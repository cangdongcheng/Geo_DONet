"""Evaluate held-out hearts or predict a new cohort on the fixed canonical mesh."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import digest, resolve_prepared_manifest, write_json
from .data_handling.loading import import_raw, load_prepared, load_raw
from .data_handling.normalization import FeatureNormalizer
from .decoder import activation_time, decode_features, load_decoder_basis
from .metrics import at_metrics, median_max_dvdt, vm_metrics
from .model import FeatureDeepONet
from .utils.plots import save_trace_plot


def load_checkpoint(path, device, prepared_manifest=None):
    # These are user-owned experiment checkpoints, which include NumPy states.
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('schema_version') != 1:
        raise ValueError('Expected a Geo_DONet schema-1 checkpoint; legacy parity is tested separately')
    manifest = resolve_prepared_manifest(
        saved['prepared_manifest'], saved['prepared_sha256'], prepared_manifest,
        relative_path=saved.get('prepared_manifest_relative'), anchor=Path(path).resolve().parent,
    )
    config = dict(saved['config'])
    model = FeatureDeepONet(**config).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return saved, model, manifest


def write_metrics(folder, rows):
    folder = Path(folder)
    with (folder / 'per_case_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(n_cases=len(rows), metrics={})
    for key in rows[0]:
        if key in ('case', 'case_name', 'fold'):
            continue
        values = np.asarray([r[key] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary['metrics'][key] = dict(mean=float(finite.mean()) if len(finite) else None,
                                     std=float(finite.std()) if len(finite) else None,
                                     n_valid=len(finite))
    write_json(folder / 'summary.json', summary)
    return summary


def evaluate(args, folder, device, prediction_only=False, output=None):
    """Predict nodewise AT/PCA features; optionally decode and compare with GT Vm."""
    # 1. Restore the model, split, decoder and training-fitted normalization.
    replacement = args.prepared_manifest
    if replacement is None and not prediction_only:
        replacement = args.input
    saved, model, manifest = load_checkpoint(args.chkpt, device, replacement)
    dataset_info, prepared, split = load_prepared(manifest)
    if saved['splits'] != split:
        raise ValueError('Checkpoint split membership disagrees with prepared dataset')
    basis = load_decoder_basis(manifest.parent / 'decoder_basis.npz', dataset_info['n_components'])
    normalizer = FeatureNormalizer.from_state(saved['normalizer'])
    folder = Path(folder)
    # 2. Select held-out training-cohort hearts or a new inference cohort.
    if prediction_only:
        raw_manifest = import_raw(args.input, folder / 'input', require_vm=False)
        data = load_raw(raw_manifest, require_vm=False)
        if not np.array_equal(data['coords'], prepared['coords']):
            raise ValueError('Prediction requires the same canonical Cobiveco node order as the decoder')
        if not np.array_equal(data['time'], basis['time']):
            raise ValueError('Prediction time grid must match the fitted decoder')
        if data['theta'].shape[1] != prepared['theta'].shape[1]:
            raise ValueError('Geometry parameter dimension differs from training')
        indices = list(range(len(data['theta'])))
    else:
        if args.input and Path(args.input).resolve() != manifest.resolve():
            # Main resolves a one-fold preparation root before calling this function.
            raise ValueError('Evaluation must use the checkpoint preparation; omit --input')
        data, indices = prepared, split['test']
    # 3. Create scratch outputs. The default archive stores compact features.
    if output is None:
        # CV calls evaluate() directly after training each fold.
        output = folder / 'evaluation'
        output.mkdir()
    else:
        output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    predictions, rows, timings = [], [], []
    coords = normalizer.coords(data['coords'])
    dt = float(basis['time'][1] - basis['time'][0])
    # 4. Predict one heart at a time and decode only when needed.
    for position, case in enumerate(indices):
        theta = torch.as_tensor(normalizer.theta(data['theta'][case:case+1]), dtype=torch.float32, device=device)
        chunks = []
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(coords), args.chunk_nodes):
                query = torch.as_tensor(coords[start:start+args.chunk_nodes], dtype=torch.float32, device=device)
                chunks.append(model(theta, query)[0].cpu().numpy())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        pred = normalizer.targets_inverse(np.concatenate(chunks)).astype(np.float32)
        predictions.append(pred)
        decoded = None
        if not prediction_only or args.save_vm:
            decoded = decode_features(pred, basis, args.chunk_nodes)
        if args.save_vm:
            # One heart per file prevents stacking a second complete Vm cohort in memory.
            np.save(output / f'vm_case_{case:04d}.npy', decoded)
        if not prediction_only:
            truth = data['vm'][case]
            true_features = data['targets'][case]
            vm_rel, vm_mae = vm_metrics(decoded, truth)
            at_rel, at_mae = at_metrics(pred[:, 0], true_features[:, 0])
            decoded_at = activation_time(decoded, basis['time'], basis['at_threshold'])
            _, decoded_at_mae = at_metrics(decoded_at, true_features[:, 0])
            true_slope = median_max_dvdt(truth, dt)
            row = dict(case=int(case), case_name=str(data['case_names'][case]),
                       vm_rel_l2=vm_rel, vm_mae_mv=vm_mae,
                       direct_at_rel_l2=at_rel, direct_at_mae_ms=at_mae,
                       decoded_at_mae_ms=decoded_at_mae,
                       decoded_at_valid_fraction=float(np.isfinite(decoded_at).mean()),
                       dvdt_fraction=median_max_dvdt(decoded, dt)/true_slope if true_slope else float('nan'))
            for channel in range(1, pred.shape[1]):
                row[f'pca_{channel}_mae'] = float(np.abs(pred[:, channel]-true_features[:, channel]).mean())
            rows.append(row)
            print(f'Case {case}: Vm MAE {vm_mae:.4f} mV; direct AT MAE {at_mae:.4f} ms', flush=True)
            if position < args.n_viz:
                ordered = np.argsort(true_features[:, 0])
                nodes = ordered[np.linspace(0, len(ordered)-1, min(4, len(ordered))).astype(int)]
                save_trace_plot(figures / f'case_{case:04d}_traces.png', basis['time'], truth,
                                decoded, nodes, true_features[:, 0], decoded_at)
    # 5. Save features, physical-unit metrics and preparation identity.
    payload = dict(pred=np.stack(predictions), target_names=np.asarray(dataset_info['target_names']),
                   test_indices=np.asarray(indices), case_names=np.asarray(data['case_names'][indices]),
                   prepared_manifest=np.asarray(str(manifest)), prepared_sha256=np.asarray(digest(manifest)),
                   prepared_manifest_relative=np.asarray(os.path.relpath(manifest, output)))
    if not prediction_only:
        payload['true'] = np.asarray(data['targets'][indices])
    np.savez_compressed(output / 'predicted_features.npz', **payload)
    write_json(output / 'prediction_info.json', dict(
        checkpoint=str(Path(args.chkpt).resolve()), prepared_manifest=str(manifest),
        model_config=saved['config'],
        n_cases=len(indices), mean_inference_seconds=float(np.mean(timings)), prediction_only=prediction_only))
    if rows:
        write_metrics(output, rows)
    return rows
