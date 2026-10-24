"""Training-only node templates and temporal PCA, with bounded memory usage."""
import os
from pathlib import Path

import numpy as np

from ..artifacts import artifact_records, digest, write_json
from ..decoder import activation_time, shift_waveforms
from .normalization import FeatureNormalizer


def activation_targets(data, threshold, chunk_nodes):
    vm, time = data['vm'], data['time']
    activation_times = np.empty(vm.shape[:2], dtype=np.float32)
    for case in range(len(vm)):
        for start in range(0, vm.shape[1], chunk_nodes):
            block = vm[case, start:start+chunk_nodes]
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite Vm in heart {data["case_names"][case]}')
            activation_times[case, start:start+chunk_nodes] = activation_time(block, time, threshold)
        if not np.isfinite(activation_times[case]).all():
            raise ValueError(f'Heart {data["case_names"][case]} has nodes without an upward threshold crossing')
    return activation_times


def align_training(data, activation_times, indices, reference_at, path, chunk_nodes):
    """Only training waveforms need an on-disk alignment workspace."""
    vm, time = data['vm'], data['time']
    aligned = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                       shape=(len(indices), *vm.shape[1:]))
    dt = float(time[1] - time[0])
    for local, case in enumerate(indices):
        for start in range(0, vm.shape[1], chunk_nodes):
            end = min(start + chunk_nodes, vm.shape[1])
            aligned[local, start:end] = shift_waveforms(
                vm[case, start:end], activation_times[case, start:end] - np.float32(reference_at), dt)
    aligned.flush()
    return aligned


def fit_basis(aligned, time, reference_at, threshold, n_components, chunk_nodes):
    """Fit the historical node-lookup template and centered residual temporal PCA."""
    n_train, n_nodes, n_frames = aligned.shape
    if n_components > n_frames:
        raise ValueError('n_components cannot exceed the number of time frames')
    # 1. Average the aligned training hearts at each canonical node.
    template = np.empty((n_nodes, n_frames), dtype=np.float32)
    for start in range(0, n_nodes, chunk_nodes):
        end = min(start + chunk_nodes, n_nodes)
        total = np.zeros((end-start, n_frames), dtype=np.float64)
        for case in range(n_train):
            total += aligned[case, start:end]
        template[start:end] = total / n_train
    # 2. Pool residual waveforms over training hearts and nodes.
    # Each residual row is one T-frame waveform. Accumulate only its T x T
    # covariance, using the same stable Chan/Welford update as the original code.
    count = 0
    mean = np.zeros(n_frames, dtype=np.float64)
    m2 = np.zeros((n_frames, n_frames), dtype=np.float64)
    for case in range(n_train):
        for start in range(0, n_nodes, chunk_nodes):
            residual = (aligned[case, start:start+chunk_nodes] - template[start:start+chunk_nodes]).astype(np.float64)
            block_count = len(residual)
            block_mean = residual.mean(axis=0)
            centered = residual - block_mean
            block_m2 = centered.T @ centered
            total = count + block_count
            delta = block_mean - mean
            m2 += block_m2 + np.outer(delta, delta) * (count * block_count / total)
            mean += delta * (block_count / total)
            count = total
    # 3. Keep the temporal eigenvectors with the largest explained variance.
    eigenvalues, eigenvectors = np.linalg.eigh(m2 / max(count - 1, 1))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    if eigenvalues.sum() <= 0:
        raise ValueError('Training waveforms have no residual variation for waveform PCA')
    return dict(node_template=template, residual_mean=mean.astype(np.float32),
                components=eigenvectors[:, order[:n_components]].astype(np.float32),
                evr=eigenvalues[:n_components] / eigenvalues.sum(), eigenvalues=eigenvalues[:n_components],
                time=np.asarray(time, dtype=np.float32), reference_at=float(reference_at),
                at_threshold=float(threshold))


def training_vm_scale(vm, train_indices, chunk_nodes):
    count, total, square = 0, 0., 0.
    for case in train_indices:
        for start in range(0, vm.shape[1], chunk_nodes):
            block = vm[case, start:start+chunk_nodes].astype(np.float64)
            count += block.size
            total += block.sum()
            square += np.square(block).sum()
    scale = float(np.sqrt(max(square/count - (total/count)**2, 0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training Vm scale must be finite and positive')
    return scale


def prepare_fold(data, activation_times, split, raw_manifest, folder, args):
    """Fit one training-only decoder, then transform all three splits with it."""
    # 1. Save this fold's heart indices and choose its phase reference.
    folder = Path(folder)
    folder.mkdir()
    write_json(folder / 'splits.json', split)
    reference_at = args.reference_at
    if reference_at is None:
        reference_at = float(np.median(activation_times[split['train']]))
    # 2. Align training waveforms and fit their node template + temporal PCA.
    workspace = folder / '_aligned.npy'
    try:
        aligned = align_training(data, activation_times, split['train'], reference_at, workspace, args.chunk_nodes)
        basis = fit_basis(aligned, data['time'], reference_at, args.at_threshold,
                          args.n_components, args.chunk_nodes)
        del aligned
    finally:
        workspace.unlink(missing_ok=True)
    np.savez(folder / 'decoder_basis.npz', **basis)
    # 3. Project every heart onto the frozen basis: [AT(x), c1(x), ..., cK(x)].
    # Validation/test projections never update the fitted template or PCA modes.
    shape = (*data['vm'].shape[:2], args.n_components + 1)
    targets = np.lib.format.open_memmap(folder / 'targets.npy', mode='w+', dtype=np.float32, shape=shape)
    targets[..., 0] = activation_times
    dt = float(data['time'][1] - data['time'][0])
    errors = []
    for case in range(shape[0]):
        total_error, count = 0., 0
        for start in range(0, shape[1], args.chunk_nodes):
            end = min(start + args.chunk_nodes, shape[1])
            waves = data['vm'][case, start:end]
            aligned_block = shift_waveforms(waves, activation_times[case, start:end] - np.float32(reference_at), dt)
            residual = aligned_block - basis['node_template'][start:end] - basis['residual_mean']
            coefficients = residual @ basis['components']
            targets[case, start:end, 1:] = coefficients
            reconstructed = basis['node_template'][start:end] + basis['residual_mean'] + coefficients @ basis['components'].T
            decoded = shift_waveforms(reconstructed, np.float32(reference_at) - activation_times[case, start:end], dt)
            total_error += float(np.abs(decoded-waves).sum(dtype=np.float64))
            count += waves.size
        errors.append(total_error / count)
    targets.flush()
    # 4. Fit geometry/feature/Vm scales using training hearts only.
    normalizer = FeatureNormalizer(
        data['theta'][split['train']],
        data['coords'],
        targets[split['train']],
    )
    np.savez(folder / 'normalizer.npz', **normalizer.state(),
             vm_scale=training_vm_scale(data['vm'], split['train'], args.chunk_nodes))
    # 5. Record oracle reconstruction diagnostics and the artifact identities.
    write_json(folder / 'diagnostics.json', dict(
        oracle_vm_mae_mv={key: float(np.mean(np.array(errors)[indices])) for key, indices in split.items()},
        explained_variance_ratio=basis['evr'].tolist(), reference_at_ms=reference_at,
        note='Oracle uses ground-truth AT and projected coefficients; this is not neural-network performance.',
    ))
    names = ['splits.json', 'decoder_basis.npz', 'normalizer.npz', 'targets.npy', 'diagnostics.json']
    write_json(folder / 'manifest.json', dict(
        kind='prepared_fold', version=1, raw_manifest=os.path.relpath(raw_manifest, folder),
        raw_manifest_sha256=digest(raw_manifest), n_components=args.n_components,
        target_names=['activation_time_ms'] + [f'pca_{k+1}' for k in range(args.n_components)],
        artifacts=artifact_records(folder, names),
    ))
