"""Train AT/PCA DeepONet with reconstructed-Vm supervision.

Read the train() function from top to bottom:
load split -> normalize -> build networks -> train -> validate -> save results.
"""
import csv
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .artifacts import digest, write_json
from .data_handling.loading import load_prepared
from .data_handling.normalization import FeatureNormalizer
from .decoder import DifferentiablePhaseDecoder, load_decoder_basis
from .model import FeatureDeepONet
from .utils.plots import save_loss_plot



def to_numpy(value):
    return value.detach().cpu().numpy()



def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def make_loss_weights(args, device):
    """Return normalised AT/PCA group weights and within-PCA EVR weights."""
    if args.at_loss_weight < 0 or args.pca_loss_weight < 0:
        raise SystemExit("loss-group weights must be non-negative")
    group_total = args.at_loss_weight + args.pca_loss_weight
    if group_total <= 0:
        raise SystemExit("at least one loss-group weight must be positive")
    at_group_weight = args.at_loss_weight / group_total
    pca_group_weight = args.pca_loss_weight / group_total

    basis = np.load(args.basis, allow_pickle=False)
    if "evr" not in basis.files or len(basis["evr"]) < args.n_components:
        raise SystemExit(f"basis needs at least {args.n_components} explained-variance values")
    pca_weights = basis["evr"][:args.n_components].astype(np.float64)
    if not np.isfinite(pca_weights).all() or pca_weights.sum() <= 0:
        raise SystemExit("basis PCA explained-variance weights are invalid")
    pca_weights /= pca_weights.sum()
    return (float(at_group_weight), float(pca_group_weight),
            tensor(pca_weights.astype(np.float32), device), pca_weights)


def grouped_feature_loss(prediction, target, at_group_weight,
                         pca_group_weight, pca_weights):
    """AT and PCA are equal-level groups; PCA modes are EVR-weighted within their group."""
    channel_mse = ((prediction - target) ** 2).mean(dim=(0, 1))
    at_mse = channel_mse[0]
    pca_mse = (channel_mse[1:] * pca_weights).sum()
    total = at_group_weight * at_mse + pca_group_weight * pca_mse
    return total, at_mse, pca_mse


def validate_training_args(args, n_nodes):
    if args.batch_size < 1 or args.nodes_per_step < 1 or args.val_nodes < 1:
        raise SystemExit("batch size and node-sample counts must be positive")
    if args.nodes_per_step > n_nodes or args.val_nodes > n_nodes:
        raise SystemExit(f"node samples cannot exceed the {n_nodes} mesh nodes")
    if args.feature_loss_weight < 0:
        raise SystemExit("--feature-loss-weight must be non-negative")


def physical_features(normalized_features, target_mean, target_std):
    """Differentiably undo per-channel target standardization."""
    return normalized_features * target_std + target_mean


def sampled_vm_target(vm, case_indices, node_indices, device):
    """Copy only a (hearts,nodes,all-times) target block to the accelerator."""
    block = vm[np.asarray(case_indices)[:, None], np.asarray(node_indices)[None, :], :]
    return tensor(block, device)


def train(args, manifest, folder, device):
    """Fit one model; select its checkpoint using validation hearts only."""
    # 1. Load the prepared data and the saved heart split.
    dataset_info, data, split = load_prepared(manifest)
    theta = data['theta']       # (hearts, geometry parameters)
    coords = data['coords']     # (canonical nodes, 4 Cobiveco coordinates)
    targets = data['targets']   # (hearts, nodes, AT + K waveform coefficients)
    data['target_names'] = np.asarray(dataset_info['target_names'])
    train_indices = np.array(split['train'])
    val_indices = np.array(split['val'])
    validate_training_args(args, len(coords))
    args.n_components = dataset_info['n_components']
    prepared_folder = Path(manifest).parent
    args.basis = str(prepared_folder / 'decoder_basis.npz')

    # 2. Restore training-fitted normalization and the frozen waveform decoder.
    with np.load(prepared_folder / 'normalizer.npz') as saved:
        normalizer = FeatureNormalizer.from_state(saved)
        vm_scale = float(saved['vm_scale'])
    vm_all = data['vm']
    vm_time = data['time']
    basis = load_decoder_basis(args.basis, args.n_components)
    args.n_train = len(train_indices)
    args.n_val = len(val_indices)

    theta_train = tensor(normalizer.theta(theta[train_indices]), device)
    target_train = tensor(normalizer.targets(targets[train_indices]), device)
    theta_val = tensor(normalizer.theta(theta[val_indices]), device)
    target_val = tensor(normalizer.targets(targets[val_indices]), device)
    coords_normalized = normalizer.coords(coords)
    target_mean = tensor(normalizer.target_mean, device).view(1, 1, -1)
    target_std = tensor(normalizer.target_std, device).view(1, 1, -1)
    at_group_weight, pca_group_weight, pca_weights, pca_weights_np = \
        make_loss_weights(args, device)
    decoder = DifferentiablePhaseDecoder(basis, args.n_components).to(device)

    # 3. Build the geometry branch, spatial trunk and AT/PCA output heads.
    model = FeatureDeepONet(
        geo_dim=theta.shape[1],
        coord_dim=coords.shape[1],
        width=args.width,
        depth=args.depth,
        output_dim=targets.shape[2],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Set up output paths and a fixed validation-node sample.
    folder = Path(folder)
    model_path = folder / 'checkpoints' / 'best.pt'
    model_path.parent.mkdir(parents=True)
    out_dir = folder / 'logs'
    out_dir.mkdir(exist_ok=True)
    figures = folder / 'figures'
    figures.mkdir()
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"data: {len(theta)} hearts, {len(coords)} nodes, targets {list(data['target_names'])}")
    print(f"split: {len(train_indices)} train / {len(val_indices)} val / "
          f"{len(theta) - len(train_indices) - len(val_indices)} test")
    print(f"model: {model.config()} | {n_params:,} params | {device}")
    print(f"waveform supervision: {args.nodes_per_step} random nodes/step x "
          f"{len(vm_time)} complete times | fixed {args.val_nodes}-node validation")
    print(f"V_m loss: MSE((decoded - truth) / {vm_scale:.5g} mV)")
    print(f"auxiliary normalized feature loss weight: {args.feature_loss_weight:g} "
          f"(AT/PCA shares {at_group_weight:.3f}/{pca_group_weight:.3f})")
    if args.feature_loss_weight > 0:
        print("PCA weights within auxiliary group: "
              + ", ".join(f"PC{k + 1}={weight:.4f}"
                          for k, weight in enumerate(pca_weights_np)))
    print(f"checkpoint -> {model_path}")

    history = []
    best_val = float("inf")
    best_epoch = -1
    order = np.arange(len(train_indices))
    val_rng = np.random.default_rng(args.seed + (10_000 if args.mode == 'cv' else 1_000))
    val_nodes = np.sort(val_rng.choice(len(coords), size=args.val_nodes,
                                       replace=False))
    val_coords = tensor(coords_normalized[val_nodes], device)
    val_vm_truth = sampled_vm_target(vm_all, val_indices, val_nodes, device)
    tic = time.time()
    loss_path = out_dir / 'loss.csv'
    # 5. Train each epoch, validate periodically, and retain the lowest validation loss.
    with open(loss_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_total", "train_vm_mse",
                         "train_at_mse_diagnostic", "train_pca_mse_diagnostic",
                         "val_total", "val_vm_mse", "val_at_mse_diagnostic",
                         "val_pca_mse_diagnostic"])
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(order)
            running_total = running_vm = running_at = running_pca = 0.0
            n_batches = 0
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                nodes = np.random.choice(len(coords), size=args.nodes_per_step,
                                         replace=False)
                coords_batch = tensor(coords_normalized[nodes], device)
                # Network: (heart geometry, node coordinates) -> normalized AT/PCA.
                prediction_normalized = model(theta_train[batch], coords_batch)
                prediction_physical = physical_features(
                    prediction_normalized, target_mean, target_std)
                # Decoder: predicted AT/PCA -> physical Vm at every time point.
                decoded_vm = decoder(prediction_physical, nodes)
                vm_truth = sampled_vm_target(vm_all, train_indices[batch], nodes, device)
                vm_loss = torch.mean(((decoded_vm - vm_truth) / vm_scale) ** 2)
                # By default feature_loss_weight is zero: only Vm supervises training.
                feature_loss, at_loss, pca_loss = grouped_feature_loss(
                    prediction_normalized, target_train[batch][:, nodes],
                    at_group_weight, pca_group_weight, pca_weights)
                loss = vm_loss + args.feature_loss_weight * feature_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_total += loss.item()
                running_vm += vm_loss.item()
                running_at += at_loss.item()
                running_pca += pca_loss.item()
                n_batches += 1
            train_total = running_total / n_batches
            train_vm = running_vm / n_batches
            train_at = running_at / n_batches
            train_pca = running_pca / n_batches

            val_total = val_vm = val_at = val_pca = np.nan
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_prediction_normalized = model(theta_val, val_coords)
                    val_prediction_physical = physical_features(
                        val_prediction_normalized, target_mean, target_std)
                    val_decoded_vm = decoder(val_prediction_physical, val_nodes)
                    val_vm_tensor = torch.mean(
                        ((val_decoded_vm - val_vm_truth) / vm_scale) ** 2)
                    val_feature, val_at_tensor, val_pca_tensor = grouped_feature_loss(
                        val_prediction_normalized, target_val[:, val_nodes],
                        at_group_weight, pca_group_weight, pca_weights)
                    val_total_tensor = (val_vm_tensor
                                        + args.feature_loss_weight * val_feature)
                    val_total = val_total_tensor.item()
                    val_vm = val_vm_tensor.item()
                    val_at = val_at_tensor.item()
                    val_pca = val_pca_tensor.item()
                if val_total < best_val:
                    best_val, best_epoch = val_total, epoch
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'config': model.config(),
                        'normalizer': normalizer.state(),
                        'target_names': data['target_names'],
                        'loss_config': {
                            'type': 'decoded_vm_mse',
                            'vm_scale': vm_scale,
                            'nodes_per_step': args.nodes_per_step,
                            'val_nodes': args.val_nodes,
                            'feature_loss_weight': args.feature_loss_weight,
                            'at_group_weight': at_group_weight,
                            'pca_group_weight': pca_group_weight,
                            'pca_weights': pca_weights_np,
                        },
                        'n_train': args.n_train,
                        'n_val': args.n_val,
                        'prepared_manifest': str(Path(manifest).resolve()),
                        'prepared_manifest_relative': os.path.relpath(manifest, model_path.parent),
                        'prepared_sha256': digest(manifest),
                        'splits': split,
                        'best_epoch': best_epoch,
                        'best_val': best_val,
                        'schema_version': 1,
                    }
                    atomic_save(checkpoint, model_path)
            history.append((epoch, train_total, train_vm, train_at, train_pca,
                            val_total, val_vm, val_at, val_pca))
            writer.writerow(history[-1])
            handle.flush()

            if epoch % args.print_every == 0 or epoch == args.epochs - 1:
                elapsed = time.time() - tic
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                message = (f"epoch {epoch:5d}/{args.epochs} | "
                           f"train {train_total:.6f} (Vm {train_vm:.6f}, "
                           f"ATdiag {train_at:.6f}, PCAdiag {train_pca:.6f}) | "
                           f"val {val_total:.6f} (Vm {val_vm:.6f}, "
                           f"ATdiag {val_at:.6f}, PCAdiag {val_pca:.6f}) | "
                           f"best {best_val:.6f} @ {best_epoch} | "
                           f"eta {eta / 60:.1f} min")
                if np.isfinite(val_total):
                    physical = to_numpy(val_prediction_physical)
                    at_mae = np.abs(physical[..., 0]
                                    - targets[val_indices][:, val_nodes, 0]).mean()
                    message += f" | sampled val AT MAE {at_mae:.3f} ms"
                print(message, flush=True)

            if args.patience > 0 and best_epoch >= 0 and epoch - best_epoch >= args.patience:
                print(f"early stop: no validation improvement for {args.patience} epochs")
                break

    # 6. Save the loss curve and the final training summary.
    if not model_path.exists():
        raise ValueError('Training did not produce a finite validation checkpoint')
    save_loss_plot(history, figures / 'loss.png')
    write_json(folder / 'training_summary.json', dict(
        epochs_completed=len(history), best_epoch=best_epoch, best_val=best_val,
        stopped_early=len(history) < args.epochs, elapsed_seconds=time.time()-tic))
    print(f"done in {(time.time() - tic) / 60:.1f} min | best val {best_val:.6f} "
          f"at epoch {best_epoch}")
    return model_path
