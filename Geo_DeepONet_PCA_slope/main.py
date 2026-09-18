"""Train/evaluate AT+slope+8PC DeepONet through reconstructed V_m loss."""
import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from opnn import FeatureDeepONet
from utils import (DifferentiableSlopeDecoder, FeatureTransform,
                   decode_numpy, decoded_diagnostics, feature_metrics,
                   load_basis, sampled_vm, vm_metrics)


DATA = "/home/svu/e1032484/scratch/geo_deeponet_pca_slope_f601_k8.npz"
BASIS = "/home/svu/e1032484/scratch/pca_at_slope_basis_f601.npz"
VM_DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
N_TRAIN, N_VAL = 95, 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geo-DeepONet: AT+slope+PCA decoded V_m supervision")
    parser.add_argument("--test-model", action="store_true")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--vm-data", default=VM_DATA)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nodes-per-step", type=int, default=2_048)
    parser.add_argument("--val-nodes", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0,
                        help="epochs without validation improvement; default 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--chunk-nodes", type=int, default=20_000)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_compact(path, n_components):
    archive = np.load(path, allow_pickle=True)
    needed = n_components + 2
    if archive["targets"].shape[-1] < needed:
        raise SystemExit(f"compact data has only {archive['targets'].shape[-1] - 2} PCs")
    return dict(theta=archive["theta"].astype(np.float32),
                coords=archive["coords"].astype(np.float32),
                targets=archive["targets"][..., :needed].astype(np.float32),
                target_names=archive["target_names"][:needed].astype(str),
                time=archive["time"].astype(np.float32),
                case_names=(archive["case_names"] if "case_names" in archive.files
                            else None))


def stem(args):
    return (f"geodeeponet_pca_slope_vmloss_k{args.n_components}_"
            f"w{args.width}_d{args.depth}_n{args.nodes_per_step}_{args.epochs}ep")


def diagnostic_mse(raw_prediction, raw_target):
    channel = ((raw_prediction - raw_target) ** 2).mean(dim=(0, 1))
    return channel[0], channel[1], channel[2:].mean()


def save_loss_plot(rows, path):
    values = np.asarray(rows, dtype=np.float64)
    valid = np.isfinite(values[:, 5])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(values[:, 0], values[:, 1], label="train V_m")
    axes[0].semilogy(values[valid, 0], values[valid, 5], label="validation V_m")
    axes[0].set_title("training objective")
    for column, label in ((2, "AT"), (3, "slope"), (4, "PCA")):
        axes[1].semilogy(values[:, 0], values[:, column], label=f"train {label}")
    for column, label in ((6, "AT"), (7, "slope"), (8, "PCA")):
        axes[1].semilogy(values[valid, 0], values[valid, column], "--",
                         label=f"val {label}")
    axes[1].set_title("feature diagnostics (not optimized)")
    for axis in axes:
        axis.set_xlabel("epoch"); axis.set_ylabel("MSE")
        axis.grid(alpha=0.3); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def train(args, device):
    data = load_compact(args.data, args.n_components)
    theta, coords, targets = data["theta"], data["coords"], data["targets"]
    train_idx = np.arange(args.n_train)
    val_idx = np.arange(args.n_train, args.n_train + args.n_val)
    if max(args.nodes_per_step, args.val_nodes) > len(coords):
        raise SystemExit("node sample exceeds mesh size")
    basis = load_basis(args.basis, args.n_components)
    transform = FeatureTransform(theta[train_idx], coords, targets[train_idx], basis)

    print(f"loading raw V_m: {args.vm_data}", flush=True)
    vm_archive = np.load(args.vm_data, allow_pickle=True)
    vm = vm_archive["vm"]
    vm_time = vm_archive["time"].astype(np.float32)
    if vm.shape[:2] != (len(theta), len(coords)) or not np.array_equal(vm_time, basis["time"]):
        raise SystemExit("raw V_m, compact data, and basis disagree")
    print("computing train-only V_m scale ...", flush=True)
    vm_scale = float(np.std(vm[:args.n_train], dtype=np.float64))

    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32,
                                               device=device)
    theta_train = to_tensor(transform.theta(theta[train_idx]))
    theta_val = to_tensor(transform.theta(theta[val_idx]))
    target_train_raw = to_tensor(transform.targets(targets[train_idx]))
    target_val_raw = to_tensor(transform.targets(targets[val_idx]))
    coords_norm = transform.coords(coords)
    decoder = DifferentiableSlopeDecoder(basis, args.n_components).to(device)
    model = FeatureDeepONet(theta.shape[1], coords.shape[1], args.width,
                            args.depth, args.n_components + 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_stem = stem(args)
    model_path = args.model_path or os.path.join("CheckPts", run_stem + ".pt")
    out_dir = args.out_dir or os.path.join("Predictions", run_stem)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"data: {len(theta)} hearts, {len(coords)} nodes, {len(vm_time)} frames")
    print(f"split: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(theta) - len(train_idx) - len(val_idx)} test")
    print(f"model: {model.config()} | {sum(p.numel() for p in model.parameters()):,} "
          f"params | {device}")
    print(f"outputs: AT + bounded slope [{transform.slope_lower:.3f}, "
          f"{transform.slope_upper:.3f}] mV/ms + {args.n_components} PCs")
    print(f"loss: reconstructed V_m MSE only; scale={vm_scale:.5g} mV")
    print(f"sampling: batch {args.batch_size} hearts x {args.nodes_per_step} nodes "
          f"x all {len(vm_time)} frames")
    print(f"checkpoint -> {model_path}")

    rng = np.random.default_rng(args.seed + 1000)
    val_nodes = np.sort(rng.choice(len(coords), args.val_nodes, replace=False))
    val_coords = to_tensor(coords_norm[val_nodes])
    val_vm = sampled_vm(vm, val_idx, val_nodes, device)
    order = np.arange(len(train_idx))
    best_val, best_epoch = float("inf"), -1
    rows = []
    start_time = time.time()
    loss_path = os.path.join(out_dir, "loss.csv")
    with open(loss_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_vm", "train_at_diag",
                         "train_slope_diag", "train_pca_diag", "val_vm",
                         "val_at_diag", "val_slope_diag", "val_pca_diag"))
        for epoch in range(args.epochs):
            model.train(); np.random.shuffle(order)
            accum = np.zeros(4, dtype=np.float64); batches = 0
            for start in range(0, len(order), args.batch_size):
                local = order[start:start + args.batch_size]
                global_cases = train_idx[local]
                nodes = np.random.choice(len(coords), args.nodes_per_step,
                                         replace=False)
                raw = model(theta_train[local], to_tensor(coords_norm[nodes]))
                physical = transform.inverse_torch(raw)
                decoded = decoder(physical, nodes)
                truth = sampled_vm(vm, global_cases, nodes, device)
                vm_loss = torch.mean(((decoded - truth) / vm_scale) ** 2)
                at_diag, slope_diag, pca_diag = diagnostic_mse(
                    raw, target_train_raw[local][:, nodes])
                optimizer.zero_grad(set_to_none=True)
                vm_loss.backward(); optimizer.step()
                accum += (vm_loss.item(), at_diag.item(), slope_diag.item(),
                          pca_diag.item())
                batches += 1
            train_values = accum / batches

            val_values = np.full(4, np.nan)
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    raw_val = model(theta_val, val_coords)
                    physical_val = transform.inverse_torch(raw_val)
                    decoded_val = decoder(physical_val, val_nodes)
                    val_vm_loss = torch.mean(((decoded_val - val_vm) / vm_scale) ** 2)
                    at_diag, slope_diag, pca_diag = diagnostic_mse(
                        raw_val, target_val_raw[:, val_nodes])
                    val_values = np.asarray((val_vm_loss.item(), at_diag.item(),
                                             slope_diag.item(), pca_diag.item()))
                if val_values[0] < best_val:
                    best_val, best_epoch = float(val_values[0]), epoch
                    torch.save(dict(model_state_dict=model.state_dict(),
                                    config=model.config(),
                                    transform=transform.state(),
                                    n_train=args.n_train, n_val=args.n_val,
                                    n_components=args.n_components,
                                    basis_file=os.path.abspath(args.basis),
                                    vm_scale=vm_scale,
                                    loss="decoded_vm_mse_only",
                                    best_val=best_val, best_epoch=best_epoch),
                               model_path)
            row = (epoch, *train_values, *val_values)
            rows.append(row); writer.writerow(row)
            if epoch % args.print_every == 0 or epoch == args.epochs - 1:
                handle.flush()
                elapsed = time.time() - start_time
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(f"epoch {epoch:5d}/{args.epochs} | "
                      f"train Vm {train_values[0]:.6f} "
                      f"(AT {train_values[1]:.4f}, slope {train_values[2]:.4f}, "
                      f"PCA {train_values[3]:.4f}) | val Vm {val_values[0]:.6f} "
                      f"(AT {val_values[1]:.4f}, slope {val_values[2]:.4f}, "
                      f"PCA {val_values[3]:.4f}) | best {best_val:.6f}@{best_epoch} "
                      f"| eta {eta / 60:.1f} min", flush=True)
            if args.patience > 0 and best_epoch >= 0 and epoch - best_epoch >= args.patience:
                print(f"early stop: no validation improvement for {args.patience} epochs")
                break
    save_loss_plot(rows, os.path.join(out_dir, "loss.png"))
    print(f"done in {(time.time() - start_time) / 60:.1f} min | "
          f"best {best_val:.6f}@{best_epoch}")


def evaluate(args, device):
    model_path = args.model_path or os.path.join("CheckPts", stem(args) + ".pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = FeatureDeepONet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    transform = FeatureTransform.from_state(checkpoint["transform"])
    n_components = int(checkpoint["n_components"])
    basis_path = checkpoint.get("basis_file", args.basis)
    basis = load_basis(basis_path, n_components)
    data = load_compact(args.data, n_components)
    test_idx = np.arange(checkpoint["n_train"] + checkpoint["n_val"],
                         len(data["theta"]))
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32,
                                               device=device)
    coords_t = to_tensor(transform.coords(data["coords"]))
    theta_test = to_tensor(transform.theta(data["theta"][test_idx]))
    predictions, inference = [], []
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.no_grad():
        for position in range(len(test_idx)):
            sync(); start = time.perf_counter()
            raw = model(theta_test[position:position + 1], coords_t)
            sync(); inference.append(time.perf_counter() - start)
            predictions.append(transform.inverse_numpy(raw[0].cpu().numpy()))
    prediction = np.stack(predictions)
    truth_features = data["targets"][test_idx, :, :n_components + 2]

    out_dir = args.out_dir or os.path.join("Predictions",
                                            os.path.splitext(os.path.basename(model_path))[0],
                                            "Test")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    def emit(message=""):
        print(message); lines.append(message)
    emit(f"model: {checkpoint['config']} | {device}")
    emit(f"checkpoint: {model_path}")
    emit(f"loss: {checkpoint.get('loss', 'unknown')}")
    emit(f"slope bounds: [{transform.slope_lower:.3f}, "
         f"{transform.slope_upper:.3f}] mV/ms")
    emit(f"feature inference: {np.mean(inference) * 1000:.2f} +/- "
         f"{np.std(inference) * 1000:.2f} ms/heart")
    emit("direct feature MAE:")
    for channel, name in enumerate(data["target_names"][:n_components + 2]):
        per_case = np.abs(prediction[..., channel]
                          - truth_features[..., channel]).mean(axis=1)
        unit = " ms" if channel == 0 else " mV/ms" if channel == 1 else ""
        emit(f"  {name:>28}: {per_case.mean():.5f} +/- {per_case.std():.5f}{unit}")

    vm_archive = np.load(args.vm_data, allow_pickle=True)
    vm = vm_archive["vm"]
    vm_l2, vm_mae = [], []
    decoded_at, decoded_slope, dvdt_fraction = [], [], []
    for position, case in enumerate(test_idx):
        decoded = decode_numpy(prediction[position], basis, args.chunk_nodes)
        truth = vm[case]
        rel, mae = vm_metrics(decoded, truth)
        at_mae, slope_mae, fraction = decoded_diagnostics(
            decoded, truth, truth_features[position, :, 0],
            truth_features[position, :, 1], basis)
        vm_l2.append(rel); vm_mae.append(mae); decoded_at.append(at_mae)
        decoded_slope.append(slope_mae); dvdt_fraction.append(fraction)
        emit(f"case {case:3d}: Vm RelL2 {rel:.5f} | MAE {mae:.3f} mV | "
             f"AT {at_mae:.3f} ms | slope {slope_mae:.3f} mV/ms | "
             f"dVdt {fraction:.3f}")
    emit("")
    emit(f"decoded V_m: Rel L2 {np.mean(vm_l2):.5f} +/- {np.std(vm_l2):.5f} | "
         f"MAE {np.mean(vm_mae):.3f} +/- {np.std(vm_mae):.3f} mV")
    emit(f"decoded AT MAE: {np.mean(decoded_at):.3f} +/- "
         f"{np.std(decoded_at):.3f} ms")
    emit(f"decoded slope MAE: {np.mean(decoded_slope):.3f} +/- "
         f"{np.std(decoded_slope):.3f} mV/ms")
    emit(f"decoded upstroke fraction: {np.mean(dvdt_fraction):.3f} +/- "
         f"{np.std(dvdt_fraction):.3f}")
    np.savez_compressed(os.path.join(out_dir, "test_features.npz"),
                        pred=prediction, true=truth_features,
                        target_names=data["target_names"][:n_components + 2],
                        test_indices=test_idx)
    np.savetxt(os.path.join(out_dir, "vm_metrics.csv"),
               np.column_stack((test_idx, vm_l2, vm_mae, decoded_at,
                                decoded_slope, dvdt_fraction)),
               delimiter=",", comments="",
               header="case,vm_rel_l2,vm_mae_mv,decoded_at_mae_ms,"
                      "decoded_slope_mae,dvdt_fraction")
    with open(os.path.join(out_dir, "test_summary.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    emit(f"outputs -> {out_dir}")


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.test_model:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
