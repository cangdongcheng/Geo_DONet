"""Train a Vm-to-ECG transfer model and evaluate saved AT/PCA predictions.

Compute-node entry point. Uses the existing transfer architectures and the
original PCA waveform decoder. Only small ECG predictions are saved.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import LinearTransfer, MLPTransfer, TemporalConvTransfer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRATCH = Path("/home/svu/e1032484/scratch")
DEFAULT_FEATURES = (ROOT / "Geo_DeepONet_PCA/Predictions/"
                    "geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep/Test/test_features.npz")
MODELS = {"linear": LinearTransfer, "mlp": MLPTransfer,
          "temporal_conv": TemporalConvTransfer}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ELECTRODES = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]


def pca_utils():
    # Both directories contain utils.py; use a unique module name.
    spec = importlib.util.spec_from_file_location(
        "phase_decoder_utils", ROOT / "Geo_DeepONet_PCA/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(path):
    print(f"Loading Vm/ECG archive on compute node: {path}", flush=True)
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in ("vm", "ecg", "time", "case_names", "coords")}
    vm, ecg, times = data["vm"], data["ecg"], data["time"]
    if vm.ndim != 3 or ecg.shape != (vm.shape[0], vm.shape[2], 10):
        raise ValueError(f"expected Vm(C,N,T), ECG(C,T,10); got {vm.shape}, {ecg.shape}")
    if len(times) != vm.shape[2] or np.any(np.diff(times) <= 0):
        raise ValueError("invalid time grid")
    if len(data["case_names"]) != len(vm) or len(data["coords"]) != vm.shape[1]:
        raise ValueError("case names / coordinates do not match Vm dimensions")
    print(f"Data: {len(vm)} hearts, {vm.shape[1]} nodes, {len(times)} frames; "
          f"dt={np.median(np.diff(times)):g} ms", flush=True)
    return data


def training_scale(values):
    """Train-only min/std without allocating another full cohort tensor."""
    count, total, square = 0, 0., 0.
    minimum = float("inf")
    for case in values:
        minimum = min(minimum, float(case.min()))
        count += case.size
        total += float(np.sum(case, dtype=np.float64))
        square += float(np.sum(np.square(case, dtype=np.float64)))
    scale = np.sqrt(max(square / count - (total / count) ** 2, 0.))
    if not np.isfinite(minimum) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training normalization")
    return minimum, float(scale)


def normalized_tensor(values, shift, scale, device):
    # torch.tensor copies even on CPU: in-place normalization must not change GT.
    return torch.tensor(values, dtype=torch.float32, device=device).sub_(shift).div_(scale)


def to_12lead(raw):
    """Existing ECG_transfer electrode convention, arbitrary leading dimensions."""
    la, ra, ll = raw[..., 0], raw[..., 1], raw[..., 2]
    wct = (ra + la + ll) / 3
    return np.stack([la-ra, ll-ra, ll-la, ra-(la+ll)/2,
                     la-(ra+ll)/2, ll-(ra+la)/2,
                     *(raw[..., j]-wct for j in range(4, 10))], axis=-1)


def metrics(pred, truth):
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    difference = pred - truth
    p = pred - pred.mean(axis=0)
    t = truth - truth.mean(axis=0)
    denominator = np.sqrt((p*p).sum(axis=0) * (t*t).sum(axis=0))
    valid = denominator > 1e-12
    correlation = np.full(pred.shape[-1], np.nan)
    correlation[valid] = (p*t).sum(axis=0)[valid] / denominator[valid]
    norm = np.linalg.norm(truth)
    return dict(rel_l2=float(np.linalg.norm(difference) / norm) if norm > 1e-12 else np.nan,
                mae=float(np.abs(difference).mean()),
                pcc=float(correlation[valid].mean()) if valid.any() else np.nan,
                pcc_valid_channels=int(valid.sum()))


def train(args, device):
    if min(args.epochs, args.batch_size, args.n_train, args.n_val) < 1:
        raise ValueError("epochs, batch size, and split sizes must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    # Refuse accidental overwrite of a completed transfer experiment.
    path = args.checkpoint
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}. Choose a new --checkpoint.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data = load_data(args.data)
    vm, ecg = data["vm"], data["ecg"]
    boundary = args.n_train + args.n_val
    if boundary >= len(vm):
        raise ValueError("split must leave held-out test hearts")
    vmin, vstd = training_scale(vm[:args.n_train])
    emin, estd = training_scale(ecg[:args.n_train])
    normalization = dict(vm_min=vmin, vm_std=vstd, ecg_min=emin, ecg_std=estd)
    train_vm = normalized_tensor(vm[:args.n_train], vmin, vstd, device)
    train_ecg = normalized_tensor(ecg[:args.n_train], emin, estd, device)
    val_vm = normalized_tensor(vm[args.n_train:boundary], vmin, vstd, device)
    val_ecg = normalized_tensor(ecg[args.n_train:boundary], emin, estd, device)
    config = dict(n_nodes=vm.shape[1], n_electrodes=10)
    model = MODELS[args.model](**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = args.out_dir or HERE / "Predictions" / path.stem
    output.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.model}: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"{args.n_train}/{args.n_val}/{len(vm)-boundary} split; {device}", flush=True)
    print("No early stopping; checkpoint selected by GT-Vm validation ECG MSE.", flush=True)
    print(f"Checkpoint: {path}", flush=True)
    best, best_epoch = float("inf"), -1
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(args.n_train)
        loss_sum = 0.
        for offset in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[offset:offset+args.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(train_vm[indices]) - train_ecg[indices]).square().mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation = float((model(val_vm) - val_ecg).square().mean().item())
        if not np.isfinite(validation) or not np.isfinite(loss_sum):
            raise ValueError(f"Non-finite loss at epoch {epoch}")
        history.append((epoch, loss_sum / args.n_train, validation))
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save(dict(model_state_dict=model.state_dict(), model_type=args.model,
                            model_config=config, normalization=normalization,
                            train_idx=np.arange(args.n_train),
                            val_idx=np.arange(args.n_train, boundary),
                            test_idx=np.arange(boundary, len(vm)),
                            time=data["time"], coords=data["coords"],
                            case_names=data["case_names"], data_path=str(args.data.resolve()),
                            electrode_names=ELECTRODES, ecg_unit="mV",
                            best_epoch=best_epoch, best_val=best,
                            training_args={k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()}), path)
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            elapsed = time.perf_counter()-start
            eta = elapsed/(epoch+1)*(args.epochs-epoch-1)/60
            print(f"epoch {epoch}/{args.epochs} | train {history[-1][1]:.6f} | "
                  f"val {validation:.6f} | best {best:.6f}@{best_epoch} | "
                  f"eta {eta:.1f} min", flush=True)
    np.savetxt(output / "loss.csv", history, delimiter=",",
               header="epoch,train_mse,val_mse", comments="")
    fig, ax = plt.subplots(figsize=(8, 4))
    values = np.array(history)
    ax.semilogy(values[:, 0], values[:, 1], label="train")
    ax.semilogy(values[:, 0], values[:, 2], label="validation")
    ax.set(xlabel="Epoch", ylabel="Normalized ECG MSE")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)
    print(f"Done in {(time.perf_counter()-start)/60:.1f} min; best epoch {best_epoch}")


def match_times(decoder_time, transfer_time):
    indices = np.searchsorted(decoder_time, transfer_time)
    if (np.any(indices >= len(decoder_time)) or
            not np.allclose(decoder_time[indices], transfer_time, atol=1e-5, rtol=0)):
        raise ValueError("Transfer time grid must be a subset of the decoder grid")
    return indices


def plot_case(path, times, truth, from_true, from_pred, name):
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(times, truth[:, j], color="black", lw=1.4, label="Simulation ECG")
        ax.plot(times, from_true[:, j], color="C0", lw=1.1, label="Transfer(GT Vm)")
        ax.plot(times, from_pred[:, j], color="C3", lw=1.1, ls="--", label="Transfer(predicted Vm)")
        ax.set_title(LEADS[j]); ax.set_ylabel("mV"); ax.grid(alpha=.2)
        if j >= 9:
            ax.set_xlabel("Time (ms)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .96))
    fig.suptitle(f"PCA-to-ECG evaluation: {name}")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=160); plt.close(fig)


def evaluate(args, device):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing transfer checkpoint: {args.checkpoint}. Run train first.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "normalization" not in checkpoint:
        raise ValueError("Legacy weights-only checkpoint: its training grid/split/scales must "
                         "be recovered explicitly. Use a checkpoint from pca_chain.py train.")
    data_path = args.data or Path(checkpoint["data_path"])
    data = load_data(data_path)
    if not np.array_equal(data["time"], checkpoint["time"]):
        raise ValueError("ECG transfer time grid differs from its training grid")
    if not np.array_equal(data["case_names"], checkpoint["case_names"]):
        raise ValueError("ECG data case ordering differs from transfer training")
    if not np.array_equal(data["coords"], checkpoint["coords"]):
        raise ValueError("ECG data node ordering/coordinates differ from transfer training")
    with np.load(args.features, allow_pickle=True) as archive:
        prediction = archive["pred"].astype(np.float32)
        case_ids = archive["test_indices"].astype(int)
        names = archive["case_names"].astype(str)
        targets = archive["target_names"].astype(str)
    if prediction.ndim != 3 or prediction.shape[1] != data["vm"].shape[1]:
        raise ValueError("Predicted features and transfer data have different node counts")
    expected = ["activation_time_ms"] + [f"pca_{k}" for k in range(1, prediction.shape[-1])]
    if list(targets) != expected:
        raise ValueError("Expected AT + PCA features from the original phase decoder (no slope)")
    if len(prediction) != len(case_ids) or len(np.unique(case_ids)) != len(case_ids):
        raise ValueError("Invalid feature case indices")
    if not set(case_ids).issubset(set(checkpoint["test_idx"])):
        raise ValueError("Feature cases overlap transfer fit/validation data or are not in this split")
    if not np.array_equal(names, data["case_names"][case_ids].astype(str)):
        raise ValueError("Saved feature case names do not match the ECG data")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predicted features")
    utils = pca_utils()
    basis = utils.load_decoder_basis(args.basis, prediction.shape[2]-1)
    time_indices = match_times(basis["time"], data["time"])
    if args.chunk_nodes < 1 or args.n_viz < 0:
        raise ValueError("chunk-nodes must be positive and n-viz non-negative")
    selected = np.arange(len(case_ids))
    if args.cases:
        if not set(args.cases).issubset(set(case_ids)):
            raise ValueError("Requested case is absent from saved test features")
        selected = np.flatnonzero(np.isin(case_ids, args.cases))
    model = MODELS[checkpoint["model_type"]](**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    norm = checkpoint["normalization"]
    output = args.out_dir or (HERE / "Predictions" / f"pca_chain_{args.checkpoint.stem}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Decoding at {len(basis['time'])} frames, then selecting {len(time_indices)} "
          "transfer frames; no temporal interpolation of ECG inputs.", flush=True)

    def transfer(vm):
        vm_tensor = normalized_tensor(vm[None], norm["vm_min"], norm["vm_std"], device)
        with torch.no_grad():
            result = model(vm_tensor)[0].cpu().numpy()
        return result * norm["ecg_std"] + norm["ecg_min"]

    rows, baseline, chained, truth_list, vm_rows = [], [], [], [], []
    for number, position in enumerate(selected):
        case = int(case_ids[position])
        # Decode on the original 1-ms basis grid BEFORE taking 5-ms samples.
        full_vm = utils.decode_features(prediction[position], basis, args.chunk_nodes)
        pred_vm = np.ascontiguousarray(full_vm[:, time_indices])
        del full_vm
        true_vm = data["vm"][case]
        ecg_a, ecg_b, ecg_gt = transfer(true_vm), transfer(pred_vm), data["ecg"][case]
        if not np.isfinite(ecg_a).all() or not np.isfinite(ecg_b).all():
            raise ValueError(f"Non-finite reconstructed ECG: case {case}")
        baseline.append(ecg_a); chained.append(ecg_b); truth_list.append(ecg_gt)
        vm_l2, vm_mae = utils.vm_metrics(pred_vm, true_vm)
        vm_rows.append((case, vm_l2, vm_mae))
        a12, b12, gt12 = map(to_12lead, (ecg_a, ecg_b, ecg_gt))
        for space, a, b, gt in (("electrodes10", ecg_a, ecg_b, ecg_gt),
                                 ("leads12", a12, b12, gt12)):
            for label, pred, reference in (("transfer_gt_vs_simulation", a, gt),
                                           ("pca_chain_vs_simulation", b, gt),
                                           ("pca_chain_vs_transfer_gt", b, a)):
                rows.append(dict(case=case, case_name=names[position], space=space,
                                 comparison=label, **metrics(pred, reference)))
        score = metrics(b12, gt12)
        print(f"case {case}: Vm MAE {vm_mae:.3f} mV | chain 12-lead "
              f"RelL2 {score['rel_l2']:.4f}, MAE {score['mae']:.6f} mV, "
              f"PCC {score['pcc']:.4f}", flush=True)
        if number < args.n_viz:
            plot_case(output / f"case{case}_12lead.png", data["time"], gt12, a12, b12, names[position])
    with (output / "per_case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savetxt(output / "vm_metrics_on_ecg_grid.csv", vm_rows, delimiter=",",
               header="case,vm_rel_l2,vm_mae_mv", comments="")
    baseline, chained, truth = map(np.stack, (baseline, chained, truth_list))
    np.savez_compressed(output / "ecg_predictions.npz", time=data["time"],
                        test_indices=case_ids[selected], case_names=names[selected],
                        true_ecg10=truth, transfer_gt_ecg10=baseline, pca_chain_ecg10=chained,
                        true_ecg12=to_12lead(truth), transfer_gt_ecg12=to_12lead(baseline),
                        pca_chain_ecg12=to_12lead(chained),
                        electrode_names=ELECTRODES, lead_names=LEADS)
    lines = ["PCA waveform -> learned ECG transfer (fixed held-out split)",
             f"N={len(selected)} hearts; {len(data['time'])} frames; units: mV",
             f"Transfer checkpoint: {args.checkpoint.resolve()}",
             f"Features: {args.features.resolve()}", f"Decoder: {args.basis.resolve()}",
             "Transfer(GT Vm) is a diagnostic baseline, not a guaranteed performance bound.",
             "PCC averages non-flat channels; valid channel counts are in per_case_metrics.csv.",
             "Mean +/- population SD across hearts (finite values only):"]
    for space in ("electrodes10", "leads12"):
        for label in ("transfer_gt_vs_simulation", "pca_chain_vs_simulation", "pca_chain_vs_transfer_gt"):
            group = [r for r in rows if r["space"] == space and r["comparison"] == label]
            lines.append(f"{space} | {label}")
            for key in ("rel_l2", "mae", "pcc"):
                values = np.array([r[key] for r in group])
                values = values[np.isfinite(values)]
                mean, std = (values.mean(), values.std()) if len(values) else (np.nan, np.nan)
                lines.append(f"  {key}: {mean:.6f} +/- {std:.6f} (N={len(values)})")
    (output / "test_summary.txt").write_text("\n".join(lines) + "\n")
    (output / "run_config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    print("\n".join(lines)); print(f"Outputs: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        part = sub.add_parser(name)
        part.add_argument("--device", default="cuda")
        part.add_argument("--checkpoint", type=Path,
                          default=HERE / "CheckPts/temporal_conv_f121_10000ep_pca_chain.pt")
        part.add_argument("--out-dir", type=Path)
        part.add_argument("--data", type=Path, default=SCRATCH / "geo_donet_data_f121.npz"
                          if name == "train" else None)
        if name == "train":
            part.add_argument("--model", choices=MODELS, default="temporal_conv")
            part.add_argument("--epochs", type=int, default=10000)
            part.add_argument("--batch-size", type=int, default=8)
            part.add_argument("--lr", type=float, default=1e-4)
            part.add_argument("--n-train", type=int, default=95)
            part.add_argument("--n-val", type=int, default=5)
            part.add_argument("--seed", type=int, default=42)
        else:
            part.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
            part.add_argument("--basis", type=Path, default=SCRATCH / "pca_phase_aligned_basis_f601.npz")
            part.add_argument("--cases", type=int, nargs="+", help="global test indices; default all saved cases")
            part.add_argument("--n-viz", type=int, default=2)
            part.add_argument("--chunk-nodes", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected_device = torch.device(arguments.device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run on an allocated GPU node or pass --device cpu.")
    (train if arguments.command == "train" else evaluate)(arguments, selected_device)
