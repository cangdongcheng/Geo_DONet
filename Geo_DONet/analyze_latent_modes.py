"""Analyse Geo-DONet latent orthogonality and effective mode use.

For branch outputs B (hearts x width) and trunk outputs T (queries x width),
Geo-DONet predicts Y = B T^T. Raw latent columns are not identifiable: for any
invertible A, (B A)(T A^{-T})^T gives the same Y. Consequently, raw-column
orthogonality is descriptive but not an invariant property of the predictor.

This script reports both raw cosine-Gram matrices and the invariant singular
spectrum of Y. The full Y is never materialised: T^T T is accumulated in
chunks, then Y Y^T = B (T^T T) B^T is diagonalised as a small matrix.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from opnn import (build_model, config_from_state_dict, is_legacy_state_dict,
                  remap_legacy_state_dict)


DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
CHECKPOINT = "CheckPts/geodonet_w300_d4_5000ep_lrsched.pt"
N_TRAIN, N_VAL = 95, 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geo-DONet latent orthogonality/effective-rank diagnostic")
    parser.add_argument("--model-path", default=CHECKPOINT)
    parser.add_argument("--data-path", default=DATA)
    parser.add_argument("--frame-step", type=int, default=5,
                        help="time stride; 5 gives the trained f121 grid")
    parser.add_argument("--cases", choices=("train", "val", "test", "all"),
                        default="train")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--mode-profiles", type=int, default=6,
                        help="leading centred effective modes plotted versus time; 0 disables")
    parser.add_argument("--rank-tol", type=float, default=1e-6,
                        help="relative singular-value cutoff for numerical rank")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def checkpoint_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = "model_chkpts_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_model(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw)
    if is_legacy_state_dict(state):
        state = remap_legacy_state_dict(state)
    config = config_from_state_dict(state)
    model = build_model(config).to(device)
    model.load_state_dict(state); model.eval()
    return model, config


def select_cases(name, n_cases, n_train, n_val):
    groups = {
        "train": np.arange(n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n_cases),
        "all": np.arange(n_cases),
    }
    return groups[name]


def query_chunk(coords, time_values, start, end):
    """Node-major (coordinate,time) queries matching Geo-DONet training."""
    n_nodes, coord_dim = coords.shape
    n_frames = end - start
    coords_tiled = coords[:, None, :].expand(n_nodes, n_frames, coord_dim)
    times_tiled = time_values[start:end][None, :, None].expand(n_nodes, n_frames, 1)
    return torch.cat((coords_tiled, times_tiled), dim=2).reshape(-1, coord_dim + 1)


def cosine_gram(gram):
    diagonal = np.clip(np.diag(gram), 0.0, None)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    output = np.zeros_like(gram, dtype=np.float64)
    np.divide(gram, denominator, out=output, where=denominator > 0)
    return np.clip(output, -1.0, 1.0)


def off_diagonal_stats(cosine):
    values = np.abs(cosine[~np.eye(len(cosine), dtype=bool)])
    return dict(mean=float(values.mean()),
                rms=float(np.sqrt(np.mean(values ** 2))),
                maximum=float(values.max()),
                above_01=float(np.mean(values > 0.1)),
                above_05=float(np.mean(values > 0.5)),
                above_09=float(np.mean(values > 0.9)))


def spectrum_from_gram(gram):
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    return np.sqrt(eigenvalues), eigenvalues, eigenvectors[:, order]


def spectrum_stats(singular, energy, rank_tol):
    if not len(singular) or singular[0] <= 0 or energy.sum() <= 0:
        return dict(rank=0, k90=0, k95=0, k99=0, k999=0,
                    stable_rank=0.0, participation=0.0, entropy_rank=0.0)
    fractions = energy / energy.sum()
    cumulative = np.cumsum(fractions)
    def needed(level):
        return int(np.searchsorted(cumulative, level) + 1)
    positive = fractions > 0
    entropy = -np.sum(fractions[positive] * np.log(fractions[positive]))
    return dict(rank=int(np.sum(singular / singular[0] > rank_tol)),
                k90=needed(0.90), k95=needed(0.95), k99=needed(0.99),
                k999=needed(0.999),
                stable_rank=float(energy.sum() / energy[0]),
                participation=float(1.0 / np.sum(fractions ** 2)),
                entropy_rank=float(np.exp(entropy)))


def effective_spectrum(branch, trunk_gram):
    return spectrum_from_gram(branch @ trunk_gram @ branch.T)


def save_spectrum_csv(path, spectra):
    max_modes = max(len(values[0]) for values in spectra.values())
    headers = ["mode"]
    for name in spectra:
        headers.extend((f"{name}_singular", f"{name}_energy_fraction",
                        f"{name}_cumulative_energy"))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(headers)
        for mode in range(max_modes):
            row = [mode + 1]
            for singular, energy in spectra.values():
                if mode < len(singular) and energy.sum() > 0:
                    fraction = energy / energy.sum()
                    row.extend((singular[mode], fraction[mode],
                                fraction[:mode + 1].sum()))
                else:
                    row.extend(("", "", ""))
            writer.writerow(row)


def save_correlation_plot(path, branch_cosine, trunk_cosine):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, values, title in zip(
            axes, (branch_cosine, trunk_cosine),
            ("branch channels (heart-centred)", "trunk basis functions")):
        image = axis.imshow(values, cmap="coolwarm", vmin=-1, vmax=1,
                            interpolation="nearest", rasterized=True)
        axis.set_title(title); axis.set_xlabel("latent channel")
        axis.set_ylabel("latent channel")
    figure.colorbar(image, ax=axes, shrink=0.8, label="cosine inner product")
    figure.savefig(path, dpi=180, bbox_inches="tight"); plt.close(figure)


def save_spectrum_plot(path, spectra, rank_tol):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (singular, energy) in spectra.items():
        if len(singular) and singular[0] > 0:
            axes[0].semilogy(np.arange(1, len(singular) + 1),
                             singular / singular[0], label=name)
        if energy.sum() > 0:
            axes[1].plot(np.arange(1, len(energy) + 1),
                         np.cumsum(energy / energy.sum()), label=name)
    axes[0].axhline(rank_tol, color="0.5", ls=":", label="rank tolerance")
    axes[0].set_ylabel("singular value / largest")
    axes[1].set_ylabel("cumulative squared-singular-value energy")
    axes[1].set_ylim(0, 1.01)
    for axis in axes:
        axis.set_xlabel("mode"); axis.grid(alpha=0.3); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def compute_mode_profiles(model, coords, time_values, chunk_frames,
                          coefficients, device):
    """Time profiles of canonical orthonormal right-singular field modes."""
    n_nodes, n_times = len(coords), len(time_values)
    n_modes = coefficients.shape[1]
    temporal_rms = np.empty((n_modes, n_times), dtype=np.float64)
    temporal_mean = np.empty((n_modes, n_times), dtype=np.float64)
    coefficients_t = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    scale = np.sqrt(n_nodes * n_times)
    norm_squares = np.zeros(n_modes, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_times, chunk_frames):
            end = min(start + chunk_frames, n_times)
            queries = query_chunk(coords, time_values, start, end)
            values = (model.trunk(queries) @ coefficients_t).reshape(
                n_nodes, end - start, n_modes)
            array = values.cpu().numpy().astype(np.float64)
            temporal_rms[:, start:end] = (
                np.sqrt(np.mean(array ** 2, axis=0)).T * scale)
            temporal_mean[:, start:end] = np.mean(array, axis=0).T * scale
            norm_squares += np.sum(array ** 2, axis=(0, 1))
            del queries, values
    return temporal_rms, temporal_mean, norm_squares


def save_mode_profiles(path, time_ms, rms, mean):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for mode in range(len(rms)):
        axes[0].plot(time_ms, rms[mode], label=f"mode {mode + 1}")
        axes[1].plot(time_ms, mean[mode], label=f"mode {mode + 1}")
    axes[0].set_ylabel("spatial RMS (global-RMS normalized)")
    axes[1].set_ylabel("spatial mean (global-RMS normalized)")
    for axis in axes:
        axis.set_xlabel("time (ms)"); axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def main():
    args = parse_args()
    if args.frame_step < 1 or args.chunk_frames < 1 or args.mode_profiles < 0:
        raise SystemExit("frame/chunk steps must be positive and mode profiles non-negative")
    if not 0 < args.rank_tol < 1:
        raise SystemExit("--rank-tol must lie between zero and one")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(args.model_path, device)

    # Access selected NPZ keys only, avoiding decompression of the 11 GB V_m.
    archive = np.load(args.data_path, allow_pickle=True)
    theta = archive["theta"].astype(np.float32)
    coords = archive["coords"].astype(np.float32)
    time_ms = archive["time"].astype(np.float32)[::args.frame_step]
    if args.n_train + args.n_val > len(theta):
        raise SystemExit("n_train + n_val exceeds the number of hearts")
    selected = select_cases(args.cases, len(theta), args.n_train, args.n_val)
    if len(selected) < 2:
        raise SystemExit("at least two hearts are needed for centred analysis")

    theta_mean = theta[:args.n_train].mean(axis=0)
    theta_std = theta[:args.n_train].std(axis=0)
    theta_std[theta_std < 1e-8] = 1.0
    coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)
    time_norm = ((time_ms - time_ms.min()) /
                 (time_ms.max() - time_ms.min() + 1e-8))
    theta_t = torch.as_tensor((theta[selected] - theta_mean) / theta_std,
                              dtype=torch.float32, device=device)
    coords_t = torch.as_tensor(coords_norm, dtype=torch.float32, device=device)
    time_t = torch.as_tensor(time_norm, dtype=torch.float32, device=device)

    out_dir = args.out_dir or os.path.join(
        "Predictions", checkpoint_stem(args.model_path), f"latent_modes_{args.cases}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"model: {config} | {device}")
    print(f"analysis: {len(selected)} {args.cases} hearts, {len(coords)} nodes, "
          f"{len(time_ms)} frames ({len(coords) * len(time_ms):,} queries)")
    print(f"output -> {out_dir}")

    with torch.no_grad():
        branch = model.branch(theta_t).cpu().numpy().astype(np.float64)
    branch_centered = branch - branch.mean(axis=0, keepdims=True)
    branch_gram = branch_centered.T @ branch_centered

    width = config["width"]
    trunk_gram = np.zeros((width, width), dtype=np.float64)
    start_time = time.time()
    print("accumulating full-grid trunk Gram matrix ...", flush=True)
    with torch.no_grad():
        for start in range(0, len(time_ms), args.chunk_frames):
            end = min(start + args.chunk_frames, len(time_ms))
            queries = query_chunk(coords_t, time_t, start, end)
            trunk = model.trunk(queries)
            # Network evaluation stays float32, but accumulate inner products
            # in float64 so the tail of the singular spectrum is not a
            # float32 summation artefact over ~6 million queries.
            trunk64 = trunk.to(torch.float64)
            trunk_gram += (trunk64.T @ trunk64).cpu().numpy()
            print(f"  frames {start:3d}:{end:3d}/{len(time_ms)}", flush=True)
            del queries, trunk, trunk64
    print(f"trunk Gram completed in {(time.time() - start_time):.1f} s", flush=True)

    branch_cosine = cosine_gram(branch_gram)
    trunk_cosine = cosine_gram(trunk_gram)
    branch_orth = off_diagonal_stats(branch_cosine)
    trunk_orth = off_diagonal_stats(trunk_cosine)

    branch_singular, branch_energy, _ = spectrum_from_gram(branch_gram)
    trunk_singular, trunk_energy, _ = spectrum_from_gram(trunk_gram)
    full_singular, full_energy, _ = effective_spectrum(branch, trunk_gram)
    centred_singular, centred_energy, centred_left = effective_spectrum(
        branch_centered, trunk_gram)
    spectra = {
        "effective_centered": (centred_singular, centred_energy),
        "effective_uncentered": (full_singular, full_energy),
        "raw_branch_centered": (branch_singular, branch_energy),
        "raw_trunk": (trunk_singular, trunk_energy),
    }
    stats = {name: spectrum_stats(singular, energy, args.rank_tol)
             for name, (singular, energy) in spectra.items()}

    profile_norms = np.empty(0)
    if args.mode_profiles:
        valid = centred_singular > centred_singular[0] * args.rank_tol
        n_profiles = min(args.mode_profiles, int(valid.sum()))
        if n_profiles:
            # v_k(q) = T(q,:) B_c^T u_k / sigma_k.
            coefficients = (branch_centered.T @ centred_left[:, :n_profiles]
                            / centred_singular[:n_profiles][None, :])
            rms, mean, profile_norms = compute_mode_profiles(
                model, coords_t, time_t, args.chunk_frames, coefficients, device)
            save_mode_profiles(os.path.join(out_dir, "effective_mode_time_profiles.png"),
                               time_ms, rms, mean)

    save_correlation_plot(os.path.join(out_dir, "raw_channel_cosine_gram.png"),
                          branch_cosine, trunk_cosine)
    save_spectrum_plot(os.path.join(out_dir, "mode_spectrum.png"), spectra,
                       args.rank_tol)
    save_spectrum_csv(os.path.join(out_dir, "mode_spectrum.csv"), spectra)
    np.savez_compressed(os.path.join(out_dir, "latent_mode_analysis.npz"),
                        selected_cases=selected, branch=branch,
                        branch_gram=branch_gram, trunk_gram=trunk_gram,
                        branch_cosine=branch_cosine, trunk_cosine=trunk_cosine,
                        effective_singular=full_singular,
                        effective_centered_singular=centred_singular,
                        raw_branch_singular=branch_singular,
                        raw_trunk_singular=trunk_singular,
                        mode_profile_norm_squares=profile_norms, time=time_ms)

    lines = [
        "=== Geo-DONet latent-mode analysis ===",
        f"checkpoint: {args.model_path}", f"model: {config}",
        f"cases: {args.cases} ({len(selected)} hearts)",
        f"grid: {len(coords)} nodes x {len(time_ms)} frames = "
        f"{len(coords) * len(time_ms):,} equally weighted queries", "",
        "Raw-channel orthogonality (absolute off-diagonal cosine):",
        f"  branch centred: mean {branch_orth['mean']:.4f}, "
        f"RMS {branch_orth['rms']:.4f}, max {branch_orth['maximum']:.4f}; "
        f">0.1 {100 * branch_orth['above_01']:.1f}%, "
        f">0.5 {100 * branch_orth['above_05']:.1f}%, "
        f">0.9 {100 * branch_orth['above_09']:.1f}%",
        f"  trunk functions: mean {trunk_orth['mean']:.4f}, "
        f"RMS {trunk_orth['rms']:.4f}, max {trunk_orth['maximum']:.4f}; "
        f">0.1 {100 * trunk_orth['above_01']:.1f}%, "
        f">0.5 {100 * trunk_orth['above_05']:.1f}%, "
        f">0.9 {100 * trunk_orth['above_09']:.1f}%", "",
        "Mode counts (energy = squared singular values):",
    ]
    for name in ("effective_centered", "effective_uncentered",
                 "raw_branch_centered", "raw_trunk"):
        values = stats[name]
        lines.append(f"  {name:>21}: numerical rank {values['rank']:3d}; "
                     f"K90/K95/K99/K99.9 = {values['k90']}/{values['k95']}/"
                     f"{values['k99']}/{values['k999']}; "
                     f"entropy rank {values['entropy_rank']:.2f}; "
                     f"participation {values['participation']:.2f}")
    lines.extend(("", "Interpretation:",
        "  effective_centered is the primary geometry-dependent V_m mode count.",
        "  Raw columns need not be orthogonal: invertible latent mixing leaves B T^T unchanged.",
        "  The effective singular spectrum is invariant to that mixing.",
        f"  With {len(selected)} hearts, centred rank is at most {len(selected) - 1}, "
        f"regardless of latent width {width}."))
    if len(profile_norms):
        lines.append("  canonical mode norm^2 check: "
                     + ", ".join(f"{value:.6f}" for value in profile_norms))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")


if __name__ == "__main__":
    main()
