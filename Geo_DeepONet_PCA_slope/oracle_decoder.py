"""Oracle test for an AT + upstroke-slope + residual-PCA V_m decoder.

The representation is deliberately tested before training another network.
For each node waveform:

1. activation time (AT) is its -10 mV upward crossing;
2. slope is the maximum positive dV/dt near AT;
3. AT shifts the waveform to a common reference time;
4. a monotone local time warp normalises the upstroke slope while leaving
   times outside a configurable window unchanged;
5. a train-only node template and temporal PCA model the remaining waveform.

Held-out hearts are reconstructed with their TRUE AT, TRUE slope, and oracle
PCA projections. This measures decoder/representation capacity only. It does
not measure whether a neural network can predict the features.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
OUT = "/home/svu/e1032484/scratch/pca_at_slope_f601"
BASIS_OUT = "/home/svu/e1032484/scratch/pca_at_slope_basis_f601.npz"
AT_THRESHOLD = -10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle AT+slope+PCA decoder analysis")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--basis-out", default=BASIS_OUT)
    parser.add_argument("--n-train", type=int, default=95)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--at-threshold", type=float, default=AT_THRESHOLD)
    parser.add_argument("--slope-window-ms", type=float, default=10.0,
                        help="measure max positive dV/dt within AT +/- this window")
    parser.add_argument("--inner-ms", type=float, default=3.0,
                        help="half-width with constant slope scaling")
    parser.add_argument("--outer-ms", type=float, default=30.0,
                        help="warp returns exactly to an identity time map here")
    parser.add_argument("--k-list", type=int, nargs="+",
                        default=[0, 1, 2, 3, 5, 8, 10, 15, 20, 30])
    parser.add_argument("--save-modes", type=int, default=30)
    parser.add_argument("--chunk-nodes", type=int, default=5_000)
    parser.add_argument("--ordinary-aligned-csv",
                        default="/home/svu/e1032484/scratch/"
                                "pca_phase_aligned_f601/recon_vs_k.csv")
    return parser.parse_args()


def activation_time(waves, time_ms, threshold):
    crossed = (waves[:, :-1] < threshold) & (waves[:, 1:] >= threshold)
    nodes = np.where(crossed.any(axis=1))[0]
    first = np.argmax(crossed, axis=1)[nodes]
    result = np.full(len(waves), np.nan, dtype=np.float32)
    v0, v1 = waves[nodes, first], waves[nodes, first + 1]
    t0, t1 = time_ms[first], time_ms[first + 1]
    result[nodes] = t0 + ((threshold - v0) / (v1 - v0 + 1e-12)
                             * (t1 - t0))
    return result


def upstroke_slope(waves, time_ms, at, window_ms):
    """Maximum positive forward dV/dt near AT, in mV/ms."""
    dt = float(np.median(np.diff(time_ms)))
    derivative = np.diff(waves, axis=1) / np.float32(dt)
    mid_time = 0.5 * (time_ms[:-1] + time_ms[1:])
    local = np.abs(mid_time[None, :] - at[:, None]) <= window_ms
    derivative = np.where(local, derivative, -np.inf)
    result = derivative.max(axis=1).astype(np.float32)
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def sample_rows(waves, sample_time, time_ms):
    """Linearly sample each waveform at its row-specific times."""
    dt = np.float32(np.median(np.diff(time_ms)))
    position = (sample_time - np.float32(time_ms[0])) / dt
    np.clip(position, 0.0, float(len(time_ms) - 1), out=position)
    left = np.floor(position).astype(np.int32)
    np.minimum(left, len(time_ms) - 2, out=left)
    fraction = position - left
    y0 = np.take_along_axis(waves, left, axis=1)
    y1 = np.take_along_axis(waves, left + 1, axis=1)
    return (y0 + fraction * (y1 - y0)).astype(np.float32)


def forward_warp_offset(normalized_offset, slopes, reference_slope,
                        inner_ms, outer_ms):
    """Map normalised-time offset to physical-time offset.

    The derivative at zero is reference_slope / slope, making the encoded
    waveform's local dV/dt approximately reference_slope. A linear shoulder
    returns the map exactly to identity at +/- outer_ms. The map is monotone.
    """
    scale = (np.float32(reference_slope) / slopes).astype(np.float32)
    if np.any(scale * inner_ms >= outer_ms):
        worst = float(np.max(scale))
        raise ValueError(f"local warp is not monotone: max scale {worst:.3f} x "
                         f"inner {inner_ms:g} >= outer {outer_ms:g}; decrease "
                         "--inner-ms or increase --outer-ms")
    absolute = np.abs(normalized_offset)
    central_end = scale[:, None] * np.float32(inner_ms)
    shoulder_slope = ((np.float32(outer_ms) - scale * np.float32(inner_ms)) /
                      np.float32(outer_ms - inner_ms))
    mapped = np.where(
        absolute <= inner_ms,
        scale[:, None] * absolute,
        np.where(absolute < outer_ms,
                 central_end + shoulder_slope[:, None] * (absolute - inner_ms),
                 absolute))
    return np.copysign(mapped, normalized_offset).astype(np.float32)


def inverse_warp_offset(physical_offset, slopes, reference_slope,
                        inner_ms, outer_ms):
    """Exact inverse of :func:`forward_warp_offset`."""
    scale = (np.float32(reference_slope) / slopes).astype(np.float32)
    central_end = scale[:, None] * np.float32(inner_ms)
    shoulder_slope = ((np.float32(outer_ms) - scale * np.float32(inner_ms)) /
                      np.float32(outer_ms - inner_ms))
    absolute = np.abs(physical_offset)
    mapped = np.where(
        absolute <= central_end,
        absolute / scale[:, None],
        np.where(absolute < outer_ms,
                 np.float32(inner_ms) + ((absolute - central_end) /
                                         shoulder_slope[:, None]),
                 absolute))
    return np.copysign(mapped, physical_offset).astype(np.float32)


def encode_waveforms(waves, at, slopes, time_ms, reference_at,
                     reference_slope, inner_ms, outer_ms):
    """Physical V_m(t) -> AT/slope-normalised waveform on ``time_ms``."""
    normalized_offset = time_ms[None, :] - np.float32(reference_at)
    physical_offset = forward_warp_offset(
        np.broadcast_to(normalized_offset, waves.shape), slopes,
        reference_slope, inner_ms, outer_ms)
    sample_time = at[:, None] + physical_offset
    return sample_rows(waves, sample_time, time_ms)


def decode_waveforms(normalized_waves, at, slopes, time_ms, reference_at,
                     reference_slope, inner_ms, outer_ms):
    """Normalised waveform -> physical V_m(t) using AT and slope controls."""
    physical_offset = time_ms[None, :] - at[:, None]
    normalized_offset = inverse_warp_offset(
        physical_offset, slopes, reference_slope, inner_ms, outer_ms)
    sample_time = np.float32(reference_at) + normalized_offset
    return sample_rows(normalized_waves, sample_time, time_ms)


def max_abs_dvdt(waves, dt):
    return np.abs(np.gradient(waves, dt, axis=1)).max(axis=1)


def metrics(reconstruction, truth, true_at, true_slope, time_ms, threshold,
            slope_window_ms):
    difference = reconstruction - truth
    rel_l2 = float(np.linalg.norm(difference) /
                   (np.linalg.norm(truth) + 1e-12))
    mae = float(np.abs(difference).mean())
    pred_at = activation_time(reconstruction, time_ms, threshold)
    valid_at = np.isfinite(pred_at) & np.isfinite(true_at)
    at_mae = float(np.abs(pred_at[valid_at] - true_at[valid_at]).mean())
    pred_slope = upstroke_slope(reconstruction, time_ms, pred_at,
                                slope_window_ms)
    valid_slope = np.isfinite(pred_slope) & np.isfinite(true_slope)
    slope_mae = float(np.abs(pred_slope[valid_slope]
                             - true_slope[valid_slope]).mean())
    dt = float(np.median(np.diff(time_ms)))
    dvdt_median = float(np.median(max_abs_dvdt(reconstruction, dt)))
    return rel_l2, mae, at_mae, slope_mae, dvdt_median


def save_trace_plot(path, time_ms, truth, reconstructions, nodes, at, slope):
    figure, axes = plt.subplots(1, len(nodes), figsize=(4 * len(nodes), 3.6),
                                sharey=True)
    for axis, node in zip(np.atleast_1d(axes), nodes):
        axis.plot(time_ms, truth[node], "k", lw=2, label="truth")
        for name, value in reconstructions.items():
            axis.plot(time_ms, value[node], lw=1.2, label=name)
        axis.set_title(f"node {node}\nAT={at[node]:.1f} ms, "
                       f"s={slope[node]:.1f} mV/ms", fontsize=8)
        axis.set_xlabel("time (ms)"); axis.grid(alpha=0.3)
    axes[0].set_ylabel("V_m (mV)"); axes[0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def main():
    args = parse_args()
    if args.inner_ms <= 0 or args.outer_ms <= args.inner_ms:
        raise SystemExit("need 0 < --inner-ms < --outer-ms")
    if args.chunk_nodes < 1 or args.slope_window_ms <= 0:
        raise SystemExit("chunk size and slope window must be positive")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.basis_out) or ".", exist_ok=True)
    log_lines = []
    def log(message=""):
        print(message, flush=True); log_lines.append(message)

    log("=== oracle AT + slope + residual-PCA decoder ===")
    log(f"data: {args.data}")
    archive = np.load(args.data, allow_pickle=True)
    vm = archive["vm"]
    time_ms = archive["time"].astype(np.float32)
    case_names = (archive["case_names"].astype(str)
                  if "case_names" in archive.files else None)
    n_cases, n_nodes, n_frames = vm.shape
    dt = float(np.median(np.diff(time_ms)))
    if not np.allclose(np.diff(time_ms), dt):
        raise SystemExit("uniform time sampling is required")
    train_idx = np.arange(args.n_train)
    val_idx = np.arange(args.n_train, args.n_train + args.n_val)
    test_idx = np.arange(args.n_train + args.n_val, n_cases)
    del val_idx
    log(f"shape: {vm.shape}; dt={dt:g} ms")
    log(f"split: {len(train_idx)} train / {args.n_val} val / "
        f"{len(test_idx)} test")

    # Features are extracted for every heart, but only train-heart summaries
    # define the decoder reference and PCA basis.
    at_all = np.empty((n_cases, n_nodes), dtype=np.float32)
    slope_all = np.empty_like(at_all)
    log("extracting AT and local upstroke slope ...")
    tic = time.time()
    for case in range(n_cases):
        at_all[case] = activation_time(vm[case], time_ms, args.at_threshold)
        slope_all[case] = upstroke_slope(vm[case], time_ms, at_all[case],
                                         args.slope_window_ms)
        if not np.isfinite(at_all[case]).all() or not np.isfinite(slope_all[case]).all():
            raise SystemExit(f"case {case}: invalid AT or slope feature")
        if (case + 1) % 10 == 0 or case + 1 == n_cases:
            log(f"  features {case + 1:3d}/{n_cases}")
    reference_at = float(np.median(at_all[train_idx]))
    reference_slope = float(np.median(slope_all[train_idx]))
    scale = reference_slope / slope_all
    log(f"train-only reference AT: {reference_at:.4f} ms")
    log(f"train-only reference slope: {reference_slope:.4f} mV/ms")
    log(f"slope range all hearts: [{slope_all.min():.3f}, "
        f"{slope_all.max():.3f}] mV/ms; warp scale range "
        f"[{scale.min():.3f}, {scale.max():.3f}]")
    if scale.max() * args.inner_ms >= args.outer_ms:
        raise SystemExit("warp is non-monotone for the observed slope range; "
                         "decrease --inner-ms or increase --outer-ms")
    log(f"feature extraction: {(time.time() - tic) / 60:.2f} min")

    # Pass 1: fold-independent phase/slope normalisation followed by a
    # train-only node lookup template.
    node_sum = np.zeros((n_nodes, n_frames), dtype=np.float64)
    log("building train-only AT+slope-normalised node template ...")
    for position, case in enumerate(train_idx):
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            encoded = encode_waveforms(
                vm[case, start:end], at_all[case, start:end],
                slope_all[case, start:end], time_ms, reference_at,
                reference_slope, args.inner_ms, args.outer_ms)
            node_sum[start:end] += encoded
        if (position + 1) % 10 == 0 or position + 1 == len(train_idx):
            log(f"  template {position + 1:3d}/{len(train_idx)}")
    node_template = (node_sum / len(train_idx)).astype(np.float32)
    del node_sum

    # Pass 2: temporal residual covariance. The mean residual is analytically
    # zero for the node-centred training data, apart from roundoff.
    covariance_sum = np.zeros((n_frames, n_frames), dtype=np.float64)
    residual_sum = np.zeros(n_frames, dtype=np.float64)
    residual_count = 0
    log("fitting train-only residual temporal PCA ...")
    for position, case in enumerate(train_idx):
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            encoded = encode_waveforms(
                vm[case, start:end], at_all[case, start:end],
                slope_all[case, start:end], time_ms, reference_at,
                reference_slope, args.inner_ms, args.outer_ms)
            residual = (encoded - node_template[start:end]).astype(np.float64)
            residual_sum += residual.sum(axis=0)
            covariance_sum += residual.T @ residual
            residual_count += len(residual)
        if (position + 1) % 10 == 0 or position + 1 == len(train_idx):
            log(f"  covariance {position + 1:3d}/{len(train_idx)}")
    residual_mean = (residual_sum / residual_count).astype(np.float32)
    covariance = (covariance_sum - residual_count *
                  np.outer(residual_mean, residual_mean)) / max(residual_count - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigval, eigvec = np.linalg.eigh(covariance)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0.0, None)
    components = eigvec[:, order].astype(np.float32)
    evr = eigval / eigval.sum()
    cum_evr = np.cumsum(evr)
    del covariance_sum, covariance
    log("")
    log("explained variance after AT+slope normalisation:")
    for fraction in (0.90, 0.95, 0.99, 0.999, 0.9999):
        k = int(np.searchsorted(cum_evr, fraction) + 1)
        log(f"  {100 * fraction:7.2f}%: K={k}")

    k_values = sorted(set(k for k in args.k_list if 0 <= k <= n_frames))
    if not k_values:
        raise SystemExit("no valid K values")
    k_max = max(k_values)
    results = {k: [] for k in k_values}
    roundtrip_results = []
    first_trace = None
    true_dvdt = []
    log("")
    log("evaluating oracle features on held-out test hearts ...")
    for test_position, case in enumerate(test_idx):
        truth = vm[case]
        encoded = np.empty_like(truth, dtype=np.float32)
        roundtrip = np.empty_like(truth, dtype=np.float32)
        coefficient = np.empty((n_nodes, k_max), dtype=np.float32)
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            encoded[start:end] = encode_waveforms(
                truth[start:end], at_all[case, start:end],
                slope_all[case, start:end], time_ms, reference_at,
                reference_slope, args.inner_ms, args.outer_ms)
            roundtrip[start:end] = decode_waveforms(
                encoded[start:end], at_all[case, start:end],
                slope_all[case, start:end], time_ms, reference_at,
                reference_slope, args.inner_ms, args.outer_ms)
            residual = (encoded[start:end] - node_template[start:end]
                        - residual_mean)
            if k_max:
                coefficient[start:end] = residual @ components[:, :k_max]
        roundtrip_results.append(metrics(
            roundtrip, truth, at_all[case], slope_all[case], time_ms,
            args.at_threshold, args.slope_window_ms))
        true_dvdt.append(float(np.median(max_abs_dvdt(truth, dt))))
        trace_recon = {}
        for k in k_values:
            normalized_reconstruction = node_template + residual_mean
            if k:
                normalized_reconstruction = (
                    normalized_reconstruction
                    + coefficient[:, :k] @ components[:, :k].T)
            reconstruction = np.empty_like(truth, dtype=np.float32)
            for start in range(0, n_nodes, args.chunk_nodes):
                end = min(start + args.chunk_nodes, n_nodes)
                reconstruction[start:end] = decode_waveforms(
                    normalized_reconstruction[start:end],
                    at_all[case, start:end], slope_all[case, start:end],
                    time_ms, reference_at, reference_slope,
                    args.inner_ms, args.outer_ms)
            results[k].append(metrics(
                reconstruction, truth, at_all[case], slope_all[case], time_ms,
                args.at_threshold, args.slope_window_ms))
            if test_position == 0 and k in (0, 2, 5, 10):
                trace_recon[f"K={k}"] = reconstruction.copy()
        if test_position == 0:
            ordered = np.argsort(at_all[case])
            nodes = ordered[np.linspace(0, len(ordered) - 1, 4).astype(int)]
            first_trace = (truth.copy(), trace_recon, nodes,
                           at_all[case].copy(), slope_all[case].copy())
        log(f"  test {test_position + 1:2d}/{len(test_idx)}")

    true_dvdt_median = float(np.mean(true_dvdt))
    rows = []
    log("")
    log("oracle AT+slope+PCA reconstruction on test hearts:")
    log(f"{'K':>4} {'cumEVR':>9} {'Vm RelL2':>9} {'Vm MAE':>8} "
        f"{'AT MAE':>8} {'slope MAE':>10} {'dVdt frac':>10}")
    for k in k_values:
        values = np.asarray(results[k], dtype=np.float64)
        means = np.nanmean(values, axis=0)
        fraction = means[4] / true_dvdt_median
        cumulative = 0.0 if k == 0 else float(cum_evr[k - 1])
        row = (k, cumulative, means[0], means[1], means[2], means[3],
               means[4], fraction)
        rows.append(row)
        log(f"{k:4d} {cumulative:9.5f} {means[0]:9.5f} {means[1]:8.4f} "
            f"{means[2]:8.4f} {means[3]:10.4f} {fraction:10.4f}")
    roundtrip_mean = np.nanmean(np.asarray(roundtrip_results), axis=0)
    roundtrip_fraction = roundtrip_mean[4] / true_dvdt_median
    log("")
    log("AT+slope warp interpolation floor:")
    log(f"  Vm RelL2 {roundtrip_mean[0]:.6f} | MAE {roundtrip_mean[1]:.5f} mV | "
        f"AT MAE {roundtrip_mean[2]:.5f} ms | slope MAE "
        f"{roundtrip_mean[3]:.5f} mV/ms | dVdt fraction {roundtrip_fraction:.5f}")

    result_csv = os.path.join(args.out, "recon_vs_k.csv")
    with open(result_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("K", "cumEVR", "vm_rel_l2", "vm_mae_mv",
                         "at_mae_ms", "slope_mae_mv_per_ms", "dvdt_median",
                         "dvdt_fraction"))
        writer.writerows(rows)
    with open(os.path.join(args.out, "roundtrip.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("vm_rel_l2", "vm_mae_mv", "at_mae_ms",
                         "slope_mae_mv_per_ms", "dvdt_median", "dvdt_fraction"))
        writer.writerow((*roundtrip_mean, roundtrip_fraction))

    n_save = min(args.save_modes, n_frames)
    np.savez(args.basis_out, node_template=node_template,
             residual_mean=residual_mean, components=components[:, :n_save],
             eigval=eigval[:n_save], evr=evr[:n_save],
             cum_evr=cum_evr[:n_save], time=time_ms,
             reference_at=np.float32(reference_at),
             reference_slope=np.float32(reference_slope),
             at_threshold=np.float32(args.at_threshold),
             slope_window_ms=np.float32(args.slope_window_ms),
             inner_ms=np.float32(args.inner_ms),
             outer_ms=np.float32(args.outer_ms),
             n_train=np.int64(args.n_train),
             decoder=np.asarray("AT_shift_plus_local_monotone_slope_warp"))

    rows_array = np.asarray(rows)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].semilogy(rows_array[:, 0], rows_array[:, 3], "o-", label="AT+slope PCA")
    axes[1].plot(rows_array[:, 0], rows_array[:, 7], "o-", label="AT+slope PCA")
    axes[2].semilogy(np.arange(1, min(50, len(evr)) + 1),
                     evr[:min(50, len(evr))], "o-")
    if os.path.exists(args.ordinary_aligned_csv):
        ordinary = np.genfromtxt(args.ordinary_aligned_csv, delimiter=",",
                                 names=True)
        axes[0].semilogy(ordinary["K"], ordinary["vm_mae"], "--",
                         label="AT-only aligned PCA")
        axes[1].plot(ordinary["K"], ordinary["dvdt_frac"], "--",
                     label="AT-only aligned PCA")
    axes[0].set_ylabel("V_m MAE (mV)")
    axes[1].set_ylabel("median max-dV/dt fraction")
    axes[2].set_ylabel("explained variance ratio")
    for axis in axes:
        axis.set_xlabel("PCA modes K"); axis.grid(alpha=0.3); axis.legend()
    figure.tight_layout(); figure.savefig(os.path.join(args.out, "recon_vs_k.png"),
                                          dpi=160); plt.close(figure)
    if first_trace is not None:
        save_trace_plot(os.path.join(args.out, "test_traces.png"), time_ms,
                        *first_trace)

    log("")
    log(f"saved table : {result_csv}")
    log(f"saved basis : {args.basis_out}")
    log(f"plots       : {args.out}/recon_vs_k.png, test_traces.png")
    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write("\n".join(log_lines) + "\n")
    log("done.")


if __name__ == "__main__":
    main()
