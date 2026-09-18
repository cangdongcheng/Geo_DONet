"""Create compact [AT, slope, PC1..PCK] targets for the slope decoder."""
import argparse
import os
import time

import numpy as np

from oracle_decoder import activation_time, encode_waveforms, upstroke_slope
from utils import load_basis


DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
BASIS = "/home/svu/e1032484/scratch/pca_at_slope_basis_f601.npz"
OUTPUT = "/home/svu/e1032484/scratch/geo_deeponet_pca_slope_f601_k8.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm-data", default=DATA)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--chunk-nodes", type=int, default=5_000)
    args = parser.parse_args()
    basis = load_basis(args.basis, args.n_components)
    archive = np.load(args.vm_data, allow_pickle=True)
    theta = archive["theta"].astype(np.float32)
    coords = archive["coords"].astype(np.float32)
    time_ms = archive["time"].astype(np.float32)
    vm = archive["vm"]
    if not np.array_equal(time_ms, basis["time"]):
        raise SystemExit("V_m and basis time grids differ")
    n_cases, n_nodes, _ = vm.shape
    targets = np.empty((n_cases, n_nodes, args.n_components + 2),
                       dtype=np.float32)
    target_names = np.asarray(["activation_time_ms", "upstroke_slope_mv_per_ms"]
                              + [f"pca_{k}" for k in range(1, args.n_components + 1)])
    start_time = time.time()
    print(f"preparing {targets.shape} [AT, slope, {args.n_components} PCs]", flush=True)
    for case in range(n_cases):
        at = activation_time(vm[case], time_ms, basis["at_threshold"])
        slope = upstroke_slope(vm[case], time_ms, at,
                               basis["slope_window_ms"])
        if not np.isfinite(at).all() or not np.isfinite(slope).all():
            raise SystemExit(f"case {case}: invalid AT/slope")
        targets[case, :, 0] = at
        targets[case, :, 1] = slope
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            encoded = encode_waveforms(
                vm[case, start:end], at[start:end], slope[start:end], time_ms,
                basis["reference_at"], basis["reference_slope"],
                basis["inner_ms"], basis["outer_ms"])
            residual = (encoded - basis["node_template"][start:end]
                        - basis["residual_mean"])
            targets[case, start:end, 2:] = residual @ basis["components"]
        elapsed = time.time() - start_time
        eta = elapsed / (case + 1) * (n_cases - case - 1)
        print(f"case {case + 1:3d}/{n_cases} | elapsed {elapsed / 60:.1f} min | "
              f"eta {eta / 60:.1f} min", flush=True)

    payload = dict(theta=theta, coords=coords, targets=targets,
                   target_names=target_names, time=time_ms,
                   basis_file=np.asarray(os.path.abspath(args.basis)))
    if "case_names" in archive.files:
        payload["case_names"] = archive["case_names"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved -> {args.output} ({os.path.getsize(args.output) / 1024**2:.1f} MB)")
    print("target ranges on first 95 hearts:")
    for channel, name in enumerate(target_names):
        value = targets[:95, :, channel]
        print(f"  {name:>28}: mean {value.mean(): .5g}, std {value.std(): .5g}, "
              f"range [{value.min():.5g}, {value.max():.5g}]")


if __name__ == "__main__":
    main()
