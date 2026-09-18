"""Export one Geo-DeepONet-PCA prediction and absolute error as VTU/PVD.

This uses the saved ``test_features.npz`` from a completed test run, so model
inference is not repeated. The fixed phase decoder reconstructs V_m for one
test heart, then each selected time frame is written on canonical.vtu with:

    Vm_pred, Vm_true, Vm_abs_error, Vm_signed_error

Open ``Vm_prediction_error.pvd`` in ParaView and colour by any point field.
"""
import argparse
import os

import meshio
import numpy as np

from utils import decode_features, load_decoder_basis, vm_metrics


FEATURES = ("Predictions/geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep/"
            "Test/test_features.npz")
BASIS = "/home/svu/e1032484/scratch/pca_phase_aligned_basis_f601.npz"
VM_DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
MESH = "/home/svu/e1032484/scratch/canonical.vtu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export one decoded Geo-DeepONet-PCA test case to VTU/PVD")
    parser.add_argument("--features", default=FEATURES,
                        help="test_features.npz produced by main.py --test-model")
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--vm-data", default=VM_DATA)
    parser.add_argument("--mesh", default=MESH)
    parser.add_argument("--case", type=int, default=100,
                        help="global heart index from the saved test set")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="write every Nth f601 frame; default 5 gives 121 files")
    parser.add_argument("--chunk-nodes", type=int, default=20_000)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def tetrahedra(mesh):
    blocks = [cell.data for cell in mesh.cells if cell.type == "tetra"]
    if not blocks:
        raise SystemExit("mesh contains no tetrahedral cells")
    return np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]


def write_series(out_dir, points, tetra, time_ms, fields, frame_stride):
    series_dir = os.path.join(out_dir, "series")
    os.makedirs(series_dir, exist_ok=True)
    rows = []
    output_frame = 0
    for frame in range(0, len(time_ms), frame_stride):
        filename = f"frame_{output_frame:04d}.vtu"
        point_data = {
            name: np.asarray(values[:, frame], dtype=np.float32)
            for name, values in fields.items()
        }
        meshio.write_points_cells(
            os.path.join(series_dir, filename), points, [("tetra", tetra)],
            point_data=point_data, binary=True)
        rows.append(
            f'    <DataSet timestep="{float(time_ms[frame])}" group="" '
            f'part="0" file="series/{filename}"/>')
        output_frame += 1
        if output_frame % 20 == 0:
            print(f"  wrote {output_frame} frames", flush=True)
    document = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>', *rows, '  </Collection>', '</VTKFile>',
    ]
    pvd_path = os.path.join(out_dir, "Vm_prediction_error.pvd")
    with open(pvd_path, "w") as handle:
        handle.write("\n".join(document) + "\n")
    return pvd_path, output_frame


def write_summary(out_dir, points, tetra, prediction, truth):
    absolute = np.abs(prediction - truth)
    point_data = {
        "Vm_time_MAE": absolute.mean(axis=1).astype(np.float32),
        "Vm_time_RMSE": np.sqrt(np.mean((prediction - truth) ** 2,
                                         axis=1)).astype(np.float32),
        "Vm_time_max_abs_error": absolute.max(axis=1).astype(np.float32),
    }
    path = os.path.join(out_dir, "error_summary.vtu")
    meshio.write_points_cells(path, points, [("tetra", tetra)],
                              point_data=point_data, binary=True)
    return path


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be positive")
    feature_archive = np.load(args.features, allow_pickle=True)
    required = {"pred", "test_indices"}
    missing = required.difference(feature_archive.files)
    if missing:
        raise SystemExit(f"{args.features} is missing {sorted(missing)}")
    test_indices = feature_archive["test_indices"].astype(int)
    positions = np.where(test_indices == args.case)[0]
    if len(positions) != 1:
        raise SystemExit(f"global case {args.case} is not uniquely present in "
                         f"saved test indices {test_indices.tolist()}")
    position = int(positions[0])
    predicted_features = feature_archive["pred"][position].astype(np.float32)
    n_components = predicted_features.shape[1] - 1
    basis = load_decoder_basis(args.basis, n_components)

    case_name = (str(feature_archive["case_names"][position])
                 if "case_names" in feature_archive.files
                 else f"case{args.case}")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.features), "vtu", case_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"case: global {args.case}, {case_name}")
    print(f"decoding {len(predicted_features)} nodes x {len(basis['time'])} frames ...",
          flush=True)
    prediction = decode_features(predicted_features, basis, args.chunk_nodes)

    print(f"loading ground-truth V_m: {args.vm_data}", flush=True)
    vm_archive = np.load(args.vm_data, allow_pickle=True)
    truth = vm_archive["vm"][args.case].astype(np.float32)
    time_ms = vm_archive["time"].astype(np.float32)
    if prediction.shape != truth.shape:
        raise SystemExit(f"prediction {prediction.shape} and truth {truth.shape} disagree")
    if not np.array_equal(time_ms, basis["time"]):
        raise SystemExit("decoder basis and raw V_m use different time grids")

    rel_l2, mae = vm_metrics(prediction, truth)
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    max_error = float(np.max(np.abs(prediction - truth)))
    print(f"metrics: RelL2 {rel_l2:.6f} | MAE {mae:.4f} mV | "
          f"RMSE {rmse:.4f} mV | max |error| {max_error:.3f} mV")

    print(f"loading canonical mesh: {args.mesh}", flush=True)
    mesh = meshio.read(args.mesh)
    tetra = tetrahedra(mesh)
    if len(mesh.points) != len(truth):
        raise SystemExit(f"mesh has {len(mesh.points)} points but fields have {len(truth)}")
    fields = {
        "Vm_pred": prediction,
        "Vm_true": truth,
        "Vm_abs_error": np.abs(prediction - truth),
        "Vm_signed_error": prediction - truth,
    }
    pvd_path, n_written = write_series(
        out_dir, mesh.points.astype(np.float32), tetra, time_ms, fields,
        args.frame_stride)
    summary_path = write_summary(out_dir, mesh.points.astype(np.float32), tetra,
                                 prediction, truth)
    with open(os.path.join(out_dir, "summary.txt"), "w") as handle:
        handle.write(
            f"case: {args.case} {case_name}\n"
            f"feature file: {args.features}\n"
            f"basis: {args.basis}\n"
            f"V_m data: {args.vm_data}\n"
            f"mesh: {args.mesh}\n"
            f"frames written: {n_written}/{len(time_ms)} "
            f"(stride {args.frame_stride})\n"
            f"V_m RelL2: {rel_l2:.8f}\n"
            f"V_m MAE: {mae:.8f} mV\n"
            f"V_m RMSE: {rmse:.8f} mV\n"
            f"V_m max abs error: {max_error:.8f} mV\n")
    print(f"PVD time series : {pvd_path} ({n_written} frames)")
    print(f"static summary  : {summary_path}")
    print("ParaView: open the .pvd and colour by Vm_pred or Vm_abs_error.")


if __name__ == "__main__":
    main()
