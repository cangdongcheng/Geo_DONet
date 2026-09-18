"""Tiny synthetic CPU tests for PCA-to-ECG; no real cardiac archive is loaded."""
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from pca_chain import evaluate, train, metrics, to_12lead, match_times, pca_utils


def expect_error(call, exception, message):
    try:
        call()
    except exception as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected {exception.__name__}: {message}")


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    electrode = np.array([[3., 1., 5., 999., 8., 9., 10., 11., 12., 13.]])
    expected = np.array([[2., 4., 2., -3., 0., 3., 5., 6., 7., 8., 9., 10.]])
    np.testing.assert_allclose(to_12lead(electrode), expected)
    electrode[:, 3] = -999
    np.testing.assert_allclose(to_12lead(electrode), expected)
    assert np.isnan(metrics(np.ones((5, 2)), np.ones((5, 2)))["pcc"])
    np.testing.assert_array_equal(match_times(np.arange(21), np.arange(0, 21, 5)),
                                  [0, 5, 10, 15, 20])
    expect_error(lambda: match_times(np.arange(21), np.array([21])), ValueError, "subset")
    utils = pca_utils()
    times = np.arange(21, dtype=np.float32)
    basis = dict(node_template=np.tile(-30 + 55*np.tanh((times-8)/1.5), (12, 1)).astype(np.float32),
                 components=np.linalg.qr(rng.normal(size=(21, 2)))[0].astype(np.float32),
                 residual_mean=np.zeros(21, np.float32), time=times,
                 reference_at=np.float32(8), at_threshold=np.float32(-10))
    features = rng.normal(size=(10, 12, 3)).astype(np.float32)
    features[..., 0] = features[..., 0] * .5 + 8
    full_vm = np.stack([utils.decode_features(f, basis, 4) for f in features])
    vm = full_vm[:, :, ::5].copy()
    ecg = (vm.transpose(0, 2, 1) @ rng.normal(size=(12, 10)) * .001).astype(np.float32)
    names = np.array([f"synthetic{i}" for i in range(10)])
    with tempfile.TemporaryDirectory(prefix="pca_ecg_smoke_") as temporary:
        root = Path(temporary)
        data_path = root / "data.npz"
        np.savez(data_path, vm=vm, ecg=ecg, time=times[::5], case_names=names,
                 coords=rng.random((12, 4)).astype(np.float32))
        basis_path = root / "basis.npz"
        np.savez(basis_path, **basis)
        feature_path = root / "features.npz"
        np.savez(feature_path, pred=features[8:], test_indices=np.array([8, 9]),
                 case_names=names[8:], target_names=["activation_time_ms", "pca_1", "pca_2"])
        for model in ("linear", "mlp", "temporal_conv"):
            checkpoint_path = root / f"{model}.pt"
            args = SimpleNamespace(command="train", epochs=2, batch_size=2, n_train=6,
                                   n_val=2, lr=1e-4, seed=42, model=model,
                                   checkpoint=checkpoint_path, data=data_path,
                                   out_dir=root / f"train_{model}", device="cpu")
            train(args, torch.device("cpu"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for label, array in (("vm", vm), ("ecg", ecg)):
                np.testing.assert_allclose(checkpoint["normalization"][f"{label}_min"], array[:6].min())
                np.testing.assert_allclose(checkpoint["normalization"][f"{label}_std"],
                                          array[:6].std(dtype=np.float64), rtol=1e-6)
            assert set(checkpoint["test_idx"]) == {8, 9}
            expect_error(lambda: train(args, torch.device("cpu")), FileExistsError, "already exists")
            evaluation = SimpleNamespace(command="evaluate", checkpoint=checkpoint_path,
                                         data=None, features=feature_path, basis=basis_path,
                                         chunk_nodes=4, cases=None, n_viz=1,
                                         out_dir=root / f"test_{model}", device="cpu")
            evaluate(evaluation, torch.device("cpu"))
            with np.load(evaluation.out_dir / "ecg_predictions.npz") as results:
                assert results["pca_chain_ecg12"].shape == (2, 5, 12)
                np.testing.assert_allclose(results["pca_chain_ecg10"], results["transfer_gt_ecg10"], atol=1e-6)
                np.testing.assert_allclose(results["true_ecg12"], to_12lead(ecg[8:]))
            assert (evaluation.out_dir / "case8_12lead.png").is_file()
            assert "pca_chain_vs_simulation" in (evaluation.out_dir / "test_summary.txt").read_text()
        # Reject a feature archive that includes an ECG-transfer training heart.
        bad_path = root / "bad_features.npz"
        np.savez(bad_path, pred=features[[0, 9]], test_indices=[0, 9],
                 case_names=names[[0, 9]], target_names=["activation_time_ms", "pca_1", "pca_2"])
        evaluation.features = bad_path
        expect_error(lambda: evaluate(evaluation, torch.device("cpu")), ValueError, "overlap")
    print("PASS: all 3 architectures, normalization, held-out guard, f601-to-f121 sampling, "
          "identity chain, metrics, lead conversion and plots (synthetic CPU only).")


if __name__ == "__main__":
    main()
