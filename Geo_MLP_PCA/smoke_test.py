"""Tiny CPU integration test; synthetic data only, no cardiac archive access.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python smoke_test.py
Exercises MLP/legacy checkpoint loading, decoder gradients, train/test CLI,
and five-fold CV including decoder split isolation. Temporary files auto-clean.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "Geo_DeepONet_PCA"
sys.path.insert(0, str(SHARED))
from opnn import FeatureDeepONet, FeatureMLP, build_feature_model
from utils import DifferentiablePhaseDecoder, activation_time, decode_features


def run(script, arguments, cwd):
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script), *map(str, arguments)],
                            cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    print(f"passed: {script.parent.name}/{script.name} {'test' if '--test-model' in arguments else 'train'}")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    for architecture, width, count in (("mlp", 200, 134806),
                                        ("mlp", 280, 255926),
                                        ("deeponet", 200, 255606)):
        model = build_feature_model(dict(architecture=architecture, width=width))
        assert sum(p.numel() for p in model.parameters()) == count

    theta = torch.randn(2, 3)
    coords = torch.randn(12, 4)
    for original in (FeatureMLP(3, 4, 8, 2, 3), FeatureDeepONet(3, 4, 8, 2, 3)):
        restored = build_feature_model(original.config())
        restored.load_state_dict(original.state_dict())
        torch.testing.assert_close(original(theta, coords), restored(theta, coords))
    model = FeatureMLP(3, 4, 8, 2, 3)
    torch.testing.assert_close(model(theta, coords)[0, 0],
                               model.net(torch.cat((theta[0], coords[0]))))

    times = np.arange(21, dtype=np.float32)
    template = np.tile(-30 + 55 * np.tanh((times - 8) / 1.5), (12, 1)).astype(np.float32)
    components = np.linalg.qr(rng.normal(size=(21, 2)))[0].astype(np.float32)
    basis = dict(node_template=template, residual_mean=np.zeros(21, np.float32),
                 components=components, time=times, reference_at=8.0, at_threshold=-10.0)
    decoder = DifferentiablePhaseDecoder(basis, 2)
    physical = model(theta, coords) + torch.tensor([8., 0., 0.])
    physical.retain_grad()
    loss = decoder(physical, np.arange(12)).square().mean()
    loss.backward()
    assert torch.isfinite(physical.grad).all()
    assert (physical.grad.abs().sum(dim=(0, 1)) > 0).all()
    assert model.net[-1].weight.grad.abs().sum() > 0

    with tempfile.TemporaryDirectory(prefix="geomlp_pca_smoke_") as temporary:
        root = Path(temporary)
        features = rng.normal(size=(10, 12, 3)).astype(np.float32)
        features[..., 0] = 8 + features[..., 0] * 0.6
        vm = np.stack([decode_features(f, basis, 4) for f in features])
        features[..., 0] = np.stack([activation_time(v, times) for v in vm])
        np.savez(root / "features.npz", theta=rng.normal(size=(10, 3)).astype(np.float32),
                 coords=rng.random((12, 4)).astype(np.float32), targets=features,
                 target_names=np.array(["activation_time_ms", "pca_1", "pca_2"]),
                 time=times, case_names=np.array([f"synthetic{i}" for i in range(10)]))
        np.savez(root / "vm.npz", vm=vm, time=times)
        np.savez(root / "basis.npz", **basis, evr=np.array([0.7, 0.3]))
        common = ["--device", "cpu", "--data", root / "features.npz",
                  "--vm-data", root / "vm.npz", "--epochs", "2", "--width", "8",
                  "--depth", "2", "--n-components", "2", "--batch-size", "2",
                  "--nodes-per-step", "4", "--val-nodes", "4", "--val-every", "1"]
        fixed = [*common, "--basis", root / "basis.npz", "--n-train", "6", "--n-val", "2"]
        for architecture, script in (("mlp", HERE / "main.py"),
                                       ("deeponet", SHARED / "main.py")):
            checkpoint = root / f"{architecture}.pt"
            run(script, [*fixed, "--model-path", checkpoint,
                         "--out-dir", root / f"{architecture}_train"], root)
            run(script, [*fixed, "--model-path", checkpoint, "--test-model",
                         "--skip-plots", "--out-dir", root / f"{architecture}_test"], root)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert saved["config"].get("architecture", "deeponet") == architecture
            output = root / f"{architecture}_test"
            with np.load(output / "test_features.npz") as archive:
                assert archive["pred"].shape == (2, 12, 3)
                assert np.isfinite(archive["pred"]).all()
            assert "decoded V_m:" in (output / "test_summary.txt").read_text()

        cv = root / "cv"
        run(HERE / "main_cv.py", [*common, "--n-val", "2", "--reference-at", "8",
                                   "--out-dir", cv], root)
        with np.load(cv / "cv_results.npz") as pooled:
            np.testing.assert_array_equal(np.sort(pooled["test_idx"]), np.arange(10))
            assert np.isfinite(pooled["vm_mae"]).all()
            for fold in range(5):
                checkpoint = torch.load(cv / f"fold_{fold}" / "model.pt",
                                        map_location="cpu", weights_only=False)
                assert checkpoint["config"]["architecture"] == "mlp"
                test = pooled["test_idx"][pooled["fold_id"] == fold]
                with np.load(cv / f"fold_{fold}" / "basis.npz") as fitted:
                    np.testing.assert_array_equal(fitted["train_idx"], checkpoint["train_idx"])
                    assert not set(fitted["train_idx"]) & set(test)
                    assert not set(fitted["train_idx"]) & set(checkpoint["val_idx"])
        assert "Geo_MLP_PCA 5-fold" in (cv / "summary.txt").read_text()
    print("PASS: model counts, legacy loading, gradients, train/test and leakage-safe CV")


if __name__ == "__main__":
    main()
