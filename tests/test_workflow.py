"""Small CPU checks: leakage, legacy numerical parity and complete workflows."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from Geo_DONet.artifacts import dated_directory, digest
from Geo_DONet.config import parse_config
from Geo_DONet.data_handling.loading import fold_manifests, load_prepared
from Geo_DONet.data_handling.splits import make_splits
from Geo_DONet.decoder import DifferentiablePhaseDecoder, decode_features
from Geo_DONet.main import run
from Geo_DONet.prepare_dataset import prepare


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT/'Geo_DONet_experiments/Geo_DeepONet_PCA'


def legacy_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='geodonet_clean_', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(10)
        self.theta = rng.normal(size=(8, 3)).astype(np.float32)
        self.coords = rng.uniform(size=(7, 4)).astype(np.float32)
        self.time = np.arange(25, dtype=np.float32) * 2
        activation = 12 + np.arange(8)[:, None]*0.6 + np.arange(7)[None, :]*0.8
        width = 2 + rng.uniform(size=(8, 7))
        self.vm = (-85 + (115 + rng.normal(size=(8, 7, 1))*2) /
                   (1 + np.exp(-(self.time[None, None, :] - activation[..., None])/width[..., None]))).astype(np.float32)
        self.raw = self.root / 'raw.npz'
        self.write_raw(self.raw, self.vm)

    def write_raw(self, path, vm):
        np.savez_compressed(path, theta=self.theta, coords=self.coords, time=self.time, vm=vm,
                            case_names=np.array([f'heart_{i}' for i in range(8)]))

    def prep(self, source=None, folds=1):
        args = parse_config(True, ['--input', str(source or self.raw), '--output', str(self.root/'prepared'),
                                  '--n-train', '4', '--n-val', '2', '--folds', str(folds),
                                  '--n-components', '2', '--chunk-nodes', '3'])
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(args)

    def train_args(self, manifest, mode='train'):
        return parse_config(False, ['--'+mode, '--input', str(manifest), '--output', str(self.root/'runs'),
                                    '--epochs', '2', '--width', '8', '--depth', '2', '--batch-size', '2',
                                    '--nodes-per-step', '4', '--val-nodes', '5', '--val-every', '1',
                                    '--n-viz', '0', '--device', 'cpu', '--patience', '0', '--chunk-nodes', '3'])

    def test_holdout_waveforms_do_not_fit_preprocessing(self):
        first = fold_manifests(self.prep())[0]
        changed = self.vm.copy()
        changed[4:] = changed[4:] * 1.15 + 2
        other = self.root/'changed.npz'
        self.write_raw(other, changed)
        second = fold_manifests(self.prep(other))[0]
        for name in ('decoder_basis.npz', 'normalizer.npz'):
            with np.load(first.parent/name) as a, np.load(second.parent/name) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    @unittest.skipUnless((LEGACY/'utils.py').is_file(), 'Legacy test-ground source is unavailable')
    def test_decoder_forward_and_gradient_match_legacy(self):
        manifest = fold_manifests(self.prep())[0]
        old = legacy_module('legacy_phase_helpers', LEGACY/'utils.py')
        with np.load(manifest.parent/'decoder_basis.npz') as archive:
            basis = {key: archive[key] for key in archive.files}
        _, data, _ = load_prepared(manifest)
        all_features = np.array(data['targets'][:2])
        np.testing.assert_array_equal(decode_features(all_features[0], basis), old.decode_features(all_features[0], basis))
        physical = all_features[:, :4]
        new_features = torch.tensor(physical, requires_grad=True)
        old_features = torch.tensor(physical, requires_grad=True)
        new_output = DifferentiablePhaseDecoder(basis, 2)(new_features, np.arange(4))
        old_output = old.DifferentiablePhaseDecoder(basis, 2)(old_features, np.arange(4))
        torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
        new_output.square().mean().backward()
        old_output.square().mean().backward()
        torch.testing.assert_close(new_features.grad, old_features.grad, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(new_features.grad).all())

    @unittest.skipUnless((LEGACY/'main.py').is_file(), 'Legacy test-ground source is unavailable')
    def test_training_checkpoint_matches_legacy(self):
        prepared = self.prep()
        manifest = fold_manifests(prepared)[0]
        _, data, _ = load_prepared(manifest)
        compact = self.root/'compact.npz'
        np.savez(compact, theta=self.theta, coords=self.coords, time=self.time, targets=data['targets'],
                 target_names=np.array(['activation_time_ms', 'pca_1', 'pca_2']))
        with contextlib.redirect_stdout(io.StringIO()):
            folder = run(self.train_args(prepared))
        checkpoint = folder/'checkpoints/best.pt'
        legacy_checkpoint = self.root/'legacy.pt'
        command = [sys.executable, str(LEGACY/'main.py'),
                   '--data', str(compact), '--basis', str(manifest.parent/'decoder_basis.npz'),
                   '--vm-data', str(self.raw), '--model-path', str(legacy_checkpoint),
                   '--out-dir', str(self.root/'legacy'), '--n-train', '4', '--n-val', '2',
                   '--n-components', '2', '--epochs', '2', '--batch-size', '2',
                   '--nodes-per-step', '4', '--val-nodes', '5', '--val-every', '1',
                   '--width', '8', '--depth', '2', '--device', 'cpu', '--patience', '0']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = torch.load(checkpoint, weights_only=False)
        expected = torch.load(legacy_checkpoint, weights_only=False)
        for key, value in expected['model_state_dict'].items():
            torch.testing.assert_close(actual['model_state_dict'][key], value, rtol=1e-6, atol=1e-7)

    def test_full_workflow_and_parallel_exports(self):
        prepared = self.prep()
        manifest = fold_manifests(prepared)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            folder = run(self.train_args(prepared))
        checkpoint = folder/'checkpoints/best.pt'
        args = parse_config(False, ['--evaluate', '--chkpt', str(checkpoint), '--output', str(self.root/'eval'),
                                    '--device', 'cpu', '--n-viz', '0', '--chunk-nodes', '3'])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluated = run(args)
        self.assertEqual(evaluated, self.root/'eval')
        self.assertTrue((folder/'config.json').exists())
        summary = json.loads((evaluated/'summary.json').read_text())
        self.assertEqual(summary['n_cases'], 2)
        self.assertIsNotNone(summary['metrics']['vm_mae_mv']['mean'])
        predict_raw = self.root/'predict.npz'
        np.savez(predict_raw, theta=self.theta[:2], coords=self.coords, time=self.time,
                 case_names=np.array(['new_a','new_b']))
        args = parse_config(False, ['--predict', '--chkpt', str(checkpoint), '--input', str(predict_raw),
                                    '--output', str(self.root/'predict'), '--device', 'cpu'])
        with contextlib.redirect_stdout(io.StringIO()):
            predicted = run(args)
        with np.load(predicted/'predicted_features.npz') as result:
            self.assertEqual(result['pred'].shape, (2, 7, 3))
            self.assertNotIn('true', result.files)
        # Saved compact features are sufficient for both visualization paths.
        import meshio
        from Geo_DONet.utils.export_vtu import main as export_vtu
        from Geo_DONet.utils.export_svg import main as export_svg
        mesh_path = self.root/'canonical.vtu'
        meshio.write_points_cells(mesh_path, self.coords[:, :3],
                                 [('tetra', np.array([[0, 1, 2, 3], [3, 4, 5, 6]]))],
                                 point_data={key: self.coords[:, k] for k, key in enumerate(('ab','rt','tm','tv'))})
        common = ['--input', str(evaluated/'predicted_features.npz'),
                  '--mesh', str(mesh_path)]
        with contextlib.redirect_stdout(io.StringIO()):
            vtu_dir = export_vtu(common + ['--frame-stride', '12'])
            svg_dir = export_svg(common + ['--times', '10', '30'])
            parallel_dir = export_vtu(common + [
                '--frame-stride', '12', '--workers', '2',
                '--output', str(evaluated/'visualization/vtu_parallel'),
            ])
        self.assertEqual(vtu_dir, evaluated/'visualization/vtu')
        self.assertEqual(svg_dir, evaluated/'visualization/svg/case_0006')
        export_info = json.loads((vtu_dir/'export_info.json').read_text())
        self.assertEqual([entry['case'] for entry in export_info['cases']], [6, 7])
        for case in [6, 7]:
            case_folder = vtu_dir/f'case_{case:04d}'
            parallel_case = parallel_dir/f'case_{case:04d}'
            self.assertTrue((case_folder/'Vm.pvd').exists())
            self.assertEqual((case_folder/'Vm.pvd').read_text(), (parallel_case/'Vm.pvd').read_text())
            self.assertEqual((case_folder/'neighbour_metrics.csv').read_text(),
                             (parallel_case/'neighbour_metrics.csv').read_text())
            for serial_frame in sorted((case_folder/'series').glob('*.vtu')):
                serial_mesh = meshio.read(serial_frame)
                parallel_mesh = meshio.read(parallel_case/'series'/serial_frame.name)
                np.testing.assert_array_equal(parallel_mesh.points, serial_mesh.points)
                np.testing.assert_array_equal(parallel_mesh.cells[0].data, serial_mesh.cells[0].data)
                for key in serial_mesh.point_data:
                    np.testing.assert_array_equal(parallel_mesh.point_data[key], serial_mesh.point_data[key])
            exported = meshio.read(case_folder/'series/frame_0000.vtu')
            np.testing.assert_array_equal(exported.points, self.coords[:, :3])
            np.testing.assert_array_equal(
                exported.cells[0].data, np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
            )
            fields = exported.point_data
            for name in ('Vm_pred', 'Vm_true', 'Vm_abs_error', 'Vm_signed_error',
                         'ndiff_pred', 'ndiff_true', 'ndiff_abs_error',
                         'signed_pred', 'signed_true', 'signed_abs_error', 'neighbour_count'):
                self.assertIn(name, fields)
            np.testing.assert_allclose(fields['Vm_true'], self.vm[case, :, 0])
            np.testing.assert_allclose(fields['Vm_abs_error'], np.abs(fields['Vm_pred']-fields['Vm_true']))
            np.testing.assert_allclose(fields['ndiff_abs_error'], np.abs(fields['ndiff_pred']-fields['ndiff_true']))
            np.testing.assert_allclose(fields['signed_error'], fields['signed_pred']-fields['signed_true'])
            self.assertTrue((case_folder/'neighbour_metrics.csv').exists())

        # Wrong mesh numbering must fail before exporting a misleading comparison.
        wrong_mesh_path = self.root/'wrong_order.vtu'
        meshio.write_points_cells(wrong_mesh_path, self.coords[:, :3],
                                 [('tetra', np.array([[0, 1, 2, 3], [3, 4, 5, 6]]))],
                                 point_data={key: self.coords[::-1, k] for k, key in enumerate(('ab','rt','tm','tv'))})
        with self.assertRaisesRegex(ValueError, 'node order'):
            export_vtu(common + ['--mesh', str(wrong_mesh_path)])
        self.assertTrue(list(svg_dir.glob('*Vm_pred.svg')))
        self.assertTrue((svg_dir/'Vm_traces.svg').exists())
        # Artifact modification is rejected before a new inference is performed.
        with (manifest.parent/'splits.json').open('a') as handle:
            handle.write(' ')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            load_prepared(manifest)

    def test_cv_isolation_coverage_and_pooling(self):
        prepared = self.prep(folds=2)
        paths = fold_manifests(prepared)
        splits = [load_prepared(path)[2] for path in paths]
        self.assertEqual(sorted(sum([s['test'] for s in splits], [])), list(range(8)))
        for split in splits:
            self.assertFalse(set(split['train']) & set(split['test']))
        with contextlib.redirect_stdout(io.StringIO()):
            folder = run(self.train_args(prepared, 'cv'))
        self.assertEqual(json.loads((folder/'summary.json').read_text())['n_cases'], 8)
        self.assertEqual(len(list(folder.glob('fold_*/checkpoints/best.pt'))), 2)

    def test_configuration_and_storage_contract(self):
        configuration = self.root/'settings.json'
        configuration.write_text(json.dumps(dict(input='raw.npz', output='runs', epochs=8, width=12)))
        args = parse_config(False, ['--train', '--config', str(configuration), '--epochs', '3'])
        self.assertEqual(args.input, str(self.raw))
        self.assertEqual(args.output, str(self.root/'runs'))
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.width, 12)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_config(False, ['--train', '--input', str(self.raw)])
        with self.assertRaises(ValueError):
            dated_directory(ROOT/'Geo_DONet/results')
        a = dated_directory(self.root/'dated')
        b = dated_directory(self.root/'dated')
        self.assertNotEqual(a, b)
        order = np.arange(125)
        np.random.RandomState(42).shuffle(order)
        actual = make_splits(125, 95, 5, 5, 42)
        self.assertEqual(actual[0]['test'], order[:25].tolist())
        self.assertEqual(actual[0]['val'], order[-5:].tolist())

    def test_named_run_and_evaluation_beside_checkpoint(self):
        prepared = self.prep()
        args = self.train_args(prepared)
        args.name = 'pca_baseline'
        with contextlib.redirect_stdout(io.StringIO()):
            model_folder = run(args)
        self.assertEqual(model_folder, self.root/'runs/pca_baseline')
        training_config = (model_folder/'config.json').read_bytes()

        evaluation_args = parse_config(False, [
            '--evaluate', '--chkpt', str(model_folder/'checkpoints/best.pt'),
            '--device', 'cpu', '--n-viz', '0',
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation_folder = run(evaluation_args)
        self.assertEqual(evaluation_folder, model_folder/'evaluation')
        self.assertTrue((evaluation_folder/'predicted_features.npz').exists())
        self.assertTrue((evaluation_folder/'config.json').exists())
        self.assertEqual((model_folder/'config.json').read_bytes(), training_config)
        self.assertFalse((evaluation_folder/'evaluation').exists())

        # Rerunning must not overwrite prior evaluation/exports or training metadata.
        with self.assertRaises(FileExistsError):
            run(parse_config(False, [
                '--evaluate', '--chkpt', str(model_folder/'checkpoints/best.pt'), '--device', 'cpu',
            ]))
        with contextlib.redirect_stdout(io.StringIO()):
            labelled = run(parse_config(False, [
                '--evaluate', '--chkpt', str(model_folder/'checkpoints/best.pt'),
                '--name', 'comparison', '--device', 'cpu', '--n-viz', '0',
            ]))
        self.assertEqual(labelled, model_folder/'evaluation_comparison')

        # Moving the complete project tree retains relative preparation links.
        moved = self.root/'moved_project'
        moved.mkdir()
        (self.root/'prepared').rename(moved/'prepared')
        (self.root/'runs').rename(moved/'runs')
        moved_model = moved/'runs/pca_baseline'
        with contextlib.redirect_stdout(io.StringIO()):
            relocated = run(parse_config(False, [
                '--evaluate', '--chkpt', str(moved_model/'checkpoints/best.pt'),
                '--name', 'relocated', '--device', 'cpu', '--n-viz', '0',
            ]))
        self.assertTrue((relocated/'summary.json').exists())

    def test_cli_help_without_data_or_ml_stack(self):
        commands = [
            ('main.py', '-h', 'Training and validation'),
            ('prepare_dataset.py', '--help', 'Heart splits and waveform PCA'),
            ('utils/export_vtu.py', '-h', '--cases'),
        ]
        for script, help_flag, expected_group in commands:
            result = subprocess.run(
                [sys.executable, '-S', str(ROOT/'Geo_DONet'/script), help_flag],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('-h, --help', result.stdout)
            self.assertIn(expected_group, result.stdout)
            self.assertIn('Examples:', result.stdout)
            if script == 'prepare_dataset.py':
                self.assertNotIn('--epochs', result.stdout)

if __name__ == '__main__':
    unittest.main()
