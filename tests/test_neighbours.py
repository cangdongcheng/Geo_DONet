"""Check mesh-neighbour definitions against a hand-calculated tetrahedron."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Geo_DONet_clean.utils.export_common import resolve_export_manifest, select_cases
from Geo_DONet_clean.utils.neighbours import build_neighbour_graph, neighbour_fields


class NeighbourTests(unittest.TestCase):
    def test_parallel_writer_error_does_not_publish_complete_series(self):
        import meshio
        from Geo_DONet_clean.utils.export_vtu import write_case

        tetrahedra = np.array([[0, 1, 2, 3]])
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        mesh = meshio.Mesh(points, [('tetra', tetrahedra)],
                           point_data={key: points[:, 0] for key in ('ab', 'rt', 'tm', 'tv')})
        graph = build_neighbour_graph(tetrahedra, 4)
        fields = {'Vm_pred': np.arange(16, dtype=np.float32).reshape(4, 4)}
        original_writer = meshio.write_points_cells

        def failing_writer(path, *args, **kwargs):
            if Path(path).name == 'frame_0001.vtu':
                raise OSError('Synthetic write failure')
            return original_writer(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            output = Path(directory)
            with patch.object(meshio, 'write_points_cells', side_effect=failing_writer):
                with self.assertRaisesRegex(OSError, 'Synthetic write failure'):
                    write_case(output, mesh, tetrahedra, np.arange(4), fields, graph, 1, workers=2)
            self.assertFalse((output/'Vm.pvd').exists())

    def test_relocated_preparation_keeps_the_saved_identity(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            folder = Path(directory)
            original = folder/'manifest.json'
            original.write_text('{"kind": "prepared_fold"}\n')
            identity = hashlib.sha256(original.read_bytes()).hexdigest()
            moved = folder/'moved.json'
            original.rename(moved)
            with self.assertRaisesRegex(FileNotFoundError, '--prepared-manifest'):
                resolve_export_manifest(original, identity)
            self.assertEqual(resolve_export_manifest(original, identity, moved), moved)
            moved.write_text('{"kind": "different_preparation"}\n')
            with self.assertRaisesRegex(ValueError, 'identity disagree'):
                resolve_export_manifest(original, identity, moved)

    def test_unique_edges_and_known_voltage_contrasts(self):
        # Repeated/shared tetrahedra must not weight a neighbour multiple times.
        tetrahedra = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]])
        graph = build_neighbour_graph(tetrahedra, n_nodes=5)
        self.assertEqual(len(graph[0]), 6)
        np.testing.assert_array_equal(graph[2], [3, 3, 3, 3, 0])

        voltage = np.array([0., 1., 3., 6., 99.])
        unsigned, signed, energy = neighbour_fields(voltage, graph)
        np.testing.assert_allclose(unsigned[:4], [10/3, 8/3, 8/3, 14/3])
        np.testing.assert_allclose(signed[:4], [-10/3, -2, 2/3, 14/3])
        self.assertEqual(energy, 84.)
        self.assertTrue(np.isnan(unsigned[4]))
        self.assertTrue(np.isnan(signed[4]))

        # A spatially uniform voltage offset cannot change neighbour differences.
        shifted_unsigned, shifted_signed, shifted_energy = neighbour_fields(voltage + 25, graph)
        np.testing.assert_array_equal(shifted_unsigned, unsigned)
        np.testing.assert_array_equal(shifted_signed, signed)
        self.assertEqual(shifted_energy, energy)

        constant_unsigned, constant_signed, constant_energy = neighbour_fields(np.full(5, -85.), graph)
        np.testing.assert_array_equal(constant_unsigned[:4], 0)
        np.testing.assert_array_equal(constant_signed[:4], 0)
        self.assertEqual(constant_energy, 0)

    def test_case_selection_is_explicit_and_defaults_to_two(self):
        context = {'indices': np.array([100, 101, 102])}
        self.assertEqual(select_cases(context, default_count=2), [100, 101])
        self.assertEqual(select_cases(context, cases=[102, 100]), [102, 100])
        self.assertEqual(select_cases(context, case=101), [101])
        with self.assertRaisesRegex(ValueError, 'absent'):
            select_cases(context, cases=[99])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            select_cases(context, cases=[100, 100])

    def test_invalid_connectivity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            build_neighbour_graph(np.array([[0, 1, 2, 9]]), n_nodes=4)
        with self.assertRaisesRegex(ValueError, 'repeated'):
            build_neighbour_graph(np.array([[0, 1, 2, 2]]), n_nodes=4)


if __name__ == '__main__':
    unittest.main()
