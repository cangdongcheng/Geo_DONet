"""Export two saved cases on the reference geometry: Vm, GT errors and neighbour contrasts."""
from pathlib import Path

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
from concurrent.futures import ThreadPoolExecutor
import time

from Geo_DONet_clean.artifacts import new_directory, write_json
from Geo_DONet_clean.utils.export_common import (
    decode_export_case,
    export_parser,
    load_export_context,
    select_cases,
)


def write_case(output, mesh, tetrahedra, times, fields, graph, frame_stride, workers=1):
    """Share one reconstructed heart across workers; write independent frame files."""
    import meshio
    import numpy as np
    from Geo_DONet_clean.utils.neighbours import neighbour_fields

    series = output / 'series'
    series.mkdir()
    entries = []
    has_truth = 'Vm_true' in fields
    frames = list(enumerate(range(0, len(times), frame_stride)))
    worker_count = min(workers, len(frames))
    coordinate_keys = ('ab', 'rt', 'tm', 'tv')
    reference_fields = {key: mesh.point_data[key] for key in coordinate_keys}
    reference_fields['neighbour_count'] = graph[2].astype(np.int32)

    def write_frame(task):
        index, frame = task

        # 1. Each worker owns its field dictionary, sharing read-only geometry/Vm.
        point_data = dict(reference_fields)
        point_data.update({key: value[:, frame] for key, value in fields.items()})

        # 2. Compute this frame's neighbour contrasts and physical-unit errors.
        predicted_unsigned, predicted_signed, predicted_energy = neighbour_fields(fields['Vm_pred'][:, frame], graph)
        point_data['ndiff_pred'] = predicted_unsigned
        point_data['signed_pred'] = predicted_signed
        metrics = {'time_ms': float(times[frame]), 'edge_energy_pred_mv2': predicted_energy}

        if has_truth:
            true_unsigned, true_signed, true_energy = neighbour_fields(fields['Vm_true'][:, frame], graph)
            unsigned_error = predicted_unsigned - true_unsigned
            signed_error = predicted_signed - true_signed
            point_data.update({
                'ndiff_true': true_unsigned,
                'ndiff_signed_error': unsigned_error,
                'ndiff_abs_error': np.abs(unsigned_error),
                'signed_true': true_signed,
                'signed_error': signed_error,
                'signed_abs_error': np.abs(signed_error),
            })
            connected = graph[2] > 0
            metrics.update({
                'edge_energy_true_mv2': true_energy,
                'ndiff_mae_mv': float(np.abs(unsigned_error[connected]).mean()),
                'signed_mae_mv': float(np.abs(signed_error[connected]).mean()),
            })

        # 3. A distinct filename per task permits concurrent compression/writes.
        filename = f'frame_{index:04d}.vtu'
        meshio.write_points_cells(
            series / filename,
            mesh.points,
            [('tetra', tetrahedra)],
            point_data=point_data,
        )
        return filename, float(times[frame]), metrics

    print(f'Writing {len(frames)} VTU frames with {worker_count} worker(s): {output}', flush=True)
    started = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        results = pool.map(write_frame, frames) if pool else map(write_frame, frames)
        with (output / 'neighbour_metrics.csv').open('w', newline='') as handle:
            columns = ['time_ms', 'edge_energy_pred_mv2']
            if has_truth:
                columns += ['edge_energy_true_mv2', 'ndiff_mae_mv', 'signed_mae_mv']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()

            # map yields results in time order even when frame writes finish out of order.
            for completed, (filename, time_ms, metrics) in enumerate(results, start=1):
                entries.append(
                    f'    <DataSet timestep="{time_ms}" group="" part="0" file="series/{filename}"/>'
                )
                writer.writerow(metrics)
                if completed == 1 or completed % 25 == 0 or completed == len(frames):
                    elapsed = time.perf_counter() - started
                    print(f'  Frames {completed}/{len(frames)} complete; {elapsed:.1f} s elapsed', flush=True)
                    handle.flush()
    finally:
        if pool is not None:
            # On Ctrl+C/error, cancel queued frames instead of finishing the whole series.
            pool.shutdown(wait=True, cancel_futures=True)

    # Publish the ParaView collection only after every frame completed successfully.
    (output / 'Vm.pvd').write_text('\n'.join([
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *entries,
        '  </Collection>',
        '</VTKFile>',
        '',
    ]))
    return len(entries)


def main(argv=None):
    parser = export_parser(__doc__, multiple_cases=True)
    parser.add_argument('--frame-stride', type=int, default=5,
                        help='Export every Nth recorded frame (default: 5; use 1 for all frames)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent frame writers sharing one reconstructed Vm field (default: 1; try 8 with 16 allocated CPUs)')
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error('--frame-stride must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')

    import numpy as np
    from Geo_DONet_clean.utils.neighbours import build_neighbour_graph

    # 1. Load shared inputs once and choose two saved cases by default.
    print('Loading predictions, verifying preparation, and reading the reference mesh...', flush=True)
    loading_started = time.perf_counter()
    context = load_export_context(args)
    cases = select_cases(context, case=args.case, cases=args.cases, default_count=2)
    mesh = context['mesh']
    blocks = [cell.data for cell in mesh.cells if cell.type == 'tetra']
    if not blocks:
        raise ValueError('Canonical reference mesh has no tetrahedral cells')
    tetrahedra = np.concatenate(blocks)
    graph = build_neighbour_graph(tetrahedra, len(mesh.points))
    times = context['basis']['time']
    print(f'Loaded {len(mesh.points)} nodes, {len(graph[0])} edges, {len(times)} times '
          f'in {time.perf_counter()-loading_started:.1f} s', flush=True)

    # 2. Decode/export each heart separately to keep host-memory use bounded.
    output = new_directory(args.output or args.input.resolve().parent / 'visualization' / 'vtu')
    records = []
    for case in cases:
        print(f'Reconstructing Vm for case {case}...', flush=True)
        decoding_started = time.perf_counter()
        fields, case_name = decode_export_case(context, case, args.chunk_nodes)
        decoding_seconds = time.perf_counter() - decoding_started
        print(f'Case {case}: Vm reconstruction finished in {decoding_seconds:.1f} s', flush=True)
        case_folder = output / f'case_{case:04d}'
        case_folder.mkdir()
        writing_started = time.perf_counter()
        frame_count = write_case(case_folder, mesh, tetrahedra, times, fields, graph,
                                 args.frame_stride, workers=args.workers)
        record = {
            'case': case,
            'case_name': case_name,
            'pvd': f'{case_folder.name}/Vm.pvd',
            'n_frames': frame_count,
            'decoding_seconds': decoding_seconds,
            'writing_seconds': time.perf_counter() - writing_started,
            'has_ground_truth': 'Vm_true' in fields,
        }
        write_json(case_folder / 'export_info.json', record)
        records.append(record)
        print(f'Case {case} ({case_name}): {case_folder / "Vm.pvd"}', flush=True)
        del fields

    # 3. Describe definitions and units beside the exported ParaView collections.
    write_json(output / 'export_info.json', {
        'features': str(args.input.resolve()),
        'prepared_manifest': str(context['manifest']),
        'mesh': str(args.mesh.resolve()),
        'cases': records,
        'frame_stride': args.frame_stride,
        'workers': args.workers,
        'n_nodes': len(mesh.points),
        'n_edges': len(graph[0]),
        'n_isolated_nodes': int(np.count_nonzero(graph[2] == 0)),
        'units': {'voltage_and_neighbour_fields': 'mV', 'time': 'ms', 'edge_energy': 'mV^2'},
        'definitions': {
            'Vm_signed_error': 'Vm_pred - Vm_true',
            'Vm_abs_error': 'abs(Vm_pred - Vm_true)',
            'ndiff': 'mean_j abs(Vm[i] - Vm[j]) over unique tetrahedral-edge neighbours j',
            'signed': 'Vm[i] - mean_j Vm[j]',
            'ndiff_signed_error': 'ndiff_pred - ndiff_true',
            'ndiff_abs_error': 'abs(ndiff_pred - ndiff_true)',
            'signed_error': 'signed_pred - signed_true',
            'signed_abs_error': 'abs(signed_pred - signed_true)',
            'isolated_nodes': 'neighbour_count is zero and neighbour contrasts are NaN',
        },
        'note': 'Neighbour contrasts are not divided by edge length. The geometry is the canonical reference mesh.',
    })
    print(f'Exported {len(cases)} case(s) to {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
