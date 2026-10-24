"""Load saved features, their fitted decoder and the matching canonical mesh."""
import argparse
from pathlib import Path

from ..artifacts import new_directory, write_json
from ..artifacts import resolve_prepared_manifest as resolve_export_manifest


def export_parser(description, multiple_cases=False):
    if multiple_cases:
        examples = (
            'Examples:\n'
            '  python utils/export_vtu.py --input /scratch/run/evaluation/predicted_features.npz '
            '--mesh /scratch/canonical.vtu --cases 100 101\n'
            '  python utils/export_vtu.py --input /scratch/run/evaluation/predicted_features.npz '
            '--mesh /scratch/canonical.vtu --case 100 --frame-stride 1\n'
            '\nWithout --case/--cases, export the first two saved cases (or one if only one exists).\n'
            'ndiff = mean absolute Vm difference to neighbouring mesh nodes.\n'
            'signed = Vm minus the neighbours\' mean Vm. Both have mV units.'
        )
    else:
        examples = (
            'Examples:\n'
            '  python utils/export_svg.py --input /scratch/run/evaluation/predicted_features.npz '
            '--mesh /scratch/canonical.vtu --case 100'
        )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', type=Path, required=True,
                        help='evaluation/predicted_features.npz; no new network inference is needed')
    parser.add_argument('--mesh', type=Path, required=True,
                        help='Canonical reference VTU with matching ab/rt/tm/tv node order')
    parser.add_argument('--prepared-manifest', type=Path,
                        help='New location of the fold manifest if preparation was moved; its saved checksum must match')
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument('--case', type=int,
                       help='Export one case index from the saved predictions')
    if multiple_cases:
        cases.add_argument('--cases', type=int, nargs='+', metavar='INDEX',
                           help='Case indices to export, e.g. --cases 100 101; default: first two saved cases')
    parser.add_argument('--output', type=Path,
                        help='Exact output folder; default: visualization beside the saved features')
    parser.add_argument('--chunk-nodes', type=int, default=20000,
                        help='Node block size for waveform decoding (default: 20000)')
    return parser


def load_export_context(args):
    """Read the mesh/decoder/raw metadata once, even when exporting several hearts."""
    import meshio
    import numpy as np
    from ..data_handling.loading import load_prepared
    from ..decoder import load_decoder_basis

    if args.chunk_nodes < 1:
        raise ValueError('chunk-nodes must be positive')

    # 1. Load compact saved predictions and their exact preprocessing identity.
    with np.load(args.input, allow_pickle=False) as archive:
        manifest = resolve_export_manifest(
            str(archive['prepared_manifest']),
            str(archive['prepared_sha256']),
            getattr(args, 'prepared_manifest', None),
            relative_path=(str(archive['prepared_manifest_relative'])
                           if 'prepared_manifest_relative' in archive.files else None),
            anchor=args.input.resolve().parent,
        )
        indices = archive['test_indices']
        case_names = archive['case_names']
        predictions = archive['pred']
        has_truth = 'true' in archive.files

    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
        raise ValueError('Saved case indices must be a nonempty integer vector')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Each saved case index must occur exactly once')
    if case_names.shape != indices.shape:
        raise ValueError('Case names and indices have different lengths')

    dataset_info, data, _ = load_prepared(manifest)
    basis = load_decoder_basis(manifest.parent / 'decoder_basis.npz', dataset_info['n_components'])
    expected_shape = (len(indices), len(data['coords']), dataset_info['n_components'] + 1)
    if predictions.shape != expected_shape or not np.isfinite(predictions).all():
        raise ValueError('Saved features must be finite and match the prepared node/channel dimensions')

    # 2. Verify the reference geometry uses the same canonical node numbering.
    mesh = meshio.read(args.mesh)
    coordinate_keys = ('ab', 'rt', 'tm', 'tv')
    if not all(key in mesh.point_data for key in coordinate_keys):
        raise ValueError('Mesh must contain ab/rt/tm/tv point fields for node-order validation')
    cobiveco = np.column_stack([mesh.point_data[key] for key in coordinate_keys])
    if cobiveco.shape != data['coords'].shape or not np.allclose(
            cobiveco, data['coords'], rtol=0, atol=1e-6):
        raise ValueError('Mesh node order differs from the prepared canonical Cobiveco grid')

    return {
        'manifest': manifest,
        'indices': indices,
        'case_names': case_names,
        'predictions': predictions,
        'has_truth': has_truth,
        'data': data,
        'basis': basis,
        'mesh': mesh,
    }


def select_cases(context, case=None, cases=None, default_count=1):
    """Preserve archive order by default; explicit indices keep the requested order."""
    available = context['indices'].tolist()
    if cases is not None:
        selected = list(cases)
    elif case is not None:
        selected = [case]
    else:
        selected = available[:default_count]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Choose one or more distinct case indices')
    missing = [index for index in selected if index not in available]
    if missing:
        raise ValueError(f'Cases {missing} are absent from saved predictions; available indices: {available}')
    return selected


def decode_export_case(context, case, chunk_nodes):
    """Decode one heart; Vm truth comes from raw data, not from oracle features."""
    import numpy as np
    from ..decoder import decode_features

    position = context['indices'].tolist().index(case)
    case_name = str(context['case_names'][position])
    predicted = decode_features(context['predictions'][position], context['basis'], chunk_nodes)
    fields = {'Vm_pred': predicted}

    if context['has_truth']:
        data = context['data']
        if case < 0 or case >= len(data['theta']) or str(data['case_names'][case]) != case_name:
            raise ValueError('Prediction and ground-truth case identities disagree')
        truth = data['vm'][case]
        fields['Vm_true'] = truth
        fields['Vm_signed_error'] = predicted - truth
        fields['Vm_abs_error'] = np.abs(predicted - truth)
    return fields, case_name


def load_export(args):
    """Single-case entry point retained for SVG export."""
    context = load_export_context(args)
    case = select_cases(context, case=args.case)[0]
    fields, case_name = decode_export_case(context, case, args.chunk_nodes)
    output = new_directory(args.output or args.input.resolve().parent / 'visualization' / 'svg' / f'case_{case:04d}')
    write_json(output / 'export_info.json', {
        'features': str(args.input.resolve()),
        'prepared_manifest': str(context['manifest']),
        'mesh': str(args.mesh.resolve()),
        'case': case,
        'case_name': case_name,
        'units': {'vm': 'mV', 'time': 'ms'},
    })
    return output, context['mesh'], context['basis']['time'], fields
