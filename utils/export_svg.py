"""Render saved predictions, waveform traces and physical-unit colorbars."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np

from Geo_DONet.artifacts import write_json
from Geo_DONet.utils.export_common import export_parser, load_export


def main(argv=None):
    parser = export_parser(__doc__)
    parser.add_argument('--times', type=float, nargs='+', help='Requested snapshot times in ms')
    args = parser.parse_args(argv)
    output, mesh, times, fields = load_export(args)
    requested = np.asarray(args.times if args.times is not None else [times[0], times[len(times)//2], times[-1]])
    if not np.isfinite(requested).all() or np.any(requested < times[0]) or np.any(requested > times[-1]):
        raise ValueError('Snapshot times must lie within the decoder time grid')
    frames = np.unique([np.argmin(np.abs(times-t)) for t in requested])
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    from Geo_DONet.utils import rendering as viz
    viz._init_plot_worker(mesh.points)
    for frame in frames:
        for name in ('Vm_pred', 'Vm_true', 'Vm_abs_error'):
            if name not in fields:
                continue
            error = name == 'Vm_abs_error'
            cmap, low, high = ((viz.CMAP_ERR, viz.VM_ERR_MIN, viz.VM_ERR_MAX) if error
                               else (viz.CMAP_VM, viz.VM_MIN, viz.VM_MAX))
            viz._render_scatter_svg((fields[name][:, frame], cmap, low, high,
                                     output / f't{float(times[frame]):g}ms_{name}.svg'))
    if 'Vm_true' in fields:
        viz.save_vm_traces(fields['Vm_pred'], fields['Vm_true'], times, output / 'Vm_traces.svg')
    viz.save_colorbar(viz.CMAP_VM, viz.VM_MIN, viz.VM_MAX, 'Vm (mV)', output / 'Vm_colorbar.svg')
    viz.save_colorbar(viz.CMAP_ERR, viz.VM_ERR_MIN, viz.VM_ERR_MAX, '|Vm error| (mV)', output / 'error_colorbar.svg')
    write_json(output / 'snapshot_times.json', dict(requested_ms=requested.tolist(), actual_ms=times[frames].tolist()))
    print(f'SVG figures: {output}')
    return output


if __name__ == '__main__':
    main()
