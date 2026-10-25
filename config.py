"""CLI arguments and JSON settings: defaults < config file < command line."""
import argparse
import json
import math
from pathlib import Path


# Default settings. Model/dataset settings are also saved with every run.
DEFAULTS = {
    # Files and runtime
    'input': None,
    'output': None,
    'chkpt': None,
    'prepared_manifest': None,
    'name': None,
    'seed': 42,
    'device': None,
    # Dataset preparation
    'n_train': 95,
    'n_val': 5,
    'folds': 1,
    'n_components': 5,
    'reference_at': None,
    'at_threshold': -10.0,
    'chunk_nodes': 20000,
    # Network and training
    'width': 200,
    'depth': 4,
    'epochs': 5000,
    'batch_size': 8,
    'nodes_per_step': 2048,
    'val_nodes': 4096,
    'lr': 5e-4,
    'val_every': 10,
    'print_every': 100,
    'patience': 1000,
    'feature_loss_weight': 0.0,
    'at_loss_weight': 0.5,
    'pca_loss_weight': 0.5,
    # Evaluation output
    'n_viz': 2,
    'save_vm': False,
}

PATH_KEYS = ('input', 'output', 'chkpt', 'prepared_manifest')
INTEGER_KEYS = (
    'seed', 'n_train', 'n_val', 'folds', 'n_components', 'chunk_nodes',
    'width', 'depth', 'epochs', 'batch_size', 'nodes_per_step', 'val_nodes',
    'val_every', 'print_every', 'patience', 'n_viz',
)
FLOAT_KEYS = ('lr', 'at_threshold', 'feature_loss_weight', 'at_loss_weight', 'pca_loss_weight')


def build_parser(preparation=False):
    """Define each flag explicitly so its meaning is visible beside its help text."""
    if preparation:
        description = 'Prepare cardiac Vm data: split hearts, align waveforms, fit training-only PCA.'
        examples = (
            'Examples:\n'
            '  python prepare_dataset.py --input /scratch/cohort.npz --config configs/pca.json\n'
            '  python prepare_dataset.py --input /scratch/cohort.npz --config configs/cv.json\n'
            '\nThe input contains theta, coords, vm, time, and case_names.\n'
            'PCA and normalization are fitted on training hearts only.'
        )
    else:
        description = 'Geo-DONet: geometry + Cobiveco coordinates -> AT/PCA features -> reconstructed Vm.'
        examples = (
            'Examples:\n'
            '  python main.py --train --input /scratch/prepared/manifest.json --config configs/pca.json\n'
            '  python main.py --evaluate --chkpt /scratch/run/checkpoints/best.pt\n'
            '  python main.py --predict --input /scratch/new_hearts.npz --chkpt /scratch/run/checkpoints/best.pt\n'
            '  python main.py --cv --input /scratch/cv_prepared/manifest.json --config configs/cv.json\n'
            '\nSettings: built-in defaults < JSON config < explicit CLI flags.\n'
            'During training, split membership and PCA dimensions come from the prepared dataset.'
        )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )

    # 1. Select an action. argparse supplies -h / --help automatically.
    if preparation:
        parser.set_defaults(mode='prepare')
    else:
        mode = parser.add_argument_group('Action (choose one)')
        actions = mode.add_mutually_exclusive_group(required=True)
        actions.add_argument('--train', dest='mode', action='store_const', const='train',
                             help='Train on one prepared split; save the best validation checkpoint')
        actions.add_argument('--evaluate', dest='mode', action='store_const', const='evaluate',
                             help='Evaluate a checkpoint on its saved held-out test hearts')
        actions.add_argument('--predict', dest='mode', action='store_const', const='predict',
                             help='Predict a new cohort without ground-truth Vm')
        actions.add_argument('--cv', dest='mode', action='store_const', const='cv',
                             help='Train/evaluate all prepared CV folds and pool test metrics')

    # 2. Input/output paths and settings file.
    files = parser.add_argument_group('Files')
    files.add_argument('--config', type=Path, metavar='JSON',
                       help='Load settings from JSON; explicit flags override this file')
    input_help = ('Raw NPZ archive or existing raw dataset.json' if preparation else
                  'Prepared manifest for train/CV; raw NPZ/manifest for predict')
    files.add_argument('--input', type=Path, metavar='PATH', help=input_help)
    files.add_argument('--output', type=Path, metavar='DIR',
                       help='Train/prepare: run root (required); evaluate/predict: exact folder (default: beside checkpoint)')
    files.add_argument('--name', metavar='NAME',
                       help='Readable training/preparation name; evaluation label or prediction cohort name')
    if not preparation:
        files.add_argument('--chkpt', type=Path, metavar='PT',
                           help='Existing checkpoint file; required for evaluate/predict')
        files.add_argument('--prepared-manifest', type=Path, metavar='JSON',
                           help='Relocated preparation fold manifest for evaluate/predict; identity must match')

    # 3. Dataset settings apply only while preparing the data.
    if preparation:
        dataset = parser.add_argument_group('Heart splits and waveform PCA')
        dataset.add_argument('--n-train', type=int, metavar='N',
                             help='Training hearts for a single split (default: 95)')
        dataset.add_argument('--n-val', type=int, metavar='N',
                             help='Validation hearts per split/fold (default: 5)')
        dataset.add_argument('--folds', type=int, metavar='N',
                             help='1 for the fixed split; N for shuffled N-fold CV (default: 1)')
        dataset.add_argument('--n-components', type=int, metavar='K',
                             help='Number of temporal waveform PCA modes (default: 5)')
        dataset.add_argument('--reference-at', type=float, metavar='MS',
                             help='Alignment reference time (default: median training AT)')
        dataset.add_argument('--at-threshold', type=float, metavar='MV',
                             help='Upward-crossing voltage used to extract AT (default: -10 mV)')
    else:
        network = parser.add_argument_group('Network and optimizer')
        network.add_argument('--width', type=int, metavar='N',
                             help='Hidden width of geometry branch and spatial trunk (default: 200)')
        network.add_argument('--depth', type=int, metavar='N',
                             help='Hidden layers in each branch/trunk (default: 4)')
        network.add_argument('--lr', type=float,
                             help='Constant Adam learning rate (default: 0.0005)')

        training = parser.add_argument_group('Training and validation')
        training.add_argument('--epochs', type=int, metavar='N',
                              help='Maximum training epochs (default: 5000)')
        training.add_argument('--batch-size', type=int, metavar='N',
                              help='Hearts in one optimizer step (default: 8)')
        training.add_argument('--nodes-per-step', type=int, metavar='N',
                              help='Fresh random mesh nodes per training batch (default: 2048)')
        training.add_argument('--val-nodes', type=int, metavar='N',
                              help='Fixed sampled nodes for validation (default: 4096)')
        training.add_argument('--val-every', type=int, metavar='N',
                              help='Validate every N epochs, and at the last epoch (default: 10)')
        training.add_argument('--print-every', type=int, metavar='N',
                              help='Print progress every N epochs (default: 100)')
        training.add_argument('--patience', type=int, metavar='N',
                              help='Epochs without improvement before stopping; 0 disables (default: 1000)')

        loss = parser.add_argument_group('Optional auxiliary feature loss')
        loss.add_argument('--feature-loss-weight', type=float,
                          help='Auxiliary AT/PCA loss weight; 0 gives Vm-only supervision (default: 0)')
        loss.add_argument('--at-loss-weight', type=float,
                          help='Relative AT weight within the auxiliary loss (default: 0.5)')
        loss.add_argument('--pca-loss-weight', type=float,
                          help='Relative PCA weight within the auxiliary loss (default: 0.5)')

        outputs = parser.add_argument_group('Evaluation and prediction outputs')
        outputs.add_argument('--n-viz', type=int, metavar='N',
                             help='Number of held-out hearts to plot; 0 disables (default: 2)')
        outputs.add_argument('--save-vm', action=argparse.BooleanOptionalAction,
                             help='Also save decoded Vm, one NPY per heart (default: disabled)')

    # 4. Shared runtime settings.
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--seed', type=int, help='Random seed (default: 42)')
    runtime.add_argument('--chunk-nodes', type=int, metavar='N',
                         help='Node block size for preprocessing/decoding/inference (default: 20000)')
    if not preparation:
        runtime.add_argument('--device', metavar='DEVICE',
                             help='cpu, cuda, or cuda:N (default: CUDA when available)')
    return parser


def load_config_file(path, parser):
    """Read JSON settings and resolve their paths relative to the JSON file."""
    path = path.expanduser().resolve()
    try:
        settings = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        parser.error(f'Cannot read config {path}: {error}')

    if not isinstance(settings, dict):
        parser.error('Config must be a JSON object')
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        parser.error('Unknown config settings: ' + ', '.join(sorted(unknown)))

    for key in PATH_KEYS:
        if settings.get(key) is not None:
            if not isinstance(settings[key], str):
                parser.error(f'{key} must be a path string or null')
            value = Path(settings[key]).expanduser()
            settings[key] = str((path.parent / value).resolve())
    return settings


def validate_settings(settings, mode, parser):
    """Reject invalid values before loading datasets or allocating an accelerator."""
    for key in PATH_KEYS + ('device',):
        if settings[key] is not None and not isinstance(settings[key], str):
            parser.error(f'{key} must be a path/string or null')
    name = settings['name']
    if name is not None and (not isinstance(name, str) or not name.strip()
                             or name in ('.', '..') or '/' in name or '\\' in name):
        parser.error('name must be one nonempty folder name without path separators')
    if mode in ('evaluate', 'predict') and name and settings['output']:
        parser.error('For evaluate/predict, choose --name or an exact --output folder')
    for key in INTEGER_KEYS:
        if type(settings[key]) is not int:
            parser.error(f'{key} must be an integer')
    for key in FLOAT_KEYS + ('reference_at',):
        value = settings[key]
        if key == 'reference_at' and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parser.error(f'{key} must be numeric')
        if not math.isfinite(value):
            parser.error(f'{key} must be finite')
    if not isinstance(settings['save_vm'], bool):
        parser.error('save_vm must be boolean')

    for key in INTEGER_KEYS:
        minimum = 0 if key in ('seed', 'patience', 'n_viz') else 1
        if settings[key] < minimum:
            parser.error(f'{key} must be at least {minimum}')
    if settings['lr'] <= 0:
        parser.error('lr must be positive')
    for key in ('feature_loss_weight', 'at_loss_weight', 'pca_loss_weight'):
        if settings[key] < 0:
            parser.error(f'{key} must be nonnegative')
    if settings['at_loss_weight'] + settings['pca_loss_weight'] <= 0:
        parser.error('At least one feature diagnostic group weight must be positive')

    if mode in ('prepare', 'train', 'cv', 'predict') and not settings['input']:
        parser.error('--input is required (in a config or on the command line)')
    if mode in ('evaluate', 'predict') and not settings['chkpt']:
        parser.error('--chkpt is required for evaluation/prediction')
    if mode in ('prepare', 'train', 'cv') and not settings['output']:
        parser.error('--output is required (in a config or on the command line)')


def parse_config(preparation=False, argv=None):
    """Load defaults, overlay the config, then apply explicit CLI overrides."""
    parser = build_parser(preparation)
    options = vars(parser.parse_args(argv))
    mode = options.pop('mode')
    config_path = options.pop('config', None)

    settings = DEFAULTS.copy()
    if config_path is not None:
        settings.update(load_config_file(config_path, parser))
    settings.update(options)

    for key in PATH_KEYS:
        if settings[key] is not None:
            settings[key] = str(Path(settings[key]).expanduser().resolve())
    validate_settings(settings, mode, parser)
    return argparse.Namespace(**settings, mode=mode)
