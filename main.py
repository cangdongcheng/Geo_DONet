"""Geo-DONet entry point: select data, create an output folder, run the chosen action."""
import json
from pathlib import Path

# Allow both `python main.py` and `python -m Geo_DONet.main`.
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Geo_DONet.artifacts import (
    checkpoint_directory, experiment_directory, log_run, new_directory,
    record_invocation, write_json,
)
from Geo_DONet.config import parse_config


def select_manifests(args):
    """Resolve the requested preparation and restore its fixed split/PCA settings."""
    from Geo_DONet.data_handling.loading import fold_manifests

    if args.mode == 'predict' or (args.mode == 'evaluate' and args.input is None):
        return []

    manifests = fold_manifests(args.input)
    if args.mode in ('train', 'evaluate') and len(manifests) != 1:
        raise ValueError('Select one prepared split, or use --cv for multiple folds')
    if args.mode == 'cv' and len(manifests) < 2:
        raise ValueError('--cv requires multiple independently prepared folds')

    if args.mode in ('train', 'cv'):
        dataset_info = json.loads(manifests[0].read_text())
        split = json.loads((manifests[0].parent / 'splits.json').read_text())
        args.n_components = dataset_info['n_components']
        args.n_train = len(split['train'])
        args.n_val = len(split['val'])
        args.folds = len(manifests)
    else:
        args.input = str(manifests[0])
    return manifests


def run(args):
    # Imports stay after parsing so -h/--help needs no PyTorch environment.
    import numpy as np
    import torch
    from Geo_DONet.cross_validation import cross_validate
    from Geo_DONet.evaluate import evaluate
    from Geo_DONet.train import set_seed, train

    # 1. Select the device and the prepared dataset.
    device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use a GPU compute node or --device cpu')
    manifests = select_manifests(args)

    # 2. Keep model outputs together; only a new training creates a new run.
    if args.mode in ('train', 'cv'):
        run_folder = experiment_directory(args.output, args.name)
    else:
        model_folder = checkpoint_directory(args.chkpt)
        if args.output is not None:
            destination = Path(args.output)
        elif args.mode == 'evaluate':
            label = 'evaluation' if args.name is None else f'evaluation_{args.name}'
            destination = model_folder / label
        else:
            cohort_name = args.name or Path(args.input).stem
            destination = model_folder / 'predictions' / cohort_name
        run_folder = new_directory(destination)
        args.output = str(run_folder)
    record_invocation(run_folder, args)

    with log_run(run_folder):
        print(f'Output: {run_folder}', flush=True)
        write_json(run_folder / 'environment.json', {
            'torch': torch.__version__,
            'numpy': np.__version__,
            'device': str(device),
            'cuda': torch.version.cuda,
        })

        # 3. Run one action. Training always uses the prepared, training-fitted decoder.
        try:
            if args.mode == 'train':
                set_seed(args.seed)
                train(args, manifests[0], run_folder, device)
            elif args.mode == 'cv':
                cross_validate(args, manifests, run_folder, device)
            elif args.mode == 'evaluate':
                evaluate(args, run_folder, device, output=run_folder)
            elif args.mode == 'predict':
                evaluate(args, run_folder, device, prediction_only=True, output=run_folder)
            write_json(run_folder / 'status.json', {'status': 'complete'})
        except BaseException as error:
            write_json(run_folder / 'status.json', {'status': 'failed', 'error': str(error)})
            raise

    return run_folder


def main():
    args = parse_config()
    run(args)


if __name__ == '__main__':
    main()
