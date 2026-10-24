"""Prepare raw Vm for a fixed split or independent CV folds."""
from pathlib import Path

# Allow both `python prepare_dataset.py` and module execution.
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Geo_DONet_clean.artifacts import experiment_directory, digest, record_invocation, write_json
from Geo_DONet_clean.config import parse_config


def prepare(args):
    """Create a reusable dataset preparation; the raw input is never modified."""
    from Geo_DONet_clean.data_handling.loading import import_raw, load_raw
    from Geo_DONet_clean.data_handling.preprocessing import activation_targets, prepare_fold
    from Geo_DONet_clean.data_handling.splits import make_splits

    # 1. Create a named (or dated) scratch folder and record preprocessing settings.
    prepared_folder = experiment_directory(args.output, args.name)
    record_invocation(prepared_folder, args)

    try:
        # 2. Load raw theta/coords/Vm/time. NPZ arrays are extracted once for mmap access.
        raw_manifest = import_raw(args.input, prepared_folder / 'raw')
        data = load_raw(raw_manifest)

        # 3. Split by heart before fitting any template, PCA basis or normalization.
        splits = make_splits(
            n_cases=len(data['theta']),
            n_train=args.n_train,
            n_val=args.n_val,
            folds=args.folds,
            seed=args.seed,
        )
        activation_times = activation_targets(data, args.at_threshold, args.chunk_nodes)

        # 4. Fit training-only waveform preprocessing separately for each split/fold.
        fold_records = []
        for fold_number, split in enumerate(splits, start=1):
            print(
                f'Preparing fold {fold_number}/{len(splits)}: '
                f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}",
                flush=True,
            )
            fold_folder = prepared_folder / f'fold_{fold_number:02d}'
            prepare_fold(data, activation_times, split, raw_manifest, fold_folder, args)
            fold_records.append({
                'path': f'{fold_folder.name}/manifest.json',
                'sha256': digest(fold_folder / 'manifest.json'),
            })

        # 5. Publish the dataset manifest only after every fold has finished.
        write_json(prepared_folder / 'manifest.json', {
            'kind': 'prepared_dataset',
            'version': 1,
            'status': 'complete',
            'folds': fold_records,
        })
    except BaseException as error:
        write_json(prepared_folder / 'failure.json', {'error': str(error)})
        raise

    manifest = prepared_folder / 'manifest.json'
    print(f'Prepared dataset: {manifest}', flush=True)
    return manifest


def main():
    args = parse_config(preparation=True)
    prepare(args)


if __name__ == '__main__':
    main()
