"""Train and evaluate each prepared fold, then pool the held-out heart metrics."""
import copy
from pathlib import Path

from .artifacts import record_invocation


def cross_validate(args, manifests, folder, device):
    from .data_handling.loading import load_prepared
    from .evaluate import evaluate, write_metrics
    from .train import set_seed, train

    # 1. Check that folds share raw data and test every heart exactly once.
    raw_identity = None
    tested_hearts = set()
    expected_hearts = set()

    for manifest in manifests:
        dataset_info, data, split = load_prepared(manifest)
        if raw_identity is None:
            raw_identity = dataset_info['raw_manifest_sha256']
        if dataset_info['raw_manifest_sha256'] != raw_identity:
            raise ValueError('CV folds must share the same raw dataset')

        test_hearts = set(split['test'])
        if tested_hearts & test_hearts:
            raise ValueError('CV test hearts overlap across folds')
        tested_hearts.update(test_hearts)
        expected_hearts = set(range(len(data['theta'])))

    if tested_hearts != expected_hearts:
        raise ValueError('CV test folds must cover every heart exactly once')

    # 2. Train one model per fold, using that fold's frozen preprocessing.
    all_metrics = []
    for fold_number, manifest in enumerate(manifests, start=1):
        fold_args = copy.deepcopy(args)
        fold_args.seed = args.seed + fold_number
        fold_args.input = str(manifest)

        fold_folder = Path(folder) / f'fold_{fold_number:02d}'
        fold_folder.mkdir()
        record_invocation(fold_folder, fold_args)

        set_seed(fold_args.seed)
        checkpoint = train(fold_args, manifest, fold_folder, device)

        # 3. Evaluate the best validation checkpoint on this fold's test hearts.
        fold_args.chkpt = str(checkpoint)
        fold_metrics = evaluate(fold_args, fold_folder, device)
        for case_metrics in fold_metrics:
            all_metrics.append({'fold': fold_number, **case_metrics})

    # 4. Pool the disjoint test folds: each heart contributes once.
    write_metrics(folder, all_metrics)
