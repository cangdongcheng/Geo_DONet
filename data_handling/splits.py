"""Heart-level fixed splits and the historical seed-42 five-fold protocol."""
import numpy as np


def make_splits(n_cases, n_train=95, n_val=5, folds=1, seed=42):
    if folds == 1:
        if n_train + n_val >= n_cases:
            raise ValueError('Train/validation counts must leave at least one test heart')
        order = np.arange(n_cases)
        return [dict(train=order[:n_train].tolist(),
                     val=order[n_train:n_train+n_val].tolist(),
                     test=order[n_train+n_val:].tolist())]
    if folds < 2 or n_cases % folds:
        raise ValueError('CV needs at least two folds, evenly dividing the heart count')
    order = np.arange(n_cases)
    np.random.RandomState(seed).shuffle(order)
    result = []
    for test in np.split(order, folds):
        remaining = order[~np.isin(order, test)]
        if len(remaining) <= n_val:
            raise ValueError('CV validation count leaves no training hearts')
        result.append(dict(train=remaining[:-n_val].tolist(), val=remaining[-n_val:].tolist(),
                           test=test.tolist()))
    return result
