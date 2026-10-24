"""Physical-unit Vm, activation-time and waveform-slope metrics."""
import numpy as np

def vm_metrics(prediction, truth):
    difference = prediction - truth
    rel_l2 = float(np.linalg.norm(difference) / (np.linalg.norm(truth) + 1e-12))
    return rel_l2, float(np.abs(difference).mean())


def at_metrics(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.any():
        return np.nan, np.nan
    difference = prediction[valid] - truth[valid]
    rel_l2 = float(np.linalg.norm(difference) / (np.linalg.norm(truth[valid]) + 1e-12))
    return rel_l2, float(np.abs(difference).mean())


def median_max_dvdt(vm, dt):
    return float(np.median(np.abs(np.gradient(vm, dt, axis=1)).max(axis=1)))
