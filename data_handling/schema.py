"""Contract: global geometry theta, canonical Cobiveco nodes, physical Vm and time."""
import numpy as np


def validate_raw(data, require_vm=True):
    theta, coords, time, names = (data[k] for k in ('theta', 'coords', 'time', 'case_names'))
    if theta.ndim != 2 or min(theta.shape) < 1:
        raise ValueError('theta must have shape (hearts, geometry_parameters)')
    if coords.ndim != 2 or coords.shape[1] != 4 or not len(coords):
        raise ValueError('coords must be shared Cobiveco coordinates (nodes, 4), in ab/rt/tm/tv order')
    if time.ndim != 1 or len(time) < 3 or not np.all(np.diff(time) > 0):
        raise ValueError('time must contain at least three increasing frames in ms')
    if not np.allclose(np.diff(time), np.diff(time)[0], rtol=1e-5, atol=1e-5):
        raise ValueError('The existing phase decoder requires a uniform time grid')
    if names.ndim != 1 or len(names) != len(theta) or len(np.unique(names)) != len(names):
        raise ValueError('case_names must uniquely identify every heart')
    for key in ('theta', 'coords', 'time'):
        if not np.issubdtype(data[key].dtype, np.floating) or not np.isfinite(data[key]).all():
            raise ValueError(f'{key} must contain finite floating-point values')
    if require_vm or 'vm' in data:
        if 'vm' not in data or data['vm'].shape != (len(theta), len(coords), len(time)):
            raise ValueError('vm must have shape (hearts, nodes, frames), matching theta/coords/time')
        if data['vm'].dtype != np.float32:
            raise ValueError('vm must be float32 in physical mV; convert explicitly before preparation')
