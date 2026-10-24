"""Training-fitted scales for geometry, coordinates and waveform features."""
import numpy as np

class FeatureNormalizer:
    """Training-only normalisation for geometry, coordinates, and each target."""

    def __init__(self, theta_train, coords, target_train):
        self.theta_mean = theta_train.mean(axis=0).astype(np.float32)
        self.theta_std = theta_train.std(axis=0).astype(np.float32)
        self.theta_std[self.theta_std < 1e-8] = 1.0
        self.coord_min = coords.min(axis=0).astype(np.float32)
        self.coord_max = coords.max(axis=0).astype(np.float32)
        self.target_mean = target_train.mean(axis=(0, 1)).astype(np.float32)
        self.target_std = target_train.std(axis=(0, 1)).astype(np.float32)
        self.target_std[self.target_std < 1e-8] = 1.0

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        for key in ("theta_mean", "theta_std", "coord_min", "coord_max",
                    "target_mean", "target_std"):
            setattr(obj, key, np.asarray(state[key], dtype=np.float32))
        return obj

    def state(self):
        return {key: getattr(self, key) for key in
                ("theta_mean", "theta_std", "coord_min", "coord_max",
                 "target_mean", "target_std")}

    def theta(self, value):
        return (value - self.theta_mean) / self.theta_std

    def coords(self, value):
        return (value - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)

    def targets(self, value):
        return (value - self.target_mean) / self.target_std

    def targets_inverse(self, value):
        return value * self.target_std + self.target_mean
