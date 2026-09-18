"""Normalisation, differentiable slope decoder, and evaluation helpers."""
import numpy as np
import torch
import torch.nn as nn

from oracle_decoder import (activation_time, decode_waveforms, max_abs_dvdt,
                            upstroke_slope)


class FeatureTransform:
    """Train-only input normalization and safe feature parameterization.

    Network channels are [AT z-score, slope logit, PCA z-scores]. Slope is
    mapped through a sigmoid to fitted physical bounds, guaranteeing a positive
    slope and a monotone decoder warp. This is a parameterization, not a direct
    feature loss.
    """
    def __init__(self, theta_train, coords, target_train, basis,
                 slope_margin=0.20):
        self.theta_mean = theta_train.mean(axis=0).astype(np.float32)
        self.theta_std = theta_train.std(axis=0).astype(np.float32)
        self.theta_std[self.theta_std < 1e-8] = 1.0
        self.coord_min = coords.min(axis=0).astype(np.float32)
        self.coord_max = coords.max(axis=0).astype(np.float32)

        self.at_mean = np.float32(target_train[..., 0].mean())
        self.at_std = np.float32(target_train[..., 0].std())
        if self.at_std < 1e-8:
            self.at_std = np.float32(1.0)
        self.pca_mean = target_train[..., 2:].mean(axis=(0, 1)).astype(np.float32)
        self.pca_std = target_train[..., 2:].std(axis=(0, 1)).astype(np.float32)
        self.pca_std[self.pca_std < 1e-8] = 1.0

        slope = target_train[..., 1]
        slope_min, slope_max = float(slope.min()), float(slope.max())
        span = max(slope_max - slope_min, 1.0)
        # Leave extrapolation room but stay comfortably inside the monotonic
        # boundary reference_slope/slope * inner < outer.
        monotone_floor = (float(basis["reference_slope"])
                          * float(basis["inner_ms"])
                          / (0.8 * float(basis["outer_ms"])))
        self.slope_lower = np.float32(max(monotone_floor,
                                          slope_min - slope_margin * span))
        self.slope_upper = np.float32(slope_max + slope_margin * span)
        if self.slope_upper <= self.slope_lower:
            raise ValueError("invalid fitted slope bounds")

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        array_keys = ("theta_mean", "theta_std", "coord_min", "coord_max",
                      "pca_mean", "pca_std")
        scalar_keys = ("at_mean", "at_std", "slope_lower", "slope_upper")
        for key in array_keys:
            setattr(obj, key, np.asarray(state[key], dtype=np.float32))
        for key in scalar_keys:
            setattr(obj, key, np.float32(state[key]))
        return obj

    def state(self):
        return {key: getattr(self, key) for key in
                ("theta_mean", "theta_std", "coord_min", "coord_max",
                 "at_mean", "at_std", "pca_mean", "pca_std",
                 "slope_lower", "slope_upper")}

    def theta(self, values):
        return (values - self.theta_mean) / self.theta_std

    def coords(self, values):
        return ((values - self.coord_min) /
                (self.coord_max - self.coord_min + 1e-8))

    def targets(self, physical):
        at = (physical[..., 0:1] - self.at_mean) / self.at_std
        ratio = ((physical[..., 1:2] - self.slope_lower) /
                 (self.slope_upper - self.slope_lower))
        ratio = np.clip(ratio, 1e-5, 1.0 - 1e-5)
        slope_logit = np.log(ratio / (1.0 - ratio))
        pca = (physical[..., 2:] - self.pca_mean) / self.pca_std
        return np.concatenate((at, slope_logit, pca), axis=-1).astype(np.float32)

    def inverse_numpy(self, raw):
        at = raw[..., 0:1] * self.at_std + self.at_mean
        slope = (self.slope_lower + (self.slope_upper - self.slope_lower) /
                 (1.0 + np.exp(-np.clip(raw[..., 1:2], -30.0, 30.0))))
        pca = raw[..., 2:] * self.pca_std + self.pca_mean
        return np.concatenate((at, slope, pca), axis=-1).astype(np.float32)

    def inverse_torch(self, raw):
        at = raw[..., 0:1] * float(self.at_std) + float(self.at_mean)
        slope = (float(self.slope_lower)
                 + float(self.slope_upper - self.slope_lower)
                 * torch.sigmoid(raw[..., 1:2]))
        pca_mean = torch.as_tensor(self.pca_mean, dtype=raw.dtype,
                                   device=raw.device)
        pca_std = torch.as_tensor(self.pca_std, dtype=raw.dtype,
                                  device=raw.device)
        pca = raw[..., 2:] * pca_std + pca_mean
        return torch.cat((at, slope, pca), dim=-1)


def load_basis(path, n_components):
    archive = np.load(path, allow_pickle=False)
    required = {"node_template", "residual_mean", "components", "time",
                "reference_at", "reference_slope", "at_threshold",
                "slope_window_ms", "inner_ms", "outer_ms"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"basis missing {sorted(missing)}")
    if archive["components"].shape[1] < n_components:
        raise ValueError(f"basis has only {archive['components'].shape[1]} modes")
    return dict(node_template=archive["node_template"].astype(np.float32),
                residual_mean=archive["residual_mean"].astype(np.float32),
                components=archive["components"][:, :n_components].astype(np.float32),
                time=archive["time"].astype(np.float32),
                reference_at=float(archive["reference_at"]),
                reference_slope=float(archive["reference_slope"]),
                at_threshold=float(archive["at_threshold"]),
                slope_window_ms=float(archive["slope_window_ms"]),
                inner_ms=float(archive["inner_ms"]),
                outer_ms=float(archive["outer_ms"]))


class DifferentiableSlopeDecoder(nn.Module):
    """Decode [AT, slope, PCA...] with a local monotone inverse time warp."""
    def __init__(self, basis, n_components):
        super().__init__()
        self.n_components = int(n_components)
        self.register_buffer("node_template",
                             torch.from_numpy(basis["node_template"]))
        self.register_buffer("residual_mean",
                             torch.from_numpy(basis["residual_mean"]))
        self.register_buffer("components",
                             torch.from_numpy(basis["components"][:, :n_components]))
        time_ms = np.asarray(basis["time"], dtype=np.float32)
        self.register_buffer("time_ms", torch.from_numpy(time_ms))
        self.reference_at = float(basis["reference_at"])
        self.reference_slope = float(basis["reference_slope"])
        self.inner_ms = float(basis["inner_ms"])
        self.outer_ms = float(basis["outer_ms"])
        self.dt = float(np.median(np.diff(time_ms)))
        self.time_start = float(time_ms[0])

    def forward(self, physical_features, node_indices):
        if physical_features.shape[-1] != self.n_components + 2:
            raise ValueError("feature count disagrees with decoder")
        nodes = torch.as_tensor(node_indices, dtype=torch.long,
                                device=physical_features.device)
        coefficients = physical_features[..., 2:]
        aligned = (self.node_template[nodes][None]
                   + self.residual_mean[None, None]
                   + torch.einsum("bsk,tk->bst", coefficients,
                                  self.components))

        at = physical_features[..., 0]
        slope = physical_features[..., 1]
        scale = self.reference_slope / slope
        central_end = scale[..., None] * self.inner_ms
        shoulder = ((self.outer_ms - scale * self.inner_ms) /
                    (self.outer_ms - self.inner_ms))[..., None]
        physical_offset = self.time_ms[None, None] - at[..., None]
        absolute = torch.abs(physical_offset)
        normalized_absolute = torch.where(
            absolute <= central_end,
            absolute / scale[..., None],
            torch.where(absolute < self.outer_ms,
                        self.inner_ms + (absolute - central_end) / shoulder,
                        absolute))
        normalized_offset = torch.copysign(normalized_absolute,
                                           physical_offset)
        position = ((self.reference_at + normalized_offset - self.time_start)
                    / self.dt)
        position = position.clamp(0.0, float(aligned.shape[-1] - 1))
        left = torch.floor(position).to(torch.long)
        left = left.clamp_max(aligned.shape[-1] - 2)
        fraction = position - left.to(position.dtype)
        y0 = torch.gather(aligned, 2, left)
        y1 = torch.gather(aligned, 2, left + 1)
        return y0 + fraction * (y1 - y0)


def sampled_vm(vm, case_indices, node_indices, device):
    block = vm[np.asarray(case_indices)[:, None],
               np.asarray(node_indices)[None, :], :]
    return torch.as_tensor(block, dtype=torch.float32, device=device)


def vm_metrics(prediction, truth):
    difference = prediction - truth
    return (float(np.linalg.norm(difference) /
                  (np.linalg.norm(truth) + 1e-12)),
            float(np.abs(difference).mean()))


def feature_metrics(prediction, truth):
    return np.abs(prediction - truth).mean(axis=0)


def decoded_diagnostics(decoded, truth, true_at, true_slope, basis):
    pred_at = activation_time(decoded, basis["time"], basis["at_threshold"])
    at_valid = np.isfinite(pred_at) & np.isfinite(true_at)
    at_mae = float(np.abs(pred_at[at_valid] - true_at[at_valid]).mean())
    pred_slope = upstroke_slope(decoded, basis["time"], pred_at,
                                basis["slope_window_ms"])
    slope_valid = np.isfinite(pred_slope) & np.isfinite(true_slope)
    slope_mae = float(np.abs(pred_slope[slope_valid]
                             - true_slope[slope_valid]).mean())
    dt = float(np.median(np.diff(basis["time"])))
    dvdt_fraction = (float(np.median(max_abs_dvdt(decoded, dt))) /
                     float(np.median(max_abs_dvdt(truth, dt))))
    return at_mae, slope_mae, dvdt_fraction


def decode_numpy(features, basis, chunk_nodes=20_000):
    aligned = (basis["node_template"] + basis["residual_mean"]
               + features[:, 2:] @ basis["components"].T)
    output = np.empty_like(aligned, dtype=np.float32)
    for start in range(0, len(features), chunk_nodes):
        end = min(start + chunk_nodes, len(features))
        output[start:end] = decode_waveforms(
            aligned[start:end], features[start:end, 0], features[start:end, 1],
            basis["time"], basis["reference_at"], basis["reference_slope"],
            basis["inner_ms"], basis["outer_ms"])
    return output
