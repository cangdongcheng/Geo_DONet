"""Plots generated only when requested, beneath the run output directory."""
import os
from pathlib import Path
import numpy as np


def _pyplot(output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output).resolve().parent / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def save_loss_plot(rows, path):
    plt = _pyplot(path)
    values = np.asarray(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    valid = np.isfinite(values[:, 5])
    axes[0].semilogy(values[:, 0], values[:, 1], label="train total")
    axes[0].semilogy(values[:, 0], values[:, 2], label="train V_m")
    axes[0].semilogy(values[valid, 0], values[valid, 5], label="validation total")
    axes[0].semilogy(values[valid, 0], values[valid, 6], label="validation V_m")
    axes[0].set_title("waveform objective")
    axes[1].semilogy(values[:, 0], values[:, 3], label="train AT diagnostic")
    axes[1].semilogy(values[:, 0], values[:, 4], label="train PCA diagnostic")
    axes[1].semilogy(values[valid, 0], values[valid, 7], "--", label="validation AT")
    axes[1].semilogy(values[valid, 0], values[valid, 8], "--", label="validation PCA")
    axes[1].set_title("normalized feature diagnostics (not loss by default)")
    for axis in axes:
        axis.set_xlabel("epoch"); axis.set_ylabel("MSE")
        axis.legend(); axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)

def save_trace_plot(path, time_ms, truth, prediction, nodes, at_true, at_pred):
    plt = _pyplot(path)
    figure, axes = plt.subplots(1, len(nodes), figsize=(4 * len(nodes), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, node in zip(axes, nodes):
        axis.plot(time_ms, truth[node], "k-", lw=2, label="GT")
        axis.plot(time_ms, prediction[node], "C3--", lw=1.5, label="prediction")
        axis.set_title(f"node {node}\nAT {at_true[node]:.1f}/{at_pred[node]:.1f} ms", fontsize=9)
        axis.set_xlabel("time (ms)"); axis.grid(alpha=0.3)
    axes[0].set_ylabel("V_m (mV)"); axes[0].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)
