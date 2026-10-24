"""Canonical-mesh SVG rendering with the original physical color scales."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VM_MIN, VM_MAX = -90.0, 50.0
VM_ERR_MIN, VM_ERR_MAX = 0.0, 20.0
CMAP_VM = "RdYlBu_r"
CMAP_ERR = "Reds"

def save_vm_traces(prediction, truth, time_ms, out_path, n_sample_nodes=5):
    """V_m(t) at a few evenly-spaced nodes. `truth` may be None (pred only)."""
    nodes = np.linspace(0, prediction.shape[0] - 1, n_sample_nodes, dtype=int)
    fig, axes = plt.subplots(len(nodes), 1, figsize=(10, 2.2 * len(nodes)), sharex=True)
    for ax, node in zip(axes, nodes):
        if truth is not None:
            ax.plot(time_ms, truth[node], "k-", lw=2.5, label="GT")
        ax.plot(time_ms, prediction[node], "r--", lw=2.0, label="Pred")
        ax.set_ylabel(f"node {node}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close()

def _init_plot_worker(xyz):
    global _WORKER_XYZ
    _WORKER_XYZ = xyz
    import matplotlib
    matplotlib.use("Agg")

def _render_scatter_svg(task):
    values, cmap, vmin, vmax, out_path = task
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6)); fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d"); ax.patch.set_alpha(0.0)
    ax.scatter(_WORKER_XYZ[:, 0], _WORKER_XYZ[:, 1], _WORKER_XYZ[:, 2],
               c=values, cmap=cmap, s=1, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_axis_off(); plt.tight_layout()
    plt.savefig(out_path, format="svg", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

def save_colorbar(cmap, vmin, vmax, label, out_path, half=False):
    """Standalone vertical colorbar SVG (inherited from Geo_DONet/main.py)."""
    fig, ax = plt.subplots(figsize=(1.2, 2.5 if half else 5))
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(sm, cax=ax)
    cb.set_label(label, fontsize=15)
    from matplotlib.ticker import MaxNLocator
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    cb.set_ticks(ticks[(ticks >= vmin) & (ticks <= vmax)])
    cb.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close()
