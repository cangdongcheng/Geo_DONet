"""Plot saved per-heart metrics; never loads Vm data or runs inference."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OPERATOR = (HERE.parent / "Geo_DeepONet_PCA/Predictions/"
            "geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep/Test/vm_metrics.csv")
MLP = (HERE / "Predictions/geomlp_pca_vmloss_k5_w200_d4_n2048_5000ep/"
       "Test/vm_metrics.csv")
METRICS = (
    ("vm_rel_l2", r"$V_m$ relative L2", "Relative L2", 1, 3),
    ("vm_mae_mv", r"$V_m$ MAE", "mV", 1, 3),
    ("decoded_at_mae_ms", "Decoded AT MAE", "ms", 1, 3),
    ("dvdt_fraction", "Upstroke retention", "% of GT max-upstroke metric", 100, 1),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-metrics", type=Path, default=OPERATOR)
    parser.add_argument("--mlp-metrics", type=Path, default=MLP)
    parser.add_argument("--out-dir", type=Path, default=HERE / "Comparison")
    args = parser.parse_args()
    sources = (args.operator_metrics, args.mlp_metrics)
    tables = []
    for path in sources:
        data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        required = {"case", *(item[0] for item in METRICS)}
        if not required.issubset(data.dtype.names or ()):
            raise ValueError(f"missing metric columns: {path}")
        if len(np.unique(data["case"])) != len(data):
            raise ValueError(f"duplicate case IDs: {path}")
        for key in required:
            if not np.isfinite(data[key]).all():
                raise ValueError(f"non-finite {key}: {path}")
        tables.append(np.sort(data, order="case"))
    if not np.array_equal(tables[0]["case"], tables[1]["case"]):
        raise ValueError("The two files must evaluate exactly the same test hearts")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = ("#2878A6", "#DB8540")
    labels = ("DeepONet\nAT + 5 PC", "MLP\nAT + 5 PC")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    rows = []
    for axis, (key, title, unit, scale, digits) in zip(axes, METRICS):
        values = [table[key] * scale for table in tables]
        means = np.array([v.mean() for v in values])
        stds = np.array([v.std(ddof=0) for v in values])
        axis.bar([0, 1], means, width=0.55, color=colors, alpha=0.9,
                 yerr=stds, capsize=5, error_kw={"elinewidth": 1.4})
        for position, (mean, std) in enumerate(zip(means, stds)):
            axis.annotate(f"{mean:.{digits}f} ± {std:.{digits}f}",
                          (position, mean + std), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=10)
        axis.set_xticks([0, 1], labels)
        axis.set_ylabel(unit)
        axis.set_title(title + ("\n(higher is better)" if scale == 100
                                else "\n(lower is better)"), fontsize=12, pad=12)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.18)
        axis.set_xlim(-0.65, 1.65)
        if scale == 100:
            axis.set_ylim(0, 112)
            axis.axhline(100, color="#777777", linestyle="--", linewidth=1)
            axis.text(0.5, 102, "GT = 100%", ha="center", color="#666666", fontsize=9)
        else:
            axis.set_ylim(0, float((means + stds).max()) * 1.27)
        rows.append((key, means[0] / scale, stds[0] / scale,
                     means[1] / scale, stds[1] / scale, len(tables[0])))

    figure.suptitle("AT + PCA with waveform supervision: DeepONet vs vanilla MLP",
                   fontsize=16, fontweight="bold", y=0.99)
    figure.text(0.5, 0.905,
                f"Same {len(tables[0])} test hearts • Bars: mean ± population SD across hearts",
                ha="center", fontsize=11, color="#555555")
    figure.text(0.5, 0.025,
                "Both use the fixed waveform decoder. Upstroke retention is the per-heart "
                "ratio of median nodewise maximum |dV/dt| (prediction / GT).",
                ha="center", fontsize=9, color="#555555")
    figure.subplots_adjust(left=0.06, right=0.985, top=0.735, bottom=0.20, wspace=0.43)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = args.out_dir / f"mlp_vs_deeponet_pca.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(figure)
    with (args.out_dir / "comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "deeponet_mean", "deeponet_std", "mlp_mean", "mlp_std", "n_hearts"))
        writer.writerows(rows)
    with (args.out_dir / "sources.txt").open("w") as handle:
        handle.write(f"DeepONet: {sources[0].resolve()}\nMLP: {sources[1].resolve()}\n"
                     "SD uses ddof=0; error bars are not confidence intervals.\n"
                     "CSV retains the original dimensionless upstroke fraction; plot shows percent.\n")
    for key, deep_mean, deep_std, mlp_mean, mlp_std, n in rows:
        print(f"{key}: DeepONet {deep_mean:.6f} +/- {deep_std:.6f}; "
              f"MLP {mlp_mean:.6f} +/- {mlp_std:.6f}; N={n}")


if __name__ == "__main__":
    main()
