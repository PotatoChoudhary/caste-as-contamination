"""
castelens/figures.py — consistent matplotlib output for all figures.
Self-tests by writing a smoke PNG (no model needed):  python -m castelens.figures
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})


def _save(fig, name):
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


def line_ci(x, ys, los, his, labels, xlabel, ylabel, title, name):
    """Dose-response style: one line per series with shaded CI."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for y, lo, hi, lab in zip(ys, los, his, labels):
        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, lo, hi, alpha=0.18)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    return _save(fig, name)


def bars_ci(labels, points, los, his, ylabel, title, name):
    """Three-bar mediation figure etc."""
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    err = [np.array(points) - np.array(los), np.array(his) - np.array(points)]
    ax.bar(x, points, yerr=err, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set(ylabel=ylabel, title=title)
    return _save(fig, name)


def heatmap(matrix, row_labels, col_labels, title, name):
    """Cross-steering matrix (rows=steered direction, cols=measured outcome)."""
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 2, 0.6 * len(row_labels) + 2))
    m = np.asarray(matrix)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-np.abs(m).max(), vmax=np.abs(m).max())
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return _save(fig, name)


if __name__ == "__main__":
    x = [-8, -4, -2, 2, 4, 8]
    y = [[-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]]
    lo = [[v - 0.1 for v in y[0]]]; hi = [[v + 0.1 for v in y[0]]]
    line_ci(x, y, lo, hi, ["V1 steer"], "alpha", "M1", "smoke", "_smoke_line.png")
    bars_ci(["full", "rand-stripped", "disgust-stripped"], [1.0, 0.95, 0.4],
            [0.9, 0.85, 0.3], [1.1, 1.05, 0.5], "caste effect", "smoke", "_smoke_bars.png")
    heatmap(np.random.randn(6, 3), ["V3", "V4", "V5", "V7", "rand", "V6"],
            ["M1", "M3", "M4"], "smoke", "_smoke_heat.png")
    print("SELF-TEST PASS (3 PNGs written to figures/)")
