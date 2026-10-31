"""Plot protocol-aware comparisons with the VirTues results.

The panels deliberately separate in-cohort performance from the MAPS comparison.
They are related context, not a single head-to-head benchmark.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_publication"


def add_value_labels(ax, bars, values, fmt="{:.3f}"):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    # Panel A: same metric, but both VirTues values and the project's value are
    # supervised/in-cohort context rather than the project's zero-shot headline.
    labels = ["VirTues\nCODEX", "VirTues\nOrion", "This paper\nsupervised ceiling"]
    values = [0.705, 0.652, 0.684]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    x = np.arange(len(labels))
    bars = ax_left.bar(x, values, color=colors, width=0.68)
    add_value_labels(ax_left, bars, values)
    ax_left.set_ylim(0, 0.82)
    ax_left.set_ylabel("Macro-F1")
    ax_left.set_xticks(x, labels)
    ax_left.set_title("A  Supervised in-cohort context")
    ax_left.text(
        0.02,
        0.97,
        "Comparable metric and setting family\n"
        "but not identical cohorts or protocols",
        transform=ax_left.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax_left.text(
        0.02,
        0.03,
        "The paper's zero-shot LOCO headline = 0.315\n"
        "and is intentionally not plotted here as a direct bar.",
        transform=ax_left.transAxes,
        va="bottom",
        fontsize=8.5,
        color="#555555",
        bbox={"facecolor": "#F2F2F2", "edgecolor": "none", "pad": 5},
    )

    # Panel B: MAPS comparison. VirTues reports relative gains; this paper has
    # directly reported absolute Danenberg and overall values.
    labels = [
        "VirTues\nDanenberg",
        "VirTues\nMoldoveanu",
        "This paper\nDanenberg",
        "This paper\noverall LOCO",
    ]
    values = [27.31, 11.31, (0.0513 / 0.0893 - 1) * 100, (0.3151 / 0.3324 - 1) * 100]
    colors = ["#54A24B", "#86BC86", "#E45756", "#B279A2"]
    x = np.arange(len(labels))
    bars = ax_right.bar(x, values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            value + (1.5 if value >= 0 else -3.5),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax_right.axhline(0, color="#333333", linewidth=0.9)
    ax_right.set_ylim(-55, 40)
    ax_right.set_ylabel("Change versus MAPS (%)")
    ax_right.set_xticks(x, labels)
    ax_right.set_title("B  MAPS comparison")
    ax_right.text(
        0.02,
        0.97,
        "Positive = method is higher than MAPS\n"
        "Negative = method is lower than MAPS",
        transform=ax_right.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax_right.text(
        0.02,
        0.03,
        "VirTues values are reported relative gains;\n"
        "this paper's values use its LOCO protocol.",
        transform=ax_right.transAxes,
        va="bottom",
        fontsize=8.5,
        color="#555555",
        bbox={"facecolor": "#F2F2F2", "edgecolor": "none", "pad": 5},
    )

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(
        "VirTues and this paper: protocol-aware comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Context only: VirTues supplies the label vocabulary at inference; "
        "this paper derives a fold-local vocabulary for zero-shot LOCO transfer.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "Figure7_VirTues_Comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure7_VirTues_Comparison.pdf", bbox_inches="tight")
    print(OUT / "Figure7_VirTues_Comparison.png")
    print(OUT / "Figure7_VirTues_Comparison.pdf")


if __name__ == "__main__":
    main()
