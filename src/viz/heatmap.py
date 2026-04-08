"""Alignment heatmap visualization (SWA-PTIS specific)."""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.amce import compute_alignment_metrics
from src.viz.style import SWA_COLOR, HUMAN_COLOR


def plot_alignment_heatmap(all_summaries, output_dir):
    """Cross-cultural JSD matrix heatmap and per-country self-alignment bar chart."""
    countries = [s["country"] for s in all_summaries]
    n = len(countries)
    jsd_matrix = np.zeros((n, n))
    for i, si in enumerate(all_summaries):
        for j, sj in enumerate(all_summaries):
            metrics = compute_alignment_metrics(si["model_amce"], sj["human_amce"])
            jsd_matrix[i, j] = metrics.get("jsd", 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    ax1 = axes[0]
    im = ax1.imshow(jsd_matrix, cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=0.5)
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(countries, rotation=45, ha="right", fontsize=10)
    ax1.set_yticklabels(countries, fontsize=10)
    ax1.set_xlabel("Human Target Country", fontsize=12)
    ax1.set_ylabel("SWA-PTIS Model (Persona Country)", fontsize=12)
    ax1.set_title("(a) Cross-Cultural JSD Matrix", fontsize=13, fontweight='bold')
    for i in range(n):
        for j in range(n):
            color = "white" if jsd_matrix[i, j] > 0.3 else "black"
            ax1.text(j, i, f"{jsd_matrix[i, j]:.3f}", ha="center", va="center",
                     fontsize=8, color=color, fontweight='bold' if i == j else 'normal')
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                              edgecolor=SWA_COLOR, linewidth=2.5)
        ax1.add_patch(rect)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Jensen-Shannon Distance", fontsize=11)

    ax2 = axes[1]
    diag_jsd = np.diag(jsd_matrix)
    colors = [SWA_COLOR if v <= np.median(diag_jsd) else '#FF9800' for v in diag_jsd]
    bars = ax2.barh(range(n), diag_jsd, color=colors, edgecolor='white', height=0.7)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(countries, fontsize=10)
    ax2.set_xlabel("JSD (Self-Alignment)", fontsize=12)
    ax2.set_title("(b) Per-Country Self-Alignment", fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, diag_jsd)):
        ax2.text(val + 0.005, i, f"{val:.3f}", va='center', fontsize=9)
    mean_jsd = np.mean(diag_jsd)
    ax2.axvline(mean_jsd, color=HUMAN_COLOR, linestyle='--', linewidth=1.5,
                label=f'Mean JSD = {mean_jsd:.3f}')
    ax2.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_alignment_heatmap.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 2] Saved -> {path}")
