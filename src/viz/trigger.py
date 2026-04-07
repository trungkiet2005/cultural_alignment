"""Decision gap visualizations (SWA-MPPI specific)."""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.viz.style import SWA_COLOR, HUMAN_COLOR


def plot_decision_gap_analysis(all_summaries, config, output_dir):
    """Three-panel figure: decision gap distribution, variance vs correction,
    and intervention strength vs alignment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    countries = [s["country"] for s in all_summaries]

    ax1 = axes[0]
    all_gaps = [s["diagnostics"]["decision_gaps"] for s in all_summaries]
    bplot = ax1.boxplot(all_gaps, tick_labels=countries, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='.', markersize=3, alpha=0.3))
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(countries)))
    for patch, color in zip(bplot['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax1.axhline(0, color=HUMAN_COLOR, linestyle='--', linewidth=1.5, label='\u03b4=0 (no preference)')
    ax1.set_ylabel("Decision Gap \u03b4 (z_right \u2212 z_left, bias-corrected)", fontsize=11)
    ax1.set_xlabel("Country", fontsize=11)
    ax1.set_title("(a) Decision Gap Distribution (Bias-Corrected)", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45); ax1.legend(fontsize=10)

    ax2 = axes[1]
    color_map = plt.cm.tab10(np.linspace(0, 1, len(all_summaries)))
    for i, s in enumerate(all_summaries):
        vars_arr = s["diagnostics"]["variances"]
        dz_arr = s["diagnostics"]["delta_z_norms"]
        n = min(len(vars_arr), len(dz_arr))
        ax2.scatter(vars_arr[:n], dz_arr[:n], alpha=0.3, s=15, color=color_map[i], label=s["country"])
    ax2.set_xlabel("Inter-Agent Variance", fontsize=11)
    ax2.set_ylabel("MPPI Correction Magnitude", fontsize=11)
    ax2.set_title("(b) When Does MPPI Push Hard?", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2, loc='upper left'); ax2.set_xscale('log')

    ax3 = axes[2]
    mean_dz = [np.mean(s["diagnostics"]["delta_z_norms"]) for s in all_summaries]
    pearson_rs = [s["alignment"].get("pearson_r", np.nan) for s in all_summaries]
    ax3.scatter(mean_dz, pearson_rs, s=150, c=SWA_COLOR, edgecolors='white', linewidth=1.5, zorder=3)
    for i, name in enumerate(countries):
        ax3.annotate(name, (mean_dz[i], pearson_rs[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel("Mean MPPI Intervention Strength", fontsize=11)
    ax3.set_ylabel("Pearson r (Alignment)", fontsize=11)
    ax3.set_title("(c) Intervention Strength vs Alignment", fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_decision_gap_analysis.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 5] Saved -> {path}")
