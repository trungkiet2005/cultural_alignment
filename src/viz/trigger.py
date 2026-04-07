"""Trigger analysis and decision gap visualizations (SWA-MPPI specific)."""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.viz.style import SWA_COLOR, HUMAN_COLOR


def plot_trigger_analysis(all_summaries, config, output_dir):
    """Four-panel figure: variance distributions, trigger rate vs JSD,
    agent reward heatmap, and MPPI intervention strength."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    countries = [s["country"] for s in all_summaries]

    ax1 = fig.add_subplot(gs[0, 0])
    all_vars = [s["diagnostics"]["variances"] for s in all_summaries]
    bplot = ax1.boxplot(all_vars, tick_labels=countries, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='.', markersize=3, alpha=0.3))
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(countries)))
    for patch, color in zip(bplot['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax1.axhline(config.tau_conflict, color=HUMAN_COLOR, linestyle='--', linewidth=2,
                label=f'Default tau = {config.tau_conflict}')
    # Show per-country calibrated tau
    for i, s in enumerate(all_summaries):
        ax1.scatter([i + 1], [s.get("tau_used", config.tau_conflict)],
                    marker='D', color='#4CAF50', s=60, zorder=5,
                    label='Calibrated tau' if i == 0 else "")
    ax1.set_ylabel("Inter-Agent Reward Variance", fontsize=11)
    ax1.set_xlabel("Country", fontsize=11)
    ax1.set_title("(a) Variance Distribution & Calibrated \u03c4 per Country", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45); ax1.legend(fontsize=9); ax1.set_yscale('log')

    ax2 = fig.add_subplot(gs[0, 1])
    trigger_rates = [s["trigger_rate"] for s in all_summaries]
    jsds = [s["alignment"].get("jsd", np.nan) for s in all_summaries]
    ax2.scatter(trigger_rates, jsds, s=120, c=SWA_COLOR, edgecolors='white', linewidth=1.5, zorder=3)
    for i, label in enumerate(countries):
        ax2.annotate(label, (trigger_rates[i], jsds[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)
    ax2.set_xlabel("MPPI Trigger Rate", fontsize=11)
    ax2.set_ylabel("Jensen-Shannon Distance", fontsize=11)
    ax2.set_title("(b) Trigger Rate vs Alignment Quality", fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    example_summary = all_summaries[0]
    reward_matrix = np.array(example_summary["diagnostics"]["agent_reward_matrix"])
    n_show = min(50, reward_matrix.shape[0])
    im3 = ax3.imshow(reward_matrix[:n_show].T, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax3.set_xlabel(f"Scenario Index (first {n_show})", fontsize=11)
    ax3.set_ylabel("Agent Index", fontsize=11)
    ax3.set_title(f"(c) Agent Rewards [{example_summary['country']}]", fontsize=12, fontweight='bold')
    ax3.set_yticks(range(reward_matrix.shape[1]))
    ax3.set_yticklabels([f"Agent {i+1}" for i in range(reward_matrix.shape[1])], fontsize=9)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label("Expected Reward", fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    mean_dz = [np.mean(s["diagnostics"]["delta_z_norms"]) for s in all_summaries]
    colors_bar = [SWA_COLOR] * len(countries)
    ax4.barh(range(len(countries)), mean_dz, color=colors_bar, edgecolor='white', height=0.7)
    ax4.set_yticks(range(len(countries))); ax4.set_yticklabels(countries, fontsize=10)
    ax4.set_xlabel("Mean MPPI Correction Magnitude", fontsize=11)
    ax4.set_title("(d) MPPI Intervention Strength per Country", fontsize=12, fontweight='bold')
    ax4.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_trigger_analysis.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 3] Saved -> {path}")


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
    ax2.axvline(config.tau_conflict, color=HUMAN_COLOR, linestyle='--', linewidth=1.5,
                label=f'Default tau = {config.tau_conflict}')
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
