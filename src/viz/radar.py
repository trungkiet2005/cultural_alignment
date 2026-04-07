"""Radar plot visualizations for model vs human moral preferences."""

import os
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from src.viz.style import CRITERIA_LABELS, HUMAN_COLOR


def plot_radar_single(model_amce, human_amce, country, alignment, ax=None, save_path=None,
                      model_label="Vanilla LLM", model_color="#9E9E9E"):
    """Plot a single radar chart comparing model and human AMCE profiles."""
    common_keys = sorted(set(model_amce.keys()) & set(human_amce.keys()))
    if len(common_keys) < 3:
        print(f"[WARN] Not enough common criteria for radar plot ({country})")
        return

    labels = [CRITERIA_LABELS.get(k, k.replace("_", "\n")) for k in common_keys]
    model_vals = [model_amce[k] for k in common_keys]
    human_vals = [human_amce[k] for k in common_keys]
    N = len(common_keys)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    model_plot = model_vals + [model_vals[0]]
    human_plot = human_vals + [human_vals[0]]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'polar': True})
        standalone = True
    else:
        standalone = False

    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=9, color='#333333')
    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="#666666", size=8)
    ax.set_ylim(0, 100)

    ax.plot(angles, model_plot, 'o-', linewidth=2.2, color=model_color,
            label=model_label, markersize=5)
    ax.fill(angles, model_plot, alpha=0.15, color=model_color)
    ax.plot(angles, human_plot, 's--', linewidth=2.0, color=HUMAN_COLOR,
            label=f'Human ({country})', markersize=5)
    ax.fill(angles, human_plot, alpha=0.08, color=HUMAN_COLOR)
    ax.plot(np.linspace(0, 2 * pi, 100), [50] * 100, ':', color='#999999', linewidth=0.8, alpha=0.6)

    jsd_str = f"JSD={alignment.get('jsd', 0):.3f}" if 'jsd' in alignment else ""
    r_str = f"r={alignment.get('pearson_r', 0):.3f}" if 'pearson_r' in alignment else ""
    ax.set_title(f"{country}\n{jsd_str}  {r_str}" if jsd_str else country,
                 size=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')

    if standalone and save_path:
        plt.tight_layout(); plt.savefig(save_path); plt.show(); plt.close()


def plot_radar_grid(all_summaries, output_dir,
                    amce_key="model_amce", alignment_key="alignment",
                    title_suffix="", file_suffix="",
                    model_label="Vanilla LLM", model_color="#9E9E9E",
                    fig_title=None):
    """Plot a grid of radar charts, one per country."""
    n = len(all_summaries)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                              subplot_kw={'polar': True})
    if n == 1: axes = np.array([axes])
    axes = axes.flatten()
    for i, summary in enumerate(all_summaries):
        m_amce = summary.get(amce_key, summary["model_amce"])
        align = summary.get(alignment_key, summary["alignment"])
        plot_radar_single(m_amce, summary["human_amce"],
                          summary["country"], align, ax=axes[i],
                          model_label=model_label, model_color=model_color)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    default_title = f"Cultural Alignment: Model vs Human Preferences{title_suffix}"
    fig.suptitle(fig_title or default_title,
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f"fig1_radar_grid{file_suffix}"
    path = os.path.join(output_dir, f"{fname}.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 1] Saved -> {path}")
