"""Bar chart visualizations for AMCE comparisons."""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.viz.style import HUMAN_COLOR


def plot_amce_comparison_bar(all_summaries, output_dir,
                             model_label="Vanilla LLM", model_color="#9E9E9E"):
    """
    Per-criterion AMCE bar chart showing model vs human
    across all countries, highlighting the bias-correction improvement.
    """
    categories = ["Species_Humans", "Gender_Female", "Age_Young",
                  "Fitness_Fit", "SocialValue_High", "Utilitarianism_More"]
    cat_labels = ["Species\n(Human)", "Gender\n(Female)", "Age\n(Young)",
                  "Fitness\n(Fit)", "Social\n(High)", "Util.\n(More)"]

    n_cats = len(categories)
    n_countries = len(all_summaries)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (cat, cat_label) in enumerate(zip(categories, cat_labels)):
        ax = axes[i]
        model_vals = [s["model_amce"].get(cat, np.nan) for s in all_summaries]
        human_vals = [s["human_amce"].get(cat, np.nan) for s in all_summaries]
        countries = [s["country"] for s in all_summaries]
        x = np.arange(n_countries)
        ax.bar(x - 0.2, model_vals, 0.4, label=model_label, color=model_color, alpha=0.85, edgecolor='white')
        ax.bar(x + 0.2, human_vals, 0.4, label='Human', color=HUMAN_COLOR, alpha=0.85, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("AMCE (%)", fontsize=10)
        ax.set_title(cat_label, fontsize=12, fontweight='bold')
        ax.axhline(50, color='gray', linestyle=':', linewidth=0.8)
        if i == 0: ax.legend(fontsize=9)

        # Per-country MAE for this criterion
        errors = [abs(m - h) for m, h in zip(model_vals, human_vals)
                  if not np.isnan(m) and not np.isnan(h)]
        if errors:
            ax.set_xlabel(f"Mean Error: {np.mean(errors):.1f} pp", fontsize=9)

    plt.suptitle(f"Per-Criterion AMCE: {model_label} vs Human Moral Machine",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig9_amce_per_criterion.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 9] Saved -> {path}")
