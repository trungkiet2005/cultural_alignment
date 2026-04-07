"""Cultural clustering visualizations (dendrogram + MDS projection)."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


def plot_cultural_clustering(all_summaries, output_dir):
    """Hierarchical clustering dendrogram and MDS projection of moral profiles."""
    countries = [s["country"] for s in all_summaries]
    all_criteria = sorted(set().union(*[s["model_amce"].keys() for s in all_summaries]))
    feature_matrix = np.array([[s["model_amce"].get(c, 50.0) for c in all_criteria]
                                for s in all_summaries])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    distances = pdist(feature_matrix, metric='euclidean')
    Z = linkage(distances, method='ward')
    dendrogram(Z, labels=countries, ax=axes[0], leaf_rotation=45,
               leaf_font_size=10, color_threshold=0.7 * max(Z[:, 2]))
    axes[0].set_title("(a) Hierarchical Clustering of Moral Profiles", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Ward Distance", fontsize=11)

    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        coords = mds.fit_transform(squareform(distances))
        axes[1].scatter(coords[:, 0], coords[:, 1], s=200, c='#2196F3',
                        edgecolors='white', linewidth=2, zorder=3)
        for i, country in enumerate(countries):
            axes[1].annotate(country, (coords[i, 0], coords[i, 1]),
                             xytext=(8, 8), textcoords='offset points', fontsize=11,
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                                       edgecolor='#90CAF9', alpha=0.8))
        axes[1].set_title("(b) MDS Projection of Moral Profiles", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("MDS Dimension 1", fontsize=11)
        axes[1].set_ylabel("MDS Dimension 2", fontsize=11)
    except ImportError:
        axes[1].text(0.5, 0.5, "sklearn not available", ha='center', va='center',
                     transform=axes[1].transAxes)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig7_cultural_clustering.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 7] Saved -> {path}")
