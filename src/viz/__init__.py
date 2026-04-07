"""Visualization utilities for SWA-MPPI experiments."""

from src.viz.style import setup_matplotlib, BASELINE_COLOR, SWA_COLOR, HUMAN_COLOR, CRITERIA_LABELS
from src.viz.radar import plot_radar_single, plot_radar_grid
from src.viz.bar_charts import plot_amce_comparison_bar
from src.viz.tables import plot_results_table
from src.viz.clustering import plot_cultural_clustering
from src.viz.heatmap import plot_alignment_heatmap
from src.viz.trigger import plot_decision_gap_analysis
from src.viz.comparison import plot_baseline_comparison, plot_comparison_table

__all__ = [
    # Style
    "setup_matplotlib",
    "BASELINE_COLOR",
    "SWA_COLOR",
    "HUMAN_COLOR",
    "CRITERIA_LABELS",
    # Shared plots
    "plot_radar_single",
    "plot_radar_grid",
    "plot_amce_comparison_bar",
    "plot_results_table",
    "plot_cultural_clustering",
    # SWA-specific plots
    "plot_alignment_heatmap",
    "plot_decision_gap_analysis",
    # Comparison plots (baseline vs SWA, 1:1 with main.py)
    "plot_baseline_comparison",
    "plot_comparison_table",
]
