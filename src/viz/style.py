"""Matplotlib style configuration and shared visual constants."""

import warnings
import matplotlib
import matplotlib.pyplot as plt

BASELINE_COLOR = "#9E9E9E"
SWA_COLOR = "#2196F3"
HUMAN_COLOR = "#E53935"

CRITERIA_LABELS = {
    "Species_Humans":       "Sparing\nHumans",
    "Age_Young":            "Sparing\nYoung",
    "Fitness_Fit":          "Sparing\nFit",
    "Gender_Female":        "Sparing\nFemales",
    "SocialValue_High":     "Sparing\nHigher Status",
    "Utilitarianism_More":  "Sparing\nMore",
}

def setup_matplotlib():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    warnings.filterwarnings("ignore", category=FutureWarning)
