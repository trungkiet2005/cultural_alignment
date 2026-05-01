#!/usr/bin/env python3
"""Quick 6-dimension radar grid: Phi-4 DISCA vs Human across 20 countries.

Reads:
  _local_run/human_amce_long.csv      (country, dimension, human_amce)
  _local_run/phi4_model_amce_long.csv (country, dimension, model_amce)

Writes:
  _local_run/radar_phi4_6dim.{pdf,png}

Run:  python _local_run/run_radar_6dim.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

# Canonical dimension order shown on the radar (clockwise from top)
DIM_ORDER = [
    "Species_Humans",
    "Gender_Female",
    "Age_Young",
    "Fitness_Fit",
    "SocialValue_High",
    "Utilitarianism_More",
]
DIM_LABELS = {
    "Species_Humans":      "Species",
    "Gender_Female":       "Gender",
    "Age_Young":           "Age",
    "Fitness_Fit":         "Fitness",
    "SocialValue_High":    "Social\nValue",
    "Utilitarianism_More": "Utilitarian",
}

REGION_ORDER = [
    ("USA", "Americas"), ("ARG", "Americas"), ("BRA", "Americas"),
    ("COL", "Americas"), ("MEX", "Americas"),
    ("GBR", "Europe"), ("DEU", "Europe"), ("ROU", "Europe"), ("SRB", "Europe"),
    ("CHN", "E. Asia"), ("JPN", "E. Asia"),
    ("IDN", "SE Asia"), ("MMR", "SE Asia"), ("MYS", "SE Asia"),
    ("THA", "SE Asia"), ("VNM", "SE Asia"),
    ("BGD", "S. Asia"), ("KGZ", "C. Asia"), ("IRN", "W. Asia"),
    ("ETH", "Africa"),
]

COL_DISCA = "#1B7C3D"
COL_HUMAN = "#2166AC"


def load_amce(path: Path, value_col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["country"]][row["dimension"]] = float(row[value_col])
    return out


def radar_axes(ax, n_axes: int):
    angles = [k / n_axes * 2 * pi for k in range(n_axes)]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIM_ORDER], size=8, color="#333")
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], color="#888", size=7)
    ax.set_ylim(0, 100)
    ax.grid(color="#dddddd", linewidth=0.6)
    return angles


def plot_one(ax, country, region, human_d, disca_d):
    angles = radar_axes(ax, len(DIM_ORDER))
    closed_angles = angles + [angles[0]]

    h = [human_d.get(d, np.nan) for d in DIM_ORDER]
    m = [disca_d.get(d, np.nan) for d in DIM_ORDER]
    h_closed = h + [h[0]]
    m_closed = m + [m[0]]

    ax.plot(closed_angles, h_closed, "s--", color=COL_HUMAN, lw=1.6,
            ms=3.5, label="Human")
    ax.fill(closed_angles, h_closed, color=COL_HUMAN, alpha=0.10)
    ax.plot(closed_angles, m_closed, "o-", color=COL_DISCA, lw=1.8,
            ms=3.5, label="Phi-4 DISCA")
    ax.fill(closed_angles, m_closed, color=COL_DISCA, alpha=0.18)
    ax.plot(np.linspace(0, 2 * pi, 80), [50] * 80, ":",
            color="#aaa", lw=0.6, alpha=0.7)

    # mean abs error across dims (in pp) as a quick alignment indicator
    err = np.nanmean(np.abs(np.array(h) - np.array(m)))
    ax.set_title(f"{country}  ({region})\nMAE={err:.1f} pp",
                 size=9.5, fontweight="bold", pad=10)


def main():
    human = load_amce(ROOT / "human_amce_long.csv", "human_amce")
    disca = load_amce(ROOT / "phi4_model_amce_long.csv", "model_amce")

    cols, rows = 5, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows),
                              subplot_kw={"polar": True})
    axes = axes.flatten()

    for ax, (iso, region) in zip(axes, REGION_ORDER):
        if iso not in human or iso not in disca:
            ax.set_visible(False)
            continue
        plot_one(ax, iso, region, human[iso], disca[iso])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.005), fontsize=11, frameon=False)
    fig.suptitle(
        "Phi-4 DISCA vs Human moral preferences across 6 dimensions (20 countries)",
        fontsize=14, fontweight="bold", y=1.025,
    )
    plt.tight_layout()

    out_pdf = ROOT / "radar_phi4_6dim.pdf"
    out_png = ROOT / "radar_phi4_6dim.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[OK] Saved -> {out_pdf}")
    print(f"[OK] Saved -> {out_png}")


if __name__ == "__main__":
    main()
