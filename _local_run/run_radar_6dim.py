#!/usr/bin/env python3
"""6-dimension radar plots: Vanilla vs DISCA vs Human across the 20 countries.

Generates three artefacts under _local_run/radar/:
  1. radar_mean_all_models.{pdf,png}   - one panel per model, AMCE averaged
                                          across the 20 countries (target plot)
  2. radar_<model>_grid.{pdf,png}      - 4x5 country grid per model
  3. radar_<model>_mean.{pdf,png}      - single mean radar per model

Reads:
  _local_run/human_amce_long.csv      (country, dimension, human_amce)
  exp_paper/result/exp24_paper_20c/<model>/swa/<slug>/{vanilla,swa}_results_<ISO>.csv

Run:  python _local_run/run_radar_6dim.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PAPER_20C = REPO / "exp_paper" / "result" / "exp24_paper_20c"
OUT = ROOT / "radar"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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
    "Age_Young":            "Age",
    "Fitness_Fit":          "Fitness",
    "SocialValue_High":    "Social\nValue",
    "Utilitarianism_More": "Utilitarian",
}
CAT_TO_DIM = {
    "Species":        "Species_Humans",
    "Gender":         "Gender_Female",
    "Age":            "Age_Young",
    "Fitness":        "Fitness_Fit",
    "SocialValue":    "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}

REGION_ORDER: List[Tuple[str, str]] = [
    ("USA", "Americas"), ("ARG", "Americas"), ("BRA", "Americas"),
    ("COL", "Americas"), ("MEX", "Americas"),
    ("GBR", "Europe"), ("DEU", "Europe"), ("ROU", "Europe"), ("SRB", "Europe"),
    ("CHN", "E. Asia"), ("JPN", "E. Asia"),
    ("IDN", "SE Asia"), ("MMR", "SE Asia"), ("MYS", "SE Asia"),
    ("THA", "SE Asia"), ("VNM", "SE Asia"),
    ("BGD", "S. Asia"), ("KGZ", "C. Asia"), ("IRN", "W. Asia"),
    ("ETH", "Africa"),
]
ALL_ISO = [iso for iso, _ in REGION_ORDER]

# (folder_under_paper_20c, slug_subdir, display_label) - paper headline 7
MODELS: List[Tuple[str, str, str]] = [
    ("llama33_70b",          "llama-3.3-70b-instruct-bnb-4bit",       "Llama-3.3-70B"),
    ("magistral_small_2509", "magistral-small-2509",                  "Magistral-Sml (24B)"),
    ("phi_4",                "phi-4",                                 "Phi-4 (14B)"),
    ("qwen3_vl_8b",          "qwen3-vl-8b-instruct-unsloth-bnb-4bit", "Qwen3-VL-8B"),
    ("hf_qwen25_7b_bf16",    "qwen2.5-7b-instruct",                   "Qwen2.5-7B"),
    ("phi35_mini",           "phi-3.5-mini-instruct",                 "Phi-3.5-mini"),
    ("gemma4_e2b",           "gemma-4-e2b-it",                        "Gemma-4-E2B"),
]

COL_DISCA   = "#1B7C3D"
COL_VANILLA = "#D6604D"
COL_HUMAN   = "#2166AC"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_human_amce() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with (ROOT / "human_amce_long.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["country"]][row["dimension"]] = float(row["human_amce"])
    return out


def amce_from_results(path: Path) -> Dict[str, float]:
    """Mean p_spare_preferred per phenomenon_category, scaled to [0, 100]."""
    if not path.exists():
        return {}
    bucket: Dict[str, List[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("phenomenon_category", "")
            if cat not in CAT_TO_DIM:
                continue
            try:
                bucket[cat].append(float(row["p_spare_preferred"]))
            except (KeyError, ValueError):
                continue
    return {
        CAT_TO_DIM[cat]: 100.0 * float(np.mean(vals))
        for cat, vals in bucket.items() if len(vals) >= 3
    }


def load_model(folder: str, slug: str) -> Tuple[Dict[str, Dict[str, float]],
                                                 Dict[str, Dict[str, float]]]:
    """Return (vanilla_amce, disca_amce), each indexed by ISO."""
    base = PAPER_20C / folder / "swa" / slug
    vanilla = {iso: amce_from_results(base / f"vanilla_results_{iso}.csv")
               for iso in ALL_ISO}
    disca   = {iso: amce_from_results(base / f"swa_results_{iso}.csv")
               for iso in ALL_ISO}
    return vanilla, disca


def mean_amce(per_country: Dict[str, Dict[str, float]],
              countries: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim in DIM_ORDER:
        vals = [per_country[c].get(dim) for c in countries
                if c in per_country and dim in per_country[c]]
        if vals:
            out[dim] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def radar_axes(ax, n_axes: int) -> List[float]:
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


def draw_polygon(ax, angles: List[float], values: List[float],
                 color: str, marker: str, ls: str, label: str,
                 alpha_fill: float = 0.12, lw: float = 1.7) -> None:
    closed_a = angles + [angles[0]]
    closed_v = values + [values[0]]
    ax.plot(closed_a, closed_v, marker=marker, ls=ls, color=color, lw=lw,
            ms=3.6, label=label)
    ax.fill(closed_a, closed_v, color=color, alpha=alpha_fill)


def panel(ax, title: str, human: Dict[str, float],
          vanilla: Dict[str, float], disca: Dict[str, float]) -> Tuple[float, float]:
    angles = radar_axes(ax, len(DIM_ORDER))
    h = [human.get(d, np.nan)   for d in DIM_ORDER]
    v = [vanilla.get(d, np.nan) for d in DIM_ORDER]
    m = [disca.get(d, np.nan)   for d in DIM_ORDER]

    draw_polygon(ax, angles, h, COL_HUMAN,   "s", "--", "Human",
                 alpha_fill=0.10, lw=1.6)
    draw_polygon(ax, angles, v, COL_VANILLA, "^", "-",  "Vanilla",
                 alpha_fill=0.10, lw=1.6)
    draw_polygon(ax, angles, m, COL_DISCA,   "o", "-",  "DISCA",
                 alpha_fill=0.16, lw=1.9)
    ax.plot(np.linspace(0, 2 * pi, 80), [50] * 80, ":",
            color="#aaa", lw=0.6, alpha=0.7)

    h_arr, v_arr, m_arr = np.array(h), np.array(v), np.array(m)
    mae_v = float(np.nanmean(np.abs(h_arr - v_arr)))
    mae_d = float(np.nanmean(np.abs(h_arr - m_arr)))
    ax.set_title(f"{title}\nMAE  V={mae_v:.1f}  →  D={mae_d:.1f} pp",
                 size=9.8, fontweight="bold", pad=10)
    return mae_v, mae_d


def add_legend(fig, axes, ncol: int = 3) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 1.005), fontsize=11, frameon=False)


# ---------------------------------------------------------------------------
# Plot routines
# ---------------------------------------------------------------------------
def plot_country_grid(label: str, slug_safe: str,
                      human: Dict[str, Dict[str, float]],
                      vanilla: Dict[str, Dict[str, float]],
                      disca: Dict[str, Dict[str, float]]) -> None:
    cols, rows = 5, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows),
                              subplot_kw={"polar": True})
    for ax, (iso, region) in zip(axes.flat, REGION_ORDER):
        if not vanilla.get(iso) or not disca.get(iso) or iso not in human:
            ax.set_visible(False)
            continue
        panel(ax, f"{iso}  ({region})", human[iso], vanilla[iso], disca[iso])
    add_legend(fig, axes)
    fig.suptitle(f"{label}: Vanilla vs DISCA vs Human across 6 dimensions (20 countries)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, OUT / f"radar_{slug_safe}_grid")


def plot_mean_per_model(per_model_means: List[Tuple[str, Dict[str, float],
                                                     Dict[str, float],
                                                     Dict[str, float]]]) -> None:
    """One panel per model, AMCE averaged across the 20 countries."""
    n = len(per_model_means)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.6 * rows),
                              subplot_kw={"polar": True})
    axes = np.atleast_2d(axes)
    for ax, (label, h_mean, v_mean, m_mean) in zip(axes.flat, per_model_means):
        panel(ax, label, h_mean, v_mean, m_mean)
    for ax in axes.flat[len(per_model_means):]:
        ax.set_visible(False)
    add_legend(fig, axes)
    fig.suptitle("Mean across 20 countries: Vanilla vs DISCA vs Human (6 dims)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, OUT / "radar_mean_all_models")


def plot_single_mean(label: str, slug_safe: str,
                     h_mean: Dict[str, float],
                     v_mean: Dict[str, float],
                     m_mean: Dict[str, float]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5),
                           subplot_kw={"polar": True})
    panel(ax, f"{label} - mean over 20 countries", h_mean, v_mean, m_mean)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10),
              fontsize=10, frameon=False)
    plt.tight_layout()
    save(fig, OUT / f"radar_{slug_safe}_mean")


def save(fig, base: Path) -> None:
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[OK] {base.name}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    human = load_human_amce()
    h_mean = mean_amce(human, ALL_ISO)

    per_model_means: List[Tuple[str, Dict[str, float],
                                Dict[str, float], Dict[str, float]]] = []

    for folder, slug, label in MODELS:
        print(f"\n=== {label} ===")
        vanilla, disca = load_model(folder, slug)
        n_v = sum(1 for c in ALL_ISO if vanilla.get(c))
        n_d = sum(1 for c in ALL_ISO if disca.get(c))
        print(f"  vanilla coverage: {n_v}/20    DISCA coverage: {n_d}/20")
        if n_v == 0 and n_d == 0:
            print(f"  [SKIP] no data for {label}")
            continue

        slug_safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "")\
                          .replace("/", "-").replace(".", "")
        plot_country_grid(label, slug_safe, human, vanilla, disca)

        v_mean = mean_amce(vanilla, [c for c in ALL_ISO if vanilla.get(c)])
        m_mean = mean_amce(disca,   [c for c in ALL_ISO if disca.get(c)])
        plot_single_mean(label, slug_safe, h_mean, v_mean, m_mean)
        per_model_means.append((label, h_mean, v_mean, m_mean))

    if per_model_means:
        plot_mean_per_model(per_model_means)

    print(f"\n[DONE] outputs in {OUT}")


if __name__ == "__main__":
    main()
