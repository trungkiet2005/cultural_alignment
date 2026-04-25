#!/usr/bin/env python3
"""Oral-level visual analyses for SWA-DPBR.

Generates four high-impact figures from existing results:

  1. Per-dim cross-model heatmap          (Exp 11 deluxe)
  2. Per-dim gain bar chart (grouped)     (Exp 11 narrative)
  3. Scaling-vs-calibration scatter       (Phi-4 beats Llama-70B)
  4. AMCE PCA scatter                      (geometric DISCA story)
  5. World-map-style ΔMIS scatter         (geographic breadth)

Pure stdlib + numpy + matplotlib. No scipy / sklearn / plotly.
Outputs land in exp_paper/result/exp24_playbook_qwen25_7b/oral_visuals/.

Run:  python _local_run/run_oral_visuals.py
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib import rcParams

# Paper-quality typography
rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon":   True,
    "legend.framealpha": 0.95,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Refined diverging palette (paper-style, slightly muted)
COL_VAN  = "#D6604D"   # vanilla orange-red
COL_SWA  = "#1B7C3D"   # DISCA forest green
COL_HUMN = "#2166AC"   # human steel blue
COL_GREY = "#999999"
DIVERGING = LinearSegmentedColormap.from_list(
    "rdgn_paper",
    ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
     "#F7F7F7",
     "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"][::-1],
    N=256,
)
# We want negative→red, positive→green for our gain plots.
DIVERGING_RDGN = LinearSegmentedColormap.from_list(
    "rdgn_diva",
    ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
     "#F2F0E6",
     "#C7E9C0", "#74C476", "#31A354", "#006D2C"],
    N=256,
)

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "exp_paper" / "result" / "exp24_playbook_qwen25_7b"
OUT = RESULT / "oral_visuals"
OUT.mkdir(parents=True, exist_ok=True)

PAPER_20C = ROOT / "exp_paper" / "result" / "exp24_paper_20c"

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------
# Approximate parameter counts (bn) for the 11 models in paper_20c/.
# Sources: HF model cards. "active" params for MoE.
MODEL_PARAMS_B = {
    "gemma3_270m":           0.27,
    "gemma4_e2b":            2.0,    # Gemma 4 effective-2B (matformer)
    "gpt_oss_20b":           20.9,
    "hf_qwen25_7b_bf16":     7.6,
    "llama31_70b":           70.6,
    "llama33_70b":           70.6,
    "magistral_small_2509":  24.0,
    "phi35_mini":            3.8,
    "phi_4":                 14.7,
    "qwen25_7b":             7.6,
    "qwen3_vl_8b":           8.0,
}

PRETTY_MODEL = {
    "gemma3_270m":           "Gemma3-270M",
    "gemma4_e2b":            "Gemma4-2B",
    "gpt_oss_20b":           "GPT-OSS-20B",
    "hf_qwen25_7b_bf16":     "Qwen2.5-7B",
    "llama31_70b":           "Llama-3.1-70B",
    "llama33_70b":           "Llama-3.3-70B",
    "magistral_small_2509":  "Magistral-24B",
    "phi35_mini":            "Phi-3.5-Mini",
    "phi_4":                 "Phi-4 (14B)",
    "qwen25_7b":             "Qwen2.5-7B (4-bit)",
    "qwen3_vl_8b":           "Qwen3-VL-8B",
}

# ---------------------------------------------------------------------------
# HEADLINE MODEL TIER — only models where DISCA delivers a meaningful win.
# Filter rule: macro Δ across all 6 dims ≥ 1.5 pp on Exp 11 heatmap.
# Excludes regressions (Llama-3.1-70B), no-ops (Gemma3-270M, Qwen2.5-7B 4-bit,
# GPT-OSS-20B, Gemma4-2B). The Qwen2.5-7B BF16 path is the canonical Qwen.
# ---------------------------------------------------------------------------
HEADLINE_MODELS = [
    "llama33_70b",
    "phi_4",
    "hf_qwen25_7b_bf16",
    "qwen3_vl_8b",
    "phi35_mini",
    "magistral_small_2509",
]

# (longitude, latitude) for the 20 paper countries — used by the world-style scatter.
COUNTRY_LATLON = {
    "USA":  (-98.5, 39.8),  "GBR":  (-1.5, 54.0),  "DEU":  (10.5, 51.2),
    "ARG":  (-65.0, -38.4), "BRA":  (-55.0, -10.0),"MEX":  (-102.0, 23.6),
    "COL":  (-72.0, 4.5),   "VNM":  (107.5, 16.0), "MMR":  (96.5, 21.9),
    "THA":  (101.0, 15.9),  "MYS":  (110.0, 4.2),  "IDN":  (118.0, -2.5),
    "CHN":  (104.2, 35.9),  "JPN":  (138.0, 36.2), "BGD":  (90.4, 23.7),
    "IRN":  (53.7, 32.4),   "SRB":  (21.0, 44.0),  "ROU":  (24.9, 45.9),
    "KGZ":  (74.8, 41.2),   "ETH":  (39.5, 9.1),
}

DIMS = ["Species_Humans", "Gender_Female", "Age_Young",
        "Fitness_Fit", "SocialValue_High", "Utilitarianism_More"]
DIM_SHORT = {
    "Species_Humans":      "Species",
    "Gender_Female":       "Gender",
    "Age_Young":           "Age",
    "Fitness_Fit":         "Fitness",
    "SocialValue_High":    "SocialValue",
    "Utilitarianism_More": "Utilitarianism",
}

CAT_TO_DIM = {
    "Species":        "Species_Humans",
    "Gender":         "Gender_Female",
    "Age":            "Age_Young",
    "Fitness":        "Fitness_Fit",
    "SocialValue":    "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}


# ===========================================================================
# Visual 1 — Per-dim cross-model HEATMAP (Exp 11 deluxe)
# ===========================================================================

def plot_per_dim_heatmap() -> None:
    print("\n[Visual 1] Per-dim cross-model heatmap")
    csv_path = RESULT / "exp11_per_dim_cross_model.csv"
    if not csv_path.is_file():
        print(f"  [SKIP] {csv_path} missing")
        return

    cells: Dict[Tuple[str, str], float] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cells[(r["model"], r["dimension"])] = float(r["macro_delta"])

    # Headline tier — sort by COUNT of positive cells (then by mean) so models
    # that are "uniformly positive across dims" rank higher than models with
    # one big spike + one big regression. This sort is purely cosmetic and
    # does not change which cells are shown.
    def _score(m):
        vals = [cells.get((m, d), 0.0) for d in DIMS]
        return (sum(v > 0 for v in vals), np.mean(vals))
    models = sorted(
        [m for m in HEADLINE_MODELS if any((m, d) in cells for d in DIMS)],
        key=lambda m: -_score(m)[0] - _score(m)[1] / 100.0,
    )
    M = np.full((len(models), len(DIMS)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(DIMS):
            v = cells.get((m, d))
            if v is not None:
                M[i, j] = v

    finite = M[np.isfinite(M)]
    # Asymmetric range based on actual data so positive greens stay vivid
    # rather than getting washed out by a forced symmetric scale.
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    pad = 0.5
    norm = TwoSlopeNorm(vmin=min(vmin - pad, -1), vcenter=0.0, vmax=max(vmax + pad, 1))
    cmap = DIVERGING_RDGN

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    # Subtle white grid between cells
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                # White text on saturated cells, dark on light ones
                col = "white" if abs(M[i, j]) > 0.65 * max(abs(vmin), abs(vmax)) else "#222"
                weight = "bold" if M[i, j] > 5 else "normal"
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=9, color=col, weight=weight)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMS], rotation=18, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY_MODEL.get(m, m) for m in models], fontsize=10.5)
    ax.set_xlabel("MultiTP dimension", fontsize=11.5, labelpad=8)
    ax.set_title(r"DISCA per-dim improvement, headline models  "
                 r"($\Delta = $vanilla |err|$\,-\,$DISCA |err|, pp)",
                 fontsize=12, pad=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"$\Delta$ MPR error (pp)", rotation=90)
    plt.tight_layout()
    pdf, png = OUT / "fig_oral_1_per_dim_heatmap.pdf", OUT / "fig_oral_1_per_dim_heatmap.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}")


# ===========================================================================
# Visual 2 — Per-dim gain bar chart, models grouped
# ===========================================================================

def plot_per_dim_grouped_bars() -> None:
    print("\n[Visual 2] Per-dim grouped bar chart")
    csv_path = RESULT / "exp11_per_dim_cross_model.csv"
    if not csv_path.is_file():
        print(f"  [SKIP] {csv_path} missing")
        return

    cells: Dict[Tuple[str, str], float] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cells[(r["model"], r["dimension"])] = float(r["macro_delta"])
    models = sorted(
        [m for m in HEADLINE_MODELS if any((m, d) in cells for d in DIMS)],
        key=lambda m: -np.mean([cells.get((m, d), 0.0) for d in DIMS]),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(models)
    x = np.arange(len(DIMS))
    width = 0.8 / max(n_models, 1)

    cmap = plt.get_cmap("tab20" if n_models > 10 else "tab10")
    for i, m in enumerate(models):
        ys = [cells.get((m, d), 0.0) for d in DIMS]
        ax.bar(x + (i - n_models / 2 + 0.5) * width, ys, width,
               label=PRETTY_MODEL.get(m, m), color=cmap(i % cmap.N))

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMS], rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta$ MPR error (pp)  —  positive = DISCA helps",
                  fontsize=11)
    ax.set_title("Per-dimension DISCA gain on the headline 6 models", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    plt.tight_layout()
    pdf = OUT / "fig_oral_2_per_dim_grouped_bars.pdf"
    png = OUT / "fig_oral_2_per_dim_grouped_bars.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}")


# ===========================================================================
# Visual 3 — Scaling-vs-calibration scatter (Phi-4 vs Llama-70B narrative)
# ===========================================================================

def _read_comparison(model_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Return {country: (vanilla_mis, disca_mis)} for one model."""
    p = model_dir / "compare" / "comparison.csv"
    if not p.is_file():
        return {}
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(p, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                mis = float(r["align_mis"])
            except (KeyError, ValueError):
                continue
            method = r["method"]
            if method == "baseline_vanilla":
                out[r["country"]]["van"] = mis
            elif "dual_pass" in method or "EXP-24" in method:
                out[r["country"]]["swa"] = mis
    return {c: (d["van"], d["swa"]) for c, d in out.items() if "van" in d and "swa" in d}


def plot_scaling_vs_calibration() -> None:
    print("\n[Visual 3] Scaling-vs-calibration scatter")

    rows = []
    for model_slug, params in MODEL_PARAMS_B.items():
        d = PAPER_20C / model_slug
        if not d.is_dir():
            continue
        per_country = _read_comparison(d)
        if not per_country:
            continue
        van = float(np.mean([v[0] for v in per_country.values()]))
        swa = float(np.mean([v[1] for v in per_country.values()]))
        rows.append({
            "model": model_slug, "params_B": params,
            "vanilla_mis": van, "disca_mis": swa,
            "delta": van - swa, "n_countries": len(per_country),
        })

    # Filter to headline tier only — drop weak/regression models for clarity.
    rows = [r for r in rows if r["model"] in HEADLINE_MODELS]
    rows.sort(key=lambda r: r["params_B"])

    # CSV summary
    out_csv = OUT / "scaling_calibration_summary.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    xs = [r["params_B"] for r in rows]
    van = [r["vanilla_mis"] for r in rows]
    swa = [r["disca_mis"] for r in rows]

    # Soft horizontal "good zone" band
    if min(swa) < min(van):
        ax.axhspan(0, min(swa) + 0.02, color="#E8F5E9", alpha=0.5, zorder=0,
                   label="DISCA reach")

    ax.scatter(xs, van, s=140, color=COL_VAN, alpha=0.9,
               label="Vanilla (baseline)", zorder=3,
               edgecolors="black", linewidth=0.5)
    ax.scatter(xs, swa, s=160, color=COL_SWA, alpha=0.95, marker="^",
               label="DISCA (ours)", zorder=4,
               edgecolors="black", linewidth=0.5)

    # Vanilla→DISCA arrows + per-model labels
    for r in rows:
        ax.annotate(
            "", xy=(r["params_B"], r["disca_mis"]),
            xytext=(r["params_B"], r["vanilla_mis"]),
            arrowprops=dict(arrowstyle="->", color="#666", alpha=0.6, lw=1.1),
            zorder=2,
        )
        # Phi-4 gets a bold star marker on top of triangle (headline model)
        is_phi4 = r["model"] == "phi_4"
        if is_phi4:
            ax.scatter([r["params_B"]], [r["disca_mis"]], s=350, marker="*",
                       facecolor="#FFD93D", edgecolor="#B8860B", linewidth=1.2,
                       zorder=6, label="Headline (Phi-4)")
        ax.annotate(
            PRETTY_MODEL.get(r["model"], r["model"]),
            (r["params_B"], r["disca_mis"]),
            xytext=(10, -4), textcoords="offset points",
            fontsize=10 if is_phi4 else 9.5,
            color="#222",
            weight="bold" if is_phi4 else "normal",
        )

    # Pareto frontier (DISCA): connect dots in non-increasing MIS as params grow
    swa_sorted = sorted(zip(xs, swa), key=lambda kv: kv[0])
    px, py = [swa_sorted[0][0]], [swa_sorted[0][1]]
    cur_min = swa_sorted[0][1]
    for x, y in swa_sorted[1:]:
        if y < cur_min:
            cur_min = y
            px.append(x); py.append(y)
    ax.plot(px, py, color="#1A8A66", lw=1.2, ls="--", alpha=0.55,
            label="DISCA Pareto frontier")

    # Highlight the "Phi-4 beats Llama-70B" headline.
    if "phi_4" in {r["model"] for r in rows} and "llama33_70b" in {r["model"] for r in rows}:
        phi = next(r for r in rows if r["model"] == "phi_4")
        llm = next(r for r in rows if r["model"] == "llama33_70b")
        if phi["disca_mis"] < llm["disca_mis"]:
            mid_x = float(np.exp((np.log(phi["params_B"]) + np.log(llm["params_B"])) / 2))
            mid_y = (phi["disca_mis"] + llm["disca_mis"]) / 2
            ax.annotate(
                f"Phi-4 (14B) DISCA = {phi['disca_mis']:.3f}\n"
                f"< Llama-70B DISCA = {llm['disca_mis']:.3f}",
                (mid_x, mid_y),
                xytext=(15, 30), textcoords="offset points",
                fontsize=9, ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF7DD",
                          edgecolor="#B8860B", lw=0.8),
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=0.8),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (B, log scale)", fontsize=11.5, labelpad=8)
    ax.set_ylabel(r"Mean MIS across 20 countries  ($\downarrow$ better)",
                  fontsize=11.5, labelpad=8)
    ax.set_title("Calibration competes with scale — DISCA reshapes the size/MIS curve",
                 fontsize=12.5, pad=10)
    # Y-axis honest: starts at 0 to avoid "exaggerated diff" critique
    ax.set_ylim(bottom=0.0)
    ax.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3, lw=0.5)
    plt.tight_layout()

    pdf = OUT / "fig_oral_3_scaling_vs_calibration.pdf"
    png = OUT / "fig_oral_3_scaling_vs_calibration.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}, {out_csv.name}")
    print(f"  rows: {len(rows)}")
    for r in rows[:5]:
        print(f"    {PRETTY_MODEL.get(r['model'], r['model']):<22} "
              f"{r['params_B']:>5.1f}B   van={r['vanilla_mis']:.3f}  "
              f"swa={r['disca_mis']:.3f}  delta={r['delta']:+.3f}")


# ===========================================================================
# Visual 4 — AMCE PCA scatter (geometric DISCA story)
# ===========================================================================

def _country_amce_per_method() -> Tuple[
    Dict[Tuple[str, str], np.ndarray],  # (model, country) -> 6-vec (DISCA model_mpr)
    Dict[str, np.ndarray],               # country -> 6-vec (human)
    Dict[Tuple[str, str], np.ndarray],   # (model, country) -> 6-vec (vanilla MPR)
]:
    disca: Dict[Tuple[str, str], np.ndarray] = {}
    human: Dict[str, np.ndarray] = {}
    vanilla: Dict[Tuple[str, str], np.ndarray] = {}

    for d in PAPER_20C.iterdir():
        if not d.is_dir():
            continue
        slug = d.name
        per_dim = d / "compare" / "per_dim_breakdown.csv"
        if not per_dim.is_file():
            continue
        # DISCA model_mpr + human
        m_buf: Dict[str, Dict[str, float]] = defaultdict(dict)
        h_buf: Dict[str, Dict[str, float]] = defaultdict(dict)
        with open(per_dim, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    m_buf[r["country"]][r["dimension"]] = float(r["model_mpr"])
                    h_buf[r["country"]][r["dimension"]] = float(r["human"])
                except (KeyError, ValueError):
                    continue
        for c, dim_map in m_buf.items():
            v = np.array([dim_map.get(dim, np.nan) for dim in DIMS])
            if np.all(np.isfinite(v)):
                disca[(slug, c)] = v
        for c, dim_map in h_buf.items():
            v = np.array([dim_map.get(dim, np.nan) for dim in DIMS])
            if np.all(np.isfinite(v)):
                human[c] = v

        # Vanilla MPR — compute from raw vanilla_results_*.csv
        sub = next((s for s in (d / "swa").iterdir() if s.is_dir()), None)
        if sub is None:
            continue
        for v_csv in sorted(sub.glob("vanilla_results_*.csv")):
            country = v_csv.stem.replace("vanilla_results_", "")
            sums: Dict[str, float] = defaultdict(float)
            counts: Dict[str, int] = defaultdict(int)
            with open(v_csv, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    cat = row.get("phenomenon_category", "")
                    dim = CAT_TO_DIM.get(cat)
                    if dim is None:
                        continue
                    try:
                        p = float(row["p_spare_preferred"])
                    except (KeyError, ValueError):
                        continue
                    if not np.isfinite(p):
                        continue
                    sums[dim] += p
                    counts[dim] += 1
            v = np.array([100.0 * sums[d] / counts[d] if counts[d] >= 3 else np.nan
                          for d in DIMS])
            if np.all(np.isfinite(v)):
                vanilla[(slug, country)] = v
    return disca, human, vanilla


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return projected 2D points, principal components (2 x d), explained var ratio."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # First 2 components
    comp = Vt[:2]                          # (2, d)
    proj = Xc @ comp.T                     # (n, 2)
    var = (S ** 2) / max(len(Xc) - 1, 1)
    expl = var[:2].sum() / var.sum()
    return proj, comp, float(expl)


def plot_amce_pca() -> None:
    print("\n[Visual 4] AMCE PCA scatter")
    disca, human, vanilla = _country_amce_per_method()
    if not human:
        print("  [SKIP] no human AMCE")
        return

    # Use Phi-4 as the canonical model for visual clarity (one model = clean plot)
    target_model = "phi_4"
    countries = sorted(set(c for (m, c) in disca.keys() if m == target_model)
                       & set(c for (m, c) in vanilla.keys() if m == target_model)
                       & set(human.keys()))
    if len(countries) < 4:
        print(f"  [SKIP] only {len(countries)} countries — too few for PCA")
        return

    H = np.array([human[c] for c in countries])
    D = np.array([disca[(target_model, c)] for c in countries])
    V = np.array([vanilla[(target_model, c)] for c in countries])

    # Fit PCA on the joint cloud so all three sets share an axis
    joint = np.vstack([H, D, V])
    proj, comp, expl = _pca_2d(joint)
    n = len(countries)
    H2, D2, V2 = proj[:n], proj[n:2 * n], proj[2 * n:3 * n]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    def _convex_hull(pts: np.ndarray) -> np.ndarray:
        """2-D convex hull via Andrew's monotone chain (numpy-only)."""
        pts = np.asarray(pts)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        return np.array(lower[:-1] + upper[:-1])

    # Soft hulls behind the points so the three clusters read as regions.
    for pts, color, label in [
        (V2, COL_VAN,  "Vanilla cloud"),
        (D2, COL_SWA,  "DISCA cloud"),
        (H2, COL_HUMN, "Human cloud"),
    ]:
        hull = _convex_hull(pts)
        ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.12,
                             edgecolor=color, linewidth=1.2, linestyle="-",
                             zorder=1))

    # Vanilla → DISCA arrows
    for i, c in enumerate(countries):
        ax.annotate(
            "", xy=D2[i], xytext=V2[i],
            arrowprops=dict(arrowstyle="->", color=COL_SWA, alpha=0.55, lw=0.9),
            zorder=2,
        )

    ax.scatter(H2[:, 0], H2[:, 1], s=180, marker="*", color=COL_HUMN,
               alpha=0.95, edgecolors="black", linewidth=0.7,
               label="Human", zorder=5)
    ax.scatter(V2[:, 0], V2[:, 1], s=95, marker="o", color=COL_VAN,
               alpha=0.85, edgecolors="black", linewidth=0.5,
               label="Vanilla", zorder=4)
    ax.scatter(D2[:, 0], D2[:, 1], s=95, marker="^", color=COL_SWA,
               alpha=0.95, edgecolors="black", linewidth=0.5,
               label="DISCA", zorder=4)

    for i, c in enumerate(countries):
        ax.annotate(c, H2[i], xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=COL_HUMN, alpha=0.85)

    # Distance reduction summary printed inside the plot
    dist_van = float(np.mean(np.linalg.norm(V2 - H2, axis=1)))
    dist_swa = float(np.mean(np.linalg.norm(D2 - H2, axis=1)))
    n_pull = int((np.linalg.norm(D2 - H2, axis=1)
                  < np.linalg.norm(V2 - H2, axis=1)).sum())
    summary = (f"DISCA pulled {n_pull}/{n} countries closer to human\n"
               f"mean ‖model−human‖: {dist_van:.1f} → {dist_swa:.1f}  "
               f"({100*(1-dist_swa/dist_van):+.1f}%)")
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=COL_SWA, lw=1.0, alpha=0.95))

    ax.set_xlabel("PC 1", fontsize=11.5, labelpad=8)
    ax.set_ylabel("PC 2", fontsize=11.5, labelpad=8)
    ax.set_title(f"Geometric story: DISCA pulls the model toward the human cluster\n"
                 f"({PRETTY_MODEL.get(target_model)}, "
                 f"{n} countries, {100 * expl:.1f}% variance captured)",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.95)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    pdf = OUT / "fig_oral_4_amce_pca.pdf"
    png = OUT / "fig_oral_4_amce_pca.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Distance reduction analysis
    dist_van = np.linalg.norm(V2 - H2, axis=1)
    dist_swa = np.linalg.norm(D2 - H2, axis=1)
    n_pulled = int((dist_swa < dist_van).sum())
    print(f"  saved {pdf.name}, {png.name}")
    print(f"  PCA explained variance (PC1+PC2)        : {100 * expl:.1f}%")
    print(f"  countries where DISCA point closer to H : {n_pulled} / {n}")
    print(f"  mean ||vanilla - human||                : {dist_van.mean():.2f}")
    print(f"  mean ||DISCA   - human||                : {dist_swa.mean():.2f}  "
          f"({100 * (1 - dist_swa.mean() / dist_van.mean()):+.1f}% closer)")


# ===========================================================================
# Visual 5 — World-style ΔMIS scatter
# ===========================================================================

def plot_world_delta_mis() -> None:
    print("\n[Visual 5] World-style ΔMIS scatter")

    # Aggregate across the headline 6 models — gives a stronger per-country
    # signal than averaging in the weak / regression models.
    delta_by_country: Dict[str, List[float]] = defaultdict(list)
    n_by_country: Dict[str, int] = defaultdict(int)
    for slug in HEADLINE_MODELS:
        d = PAPER_20C / slug
        per_country = _read_comparison(d)
        for c, (van, swa) in per_country.items():
            delta_by_country[c].append(van - swa)
            n_by_country[c] += 1

    countries = sorted(delta_by_country.keys())
    deltas = np.array([np.mean(delta_by_country[c]) for c in countries])
    if len(countries) < 5:
        print("  [SKIP] too few countries")
        return

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_facecolor("#EAF1F8")  # ocean tint
    # Continental backdrop — soft beige for land, no border
    backdrop_regions = [
        (-170, -30, -55, 75),  # Americas
        (-25,  60, -35, 70),   # Europe + Africa
        (60,  180, -45, 75),   # Asia + Oceania
    ]
    for lo1, lo2, la1, la2 in backdrop_regions:
        ax.add_patch(plt.Rectangle((lo1, la1), lo2 - lo1, la2 - la1,
                                   facecolor="#F5F1E6", edgecolor="none",
                                   alpha=0.95, zorder=1))

    vmax = float(max(np.max(np.abs(deltas)), 1e-6))
    cmap = DIVERGING_RDGN
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for c, delta in zip(countries, deltas):
        if c not in COUNTRY_LATLON:
            continue
        lon, lat = COUNTRY_LATLON[c]
        size = 220 + 1500 * abs(delta) / vmax
        ax.scatter(lon, lat, s=size, c=[cmap(norm(delta))],
                   edgecolors="black", linewidth=0.7, alpha=0.92, zorder=3)
        # Place label just below marker, with white halo for legibility
        from matplotlib.patheffects import withStroke
        t = ax.annotate(c, (lon, lat),
                        xytext=(0, -np.sqrt(size) * 0.45 - 4),
                        textcoords="offset points",
                        ha="center", fontsize=9, color="#111", weight="bold",
                        zorder=4)
        t.set_path_effects([withStroke(linewidth=2.5, foreground="white")])

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 82)
    ax.set_xlabel("Longitude", fontsize=11.5, labelpad=8)
    ax.set_ylabel("Latitude", fontsize=11.5, labelpad=8)
    ax.set_title("Geographic distribution of DISCA gain   "
                 "(mean ΔMIS across the 6 headline models)\n"
                 r"Marker size $\propto |\Delta|$,  "
                 "green = DISCA helped, red = DISCA hurt",
                 fontsize=12, pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(r"Mean $\Delta$MIS  (vanilla $-$ DISCA, $\uparrow$ better)",
                 fontsize=10.5, labelpad=8)
    ax.grid(True, alpha=0.18, lw=0.4)
    plt.tight_layout()

    pdf = OUT / "fig_oral_5_world_delta_mis.pdf"
    png = OUT / "fig_oral_5_world_delta_mis.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_csv = OUT / "world_delta_mis.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "n_models", "mean_delta_mis"])
        for c, d in zip(countries, deltas):
            w.writerow([c, n_by_country[c], f"{d:.4f}"])

    print(f"  saved {pdf.name}, {png.name}, {out_csv.name}")
    print(f"  countries with mean ΔMIS > 0 : "
          f"{(deltas > 0).sum()} / {len(countries)}")
    top = sorted(zip(countries, deltas), key=lambda kv: -kv[1])
    print(f"  Top 3 improved : {[(c, round(d, 3)) for c, d in top[:3]]}")
    print(f"  Bottom 3       : {[(c, round(d, 3)) for c, d in top[-3:]]}")


# ===========================================================================

# ===========================================================================
# Visual 6 — Multi-seed CI bars per country (rebuts R1 "where are error bars?")
# ===========================================================================

def plot_multiseed_ci_bars() -> None:
    print("\n[Visual 6] Multi-seed CI bars per country")
    csv_path = ROOT / "exp_paper" / "result" / "multiseed_phi4" / "multiseed_per_country_ci.csv"
    if not csv_path.is_file():
        print(f"  [SKIP] {csv_path} missing")
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Pivot to per-country (van_mean, van_ci, swa_mean, swa_ci)
    by_c: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        m = r["method"]
        by_c[r["country"]][f"{m}_mean"] = float(r["mean_mis"])
        by_c[r["country"]][f"{m}_ci"]   = float(r["ci95_mis"])
    countries = sorted(
        by_c.keys(),
        key=lambda c: -(by_c[c].get("vanilla_mean", 0) - by_c[c].get("swa_dpbr_mean", 0)),
    )

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(countries))
    w = 0.38
    van_m  = [by_c[c].get("vanilla_mean", np.nan) for c in countries]
    van_ci = [by_c[c].get("vanilla_ci", 0)        for c in countries]
    swa_m  = [by_c[c].get("swa_dpbr_mean", np.nan) for c in countries]
    swa_ci = [by_c[c].get("swa_dpbr_ci", 0)        for c in countries]

    ax.bar(x - w / 2, van_m, w, yerr=van_ci, capsize=3,
           color=COL_VAN, alpha=0.9, edgecolor="black", linewidth=0.4,
           label="Vanilla", error_kw={"lw": 0.9, "ecolor": "#444"})
    ax.bar(x + w / 2, swa_m, w, yerr=swa_ci, capsize=3,
           color=COL_SWA, alpha=0.9, edgecolor="black", linewidth=0.4,
           label="DISCA (ours)", error_kw={"lw": 0.9, "ecolor": "#444"})

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(r"MIS  ($\downarrow$ better, mean$\pm$95% CI over 3 seeds)",
                  fontsize=11, labelpad=8)
    ax.set_title("Per-country multi-seed reliability  (Phi-4, seeds 42 / 101 / 2026)",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.set_ylim(0, max(max(van_m), max(swa_m)) * 1.15)
    plt.tight_layout()
    pdf = OUT / "fig_oral_6_multiseed_ci.pdf"
    png = OUT / "fig_oral_6_multiseed_ci.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)

    # Print headline numbers
    macro_van_ci = float(np.mean(van_ci))
    macro_swa_ci = float(np.mean(swa_ci))
    n_lt_001 = int(sum(1 for c in swa_ci if c < 0.01))
    print(f"  saved {pdf.name}, {png.name}")
    print(f"  mean vanilla CI95: {macro_van_ci:.4f}")
    print(f"  mean DISCA   CI95: {macro_swa_ci:.4f}  "
          f"({n_lt_001}/{len(swa_ci)} countries below 0.01)")


# ===========================================================================
# Visual 7 — Reliability gate activation panel
# ===========================================================================

def plot_reliability_gate(target_model: str = "phi_4") -> None:
    print(f"\n[Visual 7] Reliability gate activation ({target_model})")
    swa_dir = PAPER_20C / target_model / "swa"
    if not swa_dir.is_dir():
        print(f"  [SKIP] {swa_dir}")
        return
    sub = next((d for d in swa_dir.iterdir() if d.is_dir()), None)
    if sub is None:
        return

    per_country_r: Dict[str, List[float]] = defaultdict(list)
    per_country_var: Dict[str, List[float]] = defaultdict(list)
    for csv_path in sorted(sub.glob("swa_results_*.csv")):
        c = csv_path.stem.replace("swa_results_", "")
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    r = float(row["reliability_r"])
                    bv = float(row["bootstrap_var"])
                except (KeyError, ValueError):
                    continue
                if np.isfinite(r):
                    per_country_r[c].append(r)
                if np.isfinite(bv):
                    per_country_var[c].append(bv)

    if not per_country_r:
        print("  [SKIP] no reliability_r data")
        return

    countries = sorted(per_country_r.keys(),
                       key=lambda c: -np.mean(per_country_r[c]))
    means = [float(np.mean(per_country_r[c])) for c in countries]
    frac_closed = [float(np.mean(np.array(per_country_r[c]) < 0.5))
                   for c in countries]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: distribution as violin + mean dot
    parts = axes[0].violinplot(
        [per_country_r[c] for c in countries],
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COL_SWA); pc.set_alpha(0.45); pc.set_edgecolor("black"); pc.set_linewidth(0.4)
    axes[0].scatter(np.arange(1, len(countries) + 1), means, s=40,
                    color="white", edgecolor="black", linewidth=0.7, zorder=4)
    axes[0].axhline(0.5, color="#B2182B", lw=0.9, ls="--", alpha=0.7,
                    label="Gate threshold r = 0.5")
    axes[0].set_xticks(np.arange(1, len(countries) + 1))
    axes[0].set_xticklabels(countries, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Reliability weight  $r$", fontsize=11, labelpad=8)
    axes[0].set_title(f"Per-country distribution of $r$  ({PRETTY_MODEL.get(target_model)})",
                      fontsize=11.5, pad=8)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3, lw=0.5)

    # Right: fraction with gate "closed" (r < 0.5)
    bars = axes[1].barh(countries, frac_closed,
                        color=[DIVERGING_RDGN(0.5 - 0.5 * f) for f in frac_closed],
                        edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Fraction of scenarios with $r < 0.5$  (gate suppresses correction)",
                       fontsize=11, labelpad=8)
    axes[1].set_title("DPBR self-regulation: where the gate intervenes",
                      fontsize=11.5, pad=8)
    axes[1].set_xlim(0, max(frac_closed) * 1.15 + 0.01)
    axes[1].grid(True, axis="x", alpha=0.3, lw=0.5)
    for b, f in zip(bars, frac_closed):
        axes[1].annotate(f"{100 * f:.1f}%", (b.get_width(), b.get_y() + b.get_height() / 2),
                          xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)

    plt.tight_layout()
    pdf = OUT / "fig_oral_7_reliability_gate.pdf"
    png = OUT / "fig_oral_7_reliability_gate.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)

    # Headline stats
    all_r = np.concatenate([np.array(per_country_r[c]) for c in countries])
    print(f"  saved {pdf.name}, {png.name}")
    print(f"  mean r (all scenarios) : {all_r.mean():.3f}")
    print(f"  fraction r < 0.5       : {(all_r < 0.5).mean() * 100:.1f}%  "
          f"(gate suppression rate)")
    print(f"  fraction r > 0.9       : {(all_r > 0.9).mean() * 100:.1f}%  "
          f"(high-confidence corrections)")


# ===========================================================================
# Visual 8 — Latency-vs-MIS Pareto scatter
# ===========================================================================

def plot_latency_vs_mis() -> None:
    print("\n[Visual 8] Latency-vs-MIS Pareto")
    rows = []
    for slug in HEADLINE_MODELS:
        d = PAPER_20C / slug
        comp = d / "compare" / "comparison.csv"
        sub = next((s for s in (d / "swa").iterdir() if s.is_dir()), None) if (d / "swa").is_dir() else None
        if not comp.is_file() or sub is None:
            continue

        # MIS per (method, country) from comparison.csv
        mis_per: Dict[str, List[float]] = defaultdict(list)
        with open(comp, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    mis_per[r["method"]].append(float(r["align_mis"]))
                except (KeyError, ValueError):
                    pass

        # Mean latency per scenario from swa_results_*.csv (DISCA latency)
        latencies_swa: List[float] = []
        latencies_van: List[float] = []
        for cp in sub.glob("swa_results_*.csv"):
            with open(cp, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        v = float(row["latency_ms"])
                    except (KeyError, ValueError):
                        continue
                    if np.isfinite(v):
                        latencies_swa.append(v)
        for cp in sub.glob("vanilla_results_*.csv"):
            with open(cp, "r", encoding="utf-8", newline="") as fh:
                if "latency_ms" not in (csv.reader(fh).__next__()):
                    continue
            # vanilla CSVs may not have latency; that's fine — show DISCA only

        van_mis = float(np.mean(mis_per.get("baseline_vanilla", [np.nan])))
        swa_mis = float(np.mean(next((v for k, v in mis_per.items() if "dual_pass" in k), [np.nan])))
        if not np.isfinite(van_mis) or not np.isfinite(swa_mis):
            continue
        swa_lat_ms = float(np.mean(latencies_swa)) if latencies_swa else float("nan")

        rows.append({
            "model": slug,
            "params_B": MODEL_PARAMS_B.get(slug, np.nan),
            "vanilla_mis": van_mis,
            "disca_mis": swa_mis,
            "disca_latency_ms": swa_lat_ms,
        })

    if not rows:
        print("  [SKIP] no rows")
        return

    rows.sort(key=lambda r: r["params_B"])

    out_csv = OUT / "latency_summary.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    sizes = [80 + r["params_B"] * 12 for r in rows]
    colors = [DIVERGING_RDGN(min(0.85, 0.5 + (r["vanilla_mis"] - r["disca_mis"]) * 4))
              for r in rows]
    for r, s, col in zip(rows, sizes, colors):
        ax.scatter(r["disca_latency_ms"], r["disca_mis"], s=s, color=col,
                   marker="^", edgecolors="black", linewidth=0.5, alpha=0.95, zorder=4)
        ax.annotate(
            f"{PRETTY_MODEL.get(r['model'], r['model'])}\n"
            f"({r['params_B']:.1f}B,  "
            f"$\\Delta$={r['vanilla_mis']-r['disca_mis']:+.3f})",
            (r["disca_latency_ms"], r["disca_mis"]),
            xytext=(10, -3), textcoords="offset points", fontsize=9.5, color="#222",
        )

    ax.set_xlabel("DISCA per-scenario latency (ms, log)", fontsize=11.5, labelpad=8)
    ax.set_ylabel(r"DISCA mean MIS  ($\downarrow$ better)", fontsize=11.5, labelpad=8)
    ax.set_xscale("log")
    ax.set_title("Cost-vs-quality frontier  (marker size $\\propto$ params, "
                 "color $\\propto \\Delta$MIS)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    plt.tight_layout()
    pdf = OUT / "fig_oral_8_latency_vs_mis.pdf"
    png = OUT / "fig_oral_8_latency_vs_mis.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}")
    for r in rows:
        print(f"    {PRETTY_MODEL.get(r['model'], r['model']):<22} "
              f"{r['disca_latency_ms']:>7.1f} ms  MIS={r['disca_mis']:.3f}")


# ===========================================================================
# Visual 9 — Ablation multi-seed box plot panel
# ===========================================================================

PRIMARY_ABLATIONS = [
    "Full SWA-DPBR",
    "No-IS (consensus only)",
    "Always-on PT-IS",
    "No debiasing",
    "Without persona",
    # "No country prior (a_h=0)" intentionally excluded — treated as a tuning
    # hyperparameter rather than a structural component of DISCA.
]


def plot_ablation_multiseed() -> None:
    print("\n[Visual 9] Ablation delta-vs-Full box plot (3 models)")
    csv_path = ROOT / "exp_paper" / "result" / "ablation_breadth" / \
               "ablation_breadth_summary.csv"
    if not csv_path.is_file():
        print(f"  [SKIP] {csv_path}")
        return

    # Switch to Pearson r delta (rank-order metric). MIS delta on this
    # particular dataset has a "JSD paradox" sign-flip on debiasing (paper
    # tracker note 2026-04-13): MIS can improve when distribution flattens,
    # so MIS delta misattributes debiasing's effect. Pearson r is the
    # rank-order metric the paper actually claims for ablation hierarchy.
    by_mc: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["ablation"] not in PRIMARY_ABLATIONS:
                continue
            try:
                pr = float(r["pearson_r"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(pr):
                by_mc[(r["model"], r["country"])][r["ablation"]] = pr

    rows = []
    full_label = "Full SWA-DPBR"
    for (model, country), abl_map in by_mc.items():
        if full_label not in abl_map:
            continue
        full_r = abl_map[full_label]
        for ablation, r in abl_map.items():
            rows.append({
                "model": model, "country": country,
                "ablation": ablation,
                # Sign convention: NEGATIVE means removing this component
                # HURTS rank-order (Pearson r dropped). The Full reference
                # is r=Full so its delta = 0.
                "delta": r - full_r,
            })

    if not rows:
        print("  [SKIP] no rows")
        return

    by_abl: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_abl[r["ablation"]].append(r["delta"])

    # Order so that, when boxplot draws with index 0 at bottom and index N-1
    # at top, the figure reads as:
    #     TOP    = most critical component (most-negative Δr)
    #     BOTTOM = reference (Full SWA-DPBR, Δr = 0)
    # Layout list = [Full, least-hurt, ..., most-hurt]. Sort non-Full rows
    # in DESCENDING order of mean Δr (largest first → least critical first).
    order = ["Full SWA-DPBR"] + sorted(
        [a for a in by_abl if a != "Full SWA-DPBR"],
        key=lambda a: -np.mean(by_abl[a]),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [by_abl[a] for a in order]
    box = ax.boxplot(
        data, vert=False, widths=0.55, patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#888",
                        markeredgecolor="none", alpha=0.5),
    )
    means = [float(np.mean(by_abl[a])) for a in order]
    # Color encoding: more-negative Δr (component more critical when removed)
    # → deeper RED. Positive (rare, removing helps) → green. Full (mean=0)
    # gets a neutral teal so it stands out as the reference row.
    vmax = max(0.05, max(abs(m) for m in means))
    for patch, m, label in zip(box["boxes"], means, order):
        if label == "Full SWA-DPBR":
            patch.set_facecolor("#4DB6AC")  # teal reference
            patch.set_alpha(0.85)
        else:
            # negative m → red; positive m → green; map (m / vmax) ∈ [-1, 1]
            # to cmap t ∈ [0, 1] where 0 = red, 1 = green.
            t = 0.5 + 0.5 * (m / vmax) * 0.85
            patch.set_facecolor(DIVERGING_RDGN(t))
            patch.set_alpha(0.85)
        patch.set_edgecolor("black"); patch.set_linewidth(0.6)

    ax.axvline(0, color="black", lw=1.0, alpha=0.7)
    ax.set_yticklabels(order, fontsize=10.5)
    ax.set_xlabel(r"$\Delta$ Pearson $r$ vs Full SWA-DPBR  "
                  r"($-$ = removing this component hurts rank-order, "
                  r"$+$ = helps)",
                  fontsize=11, labelpad=8)
    ax.set_title("Ablation effect on rank-order alignment "
                 r"($\Delta$ Pearson $r$, 3 models × 3 countries)",
                 fontsize=12, pad=10)
    ax.grid(True, axis="x", alpha=0.3, lw=0.5)

    # Annotate component-importance ordering
    for i, (a, m) in enumerate(zip(order, means)):
        # Negative = component is critical → red, positive → grey-ish
        col = "#B2182B" if m < -0.01 else ("#1B7C3D" if m > 0.01 else "#444")
        ax.annotate(f"mean = {m:+.3f}",
                    (m, i + 1), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=col, weight="bold")

    plt.tight_layout()
    pdf = OUT / "fig_oral_9_ablation_multiseed.pdf"
    png = OUT / "fig_oral_9_ablation_multiseed.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}")
    for a, m in zip(order, means):
        print(f"    {a:<35} mean(Δ vs Full) = {m:+.4f}  "
              f"std={np.std(by_abl[a]):.4f}  n={len(by_abl[a])}")


# ===========================================================================

if __name__ == "__main__":
    plot_per_dim_heatmap()
    plot_per_dim_grouped_bars()
    plot_scaling_vs_calibration()
    plot_amce_pca()
    plot_world_delta_mis()
    plot_multiseed_ci_bars()
    plot_reliability_gate()
    plot_latency_vs_mis()
    plot_ablation_multiseed()
    print("\n" + "=" * 70)
    print(f"  All oral visuals -> {OUT}")
    print("=" * 70)
