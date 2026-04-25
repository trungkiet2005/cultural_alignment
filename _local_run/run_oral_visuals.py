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
    "hf_qwen25_7b_bf16":     "Qwen2.5-7B (BF16)",
    "llama31_70b":           "Llama-3.1-70B",
    "llama33_70b":           "Llama-3.3-70B",
    "magistral_small_2509":  "Magistral-24B",
    "phi35_mini":            "Phi-3.5-Mini",
    "phi_4":                 "Phi-4 (14B)",
    "qwen25_7b":             "Qwen2.5-7B",
    "qwen3_vl_8b":           "Qwen3-VL-8B",
}

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

    models = sorted({k[0] for k in cells.keys()},
                    key=lambda m: -np.mean([cells[(m, d)] for d in DIMS if (m, d) in cells]))
    M = np.full((len(models), len(DIMS)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(DIMS):
            v = cells.get((m, d))
            if v is not None:
                M[i, j] = v

    finite = M[np.isfinite(M)]
    vmax = float(np.max(np.abs(finite))) if len(finite) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "rdgn", ["#C04E28", "#F2EDE0", "#1A8A66"], N=256,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                col = "white" if abs(M[i, j]) > vmax * 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=8.5, color=col)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMS], rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY_MODEL.get(m, m) for m in models])
    ax.set_xlabel("MultiTP dimension", fontsize=11)
    ax.set_title("DISCA macro improvement (vanilla |err| − DISCA |err|)\n"
                 "Positive = DISCA helps. Sorted by mean Δ across dims.", fontsize=11)
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
    models = sorted({k[0] for k in cells.keys()},
                    key=lambda m: -np.mean([cells.get((m, d), 0.0) for d in DIMS]))

    fig, ax = plt.subplots(figsize=(11, 5.5))
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
    ax.set_title("Per-dimension DISCA gain across 11 models", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
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

    rows.sort(key=lambda r: r["params_B"])

    # CSV summary
    out_csv = OUT / "scaling_calibration_summary.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    xs = [r["params_B"] for r in rows]
    van = [r["vanilla_mis"] for r in rows]
    swa = [r["disca_mis"] for r in rows]

    ax.scatter(xs, van, s=80, color="#C04E28", alpha=0.85,
               label="Vanilla (baseline)", zorder=3, edgecolors="black", linewidth=0.4)
    ax.scatter(xs, swa, s=80, color="#1A8A66", alpha=0.85, marker="^",
               label="DISCA (ours)", zorder=4, edgecolors="black", linewidth=0.4)

    # Connect each model's (vanilla, DISCA) pair with an arrow
    for r in rows:
        ax.annotate(
            "", xy=(r["params_B"], r["disca_mis"]),
            xytext=(r["params_B"], r["vanilla_mis"]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=0.8),
            zorder=2,
        )
        ax.annotate(
            PRETTY_MODEL.get(r["model"], r["model"]),
            (r["params_B"], r["disca_mis"]),
            xytext=(7, -3), textcoords="offset points", fontsize=8.5, color="#333",
        )

    # Pareto frontier (DISCA): connect dots in non-increasing MIS as params grow
    swa_sorted = sorted(zip(xs, swa), key=lambda kv: kv[0])
    px, py = [swa_sorted[0][0]], [swa_sorted[0][1]]
    cur_min = swa_sorted[0][1]
    for x, y in swa_sorted[1:]:
        if y < cur_min:
            cur_min = y
            px.append(x); py.append(y)
    ax.plot(px, py, color="#1A8A66", lw=1, ls="--", alpha=0.5,
            label="DISCA Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (B, log)", fontsize=11)
    ax.set_ylabel("Mean MIS across 20 countries  ↓ better", fontsize=11)
    ax.set_title("Calibration competes with scale — DISCA reshapes the size/MIS curve",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
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

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Vanilla → DISCA arrows
    for i, c in enumerate(countries):
        ax.annotate(
            "", xy=D2[i], xytext=V2[i],
            arrowprops=dict(arrowstyle="->", color="#1A8A66", alpha=0.55, lw=0.8),
            zorder=2,
        )

    ax.scatter(H2[:, 0], H2[:, 1], s=140, marker="*", color="#2D5F9A",
               alpha=0.9, edgecolors="black", linewidth=0.6, label="Human", zorder=5)
    ax.scatter(V2[:, 0], V2[:, 1], s=80, marker="o", color="#C04E28",
               alpha=0.7, edgecolors="black", linewidth=0.4, label="Vanilla model", zorder=4)
    ax.scatter(D2[:, 0], D2[:, 1], s=80, marker="^", color="#1A8A66",
               alpha=0.85, edgecolors="black", linewidth=0.4, label="DISCA model", zorder=4)

    for i, c in enumerate(countries):
        ax.annotate(c, H2[i], xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#2D5F9A")

    ax.set_xlabel(f"PC 1", fontsize=11)
    ax.set_ylabel(f"PC 2", fontsize=11)
    ax.set_title(f"Geometric story: DISCA pulls the model toward the human point\n"
                 f"({PRETTY_MODEL.get(target_model)}, "
                 f"{n} countries, {100 * expl:.1f}% variance in 2 PCs)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, lw=0.5)
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

    # Aggregate across all 11 models for a single robust per-country signal
    delta_by_country: Dict[str, List[float]] = defaultdict(list)
    n_by_country: Dict[str, int] = defaultdict(int)
    for slug in MODEL_PARAMS_B:
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

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Light continental backdrop using regional rectangles (very rough, just for context)
    backdrop_regions = [
        # (lon_min, lon_max, lat_min, lat_max, name)
        (-170, -30, -55, 75, ""),     # Americas
        (-25,  60, -35, 70, ""),      # Europe + Africa
        (60,  180, -45, 75, ""),      # Asia + Oceania
    ]
    for lo1, lo2, la1, la2, _ in backdrop_regions:
        ax.add_patch(plt.Rectangle((lo1, la1), lo2 - lo1, la2 - la1,
                                   facecolor="#F2EDE0", edgecolor="none",
                                   alpha=0.6, zorder=1))

    vmax = float(max(np.max(np.abs(deltas)), 1e-6))
    cmap = LinearSegmentedColormap.from_list(
        "rdgn_world", ["#C04E28", "#F2EDE0", "#1A8A66"], N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for c, delta in zip(countries, deltas):
        if c not in COUNTRY_LATLON:
            continue
        lon, lat = COUNTRY_LATLON[c]
        size = 200 + 1500 * abs(delta) / vmax
        ax.scatter(lon, lat, s=size, c=[cmap(norm(delta))],
                   edgecolors="black", linewidth=0.6, alpha=0.9, zorder=3)
        ax.annotate(c, (lon, lat),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#222")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic distribution of DISCA gain (mean ΔMIS across 11 models)\n"
                 "Marker size ∝ |Δ|, green = improved, orange = hurt",
                 fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, shrink=0.7)
    cb.set_label(r"Mean $\Delta$MIS  (vanilla $-$ DISCA, ↑ better)")
    ax.grid(True, alpha=0.3, lw=0.4)
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

if __name__ == "__main__":
    plot_per_dim_heatmap()
    plot_per_dim_grouped_bars()
    plot_scaling_vs_calibration()
    plot_amce_pca()
    plot_world_delta_mis()
    print("\n" + "=" * 70)
    print(f"  All oral visuals -> {OUT}")
    print("=" * 70)
