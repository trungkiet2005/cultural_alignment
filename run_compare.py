#!/usr/bin/env python3
"""
Compare Vanilla LLM Baseline vs SWA-MPPI results.

Loads the pickled per-country summaries produced by run_baseline.py and
run_swa_mppi.py, then writes side-by-side metric tables and comparison
figures (per-country alignment metrics, AMCE deltas, radar overlays).

Usage:
    python run_compare.py \
        --baseline-dir results/baseline_gemma \
        --swa-dir      results/swa_gemma \
        --output-dir   results/compare_gemma
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import resolve_output_dir
from src.viz.style import setup_matplotlib, BASELINE_COLOR, SWA_COLOR, HUMAN_COLOR
from src.viz.radar import plot_radar_single
from src.viz.comparison import plot_baseline_comparison, plot_comparison_table
from src.viz.bar_charts import plot_amce_comparison_bar


# Metrics expected inside summary["alignment"]
ALIGN_METRICS = ["jsd", "cosine_sim", "pearson_r", "spearman_rho", "mae", "rmse"]
# Lower is better for these:
LOWER_BETTER = {"jsd", "mae", "rmse"}

AMCE_KEYS = [
    "Species_Humans", "Gender_Female", "Age_Young",
    "Fitness_Fit", "SocialValue_High", "Utilitarianism_More",
]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_summaries(baseline_dir: Path, swa_dir: Path):
    bl_path = baseline_dir / "baseline_summaries.pkl"
    sw_path = swa_dir / "all_summaries.pkl"
    if not bl_path.exists():
        raise FileNotFoundError(f"Missing baseline pickle: {bl_path}")
    if not sw_path.exists():
        raise FileNotFoundError(f"Missing SWA pickle: {sw_path}")
    bl = {s["country"]: s for s in _load_pickle(bl_path)}
    sw = {s["country"]: s for s in _load_pickle(sw_path)}
    common = sorted(set(bl) & set(sw))
    if not common:
        raise RuntimeError("No overlapping countries between baseline and SWA runs.")
    missing = sorted((set(bl) | set(sw)) - set(common))
    if missing:
        print(f"[WARN] Skipping countries not in both runs: {missing}")
    return bl, sw, common


def build_metric_table(bl, sw, countries):
    rows = []
    for c in countries:
        b_align = bl[c].get("alignment", {})
        s_align = sw[c].get("alignment", {})
        row = {"country": c}
        for m in ALIGN_METRICS:
            bv = b_align.get(m, np.nan)
            sv = s_align.get(m, np.nan)
            row[f"baseline_{m}"] = bv
            row[f"swa_{m}"] = sv
            row[f"delta_{m}"] = sv - bv
            if m in LOWER_BETTER:
                row[f"swa_better_{m}"] = sv < bv
            else:
                row[f"swa_better_{m}"] = sv > bv
        rows.append(row)
    return pd.DataFrame(rows)


def build_amce_table(bl, sw, countries):
    rows = []
    for c in countries:
        h = bl[c]["human_amce"]
        bm = bl[c]["model_amce"]
        sm = sw[c]["model_amce"]
        for k in AMCE_KEYS:
            hv, bv, sv = h.get(k, np.nan), bm.get(k, np.nan), sm.get(k, np.nan)
            rows.append({
                "country": c, "category": k,
                "human": hv, "baseline": bv, "swa": sv,
                "baseline_err": bv - hv, "swa_err": sv - hv,
                "abs_baseline_err": abs(bv - hv), "abs_swa_err": abs(sv - hv),
            })
    return pd.DataFrame(rows)


def aggregate_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ALIGN_METRICS:
        b = metric_df[f"baseline_{m}"]
        s = metric_df[f"swa_{m}"]
        wins = int(metric_df[f"swa_better_{m}"].sum())
        rows.append({
            "metric": m,
            "baseline_mean": b.mean(), "baseline_std": b.std(),
            "swa_mean": s.mean(),      "swa_std": s.std(),
            "delta_mean": (s - b).mean(),
            "swa_wins": f"{wins}/{len(metric_df)}",
            "lower_is_better": m in LOWER_BETTER,
        })
    return pd.DataFrame(rows)


# ---------- Plots ----------

def plot_metric_bars(metric_df: pd.DataFrame, out_dir: Path):
    countries = metric_df["country"].tolist()
    x = np.arange(len(countries))
    w = 0.38
    for m in ALIGN_METRICS:
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(countries) + 2), 4))
        ax.bar(x - w / 2, metric_df[f"baseline_{m}"], w,
               label="Baseline", color=BASELINE_COLOR)
        ax.bar(x + w / 2, metric_df[f"swa_{m}"], w,
               label="SWA-MPPI", color=SWA_COLOR)
        arrow = " (lower better)" if m in LOWER_BETTER else " (higher better)"
        ax.set_title(f"{m}{arrow}")
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_{m}.png", dpi=150)
        plt.show(); plt.close(fig)


def plot_delta_heatmap(metric_df: pd.DataFrame, out_dir: Path):
    cols = [f"delta_{m}" for m in ALIGN_METRICS]
    M = metric_df[cols].to_numpy()
    # Flip sign on lower-better metrics so positive == SWA improves.
    for i, m in enumerate(ALIGN_METRICS):
        if m in LOWER_BETTER:
            M[:, i] = -M[:, i]
    fig, ax = plt.subplots(figsize=(1.1 * len(ALIGN_METRICS) + 2,
                                    0.4 * len(metric_df) + 2))
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ALIGN_METRICS)))
    ax.set_xticklabels(ALIGN_METRICS, rotation=30, ha="right")
    ax.set_yticks(range(len(metric_df)))
    ax.set_yticklabels(metric_df["country"])
    ax.set_title("SWA improvement over Baseline\n(positive = SWA better)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.5 * vmax else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_delta_heatmap.png", dpi=150)
    plt.close(fig)


def plot_amce_error_bars(amce_df: pd.DataFrame, out_dir: Path):
    g = amce_df.groupby("category")[["abs_baseline_err", "abs_swa_err"]].mean()
    g = g.reindex(AMCE_KEYS)
    x = np.arange(len(g))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, g["abs_baseline_err"], w,
           label="Baseline", color=BASELINE_COLOR)
    ax.bar(x + w / 2, g["abs_swa_err"], w,
           label="SWA-MPPI", color=SWA_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in g.index], fontsize=8)
    ax.set_ylabel("Mean |Model AMCE - Human AMCE|")
    ax.set_title("Per-category AMCE error (averaged over countries)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_amce_abs_error.png", dpi=150)
    plt.close(fig)


def plot_radar_overlays(bl, sw, countries, out_dir: Path):
    """Per-country radar styled like main.py, with Baseline+SWA+Human."""
    radar_dir = out_dir / "radar_overlay"
    radar_dir.mkdir(exist_ok=True)
    criteria_labels = {
        "Species_Humans": "Sparing\nHumans",
        "Age_Young": "Sparing\nYoung",
        "Fitness_Fit": "Sparing\nFit",
        "Gender_Female": "Sparing\nFemales",
        "SocialValue_High": "Sparing\nHigher Status",
        "Utilitarianism_More": "Sparing\nMore",
    }

    for c in countries:
        human = bl[c]["human_amce"]
        b_amce = bl[c]["model_amce"]
        s_amce = sw[c]["model_amce"]
        # Match main.py ordering behavior (alphabetical common keys).
        common_keys = sorted(set(AMCE_KEYS) & set(human.keys()) & set(b_amce.keys()) & set(s_amce.keys()))
        if len(common_keys) < 3:
            print(f"[WARN] Not enough common criteria for radar plot ({c})")
            continue

        labels = [criteria_labels.get(k, k.replace("_", "\n")) for k in common_keys]
        n = len(common_keys)
        angles = [i / float(n) * 2 * np.pi for i in range(n)] + [0]

        human_vals = [human[k] for k in common_keys]
        baseline_vals = [b_amce[k] for k in common_keys]
        swa_vals = [s_amce[k] for k in common_keys]

        human_plot = human_vals + [human_vals[0]]
        baseline_plot = baseline_vals + [baseline_vals[0]]
        swa_plot = swa_vals + [swa_vals[0]]

        align = sw[c].get("alignment", {})
        jsd_str = f"JSD={align.get('jsd', 0):.3f}" if "jsd" in align else ""
        r_str = f"r={align.get('pearson_r', 0):.3f}" if "pearson_r" in align else ""

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color="#333333")
        ax.set_rlabel_position(30)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="#666666", size=8)
        ax.set_ylim(0, 100)

        # Keep plotting style from main.py; add baseline as extra dashed line.
        ax.plot(angles, swa_plot, "o-", linewidth=2.2, color=SWA_COLOR, label="SWA-MPPI v3", markersize=5)
        ax.fill(angles, swa_plot, alpha=0.15, color=SWA_COLOR)
        ax.plot(angles, baseline_plot, "d--", linewidth=2.0, color=BASELINE_COLOR, label="Vanilla LLM", markersize=4.5)
        ax.plot(angles, human_plot, "s--", linewidth=2.0, color=HUMAN_COLOR, label=f"Human ({c})", markersize=5)
        ax.fill(angles, human_plot, alpha=0.08, color=HUMAN_COLOR)
        ax.plot(np.linspace(0, 2 * np.pi, 100), [50] * 100, ":", color="#999999", linewidth=0.8, alpha=0.6)

        ax.set_title(f"{c}\n{jsd_str}  {r_str}" if jsd_str else c, size=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9, framealpha=0.9, edgecolor="#cccccc")
        fig.tight_layout()
        fig.savefig(radar_dir / f"radar_overlay_{c}.png", dpi=150, bbox_inches="tight")
        fig.savefig(radar_dir / f"radar_overlay_{c}.pdf", dpi=150, bbox_inches="tight")
        plt.show(); plt.close(fig)


def print_console_summary(agg: pd.DataFrame):
    print("\n" + "=" * 72)
    print("  AGGREGATE COMPARISON  (mean over countries)")
    print("=" * 72)
    print(f"  {'metric':14s} {'baseline':>14s} {'swa':>14s} {'delta':>10s} {'swa_wins':>10s}")
    for _, r in agg.iterrows():
        print(f"  {r['metric']:14s} "
              f"{r['baseline_mean']:8.4f}±{r['baseline_std']:.3f} "
              f"{r['swa_mean']:8.4f}±{r['swa_std']:.3f} "
              f"{r['delta_mean']:+10.4f} {r['swa_wins']:>10s}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", type=Path,
                    help="Baseline output dir. If --model-name is given and this "
                         "is omitted, defaults to results/baseline/<model>.")
    ap.add_argument("--swa-dir", type=Path,
                    help="SWA-MPPI output dir. If --model-name is given and this "
                         "is omitted, defaults to results/swa_mppi/<model>.")
    ap.add_argument("--output-dir", type=Path,
                    help="Where to write comparison artefacts. Defaults to "
                         "results/compare/<model> when --model-name is given.")
    ap.add_argument("--model-name", type=str, default=None,
                    help="HF model id, used to auto-resolve dirs the same way "
                         "run_baseline.py / run_swa_mppi.py do.")
    ap.add_argument("--baseline-root", type=str, default="results/baseline")
    ap.add_argument("--swa-root", type=str, default="results/swa_mppi")
    ap.add_argument("--compare-root", type=str, default="results/compare")
    args = ap.parse_args()

    # Auto-resolve dirs from --model-name (mirrors resolve_output_dir behaviour).
    if args.model_name:
        if args.baseline_dir is None:
            args.baseline_dir = Path(resolve_output_dir(args.baseline_root, args.model_name))
        if args.swa_dir is None:
            args.swa_dir = Path(resolve_output_dir(args.swa_root, args.model_name))
        if args.output_dir is None:
            args.output_dir = Path(resolve_output_dir(args.compare_root, args.model_name))

    missing = [n for n, v in [("--baseline-dir", args.baseline_dir),
                              ("--swa-dir", args.swa_dir),
                              ("--output-dir", args.output_dir)] if v is None]
    if missing:
        ap.error(f"Missing {missing}. Pass them, or use --model-name to auto-resolve.")

    print(f"[COMPARE] baseline-dir = {args.baseline_dir}")
    print(f"[COMPARE] swa-dir      = {args.swa_dir}")
    print(f"[COMPARE] output-dir   = {args.output_dir}")

    setup_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bl, sw, countries = load_summaries(args.baseline_dir, args.swa_dir)
    print(f"[COMPARE] {len(countries)} countries: {countries}")

    metric_df = build_metric_table(bl, sw, countries)
    amce_df = build_amce_table(bl, sw, countries)
    agg = aggregate_summary(metric_df)

    metric_df.to_csv(args.output_dir / "compare_alignment_metrics.csv", index=False)
    amce_df.to_csv(args.output_dir / "compare_amce_per_category.csv", index=False)
    agg.to_csv(args.output_dir / "compare_aggregate.csv", index=False)
    print(f"[SAVE] CSVs -> {args.output_dir}")

    plot_metric_bars(metric_df, args.output_dir)
    plot_delta_heatmap(metric_df, args.output_dir)
    plot_amce_error_bars(amce_df, args.output_dir)
    plot_radar_overlays(bl, sw, countries, args.output_dir)

    # main.py-style comparison figures (Fig 8/9/10):
    # keep summary schema compatible with main.py plotting functions.
    swa_summaries_ordered = []
    for c in countries:
        s = dict(sw[c])
        s["baseline_alignment"] = bl[c]["alignment"]
        s["baseline_amce"] = bl[c]["model_amce"]
        swa_summaries_ordered.append(s)

    vanilla_metrics = {c: bl[c]["alignment"] for c in countries}
    out_dir_str = str(args.output_dir)
    print("\n[PLOT] Fig 8: Baseline vs SWA-MPPI bar comparison...")
    plot_baseline_comparison(swa_summaries_ordered, vanilla_metrics, out_dir_str)
    print("\n[PLOT] Fig 9: AMCE per-criterion bar chart...")
    plot_amce_comparison_bar(
        swa_summaries_ordered,
        out_dir_str,
        model_label="SWA-MPPI v3",
        model_color=SWA_COLOR,
    )
    print("\n[PLOT] Comparison table (publication-quality + LaTeX)...")
    plot_comparison_table(swa_summaries_ordered, vanilla_metrics, out_dir_str)

    print(f"[SAVE] Figures -> {args.output_dir}")

    print_console_summary(agg)


if __name__ == "__main__":
    main()
