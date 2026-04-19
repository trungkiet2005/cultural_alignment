#!/usr/bin/env python3
"""Pre-run triage diagnostic (Reviewer Q9).

Builds an interpretable cutoff on the vanilla decision-margin (cheap to
compute pre-deployment) that predicts whether SWA-DPBR will yield a
material MIS gain on a given (model, country) cell. Operationally:

    if mean_margin(country | model) > tau:
        deploy SWA-DPBR
    else:
        skip the IS stage, vanilla is near-optimal anyway

This script is fully post-hoc -- it joins:
  1. The per-country logit-conditioning CSV(s) from
     ``results/exp24_round2/logit_conditioning/`` and / or
     ``logit_conditioning_cross_model/logit_cond_<model>.csv``.
  2. The per-country SWA-DPBR-vs-vanilla MIS gain from each model's
     ``comparison.csv`` (or the multiseed CSV for Phi-4).

It then reports:
  * Pearson / Spearman correlation between mean_margin and MIS gain.
  * For each cutoff tau in a sweep, the (recall_high_gain, false_skip_rate).
  * The Pareto-optimal cutoff plus the recommended deployment rule.

No GPU required.

    !python exp_paper/round3/posthoc/exp_r3_pre_run_triage.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _r2_bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from exp_paper._r2_common import on_kaggle

# ─── locate input CSVs ──────────────────────────────────────────────────────
R2_BASE = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round2"
    if on_kaggle()
    else "results/exp24_round2"
)
PAPER_BASE = Path(
    "/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    if on_kaggle()
    else "results/exp24_paper_20c"
)

OUT_DIR = (R2_BASE.parent / "exp24_round3" / "pre_run_triage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default cutoff sweep over mean decision margin in [0.0, 0.5].
CUTOFFS = np.arange(0.00, 0.51, 0.025)


def _scan_logit_cond() -> pd.DataFrame:
    """Collect per-(model, country) logit-conditioning rows."""
    paths: List[Path] = []
    for sub in ["logit_conditioning_cross_model", "logit_conditioning"]:
        d = R2_BASE / sub
        if not d.exists():
            continue
        for p in d.glob("logit_cond_*.csv"):
            paths.append(p)
    rows: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "model" not in df.columns:
                # Single-model file — infer model from filename.
                df["model"] = p.stem.replace("logit_cond_", "")
            rows.append(df)
        except Exception as exc:
            print(f"  [warn] could not read {p}: {exc}")
    if not rows:
        # Fallback: try the result folder shipped with the repo.
        for p in (R2_BASE / "logit_conditioning").glob("*.csv"):
            try:
                rows.append(pd.read_csv(p))
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "mean_margin" not in df.columns:
        return pd.DataFrame()
    return df


def _scan_per_country_mis() -> pd.DataFrame:
    """Collect per-(model, country) (vanilla, SWA-DPBR) MIS pairs."""
    rows: List[pd.DataFrame] = []
    # Prefer the released multiseed CSV for Phi-4 (most accurate).
    ms = R2_BASE / "multiseed_phi4" / "multiseed_per_country_ci.csv"
    if ms.exists():
        df = pd.read_csv(ms)
        # Pivot vanilla vs SWA-DPBR
        van = df[df["method"] == "vanilla"][["country", "mean_mis"]].rename(
            columns={"mean_mis": "van_mis"})
        swa = df[df["method"] == "swa_dpbr"][["country", "mean_mis"]].rename(
            columns={"mean_mis": "swa_mis"})
        merged = van.merge(swa, on="country")
        merged["model"] = "Phi-4"
        rows.append(merged)
    # Then any per-model comparison.csv we can find.
    for p in PAPER_BASE.glob("**/comparison.csv"):
        try:
            cmp = pd.read_csv(p)
            van = cmp[cmp["method"] == "baseline_vanilla"][["country", "align_mis"]].rename(
                columns={"align_mis": "van_mis"})
            swa = cmp[cmp["method"].str.contains("dual_pass", na=False)][
                ["country", "align_mis"]].rename(columns={"align_mis": "swa_mis"})
            merged = van.merge(swa, on="country")
            merged["model"] = p.parent.parent.name  # results/exp24_paper_20c/<model>/...
            rows.append(merged)
        except Exception as exc:
            print(f"  [warn] {p}: {exc}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["model", "country"])
    out["mis_gain"]     = out["van_mis"] - out["swa_mis"]
    out["mis_gain_pct"] = 100.0 * out["mis_gain"] / out["van_mis"].replace(0, np.nan)
    return out


def _evaluate_cutoffs(joined: pd.DataFrame) -> pd.DataFrame:
    """For each tau, compute deployment-decision metrics."""
    # Define a "high-gain" cell as one with > 5% relative MIS reduction.
    high_gain = joined["mis_gain_pct"] > 5.0
    rows = []
    for tau in CUTOFFS:
        deploy = joined["mean_margin"] > tau
        n_total = len(joined)
        if n_total == 0:
            continue
        n_deploy = int(deploy.sum())
        n_skip = int((~deploy).sum())
        # True positives: deployed and high-gain.
        tp = int((deploy & high_gain).sum())
        # False negatives: skipped but actually high-gain (regret).
        fn = int((~deploy & high_gain).sum())
        # False positives: deployed but low/negative gain (waste).
        fp = int((deploy & ~high_gain).sum())
        # True negatives: skipped and low-gain (correct skip).
        tn = int((~deploy & ~high_gain).sum())
        recall = tp / max(int(high_gain.sum()), 1)
        precision = tp / max(n_deploy, 1)
        # Compute average compute saved (skipped cells * 3.6x latency premium).
        compute_saved_pct = 100.0 * n_skip / n_total
        # Average MIS regret: gains we missed by skipping high-gain cells.
        regret = float(joined.loc[~deploy & high_gain, "mis_gain"].sum())
        rows.append({
            "tau":               round(float(tau), 3),
            "n_deploy":          n_deploy,
            "n_skip":            n_skip,
            "true_positives":    tp,
            "false_negatives":   fn,
            "false_positives":   fp,
            "true_negatives":    tn,
            "recall_high_gain":  round(recall, 3),
            "precision_deploy":  round(precision, 3),
            "compute_saved_pct": round(compute_saved_pct, 1),
            "mis_regret_total":  round(regret, 4),
        })
    return pd.DataFrame(rows)


def _pick_pareto(df: pd.DataFrame) -> Optional[dict]:
    """Pick the cutoff that maximises compute_saved while keeping recall >= 0.9."""
    if df.empty:
        return None
    feasible = df[df["recall_high_gain"] >= 0.9]
    if feasible.empty:
        # Fall back to the highest-recall row.
        feasible = df.sort_values("recall_high_gain", ascending=False).head(1)
    # Among feasible, pick the largest compute_saved.
    return feasible.sort_values("compute_saved_pct", ascending=False).iloc[0].to_dict()


def _build_latex_table(cutoff_df: pd.DataFrame, recommended: Optional[dict]) -> str:
    if cutoff_df.empty:
        return "% (no data)"
    # Show 7 representative cutoffs spanning the sweep.
    keep = cutoff_df.iloc[::3].copy()  # every 3rd row → ~7 entries
    if recommended is not None:
        keep = pd.concat(
            [keep, cutoff_df[cutoff_df["tau"] == recommended["tau"]]]
        ).drop_duplicates("tau").sort_values("tau")
    lines = [
        r"\begin{tabular}{rrrrrr}\toprule",
        r"$\tau$ (margin) & deploy/skip & recall (gain$>$5\%) & precision & "
        r"compute saved (\%) & MIS regret \\\midrule",
    ]
    for _, row in keep.iterrows():
        marker = r"$^{\star}$" if (recommended and row["tau"] == recommended["tau"]) else ""
        lines.append(
            f"{row['tau']:.3f}{marker} & "
            f"{int(row['n_deploy'])}/{int(row['n_skip'])} & "
            f"{row['recall_high_gain']:.2f} & {row['precision_deploy']:.2f} & "
            f"{row['compute_saved_pct']:.1f} & {row['mis_regret_total']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    print(f"[triage] reading from R2_BASE = {R2_BASE}")
    cond = _scan_logit_cond()
    mis  = _scan_per_country_mis()
    print(f"[triage] logit-conditioning rows: {len(cond)}  "
          f"per-country MIS rows: {len(mis)}")

    if cond.empty or mis.empty:
        msg = [
            "[error] insufficient data for triage analysis.",
            "Required:",
            "  - logit_cond_*.csv from logit_conditioning(_cross_model)/",
            "  - either multiseed_per_country_ci.csv (Phi-4) or",
            "    per-model comparison.csv files.",
        ]
        raise SystemExit("\n".join(msg))

    joined = cond.merge(mis, on=["model", "country"], how="inner")
    print(f"[triage] joined cells: {len(joined)}  "
          f"(unique models: {joined['model'].nunique()})")
    joined.to_csv(OUT_DIR / "triage_joined_cells.csv", index=False)

    cutoffs = _evaluate_cutoffs(joined)
    cutoffs.to_csv(OUT_DIR / "triage_cutoff_sweep.csv", index=False)
    print(f"[saved] {OUT_DIR / 'triage_cutoff_sweep.csv'}")
    print(cutoffs.to_string(index=False))

    recommended = _pick_pareto(cutoffs)
    if recommended is not None:
        print(f"\n[recommend] tau* = {recommended['tau']:.3f}  "
              f"recall={recommended['recall_high_gain']:.2f}  "
              f"compute_saved={recommended['compute_saved_pct']:.1f}%")

    latex = _build_latex_table(cutoffs, recommended)
    (OUT_DIR / "triage_table.tex").write_text(latex, encoding="utf-8")
    print(f"[saved] {OUT_DIR / 'triage_table.tex'}")

    # Compact summary text.
    n_total = len(joined)
    high_gain = int((joined["mis_gain_pct"] > 5.0).sum())
    summary = [
        f"Pre-run triage diagnostic ({n_total} (model, country) cells)",
        "=" * 58,
        f"  high-gain cells (>5% MIS reduction): {high_gain}/{n_total}",
        f"  pearson r(mean_margin, mis_gain_pct): "
        f"{joined['mean_margin'].corr(joined['mis_gain_pct']):+.3f}",
    ]
    if recommended is not None:
        summary += [
            f"  recommended cutoff tau*: {recommended['tau']:.3f}",
            f"  at tau*, recall = {recommended['recall_high_gain']:.2f}, "
            f"compute saved = {recommended['compute_saved_pct']:.1f}%",
        ]
    out = "\n".join(summary) + "\n"
    (OUT_DIR / "triage_summary.txt").write_text(out, encoding="utf-8")
    print("\n" + out)


if __name__ == "__main__":
    main()
