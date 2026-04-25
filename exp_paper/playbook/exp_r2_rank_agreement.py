#!/usr/bin/env python3
"""Round-2 Reviewer W7 -- rank-based per-dimension agreement.

Computes Kendall tau / Spearman rho / mean rank-error over the six MultiTP
dimensions between model AMCE and human AMCE for each country, using the
existing per-country SWA-DPBR and vanilla CSVs from the main Phi-4 run.

No model reload; purely post-hoc from CSVs.

Output:
    results/exp24_round2/rank_agreement/
      ├── rank_agreement_per_country.csv   # one row per (method, country)
      └── r_vs_mis_scatter.csv             # (r, MIS) pairs per country, for
                                             the "why r can be negative while
                                             MIS decreases" scatter.

Usage:
    python exp_paper/playbook/exp_r2_rank_agreement.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
# Clones the repo on Kaggle if not already on sys.path, then adds it. Safe to
# run multiple times (idempotent: detects src/controller.py in cwd).
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _kaggle_setup import bootstrap_offline, zip_outputs as _zip_outputs

bootstrap_offline()
import glob
import os
from pathlib import Path
from typing import Dict, List

from exp_paper._r2_common import ensure_repo, on_kaggle
import pandas as pd  # noqa: E402

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.amce import (  # noqa: E402
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dim_rank_agreement,
    load_human_amce,
)
from src.config import model_slug  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
MODEL_SHORT = os.environ.get("R2_MODEL_SHORT", "phi_4")
HUMAN_AMCE_PATH = os.environ.get(
    "R2_HUMAN_AMCE_PATH",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv",
)
SWA_DIR = os.environ.get(
    "R2_SWA_DIR",
    f"/kaggle/working/cultural_alignment/results/exp24_paper_20c/"
    f"{MODEL_SHORT}/swa/{model_slug(MODEL_NAME)}",
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/rank_agreement"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "rank_agreement")
)


def _load_csv(method: str, country: str) -> pd.DataFrame:
    """Return per-scenario df for (method, country). ``method`` ∈ {swa,vanilla}."""
    name = "swa_results" if method == "swa" else "vanilla_results"
    path = os.path.join(SWA_DIR, f"{name}_{country}.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _analyse_one(method: str, country: str) -> Dict:
    df = _load_csv(method, country)
    if df.empty:
        return {"method": method, "country": country, "note": "missing_csv"}
    df = df.copy()
    df["country"] = country
    model_amce = compute_amce_from_preferences(df)
    human_amce = load_human_amce(HUMAN_AMCE_PATH, country)
    align = compute_alignment_metrics(model_amce, human_amce) if human_amce else {}
    rank = compute_per_dim_rank_agreement(model_amce, human_amce) if human_amce else {}
    return {
        "method":        method,
        "country":       country,
        "mis":           align.get("mis",         float("nan")),
        "jsd":           align.get("jsd",         float("nan")),
        "pearson_r":     align.get("pearson_r",   float("nan")),
        "spearman_rho":  align.get("spearman_rho", float("nan")),
        "kendall_tau":   rank.get("kendall_tau",  float("nan")),
        "rank_spearman": rank.get("spearman_rho", float("nan")),
        "rank_abs_err":  rank.get("rank_abs_err_mean", float("nan")),
        "n_dims":        rank.get("n_dims", 0),
    }


def main() -> None:
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for country in PAPER_20_COUNTRIES:
        for method in ("swa", "vanilla"):
            rows.append(_analyse_one(method, country))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rank_agreement_per_country.csv", index=False)
    print(f"[SAVED] {out_dir / 'rank_agreement_per_country.csv'}  ({len(df)} rows)")

    # Mini scatter table for the paper figure.
    scatter = df[df["method"] == "swa"][["country", "pearson_r", "mis"]].copy()
    scatter.columns = ["country", "swa_r", "swa_mis"]
    scatter.to_csv(out_dir / "r_vs_mis_scatter.csv", index=False)
    print(f"[SAVED] {out_dir / 'r_vs_mis_scatter.csv'}")

    # Quick summary print so the paper appendix can quote headline numbers.
    swa = df[df["method"] == "swa"]
    ven = df[df["method"] == "vanilla"]
    print("\n----- Rank-agreement summary (SWA-DPBR vs vanilla, macro over 20 countries) -----")
    for metric in ("kendall_tau", "rank_spearman", "rank_abs_err"):
        sv = swa[metric].mean(skipna=True)
        vv = ven[metric].mean(skipna=True)
        print(f"  {metric:<14s}  SWA={sv:+.3f}  vanilla={vv:+.3f}  Δ={sv - vv:+.3f}")


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE if 'RESULTS_BASE' in globals() else OUT_DIR if 'OUT_DIR' in globals() else '.')
    except Exception as _e:
        print(f'[ZIP] failed: {_e}')
