#!/usr/bin/env python3
"""Cross-model per-dimension MIS-reduction matrix.

Aggregates the per-country/per-dimension AMCE error breakdown across
all six paper models, producing a 6-models x 6-dimensions macro matrix
that answers: "where do SWA-DPBR's gains come from, and is the pattern
backbone-dependent?"

Reads the per-country ``swa_results_*.csv`` and ``baseline_results_*.csv``
written by ``exp_paper/exp_paper_<model>.py`` and computes, for each
(model, dimension) cell:
    macro_van_err   -- mean |vanilla AMCE - human AMCE| over 20 countries
    macro_swa_err   -- mean |SWA-DPBR AMCE - human AMCE| over 20 countries
    macro_delta     -- van - swa (positive = SWA helped)
    macro_pct_gain  -- 100 * delta / van

No GPU required -- pure post-hoc over existing CSVs.

Kaggle:
    !python exp_paper/round3/posthoc/exp_r3_per_dim_cross_model.py

Env overrides:
    R2_RESULTS_BASE   root of results/exp24_paper_20c/   (Kaggle default auto-set)
    R2_HUMAN_AMCE     country_specific_ACME.csv          (Kaggle default auto-set)
    R2_COUNTRIES      comma ISO3 list (default: all 20)
    R2_MODELS         comma list of (slug,short) pairs   (default: 6 paper models)
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp_paper._r2_common import on_kaggle
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.amce import (
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    load_human_amce,
)

# Six paper models. (model_short_dir, model_slug_subdir).
DEFAULT_MODELS: List[Tuple[str, str, str]] = [
    # display_name,             dir_short,                slug
    ("Llama-3.3-70B",           "llama_3_3_70b",          "meta-llama-3.3-70b-instruct"),
    ("Magistral-Small-2509",    "magistral_small_2509",   "magistral-small-2509"),
    ("Phi-4",                   "phi_4",                  "phi-4"),
    ("Qwen3-VL-8B",             "qwen3_vl_8b",            "qwen3-vl-8b-instruct"),
    ("Qwen2.5-7B",              "qwen2_5_7b",             "qwen2.5-7b-instruct"),
    ("Phi-3.5-mini",            "phi_3_5_mini",           "phi-3.5-mini-instruct"),
]

if "R2_MODELS" in os.environ:
    MODELS = []
    for tok in os.environ["R2_MODELS"].split(";"):
        parts = [s.strip() for s in tok.split(",")]
        if len(parts) >= 3:
            MODELS.append(tuple(parts[:3]))
else:
    MODELS = DEFAULT_MODELS

_PAPER_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    if on_kaggle()
    else "results/exp24_paper_20c"
)
RESULTS_BASE = os.environ.get("R2_RESULTS_BASE", _PAPER_BASE)

HUMAN_AMCE_PATH = os.environ.get(
    "R2_HUMAN_AMCE",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    if on_kaggle()
    else "WVS_data/country_specific_ACME.csv",
)
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round3/per_dim_cross_model"
    if on_kaggle()
    else "results/exp24_round3/per_dim_cross_model"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [
    "Species_Humans", "Gender_Female", "Age_Young",
    "Fitness_Fit", "SocialValue_High", "Utilitarianism_More",
]
DIM_SHORT = {
    "Species_Humans":      "Species",
    "Gender_Female":       "Gender",
    "Age_Young":           "Age",
    "Fitness_Fit":         "Fitness",
    "SocialValue_High":    "SocVal",
    "Utilitarianism_More": "Util",
}


# ─── loaders ────────────────────────────────────────────────────────────────
def _search_roots() -> List[str]:
    roots = [RESULTS_BASE]
    if on_kaggle():
        for d in sorted(glob.glob("/kaggle/input/*")):
            roots.append(d)
            for d2 in sorted(glob.glob(f"{d}/*")):
                if os.path.isdir(d2):
                    roots.append(d2)
    return roots


def _find_csv(model_short: str, model_slug: str, kind: str, country: str) -> Optional[str]:
    for root in _search_roots():
        for pat in [
            f"{root}/{model_short}/{kind}/{model_slug}/{kind}_results_{country}.csv",
            f"{root}/{model_short}/{kind}/**/{kind}_results_{country}.csv",
            f"{root}/**/{model_short}/{kind}/**/{kind}_results_{country}.csv",
            f"{root}/**/{kind}_results_{country}.csv",
        ]:
            hits = glob.glob(pat, recursive=True)
            if hits:
                return hits[0]
    return None


def _per_dim_err(results_df: pd.DataFrame, human: Dict[str, float]) -> Dict[str, float]:
    model_amce = compute_amce_from_preferences(results_df)
    breakdown = compute_per_dimension_alignment(model_amce, human)
    return {d: breakdown[d]["abs_err"] for d in breakdown}


def _process_one_cell(model_short: str, model_slug: str,
                      country: str, human: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
    van_path = _find_csv(model_short, model_slug, "baseline", country)
    swa_path = _find_csv(model_short, model_slug, "swa", country)
    if not van_path or not swa_path:
        return None
    van_df = pd.read_csv(van_path)
    swa_df = pd.read_csv(swa_path)
    return {
        "van": _per_dim_err(van_df, human),
        "swa": _per_dim_err(swa_df, human),
    }


# ─── matrix builder ──────────────────────────────────────────────────────────
def _build_macro_matrix(per_cell: Dict[Tuple[str, str], Dict[str, Dict[str, float]]]
                        ) -> pd.DataFrame:
    """One row per (model, dim). Columns: macro_van, macro_swa, delta, pct."""
    rows = []
    for (display_name, _, _) in MODELS:
        for dim in DIMS:
            vans, swas = [], []
            for country in COUNTRIES:
                cell = per_cell.get((display_name, country))
                if cell is None:
                    continue
                v = cell["van"].get(dim, np.nan)
                s = cell["swa"].get(dim, np.nan)
                if np.isfinite(v) and np.isfinite(s):
                    vans.append(v)
                    swas.append(s)
            if not vans:
                continue
            mv, ms = float(np.mean(vans)), float(np.mean(swas))
            rows.append({
                "model": display_name,
                "dimension": DIM_SHORT[dim],
                "macro_van_err":  round(mv, 4),
                "macro_swa_err":  round(ms, 4),
                "macro_delta":    round(mv - ms, 4),
                "macro_pct_gain": round(100.0 * (mv - ms) / mv, 2) if mv > 1e-9 else float("nan"),
                "n_countries":    len(vans),
            })
    return pd.DataFrame(rows)


# ─── LaTeX heatmap ───────────────────────────────────────────────────────────
def _build_latex_heatmap(df: pd.DataFrame) -> str:
    """6 models x 6 dims, cells = macro % MIS reduction."""
    if df.empty:
        return ""
    pivot = df.pivot(index="model", columns="dimension", values="macro_pct_gain")
    pivot = pivot.reindex([m[0] for m in MODELS])
    pivot = pivot[[DIM_SHORT[d] for d in DIMS]]
    lines = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{\textbf{Cross-model per-dimension MIS reduction (\%).} Each "
        r"cell is the macro-mean relative reduction in absolute AMCE error on "
        r"that dimension, averaged across the 20-country panel. Positive (green) "
        r"= SWA-DPBR helped; negative (red) = SWA-DPBR hurt. The Macro Mean "
        r"column averages within model.}",
        r"\label{tab:per_dim_cross_model}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * (len(DIMS) + 1) + r"}\toprule",
        r"Model & " + " & ".join(DIM_SHORT[d] for d in DIMS) + r" & Mean \\\midrule",
    ]
    for model_name, row in pivot.iterrows():
        cells = []
        for dim in DIMS:
            v = row[DIM_SHORT[dim]]
            if not np.isfinite(v):
                cells.append("--")
            elif v >= 5.0:
                cells.append(rf"\gain{{{v:+.1f}}}")
            elif v <= -5.0:
                cells.append(rf"\loss{{{v:+.1f}}}")
            else:
                cells.append(f"{v:+.1f}")
        # Per-model mean
        mvals = [row[DIM_SHORT[d]] for d in DIMS if np.isfinite(row[DIM_SHORT[d]])]
        mean_str = f"{np.mean(mvals):+.1f}" if mvals else "--"
        lines.append(model_name + " & " + " & ".join(cells)
                     + r" & \textbf{" + mean_str + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[per-dim x model] results_base = {RESULTS_BASE}")
    print(f"[per-dim x model] models       = {[m[0] for m in MODELS]}")
    print(f"[per-dim x model] countries    = {COUNTRIES}\n")

    per_cell: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for (display_name, model_short, model_slug) in MODELS:
        print(f"\n=== {display_name} ({model_short}/{model_slug}) ===")
        for country in COUNTRIES:
            human = load_human_amce(HUMAN_AMCE_PATH, country)
            if not human:
                continue
            cell = _process_one_cell(model_short, model_slug, country, human)
            if cell is None:
                print(f"  [miss] {country}")
                continue
            per_cell[(display_name, country)] = cell
        n_have = sum(1 for c in COUNTRIES if (display_name, c) in per_cell)
        print(f"  → {n_have}/{len(COUNTRIES)} countries available")

    if not per_cell:
        raise SystemExit(
            "[error] No per-country result CSVs found anywhere. "
            "Make sure the main per-model runs (exp_paper_<model>.py) have been "
            "completed and the result CSVs are reachable from R2_RESULTS_BASE "
            "or attached as a Kaggle input dataset."
        )

    df = _build_macro_matrix(per_cell)
    df.to_csv(OUT_DIR / "per_dim_cross_model.csv", index=False)
    print(f"\n[saved] {OUT_DIR / 'per_dim_cross_model.csv'}")

    latex = _build_latex_heatmap(df)
    (OUT_DIR / "per_dim_cross_model_heatmap.tex").write_text(latex, encoding="utf-8")
    print(f"[saved] {OUT_DIR / 'per_dim_cross_model_heatmap.tex'}")

    # Quick text summary
    lines = ["Cross-model per-dimension MIS reduction (macro %)", "=" * 55]
    for (display_name, _, _) in MODELS:
        sub = df[df["model"] == display_name]
        if sub.empty:
            continue
        best = sub.nlargest(1, "macro_pct_gain")
        worst = sub.nsmallest(1, "macro_pct_gain")
        lines.append(
            f"  {display_name:<24s}  best: {best.iloc[0]['dimension']:>7s} "
            f"({best.iloc[0]['macro_pct_gain']:+.1f}%)  "
            f"worst: {worst.iloc[0]['dimension']:>7s} "
            f"({worst.iloc[0]['macro_pct_gain']:+.1f}%)"
        )
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "per_dim_cross_model_summary.txt").write_text(summary, encoding="utf-8")
    print(f"\n{summary}")

    _zip_outputs(OUT_DIR, "round3_posthoc_per_dim_cross_model")


def _zip_outputs(out_dir: Path, label: str) -> None:
    import shutil
    dest_base = (
        Path("/kaggle/working")
        if os.path.isdir("/kaggle/input")
        else out_dir.parent.parent / "download"
    )
    dest_base.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(dest_base / label), "zip",
        root_dir=str(out_dir.parent),
        base_dir=out_dir.name,
    )
    print(f"[zip] {zip_path}")


if __name__ == "__main__":
    main()
