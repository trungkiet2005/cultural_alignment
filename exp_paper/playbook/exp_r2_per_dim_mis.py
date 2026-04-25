#!/usr/bin/env python3
"""Oral-level analysis: per-dimension AMCE error breakdown (Reviewer Q6).

Reads the per-country ``swa_results_*.csv`` and ``baseline_results_*.csv``
from the main Phi-4 paper run and decomposes the aggregate MIS reduction
into its six MultiTP moral dimensions:

    Species_Humans | Gender_Female | Age_Young
    Fitness_Fit    | SocialValue_High | Utilitarianism_More

For each country × dimension we report:
    vanilla_err   -- |vanilla AMCE - human AMCE|
    swa_err       -- |SWA-DPBR AMCE - human AMCE|
    delta         -- vanilla_err - swa_err  (positive = SWA improved)
    pct_gain      -- 100 * delta / vanilla_err

Key claims this supports:
  • Which dimensions drive the 19--24% aggregate gain?
  • Are there any regressing dimensions (delta < 0)?
  • Is the gain concentrated in a subset of dimensions or broad?

No GPU required -- pure post-hoc over existing CSVs.

Kaggle:
    !python exp_paper/playbook/exp_r2_per_dim_mis.py

Env overrides:
    R2_MODEL_SHORT    slug used to build default CSV paths (default: phi_4)
    R2_RESULTS_BASE   root of results/exp24_paper_20c/   (Kaggle default auto-set)
    R2_HUMAN_AMCE     path to country_specific_ACME.csv   (Kaggle default auto-set)
    R2_COUNTRIES      comma-separated ISO3 (default: all 20 paper countries)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
# ─────────────────────────────────────────────────────────────────────────────
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
        raise RuntimeError(
            "Not on Kaggle and not inside the repo root. "
            "Either cd into the cultural_alignment repo first, or run on Kaggle."
        )
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from exp_paper._r2_common import on_kaggle
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.amce import (
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    load_human_amce,
)

# ─── paths ──────────────────────────────────────────────────────────────────
MODEL_SHORT = os.environ.get("R2_MODEL_SHORT", "phi_4")
MODEL_SLUG  = os.environ.get("R2_MODEL_SLUG", "phi-4")  # model_slug(MODEL_NAME)

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
    "/kaggle/working/cultural_alignment/results/exp24_round2/per_dim_mis"
    if on_kaggle()
    else "results/exp24_round2/per_dim_mis"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [
    "Species_Humans", "Gender_Female", "Age_Young",
    "Fitness_Fit", "SocialValue_High", "Utilitarianism_More",
]
DIM_SHORT = {
    "Species_Humans":    "Species",
    "Gender_Female":     "Gender",
    "Age_Young":         "Age",
    "Fitness_Fit":       "Fitness",
    "SocialValue_High":  "SocVal",
    "Utilitarianism_More": "Util",
}


# ─── loaders ────────────────────────────────────────────────────────────────
def _find_csv(pattern: str) -> Optional[str]:
    # ``recursive=True`` is REQUIRED for ``**`` to actually recurse — without
    # it, glob silently treats ``**`` as a single ``*`` and most wildcard
    # patterns return nothing.
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def _search_roots() -> List[str]:
    """Roots to probe for per-country result CSVs.

    Always includes ``RESULTS_BASE``; on Kaggle also probes every directory
    directly under ``/kaggle/input/`` so an attached results dataset is
    auto-discovered.
    """
    roots = [RESULTS_BASE]
    if on_kaggle():
        # Two-level deep so we cover both /kaggle/input/<dataset>/ and
        # /kaggle/input/<dataset>/<subdir>/  (common nested layouts).
        for d in sorted(glob.glob("/kaggle/input/*")):
            roots.append(d)
            for d2 in sorted(glob.glob(f"{d}/*")):
                if os.path.isdir(d2):
                    roots.append(d2)
    return roots


def _load_results(country: str, kind: str) -> Optional[pd.DataFrame]:
    """kind = 'swa' | 'baseline'"""
    patterns: List[str] = []
    for root in _search_roots():
        patterns += [
            f"{root}/{MODEL_SHORT}/{kind}/{MODEL_SLUG}/{kind}_results_{country}.csv",
            f"{root}/{MODEL_SHORT}/{kind}/**/{kind}_results_{country}.csv",
            f"{root}/**/{kind}/**/{kind}_results_{country}.csv",
            f"{root}/**/{kind}_results_{country}.csv",
        ]
    for pat in patterns:
        p = _find_csv(pat)
        if p:
            print(f"  [HIT] {kind} {country} ← {p}")
            return pd.read_csv(p)
    print(f"[MISS] {kind} results for {country} (tried {len(patterns)} patterns)")
    return None


# ─── per-country computation ─────────────────────────────────────────────────
def _compute_per_dim(results_df: pd.DataFrame, human: Dict[str, float]) -> Dict[str, float]:
    """Return per-dim absolute error dict {dim: abs_err}."""
    model_amce = compute_amce_from_preferences(results_df)
    breakdown  = compute_per_dimension_alignment(model_amce, human)
    return {dim: breakdown[dim]["abs_err"] for dim in breakdown}


def _process_country(country: str) -> Optional[Dict]:
    human = load_human_amce(HUMAN_AMCE_PATH, country)
    if not human:
        print(f"[SKIP] {country} — no human AMCE")
        return None

    van_df = _load_results(country, "baseline")
    swa_df = _load_results(country, "swa")
    if van_df is None or swa_df is None:
        return None

    van_err = _compute_per_dim(van_df, human)
    swa_err = _compute_per_dim(swa_df, human)

    row = {"country": country}
    for dim in DIMS:
        v = van_err.get(dim, float("nan"))
        s = swa_err.get(dim, float("nan"))
        delta = v - s
        pct   = 100.0 * delta / v if v > 1e-9 else float("nan")
        row[f"{dim}_van"]   = round(v, 4)
        row[f"{dim}_swa"]   = round(s, 4)
        row[f"{dim}_delta"] = round(delta, 4)
        row[f"{dim}_pct"]   = round(pct, 2)
    return row


# ─── LaTeX heatmap table ─────────────────────────────────────────────────────
def _build_latex(df: pd.DataFrame) -> str:
    """Produce a coloured heatmap table: rows=countries, cols=dimensions.
    Cells show % gain; green = improvement, red = regression.
    """
    short = [DIM_SHORT[d] for d in DIMS]
    lines = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{Per-dimension AMCE error reduction (\%) for SWA-DPBR vs.\ "
        r"vanilla Phi-4 (14B) across 20 countries. "
        r"Positive = SWA-DPBR reduced absolute error on that dimension. "
        r"\textbf{Bold} = macro mean row. "
        r"Macro MIS gain (\%) is the aggregate headline number.}",
        r"\label{tab:per_dim_pct_gain}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(DIMS) + r"}\toprule",
        r"Country & " + " & ".join(short) + r" \\\midrule",
    ]
    pct_cols = [f"{d}_pct" for d in DIMS]
    for _, row in df[df["country"] != "MACRO"].iterrows():
        cells = []
        for col in pct_cols:
            v = row[col]
            if not np.isfinite(v):
                cells.append("--")
            elif v >= 5.0:
                cells.append(rf"\gain{{{v:+.1f}}}")
            elif v <= -5.0:
                cells.append(rf"\loss{{{v:+.1f}}}")
            else:
                cells.append(f"{v:+.1f}")
        lines.append(row["country"] + " & " + " & ".join(cells) + r" \\")

    # Macro row
    macro = df[df["country"] == "MACRO"].iloc[0] if (df["country"] == "MACRO").any() else None
    if macro is not None:
        lines.append(r"\midrule")
        cells = []
        for col in pct_cols:
            v = macro[col]
            s = f"{v:+.1f}" if np.isfinite(v) else "--"
            cells.append(rf"\textbf{{{s}}}")
        lines.append(r"\textbf{Macro mean} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[PER-DIM] results_base = {RESULTS_BASE}")
    print(f"[PER-DIM] countries    = {COUNTRIES}\n")

    rows = []
    for country in COUNTRIES:
        r = _process_country(country)
        if r is not None:
            rows.append(r)
            pcts = [r[f"{d}_pct"] for d in DIMS]
            ok = [f"{p:+.1f}%" for p in pcts if np.isfinite(p)]
            print(f"  {country:>4s}  " + "  ".join(
                f"{DIM_SHORT[d]}={r[f'{d}_pct']:+.1f}%" for d in DIMS
                if np.isfinite(r[f"{d}_pct"])
            ))

    if not rows:
        roots = _search_roots()
        msg = [
            "[ERROR] No per-country result CSVs found.",
            "",
            "This script needs the per-country files from the main Phi-4 run:",
            f"  swa_results_<COUNTRY>.csv  and  baseline_results_<COUNTRY>.csv",
            "",
            "Searched these roots (recursively):",
            *[f"  - {r}" for r in roots],
            "",
            "Fix one of these:",
            "  1) Attach a Kaggle Input dataset that contains the previous Phi-4",
            "     `results/exp24_paper_20c/phi_4/{swa,baseline}/phi-4/*.csv` files.",
            "  2) Run the main Phi-4 experiment first in this session:",
            "       !python exp_paper/models/exp_paper_phi_4.py",
            "  3) Set R2_RESULTS_BASE to the dir that contains `phi_4/swa/...`,",
            "     e.g. `/kaggle/input/<your-dataset>/results/exp24_paper_20c`.",
        ]
        raise SystemExit("\n".join(msg))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_dim_errors.csv", index=False)
    print(f"\n[SAVED] {OUT_DIR / 'per_dim_errors.csv'}")

    # Macro summary row
    macro = {"country": "MACRO"}
    for dim in DIMS:
        for suffix in ("_van", "_swa", "_delta", "_pct"):
            col = f"{dim}{suffix}"
            macro[col] = float(df[col].mean(skipna=True))
    df_with_macro = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    df_with_macro.to_csv(OUT_DIR / "per_dim_macro.csv", index=False)

    # LaTeX heatmap
    latex = _build_latex(df_with_macro)
    (OUT_DIR / "per_dim_heatmap.tex").write_text(latex, encoding="utf-8")
    print(f"[SAVED] {OUT_DIR / 'per_dim_heatmap.tex'}")

    # Summary text
    lines = ["Per-dimension AMCE breakdown — SWA-DPBR vs. vanilla Phi-4",
             "=" * 60]
    for dim in DIMS:
        pct_col = f"{dim}_pct"
        mean_pct = float(df[pct_col].mean(skipna=True))
        n_pos = int((df[pct_col] > 0).sum())
        n_neg = int((df[pct_col] < 0).sum())
        lines.append(
            f"  {DIM_SHORT[dim]:<7s}  macro gain = {mean_pct:+6.2f}%  "
            f"wins={n_pos}/{len(df)}  losses={n_neg}/{len(df)}"
        )

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "per_dim_summary.txt").write_text(summary, encoding="utf-8")
    print(f"[SAVED] {OUT_DIR / 'per_dim_summary.txt'}")
    print("\n" + summary)

    _zip_outputs(OUT_DIR, "round2_phase7_per_dim_mis")


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
    print(f"[ZIP] {zip_path}")


if __name__ == "__main__":
    main()
