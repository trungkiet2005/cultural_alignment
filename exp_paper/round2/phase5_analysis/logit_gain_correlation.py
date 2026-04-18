#!/usr/bin/env python3
"""W10 follow-up: correlation between base-model decision margin and SWA-DPBR MIS gain.

The paper's logit-conditioning diagnostic (Appendix~\\ref{app:r2_logit_conditioning})
predicts that per-country MIS improvement should correlate positively with the
vanilla model's mean decision margin. This script provides the direct evidence:
it loads the phase-3 ``logit_conditioning_per_country.csv`` and the main Phi-4
run's per-country MIS (vanilla vs SWA-DPBR) and outputs

    results/exp24_round2/phase5_analysis/
      ├── margin_vs_gain_scatter.csv   # (country, mean_margin, MIS_gain_pct) rows
      └── margin_vs_gain_summary.txt   # Pearson r, Spearman rho, n, slope

Usage (no GPU):
    python exp_paper/round2/phase5_analysis/logit_gain_correlation.py

Env overrides:
    R2_RESULTS_BASE   root of results/exp24_round2/
    R2_MAIN_COMPARISON  path to the main Phi-4 compare CSV
        (default: results/exp24_paper_20c/phi_4/compare/comparison.csv)
"""

from __future__ import annotations

# ─── self-bootstrap ─────────────────────────────────────────────────────────
import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _r2_bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/working"):
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
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

R2_BASE = Path(os.environ.get(
    "R2_RESULTS_BASE",
    "/kaggle/working/cultural_alignment/results/exp24_round2"
    if os.path.isdir("/kaggle/working")
    else "results/exp24_round2",
))
OUT_DIR = R2_BASE / "phase5_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _search_roots() -> List[Path]:
    """R2_BASE first, then every Kaggle Input dataset (1-2 levels deep)."""
    roots: List[Path] = [R2_BASE, R2_BASE.parent]  # also probe results/
    if os.path.isdir("/kaggle/input"):
        for d in sorted(glob.glob("/kaggle/input/*")):
            roots.append(Path(d))
            for d2 in sorted(glob.glob(f"{d}/*")):
                if os.path.isdir(d2):
                    roots.append(Path(d2))
    return roots


def _find_first(filename: str, override_env: Optional[str] = None) -> Optional[Path]:
    if override_env and os.environ.get(override_env):
        p = Path(os.environ[override_env])
        if p.exists():
            return p
    for root in _search_roots():
        for hit in glob.glob(f"{root}/**/{filename}", recursive=True):
            return Path(hit)
    return None


def main() -> None:
    cond_path = _find_first(
        "logit_conditioning_per_country.csv",
        override_env="R2_LOGIT_COND_CSV",
    )
    if cond_path is None:
        msg = [
            "Missing: logit_conditioning_per_country.csv",
            "",
            "Searched these roots (recursive):",
            *[f"  - {r}" for r in _search_roots()],
            "",
            "Fix one of these:",
            "  1) Run phase 3 first in this session:",
            "       !python exp_paper/round2/phase3_sensitivity/exp_r2_logit_conditioning.py",
            "  2) Attach a Kaggle Input dataset that contains the previous",
            "     `results/exp24_round2/logit_conditioning/logit_conditioning_per_country.csv`.",
            "  3) Set R2_LOGIT_COND_CSV to the absolute file path.",
        ]
        raise SystemExit("\n".join(msg))
    print(f"[USING] cond = {cond_path}")
    cond = pd.read_csv(cond_path)

    cmp_path = _find_first("comparison.csv", override_env="R2_MAIN_COMPARISON")
    if cmp_path is None:
        msg = [
            "Missing: comparison.csv (main Phi-4 paper run)",
            "",
            "Searched these roots (recursive):",
            *[f"  - {r}" for r in _search_roots()],
            "",
            "Fix one of these:",
            "  1) Run the main Phi-4 experiment first in this session:",
            "       !python exp_paper/exp_paper_phi_4.py",
            "  2) Attach a Kaggle Input dataset that contains the previous",
            "     `results/exp24_paper_20c/phi_4/compare/comparison.csv`.",
            "  3) Set R2_MAIN_COMPARISON to the absolute file path.",
        ]
        raise SystemExit("\n".join(msg))
    print(f"[USING] cmp  = {cmp_path}")
    cmp = pd.read_csv(cmp_path)

    # Extract per-country (vanilla MIS, SWA-DPBR MIS).
    van = cmp[cmp["method"] == "baseline_vanilla"][["country", "align_mis"]].rename(
        columns={"align_mis": "mis_vanilla"},
    )
    swa = cmp[cmp["method"].str.contains("dual_pass", na=False)][
        ["country", "align_mis"]].rename(columns={"align_mis": "mis_swa"})
    if van.empty or swa.empty:
        raise SystemExit(
            "Main comparison CSV missing vanilla or SWA rows. "
            "Re-check the `method` column."
        )

    m = (cond[["country", "mean_margin", "median_margin",
               "mean_entropy", "mean_abs_gap"]]
         .merge(van, on="country")
         .merge(swa, on="country"))
    m["mis_gain"] = m["mis_vanilla"] - m["mis_swa"]
    m["mis_gain_pct"] = 100.0 * m["mis_gain"] / m["mis_vanilla"].replace(0, np.nan)

    scatter = m[["country", "mean_margin", "median_margin",
                 "mean_entropy", "mis_vanilla", "mis_swa",
                 "mis_gain", "mis_gain_pct"]].copy()
    scatter.to_csv(OUT_DIR / "margin_vs_gain_scatter.csv", index=False)
    print(f"[SAVED] {OUT_DIR / 'margin_vs_gain_scatter.csv'}")

    # Correlation summary. Expected sign: +ve (higher margin → larger gain).
    lines = ["W10 logit-conditioning correlation analysis",
             "-" * 54,
             f"n_countries = {len(m)}",
             ""]
    for xcol in ("mean_margin", "median_margin", "mean_entropy", "mean_abs_gap"):
        for ycol in ("mis_gain", "mis_gain_pct"):
            x = m[xcol].values.astype(np.float64)
            y = m[ycol].values.astype(np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            r, rp = pearsonr(x[mask], y[mask])
            rho, rhop = spearmanr(x[mask], y[mask])
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            lines.append(
                f"{xcol:>15s} vs {ycol:<12s}  "
                f"Pearson r = {r:+.3f} (p={rp:.3f})  "
                f"Spearman rho = {rho:+.3f} (p={rhop:.3f})  "
                f"slope = {slope:+.4f}"
            )
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "margin_vs_gain_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # ASCII scatter hint.
    print("\nCountries ranked by mean_margin:")
    ranked = m.sort_values("mean_margin").reset_index(drop=True)
    for _, r in ranked.iterrows():
        bar_len = max(0, int(round(25.0 * float(r["mis_gain_pct"]) / 50.0))) \
                  if np.isfinite(r["mis_gain_pct"]) else 0
        bar = "#" * bar_len
        print(f"  {r['country']:>4s}  margin={r['mean_margin']:.3f}  "
              f"gain={r['mis_gain_pct']:+6.1f}%  {bar}")

    _zip_outputs(OUT_DIR, "round2_phase5_analysis")


def _zip_outputs(out_dir: Path, label: str) -> None:
    import shutil
    dest_base = (
        Path("/kaggle/working")
        if os.path.isdir("/kaggle/working")
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
