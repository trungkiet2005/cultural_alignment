#!/usr/bin/env python3
"""W9 follow-up: WVS-dim × MultiTP-dim causal impact matrix.

The phase-3 ``exp_r2_wvs_dropout.py`` runs the leave-one-WVS-dim-out study
and writes per-(drop_dim, country) rows with per-MultiTP-dimension errors.
This script turns that long-format CSV into the 10×6 impact matrix
reviewer W9 asked for:

    impact[d, cat] = mean over countries of
                     ( |abserr_cat| when WVS dim `d` is dropped )
                   - ( |abserr_cat| in the no-drop control )

A positive value means that dropping WVS dim ``d`` HURTS MultiTP
dimension ``cat``, identifying a causal coupling between them.

Outputs:
    results/exp24_round2/phase5_analysis/
      ├── wvs_dim_impact_matrix.csv   # 10-row × 6-col numeric matrix
      └── wvs_dim_impact_matrix.tex   # LaTeX heatmap-style table with
                                        top-3 couplings per row bolded
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
from typing import List, Optional

import numpy as np
import pandas as pd

R2_BASE = Path(os.environ.get(
    "R2_RESULTS_BASE",
    "/kaggle/working/cultural_alignment/results/exp24_round2"
    if os.path.isdir("/kaggle/input")
    else "results/exp24_round2",
))
OUT_DIR = R2_BASE / "phase5_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _search_roots() -> List[Path]:
    """Roots to probe for upstream phase-3 outputs.

    Always includes ``R2_BASE``; on Kaggle also adds every directly attached
    Input dataset (one and two levels deep) so a previous run's CSVs can be
    auto-discovered without setting env vars.
    """
    roots: List[Path] = [R2_BASE]
    if os.path.isdir("/kaggle/input"):
        for d in sorted(glob.glob("/kaggle/input/*")):
            roots.append(Path(d))
            for d2 in sorted(glob.glob(f"{d}/*")):
                if os.path.isdir(d2):
                    roots.append(Path(d2))
    return roots


def _find_first(filename: str, override_env: Optional[str] = None) -> Optional[Path]:
    """Find the first ``filename`` under any search root (recursive). When
    ``override_env`` is set in the environment, that path wins."""
    if override_env and os.environ.get(override_env):
        p = Path(os.environ[override_env])
        if p.exists():
            return p
    for root in _search_roots():
        # ``recursive=True`` is REQUIRED for ``**`` to actually recurse.
        for hit in glob.glob(f"{root}/**/{filename}", recursive=True):
            return Path(hit)
    return None

MT_DIMS = ["Species_Humans", "Gender_Female", "Age_Young",
           "Fitness_Fit", "SocialValue_High", "Utilitarianism_More"]
WVS_DIMS = ["religiosity", "child_rearing", "moral_acceptability",
            "social_trust", "political_participation", "national_pride",
            "happiness", "gender_equality", "materialism_orientation",
            "tolerance_diversity"]


def main() -> None:
    src = _find_first("wvs_dropout_summary.csv", override_env="R2_WVS_DROPOUT_CSV")
    if src is None:
        msg = [
            "Missing: wvs_dropout_summary.csv",
            "",
            "Searched these roots (recursive):",
            *[f"  - {r}" for r in _search_roots()],
            "",
            "Fix one of these:",
            "  1) Run phase 3 first in this session:",
            "       !python exp_paper/round2/phase3_sensitivity/exp_r2_wvs_dropout.py",
            "  2) Attach a Kaggle Input dataset that contains the previous",
            "     `results/exp24_round2/wvs_dropout/wvs_dropout_summary.csv`.",
            "  3) Set R2_WVS_DROPOUT_CSV to the absolute file path.",
        ]
        raise SystemExit("\n".join(msg))
    print(f"[USING] {src}")
    df = pd.read_csv(src)

    # Control = drop_dim == '∅'
    ctl = df[df["drop_dim"] == "∅"].set_index("country")
    if ctl.empty:
        raise SystemExit("wvs_dropout_summary.csv has no '∅' control rows.")

    rows = []
    for wvs in WVS_DIMS:
        dropped = df[df["drop_dim"] == wvs]
        if dropped.empty:
            rows.append({"wvs": wvs, **{cat: float("nan") for cat in MT_DIMS}})
            continue
        row = {"wvs": wvs}
        for cat in MT_DIMS:
            col = f"abserr_{cat}"
            if col not in dropped.columns:
                row[cat] = float("nan")
                continue
            merged = dropped.set_index("country")[col].to_frame("drop").join(
                ctl[col].to_frame("ctl"), how="inner",
            )
            if merged.empty:
                row[cat] = float("nan")
                continue
            delta = (merged["drop"] - merged["ctl"]).mean(skipna=True)
            row[cat] = float(delta)
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("wvs")[MT_DIMS]
    mat.to_csv(OUT_DIR / "wvs_dim_impact_matrix.csv")
    print(f"[SAVED] {OUT_DIR / 'wvs_dim_impact_matrix.csv'}")
    print("\nImpact matrix (per-dim |err| shift when WVS dim is dropped, averaged over countries):\n")
    print(mat.round(3).to_string())

    # LaTeX heatmap-style table with top-3 couplings per row bolded
    lines = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{WVS-dim $\times$ MultiTP-dim causal impact matrix. Cells are"
        r" the mean increase in per-dim AMCE error (pp) when the WVS dimension"
        r" is dropped from every persona, macro-averaged across the country"
        r" panel. \textbf{Bold} cells mark the top-3 largest positive couplings"
        r" per row (load-bearing WVS $\to$ MultiTP links).}",
        r"\label{tab:wvs_impact_matrix}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(MT_DIMS) + "}\\toprule",
        r"WVS dim & " + " & ".join(d.replace("_", r"\_") for d in MT_DIMS) + r" \\\midrule",
    ]
    for wvs, row in mat.iterrows():
        vals = row.values.astype(float)
        pos = np.where(np.isfinite(vals) & (vals > 0), vals, -np.inf)
        top3 = set(np.argsort(-pos)[:3]) if np.isfinite(pos).any() else set()
        cells = []
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                cells.append("--")
            elif i in top3 and v > 0:
                cells.append(rf"\textbf{{{v:+.2f}}}")
            else:
                cells.append(f"{v:+.2f}")
        lines.append(f"{wvs.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / "wvs_dim_impact_matrix.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SAVED] {OUT_DIR / 'wvs_dim_impact_matrix.tex'}")

    _zip_outputs(OUT_DIR, "round2_phase5_analysis")


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
