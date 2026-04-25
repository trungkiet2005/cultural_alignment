#!/usr/bin/env python3
"""Cross-model logit-conditioning — aggregate step (CPU-only).

Reads the per-model ``logit_cond_<model>.csv`` files produced by the two
GPU tracks (``exp_r3_logit_cond_vllm.py`` + ``exp_r3_logit_cond_unsloth.py``)
and the per-model ``comparison.csv`` from the main SWA-DPBR runs, then
computes per-model and pooled Pearson/Spearman correlations between
vanilla decision-margin and SWA-DPBR's relative MIS reduction.

Outputs:
  logit_cond_cross_model_pooled.csv         — one row per (model, country)
  logit_cond_cross_model_correlations.csv   — per-model + pooled summary
  logit_cond_cross_model_table.tex          — LaTeX table for the paper

Copy-paste into a fresh (CPU) Kaggle notebook:

    !python exp_paper/review/round3/posthoc/exp_r3_logit_cond_aggregate.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
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


_bootstrap()

import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from exp_paper._r2_common import on_kaggle

# (display_name, dir_short, slug)
MODELS: List[Tuple[str, str, str]] = [
    ("Llama-3.3-70B",        "llama_3_3_70b",        "meta-llama-3.3-70b-instruct"),
    ("Magistral-Small-2509", "magistral_small_2509", "magistral-small-2509"),
    ("Phi-4",                "phi_4",                "phi-4"),
    ("Qwen3-VL-8B",          "qwen3_vl_8b",          "qwen3-vl-8b-instruct"),
    ("Qwen2.5-7B",           "qwen2_5_7b",           "qwen2.5-7b-instruct"),
    ("Phi-3.5-mini",         "phi_3_5_mini",         "phi-3.5-mini-instruct"),
]

OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round3/logit_conditioning_cross_model"
    if on_kaggle()
    else "results/exp24_round3/logit_conditioning_cross_model"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    if on_kaggle()
    else "results/exp24_paper_20c"
)


def _load_per_country_mis(dir_short: str) -> Optional[pd.DataFrame]:
    candidates = [
        f"{PAPER_RESULTS_BASE}/{dir_short}/comparison.csv",
        f"{PAPER_RESULTS_BASE}/**/{dir_short}/**/comparison.csv",
    ]
    if on_kaggle():
        for d in sorted(glob.glob("/kaggle/input/*")):
            candidates.extend([
                f"{d}/results/exp24_paper_20c/{dir_short}/comparison.csv",
                f"{d}/**/{dir_short}/**/comparison.csv",
            ])
    for pat in candidates:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return pd.read_csv(hits[0])
    return None


def _zip_outputs(out_dir: Path, label: str) -> None:
    dest_base = (
        Path("/kaggle/working") if os.path.isdir("/kaggle/input")
        else out_dir.parent.parent / "download"
    )
    dest_base.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(dest_base / label), "zip",
        root_dir=str(out_dir.parent),
        base_dir=out_dir.name,
    )
    print(f"[zip] {zip_path}")


def main() -> None:
    from scipy.stats import pearsonr, spearmanr

    pooled_rows: List[Dict] = []
    per_model_rows: List[Dict] = []

    for (display_name, dir_short, _slug) in MODELS:
        cond_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
        if not cond_path.exists():
            print(f"[skip] {display_name}: no {cond_path}")
            continue
        cond = pd.read_csv(cond_path)
        cmp = _load_per_country_mis(dir_short)
        if cmp is None:
            print(f"[skip] {display_name}: no comparison.csv (run main {dir_short} first)")
            continue

        van = cmp[cmp["method"] == "baseline_vanilla"][["country", "align_mis"]].rename(
            columns={"align_mis": "van_mis"})
        swa = cmp[cmp["method"].str.contains("dual_pass", na=False)][["country", "align_mis"]].rename(
            columns={"align_mis": "swa_mis"})
        mis = van.merge(swa, on="country")
        mis["mis_gain"]     = mis["van_mis"] - mis["swa_mis"]
        mis["mis_gain_pct"] = 100.0 * mis["mis_gain"] / mis["van_mis"]

        merged = cond.merge(mis, on="country")
        merged["model"] = display_name
        pooled_rows.extend(merged.to_dict("records"))

        if len(merged) >= 5:
            r,  p  = pearsonr(merged["mean_margin"],  merged["mis_gain_pct"])
            rh, ph = spearmanr(merged["mean_margin"], merged["mis_gain_pct"])
            per_model_rows.append({
                "model":        display_name,
                "n":            len(merged),
                "pearson_r":    round(r, 3),
                "pearson_p":    round(p, 4),
                "spearman_rho": round(rh, 3),
                "spearman_p":   round(ph, 4),
            })
            print(f"  {display_name:<22s}  N={len(merged):>2d}  "
                  f"r={r:+.3f} (p={p:.3f})  rho={rh:+.3f}")

    if not pooled_rows:
        raise SystemExit("[error] No (logit_cond, comparison) pairs found.")

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT_DIR / "logit_cond_cross_model_pooled.csv", index=False)
    print(f"\n[saved] {OUT_DIR / 'logit_cond_cross_model_pooled.csv'}  (N={len(pooled)} cells)")

    rp, pp = pearsonr(pooled["mean_margin"],  pooled["mis_gain_pct"])
    rsp, psp = spearmanr(pooled["mean_margin"], pooled["mis_gain_pct"])
    n_models = pooled["model"].nunique()
    n_countries = pooled["country"].nunique()
    per_model_rows.append({
        "model":        f"POOLED ({n_models} x {n_countries})",
        "n":            len(pooled),
        "pearson_r":    round(rp, 3),
        "pearson_p":    round(pp, 4),
        "spearman_rho": round(rsp, 3),
        "spearman_p":   round(psp, 4),
    })
    summary = pd.DataFrame(per_model_rows)
    summary.to_csv(OUT_DIR / "logit_cond_cross_model_correlations.csv", index=False)
    print(f"[saved] {OUT_DIR / 'logit_cond_cross_model_correlations.csv'}")
    print("\n" + summary.to_string(index=False))

    lines = [
        r"\begin{tabular}{lrrr}\toprule",
        r"Model & $N$ & Pearson $r$ & Spearman $\rho$ \\\midrule",
    ]
    for _, row in summary.iterrows():
        sig = "$^{*}$" if row["pearson_p"] < 0.05 else ""
        lines.append(
            f"{row['model']} & {int(row['n'])} & "
            f"{row['pearson_r']:+.3f}{sig} & {row['spearman_rho']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "logit_cond_cross_model_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {OUT_DIR / 'logit_cond_cross_model_table.tex'}")

    _zip_outputs(OUT_DIR, "round3_posthoc_logit_cond_cross_model")


if __name__ == "__main__":
    main()
