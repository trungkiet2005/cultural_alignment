#!/usr/bin/env python3
"""Cross-model logit-conditioning vs SWA-DPBR MIS-gain correlation.

Generalises ``exp_r2_logit_conditioning.py`` (Phi-4 only) to all six
paper backbones, then computes the per-(model, country) correlation
between vanilla decision-margin and SWA-DPBR's relative MIS reduction.

Output: a single 6-models x 20-countries scatter dataset plus an
aggregated correlation table (Pearson r, Spearman rho, p-values) at
two granularities:
  • per-model:  N=20  (one (margin, gain) per country)
  • pooled:     N=120 (all model x country cells together)

The architectural-failure claim from \\S\\ref{sec:discussion}-that
poorly-conditioned decision logits cap SWA-DPBR's headroom-currently
rests on a single backbone. This script extends it to a 6-backbone
panel.

Two-step run (each step is independent):

  Step 1 (GPU, ~30 min on H100 per model): vanilla logit-conditioning
    !python exp_paper/round2/phase3_sensitivity/exp_r2_logit_conditioning_cross_model.py
  This walks every model and writes ``logit_cond_<model>.csv``.

  Step 2 (CPU, post-hoc): aggregate against SWA-DPBR per-country MIS gains
    !python exp_paper/round2/phase3_sensitivity/exp_r2_logit_conditioning_cross_model.py --aggregate-only
  Reads the step-1 CSVs + the existing per-model swa/baseline summaries,
  produces the cross-model correlation table.

Env overrides:
    R2_MODELS         comma list of model HF ids (default: 6 paper models)
    R2_COUNTRIES      comma ISO3 list (default: all 20)
    R2_N_SCENARIOS    per-country scenario count (default: 300)
    R2_BACKEND        vllm (default) | hf_native
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
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))

import argparse
import gc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()

# Six paper models. (display_name, HF id, dir_short, slug).
DEFAULT_MODELS: List[Tuple[str, str, str, str]] = [
    ("Llama-3.3-70B",        "meta-llama/Llama-3.3-70B-Instruct",
     "llama_3_3_70b",        "meta-llama-3.3-70b-instruct"),
    ("Magistral-Small-2509", "mistralai/Magistral-Small-2509",
     "magistral_small_2509", "magistral-small-2509"),
    ("Phi-4",                "microsoft/phi-4",
     "phi_4",                "phi-4"),
    ("Qwen3-VL-8B",          "Qwen/Qwen3-VL-8B-Instruct",
     "qwen3_vl_8b",          "qwen3-vl-8b-instruct"),
    ("Qwen2.5-7B",           "Qwen/Qwen2.5-7B-Instruct",
     "qwen2_5_7b",           "qwen2.5-7b-instruct"),
    ("Phi-3.5-mini",         "microsoft/Phi-3.5-mini-instruct",
     "phi_3_5_mini",         "phi-3.5-mini-instruct"),
]

if "R2_MODELS" in os.environ:
    MODELS = []
    for tok in os.environ["R2_MODELS"].split(";"):
        parts = [s.strip() for s in tok.split(",")]
        if len(parts) >= 4:
            MODELS.append(tuple(parts[:4]))
else:
    MODELS = DEFAULT_MODELS

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402

N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "300"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round2/logit_conditioning_cross_model"
    if on_kaggle()
    else "results/exp24_round2/logit_conditioning_cross_model"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    if on_kaggle()
    else "results/exp24_paper_20c"
)


# ─── per-model logit-conditioning sweep ──────────────────────────────────────
def _run_one_model(display_name: str, hf_id: str, dir_short: str) -> None:
    """Vanilla forward pass + per-scenario margin/entropy across countries."""
    install_paper_kaggle_deps()
    from src.logit_conditioning import diagnose_country  # noqa: E402
    from src.model import setup_seeds  # noqa: E402

    setup_seeds(42)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    cfg = build_cfg(hf_id, str(OUT_DIR), COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    model, tokenizer = load_model_timed(hf_id, backend=backend, load_in_4bit=False)

    rows: List[Dict] = []
    for country in COUNTRIES:
        try:
            scen = load_scenarios(cfg, country)
            agg, _ = diagnose_country(model, tokenizer, country, scen, cfg)
            rows.append({
                "model":          display_name,
                "model_short":    dir_short,
                "country":        country,
                **agg,
            })
            print(f"  ✓ {display_name} {country}  margin={agg.get('mean_margin', float('nan')):.3f}  "
                  f"entropy={agg.get('mean_entropy', float('nan')):.3f}")
        except Exception as exc:
            print(f"[error] {display_name} {country}: {exc}")
            rows.append({"model": display_name, "country": country,
                         "error": str(exc)[:500]})

    out_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")

    # cleanup
    del model, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─── post-hoc aggregation against SWA-DPBR per-country MIS gains ─────────────
def _load_per_country_mis(dir_short: str, slug: str) -> Optional[pd.DataFrame]:
    """Find vanilla-vs-SWA-DPBR per-country MIS for this model."""
    base = PAPER_RESULTS_BASE
    candidates = [
        f"{base}/{dir_short}/comparison.csv",
        f"{base}/**/{dir_short}/**/comparison.csv",
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


def _aggregate() -> None:
    """Aggregate per-model logit_cond_*.csv against per-country MIS gains."""
    from scipy.stats import pearsonr, spearmanr

    pooled_rows: List[Dict] = []
    per_model_rows: List[Dict] = []

    for (display_name, _, dir_short, slug) in MODELS:
        cond_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
        if not cond_path.exists():
            print(f"[skip] {display_name}: no {cond_path}")
            continue
        cond = pd.read_csv(cond_path)
        cmp = _load_per_country_mis(dir_short, slug)
        if cmp is None:
            print(f"[skip] {display_name}: no comparison.csv (re-run main {dir_short} first)")
            continue
        # comparison.csv format: method, country, align_mis, ...
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

        # Per-model correlation
        if len(merged) >= 5:
            r,  p  = pearsonr(merged["mean_margin"],  merged["mis_gain_pct"])
            rh, ph = spearmanr(merged["mean_margin"], merged["mis_gain_pct"])
            per_model_rows.append({
                "model":      display_name,
                "n":          len(merged),
                "pearson_r":  round(r, 3),
                "pearson_p":  round(p, 4),
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

    # Pooled correlation
    rp, pp = pearsonr(pooled["mean_margin"],  pooled["mis_gain_pct"])
    rsp, psp = spearmanr(pooled["mean_margin"], pooled["mis_gain_pct"])
    per_model_rows.append({
        "model":      "POOLED (6 x 20)",
        "n":          len(pooled),
        "pearson_r":  round(rp, 3),
        "pearson_p":  round(pp, 4),
        "spearman_rho": round(rsp, 3),
        "spearman_p":   round(psp, 4),
    })
    summary = pd.DataFrame(per_model_rows)
    summary.to_csv(OUT_DIR / "logit_cond_cross_model_correlations.csv", index=False)
    print(f"[saved] {OUT_DIR / 'logit_cond_cross_model_correlations.csv'}")
    print("\n" + summary.to_string(index=False))

    # LaTeX table
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

    _zip_outputs(OUT_DIR, "round2_phase3_logit_cond_cross_model")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip GPU sweep, only aggregate existing logit_cond_*.csv")
    args = parser.parse_args()

    if not args.aggregate_only:
        for (display_name, hf_id, dir_short, _) in MODELS:
            print(f"\n{'#' * 72}\n# {display_name}  ({hf_id})\n{'#' * 72}")
            try:
                _run_one_model(display_name, hf_id, dir_short)
            except Exception as exc:
                print(f"[error] {display_name}: {exc}")

    print(f"\n{'=' * 72}\nAggregating cross-model correlations\n{'=' * 72}")
    _aggregate()


if __name__ == "__main__":
    main()
