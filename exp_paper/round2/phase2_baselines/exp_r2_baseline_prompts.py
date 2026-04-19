#!/usr/bin/env python3
"""Round-2 prompt-only baselines (B1, B2, B3-short, B3-long, B4).

Runs the prompt-prefix baselines defined in :mod:`src.prompt_baselines` on
Phi-4 (14B) across the 20 paper countries. Each variant prepends a
cultural framing string to every scenario, then runs the same A/B
logit-gap inference as the vanilla baseline.

  • B1        Country-Tailored Prompt
  • B2        WVS Profile Prompting
  • B3_short  PRISM-Style Prompting (one-sentence prefix)
  • B3_long   PRISM-Style Prompting (three-sentence prefix)
  • B4        Country + WVS Profile  (B1 ∪ B2 — strongest prompt-only)

These are the natural counterparts to SWA-DPBR's persona ensemble: they
ask the model to speak as country X without ever supplying the
within-country disagreement signal that SWA-DPBR exploits.

Kaggle:
    !python exp_paper/round2/phase2_baselines/exp_r2_baseline_prompts.py

Env overrides:
    R2_BASELINES       comma list (default: B1,B2,B3_short,B3_long,B4)
    R2_MODEL           HF id (default: microsoft/phi-4)
    R2_COUNTRIES       comma-separated ISO3 (default: 20 paper countries)
    R2_N_SCENARIOS     per-country (default: 500)
    R2_BACKEND         vllm (default) | hf_native
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap (same pattern as the other phase-2 baselines).
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
            "Not on Kaggle and not inside the repo root."
        )
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))

import gc
import os
import time
from pathlib import Path
from typing import Dict, List

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.model import setup_seeds  # noqa: E402
from src.prompt_baseline_runner import run_prompt_baseline_country  # noqa: E402


MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
BASELINES = [
    b.strip()
    for b in os.environ.get("R2_BASELINES", "B1,B2,B3_short,B3_long,B4").split(",")
    if b.strip()
]
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/prompt_baselines"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "prompt_baselines")
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    if on_kaggle()
    else "WVS_data/country_specific_ACME.csv"
)
WVS_CSV_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    if on_kaggle()
    else "WVS_data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    scen_cache: Dict[str, pd.DataFrame] = {c: load_scenarios(cfg, c) for c in COUNTRIES}

    rows: List[Dict] = []
    for baseline in BASELINES:
        print(f"\n{'#' * 72}\n# Prompt baseline: {baseline}\n{'#' * 72}")
        for country in COUNTRIES:
            scen = scen_cache.get(country)
            if scen is None or scen.empty:
                print(f"[skip] {country} — no scenarios")
                continue
            t0 = time.time()
            try:
                out = run_prompt_baseline_country(
                    model, tokenizer, scen, country, cfg,
                    baseline=baseline,
                    wvs_csv_path=WVS_CSV_PATH,
                    human_amce_path=HUMAN_AMCE_PATH,
                )
                summary = out["summary"]
                rows.append(summary)
                # Per-country results CSV (matches diffpo / mc_dropout layout).
                out["results_df"].to_csv(
                    out_dir / f"prompt_{baseline}_results_{country}.csv", index=False)
                pd.DataFrame(rows).to_csv(out_dir / "prompt_baselines_partial.csv", index=False)
                mis = summary.get("mis", float("nan"))
                r   = summary.get("pearson_r", float("nan"))
                print(f"  ✓ {baseline:<10s} {country}  MIS={mis:.4f}  r={r:+.3f}  "
                      f"({time.time()-t0:.0f}s)")
            except Exception as exc:
                print(f"[error] {baseline} {country}: {exc}")
                rows.append({"baseline": baseline, "country": country,
                             "error": str(exc)[:500]})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "prompt_baselines_summary.csv")
    _zip_outputs(out_dir, "round2_phase2_prompt_baselines")


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
