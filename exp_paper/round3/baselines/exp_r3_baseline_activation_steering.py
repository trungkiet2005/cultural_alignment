#!/usr/bin/env python3
"""Round-2 baseline: Activation Steering (Arditi et al., NeurIPS 2024 style).

Computes a per-country cultural steering vector from contrastive WVS-pole
prompt pairs and applies it at inference via a forward hook on the
residual stream of layer ``R2_AS_LAYER`` (default 32 — transformer
midpoint for Phi-4 / Llama-3 sized models).

Hard requirement: hf_native backend (vLLM / Unsloth do not expose
Python forward hooks). Script forces ``MORAL_MODEL_BACKEND=hf_native``
unless the user overrides.

Kaggle (Phi-4 14B × 20 countries × 500 scenarios ≈ 1.5–2 h on H100):
    !python exp_paper/round3/baselines/exp_r3_baseline_activation_steering.py

Env overrides:
    R2_MODEL           HF id (default: microsoft/phi-4)
    R2_COUNTRIES       comma-separated ISO3 (default: 20 paper countries)
    R2_N_SCENARIOS     per-country (default: 500)
    R2_AS_LAYER        residual layer to inject at (default: 32)
    R2_AS_ALPHA        scaling coefficient (default: 1.5)
    R2_BACKEND         hf_native (default, required) | (anything else errors)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap.
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
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

# Activation steering MUST run on hf_native; force it unless the user has
# explicitly set the backend to something else (in which case load will fail).
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "hf_native"))

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
from src.activation_steering import run_activation_steering_country  # noqa: E402
from src.model import setup_seeds  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN     = int(os.environ.get("R2_N_SCENARIOS", "500"))
LAYER_IDX  = int(os.environ.get("R2_AS_LAYER", "32"))
ALPHA      = float(os.environ.get("R2_AS_ALPHA", "1.5"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/activation_steering"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "activation_steering")
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
    model, tokenizer = load_model_timed(MODEL_NAME, backend="hf_native",
                                        load_in_4bit=False)

    scen_cache: Dict[str, pd.DataFrame] = {c: load_scenarios(cfg, c) for c in COUNTRIES}

    rows: List[Dict] = []
    print(f"\n{'#'*72}\n# Activation steering: layer={LAYER_IDX}, alpha={ALPHA}\n{'#'*72}")
    for country in COUNTRIES:
        scen = scen_cache.get(country)
        if scen is None or scen.empty:
            print(f"[skip] {country} — no scenarios")
            continue
        t0 = time.time()
        try:
            out = run_activation_steering_country(
                model, tokenizer, scen, country, cfg,
                layer_idx=LAYER_IDX, alpha=ALPHA,
                wvs_csv_path=WVS_CSV_PATH,
                human_amce_path=HUMAN_AMCE_PATH,
            )
            summary = out["summary"]
            rows.append(summary)
            out["results_df"].to_csv(
                out_dir / f"activation_steering_results_{country}.csv", index=False)
            pd.DataFrame(rows).to_csv(out_dir / "activation_steering_partial.csv", index=False)
            mis = summary.get("mis", float("nan"))
            r   = summary.get("pearson_r", float("nan"))
            print(f"  ✓ {country}  MIS={mis:.4f}  r={r:+.3f}  ({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"[error] {country}: {exc}")
            rows.append({"method": "activation_steering", "country": country,
                         "error": str(exc)[:500]})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary(rows, out_dir, "activation_steering_summary.csv")
    _zip_outputs(out_dir, "round3_baselines_activation_steering")


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
