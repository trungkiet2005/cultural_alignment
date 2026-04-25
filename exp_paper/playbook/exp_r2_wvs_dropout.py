#!/usr/bin/env python3
"""Round-2 Reviewer W9 -- WVS-to-trolley dimension linkage via dropout.

For each WVS dimension ``d`` in :data:`src.personas.WVS_DIMS`, build a persona
panel with ``d`` omitted and re-run SWA-DPBR. Track per-MultiTP-dimension AMCE
error so we can see which trolley dimension each WVS dim is actually loading.

Output per (dropped_dim, country): MIS, JSD, per-MultiTP-dim |model-human|.
Cross-tabbed in post-processing to identify causally-implicated pairs.

Kaggle:
    !python exp_paper/review/round2/phase3_sensitivity/exp_r2_wvs_dropout.py

Env overrides:
    R2_MODEL          (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS    (default: 250)
    R2_DROP_SET       comma list from WVS_DIMS keys (default: all 10; we always
                      also run a "∅" control with no drop)
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
# Set backend BEFORE paper_runtime is imported so install_paper_kaggle_deps()
# picks the correct pip branch (vLLM vs Unsloth).
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

from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, WVS_DIMS, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "250"))
COUNTRIES = [
    c.strip()
    for c in os.environ.get("R2_COUNTRIES", "USA,VNM,DEU").split(",")
    if c.strip()
]
DROP_SET = os.environ.get(
    "R2_DROP_SET",
    ",".join(WVS_DIMS.keys()),
)
DROPS: List[str] = [s.strip() for s in DROP_SET.split(",") if s.strip()]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/wvs_dropout"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "wvs_dropout")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _flatten_perdim(per_dim: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, d in per_dim.items():
        out[f"abserr_{k}"]   = float(d.get("abs_err", float("nan")))
        out[f"signerr_{k}"]  = float(d.get("signed",  float("nan")))
    return out


def _run_dropout(model, tokenizer, cfg, country, scen, drop_key: str) -> Dict:
    """drop_key is either '∅' (control) or a single WVS dim name."""
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    patch_swa_runner_controller()

    drop_dims = set() if drop_key == "∅" else {drop_key}
    personas = build_country_personas(country, wvs_path=WVS_PATH, drop_dims=drop_dims)

    t0 = time.time()
    results_df, summary = run_country_experiment(
        model, tokenizer, country, personas, scen, cfg,
    )
    dt = time.time() - t0

    a = summary["alignment"]
    row = {
        "drop_dim":   drop_key,
        "country":    country,
        "n_scenarios": len(results_df),
        "elapsed_sec": dt,
        "mis":        a.get("mis",         float("nan")),
        "jsd":        a.get("jsd",         float("nan")),
        "pearson_r":  a.get("pearson_r",   float("nan")),
        "spearman_rho": a.get("spearman_rho", float("nan")),
        "mae":        a.get("mae",         float("nan")),
        "rmse":       a.get("rmse",        float("nan")),
        "flip_rate":  summary.get("flip_rate", float("nan")),
    }
    row.update(_flatten_perdim(summary.get("per_dimension_alignment", {})))
    return row


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(
        MODEL_NAME, RESULTS_BASE, COUNTRIES,
        n_scenarios=N_SCEN, load_in_4bit=False,
    )
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(
        MODEL_NAME, backend=backend, load_in_4bit=False,
    )

    scen_cache: Dict[str, pd.DataFrame] = {}
    for c in COUNTRIES:
        if c not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {c}: not in SUPPORTED_COUNTRIES")
            continue
        scen_cache[c] = load_scenarios(cfg, c)

    # Run the ∅ control first, then each drop.
    drop_keys = ["∅"] + DROPS
    rows: List[Dict] = []
    for drop_key in drop_keys:
        print(f"\n{'#' * 72}\n# drop_dim={drop_key}\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                row = _run_dropout(model, tokenizer, cfg, country, scen_cache[country], drop_key)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "wvs_dropout_partial.csv", index=False)
            except Exception as exc:
                print(f"[ERROR] drop={drop_key} {country}: {exc}")
                rows.append({"drop_dim": drop_key, "country": country, "error": str(exc)[:500]})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "wvs_dropout_summary.csv")


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE if 'RESULTS_BASE' in globals() else OUT_DIR if 'OUT_DIR' in globals() else '.')
    except Exception as _e:
        print(f'[ZIP] failed: {_e}')
