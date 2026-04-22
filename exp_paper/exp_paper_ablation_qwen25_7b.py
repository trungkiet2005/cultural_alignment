#!/usr/bin/env python3
"""
EXP-24 Ablation Study — Qwen2.5-7B-Instruct (HF native, bf16)
================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

Reuses all ablation controllers, metrics collection, and reporting from
``exp_paper_ablation_phi4.py``, patching only model path + Kaggle env.

Six ablation configs × 3 countries (USA, JPN, VNM) = 18 runs total.
Adds logit caching to .npz for CPU post-hoc analysis.

Setup (same as exp_paper_kaggle_qwen25_7b.py):
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input
    3. Add multitp-data dataset
    4. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_ablation_qwen25_7b.py

Env overrides:
    ABLATION_COUNTRIES      comma-separated ISO3  (default: USA,JPN,VNM)
    ABLATION_N_SCENARIOS    int                   (default: 500)
    ABLATION_SEED           int                   (default: 42)
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. KAGGLE OFFLINE BOOTSTRAP                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT_DATASET_DIR = "/kaggle/input/datasets/kit567/cultural-alignment"
MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WORK_DIR = "/kaggle/working/cultural_alignment"

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_AUTO_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Backend: HF native bf16 (matches main Qwen2.5 experiment)
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")
# ESS anchor regularisation ON (matches paper §4.2)
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")
# Ablation countries: USA, JPN, VNM for cross-model breadth (Appendix R)
os.environ.setdefault("ABLATION_COUNTRIES", "USA,JPN,VNM")


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    """Copy project from read-only input to writable working dir."""
    if _on_kaggle():
        if os.path.isdir(WORK_DIR) and os.path.isfile(
            os.path.join(WORK_DIR, "src", "controller.py")
        ):
            print(f"[SETUP] Working dir exists: {WORK_DIR}")
        else:
            if not os.path.isdir(PROJECT_DATASET_DIR):
                raise RuntimeError(
                    f"Project dataset not found at {PROJECT_DATASET_DIR}"
                )
            print(f"[SETUP] Copying project → {WORK_DIR} ...")
            shutil.copytree(PROJECT_DATASET_DIR, WORK_DIR, dirs_exist_ok=True)
        os.chdir(WORK_DIR)
        sys.path.insert(0, WORK_DIR)
        return WORK_DIR
    else:
        here = os.getcwd()
        if os.path.isfile(os.path.join(here, "src", "controller.py")):
            sys.path.insert(0, here)
            return here
        raise RuntimeError("Not on Kaggle and not inside repo root.")


def _resolve_model_path() -> str:
    """Resolve local model weights path."""
    if not _on_kaggle():
        # Local dev: use env var or skip
        return os.environ.get("MORAL_MODEL_PATH", MODEL_LOCAL_PATH)
    p = MODEL_LOCAL_PATH
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
        return p
    # Search subdirectories
    for sub in Path(p).rglob("config.json"):
        return str(sub.parent)
    candidates = [
        f"{p}/transformers/default/1",
        f"{p}/pytorch/default/1",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise RuntimeError(f"Model weights not found at {p}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. BOOTSTRAP AND IMPORT                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  KAGGLE OFFLINE — Qwen2.5-7B Ablation Study (HF native, bf16)")
print("  Cross-model breadth: USA × JPN × VNM")
print("=" * 70)

_setup_project()
MODEL_LOCAL_PATH_RESOLVED = _resolve_model_path()
print(f"[SETUP] Model path: {MODEL_LOCAL_PATH_RESOLVED}")

# Offline dep fallback
if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf 2>/dev/null || true",
        shell=True, check=False,
    )

# Now import the phi4 ablation module and patch its globals
import exp_paper.exp_paper_ablation_phi4 as _abl  # noqa: E402

# Patch model identity
_abl.MODEL_NAME = MODEL_LOCAL_PATH_RESOLVED
_abl.MODEL_SHORT = "kaggle_qwen25_7b_bf16"

# Patch results directory
_RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_ablation_qwen25_7b"
    if _on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_ablation_qwen25_7b")
)
_abl.RESULTS_BASE = _RESULTS_BASE

# Patch data paths
_abl.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_abl.WVS_DATA_PATH = WVS_DATA_PATH
_abl.HUMAN_AMCE_PATH = HUMAN_AMCE_PATH

# Patch pre-computed Full SWA-DPBR path (from main qwen25 run)
from src.config import model_slug  # noqa: E402

_KAGGLE_MAIN_RUN = (
    f"/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    f"/kaggle_qwen25_7b_bf16/swa/{model_slug(MODEL_LOCAL_PATH_RESOLVED)}"
)
_abl.FULL_SWA_BASE = os.environ.get("ABLATION_FULL_SWA_BASE") or (
    _KAGGLE_MAIN_RUN if _on_kaggle() else None
)

# Also patch _base_dpbr data paths (used by _build_cfg → SWAConfig)
import exp_model._base_dpbr as _dpbr  # noqa: E402

_dpbr.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_dpbr.WVS_DATA_PATH = WVS_DATA_PATH
_dpbr.HUMAN_AMCE_PATH = HUMAN_AMCE_PATH

print(f"[SETUP] Patched for Qwen2.5-7B ablation:")
print(f"  MODEL  : {_abl.MODEL_NAME}")
print(f"  OUTPUT : {_abl.RESULTS_BASE}")
print(f"  FULL   : {_abl.FULL_SWA_BASE}")
print(f"  MULTITP: {MULTITP_DATA_PATH}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. LOGIT CACHING — saves intermediate logit gaps to .npz                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LogitCache:
    """Accumulates per-scenario logit diagnostics for CPU post-hoc analysis."""

    def __init__(self):
        self._entries: List[Dict] = []

    def record(self, scenario_idx: int, result: Dict, category: str = ""):
        """Record relevant fields from a controller predict() result dict."""
        entry = {
            "scenario_idx": scenario_idx,
            "category": category,
            "delta_consensus": result.get("delta_consensus", float("nan")),
            "delta_opt_micro": result.get("delta_opt_micro", float("nan")),
            "delta_opt": result.get("delta_opt", float("nan")),
            "delta_star_1": result.get("delta_star_1", float("nan")),
            "delta_star_2": result.get("delta_star_2", float("nan")),
            "bootstrap_var": result.get("bootstrap_var", float("nan")),
            "reliability_r": result.get("reliability_r", float("nan")),
            "ess_pass1": result.get("ess_pass1", float("nan")),
            "ess_pass2": result.get("ess_pass2", float("nan")),
            "ess_anchor_alpha": result.get("ess_anchor_alpha", float("nan")),
            "positional_bias": result.get("positional_bias", float("nan")),
            "sigma_used": result.get("sigma_used", float("nan")),
            "logit_temp_used": result.get("logit_temp_used", float("nan")),
            "p_spare_preferred": result.get("p_spare_preferred", float("nan")),
            "mppi_flipped": bool(result.get("mppi_flipped", False)),
            "n_personas": int(result.get("n_personas", 0)),
        }
        # Per-persona logit gaps (variable length → store as separate array)
        gaps = result.get("agent_decision_gaps", [])
        if gaps:
            entry["agent_gaps"] = np.array(gaps, dtype=np.float32)
        rewards = result.get("agent_rewards", [])
        if rewards:
            entry["agent_rewards"] = np.array(rewards, dtype=np.float32)
        self._entries.append(entry)

    def save(self, path: str):
        """Flush to compressed .npz for CPU-side analysis."""
        if not self._entries:
            return
        arrays = {}
        n = len(self._entries)
        # Scalar fields → 1-D arrays
        scalar_keys = [
            "scenario_idx", "delta_consensus", "delta_opt_micro", "delta_opt",
            "delta_star_1", "delta_star_2", "bootstrap_var", "reliability_r",
            "ess_pass1", "ess_pass2", "ess_anchor_alpha", "positional_bias",
            "sigma_used", "logit_temp_used", "p_spare_preferred", "n_personas",
        ]
        for k in scalar_keys:
            dtype = np.int32 if k in ("scenario_idx", "n_personas") else np.float32
            arrays[k] = np.array([e.get(k, 0) for e in self._entries], dtype=dtype)
        arrays["mppi_flipped"] = np.array(
            [e.get("mppi_flipped", False) for e in self._entries], dtype=np.bool_
        )
        # Per-persona gaps: ragged → store per-scenario with prefix
        for i, e in enumerate(self._entries):
            if "agent_gaps" in e:
                arrays[f"s{i:04d}_agent_gaps"] = e["agent_gaps"]
            if "agent_rewards" in e:
                arrays[f"s{i:04d}_agent_rewards"] = e["agent_rewards"]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)
        sz = os.path.getsize(path)
        print(f"  [LOGIT] Saved {n} scenarios → {path} ({sz / 1024:.1f} KB)")

    def clear(self):
        self._entries.clear()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. PATCHED MAIN — wraps phi4 main() with logit caching hooks              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
)
from src.amce import (  # noqa: E402
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    compute_utilitarianism_slope,
    load_human_amce,
)
from src.config import SWAConfig  # noqa: E402
from src.constants import COUNTRY_LANG  # noqa: E402
from src.data import load_multitp_dataset  # noqa: E402
from src.model import load_model_hf_native, setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402


def main() -> None:
    seed = int(os.environ.get("ABLATION_SEED", "42"))
    n_scenarios = int(os.environ.get("ABLATION_N_SCENARIOS", "500"))
    countries = [
        c.strip()
        for c in os.environ.get("ABLATION_COUNTRIES", "USA,JPN,VNM").split(",")
        if c.strip()
    ]

    setup_seeds(seed)

    print(f"\n{'#' * 80}")
    print(f"  EXP-24 Ablation — Qwen2.5-7B-Instruct (bf16, HF native)")
    print(f"  Countries : {countries}")
    print(f"  Scenarios : {n_scenarios}  |  Seed: {seed}")
    print(f"  DPBR      : K_HALF={K_HALF}×2={K_HALF*2}  VAR_SCALE={VAR_SCALE}")
    print(f"  Ablations : {len(_abl.ABLATION_SPECS)} configurations")
    print(f"  Output    : {_abl.RESULTS_BASE}")
    print(f"{'#' * 80}\n")

    out_dir = Path(_abl.RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)
    logit_dir = out_dir / "logits"
    logit_dir.mkdir(exist_ok=True)

    # ── Load model ONCE ───────────────────────────────────────────────────────
    print(f"[LOAD] Loading {_abl.MODEL_NAME} via HF native (bf16)...")
    model, tokenizer = load_model_hf_native(
        _abl.MODEL_NAME, max_seq_length=2048, load_in_4bit=False
    )

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    all_rows: List[Dict] = []
    logit_cache = LogitCache()

    # ── Per-country loop ──────────────────────────────────────────────────────
    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country}: not in SUPPORTED_COUNTRIES")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Country: {country}")
        print("=" * 80)

        scenario_df = _abl._load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        human_amce = load_human_amce(HUMAN_AMCE_PATH, country)

        print(f"  {len(scenario_df)} scenarios | {len(personas)} personas"
              f" | {len(human_amce)} AMCE dims")

        # ── Per-ablation inner loop ───────────────────────────────────────────
        for spec_idx, spec in enumerate(_abl.ABLATION_SPECS):
            print(f"\n  {'─' * 70}")
            print(f"  [{spec_idx}/{len(_abl.ABLATION_SPECS)-1}]"
                  f"  {spec.row_label}  —  {spec.description}")
            print(f"  {'─' * 70}")

            _abl._reset_prior_state(country)
            torch.cuda.empty_cache()
            gc.collect()

            t0 = time.time()

            # Try loading pre-computed Full SWA-DPBR results
            precomp = (
                _abl._find_full_swa_csv(country)
                if spec.row_label == "Full SWA-DPBR"
                else None
            )
            if precomp is not None:
                print(f"  [FULL] Pre-computed: {precomp}")
                results_df = pd.read_csv(precomp)
                summary = _abl._reconstruct_summary(results_df, country, cfg)
            else:
                results_df, summary = _abl._run_ablation_country(
                    spec, model, tokenizer, country, personas, scenario_df, cfg
                )

            elapsed = time.time() - t0

            # ── Save per-ablation CSV ─────────────────────────────────────────
            safe_tag = (
                spec.row_label.lower()
                .replace(" ", "_").replace("(", "").replace(")", "")
                .replace("=", "eq").replace(",", "").replace("α", "a")
                .replace("/", "_")
            )
            results_df.to_csv(
                out_dir / f"{country}_{safe_tag}_results.csv", index=False
            )

            # ── Logit caching ─────────────────────────────────────────────────
            logit_cache.clear()
            for idx, row_data in results_df.iterrows():
                entry = {
                    "delta_consensus": row_data.get("delta_consensus", float("nan")),
                    "delta_opt_micro": row_data.get("delta_opt_micro", float("nan")),
                    "delta_opt": row_data.get("delta_opt", float("nan")),
                    "delta_star_1": row_data.get("delta_star_1", float("nan")),
                    "delta_star_2": row_data.get("delta_star_2", float("nan")),
                    "bootstrap_var": row_data.get("bootstrap_var", float("nan")),
                    "reliability_r": row_data.get("reliability_r", float("nan")),
                    "ess_pass1": row_data.get("ess_pass1", float("nan")),
                    "ess_pass2": row_data.get("ess_pass2", float("nan")),
                    "ess_anchor_alpha": row_data.get("ess_anchor_alpha", float("nan")),
                    "positional_bias": row_data.get("positional_bias", float("nan")),
                    "sigma_used": row_data.get("sigma_used", float("nan")),
                    "logit_temp_used": row_data.get("logit_temp_used", float("nan")),
                    "p_spare_preferred": row_data.get("p_spare_preferred", float("nan")),
                    "mppi_flipped": bool(row_data.get("mppi_flipped", False)),
                    "n_personas": int(row_data.get("n_personas", 0)),
                    "agent_decision_gaps": [],
                }
                # Parse agent_decision_gaps from CSV string repr
                raw_gaps = row_data.get("agent_decision_gaps", "")
                if isinstance(raw_gaps, str) and raw_gaps.startswith("["):
                    try:
                        import ast
                        entry["agent_decision_gaps"] = ast.literal_eval(raw_gaps)
                    except Exception:
                        pass
                elif isinstance(raw_gaps, (list, np.ndarray)):
                    entry["agent_decision_gaps"] = list(raw_gaps)
                logit_cache.record(
                    int(idx),
                    entry,
                    category=str(row_data.get("phenomenon_category", "")),
                )
            logit_cache.save(str(logit_dir / f"{country}_{safe_tag}_logits.npz"))

            # ── Collect metrics row ───────────────────────────────────────────
            row = _abl._collect_row(spec, country, results_df, summary, elapsed)
            all_rows.append(row)

            a = summary.get("alignment", {})
            print(
                f"\n  ✓  {spec.row_label} | {country}"
                f"  JSD={a.get('jsd',float('nan')):.4f}"
                f"  r={a.get('pearson_r',float('nan')):+.3f}"
                f"  MIS={a.get('mis',float('nan')):.4f}"
                f"  ({elapsed:.0f}s)"
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    summary_csv = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SAVED] {summary_csv}")

    # ── Print reports ─────────────────────────────────────────────────────────
    for country in countries:
        print(f"\n\n{'#' * 80}")
        print(f"  FINAL REPORT — {country}")
        print(f"{'#' * 80}")
        _abl.print_ablation_table(all_rows, country)
        _abl.print_per_dim_table(all_rows, country)
        _abl.print_per_dim_signed(all_rows, country)
        _abl.print_dpbr_diagnostics(all_rows, country)
        _abl.print_util_slopes(all_rows, country)

    if len(countries) > 1:
        _abl.print_cross_country_summary(summary_df)

    # LaTeX for first country
    _abl._print_latex_table(all_rows, countries[0])

    print(f"\n{'#' * 80}")
    print(f"  Ablation COMPLETE — Qwen2.5-7B-Instruct")
    print(f"  Results : {out_dir}")
    print(f"  Logits  : {logit_dir}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
