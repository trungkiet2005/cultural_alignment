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
    ABLATION_SEEDS          comma-separated ints  (optional; e.g., 11,22,33)
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


PRIMARY_METRICS = ["jsd", "pearson_r", "spearman_rho", "mae", "rmse", "mis"]
USE_REPLAY_LOGITS = os.environ.get("ABLATION_REPLAY_LOGITS", "1").strip() != "0"

# Cache key: (prompt_text, phenomenon_category, lang)
_ACTIVE_REPLAY_CACHE: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None
_ORIG_EXTRACT_GAPS = _abl.Exp24DualPassController._extract_logit_gaps


def _cache_key(user_query: str, phenomenon_category: str, lang: str) -> Tuple[str, str, str]:
    return (str(user_query), str(phenomenon_category), str(lang))


def _set_active_replay_cache(cache: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]]) -> None:
    global _ACTIVE_REPLAY_CACHE
    _ACTIVE_REPLAY_CACHE = cache


def _tensor_to_numpy_float32(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to CPU numpy float32 safely (handles bfloat16)."""
    return x.detach().to(dtype=torch.float32).cpu().numpy()


@torch.no_grad()
def _extract_logit_gaps_replay(self, user_query: str, phenomenon_category: str, lang: str):
    """Patched extractor: replay from cache when available, fallback otherwise."""
    key = _cache_key(user_query, phenomenon_category, lang)
    if _ACTIVE_REPLAY_CACHE is not None and key in _ACTIVE_REPLAY_CACHE:
        rec = _ACTIVE_REPLAY_CACHE[key]
        db = torch.tensor(float(rec["db"]), device=self.device, dtype=torch.float32)
        da = torch.tensor(rec["da"], device=self.device, dtype=torch.float32)
        return db, da, float(rec["logit_temp"])

    db, da, logit_temp = _ORIG_EXTRACT_GAPS(self, user_query, phenomenon_category, lang)
    if _ACTIVE_REPLAY_CACHE is not None:
        _ACTIVE_REPLAY_CACHE[key] = {
            "db": float(db.item()),
            "da": _tensor_to_numpy_float32(da),
            "logit_temp": float(logit_temp),
        }
    return db, da, logit_temp


def _install_replay_patch() -> None:
    """Monkey-patch base controller so all ablations can replay cached logits."""
    _abl.Exp24DualPassController._extract_logit_gaps = _extract_logit_gaps_replay


def _build_country_logit_replay_cache(
    model,
    tokenizer,
    cfg: SWAConfig,
    country: str,
    personas: List[str],
    scenario_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Build one shared cache for all ablations:
    - base extraction on original prompt
    - base extraction on swapped prompt (for debiasing path)
    """
    lang = COUNTRY_LANG.get(country, "en")
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    controller = _abl.Exp24DualPassController(
        model=model,
        tokenizer=tokenizer,
        personas=personas,
        lambda_coop=cfg.lambda_coop,
        alpha_ctl=cfg.alpha_ctl,
        K_samples=cfg.K_samples,
        noise_std=cfg.noise_std,
        temperature=cfg.temperature,
        logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=cfg.category_logit_temperatures,
        pt_alpha=cfg.pt_alpha,
        pt_beta=cfg.pt_beta,
        pt_kappa=cfg.pt_kappa,
        decision_temperature=cfg.decision_temperature,
        assistant_lang=lang,
        country_iso=country,
    )

    print(f"[REPLAY] Building logit cache for {country} ({len(scenario_df)} scenarios)...")
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        cat = str(row.get("phenomenon_category", "default"))

        k1 = _cache_key(prompt, cat, lang)
        if k1 not in cache:
            db1, da1, lt1 = _ORIG_EXTRACT_GAPS(controller, prompt, cat, lang)
            cache[k1] = {
                "db": float(db1.item()),
                "da": _tensor_to_numpy_float32(da1),
                "logit_temp": float(lt1),
            }

        swapped_prompt, swap_changed = controller._swap_positional_labels(prompt, lang)
        if swap_changed:
            k2 = _cache_key(swapped_prompt, cat, lang)
            if k2 not in cache:
                db2, da2, lt2 = _ORIG_EXTRACT_GAPS(controller, swapped_prompt, cat, lang)
                cache[k2] = {
                    "db": float(db2.item()),
                    "da": _tensor_to_numpy_float32(da2),
                    "logit_temp": float(lt2),
                }

    print(f"[REPLAY] Cache ready for {country}: {len(cache)} prompt entries")
    del controller
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def _parse_seeds() -> List[int]:
    """Parse ABLATION_SEEDS (comma list) or fallback to ABLATION_SEED."""
    raw = os.environ.get("ABLATION_SEEDS", "").strip()
    if raw:
        seeds: List[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            seeds.append(int(tok))
        if seeds:
            return seeds
    return [int(os.environ.get("ABLATION_SEED", "42"))]


def _build_multi_seed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(country,ablation) metrics across seeds.
    Returns one row per country × ablation with mean/std columns.
    """
    if summary_df.empty:
        return pd.DataFrame()

    agg_cols = [
        "jsd", "pearson_r", "spearman_rho", "mae", "rmse", "mis",
        "flip_rate", "mean_reliability_r", "elapsed_sec",
    ]
    grouped = (
        summary_df.groupby(["country", "ablation"], as_index=False)[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    return grouped


def _build_delta_vs_full(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seed deltas vs Full SWA-DPBR, then aggregate across seeds.
    Useful for reporting stability of each ablation effect.
    """
    if summary_df.empty or "seed" not in summary_df.columns:
        return pd.DataFrame()

    key_cols = ["seed", "country"]
    full = summary_df.loc[summary_df["ablation"] == "Full SWA-DPBR", key_cols + PRIMARY_METRICS]
    if full.empty:
        return pd.DataFrame()

    merged = summary_df.merge(
        full,
        on=key_cols,
        how="left",
        suffixes=("", "_full"),
    )
    for m in PRIMARY_METRICS:
        merged[f"delta_{m}_vs_full"] = merged[m] - merged[f"{m}_full"]

    delta_cols = [f"delta_{m}_vs_full" for m in PRIMARY_METRICS]
    out = (
        merged.groupby(["country", "ablation"], as_index=False)[delta_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    out.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in out.columns
    ]

    jsd_wins = (
        merged.assign(jsd_win=lambda d: d["delta_jsd_vs_full"] < 0.0)
        .groupby(["country", "ablation"], as_index=False)
        .agg(jsd_wins=("jsd_win", "sum"), n_seeds=("jsd_win", "count"))
    )
    out = out.merge(jsd_wins, on=["country", "ablation"], how="left")
    return out


def _print_multi_seed_snapshot(stability_df: pd.DataFrame) -> None:
    """Compact console view: mean±std of ΔJSD and ΔMIS vs Full."""
    if stability_df.empty:
        return
    print(f"\n{'=' * 80}")
    print("  Multi-seed stability vs Full SWA-DPBR")
    print("=" * 80)
    print(
        f"  {'Country':<8} {'Ablation':<30} "
        f"{'ΔJSD mean±std':>18} {'ΔMIS mean±std':>18} {'JSD wins':>10}"
    )
    print("  " + "-" * 76)
    for _, row in stability_df.iterrows():
        if row["ablation"] == "Full SWA-DPBR":
            continue
        djsd_m = row.get("delta_jsd_vs_full_mean", float("nan"))
        djsd_s = row.get("delta_jsd_vs_full_std", float("nan"))
        dmis_m = row.get("delta_mis_vs_full_mean", float("nan"))
        dmis_s = row.get("delta_mis_vs_full_std", float("nan"))
        wins = int(row.get("jsd_wins", 0))
        n_seeds = int(row.get("n_seeds", 0))
        print(
            f"  {str(row.get('country','')):<8} {str(row.get('ablation','')):<30} "
            f"{djsd_m:+.4f}±{djsd_s:.4f} {dmis_m:+.4f}±{dmis_s:.4f} "
            f"{wins:>2}/{n_seeds:<2}"
        )


def main() -> None:
    seeds = _parse_seeds()
    n_scenarios = int(os.environ.get("ABLATION_N_SCENARIOS", "500"))
    countries = [
        c.strip()
        for c in os.environ.get("ABLATION_COUNTRIES", "USA,JPN,VNM").split(",")
        if c.strip()
    ]

    print(f"\n{'#' * 80}")
    print(f"  EXP-24 Ablation — Qwen2.5-7B-Instruct (bf16, HF native)")
    print(f"  Countries : {countries}")
    print(f"  Scenarios : {n_scenarios}")
    print(f"  Seeds     : {seeds}")
    print(f"  DPBR      : K_HALF={K_HALF}×2={K_HALF*2}  VAR_SCALE={VAR_SCALE}")
    print(f"  Ablations : {len(_abl.ABLATION_SPECS)} configurations")
    print(f"  Replay    : {'ON' if USE_REPLAY_LOGITS else 'OFF'} (ABLATION_REPLAY_LOGITS)")
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
    country_assets: Dict[str, Dict[str, Any]] = {}

    # ── Load country data once and prebuild replay caches ─────────────────────
    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country}: not in SUPPORTED_COUNTRIES")
            continue
        scenario_df = _abl._load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        human_amce = load_human_amce(HUMAN_AMCE_PATH, country)
        asset: Dict[str, Any] = {
            "scenario_df": scenario_df,
            "personas": personas,
            "human_amce_n": len(human_amce),
            "replay_cache": None,
        }
        if USE_REPLAY_LOGITS:
            asset["replay_cache"] = _build_country_logit_replay_cache(
                model, tokenizer, cfg, country, personas, scenario_df
            )
        country_assets[country] = asset

    if USE_REPLAY_LOGITS:
        _install_replay_patch()

    # ── Per-seed × country loop ───────────────────────────────────────────────
    for seed in seeds:
        setup_seeds(seed)
        print(f"\n{'#' * 80}")
        print(f"  Seed run: {seed}")
        print(f"{'#' * 80}")

        for country in countries:
            if country not in country_assets:
                continue

            print(f"\n{'=' * 80}")
            print(f"  Country: {country} | Seed: {seed}")
            print("=" * 80)

            scenario_df = country_assets[country]["scenario_df"]
            personas = country_assets[country]["personas"]
            human_amce_n = int(country_assets[country]["human_amce_n"])
            _set_active_replay_cache(country_assets[country]["replay_cache"])

            print(f"  {len(scenario_df)} scenarios | {len(personas)} personas"
                  f" | {human_amce_n} AMCE dims")

            # ── Per-ablation inner loop ───────────────────────────────────────
            for spec_idx, spec in enumerate(_abl.ABLATION_SPECS):
                print(f"\n  {'─' * 70}")
                print(f"  [{spec_idx}/{len(_abl.ABLATION_SPECS)-1}]"
                      f"  {spec.row_label}  —  {spec.description}")
                print(f"  {'─' * 70}")

                _abl._reset_prior_state(country)
                torch.cuda.empty_cache()
                gc.collect()

                t0 = time.time()

                # For multi-seed reproducibility, re-run Full row instead of
                # reusing one precomputed CSV shared across seeds.
                can_use_precomp_full = len(seeds) == 1
                precomp = (
                    _abl._find_full_swa_csv(country)
                    if (spec.row_label == "Full SWA-DPBR" and can_use_precomp_full)
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

                # ── Save per-ablation CSV ─────────────────────────────────────
                safe_tag = (
                    spec.row_label.lower()
                    .replace(" ", "_").replace("(", "").replace(")", "")
                    .replace("=", "eq").replace(",", "").replace("α", "a")
                    .replace("/", "_")
                )
                results_df.to_csv(
                    out_dir / f"seed{seed}_{country}_{safe_tag}_results.csv", index=False
                )

                # ── Logit caching ─────────────────────────────────────────────
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
                logit_cache.save(str(logit_dir / f"seed{seed}_{country}_{safe_tag}_logits.npz"))

                # ── Collect metrics row ───────────────────────────────────────
                row = _abl._collect_row(spec, country, results_df, summary, elapsed)
                row["seed"] = seed
                all_rows.append(row)

                a = summary.get("alignment", {})
                print(
                    f"\n  ✓  {spec.row_label} | {country} | seed={seed}"
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
    summary_csv = out_dir / "ablation_summary_all_seeds.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SAVED] {summary_csv}")

    # Backward compatibility: single-seed default filename
    if len(seeds) == 1:
        compat_csv = out_dir / "ablation_summary.csv"
        summary_df.to_csv(compat_csv, index=False)
        print(f"[SAVED] {compat_csv}")

    multi_seed_df = _build_multi_seed_summary(summary_df)
    if not multi_seed_df.empty:
        multi_seed_csv = out_dir / "ablation_summary_seed_agg.csv"
        multi_seed_df.to_csv(multi_seed_csv, index=False)
        print(f"[SAVED] {multi_seed_csv}")

    stability_df = _build_delta_vs_full(summary_df)
    if not stability_df.empty:
        stability_csv = out_dir / "ablation_delta_vs_full_seed_agg.csv"
        stability_df.to_csv(stability_csv, index=False)
        print(f"[SAVED] {stability_csv}")
        _print_multi_seed_snapshot(stability_df)

    # ── Print reports ─────────────────────────────────────────────────────────
    # For detailed per-country report, use the first seed as representative.
    report_seed = seeds[0]
    report_rows = [r for r in all_rows if int(r.get("seed", report_seed)) == report_seed]
    report_df = pd.DataFrame(report_rows)
    for country in countries:
        print(f"\n\n{'#' * 80}")
        print(f"  FINAL REPORT — {country} (seed={report_seed})")
        print(f"{'#' * 80}")
        _abl.print_ablation_table(report_rows, country)
        _abl.print_per_dim_table(report_rows, country)
        _abl.print_per_dim_signed(report_rows, country)
        _abl.print_dpbr_diagnostics(report_rows, country)
        _abl.print_util_slopes(report_rows, country)

    if len(countries) > 1:
        _abl.print_cross_country_summary(report_df)

    # LaTeX table for first country from representative seed
    _abl._print_latex_table(report_rows, countries[0])

    print(f"\n{'#' * 80}")
    print(f"  Ablation COMPLETE — Qwen2.5-7B-Instruct")
    print(f"  Results : {out_dir}")
    print(f"  Logits  : {logit_dir}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
