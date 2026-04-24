#!/usr/bin/env python3
"""
DISCA Playbook Runner (Qwen2.5-7B, Kaggle offline style)
=========================================================

New standalone script for playbook experiments, keeping setup/path style
consistent with exp_paper_ablation_qwen25_7b.py.

Implemented now:
  - Experiment 1 : scenario-level disagreement vs correction logging + plot
  - Experiment 2 : country-level mean variance vs MIS improvement plot
  - Experiment 3 : multi-seed confidence intervals
  - Experiment 4 : tail-safety analysis (Step 3 defense)
  - Experiment 5 : strong baselines (vanilla / WVS prompt / MC-dropout /
                    temp scaling / DiffPO-binary) vs DISCA
  - Experiment 6 : 3x3 ablation grid (models x countries x ablations)
  - Experiment 7 : predictive failure model (regress delta-MIS on vanilla
                    stats: decision margin, logit entropy, vanilla MIS)
  - Experiment 8 : N-persona sensitivity (N in {1,2,3,4})
  - Experiment 9 : negative Pearson r diagnosis (per-dimension rank swaps)
  - Experiment 10: reliability weight histogram (uses Exp 1 CSV)
  - Experiment 11: per-dimension improvement breakdown (vanilla vs DISCA)
  - Experiment 12: WVS dimension dropout (leave-one-out over 10 dims)

Usage on Kaggle:

  !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_disca_playbook_qwen25_7b.py

Configuration: edit the CONFIG constants block right below the imports.
No CLI args, no env vars — just tweak the values and re-run the cell.
"""

from __future__ import annotations

import ast
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# 0) CONFIG — edit these constants directly, no CLI flags, no env vars.
# ============================================================================
# Which experiment to run: "1".."12" or "all".
# "all" runs every experiment; Exp 2/7/9/11 skip if their required input CSVs
# are missing (they depend on outputs from other runs).
EXPERIMENT: str = "all"

# Country panel for Exp 1, 3, 4, 5 (main sweep).
# Set USE_ALL_PAPER_COUNTRIES = True to use the full 20-country paper panel,
# otherwise COUNTRIES is used verbatim.
USE_ALL_PAPER_COUNTRIES: bool = False
COUNTRIES: List[str] = ["USA", "JPN", "DEU", "VNM", "ETH"]

# Per-country scenario count and main RNG seed.
N_SCENARIOS: int = 500
SEED: int = 42

# Multi-seed sweep for Exp 3.
EXP3_SEEDS: List[int] = [42, 101, 2026]

# Exp 6 — 3x3 ablation grid (countries x ablations, single model here).
EXP6_COUNTRIES: List[str] = ["USA", "JPN", "VNM"]
EXP6_ABLATIONS: List[str] = [
    "Full SWA-DPBR",
    "No debiasing",
    "Without persona",
    "No-IS (consensus only)",
    "Always-on PT-IS",
]

# Exp 8 — N-persona sensitivity.
EXP8_COUNTRIES: List[str] = ["USA", "JPN", "VNM"]
EXP8_N_VALUES: List[int] = [1, 2, 3, 4]

# Exp 12 — WVS dimension dropout (leave-one-out).
EXP12_COUNTRIES: List[str] = ["USA", "JPN", "VNM"]

# Output directory. "" means: use the default RESULTS_BASE defined below
# (Kaggle working dir or local results folder).
OUT_DIR_OVERRIDE: str = ""

# --- Optimisation toggles --------------------------------------------------
# Logit cache: cache (delta_base, delta_agents, logit_temp) per unique
# (country, personas, user_query, lang, phenomenon_category). Any Exp that
# reuses the same (country, scenarios, personas) triple (Exp 6, 8, 12, and
# the multiple ablation variants within Exp 6) will hit the cache instead of
# re-running the model. Cache persists to disk as pickle under
# {out_dir}/_logit_cache/<country>.pkl so subsequent sessions also skip.
USE_LOGIT_CACHE: bool = True
LOGIT_CACHE_DIR: str = ""        # "" -> {out_dir}/_logit_cache

# Share one model load across every experiment in a single "all" run. When
# False, each run_experiment_X() reloads the model (old behaviour).
SHARE_MODEL_ACROSS_EXPERIMENTS: bool = True

# External CSV paths for Exp 2/4/5/7/9/10/11.
# Leave as "" to use defaults inside OUT_DIR (script will look for files
# produced by this run or earlier runs); set an absolute path to point
# at an upstream result file.
SCENARIO_CSV: str = ""           # Exp 2/7/10: per-scenario log from Exp 1
MAIN_RESULTS_CSV: str = ""       # Exp 2: columns country,vanilla_mis,disca_mis
VANILLA_RESULTS_CSV: str = ""    # Exp 4: per-country vanilla MIS
DISCA_RESULTS_CSV: str = ""      # Exp 5/7: per-country disca+vanilla MIS
MODEL_AMCE_CSV: str = ""         # Exp 9: long-form country,dimension,model_amce
HUMAN_AMCE_LONG_CSV: str = ""    # Exp 9: long-form country,dimension,human_amce
VANILLA_PER_DIM_CSV: str = ""    # Exp 11: long-form country,dimension,abs_err
DISCA_PER_DIM_CSV: str = ""      # Exp 11: long-form country,dimension,abs_err

# ============================================================================
# 1) Kaggle offline bootstrap (mirrors ablation_qwen25_7b style)
# ============================================================================

PROJECT_DATASET_DIR = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
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
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    if _on_kaggle():
        project_src = None
        for cand in (PROJECT_DATASET_DIR, PROJECT_DATASET_DIR_ALT):
            if os.path.isdir(cand):
                project_src = cand
                break

        if os.path.isdir(WORK_DIR) and os.path.isfile(os.path.join(WORK_DIR, "src", "controller.py")):
            print(f"[SETUP] Working dir exists: {WORK_DIR}")
        else:
            if project_src is None:
                raise RuntimeError(
                    "Project dataset not found. Checked: "
                    f"{PROJECT_DATASET_DIR} and {PROJECT_DATASET_DIR_ALT}"
                )
            print(f"[SETUP] Copying project from {project_src} -> {WORK_DIR} ...")
            shutil.copytree(project_src, WORK_DIR, dirs_exist_ok=True)

        os.chdir(WORK_DIR)
        sys.path.insert(0, WORK_DIR)
        return WORK_DIR

    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        sys.path.insert(0, here)
        return here
    raise RuntimeError("Not on Kaggle and not inside repo root.")


def _resolve_model_path() -> str:
    if not _on_kaggle():
        return os.environ.get("MORAL_MODEL_PATH", MODEL_LOCAL_PATH)

    p = MODEL_LOCAL_PATH
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
        return p

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


print("=" * 70)
print("  DISCA Playbook Runner — Qwen2.5-7B (HF native, bf16)")
print("=" * 70)

_setup_project()
MODEL_LOCAL_PATH_RESOLVED = _resolve_model_path()
print(f"[SETUP] Model path: {MODEL_LOCAL_PATH_RESOLVED}")

if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm matplotlib seaborn sentencepiece protobuf 2>/dev/null || true",
        shell=True,
        check=False,
    )

# Reuse ablation helpers, patch model + data paths similarly.
import exp_paper.exp_paper_ablation_phi4 as _abl  # noqa: E402
import exp_model._base_dpbr as _dpbr  # noqa: E402
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.config import model_slug  # noqa: E402
from src.model import load_model_hf_native, setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.baseline_runner import run_baseline_vanilla  # noqa: E402
from src.mc_dropout_runner import run_baseline_mc_dropout  # noqa: E402
from src.calibration_baselines import run_baseline_calibration_scaling  # noqa: E402
from src.diffpo_binary_baseline import run_baseline_diffpo_binary  # noqa: E402
from src.prompt_baseline_runner import run_prompt_baseline_country  # noqa: E402

_abl.MODEL_NAME = MODEL_LOCAL_PATH_RESOLVED
_abl.MODEL_SHORT = "kaggle_qwen25_7b_bf16"

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_playbook_qwen25_7b"
    if _on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_playbook_qwen25_7b")
)
_abl.RESULTS_BASE = RESULTS_BASE

_abl.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_abl.WVS_DATA_PATH = WVS_DATA_PATH
_abl.HUMAN_AMCE_PATH = HUMAN_AMCE_PATH
_dpbr.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_dpbr.WVS_DATA_PATH = WVS_DATA_PATH
_dpbr.HUMAN_AMCE_PATH = HUMAN_AMCE_PATH

_abl.FULL_SWA_BASE = os.environ.get("ABLATION_FULL_SWA_BASE") or (
    f"/kaggle/working/cultural_alignment/results/exp24_paper_20c/"
    f"kaggle_qwen25_7b_bf16/swa/{model_slug(MODEL_LOCAL_PATH_RESOLVED)}"
    if _on_kaggle()
    else None
)

print("[SETUP] Patched config:")
print(f"  MODEL  : {_abl.MODEL_NAME}")
print(f"  OUTPUT : {_abl.RESULTS_BASE}")
print(f"  MULTITP: {MULTITP_DATA_PATH}")


# ============================================================================
# 1.5) Logit cache — monkey-patches controller._extract_logit_gaps
# ============================================================================
# This sits BETWEEN the controller and the HF forward pass. For each
# (country, personas, user_query, lang, phenomenon_category) it stores the
# output of _extract_logit_gaps on disk. Any later call with an identical key
# reads from cache instead of running the model.
#
# Scope of benefit:
#   Exp 6: 5 ablations share the same logit forward pass -> cache hit on 4/5
#   Exp 8: N=1,2,3,4 share the same base logit extraction for the personas
#          used; N<4 is a subset so we cache at the controller level (where
#          it computes one forward over ALL agents loaded into it)
#   Exp 12: dropout changes persona text -> different cache key -> no hit
#          (genuinely new logits)
#   Exp 3: different seeds share logits -> cache hit
#
# Footprint: (n_personas+1) * 2 floats * n_scenarios per country per run.
# ~80KB / country at 500 scenarios. Trivial.

import hashlib  # noqa: E402
import pickle  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

# Per-process in-memory cache:
#   { country: { key: (db_np_fp32, da_np_fp32, logit_temp, db_dtype_str, da_dtype_str) } }
# The dtype strings let us restore the original bf16/fp16 on cache hit.
# numpy cannot natively hold bf16 so storage is always fp32.
_LOGIT_MEM_CACHE: Dict[str, Dict[str, tuple]] = {}
_LOGIT_CACHE_LOCK = threading.Lock()
_LOGIT_CACHE_STATS = {"hits": 0, "misses": 0, "writes": 0}
_CURRENT_COUNTRY: List[str] = ["_unknown_"]          # stack-like holder
_CURRENT_PERSONAS_HASH: List[str] = ["_none_"]


def _resolve_cache_dir(out_dir: Path) -> Path:
    base = Path(LOGIT_CACHE_DIR) if LOGIT_CACHE_DIR else (out_dir / "_logit_cache")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_file_for_country(country: str, out_dir: Path) -> Path:
    return _resolve_cache_dir(out_dir) / f"{country}.pkl"


def _load_country_cache(country: str, out_dir: Path) -> None:
    """Populate _LOGIT_MEM_CACHE[country] from disk (if the pickle exists)."""
    if country in _LOGIT_MEM_CACHE:
        return
    path = _cache_file_for_country(country, out_dir)
    if path.exists():
        try:
            with open(path, "rb") as fh:
                _LOGIT_MEM_CACHE[country] = pickle.load(fh)
            print(f"[LOGIT-CACHE] loaded {len(_LOGIT_MEM_CACHE[country])} entries for {country}")
            return
        except Exception as exc:
            print(f"[LOGIT-CACHE][WARN] could not read {path}: {exc} (starting fresh)")
    _LOGIT_MEM_CACHE[country] = {}


def _flush_country_cache(country: str, out_dir: Path) -> None:
    """Persist in-memory cache for a country to disk."""
    if country not in _LOGIT_MEM_CACHE:
        return
    path = _cache_file_for_country(country, out_dir)
    try:
        with open(path, "wb") as fh:
            pickle.dump(_LOGIT_MEM_CACHE[country], fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"[LOGIT-CACHE][WARN] could not write {path}: {exc}")


def _flush_all_caches(out_dir: Path) -> None:
    for c in list(_LOGIT_MEM_CACHE.keys()):
        _flush_country_cache(c, out_dir)
    print(
        f"[LOGIT-CACHE] stats: hits={_LOGIT_CACHE_STATS['hits']} "
        f"misses={_LOGIT_CACHE_STATS['misses']} writes={_LOGIT_CACHE_STATS['writes']}"
    )


def _personas_hash(personas: List[str]) -> str:
    h = hashlib.blake2s(digest_size=10)
    for p in personas:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_key(user_query: str, phenomenon_category: str, lang: str, personas_h: str) -> str:
    h = hashlib.blake2s(digest_size=16)
    h.update(personas_h.encode("utf-8"))
    h.update(b"|")
    h.update(lang.encode("utf-8"))
    h.update(b"|")
    h.update((phenomenon_category or "default").encode("utf-8"))
    h.update(b"|")
    h.update(user_query.encode("utf-8"))
    return h.hexdigest()


def _install_logit_cache() -> None:
    """Monkey-patch controller._extract_logit_gaps with a caching wrapper."""
    from src.controller import ImplicitSWAController

    orig = ImplicitSWAController._extract_logit_gaps

    def cached_extract(self, user_query, phenomenon_category, lang):
        country = _CURRENT_COUNTRY[0]
        personas_h = _CURRENT_PERSONAS_HASH[0]
        bucket = _LOGIT_MEM_CACHE.setdefault(country, {})
        key = _cache_key(user_query, phenomenon_category, lang, personas_h)

        entry = bucket.get(key)
        if entry is not None:
            # Stored as (db_np_fp32, da_np_fp32, lt, db_dtype_str, da_dtype_str).
            # Legacy entries (tuple of 3) are still readable but cast to float32.
            if len(entry) == 5:
                db_np, da_np, lt, db_dtype_str, da_dtype_str = entry
            else:
                db_np, da_np, lt = entry
                db_dtype_str = da_dtype_str = "float32"
            with _LOGIT_CACHE_LOCK:
                _LOGIT_CACHE_STATS["hits"] += 1
            db_dtype = getattr(torch, db_dtype_str, torch.float32)
            da_dtype = getattr(torch, da_dtype_str, torch.float32)
            db = torch.as_tensor(db_np, device=self.device).to(db_dtype)
            da = torch.as_tensor(da_np, device=self.device).to(da_dtype)
            return db, da, float(lt)

        db_t, da_t, lt = orig(self, user_query, phenomenon_category, lang)
        with _LOGIT_CACHE_LOCK:
            _LOGIT_CACHE_STATS["misses"] += 1
            _LOGIT_CACHE_STATS["writes"] += 1
        # numpy does not support bf16/fp16 — cast to fp32 for storage and
        # remember the original dtype so we can restore it on cache hit.
        db_dtype_str = str(db_t.dtype).replace("torch.", "")
        da_dtype_str = str(da_t.dtype).replace("torch.", "")
        bucket[key] = (
            db_t.detach().to(torch.float32).cpu().numpy(),
            da_t.detach().to(torch.float32).cpu().numpy(),
            float(lt),
            db_dtype_str,
            da_dtype_str,
        )
        return db_t, da_t, lt

    # Patch ONLY the base controller. Subclasses (NoDebias, NoPersona, NoIS,
    # AlwaysOnIS, NoPrior) inherit from it so the patched method is reused.
    # Subclasses that override _extract_logit_gaps themselves (NoPersona)
    # call super()._extract_logit_gaps internally, so the cache still works.
    ImplicitSWAController._extract_logit_gaps = cached_extract
    # Also patch the Exp24 dual-pass controller which inherits from the same
    # base but we want to be explicit.
    try:
        from experiment_DM.exp24_dpbr_core import Exp24DualPassController
        # Only override if Exp24 had its own method (it does not; inherits).
        if "_extract_logit_gaps" in Exp24DualPassController.__dict__:
            Exp24DualPassController._extract_logit_gaps = cached_extract
    except Exception:
        pass
    print("[LOGIT-CACHE] installed monkey-patch on _extract_logit_gaps")


def _set_cache_context(country: str, personas: List[str], out_dir: Path) -> None:
    """Call before each country/personas combination so subsequent
    _extract_logit_gaps calls land in the right bucket."""
    _CURRENT_COUNTRY[0] = country
    _CURRENT_PERSONAS_HASH[0] = _personas_hash(personas)
    _load_country_cache(country, out_dir)


def _reset_prior_for(country: str) -> None:
    """Reset the hierarchical country EMA prior before a fresh ablation run,
    mirroring _reset_prior_state from exp_paper_ablation_phi4.main(). This is
    required whenever the same (country) is re-run under a different
    configuration (different ablation variant, different N personas, different
    WVS dim dropout) so the prior does not carry state across runs and make
    comparisons unfair."""
    from experiment_DM.exp24_dpbr_core import PRIOR_STATE, BootstrapPriorState
    # Drop ALL keys referring to this country (main + NoPrior's sentinel).
    for k in list(PRIOR_STATE.keys()):
        if k == country or k.endswith(f"_{country}"):
            del PRIOR_STATE[k]
    PRIOR_STATE[country] = BootstrapPriorState()


if USE_LOGIT_CACHE:
    _install_logit_cache()


# ============================================================================
# 2) Experiment 1
# ============================================================================

def _parse_agent_gaps(raw) -> List[float]:
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [float(x) for x in raw]
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            vals = ast.literal_eval(raw)
            if isinstance(vals, (list, tuple)):
                return [float(x) for x in vals]
        except Exception:
            return []
    return []


def _build_scenario_analysis_rows(country: str, scenario_df: pd.DataFrame, results_df: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    n = min(len(scenario_df), len(results_df))
    for i in range(n):
        sc = scenario_df.iloc[i]
        rr = results_df.iloc[i]

        gaps = _parse_agent_gaps(rr.get("agent_decision_gaps", []))
        if len(gaps) >= 2:
            persona_variance = float(np.var(np.array(gaps, dtype=np.float32)))
        elif len(gaps) == 1:
            persona_variance = 0.0
        else:
            persona_variance = float("nan")

        # Playbook-consistent correction magnitude: |delta_star|.
        # In controller outputs, delta_star is represented by dual-pass terms:
        #   delta_star = reliability_r * 0.5 * (delta_star_1 + delta_star_2)
        ds1 = float(rr.get("delta_star_1", np.nan))
        ds2 = float(rr.get("delta_star_2", np.nan))
        rel = float(rr.get("reliability_r", np.nan))
        if np.isfinite(ds1) and np.isfinite(ds2) and np.isfinite(rel):
            delta_star = rel * 0.5 * (ds1 + ds2)
            correction_magnitude = float(abs(delta_star))
        else:
            # Fallback for precomputed outputs that may omit delta_star_* fields.
            delta_consensus = float(rr.get("delta_consensus", np.nan))
            delta_opt_micro = float(rr.get("delta_opt_micro", np.nan))
            if np.isfinite(delta_consensus) and np.isfinite(delta_opt_micro):
                correction_magnitude = float(abs(delta_opt_micro - delta_consensus))
            else:
                correction_magnitude = float("nan")

        rows.append(
            {
                "scenario_id": f"scen_{i + 1:04d}",
                "country": country,
                "dimension": str(sc.get("phenomenon_category", "unknown")),
                "persona_variance": persona_variance,
                "correction_magnitude": correction_magnitude,
                "reliability_weight": float(rr.get("reliability_r", np.nan)),
            }
        )
    return rows


def run_experiment_1(
    model,
    tokenizer,
    countries: List[str],
    n_scenarios: int,
    seed: int,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 1: Disagreement-Correction Correlation")
    print("#" * 80)
    print(f"[EXP1] Countries={countries} | scenarios={n_scenarios} | seed={seed}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios

    full_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "Full SWA-DPBR")
    all_rows: List[Dict] = []

    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[EXP1][SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue

        print(f"[EXP1] Running country: {country}")
        scenario_df = _abl._load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        _set_cache_context(country, personas, out_dir)
        _reset_prior_for(country)
        results_df, _summary = _abl._run_ablation_country(
            full_spec, model, tokenizer, country, personas, scenario_df, cfg
        )
        all_rows.extend(_build_scenario_analysis_rows(country, scenario_df, results_df))
        _flush_country_cache(country, out_dir)

    csv_path = out_dir / (
        "scenario_analysis_all_countries.csv" if len(countries) > 5 else "scenario_analysis.csv"
    )
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"[EXP1][SAVED] {csv_path}")
    return csv_path


def plot_experiment_1(csv_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr

    df_raw = pd.read_csv(csv_path)
    df = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["persona_variance", "correction_magnitude"])
    if df.empty:
        raise ValueError(
            "No valid rows for Exp1 plot after filtering persona_variance/correction_magnitude. "
            "Check scenario_analysis CSV contents."
        )
    df["log_variance"] = np.log10(df["persona_variance"] + 1e-4)

    fig, ax = plt.subplots(figsize=(7, 5))
    country_colors = {
        "USA": "#2D5F9A",
        "JPN": "#1A8A66",
        "DEU": "#534AB7",
        "VNM": "#C04E28",
        "ETH": "#EF9F27",
    }
    for country in sorted(df["country"].unique()):
        sub = df[df["country"] == country]
        color = country_colors.get(country, "#6C757D")
        ax.scatter(sub["log_variance"], sub["correction_magnitude"], alpha=0.3, s=10, color=color, label=country)

    sorted_df = df.sort_values("log_variance")
    if len(sorted_df) >= 51:
        window = max(51, (len(sorted_df) // 20) * 2 + 1)
        if window < len(sorted_df):
            smoothed = savgol_filter(sorted_df["correction_magnitude"].values, window, 3)
            ax.plot(sorted_df["log_variance"], smoothed, "k-", linewidth=1.5, label="Smoothed trend")

    if len(df) >= 2:
        r, p = pearsonr(df["log_variance"], df["correction_magnitude"])
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        corr_text = f"Pearson r = {r:.3f}\n{p_text}"
    else:
        corr_text = f"Pearson r = n/a\nn = {len(df)}"
    ax.text(
        0.05,
        0.95,
        corr_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("Inter-persona variance, log10 S(x)", fontsize=12)
    ax.set_ylabel("Correction magnitude, |delta*|", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    pdf_path = out_dir / "figure2_scenario_correlation.pdf"
    png_path = out_dir / "figure2_scenario_correlation.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"[EXP1][SAVED] {pdf_path}")
    print(f"[EXP1][SAVED] {png_path}")

    # ── Console summary — central hypothesis of the paper ────────────────────
    print("\n" + "-" * 70)
    print("  [EXP1] Disagreement-Correction Correlation — RESULT")
    print("-" * 70)
    print(f"  n_scenarios total           : {len(df)}")
    print(f"  n_countries                 : {df['country'].nunique()}")
    if len(df) >= 2:
        r, p = pearsonr(df["log_variance"], df["correction_magnitude"])
        strength = ("STRONG" if abs(r) >= 0.4 else
                    "MODERATE" if abs(r) >= 0.2 else "WEAK")
        print(f"  Pearson r (log S vs |δ*|)   : {r:+.4f}   [{strength}]")
        print(f"  p-value                     : {('<0.001' if p < 0.001 else f'{p:.4f}')}")
    print(f"  mean persona_variance       : {df['persona_variance'].mean():.4f}")
    print(f"  mean |correction|           : {df['correction_magnitude'].mean():.4f}")
    # Per-country breakdown
    per_country = df.groupby("country").agg(
        n=("persona_variance", "size"),
        mean_var=("persona_variance", "mean"),
        mean_corr=("correction_magnitude", "mean"),
    ).round(4)
    print("\n  Per-country breakdown:")
    print(per_country.to_string())
    print("-" * 70)


# ============================================================================
# 3) Experiment 2
# ============================================================================

def run_experiment_2(scenario_csv: Path, main_results_csv: Path, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scenario_csv)
    results = pd.read_csv(main_results_csv)

    required = {"country", "vanilla_mis", "disca_mis"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            f"main_results_csv missing columns {sorted(missing)}. "
            "Expected: country, vanilla_mis, disca_mis"
        )

    country_mean_var = (
        df.groupby("country", as_index=False)["persona_variance"]
        .mean()
        .rename(columns={"persona_variance": "mean_variance"})
    )
    merged = country_mean_var.merge(results[["country", "vanilla_mis", "disca_mis"]], on="country", how="inner")
    merged["delta_mis"] = merged["vanilla_mis"] - merged["disca_mis"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1A8A66" if d > 0 else "#C04E28" for d in merged["delta_mis"]]
    ax.scatter(
        merged["mean_variance"],
        merged["delta_mis"],
        s=80,
        c=colors,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in merged.iterrows():
        ax.annotate(
            row["country"],
            (row["mean_variance"], row["delta_mis"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

    if len(merged) >= 3:
        r, p = pearsonr(merged["mean_variance"], merged["delta_mis"])
        z = np.polyfit(merged["mean_variance"], merged["delta_mis"], 1)
        x_range = np.linspace(merged["mean_variance"].min(), merged["mean_variance"].max(), 100)
        ax.plot(x_range, z[0] * x_range + z[1], "k-", linewidth=1, alpha=0.5)
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}\np = {p:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    ax.set_xlabel("Mean inter-persona variance per country", fontsize=12)
    ax.set_ylabel("MIS improvement (vanilla - DISCA)", fontsize=12)
    plt.tight_layout()

    csv_out = out_dir / "country_correlation_data.csv"
    pdf_out = out_dir / "figure3_country_correlation.pdf"
    png_out = out_dir / "figure3_country_correlation.png"
    merged.to_csv(csv_out, index=False)
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=200, bbox_inches="tight")
    print(f"[EXP2][SAVED] {csv_out}")
    print(f"[EXP2][SAVED] {pdf_out}")
    print(f"[EXP2][SAVED] {png_out}")

    print("\n" + "-" * 70)
    print("  [EXP2] Country-Level Correlation — RESULT")
    print("-" * 70)
    print(f"  n_countries                 : {len(merged)}")
    if len(merged) >= 3:
        r, p = pearsonr(merged["mean_variance"], merged["delta_mis"])
        strength = ("STRONG" if abs(r) >= 0.5 else
                    "MODERATE" if abs(r) >= 0.3 else "WEAK")
        print(f"  Pearson r (mean_S vs ΔMIS)  : {r:+.4f}   [{strength}]")
        print(f"  p-value                     : {('<0.001' if p < 0.001 else f'{p:.4f}')}")
    sorted_m = merged.sort_values("delta_mis", ascending=False)
    print(f"  countries_improved (ΔMIS>0) : {(merged['delta_mis'] > 0).sum()} / {len(merged)}")
    print("\n  Top 5 improved countries:")
    print(sorted_m.head(5)[["country", "mean_variance", "delta_mis"]].to_string(index=False))
    print("\n  Bottom 5 (failure cases):")
    print(sorted_m.tail(5)[["country", "mean_variance", "delta_mis"]].to_string(index=False))
    print("-" * 70)
    return csv_out


# ============================================================================
# 4) Experiment 3
# ============================================================================

def run_experiment_3(
    model,
    tokenizer,
    countries: List[str],
    n_scenarios: int,
    seeds: List[int],
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 3: Multi-seed Confidence Intervals")
    print("#" * 80)
    print(f"[EXP3] Countries={countries} | scenarios={n_scenarios} | seeds={seeds}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios
    full_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "Full SWA-DPBR")

    rows: List[Dict] = []
    for seed in seeds:
        setup_seeds(seed)
        # Prior state must be cleared between seeds so EMA does not leak.
        from experiment_DM.exp24_dpbr_core import PRIOR_STATE
        PRIOR_STATE.clear()
        for country in countries:
            if country not in SUPPORTED_COUNTRIES:
                continue
            print(f"[EXP3] seed={seed} country={country}")
            scenario_df = _abl._load_scenarios(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
            _set_cache_context(country, personas, out_dir)
            _results_df, summary = _abl._run_ablation_country(full_spec, model, tokenizer, country, personas, scenario_df, cfg)
            mis = float(summary.get("alignment", {}).get("mis", np.nan))
            rows.append({"seed": seed, "country": country, "mis": mis})
            _flush_country_cache(country, out_dir)

    seed_country_df = pd.DataFrame(rows)
    raw_csv = out_dir / "exp3_seed_country_mis.csv"
    seed_country_df.to_csv(raw_csv, index=False)

    country_stats = (
        seed_country_df.groupby("country", as_index=False)["mis"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mis_mean", "std": "mis_std"})
    )
    if "index" in country_stats.columns:
        country_stats = country_stats.rename(columns={"index": "country"})

    per_seed_macro = seed_country_df.groupby("seed", as_index=False)["mis"].mean().rename(columns={"mis": "macro_mis"})
    macro_mean = float(per_seed_macro["macro_mis"].mean())
    macro_std = float(per_seed_macro["macro_mis"].std(ddof=1)) if len(per_seed_macro) > 1 else float("nan")

    country_csv = out_dir / "exp3_country_stats.csv"
    macro_csv = out_dir / "exp3_macro_stats.csv"
    country_stats.to_csv(country_csv, index=False)
    pd.DataFrame([{"macro_mean_mis": macro_mean, "macro_std_across_seeds": macro_std, "n_seeds": len(seeds)}]).to_csv(
        macro_csv, index=False
    )
    print(f"[EXP3][SAVED] {raw_csv}")
    print(f"[EXP3][SAVED] {country_csv}")
    print(f"[EXP3][SAVED] {macro_csv}")

    print("\n" + "-" * 70)
    print("  [EXP3] Multi-Seed Confidence Intervals — RESULT")
    print("-" * 70)
    print(f"  seeds used                  : {seeds}")
    print(f"  macro_mean MIS              : {macro_mean:.4f}")
    print(f"  macro_std across seeds      : {macro_std:.4f}")
    if np.isfinite(macro_std):
        stability = "STABLE" if macro_std < 0.01 else "ACCEPTABLE" if macro_std < 0.05 else "UNSTABLE"
        print(f"  stability                   : [{stability}] "
              f"(paper target: macro_std < 0.01)")
    print("\n  Per-country mean ± std:")
    cs = country_stats.copy()
    cs["mis_mean"] = cs["mis_mean"].round(4)
    cs["mis_std"] = cs["mis_std"].round(4)
    print(cs.to_string(index=False))
    print("-" * 70)
    return country_csv


# ============================================================================
# 5) Experiment 4
# ============================================================================

def _load_vanilla_mis_map(vanilla_results_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(vanilla_results_csv)
    if "country" not in df.columns:
        raise ValueError("vanilla_results_csv must contain country column")
    if "mis" in df.columns:
        return {str(r["country"]): float(r["mis"]) for _, r in df.iterrows()}
    if "vanilla_mis" in df.columns:
        return {str(r["country"]): float(r["vanilla_mis"]) for _, r in df.iterrows()}
    raise ValueError("vanilla_results_csv must contain either mis or vanilla_mis column")


def run_experiment_4(
    model,
    tokenizer,
    countries: List[str],
    n_scenarios: int,
    seed: int,
    vanilla_results_csv: Path,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 4: Tail-Safety Analysis")
    print("#" * 80)
    print(f"[EXP4] Countries={countries} | scenarios={n_scenarios} | seed={seed}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)
    vanilla_mis = _load_vanilla_mis_map(vanilla_results_csv)

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios

    full_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "Full SWA-DPBR")
    consensus_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "No-IS (consensus only)")
    variant_specs = [("full", full_spec), ("consensus", consensus_spec)]
    cell_rows: List[Dict] = []

    for variant_name, spec in variant_specs:
        for country in countries:
            if country not in SUPPORTED_COUNTRIES:
                continue
            if country not in vanilla_mis:
                print(f"[EXP4][SKIP] missing vanilla MIS for {country}")
                continue

            print(f"[EXP4] variant={variant_name} country={country}")
            scenario_df = _abl._load_scenarios(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
            _set_cache_context(country, personas, out_dir)
            _reset_prior_for(country)
            _results_df, summary = _abl._run_ablation_country(spec, model, tokenizer, country, personas, scenario_df, cfg)
            mis_variant = float(summary.get("alignment", {}).get("mis", np.nan))
            delta = float(vanilla_mis[country] - mis_variant)
            cell_rows.append(
                {
                    "variant": variant_name,
                    "country": country,
                    "vanilla_mis": float(vanilla_mis[country]),
                    "variant_mis": mis_variant,
                    "delta_mis": delta,
                }
            )
        # Flush caches after each variant iteration over all countries.
        for c in countries:
            if c in SUPPORTED_COUNTRIES:
                _flush_country_cache(c, out_dir)

    cell_df = pd.DataFrame(cell_rows)
    cell_csv = out_dir / "exp4_tail_safety_cells.csv"
    cell_df.to_csv(cell_csv, index=False)

    summary_rows: List[Dict] = []
    for variant_name in sorted(cell_df["variant"].unique()):
        d = cell_df[cell_df["variant"] == variant_name]["delta_mis"].values.astype(np.float64)
        neg = d[d < 0]
        summary_rows.append(
            {
                "variant": variant_name,
                "n_cells": int(len(d)),
                "mean_improvement": float(np.mean(d)) if len(d) else float("nan"),
                "num_hurt": int((d < 0).sum()),
                "worst_case_degradation": float(np.max(-neg)) if len(neg) else 0.0,
                "std_across_cells": float(np.std(d)) if len(d) else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "exp4_tail_safety_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[EXP4][SAVED] {cell_csv}")
    print(f"[EXP4][SAVED] {summary_csv}")

    print("\n" + "-" * 70)
    print("  [EXP4] Tail-Safety Analysis — RESULT")
    print("-" * 70)
    print(summary_df.round(4).to_string(index=False))
    # Automatic interpretation.
    if {"full", "consensus"}.issubset(set(summary_df["variant"])):
        full = summary_df[summary_df["variant"] == "full"].iloc[0]
        cons = summary_df[summary_df["variant"] == "consensus"].iloc[0]
        mean_diff = full["mean_improvement"] - cons["mean_improvement"]
        wc_ratio = (cons["worst_case_degradation"] /
                    max(full["worst_case_degradation"], 1e-9))
        print(f"\n  mean_improvement diff       : {mean_diff:+.4f} "
              f"(Full − Consensus)")
        print(f"  worst-case degradation ratio: {wc_ratio:.2f}× "
              f"(Consensus vs Full)")
        print(f"  cells hurt (Full / Consensus): "
              f"{int(full['num_hurt'])} / {int(cons['num_hurt'])}")
    print("-" * 70)
    return summary_csv


# ============================================================================
# 6) Experiment 5
# ============================================================================

def _run_single_baseline_for_country(
    baseline_name: str,
    model,
    tokenizer,
    scenario_df: pd.DataFrame,
    country: str,
    cfg,
) -> Tuple[float, pd.DataFrame]:
    if baseline_name == "Vanilla":
        out = run_baseline_vanilla(model, tokenizer, scenario_df, country, cfg)
    elif baseline_name == "WVS Prompt":
        out = run_prompt_baseline_country(
            model,
            tokenizer,
            scenario_df,
            country,
            cfg,
            baseline="B2",
            wvs_csv_path=WVS_DATA_PATH,
            human_amce_path=HUMAN_AMCE_PATH,
        )
    elif baseline_name == "MC-Dropout":
        out = run_baseline_mc_dropout(model, tokenizer, scenario_df, country, cfg)
    elif baseline_name == "Temp Scaling (uses AMCE)":
        out = run_baseline_calibration_scaling(model, tokenizer, scenario_df, country, cfg, method="temperature")
    elif baseline_name == "DiffPO-binary (uses AMCE)":
        out = run_baseline_diffpo_binary(model, tokenizer, scenario_df, country, cfg)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    if "alignment" in out:
        mis = float(out.get("alignment", {}).get("mis", np.nan))
    else:
        mis = float(out.get("summary", {}).get("mis", np.nan))

    res_df = out.get("results_df", pd.DataFrame()).copy()
    if "country" not in res_df.columns:
        res_df["country"] = country
    return mis, res_df


def run_experiment_5(
    model,
    tokenizer,
    countries: List[str],
    n_scenarios: int,
    seed: int,
    disca_results_csv: Path,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 5: Strong Baselines")
    print("#" * 80)
    print(f"[EXP5] Countries={countries} | scenarios={n_scenarios} | seed={seed}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)

    disca_df = pd.read_csv(disca_results_csv)
    if not {"country", "disca_mis"}.issubset(disca_df.columns):
        raise ValueError("disca_results_csv must contain country and disca_mis")

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios

    method_names = [
        "Vanilla",
        "WVS Prompt",
        "MC-Dropout",
        "Temp Scaling (uses AMCE)",
        "DiffPO-binary (uses AMCE)",
    ]
    rows: List[Dict] = []
    detail_dir = out_dir / "exp5_baseline_country_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"[EXP5] country={country}")
        scenario_df = _abl._load_scenarios(cfg, country)
        for method in method_names:
            print(f"  [EXP5] method={method}")
            mis, res_df = _run_single_baseline_for_country(method, model, tokenizer, scenario_df, country, cfg)
            rows.append({"country": country, "method": method, "mis": mis})
            method_tag = method.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            res_df.to_csv(detail_dir / f"{country}_{method_tag}.csv", index=False)

    baseline_df = pd.DataFrame(rows)
    baseline_csv = out_dir / "exp5_baseline_country_mis.csv"
    baseline_df.to_csv(baseline_csv, index=False)

    vanilla_map = {
        str(r["country"]): float(r["mis"])
        for _, r in baseline_df[baseline_df["method"] == "Vanilla"].iterrows()
    }
    method_summary = []
    for method, mdf in baseline_df.groupby("method"):
        wins = 0
        for _, r in mdf.iterrows():
            c = str(r["country"])
            if c in vanilla_map and float(r["mis"]) < float(vanilla_map[c]):
                wins += 1
        method_summary.append(
            {
                "method": method,
                "mean_mis": float(mdf["mis"].mean()),
                "wins_vs_vanilla": int(wins),
                "n_countries": int(len(mdf)),
            }
        )

    disca_summary = {
        "method": "DISCA (ours)",
        "mean_mis": float(disca_df["disca_mis"].mean()),
        "wins_vs_vanilla": int((disca_df["disca_mis"] < disca_df.get("vanilla_mis", np.inf)).sum())
        if "vanilla_mis" in disca_df.columns
        else -1,
        "n_countries": int(len(disca_df)),
    }
    method_summary.append(disca_summary)

    summary_df = pd.DataFrame(method_summary).sort_values("mean_mis")
    summary_csv = out_dir / "exp5_baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[EXP5][SAVED] {baseline_csv}")
    print(f"[EXP5][SAVED] {summary_csv}")

    print("\n" + "-" * 70)
    print("  [EXP5] Strong Baselines — RESULT (sorted by mean MIS ↓ better)")
    print("-" * 70)
    disp = summary_df.copy()
    disp["mean_mis"] = disp["mean_mis"].round(4)
    print(disp.to_string(index=False))
    # Highlight DISCA position.
    ours_mask = disp["method"].str.contains("DISCA", case=False)
    if ours_mask.any():
        pos = int(disp.reset_index(drop=True).index[ours_mask][0]) + 1
        print(f"\n  DISCA rank                  : #{pos} out of {len(disp)} methods")
    print("-" * 70)
    return summary_csv


# ============================================================================
# 7) Experiment 6 — 3x3 ablation grid (models x countries x ablations)
# ============================================================================
#
# Playbook intent: run a matrix of ablation variants on N models x M countries
# to show the component-importance hierarchy is robust. Because this script
# is single-model (Qwen2.5-7B), we implement the 3-country x K-ablation slice
# here and leave cross-model aggregation as a post-hoc CSV merge step. The
# output CSV shape is model-generic so other model runs can be concatenated.

def run_experiment_6(
    model,
    tokenizer,
    countries: List[str],
    ablation_labels: List[str],
    n_scenarios: int,
    seed: int,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 6: 3x3 Ablation Grid")
    print("#" * 80)
    print(f"[EXP6] Countries={countries} | Ablations={ablation_labels}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios

    spec_by_label = {s.row_label: s for s in _abl.ABLATION_SPECS}
    missing = [lbl for lbl in ablation_labels if lbl not in spec_by_label]
    if missing:
        raise ValueError(f"Unknown ablation labels: {missing}. Available: {list(spec_by_label.keys())}")

    rows: List[Dict] = []
    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[EXP6][SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        scenario_df = _abl._load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        for label in ablation_labels:
            spec = spec_by_label[label]
            print(f"[EXP6] country={country} variant={label}")
            # Reset prior EMA before each variant so they are comparable —
            # mirrors exp_paper_ablation_phi4.main() which calls
            # _reset_prior_state(country) before each ablation spec.
            _reset_prior_for(country)
            # For "Without persona" the controller is called with run_personas=[""]
            # inside _run_ablation_country (see exp_paper_ablation_phi4.py:582-586),
            # so the cache key MUST use the same effective persona list to avoid
            # cross-variant cache contamination.
            effective_personas = (
                [""]
                if spec.controller_cls.__name__ == "NoPersonaController"
                else personas
            )
            _set_cache_context(country, effective_personas, out_dir)
            _results_df, summary = _abl._run_ablation_country(spec, model, tokenizer, country, personas, scenario_df, cfg)
            a = summary.get("alignment", {})
            rows.append({
                "model_slug": model_slug(_abl.MODEL_NAME),
                "country": country,
                "variant": label,
                "mis": float(a.get("mis", np.nan)),
                "pearson_r": float(a.get("pearson_r", np.nan)),
                "jsd": float(a.get("jsd", np.nan)),
                "mae": float(a.get("mae", np.nan)),
            })
        _flush_country_cache(country, out_dir)

    grid_df = pd.DataFrame(rows)
    grid_csv = out_dir / "exp6_ablation_grid.csv"
    grid_df.to_csv(grid_csv, index=False)

    # Delta-vs-Full summary (per-cell, one row per ablation-country, showing
    # how much each ablation degrades relative to Full for that country).
    if "Full SWA-DPBR" in grid_df["variant"].values:
        full_map = {
            (r["model_slug"], r["country"]): float(r["mis"])
            for _, r in grid_df[grid_df["variant"] == "Full SWA-DPBR"].iterrows()
        }
        delta_rows: List[Dict] = []
        for _, r in grid_df.iterrows():
            ref = full_map.get((r["model_slug"], r["country"]))
            delta_mis = float(r["mis"]) - ref if ref is not None else float("nan")
            delta_rows.append({
                "model_slug": r["model_slug"],
                "country": r["country"],
                "variant": r["variant"],
                "mis": r["mis"],
                "delta_mis_vs_full": delta_mis,
            })
        delta_df = pd.DataFrame(delta_rows)
        delta_csv = out_dir / "exp6_ablation_grid_delta.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"[EXP6][SAVED] {delta_csv}")

    print(f"[EXP6][SAVED] {grid_csv}")

    print("\n" + "-" * 70)
    print("  [EXP6] Ablation Grid — RESULT")
    print("-" * 70)
    # Pivot variant × country showing MIS
    try:
        pivot_mis = grid_df.pivot(index="variant", columns="country", values="mis").round(4)
        print("  MIS per (variant × country):")
        print(pivot_mis.to_string())
        if "Full SWA-DPBR" in pivot_mis.index:
            ref = pivot_mis.loc["Full SWA-DPBR"]
            delta_pivot = (pivot_mis - ref).round(4)
            print("\n  ΔMIS vs Full (positive = ablation hurt):")
            print(delta_pivot.to_string())
    except Exception as exc:
        print(f"  (pivot skipped: {exc})")
        print(grid_df.round(4).to_string(index=False))
    print("-" * 70)
    return grid_csv


# ============================================================================
# 8) Experiment 7 — Predictive failure model
# ============================================================================


def run_experiment_7(scenario_csv: Path, disca_results_csv: Path, out_dir: Path) -> Path:
    """Regress per-cell delta MIS on vanilla-pass statistics.

    Inputs:
      scenario_csv       : output of Exp1 (per-scenario with reliability_weight,
                            persona_variance). We aggregate to per-country features.
      disca_results_csv  : CSV with columns country, vanilla_mis, disca_mis.

    We approximate the playbook's "decision margin" and "logit entropy" from
    available per-scenario fields. Since these come from the DISCA pipeline
    (not a pure vanilla pass), they serve as a best-effort proxy and are
    named accordingly in the output. If a separate vanilla logits CSV is
    provided later, substitute it.
    """
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 80)
    print("  Experiment 7: Predictive Failure Model")
    print("#" * 80)

    df = pd.read_csv(scenario_csv)
    results = pd.read_csv(disca_results_csv)
    if not {"country", "vanilla_mis", "disca_mis"}.issubset(results.columns):
        raise ValueError("disca_results_csv must contain country, vanilla_mis, disca_mis")

    # Per-country features from scenario log.
    feats = (
        df.groupby("country", as_index=False)
        .agg(
            mean_variance=("persona_variance", "mean"),
            mean_correction=("correction_magnitude", "mean"),
            mean_reliability=("reliability_weight", "mean"),
        )
    )
    merged = feats.merge(results[["country", "vanilla_mis", "disca_mis"]], on="country", how="inner")
    merged["delta_mis"] = merged["vanilla_mis"] - merged["disca_mis"]

    X_cols = ["mean_variance", "mean_correction", "mean_reliability", "vanilla_mis"]
    X = merged[X_cols].values
    y = merged["delta_mis"].values
    if len(merged) < max(3, len(X_cols) + 1):
        print(f"[EXP7][WARN] Only {len(merged)} cells — regression is under-determined")
    reg = LinearRegression().fit(X, y)
    r2 = float(reg.score(X, y))

    coef_rows = [{"feature": c, "coefficient": float(coef)} for c, coef in zip(X_cols, reg.coef_)]
    coef_rows.append({"feature": "intercept", "coefficient": float(reg.intercept_)})
    coef_rows.append({"feature": "R_squared", "coefficient": r2})
    coef_df = pd.DataFrame(coef_rows)
    coef_csv = out_dir / "exp7_failure_model_coefficients.csv"
    coef_df.to_csv(coef_csv, index=False)

    # Scatter: vanilla_mis vs delta_mis (dominant predictor per playbook).
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1A8A66" if d > 0 else "#C04E28" for d in merged["delta_mis"]]
    ax.scatter(merged["vanilla_mis"], merged["delta_mis"], c=colors, s=70,
               edgecolors="black", linewidths=0.5, alpha=0.8)
    for _, row in merged.iterrows():
        ax.annotate(row["country"], (row["vanilla_mis"], row["delta_mis"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Vanilla MIS", fontsize=12)
    ax.set_ylabel("MIS improvement (vanilla - DISCA)", fontsize=12)
    ax.text(0.05, 0.95, f"R^2 = {r2:.3f}  (4 features)",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout()
    pdf = out_dir / "exp7_failure_prediction.pdf"
    png = out_dir / "exp7_failure_prediction.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    print(f"[EXP7][SAVED] {coef_csv}")
    print(f"[EXP7][SAVED] {pdf}")
    print(f"[EXP7][SAVED] {png}")

    print("\n" + "-" * 70)
    print("  [EXP7] Predictive Failure Model — RESULT")
    print("-" * 70)
    print(f"  n_cells (countries)         : {len(merged)}")
    print(f"  R² of 4-feature model       : {r2:.4f}")
    print(f"  feature coefficients (signed — larger |coef| = more important):")
    for c, coef in zip(X_cols, reg.coef_):
        print(f"    {c:<20} : {float(coef):+.4f}")
    print(f"    intercept            : {float(reg.intercept_):+.4f}")
    print("-" * 70)
    return coef_csv


# ============================================================================
# 9) Experiment 8 — N-persona sensitivity
# ============================================================================


def _truncate_personas(personas: List[str], n: int) -> List[str]:
    """Pick N personas from the 4 returned by build_country_personas.

    N=1 -> aggregate (index 3); N=2 -> young+older (0,2); N=3 -> three age
    cohorts (0,1,2); N=4 -> all. For N outside {1..4} we fall back to the
    first N available (or the full list, whichever is shorter). The controller
    treats any non-empty persona list as valid.
    """
    if len(personas) == 0:
        return personas
    if n >= len(personas):
        return list(personas)
    if n == 1:
        return [personas[-1]]
    if n == 2:
        if len(personas) >= 3:
            return [personas[0], personas[2]]
        return personas[:2]
    if n == 3:
        return personas[:3]
    return personas[:n]


def run_experiment_8(
    model,
    tokenizer,
    countries: List[str],
    n_values: List[int],
    n_scenarios: int,
    seed: int,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 8: N-Persona Sensitivity")
    print("#" * 80)
    print(f"[EXP8] Countries={countries} | N values={n_values}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios
    full_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "Full SWA-DPBR")

    rows: List[Dict] = []
    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            continue
        scenario_df = _abl._load_scenarios(cfg, country)
        base_personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        for n in n_values:
            personas = _truncate_personas(base_personas, n)
            if not personas:
                print(f"[EXP8][SKIP] country={country} N={n} — empty persona list")
                continue
            # Cache key includes persona list, so different N -> different key;
            # but when personas is identical (N>=4 full list), the cache hits.
            _set_cache_context(country, personas, out_dir)
            _reset_prior_for(country)
            print(f"[EXP8] country={country} N={n} (have {len(personas)} personas)")
            _r, summary = _abl._run_ablation_country(full_spec, model, tokenizer, country, personas, scenario_df, cfg)
            a = summary.get("alignment", {})
            rows.append({
                "country": country,
                "n_personas_requested": int(n),
                "n_personas_used": int(len(personas)),
                "mis": float(a.get("mis", np.nan)),
                "pearson_r": float(a.get("pearson_r", np.nan)),
                "jsd": float(a.get("jsd", np.nan)),
            })
        _flush_country_cache(country, out_dir)

    df_out = pd.DataFrame(rows)
    csv_out = out_dir / "exp8_n_persona_sensitivity.csv"
    df_out.to_csv(csv_out, index=False)

    # Simple line plot per country.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for country, sub in df_out.groupby("country"):
        sub = sub.sort_values("n_personas_used")
        ax.plot(sub["n_personas_used"], sub["mis"], marker="o", label=country)
    ax.set_xlabel("Number of personas (N)", fontsize=12)
    ax.set_ylabel("MIS (lower is better)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    pdf = out_dir / "exp8_n_persona_sensitivity.pdf"
    png = out_dir / "exp8_n_persona_sensitivity.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")

    print(f"[EXP8][SAVED] {csv_out}")
    print(f"[EXP8][SAVED] {pdf}")
    print(f"[EXP8][SAVED] {png}")

    print("\n" + "-" * 70)
    print("  [EXP8] N-Persona Sensitivity — RESULT")
    print("-" * 70)
    try:
        pivot_n = df_out.pivot(index="n_personas_used", columns="country", values="mis").round(4)
        print("  MIS per (N × country):")
        print(pivot_n.to_string())
        # Report knee (largest drop when increasing N by 1).
        mean_by_n = df_out.groupby("n_personas_used")["mis"].mean().sort_index()
        print("\n  mean MIS across countries by N:")
        print(mean_by_n.round(4).to_string())
        drops = mean_by_n.diff().dropna()
        if len(drops) > 0:
            knee_n = int(drops.idxmin())
            print(f"\n  largest MIS drop at N       : {knee_n} (delta={drops.min():+.4f})")
    except Exception as exc:
        print(f"  (pivot skipped: {exc})")
        print(df_out.round(4).to_string(index=False))
    print("-" * 70)
    return csv_out


# ============================================================================
# 10) Experiment 9 — Negative Pearson r diagnosis (per-dim rank swaps)
# ============================================================================


def run_experiment_9(disca_model_amce_csv: Path, human_amce_csv: Path, out_dir: Path) -> Path:
    """Diagnose per-country dimension-rank swaps for negative-r countries.

    Inputs:
      disca_model_amce_csv : long-form CSV with columns
                             country, dimension, model_amce
      human_amce_csv       : same schema but column human_amce instead.
    """
    from scipy.stats import pearsonr

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 80)
    print("  Experiment 9: Negative Pearson r Diagnosis")
    print("#" * 80)

    model_df = pd.read_csv(disca_model_amce_csv)
    human_df = pd.read_csv(human_amce_csv)
    required_m = {"country", "dimension", "model_amce"}
    required_h = {"country", "dimension", "human_amce"}
    if not required_m.issubset(model_df.columns):
        raise ValueError(f"disca_model_amce_csv missing: {required_m - set(model_df.columns)}")
    if not required_h.issubset(human_df.columns):
        raise ValueError(f"human_amce_csv missing: {required_h - set(human_df.columns)}")

    merged = model_df.merge(human_df, on=["country", "dimension"], how="inner")
    swap_rows: List[Dict] = []
    country_r_rows: List[Dict] = []

    for country, sub in merged.groupby("country"):
        sub = sub.sort_values("dimension")
        m = sub["model_amce"].values.astype(float)
        h = sub["human_amce"].values.astype(float)
        if len(sub) < 3:
            continue
        r, _ = pearsonr(m, h)
        country_r_rows.append({"country": country, "pearson_r": float(r), "n_dim": int(len(sub))})
        if r >= 0:
            continue
        # Negative r: enumerate every rank swap.
        h_order = np.argsort(-h)
        d_order = np.argsort(-m)
        h_rank = {i: int(np.where(h_order == i)[0][0]) for i in range(len(sub))}
        d_rank = {i: int(np.where(d_order == i)[0][0]) for i in range(len(sub))}
        dims = sub["dimension"].tolist()
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                if (h_rank[i] < h_rank[j]) != (d_rank[i] < d_rank[j]):
                    swap_rows.append({
                        "country": country,
                        "pearson_r": float(r),
                        "dim_a": dims[i],
                        "dim_b": dims[j],
                    })

    country_r_df = pd.DataFrame(country_r_rows)
    swaps_df = pd.DataFrame(swap_rows)
    country_csv = out_dir / "exp9_country_pearson_r.csv"
    swap_csv = out_dir / "exp9_rank_swaps.csv"
    country_r_df.to_csv(country_csv, index=False)
    swaps_df.to_csv(swap_csv, index=False)

    print(f"[EXP9][SAVED] {country_csv}")
    print(f"[EXP9][SAVED] {swap_csv}")

    print("\n" + "-" * 70)
    print("  [EXP9] Negative Pearson r Diagnosis — RESULT")
    print("-" * 70)
    n_total = len(country_r_df)
    n_neg = int((country_r_df["pearson_r"] < 0).sum()) if n_total else 0
    print(f"  countries analysed          : {n_total}")
    print(f"  countries with negative r   : {n_neg} ({(n_neg/max(n_total,1))*100:.1f}%)")
    if n_neg > 0:
        worst = country_r_df.sort_values("pearson_r").head(5)
        print("\n  Worst 5 countries (lowest r):")
        print(worst.round(4).to_string(index=False))

    if not swaps_df.empty:
        top = (
            swaps_df.groupby(["dim_a", "dim_b"], as_index=False)
            .size()
            .rename(columns={"size": "n_countries"})
            .sort_values("n_countries", ascending=False)
        )
        top_csv = out_dir / "exp9_most_common_swaps.csv"
        top.to_csv(top_csv, index=False)
        print(f"[EXP9][SAVED] {top_csv}")
        print("\n  Top swap pairs (most common rank reversals):")
        print(top.head(10).to_string(index=False))
    print("-" * 70)
    return swap_csv


# ============================================================================
# 11) Experiment 10 — Reliability weight histogram
# ============================================================================


def run_experiment_10(scenario_csv: Path, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 80)
    print("  Experiment 10: Reliability Weight Histogram")
    print("#" * 80)

    df = pd.read_csv(scenario_csv)
    if "reliability_weight" not in df.columns:
        raise ValueError("scenario_csv is missing reliability_weight (re-run Exp 1 first)")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["reliability_weight"])
    if df.empty:
        raise ValueError("No valid reliability_weight rows")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["reliability_weight"], bins=40, color="#534AB7", alpha=0.8,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="r = 0.5 (gate threshold)")
    ax.set_xlabel("Reliability weight r")
    ax.set_ylabel("Number of scenarios")
    ax.legend(fontsize=9)
    plt.tight_layout()

    pdf = out_dir / "exp10_reliability_distribution.pdf"
    png = out_dir / "exp10_reliability_distribution.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")

    stats_csv = out_dir / "exp10_reliability_stats.csv"
    pd.DataFrame([{
        "n_scenarios": int(len(df)),
        "mean_r": float(df["reliability_weight"].mean()),
        "median_r": float(df["reliability_weight"].median()),
        "frac_r_below_0_5": float((df["reliability_weight"] < 0.5).mean()),
        "frac_r_below_0_1": float((df["reliability_weight"] < 0.1).mean()),
    }]).to_csv(stats_csv, index=False)

    print(f"[EXP10][SAVED] {pdf}")
    print(f"[EXP10][SAVED] {png}")
    print(f"[EXP10][SAVED] {stats_csv}")

    print("\n" + "-" * 70)
    print("  [EXP10] Reliability Weight Distribution — RESULT")
    print("-" * 70)
    r_arr = df["reliability_weight"].values
    print(f"  n_scenarios                 : {len(r_arr)}")
    print(f"  mean r                      : {np.mean(r_arr):.4f}")
    print(f"  median r                    : {np.median(r_arr):.4f}")
    print(f"  min / max                   : {np.min(r_arr):.4f} / {np.max(r_arr):.4f}")
    print(f"  fraction r < 0.5 (gated)    : {np.mean(r_arr < 0.5)*100:.2f}%")
    print(f"  fraction r < 0.1 (suppressed): {np.mean(r_arr < 0.1)*100:.2f}%")
    print(f"  fraction r > 0.9 (trusted)  : {np.mean(r_arr > 0.9)*100:.2f}%")
    print("-" * 70)
    return stats_csv


# ============================================================================
# 12) Experiment 11 — Per-dimension improvement breakdown
# ============================================================================


def run_experiment_11(vanilla_per_dim_csv: Path, disca_per_dim_csv: Path, out_dir: Path) -> Path:
    """Compare per-dimension |model - human| between vanilla and DISCA.

    Inputs are long-form CSVs with columns: country, dimension, abs_err.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 80)
    print("  Experiment 11: Per-Dimension Improvement Breakdown")
    print("#" * 80)

    van = pd.read_csv(vanilla_per_dim_csv)
    dis = pd.read_csv(disca_per_dim_csv)
    for name, df in [("vanilla", van), ("disca", dis)]:
        if not {"country", "dimension", "abs_err"}.issubset(df.columns):
            raise ValueError(f"{name} per-dim CSV must have country, dimension, abs_err")

    van_avg = van.groupby("dimension", as_index=False)["abs_err"].mean().rename(columns={"abs_err": "vanilla_err"})
    dis_avg = dis.groupby("dimension", as_index=False)["abs_err"].mean().rename(columns={"abs_err": "disca_err"})
    merged = van_avg.merge(dis_avg, on="dimension", how="inner")
    merged["improvement"] = merged["vanilla_err"] - merged["disca_err"]
    merged = merged.sort_values("improvement", ascending=False)

    csv_out = out_dir / "exp11_per_dim_breakdown.csv"
    merged.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(merged))
    w = 0.4
    ax.bar(x - w / 2, merged["vanilla_err"], width=w, label="Vanilla", color="#C04E28", alpha=0.85)
    ax.bar(x + w / 2, merged["disca_err"], width=w, label="DISCA", color="#1A8A66", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([d.split("_")[0] for d in merged["dimension"]], rotation=20, ha="right")
    ax.set_ylabel("Mean |model - human| across countries")
    ax.legend(fontsize=9)
    plt.tight_layout()

    pdf = out_dir / "exp11_per_dim_breakdown.pdf"
    png = out_dir / "exp11_per_dim_breakdown.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")

    print(f"[EXP11][SAVED] {csv_out}")
    print(f"[EXP11][SAVED] {pdf}")
    print(f"[EXP11][SAVED] {png}")

    print("\n" + "-" * 70)
    print("  [EXP11] Per-Dimension Improvement Breakdown — RESULT")
    print("-" * 70)
    print("  Sorted by improvement (vanilla_err − disca_err):")
    print(merged.round(4).to_string(index=False))
    n_improved = int((merged["improvement"] > 0).sum())
    print(f"\n  dimensions improved         : {n_improved} / {len(merged)}")
    if n_improved > 0:
        best = merged.iloc[0]
        print(f"  best-improved dimension     : {best['dimension']} "
              f"(Δ={best['improvement']:+.4f})")
    print("-" * 70)
    return csv_out


# ============================================================================
# 13) Experiment 12 — WVS dimension dropout (leave-one-out)
# ============================================================================

EXP12_WVS_DIMS: List[str] = [
    "religiosity",
    "child_rearing",
    "moral_acceptability",
    "social_trust",
    "political_participation",
    "national_pride",
    "happiness",
    "gender_equality",
    "materialism_orientation",
    "tolerance_diversity",
]


def run_experiment_12(
    model,
    tokenizer,
    countries: List[str],
    n_scenarios: int,
    seed: int,
    out_dir: Path,
) -> Path:
    print("\n" + "#" * 80)
    print("  Experiment 12: WVS Dimension Dropout (leave-one-out)")
    print("#" * 80)
    print(f"[EXP12] Countries={countries}")

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_seeds(seed)

    cfg = _abl._build_cfg(countries, load_in_4bit=False)
    cfg.n_scenarios = n_scenarios
    full_spec = next(s for s in _abl.ABLATION_SPECS if s.row_label == "Full SWA-DPBR")

    rows: List[Dict] = []
    # Reference: no dropout (all 10 dims active).
    for country in countries:
        if country not in SUPPORTED_COUNTRIES:
            continue
        scenario_df = _abl._load_scenarios(cfg, country)
        print(f"[EXP12] country={country} dim_dropped=<none> (reference)")
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        _set_cache_context(country, personas, out_dir)
        _reset_prior_for(country)
        _r, summary = _abl._run_ablation_country(full_spec, model, tokenizer, country, personas, scenario_df, cfg)
        rows.append({
            "country": country,
            "dim_dropped": "(none)",
            "mis": float(summary.get("alignment", {}).get("mis", np.nan)),
            "pearson_r": float(summary.get("alignment", {}).get("pearson_r", np.nan)),
        })
        for dim in EXP12_WVS_DIMS:
            print(f"[EXP12] country={country} dim_dropped={dim}")
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH, drop_dims={dim})
            _set_cache_context(country, personas, out_dir)
            _reset_prior_for(country)
            _r2, summary2 = _abl._run_ablation_country(full_spec, model, tokenizer, country, personas, scenario_df, cfg)
            rows.append({
                "country": country,
                "dim_dropped": dim,
                "mis": float(summary2.get("alignment", {}).get("mis", np.nan)),
                "pearson_r": float(summary2.get("alignment", {}).get("pearson_r", np.nan)),
            })
        _flush_country_cache(country, out_dir)

    df_out = pd.DataFrame(rows)
    raw_csv = out_dir / "exp12_wvs_dropout_raw.csv"
    df_out.to_csv(raw_csv, index=False)

    # Delta vs reference, averaged across the provided countries.
    ref = df_out[df_out["dim_dropped"] == "(none)"][["country", "mis"]].rename(columns={"mis": "ref_mis"})
    joined = df_out.merge(ref, on="country", how="left")
    joined["delta_mis"] = joined["mis"] - joined["ref_mis"]

    agg = (
        joined[joined["dim_dropped"] != "(none)"]
        .groupby("dim_dropped", as_index=False)["delta_mis"]
        .mean()
        .sort_values("delta_mis", ascending=False)
    )
    agg_csv = out_dir / "exp12_wvs_dropout_summary.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"[EXP12][SAVED] {raw_csv}")
    print(f"[EXP12][SAVED] {agg_csv}")

    print("\n" + "-" * 70)
    print("  [EXP12] WVS Dimension Dropout — RESULT")
    print("-" * 70)
    print(f"  countries used              : {countries}")
    print(f"  WVS dimensions tested       : {len(EXP12_WVS_DIMS)}")
    print("\n  Load-bearing ranking (higher ΔMIS = dropping this dim hurts more):")
    agg_show = agg.copy()
    agg_show["delta_mis"] = agg_show["delta_mis"].round(4)
    print(agg_show.to_string(index=False))
    if len(agg_show) > 0:
        critical = agg_show[agg_show["delta_mis"] > 0.03]["dim_dropped"].tolist()
        if critical:
            print(f"\n  critical dims (ΔMIS > 0.03) : {critical}")
    print("-" * 70)
    return agg_csv


# ============================================================================
# 14) CLI
# ============================================================================

def main() -> None:
    # All configuration is read from the CONFIG constants at the top of this
    # file. Edit those to change behaviour, then just re-run.
    experiment = EXPERIMENT
    n_scenarios = N_SCENARIOS
    seed = SEED
    seeds = list(EXP3_SEEDS)

    out_dir = Path(OUT_DIR_OVERRIDE) if OUT_DIR_OVERRIDE else Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = list(PAPER_20_COUNTRIES) if USE_ALL_PAPER_COUNTRIES else list(COUNTRIES)

    def _opt_path(explicit: str, fallback_name: str) -> Path:
        return Path(explicit) if explicit else out_dir / fallback_name

    scenario_csv = _opt_path(SCENARIO_CSV, "scenario_analysis_all_countries.csv")
    main_results_csv = _opt_path(MAIN_RESULTS_CSV, "main_results_phi4.csv")
    vanilla_csv = _opt_path(VANILLA_RESULTS_CSV, "vanilla_qwen25_7b.csv")
    disca_csv = _opt_path(DISCA_RESULTS_CSV, "disca_qwen25_7b.csv")
    model_amce_csv = Path(MODEL_AMCE_CSV) if MODEL_AMCE_CSV else Path("")
    human_amce_long = Path(HUMAN_AMCE_LONG_CSV) if HUMAN_AMCE_LONG_CSV else Path("")
    vanilla_per_dim_csv = Path(VANILLA_PER_DIM_CSV) if VANILLA_PER_DIM_CSV else Path("")
    disca_per_dim_csv = Path(DISCA_PER_DIM_CSV) if DISCA_PER_DIM_CSV else Path("")

    exp6_countries = list(EXP6_COUNTRIES)
    exp6_ablations = list(EXP6_ABLATIONS)
    exp8_countries = list(EXP8_COUNTRIES)
    exp8_n_values = list(EXP8_N_VALUES)
    exp12_countries = list(EXP12_COUNTRIES)

    print("\n" + "=" * 70)
    print("  DISCA PLAYBOOK — session start")
    print("=" * 70)
    print(f"  experiment                  : {experiment}")
    print(f"  countries                   : {countries}")
    print(f"  n_scenarios / seed          : {n_scenarios} / {seed}")
    print(f"  output dir                  : {out_dir}")
    print(f"  USE_LOGIT_CACHE             : {USE_LOGIT_CACHE}")
    print(f"  SHARE_MODEL_ACROSS_EXPS     : {SHARE_MODEL_ACROSS_EXPERIMENTS}")
    print("=" * 70)

    import time
    session_t0 = time.time()
    exp_timings: Dict[str, float] = {}

    def _timed(name: str, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        exp_timings[name] = time.time() - t0
        return result

    # ── Determine which experiments actually need a model loaded ──────────────
    model_free_exps = {"2", "7", "9", "10", "11"}
    needs_model_single = experiment not in model_free_exps and experiment != "all"
    needs_model_all = experiment == "all"

    # ── Load model once (shared across every experiment) ──────────────────────
    model = tokenizer = None
    if (needs_model_single or needs_model_all) and SHARE_MODEL_ACROSS_EXPERIMENTS:
        print(f"[PLAYBOOK] Loading model once: {_abl.MODEL_NAME}")
        model, tokenizer = load_model_hf_native(_abl.MODEL_NAME, max_seq_length=2048, load_in_4bit=False)

    def _ensure_model():
        nonlocal model, tokenizer
        if model is None:
            print(f"[PLAYBOOK] Loading model on demand: {_abl.MODEL_NAME}")
            model, tokenizer = load_model_hf_native(_abl.MODEL_NAME, max_seq_length=2048, load_in_4bit=False)
        return model, tokenizer

    try:
        # ── Single-experiment dispatch ────────────────────────────────────────
        if experiment == "1":
            m, tk = _ensure_model()
            csv_path = _timed("Exp1", run_experiment_1, m, tk,
                              countries=countries, n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)
            _timed("Exp1-plot", plot_experiment_1, csv_path=csv_path, out_dir=out_dir)
            return

        if experiment == "2":
            if not scenario_csv.exists():
                raise FileNotFoundError(f"scenario CSV not found: {scenario_csv} (set SCENARIO_CSV)")
            if not main_results_csv.exists():
                raise FileNotFoundError(f"main results CSV not found: {main_results_csv} (set MAIN_RESULTS_CSV)")
            _timed("Exp2", run_experiment_2,
                   scenario_csv=scenario_csv, main_results_csv=main_results_csv, out_dir=out_dir)
            return

        if experiment == "3":
            m, tk = _ensure_model()
            _timed("Exp3", run_experiment_3, m, tk,
                   countries=countries, n_scenarios=n_scenarios, seeds=seeds, out_dir=out_dir)
            return

        if experiment == "4":
            if not vanilla_csv.exists():
                raise FileNotFoundError(f"vanilla results CSV not found: {vanilla_csv} (set VANILLA_RESULTS_CSV)")
            m, tk = _ensure_model()
            _timed("Exp4", run_experiment_4, m, tk,
                   countries=countries, n_scenarios=n_scenarios, seed=seed,
                   vanilla_results_csv=vanilla_csv, out_dir=out_dir)
            return

        if experiment == "5":
            if not disca_csv.exists():
                raise FileNotFoundError(f"disca results CSV not found: {disca_csv} (set DISCA_RESULTS_CSV)")
            m, tk = _ensure_model()
            _timed("Exp5", run_experiment_5, m, tk,
                   countries=countries, n_scenarios=n_scenarios, seed=seed,
                   disca_results_csv=disca_csv, out_dir=out_dir)
            return

        if experiment == "6":
            m, tk = _ensure_model()
            _timed("Exp6", run_experiment_6, m, tk,
                   countries=exp6_countries, ablation_labels=exp6_ablations,
                   n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)
            return

        if experiment == "7":
            if not scenario_csv.exists():
                raise FileNotFoundError(f"scenario CSV not found: {scenario_csv}")
            if not disca_csv.exists():
                raise FileNotFoundError(f"disca/main results CSV not found: {disca_csv}")
            _timed("Exp7", run_experiment_7,
                   scenario_csv=scenario_csv, disca_results_csv=disca_csv, out_dir=out_dir)
            return

        if experiment == "8":
            m, tk = _ensure_model()
            _timed("Exp8", run_experiment_8, m, tk,
                   countries=exp8_countries, n_values=exp8_n_values,
                   n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)
            return

        if experiment == "9":
            if (str(model_amce_csv) == "" or str(human_amce_long) == ""
                    or not model_amce_csv.exists() or not human_amce_long.exists()):
                raise FileNotFoundError(
                    "Exp 9 requires MODEL_AMCE_CSV and HUMAN_AMCE_LONG_CSV "
                    "(both long-form CSVs with country,dimension,{model,human}_amce)"
                )
            _timed("Exp9", run_experiment_9,
                   disca_model_amce_csv=model_amce_csv, human_amce_csv=human_amce_long, out_dir=out_dir)
            return

        if experiment == "10":
            if not scenario_csv.exists():
                raise FileNotFoundError(f"scenario CSV not found: {scenario_csv} (run Exp 1 first)")
            _timed("Exp10", run_experiment_10, scenario_csv=scenario_csv, out_dir=out_dir)
            return

        if experiment == "11":
            if not vanilla_per_dim_csv.exists() or not disca_per_dim_csv.exists():
                raise FileNotFoundError("Exp 11 requires VANILLA_PER_DIM_CSV and DISCA_PER_DIM_CSV")
            _timed("Exp11", run_experiment_11,
                   vanilla_per_dim_csv=vanilla_per_dim_csv,
                   disca_per_dim_csv=disca_per_dim_csv, out_dir=out_dir)
            return

        if experiment == "12":
            m, tk = _ensure_model()
            _timed("Exp12", run_experiment_12, m, tk,
                   countries=exp12_countries, n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)
            return

        # ── experiment == "all" : run everything we can, skip what's missing ──
        print("\n[PLAYBOOK] Running full sequence (Exp 1..12 where inputs are available)")
        m, tk = _ensure_model()
        skipped: List[str] = []

        # Exp 1
        csv_exp1 = _timed("Exp1", run_experiment_1, m, tk,
                          countries=countries, n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)
        _timed("Exp1-plot", plot_experiment_1, csv_path=csv_exp1, out_dir=out_dir)
        exp1_csv_auto = csv_exp1 if csv_exp1.exists() else scenario_csv

        # Exp 2 (model-free)
        if exp1_csv_auto.exists() and main_results_csv.exists():
            _timed("Exp2", run_experiment_2,
                   scenario_csv=exp1_csv_auto, main_results_csv=main_results_csv, out_dir=out_dir)
        else:
            skipped.append("Exp2")
            print(f"[PLAYBOOK][SKIP] Exp2 needs: {exp1_csv_auto} and {main_results_csv}")

        # Exp 3
        _timed("Exp3", run_experiment_3, m, tk,
               countries=countries, n_scenarios=n_scenarios, seeds=seeds, out_dir=out_dir)

        # Exp 4
        if vanilla_csv.exists():
            _timed("Exp4", run_experiment_4, m, tk,
                   countries=countries, n_scenarios=n_scenarios, seed=seed,
                   vanilla_results_csv=vanilla_csv, out_dir=out_dir)
        else:
            skipped.append("Exp4")
            print(f"[PLAYBOOK][SKIP] Exp4 needs: {vanilla_csv}")

        # Exp 5
        if disca_csv.exists():
            _timed("Exp5", run_experiment_5, m, tk,
                   countries=countries, n_scenarios=n_scenarios, seed=seed,
                   disca_results_csv=disca_csv, out_dir=out_dir)
        else:
            skipped.append("Exp5")
            print(f"[PLAYBOOK][SKIP] Exp5 needs: {disca_csv}")

        # Exp 6
        _timed("Exp6", run_experiment_6, m, tk,
               countries=exp6_countries, ablation_labels=exp6_ablations,
               n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)

        # Exp 7 (model-free)
        if exp1_csv_auto.exists() and disca_csv.exists():
            _timed("Exp7", run_experiment_7,
                   scenario_csv=exp1_csv_auto, disca_results_csv=disca_csv, out_dir=out_dir)
        else:
            skipped.append("Exp7")
            print(f"[PLAYBOOK][SKIP] Exp7 needs: {exp1_csv_auto} and {disca_csv}")

        # Exp 8
        _timed("Exp8", run_experiment_8, m, tk,
               countries=exp8_countries, n_values=exp8_n_values,
               n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)

        # Exp 9 (model-free)
        if model_amce_csv.exists() and human_amce_long.exists():
            _timed("Exp9", run_experiment_9,
                   disca_model_amce_csv=model_amce_csv, human_amce_csv=human_amce_long, out_dir=out_dir)
        else:
            skipped.append("Exp9")
            print(f"[PLAYBOOK][SKIP] Exp9 needs MODEL_AMCE_CSV and HUMAN_AMCE_LONG_CSV")

        # Exp 10 (model-free)
        if exp1_csv_auto.exists():
            _timed("Exp10", run_experiment_10, scenario_csv=exp1_csv_auto, out_dir=out_dir)
        else:
            skipped.append("Exp10")
            print(f"[PLAYBOOK][SKIP] Exp10 needs scenario CSV")

        # Exp 11 (model-free)
        if vanilla_per_dim_csv.exists() and disca_per_dim_csv.exists():
            _timed("Exp11", run_experiment_11,
                   vanilla_per_dim_csv=vanilla_per_dim_csv,
                   disca_per_dim_csv=disca_per_dim_csv, out_dir=out_dir)
        else:
            skipped.append("Exp11")
            print(f"[PLAYBOOK][SKIP] Exp11 needs VANILLA_PER_DIM_CSV and DISCA_PER_DIM_CSV")

        # Exp 12
        _timed("Exp12", run_experiment_12, m, tk,
               countries=exp12_countries, n_scenarios=n_scenarios, seed=seed, out_dir=out_dir)

        print("\n[PLAYBOOK] Sequence complete.")
        if skipped:
            print(f"[PLAYBOOK] Skipped (missing inputs): {', '.join(skipped)}")
    finally:
        session_elapsed = time.time() - session_t0
        print("\n" + "=" * 70)
        print("  DISCA PLAYBOOK — session summary")
        print("=" * 70)
        print(f"  total wall-clock            : {session_elapsed:.1f}s "
              f"({session_elapsed/60:.1f} min)")
        if exp_timings:
            print("  per-experiment timing:")
            for name, sec in sorted(exp_timings.items(),
                                    key=lambda kv: kv[1], reverse=True):
                print(f"    {name:<15} : {sec:7.1f}s ({sec/60:5.1f} min)")
        if USE_LOGIT_CACHE:
            print(f"  logit-cache stats           : "
                  f"hits={_LOGIT_CACHE_STATS['hits']}  "
                  f"misses={_LOGIT_CACHE_STATS['misses']}  "
                  f"writes={_LOGIT_CACHE_STATS['writes']}")
            total = _LOGIT_CACHE_STATS['hits'] + _LOGIT_CACHE_STATS['misses']
            if total > 0:
                hit_rate = _LOGIT_CACHE_STATS['hits'] / total * 100
                print(f"  cache hit rate              : {hit_rate:.1f}%")
        print(f"  output directory            : {out_dir}")
        print("=" * 70)

        if USE_LOGIT_CACHE:
            _flush_all_caches(out_dir)
        if model is not None:
            del model, tokenizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()

