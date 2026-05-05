#!/usr/bin/env python3
"""
Paper sweep — Qwen2.5-7B-Instruct — N-pass DPBR ablation, 20 countries
======================================================================

Question: the paper uses dual-pass (N=2). What about N = 1, 3, 4, 5, 6, …?

This script answers that without paying 6× GPU cost. It runs the model **once**
across all 20 countries and caches per-scenario debiased logit gaps, then
replays the cache through arbitrary N-pass DPBR variants in pure numpy.

Generalised DPBR rule (matches paper exactly at N=2):
    bootstrap_var = 2 · Var_ddof1(δ*₁,…,δ*ₙ)        # = (a-b)² when N=2
    r             = exp(-bootstrap_var / VAR_SCALE)
    δ*            = r · mean(δ*₁,…,δ*ₙ)
At N=1: r = 1, δ* = δ*₁.

Kaggle OFFLINE setup (Internet OFF — same pattern as exp_paper_ablation_qwen25_7b.py):
  Required Kaggle inputs:
    1. Repo dataset:  cultural_alignment   (any version with src/, exp_paper/,
                       experiment_DM/exp_npass_dpbr.py present)
    2. Model input:   Qwen/Qwen2.5-7B-Instruct  (mounted under
                       /kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1,
                       or override with MORAL_MODEL_PATH env)
    3. Data dataset:  trungkiet/mutltitp-data   (provides MultiTP + WVS + ACME)

  Hard-wired paths (do NOT git-clone / pip-download — Internet stays OFF):
    PROJECT_DATASET_DIR     = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
    PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
    MODEL_LOCAL_PATH        = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
    MULTITP_DATA_PATH       = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
    WVS_DATA_PATH           = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    HUMAN_AMCE_PATH         = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    WORK_DIR                = "/kaggle/working/cultural_alignment"

  Backend: HF native bf16 (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
  Qwen2.5-7B @ 7.6B does NOT fit on a single 16 GB T4 once 5-agent batched
  forward + activations are added (bare weights ≈ 15 GB; OOM during DPBR
  extraction). On Kaggle pick the **GPU T4 x2** accelerator so the script
  shards the model across both GPUs via accelerate (MORAL_DEVICE_MAP=auto,
  default in this script). Single-GPU alternatives: L4 (24 GB) or P100 (16 GB,
  marginal).

Run:
    !python /kaggle/working/cultural_alignment/exp_paper/models/exp_paper_phi35_mini_npass_ablation.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path as _P

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  KAGGLE OFFLINE BOOTSTRAP — no Internet, no git clone, no pip download    ║
# ║  (mirrors exp_paper/ablation/exp_paper_ablation_qwen25_7b.py pattern)     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  KAGGLE OFFLINE — Qwen2.5-7B N-pass DPBR Ablation (HF native, bf16)")
print("  20 paper countries  ×  N ∈ {1, 2, 3, 4, 5, 6}")
print("=" * 70)

# Repo (mounted as Kaggle Dataset)
PROJECT_DATASET_DIR     = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
WORK_DIR                = "/kaggle/working/cultural_alignment"

# Qwen2.5-7B-Instruct (mounted as Kaggle Model input). Override with env
# MORAL_MODEL_PATH if your Kaggle Model dataset is at a different mount point.
MODEL_LOCAL_PATH = os.environ.get(
    "MORAL_MODEL_PATH",
    "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
)

# Datasets (mounted from /kaggle/input)
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# ── Offline env vars (set BEFORE any HF / torch / transformers import) ─────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_AUTO_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Backend: HF native bf16 — most reliable on offline Kaggle (no Unsloth weight
# downloads, no vLLM JIT). Qwen2.5-7B @ 7.6B is too big for a single T4 16 GB
# once activations + 5-agent batched forward are added. Spread across both T4s
# via accelerate device_map="auto" (Kaggle "GPU T4 x2" accelerator).
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")
os.environ.setdefault("MORAL_DEVICE_MAP", "auto")  # T4×2 sharding
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    """Copy repo from read-only /kaggle/input to writable /kaggle/working.
    Local dev path: trust the existing cwd if it's a repo root."""
    if _on_kaggle():
        project_src = next(
            (p for p in (PROJECT_DATASET_DIR, PROJECT_DATASET_DIR_ALT)
             if os.path.isdir(os.path.join(p, "src"))),
            None,
        )
        if os.path.isdir(WORK_DIR) and os.path.isfile(
            os.path.join(WORK_DIR, "src", "controller.py")
        ):
            print(f"[OFFLINE] Reusing existing {WORK_DIR}")
        else:
            if project_src is None:
                raise RuntimeError(
                    "Project dataset not found. Looked in:\n"
                    f"  {PROJECT_DATASET_DIR}\n  {PROJECT_DATASET_DIR_ALT}"
                )
            print(f"[OFFLINE] Copying repo: {project_src} → {WORK_DIR}")
            shutil.copytree(project_src, WORK_DIR, dirs_exist_ok=True)
        os.chdir(WORK_DIR)
        sys.path.insert(0, WORK_DIR)
        return WORK_DIR
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        sys.path.insert(0, here)
        return here
    raise RuntimeError(
        "Not on Kaggle and not inside a repo root with src/controller.py."
    )


def _resolve_model_path() -> str:
    """Resolve local Qwen2.5-7B-Instruct weights — never hits the network.
    Tries MODEL_LOCAL_PATH directly, then any subdir containing config.json,
    then a few common Kaggle Model layout fallbacks."""
    if not _on_kaggle():
        return MODEL_LOCAL_PATH
    p = MODEL_LOCAL_PATH
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
        return p
    if os.path.isdir(p):
        for sub in _P(p).rglob("config.json"):
            return str(sub.parent)
    candidates = [
        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/2",
        f"{p}/transformers/default/1",
        f"{p}/transformers/7b-instruct/1",
        f"{p}/pytorch/default/1",
        "/kaggle/input/qwen-2.5/transformers/7b-instruct/1",
        "/kaggle/input/qwen2.5-7b-instruct/transformers/default/1",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise RuntimeError(
        f"Qwen2.5-7B weights not found. Looked in:\n  {p}\nand common subdirs.\n"
        "Add the Kaggle Model 'Qwen/Qwen2.5-7B-Instruct' as input, then "
        "set MORAL_MODEL_PATH=/kaggle/input/<your-mount>."
    )


_setup_project()
MODEL_LOCAL_PATH_RESOLVED = _resolve_model_path()
print(f"[OFFLINE] Model: {MODEL_LOCAL_PATH_RESOLVED}")

# Defensive offline-only pip fallback (no network — only --no-index --no-deps
# from the pre-installed Kaggle wheel cache).
if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf 2>/dev/null || true",
        shell=True, check=False,
    )

# ─── Now safe to import paper runtime + repo modules ─────────────────────────
from exp_paper.paper_runtime import configure_paper_env  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()

# Defensively patch _base_dpbr globals so any indirect import sees offline paths.
import exp_model._base_dpbr as _dpbr  # noqa: E402

_dpbr.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_dpbr.WVS_DATA_PATH     = WVS_DATA_PATH
_dpbr.HUMAN_AMCE_PATH   = HUMAN_AMCE_PATH

# ─── Stdlib + heavy deps (after sys.path is set) ─────────────────────────────
import gc                                                     # noqa: E402
import json                                                   # noqa: E402
import time                                                   # noqa: E402
from pathlib import Path                                      # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple      # noqa: E402

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from tqdm.auto import tqdm                                    # noqa: E402

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from src.amce import (                                        # noqa: E402
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    compute_utilitarianism_slope,
    load_human_amce,
)
from src.baseline_runner import run_baseline_vanilla          # noqa: E402
from src.config import SWAConfig, resolve_output_dir          # noqa: E402
from src.constants import COUNTRY_LANG                        # noqa: E402
from src.data import load_multitp_dataset                     # noqa: E402
from src.model import load_model_hf_native, setup_seeds       # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.scenarios import generate_multitp_scenarios          # noqa: E402

from experiment_DM.exp24_dpbr_core import (                   # noqa: E402
    BootstrapPriorState,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
)
from experiment_DM.exp_npass_dpbr import (                    # noqa: E402
    LoggingDPBRController,
    cache_path_for,
    load_logit_cache,
    npass_bootstrap_var,
    replay_country_npass,
    save_logit_cache,
)
from exp_paper.paper_countries import (                       # noqa: E402
    PAPER_20_COUNTRIES,
    RESULTS_BASE_EXP24_20C,
)


# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = MODEL_LOCAL_PATH_RESOLVED  # local Kaggle Model path (offline)
MODEL_SHORT  = "qwen25_7b"
EXP_BASE     = "EXP-24-NPASS"

N_SCENARIOS  = 500
BATCH_SIZE   = 1
SEED         = 42
LAMBDA_COOP  = 0.70

# Default ablation grid. Override with EXP24_NPASS_GRID="1,2,3,4,5,6,8,12".
N_PASSES_GRID: List[int] = [
    int(v) for v in os.environ.get("EXP24_NPASS_GRID", "1,2,3,4,5,6").split(",") if v.strip()
]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _build_cfg(model_name: str, swa_root: str, target_countries: Sequence[str]) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(target_countries),
        load_in_4bit=False,  # HF native bf16 (offline-friendly, fits on T4)
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=swa_root,
        lambda_coop=LAMBDA_COOP,
        K_samples=128,
    )


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _row_meta(idx: int, row: pd.Series, lang: str) -> Dict:
    return {
        "scenario_idx": int(idx),
        "Prompt": row.get("Prompt", row.get("prompt", "")),
        "phenomenon_category": row.get("phenomenon_category", "default"),
        "this_group_name": row.get("this_group_name", "Unknown"),
        "preferred_on_right": int(bool(row.get("preferred_on_right", 1))),
        "n_left": int(row.get("n_left", 1)),
        "n_right": int(row.get("n_right", 1)),
        "lang": lang,
    }


def _summarise_rows(results_df: pd.DataFrame, country: str, cfg: SWAConfig,
                    method: str, model_name: str) -> Dict:
    """Compute alignment metrics + per-dimension breakdown for a results_df."""
    model_amce = compute_amce_from_preferences(results_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment  = compute_alignment_metrics(model_amce, human_amce)
    per_dim    = compute_per_dimension_alignment(model_amce, human_amce)
    util       = compute_utilitarianism_slope(results_df)

    def _mea(c: str) -> float:
        return float(results_df[c].mean()) if c in results_df.columns and len(results_df) else float("nan")

    flip_rate = float(results_df["mppi_flipped"].mean()) if "mppi_flipped" in results_df.columns else float("nan")

    return {
        "model": model_name,
        "method": method,
        "country": country,
        **{f"align_{k}": v for k, v in alignment.items()},
        "flip_rate": flip_rate,
        "n_scenarios": len(results_df),
        "mean_reliability_r": _mea("reliability_r"),
        "mean_bootstrap_var": _mea("bootstrap_var"),
        "mean_ess_min": _mea("ess_min"),
        "mean_ess_anchor_alpha": _mea("ess_anchor_alpha"),
        "mean_positional_bias": _mea("positional_bias"),
        "utilitarianism_slope_hat": float(util.get("slope_hat", float("nan"))),
        "utilitarianism_slope_n": int(util.get("n_obs", 0) or 0),
        "per_dim_alignment": per_dim,
    }


# ─── PHASE 1: extraction (run model once, cache logits per country) ──────────
def run_extraction_for_country(
    model, tokenizer, country: str, personas: List[str],
    scen: pd.DataFrame, cfg: SWAConfig, out_dir: Path,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
    """Run vanilla baseline + standard DPBR (N=2) with logit logging.

    Returns
    -------
    (vanilla_results_df, vanilla_summary, dpbr_results_df, dpbr_summary).
    Side effect: writes the per-scenario logit cache parquet next to ``out_dir``.
    """
    lang = COUNTRY_LANG.get(country, "en")
    scen = scen.copy()
    if "lang" not in scen.columns:
        scen["lang"] = lang

    # 1) Vanilla baseline (no personas, no IS).
    print(f"  [VANILLA] {country} …")
    bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
    bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
    vanilla_summary = {
        "model": cfg.model_name, "method": "baseline_vanilla", "country": country,
        **{f"align_{k}": v for k, v in bl["alignment"].items()},
        "flip_rate": float("nan"), "n_scenarios": len(bl["results_df"]),
    }

    # 2) DPBR-with-logging (N=2, paper-canonical). Custom loop so we can attach
    #    per-scenario metadata to the controller's cache records.
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    controller = LoggingDPBRController(
        model=model, tokenizer=tokenizer, personas=personas,
        lambda_coop=cfg.lambda_coop, alpha_ctl=cfg.alpha_ctl,
        K_samples=cfg.K_samples, noise_std=cfg.noise_std,
        temperature=cfg.temperature, logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=cfg.category_logit_temperatures,
        pt_alpha=cfg.pt_alpha, pt_beta=cfg.pt_beta, pt_kappa=cfg.pt_kappa,
        decision_temperature=cfg.decision_temperature,
        assistant_lang=lang, country_iso=country,
    )

    rows: List[Dict] = []
    print(f"  [DPBR-N2 + cache] {country} …")
    for idx, row in tqdm(scen.iterrows(), total=len(scen), desc=f"DPBR/log [{country}]"):
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        meta = _row_meta(int(idx), row, lang)
        controller.attach_meta(meta)
        t0 = time.time()
        pred = controller.predict(
            prompt,
            preferred_on_right=bool(meta["preferred_on_right"]),
            phenomenon_category=meta["phenomenon_category"],
            lang=lang,
        )
        latency = time.time() - t0
        rewards = np.asarray(pred["agent_rewards"], dtype=float)
        rows.append({
            "country": country, "scenario_idx": int(idx),
            "Prompt": prompt,
            "phenomenon_category": meta["phenomenon_category"],
            "this_group_name": meta["this_group_name"],
            "preferred_on_right": meta["preferred_on_right"],
            "n_left": meta["n_left"], "n_right": meta["n_right"],
            "p_left": float(pred["p_left"]),
            "p_right": float(pred["p_right"]),
            "p_spare_preferred": float(pred["p_spare_preferred"]),
            "delta_consensus": float(pred["delta_consensus"]),
            "delta_opt": float(pred["delta_opt"]),
            "delta_opt_micro": float(pred.get("delta_opt_micro", pred["delta_opt"])),
            "reliability_r": float(pred.get("reliability_r", float("nan"))),
            "bootstrap_var": float(pred.get("bootstrap_var", float("nan"))),
            "ess_pass1": float(pred.get("ess_pass1", float("nan"))),
            "ess_pass2": float(pred.get("ess_pass2", float("nan"))),
            "ess_anchor_alpha": float(pred.get("ess_anchor_alpha", float("nan"))),
            "positional_bias": float(pred.get("positional_bias", 0.0)),
            "logit_temp_used": float(pred["logit_temp_used"]),
            "mppi_flipped": bool(pred["mppi_flipped"]),
            "delta_z_norm": float(pred["delta_z_norm"]),
            "mppi_variance": float(pred["variance"]),
            "agent_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "agent_reward_std": float(rewards.std()) if rewards.size else 0.0,
            "latency_ms": latency * 1000.0,
        })

    dpbr_results_df = pd.DataFrame(rows)
    dpbr_results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
    dpbr_summary = _summarise_rows(
        dpbr_results_df, country, cfg, method=f"{EXP_BASE}_dual_pass", model_name=cfg.model_name,
    )

    # 3) Persist the logit cache (one parquet per country).
    cache_df = pd.DataFrame(controller._cache_records)
    save_logit_cache(out_dir, country, cache_df)
    print(f"  [CACHE] saved {len(cache_df)} rows → {cache_path_for(out_dir, country)}")

    # Free the controller (model / tokenizer survive the call).
    del controller
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return bl["results_df"], vanilla_summary, dpbr_results_df, dpbr_summary


# ─── PHASE 2: replay (no model — pure numpy) ─────────────────────────────────
def run_replay_for_n(
    out_dir: Path, countries: Sequence[str], cfg: SWAConfig, n_passes: int,
) -> List[Dict]:
    """Replay each country's cached logits through ``n_passes`` IS passes.

    Prints a compact per-country line so progress is visible:
        [N=4] [ 7/20] VNM  MIS=0.5012  r=-0.412  rel_r=0.93  flip=14.2%
    """
    rows: List[Dict] = []
    method = f"{EXP_BASE}_n{n_passes}"
    t0_grid = time.time()
    for ci, country in enumerate(countries):
        cache_df = load_logit_cache(out_dir, country)
        if cache_df is None or cache_df.empty:
            print(f"  [N={n_passes}] [{ci+1:>2}/{len(countries)}] {country}: no cache, skipping")
            continue
        t0 = time.time()
        results_df = replay_country_npass(
            cache_df, country_iso=country, n_passes=n_passes,
            K_per_pass=K_HALF, var_scale=VAR_SCALE,
            lambda_coop=cfg.lambda_coop,
            pt_alpha=cfg.pt_alpha, pt_beta=cfg.pt_beta, pt_kappa=cfg.pt_kappa,
            beta_temp=cfg.temperature, rho_eff=0.1,
            decision_temperature=cfg.decision_temperature,
            noise_std_floor=cfg.noise_std,
            seed=SEED,
        )
        dt = time.time() - t0
        results_df.to_csv(out_dir / f"npass{n_passes}_results_{country}.csv", index=False)
        summary = _summarise_rows(results_df, country, cfg, method=method, model_name=cfg.model_name)
        rows.append(summary)
        mis = summary["align_mis"]
        prs = summary["align_pearson_r"]
        rel = summary["mean_reliability_r"]
        flp = summary["flip_rate"] * 100
        bv  = summary["mean_bootstrap_var"]
        print(f"  [N={n_passes}] [{ci+1:>2}/{len(countries)}] {country}  "
              f"MIS={mis:.4f}  r={prs:+.3f}  rel_r={rel:.3f}  bvar={bv:.4f}  "
              f"flip={flp:5.1f}%  ({dt:5.2f}s)")
    if rows:
        agg_mis = float(np.mean([r["align_mis"] for r in rows]))
        agg_r   = float(np.mean([r["align_pearson_r"] for r in rows]))
        agg_rel = float(np.nanmean([r["mean_reliability_r"] for r in rows]))
        agg_bv  = float(np.nanmean([r["mean_bootstrap_var"] for r in rows]))
        agg_flp = float(np.mean([r["flip_rate"] for r in rows])) * 100
        elapsed = time.time() - t0_grid
        print(f"  ── N={n_passes} mean-of-{len(rows)}: "
              f"MIS={agg_mis:.4f}  r={agg_r:+.3f}  rel_r={agg_rel:.3f}  "
              f"bvar={agg_bv:.4f}  flip={agg_flp:5.1f}%  ({elapsed:5.1f}s) ──")
    return rows


# ─── Paper-grade reporting ───────────────────────────────────────────────────
def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = SEED) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean. Empty / all-nan input → (nan, nan)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for b in range(n_boot):
        means[b] = arr[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def _agg_with_ci(per_country_rows: List[Dict], metric_keys: Sequence[str]) -> Dict:
    """Mean ± std + 95% bootstrap CI for each metric, computed across countries."""
    out: Dict = {}
    for k in metric_keys:
        vals = np.asarray([r.get(k, float("nan")) for r in per_country_rows], dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            out[f"{k}_mean"] = out[f"{k}_std"] = float("nan")
            out[f"{k}_ci_lo"] = out[f"{k}_ci_hi"] = float("nan")
            continue
        out[f"{k}_mean"] = float(finite.mean())
        out[f"{k}_std"]  = float(finite.std(ddof=1)) if finite.size >= 2 else 0.0
        lo, hi = _bootstrap_ci(finite)
        out[f"{k}_ci_lo"] = lo
        out[f"{k}_ci_hi"] = hi
    return out


def _aggregate_per_dim_by_n(all_replay_rows: List[Dict]) -> pd.DataFrame:
    """Per-dimension MIS (|model − human|) averaged across countries, per N.

    Output: rows = dimension, columns = N values (sorted ascending).
    Uses ``compute_per_dimension_alignment``'s 'abs_err' field when available,
    falling back to |model − human|.
    """
    rec: Dict[Tuple[int, str], List[float]] = {}
    for r in all_replay_rows:
        n = int(r["method"].split("_n")[-1])
        pda = r.get("per_dim_alignment") or {}
        for dim, dd in pda.items():
            err = dd.get("abs_err")
            if err is None:
                hv = dd.get("human", float("nan"))
                mv = dd.get("model", float("nan"))
                err = abs(float(mv) - float(hv))
            rec.setdefault((n, dim), []).append(float(err))
    if not rec:
        return pd.DataFrame()
    n_values = sorted({k[0] for k in rec})
    dims     = sorted({k[1] for k in rec})
    rows = []
    for dim in dims:
        row = {"dimension": dim}
        for n in n_values:
            vals = rec.get((n, dim), [])
            row[f"N{n}_mean_abs_err"] = float(np.mean(vals)) if vals else float("nan")
            row[f"N{n}_std_abs_err"]  = float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _load_vanilla_per_country(cmp_root: Path) -> Optional[pd.DataFrame]:
    """Load Phase-1 vanilla baseline rows (per country) for wins-vs-vanilla.
    Survives across runs since the file is written to disk."""
    p = Path(cmp_root) / "phase1_vanilla_and_dpbr_n2.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df[df["method"] == "baseline_vanilla"].copy()


def _wins_vs_vanilla(per_country_rows: List[Dict], vanilla_df: Optional[pd.DataFrame]
                    ) -> Tuple[int, int, float, Tuple[float, float]]:
    """Per-N: count of countries where DPBR beats vanilla on MIS, plus
    paired Δ = MIS_vanilla − MIS_DPBR (positive = DPBR better).

    Returns (wins, total, mean_delta, ci_for_delta)."""
    if vanilla_df is None or vanilla_df.empty:
        return (0, 0, float("nan"), (float("nan"), float("nan")))
    vmap = dict(zip(vanilla_df["country"], vanilla_df["align_mis"]))
    deltas = []
    wins = 0
    for r in per_country_rows:
        v = vmap.get(r["country"])
        if v is None or np.isnan(v):
            continue
        d = float(v) - float(r["align_mis"])
        deltas.append(d)
        if d > 0:
            wins += 1
    if not deltas:
        return (0, 0, float("nan"), (float("nan"), float("nan")))
    arr = np.asarray(deltas, dtype=float)
    return (wins, len(deltas), float(arr.mean()), _bootstrap_ci(arr))


def _build_paper_report(
    all_replay_rows: List[Dict],
    cmp_root: Path,
    cfg: SWAConfig,
    countries: Sequence[str],
) -> None:
    """Compute + persist + print paper-grade aggregates.

    Writes:
      - ``npass_ablation_per_country.csv``      (already implied; we re-write here for safety)
      - ``npass_ablation_summary.csv``          (means only — kept for backward compat)
      - ``npass_ablation_summary_with_ci.csv``  (mean ± std + 95% CI; wins-vs-vanilla; Δ vs vanilla)
      - ``npass_per_dim_by_N.csv``              (per-dimension MIS for each N)
      - ``npass_paper_table.md``                (markdown — paste into LaTeX/paper)
    """
    cmp_root = Path(cmp_root)
    cmp_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-country flat dataframe (drop the per_dim_alignment dict for CSV).
    per_country_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "per_dim_alignment"} for r in all_replay_rows]
    )
    per_country_df.to_csv(cmp_root / "npass_ablation_per_country.csv", index=False)

    # 2) Per-N: split rows by N, compute mean / std / CI for the headline metrics.
    metric_keys = (
        "align_mis", "align_pearson_r", "align_jsd", "align_mae",
        "flip_rate", "mean_reliability_r", "mean_bootstrap_var",
        "mean_ess_anchor_alpha", "mean_ess_min", "mean_positional_bias",
    )
    by_n: Dict[int, List[Dict]] = {}
    for r in all_replay_rows:
        n = int(r["method"].split("_n")[-1])
        by_n.setdefault(n, []).append(r)
    n_values = sorted(by_n)

    vanilla_df = _load_vanilla_per_country(cmp_root)

    summary_rows = []
    summary_with_ci_rows = []
    for n in n_values:
        rows_n = by_n[n]
        wins, total, mean_delta, (delta_lo, delta_hi) = _wins_vs_vanilla(rows_n, vanilla_df)
        agg = _agg_with_ci(rows_n, metric_keys)
        summary_rows.append({
            "n_passes": n, "method": f"{EXP_BASE}_n{n}", "n_countries": len(rows_n),
            **{k.replace("align_", "mean_").replace("flip_rate", "mean_flip_rate")
               .replace("mean_reliability_r", "mean_reliability_r")
               .replace("mean_bootstrap_var", "mean_bootstrap_var"):
               agg[f"{k}_mean"] for k in metric_keys},
        })
        summary_with_ci_rows.append({
            "n_passes": n, "method": f"{EXP_BASE}_n{n}", "n_countries": len(rows_n),
            **agg,
            "wins_vs_vanilla": wins, "wins_total": total,
            "delta_mis_vs_vanilla_mean": mean_delta,
            "delta_mis_vs_vanilla_ci_lo": delta_lo,
            "delta_mis_vs_vanilla_ci_hi": delta_hi,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("n_passes").reset_index(drop=True)
    summary_with_ci_df = pd.DataFrame(summary_with_ci_rows).sort_values("n_passes").reset_index(drop=True)

    summary_df.to_csv(cmp_root / "npass_ablation_summary.csv", index=False)
    summary_with_ci_df.to_csv(cmp_root / "npass_ablation_summary_with_ci.csv", index=False)

    # 3) Per-dimension MIS by N.
    per_dim_df = _aggregate_per_dim_by_n(all_replay_rows)
    if not per_dim_df.empty:
        per_dim_df.to_csv(cmp_root / "npass_per_dim_by_N.csv", index=False)

    # 4) Markdown paper table.
    md_path = cmp_root / "npass_paper_table.md"
    md_lines: List[str] = []
    md_lines.append(f"# {EXP_BASE} ablation — Qwen2.5-7B-Instruct, 20 countries\n")
    md_lines.append(f"Generalised DPBR rule: bootstrap_var = 2·Var_ddof1(δ*₁,…,δ*ₙ); ")
    md_lines.append(f"r = exp(−bv / {VAR_SCALE}); δ* = r·mean(δ*ᵢ).\n")
    md_lines.append(f"K per pass = {K_HALF}.  λ_coop = {LAMBDA_COOP}.  N_scenarios = {cfg.n_scenarios}.\n\n")
    md_lines.append("## Headline metrics (mean ± std across 20 countries; 95% bootstrap CI)\n")
    md_lines.append("| N | MIS ↓ | Pearson r ↑ | JSD ↓ | MAE ↓ | Flip% | mean rel_r | wins/20 | Δ MIS vs vanilla |")
    md_lines.append("|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in summary_with_ci_df.iterrows():
        n = int(r["n_passes"])
        wt = int(r["wins_total"]) if not np.isnan(r["wins_total"]) else 0
        wn = int(r["wins_vs_vanilla"]) if not np.isnan(r["wins_vs_vanilla"]) else 0
        wins_str = f"{wn}/{wt}" if wt > 0 else "—"
        delta_str = (
            f"{r['delta_mis_vs_vanilla_mean']:+.4f} "
            f"[{r['delta_mis_vs_vanilla_ci_lo']:+.4f},{r['delta_mis_vs_vanilla_ci_hi']:+.4f}]"
            if not np.isnan(r["delta_mis_vs_vanilla_mean"]) else "—"
        )
        md_lines.append(
            f"| {n} "
            f"| {r['align_mis_mean']:.4f}±{r['align_mis_std']:.4f} "
              f"[{r['align_mis_ci_lo']:.4f},{r['align_mis_ci_hi']:.4f}] "
            f"| {r['align_pearson_r_mean']:+.3f}±{r['align_pearson_r_std']:.3f} "
            f"| {r['align_jsd_mean']:.4f}±{r['align_jsd_std']:.4f} "
            f"| {r['align_mae_mean']:.2f}±{r['align_mae_std']:.2f} "
            f"| {r['flip_rate_mean']*100:.1f}% "
            f"| {r['mean_reliability_r_mean']:.3f} "
            f"| {wins_str} "
            f"| {delta_str} |"
        )
    md_lines.append("")

    if not per_dim_df.empty:
        md_lines.append("\n## Per-dimension mean |error| (model − human MPR), averaged across 20 countries\n")
        n_cols = [c for c in per_dim_df.columns if c.startswith("N") and c.endswith("_mean_abs_err")]
        n_label = [int(c.split("_")[0][1:]) for c in n_cols]
        header = "| dimension | " + " | ".join(f"N={n}" for n in n_label) + " |"
        sep    = "|---" + "|---:" * len(n_cols) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for _, dr in per_dim_df.iterrows():
            cells = [f"{dr[c]:.2f}" for c in n_cols]
            md_lines.append(f"| {dr['dimension']} | " + " | ".join(cells) + " |")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ── Console: full paper-grade printout ─────────────────────────────────
    print(f"\n{'#'*88}")
    print(f"# {EXP_BASE} FINAL REPORT — {MODEL_NAME}")
    print(f"{'#'*88}")
    print(f"\nFiles written under  {cmp_root}:")
    for fn in ("npass_ablation_per_country.csv",
               "npass_ablation_summary.csv",
               "npass_ablation_summary_with_ci.csv",
               "npass_per_dim_by_N.csv",
               "npass_paper_table.md"):
        p = cmp_root / fn
        size = p.stat().st_size if p.is_file() else 0
        flag = "✓" if size > 0 else "·"
        print(f"  {flag} {fn:<42s}  ({size:>7d} bytes)")

    # Headline table — wide, with std + CI + wins.
    print(f"\n┌─ HEADLINE (mean ± std across {len(countries)} countries; 95% bootstrap CI) ─┐")
    print(f"  {'N':>2}  "
          f"{'MIS ↓':>22}  {'r ↑':>14}  {'JSD ↓':>14}  "
          f"{'MAE ↓':>13}  {'flip%':>6}  {'rel_r':>6}  {'wins/20':>8}  {'ΔMIS vs vanilla':>22}")
    print("  " + "-" * 134)
    for _, r in summary_with_ci_df.iterrows():
        n = int(r["n_passes"])
        wt = int(r["wins_total"]) if not np.isnan(r["wins_total"]) else 0
        wn = int(r["wins_vs_vanilla"]) if not np.isnan(r["wins_vs_vanilla"]) else 0
        wins_str = f"{wn:>2}/{wt:<2}" if wt > 0 else "—"
        if not np.isnan(r["delta_mis_vs_vanilla_mean"]):
            d_str = (f"{r['delta_mis_vs_vanilla_mean']:+.4f} "
                     f"[{r['delta_mis_vs_vanilla_ci_lo']:+.4f},{r['delta_mis_vs_vanilla_ci_hi']:+.4f}]")
        else:
            d_str = "—"
        mis_str = (f"{r['align_mis_mean']:.4f}±{r['align_mis_std']:.4f} "
                   f"[{r['align_mis_ci_lo']:.4f},{r['align_mis_ci_hi']:.4f}]")
        r_str   = f"{r['align_pearson_r_mean']:+.3f}±{r['align_pearson_r_std']:.3f}"
        jsd_str = f"{r['align_jsd_mean']:.4f}±{r['align_jsd_std']:.4f}"
        mae_str = f"{r['align_mae_mean']:5.2f}±{r['align_mae_std']:4.2f}"
        print(f"  {n:>2}  "
              f"{mis_str:>22}  {r_str:>14}  {jsd_str:>14}  "
              f"{mae_str:>13}  {r['flip_rate_mean']*100:>5.1f}%  "
              f"{r['mean_reliability_r_mean']:>6.3f}  {wins_str:>8}  {d_str:>22}")

    # Per-dimension table.
    if not per_dim_df.empty:
        print(f"\n┌─ PER-DIMENSION |error| (mean abs MPR error vs human, averaged across countries) ─┐")
        n_cols = [c for c in per_dim_df.columns if c.startswith("N") and c.endswith("_mean_abs_err")]
        n_label = [int(c.split("_")[0][1:]) for c in n_cols]
        header = f"  {'dimension':<24s}  " + "  ".join(f"{f'N={n}':>8s}" for n in n_label)
        print(header)
        print("  " + "-" * (24 + 10 * len(n_cols)))
        for _, dr in per_dim_df.iterrows():
            cells = "  ".join(f"{dr[c]:>8.2f}" for c in n_cols)
            print(f"  {dr['dimension']:<24s}  {cells}")

    # Recommendation: best N by MIS, plus by Δ-vs-vanilla.
    best_mis = summary_with_ci_df.sort_values("align_mis_mean").iloc[0]
    print(f"\n[BEST by MIS]      N={int(best_mis['n_passes'])}: "
          f"MIS={best_mis['align_mis_mean']:.4f}±{best_mis['align_mis_std']:.4f} "
          f"(95% CI [{best_mis['align_mis_ci_lo']:.4f},{best_mis['align_mis_ci_hi']:.4f}])")
    if "delta_mis_vs_vanilla_mean" in summary_with_ci_df.columns and \
       summary_with_ci_df["delta_mis_vs_vanilla_mean"].notna().any():
        best_delta = summary_with_ci_df.sort_values("delta_mis_vs_vanilla_mean", ascending=False).iloc[0]
        print(f"[BEST by Δ-vs-van] N={int(best_delta['n_passes'])}: "
              f"ΔMIS={best_delta['delta_mis_vs_vanilla_mean']:+.4f} "
              f"(95% CI [{best_delta['delta_mis_vs_vanilla_ci_lo']:+.4f},"
              f"{best_delta['delta_mis_vs_vanilla_ci_hi']:+.4f}]) "
              f"wins={int(best_delta['wins_vs_vanilla'])}/{int(best_delta['wins_total'])}")

    # N=2 sanity check vs Phase-1 canonical row (if available).
    if vanilla_df is not None:
        # Try loading canonical N=2 row from phase1 file.
        p1 = Path(cmp_root) / "phase1_vanilla_and_dpbr_n2.csv"
        if p1.is_file():
            p1df = pd.read_csv(p1)
            canon = p1df[p1df["method"] == f"{EXP_BASE}_dual_pass"]
            if not canon.empty:
                canon_mis = float(canon["align_mis"].mean())
                replayed_n2 = summary_with_ci_df[summary_with_ci_df["n_passes"] == 2]
                if not replayed_n2.empty:
                    rep_mis = float(replayed_n2["align_mis_mean"].iloc[0])
                    diff = rep_mis - canon_mis
                    print(f"\n[SANITY] N=2 replay MIS = {rep_mis:.4f}  "
                          f"|  Phase-1 canonical N=2 MIS = {canon_mis:.4f}  "
                          f"|  Δ = {diff:+.4f}  "
                          f"(should be ~0; large drift → cache or RNG bug)")

    # Markdown preview to console (first ~30 lines).
    print(f"\n┌─ npass_paper_table.md  (preview — full file at {md_path}) ─┐")
    md_text = md_path.read_text(encoding="utf-8")
    for line in md_text.splitlines()[:32]:
        print(f"  {line}")
    if len(md_text.splitlines()) > 32:
        print(f"  ... ({len(md_text.splitlines()) - 32} more lines in file)")

    print(f"\n[{EXP_BASE}] DONE — {cmp_root}")


# ─── Main orchestrator ───────────────────────────────────────────────────────
def main() -> None:
    setup_seeds(SEED)
    countries = list(PAPER_20_COUNTRIES)

    rb       = f"{RESULTS_BASE_EXP24_20C}_npass_ablation"
    swa_root = f"{rb}/{MODEL_SHORT}/swa"
    cmp_root = f"{rb}/{MODEL_SHORT}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    cfg     = _build_cfg(MODEL_NAME, swa_root, target_countries=countries)
    out_dir = Path(swa_root) / resolve_output_dir("", MODEL_NAME).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    print(f"\n{'='*72}")
    print(f"  {EXP_BASE}: N-pass DPBR ablation [{MODEL_NAME}]")
    print(f"{'='*72}")
    print(f"[GRID] N values        = {N_PASSES_GRID}")
    print(f"[CONFIG] K_per_pass    = {K_HALF}")
    print(f"[CONFIG] VAR_SCALE     = {VAR_SCALE}")
    print(f"[CONFIG] λ_coop        = {LAMBDA_COOP}")
    print(f"[CONFIG] N_scenarios   = {N_SCENARIOS}  ×  {len(countries)} countries")
    print(f"[CONFIG] cache dir     = {out_dir}")
    print(f"[OFFLINE] MultiTP      = {MULTITP_DATA_PATH}")
    print(f"[OFFLINE] WVS          = {WVS_DATA_PATH}")
    print(f"[OFFLINE] human AMCE   = {HUMAN_AMCE_PATH}")

    # ── PHASE 1: load model once, sweep all countries ──────────────────────
    skip_extraction = all(cache_path_for(out_dir, c).is_file() for c in countries)
    extraction_rows: List[Dict] = []
    if skip_extraction and os.environ.get("EXP24_NPASS_FORCE_EXTRACT", "0") != "1":
        print("\n[PHASE-1] all per-country logit caches present → skipping extraction.")
        print("          (set EXP24_NPASS_FORCE_EXTRACT=1 to recompute)")
    else:
        t_phase1 = time.time()
        print("\n[PHASE-1] extracting logits (one-time GPU sweep across 20 countries)")
        model, tokenizer = load_model_hf_native(
            MODEL_NAME, max_seq_length=2048, load_in_4bit=False,
        )
        try:
            for ci, country in enumerate(countries):
                if country not in SUPPORTED_COUNTRIES:
                    print(f"[SKIP] unsupported country: {country}")
                    continue
                t_country = time.time()
                print(f"\n[{ci+1:>2}/{len(countries)}] {EXP_BASE} | {country}")
                scen     = _load_scen(cfg, country)
                personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
                _, vanilla_summary, _, dpbr_summary = run_extraction_for_country(
                    model, tokenizer, country, personas, scen, cfg, out_dir,
                )
                extraction_rows.extend([vanilla_summary, dpbr_summary])
                v_mis = vanilla_summary.get("align_mis", float("nan"))
                d_mis = dpbr_summary.get("align_mis", float("nan"))
                d_r   = dpbr_summary.get("align_pearson_r", float("nan"))
                print(f"  [{country}] vanilla MIS={v_mis:.4f}  "
                      f"DPBR-N2 MIS={d_mis:.4f}  r={d_r:+.3f}  "
                      f"Δ={v_mis - d_mis:+.4f}  ({time.time()-t_country:5.1f}s)")
                torch.cuda.empty_cache()
                gc.collect()
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Persist phase-1 summaries.
        if extraction_rows:
            pd.DataFrame([{k: v for k, v in r.items() if k != "per_dim_alignment"}
                          for r in extraction_rows]).to_csv(
                Path(cmp_root) / "phase1_vanilla_and_dpbr_n2.csv", index=False,
            )
            v_rows = [r for r in extraction_rows if r["method"] == "baseline_vanilla"]
            d_rows = [r for r in extraction_rows if r["method"].endswith("_dual_pass")]
            v_mis_avg = float(np.mean([r["align_mis"] for r in v_rows])) if v_rows else float("nan")
            d_mis_avg = float(np.mean([r["align_mis"] for r in d_rows])) if d_rows else float("nan")
            print(f"\n[PHASE-1] mean across {len(v_rows)} countries:  "
                  f"vanilla MIS = {v_mis_avg:.4f}   "
                  f"DPBR-N2 MIS = {d_mis_avg:.4f}   "
                  f"Δ = {v_mis_avg - d_mis_avg:+.4f}")
        print(f"[PHASE-1] DONE in {(time.time()-t_phase1)/60:.1f} min — caches under {out_dir}")

    # ── PHASE 2: replay each N from cached logits ──────────────────────────
    t_phase2 = time.time()
    print("\n[PHASE-2] replaying cached logits for each N (no GPU required)")
    all_replay_rows: List[Dict] = []
    for n in N_PASSES_GRID:
        print(f"\n  ── N = {n}  (K_per_pass={K_HALF}, total IS samples = {n * K_HALF}) ──")
        all_replay_rows.extend(run_replay_for_n(out_dir, countries, cfg, n_passes=n))

    if not all_replay_rows:
        print("[PHASE-2] no replay rows produced — aborting")
        return
    print(f"\n[PHASE-2] DONE in {(time.time()-t_phase2)/60:.1f} min "
          f"({len(all_replay_rows)} (N,country) cells)")

    # ── Build paper-grade report (CSV + markdown + rich console printout) ──
    _build_paper_report(all_replay_rows, Path(cmp_root), cfg, countries)


if __name__ == "__main__":
    main()
