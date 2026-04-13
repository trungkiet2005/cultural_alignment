#!/usr/bin/env python3
"""
Persona + Debiasing Baseline Sweep — All Paper Models × 20 Countries
======================================================================
Stripped-down SWA-DPBR that keeps ONLY the two load-bearing components:
  • WVS cultural personas  (4 agents per country)
  • Positional A↔B debiasing  (linear cancellation of framing artifacts)

Everything else is disabled:
  ✗  Importance sampling         (δ* ≡ 0)
  ✗  Dual-pass reliability       (r ≡ 1, but irrelevant since IS=0)
  ✗  Hierarchical country prior  (α_h ≡ 0)

Purpose: verify whether persona+debiasing alone reproduces (or exceeds) the
MIS gains reported in Table 3 of the paper, and provide a fast cross-model
baseline for the 20 PAPER_20_COUNTRIES pool.

Optimised for a single H100 80 GB SXM (single-GPU):
  • vLLM bf16 with gpu_memory_utilization=0.92
  • max_model_len=4096 (H100 VRAM headroom)
  • enable_chunked_prefill via env (reduces peak memory spikes)
  • Auto 4-bit for ≥70B models (stays in 80 GB)
  • Model weights freed + CUDA cache cleared between models

Usage — Kaggle notebook cell:
    !python exp_paper/exp_persona_debiasing_sweep.py

Local (inside repo root):
    python exp_paper/exp_persona_debiasing_sweep.py

Environment overrides:
    PD_MODELS           comma-separated model shorts to run, e.g. phi_4,llama33_70b
                        (default: all registry models)
    PD_COUNTRIES        comma-separated ISO-3 codes (default: all 20 paper countries)
    PD_N_SCENARIOS      int (default: 500)
    PD_SEED             int (default: 42)
    MORAL_MODEL_BACKEND vllm|hf_native|unsloth  (default: vllm)
    MORAL_VLLM_GPU_MEM  float (default: 0.92 — H100 headroom)
    PD_LOAD_TIMEOUT     int minutes per model load (default: 20)
    PD_SKIP_DONE        1 to skip countries whose CSV already exists (default: 1)
"""

from __future__ import annotations

import gc
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Repo bootstrap (identical to all exp_paper/* entry scripts) ───────────────
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        sys.path.insert(0, here)
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


_ensure_repo()

# ── H100 defaults (set BEFORE paper_runtime / vllm_env are imported) ─────────
# H100 80 GB can push memory utilisation higher than the Kaggle T4/P100 default.
os.environ.setdefault("MORAL_MODEL_BACKEND",  "vllm")
os.environ.setdefault("MORAL_VLLM_GPU_MEM",   "0.92")   # ~73.6 GB usable
os.environ.setdefault("MORAL_VLLM_MAX_SEQ_LEN", "4096") # longer ctx for H100
# Chunked prefill reduces peak VRAM spikes on H100 (vLLM ≥0.4)
os.environ.setdefault("VLLM_ENABLE_CHUNKED_PREFILL", "1")
# ESS anchor regularisation ON (matches paper §4.2) — harmless with IS=0
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")

from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env(vllm_gpu_mem_default="0.92")

from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

# ── Core imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch

try:
    torch._dynamo.config.disable = True          # type: ignore[attr-defined]
    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except Exception:
    pass

from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    Exp24DualPassController,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
    _use_ess_anchor_reg,
    ess_anchor_blend_alpha,
    positional_bias_logit_gap,
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
from src.model import load_model, load_model_hf_native, setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402

# ── Run configuration ─────────────────────────────────────────────────────────
N_SCENARIOS:    int   = int(os.environ.get("PD_N_SCENARIOS", "500"))
SEED:           int   = int(os.environ.get("PD_SEED",        "42"))
LOAD_TIMEOUT:   int   = int(os.environ.get("PD_LOAD_TIMEOUT", "20"))
SKIP_DONE:      bool  = os.environ.get("PD_SKIP_DONE", "1") == "1"
LAMBDA_COOP:    float = 0.70

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp_persona_debiasing"
    if _on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp_persona_debiasing")
)

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

DIM_KEYS = [
    "Species_Humans",
    "Gender_Female",
    "Age_Young",
    "Fitness_Fit",
    "SocialValue_High",
    "Utilitarianism_More",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PersonaDebiasingController
# Keeps personas + debiasing; disables IS and country prior entirely.
# ═══════════════════════════════════════════════════════════════════════════════

class _NullPriorState(BootstrapPriorState):
    """Prior state that always returns delta_opt_micro unchanged (α_h ≡ 0)."""

    def apply_prior(self, delta_opt_micro: float) -> float:
        return delta_opt_micro


class PersonaDebiasingController(Exp24DualPassController):
    """
    SWA-DPBR with ONLY persona + debiasing enabled.

    Pipeline per scenario:
      1.  Forward pass on original prompt  → δ_base_1, δ_agents_1
      2.  Forward pass on A↔B-swapped prompt → δ_base_2, δ_agents_2
      3.  Linear debiasing:
              δ_base   = (δ_base_1   − δ_base_2)   / 2
              δ_agents = (δ_agents_1 − δ_agents_2) / 2
      4.  Persona consensus anchor: δ_anchor = mean(δ_agents)
      5.  δ_opt_micro = δ_anchor          (no IS correction)
      6.  No country prior: δ_opt_final = δ_opt_micro
      7.  p_right = σ(δ_opt_final / T_decision)
    """

    # ── IS disabled: both passes return zero correction ───────────────────────
    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        K: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """Return δ* = 0; ESS = 1.0 (no IS applied)."""
        return torch.zeros((), device=device), 1.0

    # ── Country prior disabled ─────────────────────────────────────────────────
    def _get_prior(self) -> "_NullPriorState":
        key = f"__pd_{self.country}"
        if key not in PRIOR_STATE:
            PRIOR_STATE[key] = _NullPriorState()
        return PRIOR_STATE[key]  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# All models tested in Table 2 / Table 3 of the paper.
# Sorted by nominal parameter count (descending) — matches paper ordering.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    short:        str   # used for directory names and CSV column headers
    model_name:   str   # HuggingFace model id
    backend:      str   # "vllm" | "unsloth" | "hf_native"
    load_in_4bit: bool  # 4-bit quantisation (unsloth or vLLM bnb)
    params_b:     float # nominal parameter count in billions (for sorting)
    notes:        str = ""


# fmt: off
MODEL_REGISTRY: List[ModelSpec] = [
    # ── ≥70 B: 4-bit to fit H100 80 GB ──────────────────────────────────────
    ModelSpec("qwen25_72b",        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
              "unsloth", True,  72.0, "Qwen2.5-72B (4-bit, Unsloth)"),
    ModelSpec("llama33_70b",       "meta-llama/Llama-3.3-70B-Instruct",
              "vllm",    True,  70.0, "Llama-3.3-70B (vLLM auto-quant)"),
    ModelSpec("llama31_70b",       "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
              "unsloth", True,  70.0, "Llama-3.1-70B (4-bit, Unsloth)"),
    ModelSpec("gpt_oss_20b",       "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
              "unsloth", True,  20.0, "GPT-OSS-20B (4-bit, Unsloth)"),
    # ── 20–30 B: bf16 fits in 80 GB ──────────────────────────────────────────
    ModelSpec("magistral_small",   "mistralai/Magistral-Small-2509",
              "vllm",    False, 24.0, "Magistral-Small-2509 (24B, bf16)"),
    # ── 8–15 B: bf16 fits comfortably ────────────────────────────────────────
    ModelSpec("phi_4",             "microsoft/phi-4",
              "vllm",    False, 14.0, "Phi-4 (14B, bf16)"),
    ModelSpec("hf_gemma2_9b_bf16", "google/gemma-2-9b-it",
              "hf_native", False, 9.0, "Gemma-2-9B (bf16, HF-native)"),
    ModelSpec("llama31_8b",        "meta-llama/Meta-Llama-3.1-8B-Instruct",
              "vllm",    False,  8.0, "Llama-3.1-8B (bf16)"),
    ModelSpec("qwen3_vl_8b",       "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
              "unsloth", True,   8.0, "Qwen3-VL-8B (4-bit, Unsloth)"),
    # ── 6–8 B ────────────────────────────────────────────────────────────────
    ModelSpec("mistral_v03",       "mistralai/Mistral-7B-Instruct-v0.3",
              "vllm",    False,  7.0, "Mistral-7B v0.3 (bf16)"),
    ModelSpec("mistral_v02",       "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
              "unsloth", True,   7.0, "Mistral-7B v0.2 (4-bit, Unsloth)"),
    ModelSpec("gemma_7b",          "google/gemma-7b-it",
              "vllm",    False,  7.0, "Gemma-7B (bf16)"),
    ModelSpec("hf_qwen25_7b_bf16", "Qwen/Qwen2.5-7B-Instruct",
              "hf_native", False, 7.0, "Qwen2.5-7B (bf16, HF-native)"),
    ModelSpec("qwen25_7b",         "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
              "unsloth", True,   7.0, "Qwen2.5-7B (4-bit, Unsloth)"),
    # ── 2–5 B ────────────────────────────────────────────────────────────────
    ModelSpec("qwen3_4b_thinking", "Qwen/Qwen3-4B-Thinking-2507",
              "vllm",    False,  4.0, "Qwen3-4B-Thinking (bf16)"),
    ModelSpec("phi35_mini",        "microsoft/Phi-3.5-mini-instruct",
              "vllm",    False,  3.8, "Phi-3.5-mini (3.8B, bf16)"),
    ModelSpec("gemma4_e2b",        "unsloth/gemma-4-E2B-it",
              "unsloth", True,   2.0, "Gemma-4-E2B (4-bit, Unsloth)"),
    # ── ≤2 B ─────────────────────────────────────────────────────────────────
    ModelSpec("llama32_1b",        "meta-llama/Llama-3.2-1B-Instruct",
              "vllm",    False,  1.0, "Llama-3.2-1B (bf16)"),
    ModelSpec("qwen35_08b",        "Qwen/Qwen3.5-0.8B",
              "vllm",    False,  0.8, "Qwen3.5-0.8B (bf16)"),
    ModelSpec("gemma3_270m",       "google/gemma-3-270m-it",
              "vllm",    False,  0.27, "Gemma-3-270M (bf16)"),
]
# fmt: on

# Build lookup: short → ModelSpec
_MODEL_LOOKUP: Dict[str, ModelSpec] = {m.short: m for m in MODEL_REGISTRY}


def _resolve_models() -> List[ModelSpec]:
    """Return the subset of models to run (env PD_MODELS, else all)."""
    raw = os.environ.get("PD_MODELS", "").strip()
    if not raw:
        return list(MODEL_REGISTRY)
    selected: List[ModelSpec] = []
    for s in raw.split(","):
        s = s.strip()
        if s not in _MODEL_LOOKUP:
            print(f"[WARN] Unknown model short '{s}' — skipping.")
            continue
        selected.append(_MODEL_LOOKUP[s])
    return selected


def _resolve_countries() -> List[str]:
    raw = os.environ.get("PD_COUNTRIES", "").strip()
    if not raw:
        return list(PAPER_20_COUNTRIES)
    return [c.strip() for c in raw.split(",") if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# H100-aware model loading
# ═══════════════════════════════════════════════════════════════════════════════

class _LoadTimeout(Exception):
    pass


def _load_model_timed(spec: ModelSpec, timeout_minutes: int = LOAD_TIMEOUT):
    """
    Load *spec* with timeout guard.
    • vLLM bf16: medium/small models; auto-quant for ≥70B via VLLM env.
    • Unsloth 4-bit: large models or when spec.backend == "unsloth".
    • hf_native: BF16 via Transformers (for models that need it).

    H100 tweaks applied through environment:
      MORAL_VLLM_GPU_MEM=0.92, MORAL_VLLM_MAX_SEQ_LEN=4096,
      VLLM_ENABLE_CHUNKED_PREFILL=1, VLLM_TP auto from GPU count.
    """
    def _do_load():
        if spec.backend == "vllm":
            from src.vllm_causal import load_model_vllm
            return load_model_vllm(
                spec.model_name,
                max_seq_length=int(os.environ.get("MORAL_VLLM_MAX_SEQ_LEN", "4096")),
                load_in_4bit=spec.load_in_4bit,
            )
        elif spec.backend == "hf_native":
            return load_model_hf_native(
                spec.model_name, max_seq_length=4096, load_in_4bit=False
            )
        else:
            # unsloth — 4-bit BnB
            return load_model(
                spec.model_name, max_seq_length=4096, load_in_4bit=spec.load_in_4bit
            )

    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        print(f"[LOAD] SIGALRM unavailable on {sys.platform} — loading without timeout")
        return _do_load()

    def _handler(signum, frame):
        raise _LoadTimeout(
            f"Model load exceeded {timeout_minutes} min. "
            "Adjust PD_LOAD_TIMEOUT or check VRAM / network."
        )

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_minutes * 60)
    print(f"[LOAD] timeout={timeout_minutes} min  (override: PD_LOAD_TIMEOUT)")
    try:
        result = _do_load()
        signal.alarm(0)
        return result
    except _LoadTimeout as exc:
        signal.alarm(0)
        print(f"\n[LOAD][ERROR] {exc}")
        raise (SystemExit(str(exc)) if _on_kaggle() else RuntimeError(str(exc))) from exc
    finally:
        signal.signal(signal.SIGALRM, prev)


def _free_model(model, tokenizer) -> None:
    """Delete model/tokenizer tensors and flush GPU caches between model loads."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cfg(spec: ModelSpec, countries: List[str]) -> SWAConfig:
    return SWAConfig(
        model_name=spec.model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=1,
        target_countries=list(countries),
        load_in_4bit=spec.load_in_4bit,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=RESULTS_BASE,
        lambda_coop=LAMBDA_COOP,
        K_samples=K_HALF * 2,  # kept for cfg compat; not used by PD controller
    )


def _load_scenarios(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    df   = load_multitp_dataset(
        data_base_path=cfg.multitp_data_path,
        lang=lang,
        translator=cfg.multitp_translator,
        suffix=cfg.multitp_suffix,
        n_scenarios=cfg.n_scenarios,
    )
    df = df.copy()
    df["lang"] = lang
    return df


def _reset_prior(country: str) -> None:
    """Reset only this country's prior-state entry before each run."""
    key = f"__pd_{country}"
    PRIOR_STATE.pop(key, None)
    PRIOR_STATE[key] = _NullPriorState()


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df.columns and df[col].notna().any() else float("nan")


def _safe_std(df: pd.DataFrame, col: str) -> float:
    return (
        float(df[col].std(ddof=1))
        if col in df.columns and len(df) >= 2 and df[col].notna().any()
        else float("nan")
    )


def _run_country(
    model,
    tokenizer,
    spec: ModelSpec,
    country: str,
    personas: List[str],
    scenario_df: pd.DataFrame,
    cfg: SWAConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """Patch swa_runner with PersonaDebiasingController and run one country."""
    import src.swa_runner as _swa_runner
    _swa_runner.ImplicitSWAController = PersonaDebiasingController  # type: ignore[attr-defined]
    return run_country_experiment(model, tokenizer, country, personas, scenario_df, cfg)


def _collect_row(
    spec: ModelSpec,
    country: str,
    results_df: pd.DataFrame,
    summary: Dict,
    elapsed_sec: float,
) -> Dict:
    align = summary.get("alignment", {})
    util  = summary.get("utilitarianism_slope", {})

    row: Dict = {
        "model_short":  spec.short,
        "model_name":   spec.model_name,
        "params_b":     spec.params_b,
        "backend":      spec.backend,
        "country":      country,
        "n_scenarios":  summary.get("n_scenarios", len(results_df)),
        # ── Primary alignment metrics ─────────────────────────────────────────
        "jsd":          align.get("jsd",         float("nan")),
        "pearson_r":    align.get("pearson_r",    float("nan")),
        "pearson_p":    align.get("pearson_p",    float("nan")),
        "spearman_rho": align.get("spearman_rho", float("nan")),
        "spearman_p":   align.get("spearman_p",   float("nan")),
        "mae":          align.get("mae",          float("nan")),
        "rmse":         align.get("rmse",         float("nan")),
        "mis":          align.get("mis",          float("nan")),
        "cosine_sim":   align.get("cosine_sim",   float("nan")),
        "n_criteria":   int(align.get("n_criteria", 0)),
        # ── Process ───────────────────────────────────────────────────────────
        "flip_rate":            summary.get("flip_rate",         float("nan")),
        "mean_latency_ms":      summary.get("mean_latency_ms",   float("nan")),
        "mean_positional_bias": _safe_mean(results_df, "positional_bias"),
        "std_positional_bias":  _safe_std(results_df,  "positional_bias"),
        "mean_delta_consensus": _safe_mean(results_df, "delta_consensus"),
        "mean_variance":        _safe_mean(results_df, "variance"),
        # ── Util OLS (diagnostic) ─────────────────────────────────────────────
        "util_slope_hat": float(util.get("slope_hat", float("nan")) or float("nan")),
        "util_slope_se":  float(util.get("slope_se",  float("nan")) or float("nan")),
        "util_n_obs":     int(util.get("n_obs", 0) or 0),
        # ── Timing ────────────────────────────────────────────────────────────
        "elapsed_sec": elapsed_sec,
    }

    model_amce = summary.get("model_amce", {})
    human_amce = summary.get("human_amce", {})
    per_dim    = summary.get("per_dimension_alignment", {})

    for dk in DIM_KEYS:
        row[f"model_{dk}"] = float(model_amce.get(dk, float("nan")))
        row[f"human_{dk}"] = float(human_amce.get(dk, float("nan")))

    for dk, dd in per_dim.items():
        row[f"abserr_{dk}"]  = float(dd.get("abs_err", float("nan")))
        row[f"signerr_{dk}"] = float(dd.get("signed",  float("nan")))

    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Report helpers
# ═══════════════════════════════════════════════════════════════════════════════

_G  = "\033[92m"
_R  = "\033[91m"
_B  = "\033[1m"
_D  = "\033[2m"
_Y  = "\033[93m"
_RS = "\033[0m"


def _fmt(val: float, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if np.isfinite(val) else "  —   "


def _d(val: float, ref: float, lower_better: bool = True, dec: int = 3) -> str:
    if not (np.isfinite(val) and np.isfinite(ref)):
        return f"{'—':>8}"
    d    = val - ref
    absd = abs(d)
    if absd < 5e-5:
        return f"{'±0':>8}"
    sign  = "+" if d > 0 else "−"
    worse = (d > 0) if lower_better else (d < 0)
    c     = _R if worse else _G
    return f"{c}{sign}.{absd:0{dec+2}.{dec}f}{_RS}"


def print_model_summary(rows: List[Dict], spec: ModelSpec) -> None:
    """Per-model summary: one row per country, sorted by MIS."""
    mr = [r for r in rows if r["model_short"] == spec.short]
    if not mr:
        return
    mr.sort(key=lambda x: x.get("mis", float("inf")))

    sep  = "═" * 105
    thin = "─" * 105
    print(f"\n{sep}")
    print(
        f"  {_B}Persona+Debiasing — {spec.notes}{_RS}"
        f"  (n_scenarios≈{N_SCENARIOS}  backend={spec.backend}  4bit={spec.load_in_4bit})"
    )
    print(sep)
    print(
        f"  {'Country':<8}"
        f"  {'MIS↓':>7}  {'JSD↓':>7}  {'r↑':>6}  {'ρ↑':>6}"
        f"  {'MAE↓':>6}  {'RMSE↓':>6}  {'flip%':>6}"
    )
    print(thin)
    for r in mr:
        print(
            f"  {r['country']:<8}"
            f"  {_fmt(r['mis'],  '.4f'):>7}"
            f"  {_fmt(r['jsd'],  '.4f'):>7}"
            f"  {_fmt(r['pearson_r'],  '+.3f'):>6}"
            f"  {_fmt(r['spearman_rho'],'+.3f'):>6}"
            f"  {_fmt(r['mae'],  '.2f'):>6}"
            f"  {_fmt(r['rmse'], '.2f'):>6}"
            f"  {r['flip_rate']*100 if np.isfinite(r.get('flip_rate', float('nan'))) else 0.0:>6.1f}"
        )
    print(thin)
    mis_vals = [r["mis"] for r in mr if np.isfinite(r.get("mis", float("nan")))]
    if mis_vals:
        print(
            f"  {_B}{'Mean':.<8}{_RS}"
            f"  {_fmt(float(np.mean(mis_vals)), '.4f'):>7}"
        )


def print_cross_model_summary(df: pd.DataFrame) -> None:
    """Final table: all models × mean-across-countries metrics."""
    if df.empty:
        return
    metrics = ["mis", "jsd", "pearson_r", "spearman_rho", "mae", "rmse"]
    grp = df.groupby(["model_short", "params_b"])[metrics].mean().reset_index()
    grp.sort_values("params_b", ascending=False, inplace=True)

    ref_mis = grp["mis"].min()  # best model as reference

    sep  = "═" * 95
    thin = "─" * 95
    print(f"\n\n{sep}")
    print(f"  {_B}Cross-Model Summary — Persona+Debiasing — {df['country'].nunique()} countries{_RS}")
    print(sep)
    print(
        f"  {'Model':<22}  {'Params':>7}"
        f"  {'MIS↓':>7}  {'ΔMIS':>8}"
        f"  {'JSD↓':>7}  {'r↑':>6}  {'ρ↑':>6}  {'MAE↓':>6}  {'RMSE↓':>6}"
    )
    print(thin)
    for _, r in grp.iterrows():
        mis  = r["mis"]
        delt = _d(mis, ref_mis, lower_better=True)
        print(
            f"  {r['model_short']:<22}  {r['params_b']:>5.1f}B"
            f"  {_fmt(mis,         '.4f'):>7}  {delt}"
            f"  {_fmt(r['jsd'],         '.4f'):>7}"
            f"  {_fmt(r['pearson_r'],   '+.3f'):>6}"
            f"  {_fmt(r['spearman_rho'],'+.3f'):>6}"
            f"  {_fmt(r['mae'],         '.2f'):>6}"
            f"  {_fmt(r['rmse'],        '.2f'):>6}"
        )
    print(thin)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    setup_seeds(SEED)

    models    = _resolve_models()
    countries = _resolve_countries()

    print(f"\n{'#' * 80}")
    print(f"  {_B}Persona + Debiasing Baseline Sweep{_RS}")
    print(f"  Models    : {len(models)} ({', '.join(m.short for m in models)})")
    print(f"  Countries : {len(countries)} ({', '.join(countries)})")
    print(f"  Scenarios : {N_SCENARIOS}  |  Seed: {SEED}")
    print(f"  Backend   : {os.environ.get('MORAL_MODEL_BACKEND', 'vllm')}")
    print(f"  VRAM util : {os.environ.get('MORAL_VLLM_GPU_MEM', '0.92')}")
    print(f"  Output    : {RESULTS_BASE}")
    print(f"  Skip done : {SKIP_DONE}")
    print(f"{'#' * 80}\n")

    out_root = Path(RESULTS_BASE)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for m_idx, spec in enumerate(models):
        print(f"\n{'=' * 80}")
        print(
            f"  [{m_idx + 1}/{len(models)}]  {_B}{spec.notes}{_RS}"
            f"  backend={spec.backend}  4bit={spec.load_in_4bit}"
        )
        print("=" * 80)

        # ── Per-model output directory ────────────────────────────────────────
        model_dir = out_root / spec.short
        model_dir.mkdir(parents=True, exist_ok=True)

        # ── Determine which countries still need running ──────────────────────
        pending = []
        for c in countries:
            csv_path = model_dir / f"{c}_pd_results.csv"
            if SKIP_DONE and csv_path.exists():
                print(f"  [SKIP] {c}: {csv_path.name} already exists")
                # Try to reload and add to all_rows
                try:
                    done_df = pd.read_csv(csv_path)
                    # Reconstruct a summary row from the CSV
                    # (best-effort; some fields may be missing)
                    pass
                except Exception:
                    pass
            else:
                pending.append(c)

        if not pending:
            print(f"  [SKIP] All countries done for {spec.short}")
            continue

        # ── Load model (once per model spec) ─────────────────────────────────
        print(f"\n[LOAD] {spec.model_name}  (timeout={LOAD_TIMEOUT} min)")
        t_load = time.time()
        try:
            model, tokenizer = _load_model_timed(spec, timeout_minutes=LOAD_TIMEOUT)
        except (SystemExit, RuntimeError) as exc:
            print(f"[LOAD][FAIL] {exc} — skipping model {spec.short}")
            continue
        print(f"[LOAD] done in {time.time() - t_load:.1f}s")

        cfg = _build_cfg(spec, pending)

        # ── Per-country loop ──────────────────────────────────────────────────
        for c_idx, country in enumerate(pending):
            if country not in SUPPORTED_COUNTRIES:
                print(f"  [SKIP] {country}: not in SUPPORTED_COUNTRIES")
                continue

            print(f"\n  [{c_idx + 1}/{len(pending)}]  Country: {_B}{country}{_RS}")

            scenario_df = _load_scenarios(cfg, country)
            personas    = build_country_personas(country, wvs_path=WVS_DATA_PATH)

            print(
                f"  Loaded {len(scenario_df)} scenarios"
                f"  |  {len(personas)} personas  |  lang={COUNTRY_LANG.get(country, 'en')}"
            )

            _reset_prior(country)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            t_start = time.time()
            try:
                results_df, summary = _run_country(
                    model, tokenizer, spec, country, personas, scenario_df, cfg
                )
            except Exception as exc:
                print(f"  [ERROR] {country}: {exc}")
                continue
            elapsed = time.time() - t_start

            # Save per-country CSV
            csv_path = model_dir / f"{country}_pd_results.csv"
            results_df.to_csv(csv_path, index=False)

            row = _collect_row(spec, country, results_df, summary, elapsed)
            all_rows.append(row)

            a = summary.get("alignment", {})
            print(
                f"  ✓  {spec.short} | {country}"
                f"  MIS={_fmt(a.get('mis',        float('nan')), '.4f')}"
                f"  JSD={_fmt(a.get('jsd',        float('nan')), '.4f')}"
                f"  r={_fmt(a.get('pearson_r',    float('nan')), '+.3f')}"
                f"  MAE={_fmt(a.get('mae',         float('nan')), '.2f')}"
                f"  ({elapsed:.0f}s)"
            )

        # ── Save incremental summary CSV after each model ─────────────────────
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                out_root / "pd_sweep_summary.csv", index=False
            )
            print(f"\n[SAVED] pd_sweep_summary.csv  ({len(all_rows)} rows so far)")

        # ── Free model VRAM before loading next ───────────────────────────────
        print(f"\n[FREE] Unloading {spec.short}")
        _free_model(model, tokenizer)
        print("[FREE] GPU cache cleared")

    # ── Final summary ─────────────────────────────────────────────────────────
    if not all_rows:
        print("\n[WARN] No results collected — check model/country settings.")
        return

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(out_root / "pd_sweep_summary.csv", index=False)
    print(f"\n[SAVED] Final pd_sweep_summary.csv  ({len(summary_df)} rows)")

    # ── Per-model report ──────────────────────────────────────────────────────
    done_specs = [m for m in models if any(r["model_short"] == m.short for r in all_rows)]
    for spec in done_specs:
        print_model_summary(all_rows, spec)

    print_cross_model_summary(summary_df)

    print(f"\n{'#' * 80}")
    print(f"  {_B}Persona+Debiasing Sweep COMPLETE{_RS}")
    print(f"  Results: {out_root}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
