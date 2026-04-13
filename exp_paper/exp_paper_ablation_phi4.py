#!/usr/bin/env python3
"""
EXP-24 Ablation Study — Phi-4 (14B), USA
==========================================
Systematic component-level analysis of SWA-DPBR to identify which design
choices drive cultural-alignment gains.  Mirrors the paper's ablation table
(§Ablation: Identifying Load-Bearing Components) using Phi-4 instead of
Qwen2.5-72B, allowing direct cross-model comparison.

Six configurations are evaluated sequentially using a single model load:

  Row   Configuration              What is disabled
  ────  ─────────────────────────  ──────────────────────────────────────────
  Full  Full SWA-DPBR              —  (reference row)
  1     No-IS (consensus only)     Importance sampling  (δ* ≡ 0; anchor only)
  2     Always-on PT-IS            Dual-pass DPBR reliability  (r ≡ 1 always)
  3     No debiasing               Positional-label A↔B swap
  4     Without persona            Cultural personas  (agents = base model)
  5     No country prior (α_h=0)   Hierarchical country prior

Metrics per configuration × country:
  Primary  : JSD (↓), Pearson r (↑), Spearman ρ (↑), MAE (↓), RMSE (↓), MIS (↓)
  Shape    : Centred cosine similarity (≡ Pearson r — sanity check)
  DPBR     : mean/std reliability_r, bootstrap_var, ESS₁, ESS₂, ESS-anchor α
  Process  : flip rate, positional bias, delta_consensus, final prior state
  Per-dim  : |model−human| for Species, Gender, Age, Fitness, SocialValue, Util
  Util OLS : slope b_hat + SE (proper continuous-AMCE; diagnostic only, ∉ JSD)

Usage
-----
Kaggle notebook:
    !python exp_paper/exp_paper_ablation_phi4.py

Local (inside repo root):
    python exp_paper/exp_paper_ablation_phi4.py

Environment overrides:
    ABLATION_COUNTRIES      comma-separated ISO3  (default: USA)
    ABLATION_N_SCENARIOS    int                   (default: 500 — matches exp_paper_phi_4.py)
    ABLATION_SEED           int                   (default: 42)
    MORAL_MODEL_BACKEND     unsloth|hf_native|vllm (default: vllm — matches
                            the main EXP-24-PHI_4 experiment which was run
                            with vLLM BF16 full precision on Kaggle; Unsloth
                            4-bit quantisation produces near-uniform logits
                            for Phi-4 on moral reasoning, causing MIS≈0.51
                            instead of the main-exp MIS≈0.24)
    EXP24_K_HALF            int  — override K per IS pass (default: 64)
    EXP24_VAR_SCALE         float — override DPBR scale  (default: 0.04)
"""

from __future__ import annotations

import gc
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Repo bootstrap (same pattern as all exp_paper/* entry scripts) ────────────
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

# ── Backend default: vLLM (BF16) — matches the main EXP-24-PHI_4 run ─────────
# Unsloth 4-bit quantisation causes near-flat logits for Phi-4 on moral
# reasoning (delta_consensus ≈ 0.003 after debiasing → all MPR ≈ 50%),
# producing MIS ≈ 0.51 instead of the vLLM baseline MIS ≈ 0.24.
# The user can still force Unsloth with: MORAL_MODEL_BACKEND=unsloth
os.environ.setdefault("MORAL_MODEL_BACKEND", "vllm")

# ── Env / deps (must run BEFORE exp24_dpbr_core is imported) ─────────────────
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()

from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

# EXP-24 env: ESS anchor regularisation ON by default (matches paper §4.2)
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")

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
    dpbr_reliability_weight,
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
from src.config import model_slug  # noqa: E402
from src.data import load_multitp_dataset  # noqa: E402
from src.model import load_model, load_model_hf_native, setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

# ── Run configuration ─────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-4"
MODEL_SHORT = "phi_4"

ABLATION_COUNTRIES: List[str] = [
    c.strip()
    for c in os.environ.get("ABLATION_COUNTRIES", "USA").split(",")
    if c.strip()
]
# N_SCENARIOS matches exp_paper_phi_4.py (_base_dpbr.N_SCENARIOS = 500) so that
# the Full SWA-DPBR reference row is evaluated on the same scenario set.
N_SCENARIOS: int = int(os.environ.get("ABLATION_N_SCENARIOS", "500"))
SEED: int = int(os.environ.get("ABLATION_SEED", "42"))
LAMBDA_COOP: float = 0.70
LOAD_TIMEOUT_MINUTES: int = int(os.environ.get("ABLATION_LOAD_TIMEOUT_MINUTES", "15"))

# ── Pre-computed Full SWA-DPBR results (optional) ─────────────────────────────
# If ABLATION_FULL_SWA_BASE is set (or the Kaggle default path exists from a
# previous exp_paper_phi_4.py run in the same session), the "Full SWA-DPBR"
# row is loaded from swa_results_<country>.csv instead of being re-run.
# When the file does not exist the script falls back to running Full normally.
# Default path mirrors _base_dpbr.run_for_model():
#   {results_base}/{model_short}/swa/{model_slug(MODEL_NAME)}/swa_results_<C>.csv
_KAGGLE_MAIN_RUN_SWA = (
    f"/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    f"/{MODEL_SHORT}/swa/{model_slug(MODEL_NAME)}"
)
FULL_SWA_BASE: Optional[str] = os.environ.get("ABLATION_FULL_SWA_BASE") or (
    _KAGGLE_MAIN_RUN_SWA if _on_kaggle() else None
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_ablation_phi4"
    if _on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_ablation_phi4")
)
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# Canonical dimension keys produced by compute_amce_from_preferences()
DIM_KEYS = [
    "Species_Humans",
    "Gender_Female",
    "Age_Young",
    "Fitness_Fit",
    "SocialValue_High",
    "Utilitarianism_More",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Ablation controller subclasses
# Each subclass disables EXACTLY ONE component of SWA-DPBR so the marginal
# contribution of that component is isolated.
# ═══════════════════════════════════════════════════════════════════════════════


class NoISController(Exp24DualPassController):
    """Ablation 1 — No-IS (consensus only): importance sampling disabled.

    Both IS passes return δ* = 0, so the final micro decision collapses to:
        δ_opt_micro = α_ESS · anchor + (1 − α_ESS) · δ_base   (pure consensus)

    This measures the contribution of IS above the persona-consensus baseline.
    The dual-pass structure, debiasing, and hierarchical prior remain active.
    """

    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        K: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        # Return zero IS correction; ESS = 1.0 (no collapse)
        return torch.zeros((), device=device), 1.0


class AlwaysOnISController(Exp24DualPassController):
    """Ablation 2 — Always-on PT-IS: dual-pass reliability weight bypassed (r ≡ 1).

    Both IS passes run normally, but the bootstrap reliability weight is fixed
    at r = 1.0 regardless of the bootstrap variance (δ*₁ − δ*₂)².  This
    isolates the contribution of the DPBR filter: when r ≡ 1, the algorithm
    always applies the IS correction whether or not the two passes agree.
    All other components (debiasing, persona, country prior) remain active.
    """

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        # ── Forward pass (identical to Exp24DualPassController.predict) ────────
        db1, da1, logit_temp = self._extract_logit_gaps(
            user_query, phenomenon_category, lang
        )
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(
                swapped_query, phenomenon_category, lang
            )
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1
        positional_bias = positional_bias_logit_gap(db1, db2, swap_changed)

        sigma = max(
            float(delta_agents.std(unbiased=True).item())
            if delta_agents.numel() >= 2
            else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()
        device = self.device

        ds1, ess1 = self._single_is_pass(
            delta_base, delta_agents, anchor, sigma, K_HALF, device
        )
        ds2, ess2 = self._single_is_pass(
            delta_base, delta_agents, anchor, sigma, K_HALF, device
        )

        # ── ABLATION: bypass reliability weight — r ≡ 1.0 ────────────────────
        bootstrap_var = float((ds1 - ds2).pow(2).item())
        r             = 1.0                     # force full IS acceptance
        delta_star    = (ds1 + ds2) / 2.0      # no soft attenuation
        # ─────────────────────────────────────────────────────────────────────

        ess_min = min(ess1, ess2)
        if _use_ess_anchor_reg():
            alpha_reg = ess_anchor_blend_alpha(ess_min, self.rho_eff)
            delta_opt_micro = float(
                (alpha_reg * anchor + (1.0 - alpha_reg) * delta_base + delta_star).item()
            )
        else:
            alpha_reg = 1.0
            delta_opt_micro = float((anchor + delta_star).item())

        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = (
            float(delta_agents.var(unbiased=True).item())
            if delta_agents.numel() > 1
            else 0.0
        )

        def _p_pref_micro(d_s: torch.Tensor) -> float:
            dm = float((anchor + d_s).item())
            pr = torch.sigmoid(
                torch.tensor(dm / self.decision_temperature)
            ).item()
            return pr if preferred_on_right else 1.0 - pr

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_star_1": float(ds1.item()),
            "delta_star_2": float(ds2.item()),
            "bootstrap_var": bootstrap_var,
            "reliability_r": r,
            "ess_pass1": ess1,
            "ess_pass2": ess2,
            "ess_anchor_alpha": alpha_reg,
            "ess_anchor_reg_enabled": _use_ess_anchor_reg(),
            "delta_country": st["delta_country"],
            "alpha_h": st["alpha_h"],
            "prior_step": st["step"],
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_is_pass1_micro": _p_pref_micro(ds1),
            "p_spare_preferred_is_pass2_micro": _p_pref_micro(ds2),
            "positional_bias": positional_bias,
        }


class NoDebiasController(Exp24DualPassController):
    """Ablation 3 — No debiasing: positional A↔B swap disabled.

    _swap_positional_labels() always reports that no swap occurred, so the
    controller never performs the second forward pass and the raw positional
    bias of the A-first presentation is retained in δ_base and δ_agents.
    All other components (IS, dual-pass, persona, country prior) remain active.
    """

    def _swap_positional_labels(
        self, user_query: str, lang: str
    ) -> Tuple[str, bool]:
        # Signal: no swap, no second pass — positional bias unremoved
        return user_query, False


class NoPersonaController(Exp24DualPassController):
    """Ablation 4 — Without persona: cultural agents replaced by base model.

    _extract_logit_gaps() is overridden to return delta_agents = delta_base
    (shape (1,)).  This collapses the N-agent ensemble to a single agent whose
    logit gap equals the base model, so:
        anchor  = δ_base
        sigma   = noise_std  (floor, since std of a 1-element tensor = 0)
        IS optimises around δ_base  (no cultural-persona signal)

    The override eliminates cultural persona influence while keeping debiasing,
    IS, dual-pass, and the country prior active.
    """

    def _extract_logit_gaps(
        self,
        user_query: str,
        phenomenon_category: str,
        lang: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        db, _da, logit_temp = super()._extract_logit_gaps(
            user_query, phenomenon_category, lang
        )
        # Replace all persona agents with a single copy of the base gap
        da = db.detach().clone().unsqueeze(0)   # shape (1,)
        return db, da, logit_temp


class NoPriorController(Exp24DualPassController):
    """Ablation 5 — No country prior (α_h = 0): hierarchical prior disabled.

    A custom BootstrapPriorState subclass is injected whose apply_prior()
    method always returns delta_opt_micro unchanged.  The EMA update still
    runs (so delta_country drifts for diagnostics), but α_h is never applied
    to the final decision gap.
    """

    class _NullPriorState(BootstrapPriorState):
        """BootstrapPriorState with apply_prior() hard-wired to identity."""

        def apply_prior(self, delta_opt_micro: float) -> float:
            # Run EMA update for diagnostics — never blend with country prior
            return delta_opt_micro

    def _get_prior(self) -> "NoPriorController._NullPriorState":
        key = f"__noprior_{self.country}"
        if key not in PRIOR_STATE:
            PRIOR_STATE[key] = NoPriorController._NullPriorState()
        return PRIOR_STATE[key]  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AblationSpec:
    row_label: str       # Paper-style row label (matches §Ablation table)
    controller_cls: type  # Subclass of Exp24DualPassController to use
    description: str     # One-line description for logging


ABLATION_SPECS: List[AblationSpec] = [
    AblationSpec(
        row_label="Full SWA-DPBR",
        controller_cls=Exp24DualPassController,
        description="All components enabled  [reference]",
    ),
    AblationSpec(
        row_label="No-IS (consensus only)",
        controller_cls=NoISController,
        description="IS disabled — δ* ≡ 0; only anchor/consensus contributes",
    ),
    AblationSpec(
        row_label="Always-on PT-IS",
        controller_cls=AlwaysOnISController,
        description="Dual-pass reliability weight bypassed (r ≡ 1 always)",
    ),
    AblationSpec(
        row_label="No debiasing",
        controller_cls=NoDebiasController,
        description="Positional A↔B swap disabled; raw positional bias retained",
    ),
    AblationSpec(
        row_label="Without persona",
        controller_cls=NoPersonaController,
        description="Cultural personas removed; agents = base model clone",
    ),
    AblationSpec(
        row_label="No country prior (a_h=0)",
        controller_cls=NoPriorController,
        description="Hierarchical country EMA prior disabled (α_h ≡ 0)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _build_cfg(target_countries: List[str], load_in_4bit: bool = False) -> SWAConfig:
    return SWAConfig(
        model_name=MODEL_NAME,
        n_scenarios=N_SCENARIOS,
        batch_size=1,
        target_countries=list(target_countries),
        load_in_4bit=load_in_4bit,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=RESULTS_BASE,
        lambda_coop=LAMBDA_COOP,
        K_samples=K_HALF * 2,
    )


def _load_scenarios(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    df = load_multitp_dataset(
        data_base_path=cfg.multitp_data_path,
        lang=lang,
        translator=cfg.multitp_translator,
        suffix=cfg.multitp_suffix,
        n_scenarios=cfg.n_scenarios,
    )
    df = df.copy()
    df["lang"] = lang
    return df


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col in df.columns and df[col].notna().any():
        return float(df[col].mean())
    return float("nan")


def _safe_std(df: pd.DataFrame, col: str) -> float:
    if col in df.columns and len(df) >= 2 and df[col].notna().any():
        return float(df[col].std(ddof=1))
    return float("nan")


def _reset_prior_state(country: str) -> None:
    """Clear and reinitialise all prior-state entries before each ablation."""
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    PRIOR_STATE[f"__noprior_{country}"] = NoPriorController._NullPriorState()


def _find_full_swa_csv(country: str) -> Optional[str]:
    """Return path to pre-computed swa_results_<country>.csv from the main run,
    or None if FULL_SWA_BASE is unset / file does not exist."""
    if not FULL_SWA_BASE:
        return None
    p = os.path.join(FULL_SWA_BASE, f"swa_results_{country}.csv")
    if os.path.isfile(p):
        return p
    return None


def _reconstruct_summary(results_df: pd.DataFrame, country: str, cfg: "SWAConfig") -> Dict:
    """Recompute the summary dict from a pre-loaded results_df.

    Mirrors the tail of src/swa_runner.run_country_experiment() so that
    metric computation is identical to a live run.
    """
    model_amce = compute_amce_from_preferences(results_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment  = compute_alignment_metrics(model_amce, human_amce)
    per_dim    = compute_per_dimension_alignment(model_amce, human_amce)
    util_slope = compute_utilitarianism_slope(results_df)

    flip_col   = "mppi_flipped"
    flip_count = int(results_df[flip_col].sum()) if flip_col in results_df.columns else 0
    n          = len(results_df)

    return {
        "country":                country,
        "n_scenarios":            n,
        "flip_rate":              flip_count / max(1, n),
        "flip_count":             flip_count,
        "mean_variance":          float(results_df["mppi_variance"].mean())
                                  if "mppi_variance" in results_df.columns else float("nan"),
        "mean_delta_z_norm":      float(results_df["delta_z_norm"].mean())
                                  if "delta_z_norm" in results_df.columns else float("nan"),
        "mean_decision_gap":      float(results_df["delta_consensus"].mean())
                                  if "delta_consensus" in results_df.columns else float("nan"),
        "model_amce":             model_amce,
        "human_amce":             human_amce,
        "alignment":              alignment,
        "per_dimension_alignment": per_dim,
        "utilitarianism_slope":   util_slope,
    }


def _run_ablation_country(
    spec: AblationSpec,
    model,
    tokenizer,
    country: str,
    personas: List[str],
    scenario_df: pd.DataFrame,
    cfg: SWAConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """Patch swa_runner's controller and run one country for one ablation."""
    import src.swa_runner as _swa_runner

    _swa_runner.ImplicitSWAController = spec.controller_cls  # type: ignore[attr-defined]

    # "Without persona": pass a single empty-string persona so that
    # NoPersonaController._extract_logit_gaps can still call super() safely.
    run_personas = (
        [""]
        if spec.controller_cls is NoPersonaController
        else personas
    )
    return run_country_experiment(
        model, tokenizer, country, run_personas, scenario_df, cfg
    )


def _collect_row(
    spec: AblationSpec,
    country: str,
    results_df: pd.DataFrame,
    summary: Dict,
    elapsed_sec: float,
) -> Dict:
    """Flatten all metrics from a single (ablation × country) run into a dict."""
    align = summary.get("alignment", {})
    util  = summary.get("utilitarianism_slope", {})

    # Prior state key depends on ablation type
    prior_key = (
        f"__noprior_{country}"
        if spec.controller_cls is NoPriorController
        else country
    )
    prior_st = (PRIOR_STATE.get(prior_key) or BootstrapPriorState()).stats

    row: Dict = {
        "ablation":    spec.row_label,
        "country":     country,
        "n_scenarios": summary.get("n_scenarios", len(results_df)),
        # ── Primary alignment metrics ──────────────────────────────────────────
        "jsd":          align.get("jsd",          float("nan")),
        "pearson_r":    align.get("pearson_r",     float("nan")),
        "spearman_rho": align.get("spearman_rho",  float("nan")),
        "mae":          align.get("mae",           float("nan")),
        "rmse":         align.get("rmse",          float("nan")),
        "mis":          align.get("mis",           float("nan")),
        "cosine_sim":   align.get("cosine_sim",    float("nan")),
        "n_criteria":   int(align.get("n_criteria", 0)),
        # ── DPBR process diagnostics ──────────────────────────────────────────
        "flip_rate":              summary.get("flip_rate",       float("nan")),
        "mean_reliability_r":     _safe_mean(results_df, "reliability_r"),
        "std_reliability_r":      _safe_std(results_df,  "reliability_r"),
        "mean_bootstrap_var":     _safe_mean(results_df, "bootstrap_var"),
        "std_bootstrap_var":      _safe_std(results_df,  "bootstrap_var"),
        "mean_ess_pass1":         _safe_mean(results_df, "ess_pass1"),
        "mean_ess_pass2":         _safe_mean(results_df, "ess_pass2"),
        "mean_ess_anchor_alpha":  _safe_mean(results_df, "ess_anchor_alpha"),
        "mean_positional_bias":   _safe_mean(results_df, "positional_bias"),
        "std_positional_bias":    _safe_std(results_df,  "positional_bias"),
        "mean_delta_consensus":   _safe_mean(results_df, "delta_consensus"),
        "mean_delta_opt":         _safe_mean(results_df, "delta_opt"),
        "mean_delta_opt_micro":   _safe_mean(results_df, "delta_opt_micro"),
        "std_delta_opt_micro":    _safe_std(results_df,  "delta_opt_micro"),
        # ── Country prior state at end of run ─────────────────────────────────
        "final_delta_country": float(prior_st.get("delta_country", float("nan"))),
        "final_alpha_h":       float(prior_st.get("alpha_h",       float("nan"))),
        "prior_steps":         int(prior_st.get("step", 0)),
        # ── Utilitarianism OLS slope (diagnostic; NOT in JSD) ─────────────────
        "util_slope_hat": float(util.get("slope_hat", float("nan")) or float("nan")),
        "util_slope_se":  float(util.get("slope_se",  float("nan")) or float("nan")),
        "util_n_obs":     int(util.get("n_obs", 0) or 0),
        # ── Timing ────────────────────────────────────────────────────────────
        "elapsed_sec": elapsed_sec,
    }

    # Per-dimension model/human AMCEs and errors
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
# Rich reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

_CLR_GREEN  = "\033[92m"
_CLR_RED    = "\033[91m"
_CLR_BOLD   = "\033[1m"
_CLR_RESET  = "\033[0m"
_CLR_DIM    = "\033[2m"
_CLR_YELLOW = "\033[93m"


def _delta_fmt(
    val: float,
    ref: float,
    lower_is_better: bool = True,
    decimals: int = 3,
) -> str:
    """Return a coloured Δ string (+.xxx / −.xxx) vs the reference row."""
    if not (np.isfinite(val) and np.isfinite(ref)):
        return f"{'—':>8}"
    d    = val - ref
    absd = abs(d)
    if absd < 5e-5:
        return f"{'±.000':>8}"
    sign = "+" if d > 0 else "−"
    worse = (d > 0) if lower_is_better else (d < 0)
    clr   = _CLR_RED if worse else _CLR_GREEN
    return f"{clr}{sign}.{absd:0{decimals+2}.{decimals}f}{_CLR_RESET}"


def _fmt(val: float, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if np.isfinite(val) else "  —   "


def print_ablation_table(rows: List[Dict], country: str) -> None:
    """Print the paper-style ablation table with Δ-columns for each metric."""
    cr = [r for r in rows if r["country"] == country]
    if not cr:
        return

    ref = next((r for r in cr if r["ablation"] == "Full SWA-DPBR"), cr[0])
    w   = max(len(r["ablation"]) for r in cr) + 2

    sep = "═" * 108
    thin = "─" * 108
    print(f"\n{sep}")
    print(
        f"  Ablation table — {_CLR_BOLD}{country}{_CLR_RESET}"
        f"  (n={ref['n_scenarios']}  K={K_HALF}×2={K_HALF * 2}"
        f"  VAR_SCALE={VAR_SCALE})"
    )
    print(sep)
    print(
        f"  {'#':>2}  {'Configuration':<{w}}"
        f"  {'JSD↓':>7} {'ΔJSD':>8}"
        f"  {'r↑':>6} {'Δr':>8}"
        f"  {'MAE↓':>6} {'ΔMAE':>8}"
        f"  {'MIS↓':>6} {'ΔMIS':>8}"
        f"  {'ρ↑':>6}"
        f"  {'flip%':>5}  {'rel_r':>5}"
    )
    print(thin)

    row_labels = {0: "   ", 1: "  1", 2: "  2", 3: "  3", 4: "  4", 5: "  5"}
    for i, row in enumerate(cr):
        is_ref = row["ablation"] == "Full SWA-DPBR"
        b, r_  = (_CLR_BOLD, _CLR_RESET) if is_ref else ("", "")
        prefix = row_labels.get(i, f"  {i}")

        jsd_d = _delta_fmt(row["jsd"],        ref["jsd"],        lower_is_better=True)
        r_d   = _delta_fmt(row["pearson_r"],   ref["pearson_r"],  lower_is_better=False)
        mae_d = _delta_fmt(row["mae"],         ref["mae"],        lower_is_better=True)
        mis_d = _delta_fmt(row["mis"],         ref["mis"],        lower_is_better=True)

        relr  = row.get("mean_reliability_r", float("nan"))
        flip  = row.get("flip_rate", float("nan"))

        print(
            f"{prefix}  {b}{row['ablation']:<{w}}{r_}"
            f"  {_fmt(row['jsd'],'.4f'):>7} {jsd_d}"
            f"  {_fmt(row['pearson_r'],'+.3f'):>6} {r_d}"
            f"  {_fmt(row['mae'],'.2f'):>6} {mae_d}"
            f"  {_fmt(row['mis'],'.4f'):>6} {mis_d}"
            f"  {_fmt(row['spearman_rho'],'+.3f'):>6}"
            f"  {flip*100:>5.1f}"
            f"  {_fmt(relr,'.3f'):>5}"
        )

    print(thin)
    print(
        f"  {_CLR_DIM}Δ = ablation − Full; "
        f"{_CLR_GREEN}green{_CLR_RESET}{_CLR_DIM} = improvement, "
        f"{_CLR_RED}red{_CLR_RESET}{_CLR_DIM} = degradation{_CLR_RESET}"
    )


def print_per_dim_table(rows: List[Dict], country: str) -> None:
    """Print |model − human| per dimension for each ablation configuration."""
    cr = [r for r in rows if r["country"] == country]
    if not cr:
        return

    ref = next((r for r in cr if r["ablation"] == "Full SWA-DPBR"), cr[0])
    w   = max(len(r["ablation"]) for r in cr) + 2
    thin = "─" * 100

    print(f"\n  {_CLR_BOLD}Per-Dimension |model−human| (pp)  — {country}{_CLR_RESET}")
    hdr = f"  {'Configuration':<{w}}"
    for dk in DIM_KEYS:
        hdr += f"  {dk.split('_')[0]:>8}"
    hdr += f"  {'RMSE':>6}  {'RMSE Δ':>8}"
    print(hdr)
    print(f"  {thin}")

    for row in cr:
        is_ref = row["ablation"] == "Full SWA-DPBR"
        b, r_  = (_CLR_BOLD, _CLR_RESET) if is_ref else ("", "")
        line   = f"  {b}{row['ablation']:<{w}}{r_}"

        for dk in DIM_KEYS:
            ae = row.get(f"abserr_{dk}", float("nan"))
            line += f"  {_fmt(ae, '8.2f')}"

        rmse  = row.get("rmse", float("nan"))
        rmse_d = _delta_fmt(rmse, ref["rmse"], lower_is_better=True)
        line += f"  {_fmt(rmse, '6.2f')}  {rmse_d}"
        print(line)


def print_per_dim_signed(rows: List[Dict], country: str) -> None:
    """Print signed errors (model − human) to show direction of misalignment."""
    cr = [r for r in rows if r["country"] == country]
    if not cr:
        return

    w    = max(len(r["ablation"]) for r in cr) + 2
    thin = "─" * 100

    print(f"\n  {_CLR_BOLD}Per-Dimension signed error (model−human, pp)  — {country}{_CLR_RESET}")
    hdr = f"  {'Configuration':<{w}}"
    for dk in DIM_KEYS:
        hdr += f"  {dk.split('_')[0]:>8}"
    print(hdr)
    print(f"  {thin}")

    for row in cr:
        is_ref = row["ablation"] == "Full SWA-DPBR"
        b, r_  = (_CLR_BOLD, _CLR_RESET) if is_ref else ("", "")
        line   = f"  {b}{row['ablation']:<{w}}{r_}"
        for dk in DIM_KEYS:
            se  = row.get(f"signerr_{dk}", float("nan"))
            if not np.isfinite(se):
                line += f"  {'—':>8}"
            else:
                clr = _CLR_GREEN if se >= 0 else _CLR_RED
                line += f"  {clr}{se:>+8.2f}{_CLR_RESET}"
        print(line)


def print_dpbr_diagnostics(rows: List[Dict], country: str) -> None:
    """Print DPBR-internal diagnostics: reliability, bootstrap var, ESS, bias."""
    cr = [r for r in rows if r["country"] == country]
    if not cr:
        return

    w    = max(len(r["ablation"]) for r in cr) + 2
    thin = "─" * 110

    print(f"\n  {_CLR_BOLD}DPBR Internal Diagnostics  — {country}{_CLR_RESET}")
    print(
        f"  {'Configuration':<{w}}"
        f"  {'rel_r μ':>8} {'rel_r σ':>8}"
        f"  {'bvar μ':>8} {'bvar σ':>8}"
        f"  {'ESS₁':>6} {'ESS₂':>6}"
        f"  {'ESS-α':>6}"
        f"  {'pos_b μ':>8} {'pos_b σ':>8}"
        f"  {'δ_cntry':>8} {'α_h':>5}"
    )
    print(f"  {thin}")

    for row in cr:
        is_ref = row["ablation"] == "Full SWA-DPBR"
        b, r_  = (_CLR_BOLD, _CLR_RESET) if is_ref else ("", "")
        print(
            f"  {b}{row['ablation']:<{w}}{r_}"
            f"  {_fmt(row.get('mean_reliability_r',float('nan')),'.4f'):>8}"
            f"  {_fmt(row.get('std_reliability_r', float('nan')),'.4f'):>8}"
            f"  {_fmt(row.get('mean_bootstrap_var',float('nan')),'.5f'):>8}"
            f"  {_fmt(row.get('std_bootstrap_var', float('nan')),'.5f'):>8}"
            f"  {_fmt(row.get('mean_ess_pass1',    float('nan')),'.3f'):>6}"
            f"  {_fmt(row.get('mean_ess_pass2',    float('nan')),'.3f'):>6}"
            f"  {_fmt(row.get('mean_ess_anchor_alpha',float('nan')),'.3f'):>6}"
            f"  {_fmt(row.get('mean_positional_bias',  float('nan')),'.4f'):>8}"
            f"  {_fmt(row.get('std_positional_bias',   float('nan')),'.4f'):>8}"
            f"  {_fmt(row.get('final_delta_country',   float('nan')),'.4f'):>8}"
            f"  {_fmt(row.get('final_alpha_h',         float('nan')),'.3f'):>5}"
        )


def print_util_slopes(rows: List[Dict], country: str) -> None:
    """Print Utilitarianism OLS slope diagnostics."""
    cr = [r for r in rows if r["country"] == country]
    if not cr or all(r.get("util_n_obs", 0) < 3 for r in cr):
        return

    w    = max(len(r["ablation"]) for r in cr) + 2
    thin = "─" * 80
    print(f"\n  {_CLR_DIM}Utilitarianism OLS slope  (diagnostic; NOT in JSD/MIS)  — {country}{_CLR_RESET}")
    print(
        f"  {'Configuration':<{w}}"
        f"  {'b_hat':>7} {'SE(b)':>7} {'n_obs':>5}"
    )
    print(f"  {thin}")
    for row in cr:
        n   = row.get("util_n_obs", 0)
        bh  = row.get("util_slope_hat", float("nan"))
        se  = row.get("util_slope_se",  float("nan"))
        if n >= 3:
            print(
                f"  {row['ablation']:<{w}}"
                f"  {_fmt(bh,'+.4f'):>7} {_fmt(se,'.4f'):>7} {n:>5}"
            )


def print_cross_country_summary(df: pd.DataFrame) -> None:
    """Print averaged metrics across all countries per ablation."""
    if df.empty or df["country"].nunique() < 2:
        return

    metrics = ["jsd", "pearson_r", "spearman_rho", "mae", "rmse", "mis"]
    grp = df.groupby("ablation")[metrics].mean()

    # Sort to put Full first
    order = [s.row_label for s in ABLATION_SPECS]
    grp = grp.reindex([o for o in order if o in grp.index])

    print(f"\n{'═' * 80}")
    print(f"  {_CLR_BOLD}Cross-Country Mean Metrics{_CLR_RESET}"
          f"  (n_countries={df['country'].nunique()})")
    print("═" * 80)
    print(
        f"  {'Configuration':<35}"
        f"  {'JSD↓':>7} {'r↑':>7} {'ρ↑':>7}"
        f"  {'MAE↓':>7} {'RMSE↓':>7} {'MIS↓':>7}"
    )
    print("─" * 80)

    ref_row = grp.loc["Full SWA-DPBR"] if "Full SWA-DPBR" in grp.index else None
    for name, row in grp.iterrows():
        is_ref = name == "Full SWA-DPBR"
        b, r_  = (_CLR_BOLD, _CLR_RESET) if is_ref else ("", "")
        print(
            f"  {b}{name:<35}{r_}"
            f"  {_fmt(row['jsd'],       '.4f'):>7}"
            f"  {_fmt(row['pearson_r'], '+.3f'):>7}"
            f"  {_fmt(row['spearman_rho'],'+.3f'):>7}"
            f"  {_fmt(row['mae'],       '.2f'):>7}"
            f"  {_fmt(row['rmse'],      '.2f'):>7}"
            f"  {_fmt(row['mis'],       '.4f'):>7}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Model load with timeout — mirrors exp_model/_base_dpbr.py safety pattern
# ═══════════════════════════════════════════════════════════════════════════════


class _LoadTimeout(Exception):
    pass


def _load_model_timed(
    backend: str,
    use_4bit: bool,
    timeout_minutes: int = LOAD_TIMEOUT_MINUTES,
):
    """
    Load the model and abort if it hangs beyond *timeout_minutes*.

    On Kaggle (Linux): uses signal.SIGALRM so a frozen download/OOM does
    not waste the full GPU session.  On Windows SIGALRM is unavailable and
    the load proceeds without a timeout.

    Override the limit with env var ABLATION_LOAD_TIMEOUT_MINUTES.
    """
    def _do_load():
        if backend == "vllm":
            # Use load_model (routes to _load_model_vllm → VllmCausalLogitModel),
            # which is the exact same path as the main EXP-24-PHI_4 run.
            # Do NOT use src.vllm_causal.load_model_vllm — it returns a different
            # wrapper class (VllmCausalWrapper) with a different inference path,
            # causing mismatched metrics vs the main run.
            return load_model(MODEL_NAME, max_seq_length=2048, load_in_4bit=False)
        elif backend == "hf_native":
            return load_model_hf_native(MODEL_NAME, max_seq_length=2048, load_in_4bit=False)
        else:
            return load_model(MODEL_NAME, max_seq_length=2048, load_in_4bit=use_4bit)

    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        print(f"[LOAD] SIGALRM unavailable on {sys.platform} — loading without timeout")
        return _do_load()

    def _handler(signum, frame):
        raise _LoadTimeout(
            f"Model load exceeded {timeout_minutes} minute(s). "
            "Check VRAM / network. Override: ABLATION_LOAD_TIMEOUT_MINUTES."
        )

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_minutes * 60)
    print(f"[LOAD] timeout={timeout_minutes} min  "
          f"(override: ABLATION_LOAD_TIMEOUT_MINUTES)")
    try:
        result = _do_load()
        signal.alarm(0)
        return result
    except _LoadTimeout as exc:
        signal.alarm(0)
        print(f"\n[LOAD][ERROR] {exc}")
        err = SystemExit("[LOAD] Stopping Kaggle run to avoid wasting GPU.") if _on_kaggle() \
              else RuntimeError(str(exc))
        raise err from exc
    finally:
        signal.signal(signal.SIGALRM, prev)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    setup_seeds(SEED)

    print(f"\n{'#' * 80}")
    print(f"  EXP-24 Ablation Study — {_CLR_BOLD}Phi-4 (14B) — {MODEL_NAME}{_CLR_RESET}")
    print(f"  Countries : {ABLATION_COUNTRIES}")
    print(f"  Scenarios : {N_SCENARIOS}  |  Seed: {SEED}")
    print(f"  DPBR      : K_HALF={K_HALF} × 2 = {K_HALF * 2} IS samples/scenario")
    print(f"              VAR_SCALE={VAR_SCALE}  (r=exp(−var/s))")
    print(f"  Ablations : {len(ABLATION_SPECS)} configurations")
    print(f"  Output    : {RESULTS_BASE}")
    print(f"{'#' * 80}\n")

    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ONCE — all ablations share the same weights ────────────────
    # vllm / hf_native → BF16 full precision (matches main EXP-24-PHI_4 run)
    # unsloth          → 4-bit (legacy fallback; produces flat logits for Phi-4)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    use_4bit = (backend == "unsloth")
    print(f"[LOAD] backend={backend}  model={MODEL_NAME}  4bit={use_4bit}")

    model, tokenizer = _load_model_timed(backend, use_4bit)

    cfg = _build_cfg(ABLATION_COUNTRIES, load_in_4bit=use_4bit)
    all_rows: List[Dict] = []

    # ── Per-country loop ──────────────────────────────────────────────────────
    for country in ABLATION_COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country}: not in SUPPORTED_COUNTRIES")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Country: {_CLR_BOLD}{country}{_CLR_RESET}")
        print("=" * 80)

        scenario_df = _load_scenarios(cfg, country)
        personas    = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        human_amce  = load_human_amce(HUMAN_AMCE_PATH, country)

        print(
            f"  Loaded {len(scenario_df)} scenarios"
            f"  |  {len(personas)} personas"
            f"  |  {len(human_amce)} human AMCE dimensions"
        )

        # ── Per-ablation inner loop ───────────────────────────────────────────
        for spec_idx, spec in enumerate(ABLATION_SPECS):
            print(f"\n  {'─' * 70}")
            print(
                f"  [{spec_idx}/{len(ABLATION_SPECS) - 1}]"
                f"  {_CLR_BOLD}{spec.row_label}{_CLR_RESET}"
                f"  —  {spec.description}"
            )
            print(f"  {'─' * 70}")

            _reset_prior_state(country)
            torch.cuda.empty_cache()
            gc.collect()

            t_start = time.time()
            # Full SWA-DPBR: load pre-computed results from exp_paper_phi_4.py
            # if available, so the reference row matches the main 500-scenario run.
            _precomp_csv = (
                _find_full_swa_csv(country)
                if spec.row_label == "Full SWA-DPBR"
                else None
            )
            if _precomp_csv is not None:
                print(f"  [FULL] Loading pre-computed results: {_precomp_csv}")
                results_df = pd.read_csv(_precomp_csv)
                summary    = _reconstruct_summary(results_df, country, cfg)
            else:
                results_df, summary = _run_ablation_country(
                    spec, model, tokenizer, country, personas, scenario_df, cfg
                )
            elapsed = time.time() - t_start

            # Save per-ablation CSV
            safe_tag = (
                spec.row_label
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "eq")
                .replace(",", "")
                .replace("α", "a")
                .replace("/", "_")
            )
            results_df.to_csv(
                out_dir / f"{country}_{safe_tag}_results.csv", index=False
            )

            row = _collect_row(spec, country, results_df, summary, elapsed)
            all_rows.append(row)

            a = summary.get("alignment", {})
            print(
                f"\n  ✓  {spec.row_label} | {country}"
                f"  JSD={_fmt(a.get('jsd',  float('nan')), '.4f')}"
                f"  r={_fmt(a.get('pearson_r', float('nan')), '+.3f')}"
                f"  ρ={_fmt(a.get('spearman_rho', float('nan')), '+.3f')}"
                f"  MAE={_fmt(a.get('mae', float('nan')), '.2f')}"
                f"  MIS={_fmt(a.get('mis', float('nan')), '.4f')}"
                f"  RMSE={_fmt(a.get('rmse', float('nan')), '.2f')}"
                f"  ({elapsed:.0f}s)"
            )

    # ── Cleanup model ─────────────────────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    summary_csv = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SAVED] {summary_csv}")

    # ── Print all report tables ───────────────────────────────────────────────
    for country in ABLATION_COUNTRIES:
        print(f"\n\n{'#' * 80}")
        print(f"  FINAL REPORT — {country}")
        print(f"{'#' * 80}")

        print_ablation_table(all_rows, country)
        print_per_dim_table(all_rows, country)
        print_per_dim_signed(all_rows, country)
        print_dpbr_diagnostics(all_rows, country)
        print_util_slopes(all_rows, country)

    if len(ABLATION_COUNTRIES) > 1:
        print_cross_country_summary(summary_df)

    # ── LaTeX snippet ─────────────────────────────────────────────────────────
    _print_latex_table(all_rows, ABLATION_COUNTRIES[0])

    print(f"\n{'#' * 80}")
    print(f"  EXP-24 Ablation COMPLETE  —  Phi-4 (14B)")
    print(f"  Results: {out_dir}")
    print(f"{'#' * 80}\n")


# ── LaTeX snippet generator ───────────────────────────────────────────────────

def _print_latex_table(rows: List[Dict], country: str) -> None:
    """Emit a copy-pasteable LaTeX ablation table (matches paper §Ablation)."""
    cr = [r for r in rows if r["country"] == country]
    if not cr:
        return

    ref = next((r for r in cr if r["ablation"] == "Full SWA-DPBR"), cr[0])

    def _δ(val: float, ref_val: float, low: bool = True) -> str:
        if not (np.isfinite(val) and np.isfinite(ref_val)):
            return "--"
        d    = val - ref_val
        sign = "+" if d > 0 else "$-$"
        tag  = r"\loss" if ((d > 0) == low) else r"\gain"
        return f"{tag}{{{sign}.{abs(d):05.3f}}}"

    lines = [
        r"\begin{table}[t]",
        (
            rf"\caption{{Ablation on Phi-4 (14B), {country} ({ref['n_scenarios']} scenarios),"
            r" \textbf{SWA-DPBR} with one"
        ),
        (
            r"$K$-sample importance-sampling batch per scenario"
            r" (dual-pass reliability disabled for isolation)."
        ),
        (
            rf"Full configuration: JSD = {ref['jsd']:.4f},"
            rf" $r$ = {ref['pearson_r']:.3f},"
            rf" MIS = {ref['mis']:.4f}."
            r"}"
        ),
        r"\label{tab:ablation_phi4}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\# & Configuration & JSD $\downarrow$ & $\Delta$JSD & "
        r"$r$ $\uparrow$ & $\Delta r$ & MIS $\downarrow$ & $\Delta$MIS \\",
        r"\midrule",
        (
            rf"  & \textbf{{Full SWA-DPBR}} & \textbf{{{ref['jsd']:.4f}}} & -- &"
            rf" \textbf{{{ref['pearson_r']:.3f}}} & -- &"
            rf" \textbf{{{ref['mis']:.4f}}} & -- \\"
        ),
        r"\midrule",
    ]

    row_idx = {spec.row_label: i for i, spec in enumerate(ABLATION_SPECS) if i > 0}
    for row in cr:
        if row["ablation"] == "Full SWA-DPBR":
            continue
        idx = row_idx.get(row["ablation"], "?")
        jsd_d = _δ(row["jsd"],       ref["jsd"],       low=True)
        r_d   = _δ(row["pearson_r"],  ref["pearson_r"], low=False)
        mis_d = _δ(row["mis"],        ref["mis"],       low=True)
        lines.append(
            rf"{idx} & {row['ablation']}"
            rf" & {row['jsd']:.4f} & {jsd_d}"
            rf" & {row['pearson_r']:.3f} & {r_d}"
            rf" & {row['mis']:.4f} & {mis_d} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    print(f"\n{'─' * 70}")
    print("  LaTeX ablation table snippet:")
    print("─" * 70)
    print("\n".join(lines))
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
