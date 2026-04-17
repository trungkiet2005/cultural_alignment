#!/usr/bin/env python3
"""
EXP-01: Stratified Hierarchical IS with Confidence Gating (SHIS-CG)
=====================================================================

**Motivation** (from experiment_DM/EXP-09 analysis):

EXP-09 (Hierarchical IS) achieves the best mean MIS (0.3975) across all
experiment_DM methods, but has three key weaknesses:

1. **Single global prior**: One delta_country across ALL dimension categories.
   SocialValue errors (consistently the worst dimension) contaminate Species/
   Gender priors, preventing targeted corrections.

2. **High flip rate (14-18%)**: The country prior can override clear micro-IS
   signals when agents strongly agree, causing unnecessary decision flips.

3. **No anchor regularization**: Extreme IS corrections from poorly calibrated
   models (Gemma, Mistral) get amplified by the prior, worsening regression.

**EXP-01 Fix: Three orthogonal improvements over EXP-09**

**Innovation 1 — Category-Stratified Prior**:
Instead of 1 global delta_country, maintain 6 category-specific priors (one per
phenomenon: Species, Gender, Age, Fitness, SocialValue, Utilitarianism). Each
category prior is initialized from the global prior after warmup and diverges
independently. When a category has < N_CAT_MATURE observations, blend with the
global prior for stability.

    delta_prior(cat) = w_cat * delta_cat + (1 - w_cat) * delta_global
    w_cat = min(n_cat_obs / N_CAT_MATURE, 1.0)

**Innovation 2 — Confidence-Gated Prior Application**:
Modulate alpha_h by agent agreement (inverse normalized variance). When agents
strongly agree (low variance -> high confidence), trust micro-IS more since it
has a clear signal. When agents disagree (high variance -> low confidence), lean
on the prior for stability.

    confidence = exp(-agent_variance / (sigma^2 * CONFIDENCE_SCALE))
    alpha_eff = alpha_h * (1 - confidence)

This directly targets the flip% problem: clear micro-IS signals pass through
unmodified, while noisy scenarios benefit from the prior.

**Innovation 3 — ESS-Adaptive Anchor Regularization** (from EXP-05):
After the hierarchical update, softly pull toward the agent consensus mean.
The regularization strength is inversely proportional to ESS quality:

    ess_ratio = k_eff / K
    anchor_weight = ANCHOR_STRENGTH * max(0, 1 - ess_ratio / (rho_eff * 3))
    delta_final = (1 - anchor_weight) * delta_hier + anchor_weight * anchor

When IS works well (high ESS) -> no regularization.
When IS collapses (low ESS) -> anchor regularization prevents extreme corrections.

**Expected improvements vs EXP-09**:
- Lower mean MIS through targeted category-specific corrections
- Reduced flip% through confidence gating (only apply prior when agents disagree)
- Better Gemma/Mistral performance through anchor regularization
- More robust prior through stratified structure

Usage on Kaggle
---------------
    !python experiment/exp01_stratified_hier_is.py
"""

# ============================================================================
# Step 0: env
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle():
    return os.path.isdir("/kaggle/working")


def _ensure_repo():
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle, not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE],
            check=True,
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps():
    if not _on_kaggle():
        return
    for c in [
        "pip install -q bitsandbytes scipy tqdm",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]:
        subprocess.run(c, shell=True, check=False)


_REPO_DIR = _ensure_repo()
_install_deps()


# ============================================================================
# Step 2: imports
# ============================================================================
import gc, shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn.functional as F
import pandas as pd

from src.config import BaselineConfig, SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.controller import ImplicitSWAController
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment
from src.baseline_runner import run_baseline_vanilla


# ============================================================================
# Step 3: experiment configuration
# ============================================================================
EXP_ID = "EXP-01"
EXP_NAME = "stratified_hier_is"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

# ---------------------------------------------------------------------------
# Stratified Hierarchical IS hyperparameters
# ---------------------------------------------------------------------------
N_WARMUP           = 30    # Global warmup (shorter: category priors activate sooner)
DECAY_TAU          = 120   # Annealing decay (slower than EXP-09's 100 for stability)
BETA_EMA           = 0.1   # Global prior EMA rate (same as EXP-09)
BETA_CAT           = 0.15  # Category prior EMA rate (faster: fewer obs per category)
N_CAT_MATURE       = 15    # Obs before fully trusting a category prior
CONFIDENCE_SCALE   = 2.0   # Variance scaling for confidence gating
ANCHOR_REG_STRENGTH = 0.25 # Max anchor regularization weight when ESS collapses

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_ROOT = "/kaggle/working/cultural_alignment/results/exp01_baseline"
SWA_ROOT  = "/kaggle/working/cultural_alignment/results/exp01_shis/swa"
CMP_ROOT  = "/kaggle/working/cultural_alignment/results/exp01_shis/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)


# ============================================================================
# Step 4: Stratified Prior State
# ============================================================================
class StratifiedPriorState:
    """
    Category-stratified country-level prior with confidence gating.

    Three-level hierarchy:
      Level 0: Global country prior (most stable, least specific)
      Level 1: Per-category country prior (targeted, fewer observations)
      Level 2: Scenario micro-IS (most specific, single observation)

    Blending:
      delta_prior(cat) = w_cat * delta_cat + (1 - w_cat) * delta_global
      w_cat = min(n_cat_obs / N_CAT_MATURE, 1.0)

    Confidence gating (anti-flip):
      confidence = exp(-agent_variance / (sigma^2 * CONFIDENCE_SCALE))
      alpha_eff = alpha_h * (1 - confidence)
      High confidence (agents agree) -> trust micro-IS -> alpha_eff ~ 0
      Low confidence (agents disagree) -> trust prior -> alpha_eff ~ alpha_h
    """

    def __init__(
        self,
        beta_global: float = BETA_EMA,
        beta_cat: float = BETA_CAT,
        decay_tau: float = DECAY_TAU,
        n_warmup: int = N_WARMUP,
        n_cat_mature: int = N_CAT_MATURE,
        confidence_scale: float = CONFIDENCE_SCALE,
    ):
        self.beta_global = beta_global
        self.beta_cat = beta_cat
        self.decay_tau = decay_tau
        self.n_warmup = n_warmup
        self.n_cat_mature = n_cat_mature
        self.confidence_scale = confidence_scale

        self.delta_global: float = 0.0
        self.cat_priors: Dict[str, float] = {}
        self.cat_counts: Dict[str, int] = {}
        self.step: int = 0
        self._history: List[float] = []

    # ----- Annealing schedule -----
    def alpha_h(self) -> float:
        """Base annealing: 0 during warmup, then 1 - exp(-t/tau)."""
        if self.step < self.n_warmup:
            return 0.0
        t = self.step - self.n_warmup
        return 1.0 - np.exp(-t / self.decay_tau)

    # ----- Blended prior retrieval -----
    def get_blended_prior(self, category: str) -> float:
        """Blend category-specific and global priors based on maturity."""
        if category not in self.cat_priors:
            return self.delta_global
        w = min(self.cat_counts.get(category, 0) / self.n_cat_mature, 1.0)
        return w * self.cat_priors[category] + (1.0 - w) * self.delta_global

    # ----- Confidence gating -----
    def confidence_gate(self, agent_variance: float, sigma: float) -> float:
        """
        Returns a value in [0, 1] that scales alpha_h:
          0 = agents agree perfectly -> don't apply prior (trust micro-IS)
          1 = agents disagree wildly -> fully apply prior
        """
        if sigma <= 0:
            return 1.0
        normalized_var = agent_variance / (sigma ** 2 * self.confidence_scale)
        confidence = np.exp(-normalized_var)
        return 1.0 - confidence

    # ----- Apply prior to micro-IS result -----
    def apply_prior(
        self,
        delta_micro: float,
        category: str,
        agent_variance: float,
        sigma: float,
    ) -> float:
        """
        Mix micro-IS with stratified prior, gated by confidence.

        During warmup: pure micro-IS (no prior applied).
        After warmup:
          prior = blended(category, global)
          alpha_eff = alpha_h * confidence_gate
          result = alpha_eff * prior + (1 - alpha_eff) * micro
        """
        if self.step < self.n_warmup:
            return delta_micro

        prior = self.get_blended_prior(category)
        alpha = self.alpha_h()
        gate = self.confidence_gate(agent_variance, sigma)
        alpha_eff = alpha * gate

        return alpha_eff * prior + (1.0 - alpha_eff) * delta_micro

    # ----- Update priors with new observation -----
    def update(self, delta_micro: float, category: str) -> None:
        """Update both global and category-specific priors via EMA."""
        self.step += 1

        # Global prior (EMA)
        self.delta_global = (
            (1.0 - self.beta_global) * self.delta_global
            + self.beta_global * delta_micro
        )

        # Category prior (EMA, initialize from current global)
        if category not in self.cat_priors:
            self.cat_priors[category] = self.delta_global
            self.cat_counts[category] = 0
        self.cat_priors[category] = (
            (1.0 - self.beta_cat) * self.cat_priors[category]
            + self.beta_cat * delta_micro
        )
        self.cat_counts[category] = self.cat_counts.get(category, 0) + 1

        self._history.append(delta_micro)

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step,
            "delta_global": self.delta_global,
            "alpha_h": self.alpha_h(),
            "n_categories": len(self.cat_priors),
            "cat_priors": dict(self.cat_priors),
            "cat_counts": dict(self.cat_counts),
        }


# Global state: reset per country x model
_COUNTRY_PRIOR_STATE: Dict[str, StratifiedPriorState] = {}


# ============================================================================
# Step 5: Stratified Hierarchical Controller
# ============================================================================
class Exp01StratifiedController(ImplicitSWAController):
    """
    Stratified Hierarchical IS with Confidence Gating (SHIS-CG).

    Three innovations over EXP-09:
      1. Category-stratified prior (6 priors instead of 1 global)
      2. Confidence-gated prior application (anti-flip mechanism)
      3. ESS-adaptive anchor regularization (prevents extreme corrections)
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior_state(self) -> StratifiedPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = StratifiedPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(
        self,
        user_query,
        preferred_on_right=True,
        phenomenon_category="default",
        lang="en",
    ):
        # ---- Standard SWA-PTIS: two-pass positional debiasing ----
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

        delta_base = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ---- Adaptive proposal sigma (floored at noise_std) ----
        sigma = max(
            float(delta_agents.std(unbiased=True).item())
            if delta_agents.numel() >= 2
            else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()
        K, device = self.K, self.device

        # ---- K-sample IS with Prospect-Theory utility ----
        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (
            delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)
        ).abs()
        g_per_agent = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent = self._pt_value(g_per_agent)
        mean_v = v_per_agent.mean(dim=1)

        g_cons = (
            (delta_base - anchor).abs() - (delta_tilde - anchor).abs()
        ) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        # ---- ESS gate ----
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_ratio = float(k_eff.item()) / K
        if ess_ratio < self.rho_eff:
            delta_star = torch.zeros((), device=device)
        else:
            delta_star = torch.sum(w * eps)

        delta_opt_micro = float((anchor + delta_star).item())
        agent_variance = (
            float(delta_agents.var(unbiased=True).item())
            if delta_agents.numel() > 1
            else 0.0
        )
        anchor_f = float(anchor.item())

        # ===========================================================
        # INNOVATION 1+2: Stratified Prior + Confidence Gating
        # ===========================================================
        prior_state = self._get_prior_state()
        delta_opt_hier = prior_state.apply_prior(
            delta_opt_micro, phenomenon_category, agent_variance, sigma
        )

        # ===========================================================
        # INNOVATION 3: ESS-Adaptive Anchor Regularization
        # Poor ESS -> pull back toward agent consensus to prevent
        # extreme corrections (especially helps Gemma/Mistral)
        # ===========================================================
        if ess_ratio < self.rho_eff:
            reg_weight = ANCHOR_REG_STRENGTH
        else:
            reg_weight = ANCHOR_REG_STRENGTH * max(
                0.0, 1.0 - ess_ratio / (self.rho_eff * 3.0)
            )

        delta_opt_final = (
            (1.0 - reg_weight) * delta_opt_hier + reg_weight * anchor_f
        )

        # Update prior state with micro-IS result
        prior_state.update(delta_opt_micro, phenomenon_category)

        prior_stats = prior_state.stats

        # ---- Final probability ----
        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": agent_variance,
            "sigma_used": sigma,
            "mppi_flipped": (anchor_f > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - anchor_f),
            "delta_consensus": anchor_f,
            "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_global": prior_stats["delta_global"],
            "alpha_h": prior_stats["alpha_h"],
            "prior_step": prior_stats["step"],
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


# Patch runner to use our controller
_swa_runner_mod.ImplicitSWAController = Exp01StratifiedController


# ============================================================================
# Step 6: Runner helpers
# ============================================================================
def _free_model_cache(model_name):
    safe = "models--" + model_name.replace("/", "--")
    for root in [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]:
        if not root:
            continue
        hub_dir = (
            root
            if os.path.basename(root.rstrip("/")) == "hub"
            else os.path.join(root, "hub")
        )
        target = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name):
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=SWA_ROOT,
        lambda_coop=0.7,
        K_samples=128,
    )


def _build_baseline_config(model_name):
    return BaselineConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=BASE_ROOT,
    )


def _load_country_scenarios(cfg, country):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path,
            lang=lang,
            translator=cfg.multitp_translator,
            suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_baseline_for_model(model, tokenizer, model_name):
    """Run vanilla LLM baseline (no personas, no IS)."""
    cfg = _build_baseline_config(model_name)
    out_dir = Path(BASE_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# BASELINE [{model_name}]\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(
            f"\n[{ci+1}/{len(cfg.target_countries)}] Baseline "
            f"{model_name} | {country}"
        )
        scen = _load_country_scenarios(cfg, country)
        bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        bl["results_df"].to_csv(
            out_dir / f"vanilla_results_{country}.csv", index=False
        )
        rows.append(
            {
                "model": model_name,
                "method": "baseline_vanilla",
                "country": country,
                **{f"align_{k}": v for k, v in bl["alignment"].items()},
                "n_scenarios": len(bl["results_df"]),
            }
        )
        torch.cuda.empty_cache()
        gc.collect()
    return rows


def _run_swa_for_model(model, tokenizer, model_name):
    """Run SHIS-CG (our method) on every TARGET_COUNTRY for one model."""
    cfg = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} SHIS-CG [{model_name}]\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue

        # Reset prior state for each country x model combination
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = StratifiedPriorState()
        print(
            f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} "
            f"{model_name} | {country} (prior reset)"
        )

        scen = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        # Inject country into controller via monkey-patch
        orig_init = Exp01StratifiedController.__init__

        def patched_init(self, *args, country=country, **kwargs):
            orig_init(self, *args, country=country, **kwargs)

        Exp01StratifiedController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp01StratifiedController

        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg
        )
        Exp01StratifiedController.__init__ = orig_init  # restore

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        prior_stats = _COUNTRY_PRIOR_STATE.get(
            country, StratifiedPriorState()
        ).stats

        rows.append(
            {
                "model": model_name,
                "method": f"{EXP_ID}_shis_cg",
                "country": country,
                **{f"align_{k}": v for k, v in summary["alignment"].items()},
                "flip_rate": summary["flip_rate"],
                "mean_latency_ms": summary["mean_latency_ms"],
                "n_scenarios": summary["n_scenarios"],
                "final_delta_global": prior_stats["delta_global"],
                "final_alpha_h": prior_stats["alpha_h"],
                "n_cat_priors": prior_stats["n_categories"],
            }
        )
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 7: Output — 3-column comparison table
# ============================================================================
def _print_comparison_table(cmp_df: pd.DataFrame) -> None:
    """
    Print the paper-ready comparison table with 3 columns:
      Vanilla MIS | Method MIS | Improvement %
    One block per model. Saves mis_comparison.csv.
    """
    if cmp_df.empty or "method" not in cmp_df.columns:
        return
    if "align_mis" not in cmp_df.columns:
        print("\n[WARN] align_mis column not found — skipping comparison.")
        return

    methods_present = set(cmp_df["method"])
    if "baseline_vanilla" not in methods_present:
        print("\n[WARN] No baseline_vanilla rows — cannot compare.")
        return
    method_names = [m for m in methods_present if m != "baseline_vanilla"]
    if not method_names:
        return
    method_name = method_names[0]

    out_rows: List[dict] = []
    width = 78

    all_vanilla, all_method = [], []

    for model_name in cmp_df["model"].drop_duplicates().tolist():
        mdf = cmp_df[cmp_df["model"] == model_name]
        baseline = (
            mdf[mdf["method"] == "baseline_vanilla"]
            .drop_duplicates("country")
            .set_index("country")["align_mis"]
        )
        method = (
            mdf[mdf["method"] == method_name]
            .drop_duplicates("country")
            .set_index("country")["align_mis"]
        )
        common = [c for c in baseline.index if c in method.index]
        if not common:
            continue

        b = baseline.loc[common].astype(float)
        m = method.loc[common].astype(float)
        short_model = model_name.split("/")[-1] if "/" in model_name else model_name

        print(f"\n{'='*width}")
        print(f"  {EXP_ID} SHIS-CG  |  MIS Comparison  |  lower = better")
        print(f"  MODEL: {short_model}")
        print(f"{'='*width}")
        print(
            f"  {'Country':>8}  {'Vanilla MIS':>12}  "
            f"{'Method MIS':>12}  {'Improve %':>10}"
        )
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")

        wins = 0
        for country in common:
            bv = float(b.loc[country])
            mv = float(m.loc[country])
            imp = (100.0 * (bv - mv) / bv) if bv != 0 else 0.0
            marker = "+" if imp >= 0 else ""
            tag = "  \u2713" if imp > 0 else "  \u2717"
            print(
                f"  {country:>8}  {bv:12.4f}  {mv:12.4f}  "
                f"{marker}{imp:9.2f}%{tag}"
            )
            if imp > 0:
                wins += 1
            all_vanilla.append(bv)
            all_method.append(mv)
            out_rows.append(
                {
                    "model": model_name,
                    "country": country,
                    "vanilla_mis": bv,
                    "method_mis": mv,
                    "improve_pct": imp,
                }
            )

        mean_b = float(b.mean())
        mean_m = float(m.mean())
        mean_imp = (100.0 * (mean_b - mean_m) / mean_b) if mean_b != 0 else 0.0
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
        d = "+" if mean_imp >= 0 else ""
        print(
            f"  {'MEAN':>8}  {mean_b:12.4f}  {mean_m:12.4f}  "
            f"{d}{mean_imp:9.2f}%"
        )
        print(f"  Wins: {wins}/{len(common)}")
        print(f"{'='*width}")

    # Global summary across all models
    if all_vanilla and all_method:
        gv = np.mean(all_vanilla)
        gm = np.mean(all_method)
        gi = (100.0 * (gv - gm) / gv) if gv != 0 else 0.0
        print(f"\n{'*'*width}")
        print(f"  GLOBAL MEAN (all models x countries)")
        print(f"  Vanilla MIS: {gv:.4f}  |  Method MIS: {gm:.4f}  |  Improve: +{gi:.2f}%")
        print(f"{'*'*width}")

    if out_rows:
        out_path = Path(CMP_ROOT) / "mis_comparison.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"\n[SAVE] MIS comparison -> {out_path}  ({len(out_rows)} rows)")


# ============================================================================
# Step 8: main
# ============================================================================
def main():
    setup_seeds(SEED)
    for d in (BASE_ROOT, SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] Stratified Hierarchical IS with Confidence Gating (SHIS-CG)")
    print(f"[CONFIG] N_WARMUP={N_WARMUP}, DECAY_TAU={DECAY_TAU}")
    print(f"[CONFIG] BETA_EMA={BETA_EMA}, BETA_CAT={BETA_CAT}")
    print(f"[CONFIG] N_CAT_MATURE={N_CAT_MATURE}, CONFIDENCE_SCALE={CONFIDENCE_SCALE}")
    print(f"[CONFIG] ANCHOR_REG_STRENGTH={ANCHOR_REG_STRENGTH}")

    all_rows: List[dict] = []

    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}")
        print(f"  MODEL {mi+1}/{len(MODELS)}: {model_name}")
        print(f"{'='*70}")
        model, tokenizer = load_model(
            model_name, max_seq_length=2048, load_in_4bit=True
        )
        try:
            # Run vanilla baseline first
            all_rows.extend(
                _run_baseline_for_model(model, tokenizer, model_name)
            )
            # Run our method
            all_rows.extend(
                _run_swa_for_model(model, tokenizer, model_name)
            )
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _free_model_cache(model_name)

        # Incremental save
        pd.DataFrame(all_rows).to_csv(
            Path(CMP_ROOT) / "comparison.csv", index=False
        )

    # Final save + comparison table
    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    _print_comparison_table(cmp_df)

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
