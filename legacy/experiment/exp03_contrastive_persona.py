#!/usr/bin/env python3
"""
EXP-03: Contrastive Persona Decoding + Dim-Adaptive PT + Stratified Prior
===========================================================================

**Why EXP-02 is not enough:**

EXP-02's dimension-adaptive PT addresses the PT parameter mismatch, but the
ROOT anchor bias remains: all WVS-derived personas are egalitarian, all share
the same instruction-tuning bias from the frozen LLM, and the utilitarian
persona is country-invariant. These biases are ADDITIVE and STRUCTURAL — they
shift the anchor by a constant offset regardless of target country.

The cultural SIGNAL (what makes Japan different from Brazil) is a small
perturbation on top of this large shared bias.

**EXP-03 Fix: Contrastive Persona Decoding (CPD)**

Run BOTH country-specific and "world-average" personas on the same scenario.
The difference isolates the pure cultural signal:

    cultural_signal = delta_country - delta_world

In logit space (contrastive decoding, Li et al. 2023):
    delta_corrected = (1 + lambda) * delta_country - lambda * delta_world
                    = delta_country + lambda * cultural_signal

Implementation: pass both country (N=4) and world-average (N=4) personas
to the controller as a single 8-persona pool. In predict(), split them
and compute the contrastive correction before PT-IS.

**Combined innovations (full stack):**
  1. Contrastive Persona Decoding (this experiment)
  2. Dimension-specific kappa/sigma (from EXP-02)
  3. Category-stratified hierarchical prior (from EXP-01)
  4. Confidence-gated prior application (from EXP-01)
  5. ESS-adaptive anchor regularization (from EXP-01)

**Expected improvements:**
  - SocialValue: egalitarian bias cancels in the subtraction
  - Better cultural specificity: signal that differentiates countries is amplified
  - Reduced Gemma over-correction: shared bias removed before PT-IS
  - Cross-model: works for any LLM (shared bias is model-specific but consistent)
  - Cross-language: contrastive subtraction is language-agnostic

Usage on Kaggle
---------------
    !python experiment/exp03_contrastive_persona.py
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
EXP_ID = "EXP-03"
EXP_NAME = "contrastive_persona"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

# Contrastive decoding strength
LAMBDA_CONTRAST = 0.5  # alpha in (1+alpha) * country - alpha * world

# Dimension-Specific PT Parameters (from EXP-02)
DIMENSION_PT_PARAMS: Dict[str, Dict[str, float]] = {
    "Species":        {"kappa": 3.00, "alpha": 0.88, "beta": 0.88, "sigma_scale": 0.8},
    "Gender":         {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0},
    "Age":            {"kappa": 1.75, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.2},
    "Fitness":        {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0},
    "SocialValue":    {"kappa": 1.25, "alpha": 0.85, "beta": 0.90, "sigma_scale": 1.5},
    "Utilitarianism": {"kappa": 2.00, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.1},
}
DEFAULT_PT_PARAMS = {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0}

# Stratified Prior hyperparameters (from EXP-01)
N_WARMUP            = 30
DECAY_TAU           = 120
BETA_EMA            = 0.1
BETA_CAT            = 0.15
N_CAT_MATURE        = 15
CONFIDENCE_SCALE    = 2.0
ANCHOR_REG_STRENGTH = 0.25

# Paths
BASE_ROOT = "/kaggle/working/cultural_alignment/results/exp03_baseline"
SWA_ROOT  = "/kaggle/working/cultural_alignment/results/exp03_contrastive/swa"
CMP_ROOT  = "/kaggle/working/cultural_alignment/results/exp03_contrastive/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)


# ============================================================================
# Step 4a: World-Average Personas
# ============================================================================
WORLD_AVERAGE_PERSONAS: List[str] = [
    (
        "You are a young adult (age 18-35) representing the global average. "
        "You hold moderate views that reflect the worldwide median on social "
        "values: moderate religiosity, moderate gender equality support, "
        "moderate social trust, and moderate tolerance for diversity. "
        "You do not identify with any particular country or culture. "
        "When making moral judgments, you weigh all considerations equally "
        "without strong bias toward any particular moral framework."
    ),
    (
        "You are a middle-aged adult (age 36-55) representing the global average. "
        "Your values reflect the worldwide median across all cultures: moderate "
        "conservatism balanced with progressive views, average levels of social "
        "trust and institutional confidence. You approach moral dilemmas with "
        "a balanced perspective that does not favor any specific cultural tradition."
    ),
    (
        "You are an older adult (age 55+) representing the global average. "
        "Your values reflect the accumulated wisdom of diverse global traditions: "
        "moderate religiosity, respect for both individual rights and collective "
        "responsibility, and pragmatic moral reasoning. You do not identify with "
        "any particular national culture."
    ),
    (
        "You are a moral philosopher committed to maximizing the total number "
        "of lives saved. When faced with a trolley-problem dilemma, you always "
        "prefer the option that saves more people, regardless of their social "
        "status, age, gender, or species."
    ),
]

N_WORLD_PERSONAS = len(WORLD_AVERAGE_PERSONAS)


# ============================================================================
# Step 4b: Stratified Prior State (from EXP-01)
# ============================================================================
class StratifiedPriorState:
    """Category-stratified country-level prior with confidence gating."""

    def __init__(self):
        self.delta_global: float = 0.0
        self.cat_priors: Dict[str, float] = {}
        self.cat_counts: Dict[str, int] = {}
        self.step: int = 0

    def alpha_h(self):
        if self.step < N_WARMUP:
            return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def get_blended_prior(self, category):
        if category not in self.cat_priors:
            return self.delta_global
        w = min(self.cat_counts.get(category, 0) / N_CAT_MATURE, 1.0)
        return w * self.cat_priors[category] + (1.0 - w) * self.delta_global

    def confidence_gate(self, agent_variance, sigma):
        if sigma <= 0: return 1.0
        return 1.0 - np.exp(-agent_variance / (sigma ** 2 * CONFIDENCE_SCALE))

    def apply_prior(self, delta_micro, category, agent_variance, sigma):
        if self.step < N_WARMUP: return delta_micro
        prior = self.get_blended_prior(category)
        alpha_eff = self.alpha_h() * self.confidence_gate(agent_variance, sigma)
        return alpha_eff * prior + (1.0 - alpha_eff) * delta_micro

    def update(self, delta_micro, category):
        self.step += 1
        self.delta_global = (1.0 - BETA_EMA) * self.delta_global + BETA_EMA * delta_micro
        if category not in self.cat_priors:
            self.cat_priors[category] = self.delta_global
            self.cat_counts[category] = 0
        self.cat_priors[category] = (1.0 - BETA_CAT) * self.cat_priors[category] + BETA_CAT * delta_micro
        self.cat_counts[category] = self.cat_counts.get(category, 0) + 1

    @property
    def stats(self):
        return {"step": self.step, "delta_global": self.delta_global, "alpha_h": self.alpha_h()}


_COUNTRY_PRIOR_STATE: Dict[str, StratifiedPriorState] = {}


# ============================================================================
# Step 5: Contrastive + Dim-Adaptive Controller
# ============================================================================
class Exp03ContrastiveController(ImplicitSWAController):
    """
    Contrastive Persona Decoding + Dim-Adaptive PT + Stratified Prior.

    The controller receives 8 personas: first N_country are country-specific,
    remaining N_world are world-average. In predict(), it splits them, computes
    the contrastive correction, then applies dim-adaptive PT-IS.

    Innovations:
      1. Contrastive decoding: anchor = anchor_country + lambda * cultural_signal
      2. Dimension-specific kappa/sigma (from EXP-02)
      3. Category-stratified prior + confidence gating (from EXP-01)
      4. ESS-adaptive anchor regularization (from EXP-01)
    """

    def __init__(self, *args, country: str = "UNKNOWN",
                 n_country_personas: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country
        self.n_country_personas = n_country_personas

    def _get_prior_state(self) -> StratifiedPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = StratifiedPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True,
                phenomenon_category="default", lang="en"):
        # ── Dimension-specific PT params ──
        dim_p = DIMENSION_PT_PARAMS.get(phenomenon_category, DEFAULT_PT_PARAMS)
        kappa_d, alpha_d, beta_d = dim_p["kappa"], dim_p["alpha"], dim_p["beta"]
        sigma_scale = dim_p["sigma_scale"]

        # ── Two-pass debiasing (evaluates ALL 8 personas) ──
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
        delta_agents_all = (da1 - da2) / 2.0 if swap_changed else da1

        # ── Split into country and world personas ──
        n_c = self.n_country_personas
        delta_country = delta_agents_all[:n_c]
        delta_world = delta_agents_all[n_c:]

        # ── Contrastive correction ──
        n_min = min(delta_country.numel(), delta_world.numel())
        if n_min > 0 and delta_world.numel() > 0:
            cultural_signal = delta_country[:n_min] - delta_world[:n_min]
            mean_signal = cultural_signal.mean()
            anchor_country = delta_country.mean()
            # delta_corrected = (1+lambda) * country - lambda * world
            anchor_corrected = anchor_country + LAMBDA_CONTRAST * mean_signal
        else:
            anchor_corrected = delta_agents_all.mean()
            mean_signal = torch.tensor(0.0)

        # Use country personas for PT-IS (they carry the cultural-corrected signal)
        delta_agents = delta_country

        # ── Dimension-scaled adaptive sigma ──
        base_sigma = max(
            float(delta_agents.std(unbiased=True).item())
            if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        sigma = base_sigma * sigma_scale
        K, device = self.K, self.device

        # ── K-sample IS with dim-specific PT ──
        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor_corrected + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (
            delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)
        ).abs()
        g_per_agent = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        v_per_agent = torch.where(
            g_per_agent >= 0,
            g_per_agent.abs().pow(alpha_d),
            -kappa_d * g_per_agent.abs().pow(beta_d),
        )
        mean_v = v_per_agent.mean(dim=1)

        g_cons = (
            (delta_base - anchor_corrected).abs()
            - (delta_tilde - anchor_corrected).abs()
        ) / sigma
        v_cons = torch.where(
            g_cons >= 0, g_cons.abs().pow(alpha_d),
            -kappa_d * g_cons.abs().pow(beta_d),
        )

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_ratio = float(k_eff.item()) / K
        delta_star = (
            torch.sum(w * eps) if ess_ratio >= self.rho_eff
            else torch.zeros((), device=device)
        )

        delta_opt_micro = float((anchor_corrected + delta_star).item())
        agent_variance = (
            float(delta_agents.var(unbiased=True).item())
            if delta_agents.numel() > 1 else 0.0
        )
        anchor_f = float(anchor_corrected.item())

        # ── Stratified Prior + Confidence Gating ──
        prior_state = self._get_prior_state()
        delta_opt_hier = prior_state.apply_prior(
            delta_opt_micro, phenomenon_category, agent_variance, sigma
        )

        # ── ESS-Adaptive Anchor Reg ──
        if ess_ratio < self.rho_eff:
            reg_weight = ANCHOR_REG_STRENGTH
        else:
            reg_weight = ANCHOR_REG_STRENGTH * max(
                0.0, 1.0 - ess_ratio / (self.rho_eff * 3.0)
            )
        delta_opt_final = (1.0 - reg_weight) * delta_opt_hier + reg_weight * anchor_f

        prior_state.update(delta_opt_micro, phenomenon_category)

        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right

        return {
            "p_right": p_right, "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": agent_variance, "sigma_used": sigma,
            "mppi_flipped": (anchor_f > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - anchor_f),
            "delta_consensus": anchor_f, "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "cultural_signal": float(mean_signal.item()),
            "lambda_contrast": LAMBDA_CONTRAST,
            "kappa_d": kappa_d,
            "delta_global": prior_state.stats["delta_global"],
            "alpha_h": prior_state.stats["alpha_h"],
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp03ContrastiveController


# ============================================================================
# Step 6: Runner helpers
# ============================================================================
def _free_model_cache(model_name):
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root: continue
        hub_dir = (root if os.path.basename(root.rstrip("/")) == "hub"
                   else os.path.join(root, "hub"))
        target = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try: shutil.rmtree(target)
            except Exception: pass


def _build_swa_config(model_name):
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=SWA_ROOT,
        lambda_coop=0.7, K_samples=128,
    )


def _build_baseline_config(model_name):
    return BaselineConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=BASE_ROOT,
    )


def _load_country_scenarios(cfg, country):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy(); df["lang"] = lang
    return df


def _run_baseline_for_model(model, tokenizer, model_name):
    cfg = _build_baseline_config(model_name)
    out_dir = Path(BASE_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True); cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# BASELINE [{model_name}]\n{'#'*70}")
    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] Baseline {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)
        bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
        rows.append({"model": model_name, "method": "baseline_vanilla", "country": country,
                      **{f"align_{k}": v for k, v in bl["alignment"].items()},
                      "n_scenarios": len(bl["results_df"])})
        torch.cuda.empty_cache(); gc.collect()
    return rows


def _run_swa_for_model(model, tokenizer, model_name):
    cfg = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True); cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} CONTRASTIVE [{model_name}]\n{'#'*70}")
    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = StratifiedPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")

        scen = _load_country_scenarios(cfg, country)
        country_personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        n_country = len(country_personas)

        # Combine country + world-average personas
        all_personas = country_personas + WORLD_AVERAGE_PERSONAS

        # Inject country + n_country_personas into controller
        orig_init = Exp03ContrastiveController.__init__
        def patched_init(self, *args, country=country,
                         n_country_personas=n_country, **kwargs):
            orig_init(self, *args, country=country,
                      n_country_personas=n_country_personas, **kwargs)
        Exp03ContrastiveController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp03ContrastiveController

        results_df, summary = run_country_experiment(
            model, tokenizer, country, all_personas, scen, cfg
        )
        Exp03ContrastiveController.__init__ = orig_init
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        rows.append({"model": model_name, "method": f"{EXP_ID}_contrastive",
                      "country": country,
                      **{f"align_{k}": v for k, v in summary["alignment"].items()},
                      "flip_rate": summary["flip_rate"],
                      "mean_latency_ms": summary["mean_latency_ms"],
                      "n_scenarios": summary["n_scenarios"]})
        torch.cuda.empty_cache(); gc.collect()
    return rows


# ============================================================================
# Step 7: Output
# ============================================================================
def _print_comparison_table(cmp_df):
    if cmp_df.empty or "align_mis" not in cmp_df.columns: return
    methods_present = set(cmp_df["method"])
    if "baseline_vanilla" not in methods_present: return
    method_name = [m for m in methods_present if m != "baseline_vanilla"]
    if not method_name: return
    method_name = method_name[0]
    out_rows, all_v, all_m = [], [], []
    width = 78
    for model_name in cmp_df["model"].drop_duplicates().tolist():
        mdf = cmp_df[cmp_df["model"] == model_name]
        bl = mdf[mdf["method"] == "baseline_vanilla"].drop_duplicates("country").set_index("country")["align_mis"]
        mt = mdf[mdf["method"] == method_name].drop_duplicates("country").set_index("country")["align_mis"]
        common = [c for c in bl.index if c in mt.index]
        if not common: continue
        b, m = bl.loc[common].astype(float), mt.loc[common].astype(float)
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n{'='*width}")
        print(f"  {EXP_ID} CONTRASTIVE  |  MIS Comparison  |  lower = better")
        print(f"  MODEL: {short}")
        print(f"{'='*width}")
        print(f"  {'Country':>8}  {'Vanilla MIS':>12}  {'Method MIS':>12}  {'Improve %':>10}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
        wins = 0
        for c in common:
            bv, mv = float(b.loc[c]), float(m.loc[c])
            imp = (100.0 * (bv - mv) / bv) if bv != 0 else 0.0
            tag = "  \u2713" if imp > 0 else "  \u2717"
            sign = "+" if imp >= 0 else ""
            print(f"  {c:>8}  {bv:12.4f}  {mv:12.4f}  {sign}{imp:9.2f}%{tag}")
            if imp > 0: wins += 1
            all_v.append(bv); all_m.append(mv)
            out_rows.append({"model": model_name, "country": c,
                             "vanilla_mis": bv, "method_mis": mv, "improve_pct": imp})
        mean_b, mean_m = float(b.mean()), float(m.mean())
        mean_imp = (100.0 * (mean_b - mean_m) / mean_b) if mean_b != 0 else 0.0
        d = "+" if mean_imp >= 0 else ""
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
        print(f"  {'MEAN':>8}  {mean_b:12.4f}  {mean_m:12.4f}  {d}{mean_imp:9.2f}%")
        print(f"  Wins: {wins}/{len(common)}")
        print(f"{'='*width}")
    if all_v and all_m:
        gv, gm = np.mean(all_v), np.mean(all_m)
        gi = (100.0 * (gv - gm) / gv) if gv != 0 else 0.0
        print(f"\n{'*'*width}")
        print(f"  GLOBAL MEAN (all models x countries)")
        print(f"  Vanilla: {gv:.4f}  |  Method: {gm:.4f}  |  Improve: +{gi:.2f}%")
        print(f"  EXP-09 benchmark: 0.3975  |  Beat EXP-09: {'YES' if gm < 0.3975 else 'NO'}")
        print(f"{'*'*width}")
    if out_rows:
        pd.DataFrame(out_rows).to_csv(Path(CMP_ROOT) / "mis_comparison.csv", index=False)


# ============================================================================
# Step 8: main
# ============================================================================
def main():
    setup_seeds(SEED)
    for d in (BASE_ROOT, SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] Contrastive Persona Decoding + Dim-Adaptive PT + Stratified Prior")
    print(f"[CONFIG] LAMBDA_CONTRAST={LAMBDA_CONTRAST}")
    print(f"[CONFIG] World-average personas: {N_WORLD_PERSONAS}")
    print(f"[CONFIG] Total personas per country: country(4) + world({N_WORLD_PERSONAS}) = {4 + N_WORLD_PERSONAS}")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_baseline_for_model(model, tokenizer, model_name))
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    _print_comparison_table(cmp_df)
    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
