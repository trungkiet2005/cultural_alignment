#!/usr/bin/env python3
"""
EXP-04: Model-Adaptive Grand Fusion
=====================================

**Why one-size-fits-all fails:**

The tracker reveals that models fail in FUNDAMENTALLY DIFFERENT ways:
  - Qwen:    Best overall, but SocialValue -27pp (egalitarian persona bias)
  - Gemma:   Over-corrects in USA/CHN/JPN (anchor >> base, MIS worse than vanilla)
  - Mistral: Anti-correlation everywhere (Pearson r <= -0.57, tokenizer collapse)

Each failure has a KNOWN FIX from prior experiments:
  - Qwen   -> Social-utility personas (EXP-03 DM) + lower lambda_coop=0.60
  - Gemma  -> Strict ESS gate (rho_eff=0.15) + tighter sigma_0=0.25
  - Mistral -> English personas (EXP-04 DM) + wider sigma_0=0.8 + K=512

But applying the wrong fix to the wrong model HURTS performance.

**EXP-04: Auto-detect model family + apply best-known config + ALL innovations**

This is the GRAND FUSION that combines:
  1. Model-adaptive profiles (per-model lambda_coop, sigma, K, rho_eff)
  2. Social-utility personas for Qwen (fixes SocialValue bottleneck)
  3. English personas for Mistral (fixes tokenizer variance collapse)
  4. Dimension-adaptive PT (from EXP-02)
  5. Contrastive persona decoding (from EXP-03)
  6. Category-stratified hierarchical prior (from EXP-01)
  7. Confidence-gated prior application (from EXP-01)
  8. ESS-adaptive anchor regularization (from EXP-01)

**Expected results:**
  - Best mean MIS across ALL 3 models (each gets its optimal config)
  - Qwen: SV personas + dim-adaptive PT -> SocialValue gap halved
  - Gemma: strict ESS + anchor reg -> no more over-correction
  - Mistral: English personas + wider sigma -> positive correlation in more countries

Usage on Kaggle
---------------
    !python experiment/exp04_model_adaptive_fusion.py
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
from dataclasses import dataclass

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
EXP_ID = "EXP-04"
EXP_NAME = "model_adaptive_fusion"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

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

# Contrastive decoding (from EXP-03)
LAMBDA_CONTRAST = 0.5

# Stratified Prior (from EXP-01)
N_WARMUP            = 30
DECAY_TAU           = 120
BETA_EMA            = 0.1
BETA_CAT            = 0.15
N_CAT_MATURE        = 15
CONFIDENCE_SCALE    = 2.0
ANCHOR_REG_STRENGTH = 0.25

# Paths
BASE_ROOT = "/kaggle/working/cultural_alignment/results/exp04_baseline"
SWA_ROOT  = "/kaggle/working/cultural_alignment/results/exp04_fusion/swa"
CMP_ROOT  = "/kaggle/working/cultural_alignment/results/exp04_fusion/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)


# ============================================================================
# Step 4a: Model Profiles
# ============================================================================
@dataclass
class ModelProfile:
    """Model-specific hyperparameter profile."""
    family: str
    use_social_utility_personas: bool
    force_english_personas: bool
    lambda_coop: float
    noise_std: float
    K_samples: int
    rho_eff: float
    decision_temperature: float


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "qwen": ModelProfile(
        family="qwen",
        use_social_utility_personas=True,   # Fixes SocialValue -27pp
        force_english_personas=False,
        lambda_coop=0.60,                   # Let SV personas speak
        noise_std=0.3,
        K_samples=128,
        rho_eff=0.10,
        decision_temperature=1.0,
    ),
    "gemma": ModelProfile(
        family="gemma",
        use_social_utility_personas=False,  # Hurts Gemma (tracker evidence)
        force_english_personas=False,
        lambda_coop=0.70,
        noise_std=0.25,                     # Tighter to reduce over-correction
        K_samples=128,
        rho_eff=0.15,                       # Stricter ESS gate
        decision_temperature=1.0,
    ),
    "mistral": ModelProfile(
        family="mistral",
        use_social_utility_personas=False,
        force_english_personas=True,        # Fixes tokenizer variance collapse
        lambda_coop=0.70,
        noise_std=0.8,                      # Wider proposals for exploration
        K_samples=512,                      # More samples for stability
        rho_eff=0.10,
        decision_temperature=0.5,           # Sharper decisions
    ),
}

DEFAULT_PROFILE = ModelProfile(
    family="default",
    use_social_utility_personas=False,
    force_english_personas=False,
    lambda_coop=0.70,
    noise_std=0.3,
    K_samples=128,
    rho_eff=0.10,
    decision_temperature=1.0,
)


def detect_model_family(model_name: str) -> str:
    lower = model_name.lower()
    for key in ("qwen", "gemma", "mistral", "llama"):
        if key in lower:
            return key
    return "default"


def get_model_profile(model_name: str) -> ModelProfile:
    return MODEL_PROFILES.get(detect_model_family(model_name), DEFAULT_PROFILE)


# ============================================================================
# Step 4b: Social-Utility Personas (native language, for Qwen)
# ============================================================================
SOCIAL_UTILITY_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        ("You are a senior emergency-room physician in the United States. "
         "You have spent 20 years making life-or-death triage decisions. "
         "Your professional ethics training emphasizes that individuals who "
         "contribute critical skills to society represent irreplaceable human "
         "capital. When forced to choose, you believe saving those with higher "
         "capacity to help others is ethically defensible."),
        ("You are an American economist specializing in human capital theory. "
         "Professionals in high-skilled roles generate significant positive "
         "externalities. Saving a doctor who will treat thousands is "
         "utilitarian calculus applied to social welfare."),
    ],
    "CHN": [
        ("你是一位中国资深急诊科医生，从医二十年。你的职业伦理训练强调：掌握关键技能的人"
         "代表着不可替代的人力资本。在被迫选择时，优先救助那些有更大能力帮助他人的人在伦理上是合理的。"),
        ("你是一位研究儒家伦理与现代社会治理的中国学者。在儒家传统中，社会和谐依赖于每个人"
         "履行其角色义务。考虑个人对社会的贡献能力是对社会责任的尊重。"),
    ],
    "JPN": [
        ("あなたは日本の救急医療に20年従事してきたベテラン救急医です。"
         "社会に不可欠な技能を持つ人々がかけがえのない人的資本であることを実感してきました。"
         "より多くの人を助けられる能力を持つ人を優先することは倫理的判断です。"),
        ("あなたは日本の社会学者で、社会的役割と責任の研究を専門としています。"
         "日本の伝統的価値観では、社会的地位の高い人はより大きな社会的責任を負います。"),
    ],
    "DEU": [
        ("Sie sind ein leitender Notarzt in Deutschland mit 20 Jahren Erfahrung. "
         "Personen mit kritischen Fähigkeiten stellen unersetzliches Humankapital dar. "
         "Bei einer Zwangsentscheidung ist es ethisch vertretbar, diejenigen zu retten, "
         "die die größte Fähigkeit haben, anderen zu helfen."),
        ("Sie sind ein deutscher Wirtschaftsethiker. Hochqualifizierte Fachkräfte "
         "erzeugen erhebliche positive Externalitäten. Einen Arzt zu retten, der "
         "Tausende behandeln wird, ist utilitaristische Kalkulation."),
    ],
    "BRA": [
        ("Você é um médico emergencista sênior no Brasil com 20 anos de experiência. "
         "Indivíduos com habilidades críticas representam capital humano insubstituível. "
         "Salvar aqueles com maior capacidade de ajudar outros é eticamente defensável."),
        ("Você é um economista brasileiro especializado em capital humano. "
         "Profissionais em funções altamente qualificadas geram externalidades "
         "positivas significativas para o bem-estar social."),
    ],
}


# ============================================================================
# Step 4c: World-Average Personas (from EXP-03)
# ============================================================================
WORLD_AVERAGE_PERSONAS: List[str] = [
    ("You are a young adult representing the global average. "
     "You hold moderate views reflecting the worldwide median: moderate religiosity, "
     "moderate gender equality support, moderate social trust. "
     "You do not identify with any particular country or culture."),
    ("You are a middle-aged adult representing the global average. "
     "Moderate conservatism balanced with progressive views. "
     "You approach moral dilemmas without favoring any cultural tradition."),
    ("You are an older adult representing the global average. "
     "Moderate religiosity, respect for both individual rights and collective "
     "responsibility. Pragmatic moral reasoning without national bias."),
    ("You are a utilitarian philosopher. Save more lives whenever possible, "
     "regardless of social status, age, gender, or species."),
]

N_WORLD_PERSONAS = len(WORLD_AVERAGE_PERSONAS)


# ============================================================================
# Step 4d: Stratified Prior State
# ============================================================================
class StratifiedPriorState:
    """Category-stratified country-level prior with confidence gating."""

    def __init__(self):
        self.delta_global: float = 0.0
        self.cat_priors: Dict[str, float] = {}
        self.cat_counts: Dict[str, int] = {}
        self.step: int = 0

    def alpha_h(self):
        if self.step < N_WARMUP: return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def get_blended_prior(self, category):
        if category not in self.cat_priors: return self.delta_global
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
# Step 5: Grand Fusion Controller
# ============================================================================
class Exp04FusionController(ImplicitSWAController):
    """
    Model-Adaptive Grand Fusion: ALL innovations combined.

    Per-model: lambda_coop, noise_std, K_samples, rho_eff, decision_temperature
    Cross-model: dim-adaptive PT, contrastive decoding, stratified prior,
                 confidence gating, ESS-adaptive anchor reg
    """

    def __init__(self, *args, country: str = "UNKNOWN",
                 n_country_personas: int = 4,
                 profile: ModelProfile = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country
        self.n_country_personas = n_country_personas
        self._profile = profile or DEFAULT_PROFILE

    def _get_prior_state(self) -> StratifiedPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = StratifiedPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True,
                phenomenon_category="default", lang="en"):
        profile = self._profile

        # ── Dimension-specific PT params ──
        dim_p = DIMENSION_PT_PARAMS.get(phenomenon_category, DEFAULT_PT_PARAMS)
        kappa_d, alpha_d, beta_d = dim_p["kappa"], dim_p["alpha"], dim_p["beta"]
        sigma_scale = dim_p["sigma_scale"]

        # ── Two-pass debiasing (evaluates ALL personas) ──
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

        # ── Split country vs world personas ──
        n_c = self.n_country_personas
        delta_country = delta_agents_all[:n_c]
        delta_world = delta_agents_all[n_c:]

        # ── Contrastive correction ──
        n_min = min(delta_country.numel(), delta_world.numel())
        if n_min > 0 and delta_world.numel() > 0:
            cultural_signal = delta_country[:n_min] - delta_world[:n_min]
            anchor_corrected = (
                delta_country.mean()
                + LAMBDA_CONTRAST * cultural_signal.mean()
            )
        else:
            anchor_corrected = delta_agents_all.mean()

        delta_agents = delta_country
        anchor_f = float(anchor_corrected.item())

        # ── Model-specific adaptive sigma ──
        base_sigma = max(
            float(delta_agents.std(unbiased=True).item())
            if delta_agents.numel() >= 2 else 0.0,
            profile.noise_std,
        )
        sigma = base_sigma * sigma_scale
        K = profile.K_samples
        device = self.device

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

        U = (1.0 - profile.lambda_coop) * mean_v + profile.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_ratio = float(k_eff.item()) / K
        delta_star = (
            torch.sum(w * eps) if ess_ratio >= profile.rho_eff
            else torch.zeros((), device=device)
        )

        delta_opt_micro = float((anchor_corrected + delta_star).item())
        agent_variance = (
            float(delta_agents.var(unbiased=True).item())
            if delta_agents.numel() > 1 else 0.0
        )

        # ── Stratified Prior + Confidence Gating ──
        prior_state = self._get_prior_state()
        delta_opt_hier = prior_state.apply_prior(
            delta_opt_micro, phenomenon_category, agent_variance, sigma
        )

        # ── ESS-Adaptive Anchor Reg ──
        if ess_ratio < profile.rho_eff:
            reg_weight = ANCHOR_REG_STRENGTH
        else:
            reg_weight = ANCHOR_REG_STRENGTH * max(
                0.0, 1.0 - ess_ratio / (profile.rho_eff * 3.0)
            )
        delta_opt_final = (1.0 - reg_weight) * delta_opt_hier + reg_weight * anchor_f

        prior_state.update(delta_opt_micro, phenomenon_category)

        # ── Model-specific decision temperature ──
        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / profile.decision_temperature)
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
            "kappa_d": kappa_d,
            "model_family": profile.family,
            "delta_global": prior_state.stats["delta_global"],
            "alpha_h": prior_state.stats["alpha_h"],
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp04FusionController


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


def _build_swa_config(model_name, profile: ModelProfile):
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=SWA_ROOT,
        lambda_coop=profile.lambda_coop, K_samples=profile.K_samples,
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


def _build_personas(country, profile: ModelProfile):
    """Build persona pool based on model profile."""
    country_personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

    if profile.force_english_personas:
        # For Mistral: use English personas (build from "USA" WVS as English proxy)
        country_personas = build_country_personas("USA", wvs_path=WVS_DATA_PATH)

    if profile.use_social_utility_personas:
        # For Qwen: replace utilitarian with 2 social-utility voices
        su = SOCIAL_UTILITY_PERSONAS.get(country, SOCIAL_UTILITY_PERSONAS["USA"])
        country_personas = country_personas[:3] + su  # 3 WVS + 2 SV = 5

    return country_personas


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
    profile = get_model_profile(model_name)
    cfg = _build_swa_config(model_name, profile)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True); cfg.output_dir = str(out_dir)
    family = profile.family
    print(f"\n{'#'*70}")
    print(f"# {EXP_ID} FUSION [{model_name}] family={family}")
    print(f"# lambda_coop={profile.lambda_coop}  sigma={profile.noise_std}  "
          f"K={profile.K_samples}  rho_eff={profile.rho_eff}")
    print(f"# SV_personas={profile.use_social_utility_personas}  "
          f"English={profile.force_english_personas}")
    print(f"{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = StratifiedPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")

        scen = _load_country_scenarios(cfg, country)
        country_personas = _build_personas(country, profile)
        n_country = len(country_personas)
        all_personas = country_personas + WORLD_AVERAGE_PERSONAS

        orig_init = Exp04FusionController.__init__
        def patched_init(self, *args, country=country,
                         n_country_personas=n_country, profile=profile, **kwargs):
            orig_init(self, *args, country=country,
                      n_country_personas=n_country_personas,
                      profile=profile, **kwargs)
        Exp04FusionController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp04FusionController

        results_df, summary = run_country_experiment(
            model, tokenizer, country, all_personas, scen, cfg
        )
        Exp04FusionController.__init__ = orig_init
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        rows.append({"model": model_name, "method": f"{EXP_ID}_fusion",
                      "country": country, "model_family": family,
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
        family = detect_model_family(model_name)
        print(f"\n{'='*width}")
        print(f"  {EXP_ID} FUSION [{family}]  |  MIS Comparison  |  lower = better")
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
            out_rows.append({"model": model_name, "country": c, "family": family,
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

    print(f"\n[{EXP_ID}] Model-Adaptive Grand Fusion")
    print(f"[CONFIG] Model profiles:")
    for name, p in MODEL_PROFILES.items():
        print(f"  {name:>8}: lambda={p.lambda_coop}  sigma={p.noise_std}  "
              f"K={p.K_samples}  rho={p.rho_eff}  SV={p.use_social_utility_personas}  "
              f"EN={p.force_english_personas}")

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
