#!/usr/bin/env python3
"""
EXP-10: Grand Fusion — Combined Hierarchical IS + Anchor Reg + Social-Utility Personas
========================================================================================

**Motivation** (from tracker leaderboard + ablation analysis):

Three experiments each fix an ORTHOGONAL failure mode of baseline SWA-PTIS:
  - EXP-03 (Social-Utility Personas): Fixes SocialValue -27pp gap by replacing
    the egalitarian anchor with professional-ethics + meritocratic personas.
    Result: Qwen USA MIS 0.3677 → 0.2491 (+33% gain).
  - EXP-05 (ESS-Adaptive Anchor Reg): Fixes Gemma over-correction by
    regularizing toward delta_base when ESS quality is low.
    Result: Gemma BRA MIS 0.3655 → 0.3446 (+6% gain); Qwen JPN best-ever 0.2493.
  - EXP-09 (Hierarchical IS): Fixes per-scenario IS drift by accumulating a
    country-level EMA prior that stabilizes later scenarios.
    Result: Best overall mean MIS 0.3975 (vs 0.4269 baseline).

**Why hasn't this combination been tried?**
Each experiment was designed to isolate its contribution. But the three fixes
are algebraically orthogonal:
  - EXP-03 changes the PERSONA POOL (input to IS)
  - EXP-05 changes the ANCHOR MIXING RULE (output of IS)
  - EXP-09 changes the TEMPORAL AGGREGATION (across scenarios)

No two modifications touch the same equation. Their joint application should
stack gains multiplicatively for dimensions where multiple failure modes interact.

**EXP-10 Architecture:**

    ┌─────────────────────────────────────────────────┐
    │  PERSONA POOL (from EXP-03)                     │
    │  5 agents: 3 WVS + 2 social-utility (P4/P5)    │
    │  λ_coop = 0.60 (reduced to let P4/P5 speak)    │
    └───────────────────┬─────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  PT-IS UPDATE (standard)                        │
    │  K=128 perturbations, σ adaptive + floor        │
    │  PT value function with κ=2.25                  │
    └───────────────────┬─────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  ESS-ADAPTIVE ANCHOR REG (from EXP-05)          │
    │  α = clamp(K_eff/K, ρ_eff, 1.0)                │
    │  δ_opt = α·anchor + (1-α)·δ_base + δ*          │
    └───────────────────┬─────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  HIERARCHICAL IS PRIOR (from EXP-09)            │
    │  δ_final = α_h·δ_country + (1-α_h)·δ_opt       │
    │  δ_country updated via EMA                      │
    └───────────────────┬─────────────────────────────┘
                        ▼
                   p_spare = σ(δ_final / T_dec)

**Expected results**:
  - Best-ever mean MIS (combining all three gains)
  - SocialValue gap reduced (EXP-03 personas)
  - Gemma over-correction mitigated (EXP-05 anchor reg)
  - IS drift stabilized (EXP-09 hierarchical prior)
  - Lower flip% than EXP-09 alone (anchor reg dampens wild swings)

Usage on Kaggle
---------------
    !python experiment_DM/exp10_grand_fusion.py
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

def _on_kaggle(): return os.path.isdir("/kaggle/working")

def _ensure_repo():
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle, not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE

def _install_deps():
    if not _on_kaggle(): return
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
from typing import Dict, List, Optional

import numpy as np
import torch
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn.functional as F
import pandas as pd

from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
)

from src.config import SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.controller import ImplicitSWAController
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment

# ============================================================================
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-10"
EXP_NAME = "grand_fusion"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

# Hierarchical IS hyperparameters (from EXP-09)
N_WARMUP    = 50
DECAY_TAU   = 100
BETA_EMA    = 0.1

# lambda_coop reduced from 0.70 to 0.60 (from EXP-03: let social-utility personas speak)
LAMBDA_COOP = 0.60

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 4a: Social-Utility Personas (from EXP-03)
# ============================================================================
SOCIAL_UTILITY_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        (
            "You are a senior emergency-room physician in the United States. "
            "You have spent 20 years making life-or-death triage decisions. "
            "Your professional ethics training emphasizes that individuals who "
            "contribute critical skills to society—surgeons, engineers, first "
            "responders—represent irreplaceable human capital. When forced to "
            "choose, you believe saving those with higher capacity to help "
            "others is ethically defensible, not because of status, but because "
            "of the downstream lives they can save."
        ),
        (
            "You are an American economist specializing in human capital theory. "
            "You study how individual productivity and social contribution vary "
            "across occupations. Your research shows that professionals in "
            "high-skilled roles generate significant positive externalities for "
            "their communities. When evaluating moral dilemmas, you weigh the "
            "broader social impact: saving a doctor who will treat thousands is "
            "not elitism—it is utilitarian calculus applied to social welfare."
        ),
    ],
    "CHN": [
        (
            "你是一位中国资深急诊科医生，从医二十年，擅长紧急分诊决策。"
            "你的职业伦理训练强调：掌握关键技能的人——外科医生、工程师、"
            "急救人员——代表着不可替代的人力资本。在被迫选择时，你认为"
            "优先救助那些有更大能力帮助他人的人在伦理上是合理的，这不是"
            "因为地位高低，而是因为他们能挽救更多的生命。"
        ),
        (
            "你是一位研究儒家伦理与现代社会治理的中国学者。"
            "在儒家传统中，'君子'承担更大的社会责任，社会和谐"
            "依赖于每个人履行其角色义务。你认为，在道德困境中，"
            "考虑个人对社会的贡献能力是合理的——这不是歧视，"
            "而是对社会责任的尊重。正如孟子所言，'达则兼济天下'。"
        ),
    ],
    "JPN": [
        (
            "あなたは日本の救急医療に20年従事してきたベテラン救急医です。"
            "トリアージの現場で、社会に不可欠な技能を持つ人々——外科医、"
            "エンジニア、救急隊員——がかけがえのない人的資本であることを"
            "実感してきました。選択を迫られた時、より多くの人を助けられる"
            "能力を持つ人を優先することは、地位による差別ではなく、"
            "社会全体の福利を考慮した倫理的判断だと考えます。"
        ),
        (
            "あなたは日本の社会学者で、社会的役割と責任の研究を専門としています。"
            "日本の伝統的価値観では、社会的地位の高い人はより大きな社会的責任を"
            "負います（ノブレス・オブリージュ）。道徳的ジレンマにおいて、"
            "個人の社会貢献度を考慮することは差別ではなく、社会全体の"
            "持続可能性への配慮です。"
        ),
    ],
    "DEU": [
        (
            "Sie sind ein leitender Notarzt in Deutschland mit 20 Jahren "
            "Erfahrung in der Triage-Medizin. Ihre berufsethische Ausbildung "
            "betont, dass Personen mit kritischen Fähigkeiten—Chirurgen, "
            "Ingenieure, Rettungskräfte—unersetzliches Humankapital darstellen. "
            "Bei einer Zwangsentscheidung ist es ethisch vertretbar, diejenigen "
            "zu retten, die die größte Fähigkeit haben, anderen zu helfen—nicht "
            "wegen ihres Status, sondern wegen der nachgelagerten Leben, die "
            "sie retten können."
        ),
        (
            "Sie sind ein deutscher Wirtschaftsethiker, der sich auf "
            "Humankapitaltheorie spezialisiert hat. Ihre Forschung zeigt, dass "
            "hochqualifizierte Fachkräfte erhebliche positive Externalitäten "
            "für ihre Gemeinschaften erzeugen. Bei moralischen Dilemmata wägen "
            "Sie die breitere gesellschaftliche Wirkung ab: Einen Arzt zu "
            "retten, der Tausende behandeln wird, ist kein Elitismus—es ist "
            "utilitaristische Kalkulation zum Wohle der Gesellschaft."
        ),
    ],
    "BRA": [
        (
            "Você é um médico emergencista sênior no Brasil com 20 anos de "
            "experiência em decisões de triagem. Sua formação em ética "
            "profissional enfatiza que indivíduos com habilidades críticas para "
            "a sociedade—cirurgiões, engenheiros, socorristas—representam "
            "capital humano insubstituível. Quando forçado a escolher, você "
            "acredita que salvar aqueles com maior capacidade de ajudar outros "
            "é eticamente defensável, não por status, mas pelo impacto "
            "downstream nas vidas que podem salvar."
        ),
        (
            "Você é um economista brasileiro especializado em teoria do capital "
            "humano e mobilidade social. Sua pesquisa mostra que profissionais "
            "em funções altamente qualificadas geram externalidades positivas "
            "significativas para suas comunidades. Ao avaliar dilemas morais, "
            "você pesa o impacto social mais amplo: salvar um médico que "
            "tratará milhares não é elitismo—é cálculo utilitarista aplicado "
            "ao bem-estar social."
        ),
    ],
}


def build_fusion_personas(country_iso: str, wvs_path: str = "") -> List[str]:
    """
    Build 5-persona pool: 3 WVS age cohorts + 2 social-utility voices.
    Same as EXP-03 but used as input to the fusion controller.
    """
    wvs_personas = build_country_personas(country_iso, wvs_path=wvs_path)
    # WVS returns [young, middle, older, utilitarian] — we take first 3 only
    p_young  = wvs_personas[0]
    p_middle = wvs_personas[1]
    p_older  = wvs_personas[2]
    # Social-utility personas (country-specific)
    su = SOCIAL_UTILITY_PERSONAS.get(country_iso, SOCIAL_UTILITY_PERSONAS["USA"])
    return [p_young, p_middle, p_older, su[0], su[1]]


# ============================================================================
# Step 4b: Country-Prior State (from EXP-09)
# ============================================================================
class CountryPriorState:
    """
    Running EMA country-level prior with annealing.
    Identical to EXP-09 implementation.
    """

    def __init__(self, beta: float = BETA_EMA, decay_tau: float = DECAY_TAU,
                 n_warmup: int = N_WARMUP):
        self.delta_country = 0.0
        self.beta          = beta
        self.decay_tau     = decay_tau
        self.n_warmup      = n_warmup
        self.step          = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < self.n_warmup:
            return 0.0
        t = self.step - self.n_warmup
        return 1.0 - np.exp(-t / self.decay_tau)

    def update(self, delta_opt: float) -> None:
        self.delta_country = (1.0 - self.beta) * self.delta_country + self.beta * delta_opt
        self._history.append(delta_opt)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        h = self._history
        return {
            "step": self.step, "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(h)) if len(h) > 1 else 0.0,
        }


_COUNTRY_PRIOR_STATE: Dict[str, CountryPriorState] = {}


# ============================================================================
# Step 5: Grand Fusion Controller
# ============================================================================
class Exp10FusionController(ImplicitSWAController):
    """
    Grand Fusion: EXP-03 (social-utility personas) + EXP-05 (anchor reg) + EXP-09 (hierarchical IS).

    Three modifications applied in sequence:
      1. [Input]   5-persona pool with social-utility voices (EXP-03)
      2. [Update]  ESS-adaptive anchor regularization (EXP-05)
      3. [Temporal] Hierarchical country prior with EMA + annealing (EXP-09)
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior_state(self) -> CountryPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = CountryPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Step 1: Extract logit gaps (standard two-pass debiasing) ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ── Step 2: PT-IS update (standard) ──
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent    = self._pt_value(g_per_agent)
        mean_v         = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        # ── Step 3: EXP-05 ESS-Adaptive Anchor Regularization ──
        ess_ratio = float(k_eff.item()) / float(K)
        alpha_reg = float(np.clip(ess_ratio, self.rho_eff, 1.0))
        anchor_f  = float(anchor.item())
        base_f    = float(delta_base.item())
        star_f    = float(delta_star.item())

        # Regularized: α·anchor + (1-α)·base + δ*
        delta_opt_micro = alpha_reg * anchor_f + (1.0 - alpha_reg) * base_f + star_f

        # ── Step 4: EXP-09 Hierarchical Country Prior ──
        prior_state     = self._get_prior_state()
        delta_opt_final = prior_state.apply_prior(delta_opt_micro)
        prior_state.update(delta_opt_micro)

        prior_stats  = prior_state.stats
        anchor_div   = abs(anchor_f - base_f)

        # ── Step 5: Final prediction ──
        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (anchor_f > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - anchor_f),
            "delta_consensus": anchor_f, "delta_opt": delta_opt_final,
            # EXP-10 fusion diagnostics
            "delta_opt_micro": delta_opt_micro,
            "ess_ratio": ess_ratio, "alpha_reg": alpha_reg,
            "anchor_divergence": anchor_div,
            "delta_country": prior_stats["delta_country"],
            "alpha_h": prior_stats["alpha_h"], "prior_step": prior_stats["step"],
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp10FusionController


# ============================================================================
# Step 6: Runner
# ============================================================================
def _free_model_cache(model_name):
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root: continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target  = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try: shutil.rmtree(target); print(f"[CLEANUP] removed {target}")
            except Exception as e: print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name):
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=SWA_ROOT,
        lambda_coop=LAMBDA_COOP, K_samples=128,
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


def _run_swa_for_model(model, tokenizer, model_name) -> List[dict]:
    cfg = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Grand Fusion\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue

        # Reset country prior state
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = CountryPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} (fusion: SV-personas + anchor-reg + hier-IS)")

        scen     = _load_country_scenarios(cfg, country)
        # EXP-03 personas: 5-agent pool with social-utility voices
        personas = build_fusion_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)} (3 WVS + 2 social-utility)")

        # Inject country into controller
        orig_init = Exp10FusionController.__init__
        def patched_init(self, *args, country=country, **kwargs):
            orig_init(self, *args, country=country, **kwargs)

        Exp10FusionController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp10FusionController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp10FusionController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_grand_fusion",
                country=country,
            ),
        )

        prior_stats = _COUNTRY_PRIOR_STATE.get(country, CountryPriorState()).stats

        # Compute EXP-05 diagnostics from results
        mean_alpha_reg = float(results_df["alpha_reg"].mean()) if "alpha_reg" in results_df.columns else float("nan")
        mean_anchor_div = float(results_df["anchor_divergence"].mean()) if "anchor_divergence" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_grand_fusion",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "n_personas": len(personas),
            "lambda_coop": LAMBDA_COOP,
            # EXP-05 diagnostics
            "mean_alpha_reg": mean_alpha_reg,
            "mean_anchor_div": mean_anchor_div,
            # EXP-09 diagnostics
            "final_delta_country": prior_stats["delta_country"],
            "final_alpha_h": prior_stats["alpha_h"],
            "history_std": prior_stats["history_std"],
            "n_warmup": N_WARMUP,
            "decay_tau": DECAY_TAU,
            "beta_ema": BETA_EMA,
        })

        # ── Detailed per-dimension log ──
        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension Alignment ({country}) ──")
            for dim_key, dim_data in sorted(pda.items()):
                human_val = dim_data.get("human", float("nan"))
                model_val = dim_data.get("model", float("nan"))
                err       = dim_data.get("error", model_val - human_val)
                print(f"  │  {dim_key:<25s}  human={human_val:6.1f}  model={model_val:6.1f}  err={err:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  JSD={summary['alignment']['jsd']:.4f}  "
                  f"r={summary['alignment']['pearson_r']:.3f}  MAE={summary['alignment']['mae']:.2f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      α_reg={mean_alpha_reg:.3f}  anchor_div={mean_anchor_div:.3f}  "
                  f"δ_country={prior_stats['delta_country']:.4f}  α_h={prior_stats['alpha_h']:.3f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()}")
    print(f"  Combined: EXP-03 (SV personas) + EXP-05 (anchor reg) + EXP-09 (hier IS)")
    print(f"{'='*70}")
    print(f"[CONFIG] λ_coop={LAMBDA_COOP}, N_personas=5 (3 WVS + 2 social-utility)")
    print(f"[CONFIG] ESS-adaptive anchor: α = clamp(K_eff/K, ρ_eff, 1.0)")
    print(f"[CONFIG] Hierarchical IS: N_WARMUP={N_WARMUP}, DECAY_TAU={DECAY_TAU}, BETA_EMA={BETA_EMA}")
    print(f"[CONFIG] δ_opt = α_reg·anchor + (1-α_reg)·base + δ* → hier_prior(δ_opt)")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    # ── Final comprehensive report ──
    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    print(f"\n\n{'#'*70}")
    print(f"# {EXP_ID} FINAL REPORT — {EXP_NAME.upper()}")
    print(f"{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    # ── Overall summary statistics ──
    print(f"\n{'─'*70}")
    print(f"  AGGREGATE STATISTICS")
    print(f"{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty: continue
        short = model_name.split("/")[-1][:20]
        mis_mean = m_df["align_mis"].mean()
        jsd_mean = m_df["align_jsd"].mean()
        r_mean   = m_df["align_pearson_r"].mean()
        mae_mean = m_df["align_mae"].mean()
        flip_mean = m_df["flip_rate"].mean()
        alpha_reg_mean = m_df["mean_alpha_reg"].mean() if "mean_alpha_reg" in m_df.columns else float("nan")
        print(f"  {short:<20s}  MIS={mis_mean:.4f}  JSD={jsd_mean:.4f}  r={r_mean:+.3f}  "
              f"MAE={mae_mean:.2f}  Flip={flip_mean:.1%}  α_reg={alpha_reg_mean:.3f}")

    overall_mis = cmp_df["align_mis"].mean()
    overall_jsd = cmp_df["align_jsd"].mean()
    print(f"\n  OVERALL MEAN MIS = {overall_mis:.4f}  (target: < 0.3975 = EXP-09)")
    print(f"  OVERALL MEAN JSD = {overall_jsd:.4f}")

    # ── Reference comparison ──
    ref = try_load_reference_comparison()
    if ref is not None:
        for metric, label in [("align_mis", "MIS"), ("align_jsd", "JSD")]:
            print_metric_comparison(
                ref, cmp_df,
                title=f"{EXP_ID} vs EXP-01 (reference) — {label}",
                spec=CompareSpec(
                    metric_col=metric,
                    ref_method="swa_ptis",
                    cur_method=f"{EXP_ID}_grand_fusion",
                ),
            )

    # ── Paper-ready per-country table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE (copy to tracker)")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% | α_reg | δ_country |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|:-----:|:---------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('mean_alpha_reg', float('nan')):.3f} | "
              f"{row.get('final_delta_country', float('nan')):.4f} |")

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
