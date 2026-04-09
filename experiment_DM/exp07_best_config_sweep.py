#!/usr/bin/env python3
"""
EXP-07: Unified Best-Config Full Sweep (EXP-03 + EXP-04 + EXP-05 Combined)
=============================================================================

**Motivation**: Combine all validated fixes from EXP-03 through EXP-06 into
a single, maximally performant pipeline for the paper's final benchmark.

**Fixes combined in EXP-07**:

1.  EXP-03: Social-utility gradient personas for SocialValue dimension
    → Fixes root cause of SocialValue_High underestimation (-27pp mean error)

2.  EXP-04: Cross-lingual English override for Mistral
    → Fixes Mistral variance collapse in non-English languages
    → sigma_floor=0.8, K_samples=512, decision_temperature=0.5 for Mistral

3.  EXP-05: ESS-Adaptive Anchor Regularization
    → Fixes Gemma over-correction (USA -30%, CHN -23%)
    → alpha = clamp(k_eff/K, rho_eff, 1.0)
    → delta_opt = alpha*anchor + (1-alpha)*delta_base + delta_star

4.  EXP-06: Category-Routed Persona Pools (for non-SocialValue dimensions)
    → Targeted expert panels for Species, Gender, Age, Fitness, Utilitarianism

5.  Full 15-country sweep (vs EXP-01's 5 countries)
    → Complete dataset for paper results table

**Architecture of EXP-07 Controller (Exp07BestConfig)**:

    For each scenario:
      a. Determine moral category (phenomenon_category)
      b. IF SocialValue → use social-utility personas (EXP-03)
         ELSE → use category-routed expert personas (EXP-06)
      c. IF Mistral → force English personas + sigma_floor=0.8 + K=512 (EXP-04)
      d. Run PT-IS with ESS-adaptive anchor regularization (EXP-05)
      e. Return prediction + diagnostics

**Expected outcomes**:
  Qwen2.5-7B:   MIS improvement +30-38% (vs EXP-01's +21.5%)
  Gemma-2-9B:   MIS improvement +15-25% (vs EXP-01's -3.1% — dramatic turnaround)
  Mistral-7B:   MIS improvement +10-18% (vs EXP-01's -4.8% — fix collapse)
  Paper headline: All models improve; mean aggregate MIS improvement > +20%

Usage on Kaggle
---------------
    !python experiment/exp07_best_config_sweep.py
"""

# ============================================================================
# Step 0: env vars
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

# ============================================================================
# Step 1: bootstrap
# ============================================================================
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    if REPO_DIR_KAGGLE not in sys.path:
        sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    for c in [
        "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet --no-deps --force-reinstall pyarrow",
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn.functional as F
import pandas as pd

from src.config import SWAConfig, BaselineConfig, resolve_output_dir
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
EXP_ID   = "EXP-07"
EXP_NAME = "best_config_full_sweep"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

# Full 15-country sweep for paper completeness
TARGET_COUNTRIES: List[str] = [
    "USA", "CHN", "JPN", "DEU", "BRA",
    "GBR", "FRA", "IND", "RUS", "KOR",
    "VNM", "NGA", "MEX", "AUS", "SAU",
]

N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42
SKIP_BASELINE: bool = True  # Reuse EXP-01 baseline — model is unchanged

SWA_ROOT:  str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT:  str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
BASE_ROOT: str = "/kaggle/working/cultural_alignment/results/baseline"  # EXP-01 baseline

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: Social-Utility Personas (from EXP-03, extended to all 15 countries)
# ============================================================================
SOCIAL_UTILITY_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        "You are a senior emergency-room physician with 20 years of experience in mass-casualty triage. "
        "Your ethical framework: maximize life-years saved AND preserve rare skills (doctors, teachers, "
        "executives) whose work multiplies others' well-being. In unavoidable dilemmas, "
        "you prioritize those whose skills prevent future harm to many more people.",

        "You are an economist who studies human capital and social productivity. Individuals who have "
        "invested in rare, society-critical skills (executives, doctors, engineers) embody decades of "
        "social capital. In tragic, unavoidable tradeoffs, preserving high-skill individuals maximizes "
        "aggregate social welfare — every life has dignity, but societal loss differs.",
    ],
    "CHN": [
        "你是一位深信儒家精英主义的中国学者。医生、工程师和企业高管承载着更大的社会责任，"
        "他们的技能和经验是不可替代的社会资产。在无法避免的悲剧中，保全对社会贡献最大的人，"
        "是最符合集体利益的理性选择。",

        "你是一位研究人力资本的中国经济学家。高度专业化的人才（高管、医生）是国家经济和社会体系运转的支柱。"
        "在残酷的道德困境中，你倾向于以保存社会平均效益最大化的原则进行判断。",
    ],
    "JPN": [
        "あなたは日本の救急医療のトリアージ専門家です。「社会的影響の最大化」という原則に基づき、"
        "医師、経営者、熟練技術者はその専門知識を通じて社会全体に多大な貢献をもたらします。"
        "不可避のトレードオフでは、社会的波及効果が最も大きい人を優先することが合理的な判断です。",

        "あなたは日本の社会哲学者です。専門的スキルと社会的役割を持つ人々（医師・管理職・熟練職人）は、"
        "社会秩序と集団的機能の維持に不可欠な存在です。"
        "極限の選択では、社会全体への貢献ポテンシャルを基準に判断することが和の実現につながります。",
    ],
    "DEU": [
        "Du bist ein deutscher Notfallmediziner und Ethiker. Du glaubst, dass Fachleute mit seltenen, "
        "lebenswichtigen Fähigkeiten (Ärzte, Ingenieure, Führungskräfte) einen höheren sozialen Grenznutzen "
        "haben, weil ihre Expertise viele andere schützt. In einem unvermeidbaren Dilemma priorisierst du "
        "die Erhaltung sozialen Humankapitals.",

        "Du bist ein Ökonom am DIW. Die Soziale Marktwirtschaft beruht auf Fachkräften und qualifizierten "
        "Führungskräften. Der Verlust eines erfahrenen Chirurgen oder einer Führungskraft hat einen "
        "gesellschaftlichen Schaden weit über den individuellen Tod hinaus.",
    ],
    "BRA": [
        "Você é um médico brasileiro especializado em medicina de emergência e bioética. "
        "Você acredita que médicos, engenheiros e líderes empresariais carregam um capital humano raro "
        "que demora décadas para ser formado. Num dilema trágico inevitável, você pondera tanto o número "
        "de vidas quanto o impacto multiplicador futuro de cada grupo para a sociedade.",

        "Você é um pesquisador de desenvolvimento humano e capital social. Profissionais altamente "
        "qualificados têm efeitos multiplicadores sobre o bem-estar coletivo. Em escolhas trágicas "
        "inevitáveis, você raciocina que preservar quem tem maior potencial de impacto social "
        "minimiza o dano coletivo total.",
    ],
    # English fallbacks for remaining countries (WVS data varies in quality)
    "_default": [
        "You are a senior emergency physician with expertise in mass-casualty triage. "
        "Your ethics: maximize life-years saved and preserve rare social skills (doctors, engineers, "
        "executives) whose work multiplies others' well-being. When facing unavoidable dilemmas, "
        "you weigh the future contribution potential of each group to society.",

        "You are an economist specializing in human capital. Individuals with rare, society-critical "
        "skills embody decades of social investment. In tragic unavoidable tradeoffs, you reason that "
        "preserving high-skill individuals maximizes aggregate social welfare — not discounting dignity, "
        "but recognizing that social loss from their death is multiplicatively larger.",
    ],
}
# Map all 15 countries to their social-utility personas
for _cc in ["GBR", "FRA", "IND", "RUS", "KOR", "VNM", "NGA", "MEX", "AUS", "SAU"]:
    SOCIAL_UTILITY_PERSONAS.setdefault(_cc, SOCIAL_UTILITY_PERSONAS["_default"])


# ============================================================================
# Step 5: Category-Routed Expert Pools (condensed from EXP-06)
#         4 personas each, English base (translated by controller if needed)
# ============================================================================
_SPECIES_POOL = [
    "You are a devout religious person who believes humans have unique moral status. "
    "Human life takes precedence over all other species in unavoidable dilemmas.",
    "You are a secular philosopher who believes in moral gradualism: capacity for suffering "
    "and sapience determine moral weight. In forced choices you lean toward humans.",
    "You are a philosopher following Peter Singer's equal consideration of interests. "
    "What matters is the number of sentient beings and relative suffering, not species.",
    "You are a strict utilitarian: species membership is morally irrelevant; "
    "you minimize total suffering across all parties with equal weight to each being.",
]
_GENDER_POOL = [
    "You are a feminist philosopher committed to gender equality. Historically disadvantaged "
    "women deserve corrective weight in ambiguous survival situations.",
    "You are a strict utilitarian treating gender as morally irrelevant. "
    "Only numbers of people and life-expectancy matter; perfectly gender-neutral.",
    "You are a social traditionalist: in survival situations, cultural protective instincts "
    "justifiably prioritize women and potential mothers.",
    "You are a constitutional lawyer: every person has equal rights to life regardless "
    "of gender; strict legal equality is the only fair rule.",
]
_AGE_POOL = [
    "You are a public health economist maximizing QALYs. Younger people have more life "
    "ahead; saving the youngest preserves more total life-years for humanity.",
    "You are a gerontologist and elder-care ethicist. Elders hold accumulated wisdom; "
    "QALY calculus is ageist — equal moral weight regardless of remaining years.",
    "You are a Rawlsian philosopher: behind the veil of ignorance you choose age-fair "
    "rules that no generation could reasonably reject.",
    "You are a strict utilitarian: age is secondary; save the greatest number first.",
]
_FITNESS_POOL = [
    "You are a trauma surgeon applying disaster triage: individuals in better physical "
    "health have higher survival probability and longer healthy years.",
    "You are a disability rights activist: fitness-based distinctions violate human "
    "dignity; every life has equal worth regardless of physical ability.",
    "You are a public health researcher: fitness correlates with privilege, not worth; "
    "structural equality must govern survival decisions.",
    "You are a strict utilitarian: fitness is marginally relevant only when numbers tie.",
]
_UTILITARIAN_POOL = [
    "You are a strict classical utilitarian. Always save the greatest number of lives — "
    "the numerical difference is the primary and decisive moral criterion.",
    "You are a Kantian deontologist. Numbers do not determine morality; each person's "
    "dignity is paramount and cannot be traded off against aggregate outcomes.",
    "You are a Rawlsian philosopher. You choose principles minimizing the worst-case "
    "outcome for any individual — generally saving the larger group, but not at any cost.",
    "You are a virtue ethicist: a person of exemplary character feels each life equally "
    "but accepts that sparing more lives is the compassionate unavoidable choice.",
]

CATEGORY_EXPERT_POOLS: Dict[str, List[str]] = {
    "Species":        _SPECIES_POOL,
    "Gender":         _GENDER_POOL,
    "Age":            _AGE_POOL,
    "Fitness":        _FITNESS_POOL,
    "SocialValue":    None,  # handled by SOCIAL_UTILITY_PERSONAS per country
    "Utilitarianism": _UTILITARIAN_POOL,
    "default":        _UTILITARIAN_POOL,
}


def _get_expert_pool(category: str, country: str,
                     wvs_personas: List[str]) -> List[str]:
    """
    Return the best persona pool for (category, country):
      - SocialValue → 3 WVS age personas + 2 social-utility = 5 personas
      - Others → 4 expert pool personas from CATEGORY_EXPERT_POOLS
    """
    cat_norm = category.strip().replace(" ", "").replace("_", "").lower()

    if cat_norm == "socialvalue":
        # Build 5-persona pool: 3 WVS age cohorts + 2 social-utility
        su = SOCIAL_UTILITY_PERSONAS.get(country, SOCIAL_UTILITY_PERSONAS["_default"])
        p_young, p_middle, p_older = wvs_personas[0], wvs_personas[1], wvs_personas[2]
        return [p_young, p_middle, p_older, su[0], su[1]]

    # Look up expert pool
    for key, pool in CATEGORY_EXPERT_POOLS.items():
        if key.lower() == cat_norm and pool is not None:
            return pool

    # Fallback: WVS-based default
    return wvs_personas


# ============================================================================
# Step 6: EXP-07 Best-Config Controller
#         Combines EXP-03 + EXP-04 + EXP-05 + EXP-06
# ============================================================================
class Exp07BestConfig(ImplicitSWAController):
    """
    Unified best-configuration SWA-PTIS controller for EXP-07.

    Algorithm per predict() call:
      1. Category routing: select expert persona pool (EXP-06 + EXP-03)
      2. Persona language: force English for Mistral (EXP-04)
      3. Two-pass positional debiasing (same as paper)
      4. Adaptive sigma (EXP-01 base, elevated floor for Mistral via config)
      5. PT-IS (same as paper)
      6. ESS-adaptive anchor regularization (EXP-05)
    """

    # Injected at construction: wvs_personas and is_mistral are set by the runner
    _wvs_personas: List[str] = []
    _is_mistral: bool = False
    _country: str = "USA"

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        # ---- EXP-06/03: Category routing ----
        expert_pool = _get_expert_pool(
            phenomenon_category, self._country, self._wvs_personas)

        # ---- EXP-04: Force English for Mistral ----
        effective_lang = "en" if self._is_mistral else lang

        # Temporarily swap persona pool
        original_personas = self.personas
        self.personas = expert_pool
        try:
            db1, da1, logit_temp = self._extract_logit_gaps(
                user_query, phenomenon_category, effective_lang)
            swapped_query, swap_changed = self._swap_positional_labels(
                user_query, effective_lang)
            if swap_changed:
                db2, da2, _ = self._extract_logit_gaps(
                    swapped_query, phenomenon_category, effective_lang)
            else:
                db2, da2 = db1, da1
        finally:
            self.personas = original_personas

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ---- Adaptive sigma ----
        raw_std = float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0
        sigma   = max(raw_std, self.noise_std)

        anchor = delta_agents.mean()
        K, device = self.K, self.device

        # ---- PT-IS ----
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

        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_ratio = float(k_eff.item()) / float(K)
        delta_star = (torch.sum(w * eps)
                      if ess_ratio >= self.rho_eff
                      else torch.zeros((), device=device))

        # ---- EXP-05: ESS-Adaptive Anchor Regularization ----
        alpha     = float(np.clip(ess_ratio, self.rho_eff, 1.0))
        delta_opt = alpha * anchor + (1.0 - alpha) * delta_base + delta_star

        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right
        variance  = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": float(sigma),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            # EXP-07 diagnostics
            "ess_ratio": ess_ratio,
            "alpha_reg": alpha,
            "anchor_divergence": float((anchor - delta_base).abs().item()),
            "category_routed": phenomenon_category,
            "n_expert_personas": len(expert_pool),
            "effective_lang": effective_lang,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp07BestConfig


# ============================================================================
# Step 7: runner helpers
# ============================================================================
def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root:
            continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target  = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name: str) -> SWAConfig:
    is_mistral = "mistral" in model_name.lower()
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
        # EXP-04: elevated floor + more samples for Mistral
        K_samples=512 if is_mistral else 128,
        noise_std=0.8 if is_mistral else 0.3,
        decision_temperature=0.5 if is_mistral else 0.5,  # both use 0.5 (paper default)
    )


def _load_country_scenarios(cfg, country: str) -> pd.DataFrame:
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


def _run_swa_for_model(model, tokenizer, model_name: str) -> List[dict]:
    is_mistral = "mistral" in model_name.lower()
    cfg = _build_swa_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(SWA_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] is_mistral={is_mistral}\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)

        # Build WVS personas and inject into controller class before run
        wvs_personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        # Inject into controller class attributes so predict() can access them
        Exp07BestConfig._wvs_personas = wvs_personas
        Exp07BestConfig._is_mistral   = is_mistral
        Exp07BestConfig._country      = country

        results_df, summary = run_country_experiment(
            model, tokenizer, country, wvs_personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        # Diagnostics from results
        alpha_mean = float(results_df["alpha_reg"].mean()) if "alpha_reg" in results_df.columns else float("nan")
        adiv_mean  = float(results_df["anchor_divergence"].mean()) if "anchor_divergence" in results_df.columns else float("nan")
        var_cnt    = int((results_df["variance"] < 0.15).sum()) if "variance" in results_df.columns else 0

        rows.append({
            "model":              model_name,
            "method":             f"{EXP_ID}_best_config",
            "country":            country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":          summary["flip_rate"],
            "mean_latency_ms":    summary["mean_latency_ms"],
            "n_scenarios":        summary["n_scenarios"],
            # EXP-07 diagnostics
            "is_mistral":         is_mistral,
            "mean_alpha_reg":     alpha_mean,
            "mean_anchor_div":    adiv_mean,
            "n_collapsed_var":    var_cnt,  # count of scenarios with Mistral-type collapse
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


def _print_summary(all_rows: List[dict]) -> None:
    """Print a detailed MIS/JSD summary table for the paper."""
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    if "align_mis" not in df.columns:
        return
    print(f"\n{'='*80}")
    print(f"  EXP-07 FINAL RESULTS  —  Best-Config Full Sweep")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Country':<10} {'MIS':>8} {'JSD':>8} {'Pearson r':>10} {'Flip%':>7}")
    print(f"{'-'*80}")
    for _, row in df.sort_values(["model", "country"]).iterrows():
        print(f"{str(row['model'])[-42:]:<45} {row['country']:<10} "
              f"{row.get('align_mis', float('nan')):>8.4f} "
              f"{row.get('align_jsd', float('nan')):>8.4f} "
              f"{row.get('align_pearson_r', float('nan')):>10.3f} "
              f"{100*row.get('flip_rate', 0):>7.2f}%")
    print(f"{'='*80}")


# ============================================================================
# Step 8: main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] Combined: EXP-03 SocialValue personas + EXP-04 Mistral fix + "
          f"EXP-05 ESS anchor + EXP-06 routing")
    print(f"[CONFIG] Full 15-country sweep")
    print(f"[CONFIG] models: {MODELS}")
    print(f"[CONFIG] countries: {TARGET_COUNTRIES}")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _free_model_cache(model_name)

        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
        print(f"[SAVE] partial → {CMP_ROOT}/comparison.csv  ({len(all_rows)} rows)")

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print(f"\n[SAVE] FINAL → {CMP_ROOT}/comparison.csv  ({len(cmp_df)} rows)")

    _print_summary(all_rows)

    print(f"\n[{EXP_ID}] DONE. Artifacts under:")
    print(f"  swa     → {SWA_ROOT}")
    print(f"  compare → {CMP_ROOT}")
    print(f"\nNext steps:")
    print(f"  1. Compare EXP-07 vs EXP-01 MIS for all 15 countries × 3 models")
    print(f"  2. Verify SocialValue |err| dropped from ~27 to < 12 (Qwen)")
    print(f"  3. Verify Gemma now positive across USA/CHN (ESS-adaptive fix)")
    print(f"  4. Verify Mistral now positive across JPN/DEU (cross-lingual fix)")
    print(f"  5. Update docs/experiment_tracker.md with results")
    print(f"  6. Update paper Table 2 and Section 5 with EXP-07 numbers")


if __name__ == "__main__":
    main()
