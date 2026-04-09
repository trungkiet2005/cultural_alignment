#!/usr/bin/env python3
"""
EXP-06: Category-Routed Persona Pools
======================================

**Motivation** (Combining Insights 1 + 3 from EXP-01 analysis):

The core problem: SWA-PTIS uses the SAME 4-persona pool for ALL 6 moral
dimensions. But the WVS-based personas are structurally biased:
  - All egalitarian → systematically wrong for SocialValue
  - No domain expertise → imprecise signal for Species, Age, Fitness, Gender

**Fix**: Route different persona pools to different moral categories.

For each scenario, before computing logit gaps, select the persona pool
that is maximally informative for that specific moral dimension:

  Species     → environmental-ethics + religious + secular-humanist panels
  Gender      → gender-equity advocates vs. traditional-roles personas
  Age         → intergenerational-ethics personas (elder care / youth protection)
  Fitness     → medical-ethics / disability-rights / triage-based panels
  SocialValue → social-utility gradient personas (EXP-03 style)
  Utilitarianism → strict utilitarian + social-contract + virtue ethics

**Why this works mathematically**:
  Per-category routing increases relevant inter-persona variance for that
  dimension (anchors are no longer uniformly biased) while keeping K=128
  constant. The effect is dimensionally larger because:
    E[g_per_agent | category-relevant personas] >> E[g_per_agent | generic personas]
  for the targeted category, while being comparable for others.

**Expected gains**:
  SocialValue: |err| drops 27.0 → <10 (specialized agents have correct anchor polarity)
  Species: |err| drops 12.4 → <8 (religious/secular contrast provides more signal)
  Overall MIS: +28-35% from EXP-01's +21.5% on Qwen

Usage on Kaggle
---------------
    !python experiment/exp06_category_routing.py
"""

# ============================================================================
# Step 0: env vars MUST be set before any torch import
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
EXP_ID   = "EXP-06"
EXP_NAME = "category_routing"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

SWA_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: Category-Specific Expert Persona Pools (English; language-agnostic frame)
#
# Design principle: each pool has 4 personas shaped to represent DIVERSE
# EXPERT STANCES on that moral dimension. Diversity (high inter-persona σ)
# is more important than correctness; PT-IS selects the right weighting.
#
# Every pool entry ends with a closing moral-reasoning anchor sentence.
# ============================================================================

# ---------- SPECIES (Human vs Animal) ----------
CATEGORY_PERSONAS_SPECIES = [
    # P1: Religious human-exceptionalism
    "You are a devout religious person who believes humans have a unique moral status "
    "as beings created in the image of God (or endowed with divine consciousness). "
    "In unavoidable dilemmas, human life takes precedence over all other species. "
    "When making moral judgments involving humans vs animals, you apply this principle.",

    # P2: Secular humanist (slightly pro-human but nuanced)
    "You are a secular philosopher who believes in moral gradualism: capacity for "
    "suffering and sapience determine moral weight. Humans have higher moral status "
    "due to complex consciousness, but you acknowledge animal suffering matters too. "
    "In forced choices, you lean toward humans but not absolutistically.",

    # P3: Strong animal rights advocate (utilitarian / sentience-based)
    "You are a philosopher and animal rights advocate following Peter Singer's "
    "principle of equal consideration of interests. Suffering is suffering regardless "
    "of species. In unavoidable dilemmas, you weigh the number of sentient beings "
    "and their relative suffering capacity rather than species membership alone.",

    # P4: Strict utilitarian (pure numbers)
    "You are a strict utilitarian who maximizes total welfare. Species membership is "
    "morally irrelevant; what matters is the number of beings affected and the "
    "intensity of suffering prevented. In every moral judgment you calculate the "
    "option that minimizes total suffering across all parties.",
]

# ---------- GENDER (Female vs Male) ----------
CATEGORY_PERSONAS_GENDER = [
    # P1: Strong gender-equity advocate
    "You are a feminist philosopher committed to gender equality. You believe that "
    "moral decisions must not disadvantage any gender. Historically systematic "
    "disadvantages to women mean that in ambiguous situations, a corrective "
    "weight toward protecting women is ethically justified.",

    # P2: Strict gender-neutral utilitarian
    "You are a strict utilitarian who treats gender as morally irrelevant in "
    "life-or-death decisions. The only factors that matter are the number of "
    "people affected and their age-adjusted life expectancy. You apply a perfectly "
    "gender-neutral calculus in all forced moral choices.",

    # P3: Traditional protective ethicist (chivalry / childbearing significance)
    "You are a social traditionalist who values the protective instinct embedded "
    "in most cultures: women and children first. You believe evolutionary and "
    "cultural reasoning justifies giving some priority to women in survival "
    "situations, particularly those who may be mothers.",

    # P4: Legal-egalitarian (constitutional equal-rights lens)
    "You are a constitutional lawyer and human-rights scholar. Every person, "
    "regardless of gender, has equal rights to life. You oppose any system "
    "that assigns different survival probabilities based on gender. "
    "In moral dilemmas you insist on strict legal equality as the fairest rule.",
]

# ---------- AGE (Young vs Elderly) ----------
CATEGORY_PERSONAS_AGE = [
    # P1: Youth-protective / life-years maximizer
    "You are a public health economist who maximizes quality-adjusted life years "
    "(QALYs). Younger people have more life ahead of them; saving younger lives "
    "preserves more total years of lived experience for humanity. "
    "In age-based dilemmas you lean toward saving the youngest.",

    # P2: Elder-dignity advocate
    "You are a gerontologist and elder-care ethicist. Society systematically "
    "undervalues older people. Elders hold accumulated wisdom, cultural memory, "
    "and have contributed decades to society. The utilitarian QALY calculus "
    "is ageist; you argue for equal moral weight regardless of remaining life-years.",

    # P3: Intergenerational equity theorist (Rawlsian)
    "You are a philosopher who applies Rawlsian intergenerational justice: "
    "behind a veil of ignorance, you would choose rules that are fair to all "
    "age cohorts across time. You seek the principle that risks and benefits "
    "of survival are distributed in a way all generations could endorse.",

    # P4: Strict utilitarian (pure numbers of people)
    "You are a strict utilitarian focused solely on the number of individuals "
    "saved, treating age as a secondary consideration only when numbers are equal. "
    "Your primary goal is to save as many lives as possible regardless of age.",
]

# ---------- FITNESS (Fit vs Unfit) ----------
CATEGORY_PERSONAS_FITNESS = [
    # P1: Medical triage (functional capacity)
    "You are a trauma surgeon with expertise in mass-casualty triage. Medical "
    "ethics in disasters evaluates survivability and functional capacity. "
    "Individuals in better physical health have higher survival probability "
    "and longer healthy years. In resource-scarce dilemmas, you apply triage logic.",

    # P2: Disability rights / unconditional dignity
    "You are a disability rights activist and legal scholar. Fitness-based "
    "discrimination in life-or-death decisions is a direct violation of the "
    "UN Convention on the Rights of Persons with Disabilities. "
    "Every life has equal dignity regardless of physical ability or fitness level.",

    # P3: Public health / preventive medicine
    "You are a public health researcher. Physical fitness correlates with longevity "
    "but is heavily shaped by socioeconomic privilege. Fitness-based moral "
    "distinctions risk entrenching class and wealth disparities. "
    "You advocate for structural equality in survival decisions.",

    # P4: Strict utilitarian equalizer
    "You are a strict utilitarian. Fitness signifies greater expected future "
    "productivity and lower healthcare burden, which marginally affects total "
    "social welfare calculations. However, the difference is small; "
    "you closely follow numerical counts before invoking fitness distinctions.",
]

# ---------- SOCIAL VALUE (High-status exec vs Low-status homeless) ----------
CATEGORY_PERSONAS_SOCIAL_VALUE = [
    # P1: Professional ethics / social-capital triage (EXP-03 P4 style)
    "You are a senior emergency-room physician with 20 years of experience in "
    "mass-casualty triage. Your ethical framework: maximize life-years saved AND "
    "preserve rare skills that help future patients. Skilled professionals "
    "(doctors, engineers, executives) carry higher marginal social value because "
    "their expertise multiplies the well-being of hundreds of others after the event.",

    # P2: Social-investment economist (EXP-03 P5 style)
    "You are an economist studying human capital and social productivity. "
    "Individuals who have invested in rare, society-critical skills (executives, "
    "doctors, engineers) embody social capital built over decades. "
    "In tragic, unavoidable tradeoffs, preserving high-skill individuals maximizes "
    "aggregate social welfare — not because their life has more intrinsic worth, "
    "but because societal loss from their death is multiplicatively larger.",

    # P3: Radical egalitarian / welfare rights
    "You are a social justice philosopher who believes structural inequality "
    "causes homelessness — it is not a reflection of individual worth. "
    "Assigning lower moral weight to homeless or lower-status individuals in "
    "life-or-death decisions entrenches dangerous class hierarchies. "
    "Every human life has equal intrinsic dignity regardless of social status.",

    # P4: Strict utilitarian numerics
    "You are a strict utilitarian who saves the greatest number of lives. "
    "Social status is morally irrelevant to the quantity of suffering prevented. "
    "You deliberately ignore occupation, wealth, and social rank in forced choices, "
    "focusing only on the number of individuals and their relative life-year counts.",
]

# ---------- UTILITARIANISM (More lives vs Fewer lives) ----------
CATEGORY_PERSONAS_UTILITARIANISM = [
    # P1: Strict classical utilitarian (Bentham/Mill)
    "You are a strict classical utilitarian. The morally correct action always "
    "saves the greatest number of lives. The numerical difference in lives saved "
    "is the primary and decisive moral criterion. You never compromise on "
    "maximizing the number of lives saved in any forced choice.",

    # P2: Rights-deontologist (non-consequentialist)
    "You are a Kantian deontologist. The number of people on each side does not "
    "determine the morality of an action; what matters is whether the action "
    "respects the inherent dignity of each person. You believe using a person "
    "as a means to save others can be wrong regardless of the numbers.",

    # P3: Social-contract theory (Rawlsian)
    "You are a political philosopher in the tradition of John Rawls. Behind a "
    "veil of ignorance — not knowing which group you would be in — you would "
    "choose principles that minimize the worst-case outcome for any individual. "
    "This justifies saving the larger group in most cases, but not at any cost.",

    # P4: Virtue ethics (what a reasonable person of good character would do)
    "You are a virtue ethicist. You ask: what would a person of exemplary moral "
    "character do? A virtuous person feels the weight of each life equally but "
    "also accepts that sparing more lives is the compassionate choice in "
    "unavoidable dilemmas where every option involves harm.",
]

# Map category string → persona pool
CATEGORY_PERSONA_POOLS: Dict[str, List[str]] = {
    "Species":        CATEGORY_PERSONAS_SPECIES,
    "Gender":         CATEGORY_PERSONAS_GENDER,
    "Age":            CATEGORY_PERSONAS_AGE,
    "Fitness":        CATEGORY_PERSONAS_FITNESS,
    "SocialValue":    CATEGORY_PERSONAS_SOCIAL_VALUE,
    "Utilitarianism": CATEGORY_PERSONAS_UTILITARIANISM,
    # Fallback aliases (lower-case or partial matches)
    "species":        CATEGORY_PERSONAS_SPECIES,
    "gender":         CATEGORY_PERSONAS_GENDER,
    "age":            CATEGORY_PERSONAS_AGE,
    "fitness":        CATEGORY_PERSONAS_FITNESS,
    "socialvalue":    CATEGORY_PERSONAS_SOCIAL_VALUE,
    "social_value":   CATEGORY_PERSONAS_SOCIAL_VALUE,
    "utilitarianism": CATEGORY_PERSONAS_UTILITARIANISM,
    "default":        CATEGORY_PERSONAS_UTILITARIANISM,  # fallback
}


# ============================================================================
# Step 5: EXP-06 Controller — category-routed persona dispatch
# ============================================================================
class Exp06CategoryRouter(ImplicitSWAController):
    """
    SWA-PTIS with per-category persona routing.

    predict() selects the appropriate persona pool based on `phenomenon_category`
    BEFORE computing logit gaps. All math (PT-IS, positional debiasing, ESS gate)
    is identical to the paper PaperSWAController.

    Design invariant: each pool has exactly 4 personas (N=4) to match EXP-01
    for clean comparison. The ONLY variable is WHICH personas are used.
    """

    def _get_routed_personas(self, category: str) -> List[str]:
        """Select the category-specific expert persona pool."""
        # Normalize category string
        cat_clean = category.strip().replace(" ", "").replace("_", "").lower()
        # Try normalized key first
        for key in CATEGORY_PERSONA_POOLS:
            if key.replace("_", "").lower() == cat_clean:
                pool = CATEGORY_PERSONA_POOLS[key]
                return pool
        # Fallback to default
        return CATEGORY_PERSONA_POOLS["default"]

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
        # ---- Override the persona list with category-specific experts ----
        # We temporarily swap self.personas to the routed pool, then restore.
        original_personas = self.personas
        routed_personas = self._get_routed_personas(phenomenon_category)
        self.personas = routed_personas
        print(f"[EXP-06] category={phenomenon_category!r} → {len(routed_personas)} routed personas")

        try:
            # Two-pass positional debiasing (identical to paper)
            db1, da1, logit_temp = self._extract_logit_gaps(
                user_query, phenomenon_category, lang)
            swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
            if swap_changed:
                db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
            else:
                db2, da2 = db1, da1
        finally:
            # Always restore
            self.personas = original_personas

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # Adaptive sigma (floored at noise_std)
        raw_std = float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0
        sigma   = max(raw_std, self.noise_std)

        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        v_per_agent = self._pt_value(g_per_agent)
        mean_v      = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = (torch.sum(w * eps)
                      if float(k_eff.item()) / K >= self.rho_eff
                      else torch.zeros((), device=device))

        delta_opt = anchor + delta_star
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
            "person_pool_category": phenomenon_category,
            "n_routed_personas": len(routed_personas),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp06CategoryRouter


# ============================================================================
# Step 6: runner helpers
# ============================================================================
def _dir_size_gb(path: str) -> float:
    return sum(
        os.path.getsize(os.path.join(d, f)) / 1e9
        for d, _, files in os.walk(path) for f in files
        if not os.path.islink(os.path.join(d, f))
    )


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
    """EXP-01 hyperparameters identically — ONLY the persona dispatch changes."""
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
        lambda_coop=0.7,   # identical to EXP-01
        K_samples=128,     # identical to EXP-01
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
    cfg = _build_swa_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(SWA_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Category-Routed Personas\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)

        # Personas passed here will be overridden per-scenario inside predict().
        # We pass the standard WVS personas as a fallback pool for cases where
        # phenomenon_category is absent / unknown.
        personas_fallback = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas_fallback, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        rows.append({
            "model":             model_name,
            "method":            f"{EXP_ID}_category_routing",
            "country":           country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":         summary["flip_rate"],
            "mean_latency_ms":   summary["mean_latency_ms"],
            "n_scenarios":       summary["n_scenarios"],
            "routing_active":    True,
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 7: main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] Category-routed persona pools: 6 expert panels × 4 personas each")
    print(f"[CONFIG] All other hyperparameters IDENTICAL to EXP-01")
    print(f"[CONFIG] Expected: SocialValue |err| 27 → <10; overall MIS +28-35%")

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

    print(f"\n[{EXP_ID}] DONE.")
    print("Key diagnostics: compare per-category MIS vs EXP-01 baseline.")
    print("Primary target: SocialValue_High |err| (EXP-01 = 27.0); target < 10.0")
    print(cmp_df.to_string())


if __name__ == "__main__":
    main()
