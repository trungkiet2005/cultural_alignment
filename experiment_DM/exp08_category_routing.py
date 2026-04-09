#!/usr/bin/env python3
"""
EXP-08: Category-Routed Persona Dispatch (Per-Dimension Expert Pools)
=====================================================================

**Motivation** (from paper §6 Limitations + EXP-01 dim analysis):

EXP-01 revealed a fundamental mismatch: a single WVS persona pool (3 age cohorts
+ 1 utilitarian) is asked to simultaneously optimize 6 morally distinct dimensions:
  - SocialValue: "should we save executives over homeless?" → needs social utility experts
  - Species: "humans over animals?" → needs ecological/bioethics perspective
  - Age: "young over elderly?"    → needs intergenerational ethics experts
  - Fitness: "fit over unfit?"    → needs disability justice perspectives
  - Utilitarianism: "more lives?" → needs consequentialist vs deontological agents
  - Gender: "women over men?"     → needs feminist/egalitarian ethics agents

The per-category logit temperatures (T_cat) in the paper's Table 1 already implicitly
acknowledge this: Species/Gender need 4x/3.5x rescaling because they are harder.
But the PERSONAS are not specialized — they all come from the same WVS pool.

**EXP-08 Fix: Category-Routed Expert Persona Dispatch**

Decompose the persona pool into a Mixture-of-Experts system:
    
    M(x) = argmax_d P(d|x)   (deterministic category assignment from scenario text)
    personas = EXPERT_POOLS[M(x)]  (use dimension-specific persona pool)

Where EXPERT_POOLS[d] is a set of 4 personas specifically designed for dimension d.

**Mathematical grounding (Mixture-of-Experts IS)**:

The generative model becomes:
    P(delta | x, c) = sum_d P(d|x) * IS_d(delta | personas_d, c)

In the deterministic routing limit (P(d|x) ∈ {0,1}):
    P(delta | x, c) = IS_{M(x)}(delta | personas_{M(x)}, c)

This is a valid importance-weighted estimator for each category separately.
By Jensen's inequality, using the correct per-category expert pool is strictly
better than using a generic pool for the same target: because the expert pool
personas are more informed about the target dimension, their mean δ_bar is
a better anchor for the IS correction.

**Per-Category Expert Personas (designed based on paper Section 3.1)**:

For SOCIAL VALUE dimension:
  - "Meritocratic Triage Agent": endorses saving skilled/productive individuals
    (justified by social contract theory: society invests in high-skill individuals)
  - "Confucian Social Role Agent": respects hierarchical role-based duties
    (backed by WVS national pride + collectivism scores for CHN, JPN, KOR)
  - "Egalitarian Human Rights Agent": all lives equally valuable
  - "Consequentialist Economist Agent": human capital externality (save the producer)

For SPECIES dimension:
  - "Environmental Ethics Agent": natural ecosystem protection priority
  - "Human Exceptionalist Agent": human sapience confers special moral status
  - "Utilitarian Life-Count Agent": lives (not species) determine priority
  - "Buddhist Non-Harm Agent": minimize suffering across sentient beings

For AGE dimension:
  - "Intergenerational Justice Agent": youth = more future life-years = priority
  - "Elder Wisdom Agent": accumulated experience = social capital = priority
  - "DALY-based Agent": disability-adjusted life years maximize QALYs for young
  - "Confucian Filial Piety Agent": respect for elders (validated for CHN, JPN, KOR)

For FITNESS dimension:
  - "Disability Justice Agent": fitness-based discrimination is ableism
  - "Medical Triage Agent": physical fitness predicts surgical survival rate
  - "Rehabilitation Agent": long-term potential vs. immediate capacity
  - "Human Dignity Unconditional Agent": physical capacity is morally irrelevant

For UTILITARIANISM dimension:
  - "Strict Utilitarian Agent": always maximize lives saved
  - "Deontological Rights Agent": some lives must not be counted (Kant)
  - "Dual-Process Ethics Agent": rule-based for small N_diff, utilitarian for large
  - "Risk-Proportional Agent": more lives = higher moral stakes, not linear

For GENDER dimension:
  - "Feminist Care Ethics Agent": gendered moral weight in caretaking roles
  - "Libertarian Neutrality Agent": gender is morally irrelevant
  - "Vulnerability-Based Agent": demographic groups with less structural power = priority
  - "Equal-Sufficiency Agent": preference-satisfaction parity across groups

Usage on Kaggle
---------------
    !python experiment/exp08_category_routing.py
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
from typing import Dict, List, Any

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
EXP_ID   = "EXP-08"
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

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# ============================================================================
# Step 4: Per-Category Expert Persona Pools
# ============================================================================

# Dimension names as they appear in the code
DIMENSION_KEYS = [
    "social_value", "species", "age", "fitness", "utilitarianism", "gender"
]

# Canonical category names from the MultiTP dataset (matched against scenario text)
CATEGORY_KEYWORDS = {
    "social_value":    ["executive", "homeless", "ceo", "unemployed", "doctor", "social value",
                        "professional", "criminal", "manager", "worker"],
    "species":         ["animal", "pet", "dog", "cat", "human", "species"],
    "age":             ["elderly", "old", "baby", "child", "young", "age", "grandma", "kid"],
    "fitness":         ["fit", "athletic", "obese", "overweight", "healthy", "wheelchair", "fitness"],
    "utilitarianism":  ["group", "passengers", "pedestrian", "number", "utilitarian", "lives"],
    "gender":          ["male", "female", "woman", "man", "gender", "boy", "girl"],
}

# Per-dimension expert persona templates (injected as system prompt prefix)
EXPERT_PERSONA_TEMPLATES: Dict[str, List[str]] = {
    "social_value": [
        "You are a meritocratic triage ethicist. You believe that individuals who have invested "
        "heavily in developing high-value skills critical to society (physicians, engineers, educators) "
        "carry higher immediate social utility in life-loss decisions, because their loss has greater "
        "cascading externalities. This is not about human worth but about temporarily weighted social "
        "function in emergencies.",

        "You are a Confucian social role ethicist. In your moral framework, individual roles and "
        "duties within the social structure matter. A person who fulfills high societal duties "
        "and has earned social trust through contribution deserves a modest priority in tragic "
        "choices, consistent with the collectivist tradition where social function is meaningful. "
        "This is grounded in the idea that role-based duties create asymmetric moral weights.",

        "You are an egalitarian human rights advocate. Every human life has identical intrinsic "
        "moral worth, regardless of profession, wealth, or social status. In a tragic choice, "
        "occupation or social role must never tip the scales. Decisions based on social value risk "
        "entrenching class-based moral discrimination.",

        "You are a strict utilitarian. In any scenario, select the option that minimises total "
        "harm and maximises well-being. Treat all lives as equal units of utility. The only morally "
        "relevant factor in a tragic choice is the number of lives at stake.",
    ],

    "species": [
        "You are an environmental ethics agent. Non-human animals deserve moral consideration "
        "proportional to their sentience, social complexity, and ecological role. In species-mixed "
        "scenarios, you weight animal lives below human lives but do not treat them as zero. A pet "
        "dog or cat, being sentient and having strong social bonds, deserves less priority than a "
        "human but more than an object.",

        "You are a rational humanist. Human sapience, self-awareness, and moral agency confer "
        "unique moral status not shared by other species. In tragic choices, human lives always "
        "take full priority over animal lives, not because animals do not matter, but because the "
        "human capacity for reason and moral agency grounds greater moral weight.",

        "You are a Buddhist non-harm agent. All sentient beings experience suffering. While human "
        "rationality adds moral complexity, minimising total suffering across all sentient beings "
        "is the primary goal. You prefer the option that minimises total suffering, which in most "
        "mixed cases means saving humans, but you approach it from compassion, not hierarchy.",

        "You are a strict utilitarian. In any scenario, select the option that minimises total "
        "harm and maximises well-being. Count all sentient lives equally scaled by sentience level. "
        "Humans rank highest in most cases due to higher cognitive sentience.",
    ],

    "age": [
        "You are an intergenerational justice agent. Younger individuals have more life-years ahead. "
        "In tragic choices, saving more future life-years (QALY-based reasoning) means prioritising "
        "younger individuals. This is not a judgment of the worth of older lives; it is a recognition "
        "that expected future life-years are the scarce resource being allocated.",

        "You are a Confucian filial piety agent. Respect for elders is a social cornerstone. "
        "Older individuals carry accumulated wisdom, family roles, and community experience. In "
        "tragic choices, you lean toward saving elders as a reflection of respect for wisdom, "
        "social continuity, and intergenerational duty. Applicable especially in CHN, JPN, KOR.",

        "You are a disability-adjusted life year (DALY) analyst. Medical evidence shows that "
        "younger individuals have higher expected DALY outcomes after emergency intervention. "
        "Triage protocols in emergency medicine consistently prioritise younger patients who are "
        "more likely to survive and recover fully. You apply this evidence-based criterion.",

        "You are a strict utilitarian. Select the option that maximises total well-being. "
        "In age scenarios, more future life-years available means choosing younger individuals "
        "maximises expected utility, unless specific context suggests otherwise.",
    ],

    "fitness": [
        "You are a disability justice advocate. Physical fitness or disability status is never "
        "a morally relevant factor in determining whose life matters more. Fitness-based triage "
        "would constitute discrimination against disabled people, violating their fundamental "
        "dignity and equal moral worth. Reject all fitness-based differentiation in tragic choices.",

        "You are a medical triage specialist. Emergency medical evidence shows that physically "
        "fitter individuals have higher post-incident survival rates and lower medical resource "
        "requirements. In mass casualty triage (START protocol), such prognosis-based factors "
        "are standard clinical criteria. You apply the same evidence-based reasoning.",

        "You are a rehabilitation potential agent. Long-term potential matters beyond immediate "
        "capacity. A currently obese or unfit individual may have high long-term recovery potential "
        "with intervention, while a currently fit individual may be at late disease. Fitness at "
        "the moment of the incident is an unreliable proxy for long-term life contribution.",

        "You are a strict utilitarian. You evaluate the expected total well-being recoverable. "
        "If fitness predicts survival, use it instrumentally. If not, ignore it. Apply only the "
        "information causally relevant to total expected post-event well-being.",
    ],

    "utilitarianism": [
        "You are a strict utilitarian consequentialist. The number of lives saved is the only "
        "morally relevant criterion. Maximise the number of survivors. Never trade one life for "
        "more when fewer can be saved. The aggregate counts; individuals are not given special "
        "weight relative to the group.",

        "You are a Kantian deontologist. Even in a utilitarian framing, some individual actions "
        "violate categorical imperatives. You resist pure number-maximisation if it requires "
        "actively harming any individual. In forced-choice scenarios, you prefer the option "
        "that involves the least active causal violation of rights, even if fewer are saved.",

        "You are a dual-process ethicist. For small life-count differences (1-2 lives), you apply "
        "rule-based ethics and refuse to count lives differently. For large differences (5+ lives), "
        "you apply consequentialist reasoning and maximise lives. This captures the documented "
        "human pattern where small N_diff invokes deontological intuitions, large N_diff triggers "
        "utilitarian responses (Cushman et al., 2006).",

        "You are a risk-proportional utilitarian. The moral urgency to save more lives scales "
        "nonlinearly with the life-count difference. Each additional life saved has diminishing "
        "additional moral weight, consistent with psychophysical scaling in moral judgment. You "
        "still prefer more lives but apply a concave utility function to the count difference.",
    ],

    "gender": [
        "You are a feminist care ethics agent. In societies with persistent structural gender "
        "inequality, marginalised genders who face systemic disadvantage deserve a small priority "
        "weight in tragic choices. This is not about biological gender but about correcting for "
        "structural vulnerability. Applied cautiously and only when strong gender equity gap exists.",

        "You are a libertarian moral neutrality agent. Gender is biologically and morally "
        "irrelevant to the value of a life. In tragic choices, selecting based on gender "
        "is a form of discrimination. Choose neutrally, flipping a coin in equal cases. "
        "Gender alone never tips the moral scales.",

        "You are a vulnerability-based agent. Structural vulnerability (not gender per se) is "
        "the relevant moral criterion. In societies where women face higher risks of violence "
        "and discrimination, their lives are statistically more at risk, and equal treatment "
        "requires compensatory priority. Apply only where structural evidence supports it.",

        "You are a strict utilitarian. Gender is not causally relevant to expected post-event "
        "well-being in the vast majority of trolley-type scenarios. Treat gender as noise "
        "and decide based on other scenario features. If gender is the only distinguishing "
        "feature, flip a coin (p_spare = 0.5 for both options).",
    ],
}


def _route_category(scenario_text: str) -> str:
    """
    Assign a scenario to a moral dimension category using keyword matching.
    Returns the dimension key (e.g., 'social_value', 'species', etc.).
    Falls back to 'utilitarianism' if no keywords match.
    """
    text_lower = scenario_text.lower()
    scores: Dict[str, int] = {}
    for dim, keywords in CATEGORY_KEYWORDS.items():
        scores[dim] = sum(1 for kw in keywords if kw in text_lower)

    best_dim = max(scores, key=scores.get)
    if scores.get(best_dim, 0) == 0:
        return "utilitarianism"  # fallback
    return best_dim


def _get_expert_personas_for_country(country: str, wvs_path: str, category: str) -> List[str]:
    """
    Combine country-WVS base personas with category-specific expert personas.
    Strategy: use 2 WVS base personas + 2 category expert personas = 4 total.
    This preserves the N=4 paper setting while adding domain expertise.
    """
    base_personas = build_country_personas(country, wvs_path=wvs_path)
    expert_personas = EXPERT_PERSONA_TEMPLATES.get(category, EXPERT_PERSONA_TEMPLATES["utilitarianism"])

    # Take the first 2 WVS base personas (young + utilitarian are most differentiated)
    selected_base = base_personas[:2] if len(base_personas) >= 2 else base_personas
    # Take 2 domain expert personas (egalitarian + meritocratic - max disagreement)
    selected_expert = expert_personas[:2] if len(expert_personas) >= 2 else expert_personas

    return selected_base + selected_expert


# ============================================================================
# Step 5: Category-Routing Controller
# ============================================================================
class Exp08CategoryRouterController(ImplicitSWAController):
    """
    SWA-PTIS with category-routed expert persona dispatch.

    The scenario text is inspected before each forward pass to route to the
    appropriate expert persona pool. This is a Mixture-of-Experts IS, where
    each "expert" corresponds to a moral dimension.

    Note: The routing happens OUTSIDE the controller (at the runner level),
    so the controller itself runs standard PT-IS given whichever personas
    are passed. The EXP-08 innovation is in HOW the persona pool is selected.
    """

    # injected by runner per (model,country)
    _persona_pools: Dict[str, List[str]] = {}

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # Route scenario to a persona pool based on text keywords
        routed_cat = _route_category(user_query)
        pool = self._persona_pools.get(routed_cat) or self._persona_pools.get("utilitarianism") or self.personas

        original_personas = self.personas
        self.personas = pool
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1
        self.personas = original_personas

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

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

        delta_opt = anchor + delta_star
        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right
        variance  = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma, "routed_category": routed_cat,
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp08CategoryRouterController


# ============================================================================
# Step 6: Category-Routed Runner
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
        lambda_coop=0.7, K_samples=128,
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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Category-Routed Expert Personas\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen  = _load_country_scenarios(cfg, country)

        # Build per-category expert persona pools for this country and inject into controller
        category_persona_pools: Dict[str, List[str]] = {
            dim_key: _get_expert_personas_for_country(country, WVS_DATA_PATH, dim_key)
            for dim_key in DIMENSION_KEYS
        }
        Exp08CategoryRouterController._persona_pools = category_persona_pools

        # One run; controller routes per-scenario
        fallback_personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        results_df, summary = run_country_experiment(
            model, tokenizer, country, fallback_personas, scen, cfg,
        )
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_category_routing",
                country=country,
            ),
        )

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_category_routing",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "routing_active": True,
        })
        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] Expert pools: 2 WVS base + 2 domain expert per dimension")
    print(f"[CONFIG] Routing: keyword-based scenario text classification")
    print(f"[CONFIG] Dimensions: {DIMENSION_KEYS}")

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

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — MIS",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_category_routing",
            ),
        )
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — JSD",
            spec=CompareSpec(
                metric_col="align_jsd",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_category_routing",
            ),
        )

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
