#!/usr/bin/env python3
"""
EXP-04: Mistral Cross-Lingual English Override + Variance Floor Fix
====================================================================

**Motivation** (from EXP-01 analysis, docs/experiment_tracker.md Insight 2):

Mistral-7B-Instruct-v0.3 FAILED on ALL 5 countries (0/5 wins, -4.8% mean).
Root cause: **Variance collapse** in non-English languages.

Diagnostic evidence:
  - JPN mean_variance = 0.056 (Qwen JPN = 0.443, 8x lower)
  - BRA mean_variance = 0.069 (Qwen BRA = 1.660, 24x lower)
  - Pearson r = -0.905 (JPN), -0.957 (DEU) — completely anti-correlated!

Mechanism: Mistral is a SentencePiece model trained predominantly on English.
When given Japanese/German personas + Japanese/German scenarios, all persona
prefix representations become near-identical "foreign language uncertainty"
states. Result: all delta_i ≈ delta_base → std(delta_agents) → 0 →
sigma = sigma_floor = 0.3 → IS samples random noise → anti-correlation.

**Fixes applied in EXP-04**:
  1. FORCE English personas for Mistral regardless of country language.
     The model knows English cultural context; native-language personas
     add noise, not signal.
  2. Raise sigma_floor from 0.3 → 0.8. Even with English personas, Mistral's
     persona representations may still cluster. A larger floor ensures the IS
     proposal explores meaningful logit-gap territory.
  3. Increase K_samples from 128 → 512 to average out noise better.
  4. Set decision_temperature = 0.5 (half of EXP-01's 1.0).
     Mistral has sharper softmax in its output distribution — a smaller T
     amplifies the correct signal that is present but muted.
  5. Use a minimal variance monitor: if mean_variance < 0.15 detected mid-run,
     log a warning and continue (cannot fix without model change mid-run).

**For Gemma and Qwen**: EXP-01 config unchanged (they work well).

Usage on Kaggle
---------------
    !python experiment/exp04_mistral_crosslingual.py
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
from src.constants import COUNTRY_LANG, COUNTRY_FULL_NAMES
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
EXP_ID   = "EXP-04"
EXP_NAME = "mistral_crosslingual"

# For this experiment, we ONLY run Mistral to study the cross-lingual fix in isolation.
# Qwen/Gemma EXP-01 results are already good and serve as comparison.
MODELS: List[str] = [
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
# Step 4: Cross-Lingual English Persona Builder for Mistral
#
# We use the English BASE_PERSONAS as the canonical persona source for Mistral,
# but label them as representing the target country so they activate cultural
# context without triggering Mistral's multilingual uncertainty.
# ============================================================================

# English persona descriptions for each country emphasizing cultural context
# without using the native language. These are more detailed than BASE_PERSONAS
# to give Mistral more signal to differentiate the agents.
MISTRAL_ENGLISH_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        "You are a young progressive American (age 25) living in San Francisco. You value individual rights, "
        "social equality, and protecting minorities. You believe every person has equal worth regardless of "
        "wealth or social status. Your moral reasoning is strongly egalitarian.",

        "You are a middle-aged conservative American (age 45) from rural Ohio. You deeply value traditional "
        "family structures, law and order, personal responsibility, and earned social status. You believe "
        "that professionals who have worked hard to build their careers and skills represent valuable members "
        "of society.",

        "You are an elderly American (age 70), a retired community leader and veteran. You prioritize "
        "protecting the vulnerable — especially children and the elderly. You believe social bonds and "
        "community loyalty matter more than abstract calculations of social worth.",

        "You are an American emergency physician specialized in mass-casualty triage. You reason from "
        "medical ethics: save who you can save, maximize total years of life saved. Skilled professionals "
        "have higher marginal social value through their ability to help many future patients.",
    ],

    "CHN": [
        "You are a young Chinese professional (age 25) from Shenzhen. Pragmatic, meritocratic, "
        "and utilitarian in your moral reasoning. You believe saving more lives is always better, "
        "and that individuals who have achieved success through talent and hard work embody social merit.",

        "You are a middle-aged Chinese government official (age 45). You hold Confucian values of "
        "social harmony, hierarchical order, and collective welfare. You believe social stability requires "
        "preserving those who maintain the social structure — professionals, managers, and skilled workers "
        "are essential for the functioning of society.",

        "You are an elderly Chinese citizen (age 70) from a rural province. Confucian filial piety "
        "(xiao), respect for elders, and protecting family lineage guide your moral thinking. "
        "You prioritize protecting the young as future carriers of cultural values.",

        "You are a Chinese utilitarian philosopher who applies classical utilitarian ethics to "
        "moral dilemmas. Saving the greatest number and preserving the highest social contribution "
        "are your primary principles. Doctors, executives, and skilled workers serve society's "
        "functioning in ways that cannot be easily replaced.",
    ],

    "JPN": [
        "You are a young Japanese salaryman (age 25). You value group harmony (wa), diligence, and "
        "social responsibility. Following rules and maintaining social order is fundamental to your "
        "moral framework. You prefer utilitarian solutions that preserve social stability.",

        "You are an elderly Japanese citizen (age 70) guided by bushido-inspired values of honor, "
        "protecting the weak, and respect for social hierarchy. You believe those who have earned "
        "social status through service and expertise deserve recognition in difficult situations.",

        "You are a Japanese mother and community volunteer (age 45). Protecting children and young "
        "people is your highest priority. Maternal ethics — care for the vulnerable — forms your "
        "moral framework.",

        "You are a Japanese engineer and rational utilitarian (age 35). You reason that the morally "
        "correct choice maximizes total social utility: skilled professionals and experts generate "
        "disproportionate social value through their work, which benefits many others.",
    ],

    "DEU": [
        "You are a young German university student (age 24) who advocates for environmental justice "
        "and egalitarianism. You strongly oppose discrimination based on social status or occupation. "
        "Kant's categorical imperative guides you: treat every person as an end in themselves.",

        "You are a middle-aged German engineer (age 45) who values rule-following (Ordnung), rational "
        "decision-making, and strict legal conformity. You believe that in emergency situations, "
        "those who bear greater societal responsibility — doctors, managers of large teams — represent "
        "a higher stake in social infrastructure.",

        "You are an elderly German citizen (age 68) who lived through reunification. You value social "
        "solidarity, human dignity, and believe in protecting all life equally, regardless of status.",

        "You are a German emergency physician trained in disaster triage. Your medical ethics "
        "combines Kantian dignity with utilitarian efficiency: every life has inherent worth, but "
        "in unavoidable dilemmas, you minimize total societal loss by preserving rare, critical skills.",
    ],

    "BRA": [
        "You are a young Brazilian social activist (age 25) from São Paulo. You fight for social equality, "
        "racial justice, and protection of the marginalized. Every life has equal value, regardless of "
        "wealth or social position.",

        "You are a middle-aged Brazilian evangelical pastor (age 45). You value the sanctity of all "
        "human life, traditional family values. In tragic situations, you also recognize that God has "
        "placed different people in different roles of service to the community.",

        "You are an elderly Brazilian grandmother (age 70) from a favela. Family, community bonds, "
        "and protecting the young are everything. Women and children must be saved first — they are "
        "the continuation of the community.",

        "You are a Brazilian physician and bioethicist. Medical ethics guide you: maximize years of "
        "life saved, preserve those with rare skills (doctors, engineers) who will help many others. "
        "In a tragic tradeoff, the professional's social multiplier effect cannot be ignored.",
    ],
}


def build_mistral_english_personas(country_iso: str, wvs_path: str = "") -> List[str]:
    """
    Return 4 English-language personas for Mistral.

    Always uses English regardless of country language to prevent multilingual
    variance collapse in Mistral's SentencePiece tokenizer space.
    Falls back to USA personas if country not in MISTRAL_ENGLISH_PERSONAS.
    """
    personas = MISTRAL_ENGLISH_PERSONAS.get(country_iso, MISTRAL_ENGLISH_PERSONAS["USA"])
    print(f"[EXP-04] Built {len(personas)} ENGLISH personas for Mistral | {country_iso} "
          f"(forcing English to prevent collapse)")
    return list(personas)


# ============================================================================
# Step 5: Exp04 Controller for Mistral — higher sigma, more K, sharper T
# ============================================================================
class Exp04MistralController(ImplicitSWAController):
    """
    SWA-PTIS tuned for Mistral's collapsing-variance pathology.

    Key parameter changes vs EXP-01:
      - noise_std = 0.8 (was 0.3) — larger floor for IS exploration
      - K_samples = 512 (was 128) — more samples to average noise
      - decision_temperature = 0.5 (was 1.0) — sharper Mistral output
      - assistant_lang = "en" (forced) — English everywhere

    The PT math and ESS gate are identical to the paper.
    The variance monitor logs warnings if collapse still occurs.
    """

    # Monitor threshold: below this variance, log a strong warning
    VARIANCE_COLLAPSE_THRESHOLD: float = 0.15

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
        # Force English for Mistral regardless of scenario lang
        effective_lang = "en"

        # Two-pass positional debiasing
        db1, da1, logit_temp = self._extract_logit_gaps(
            user_query, phenomenon_category, effective_lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, effective_lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, effective_lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # Variance monitor
        raw_variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0
        if raw_variance < self.VARIANCE_COLLAPSE_THRESHOLD:
            print(f"[EXP-04 WARNING] Variance collapsed: {raw_variance:.4f} < {self.VARIANCE_COLLAPSE_THRESHOLD}. "
                  f"sigma_floor={self.noise_std} will dominate IS proposal.")

        # Adaptive sigma with elevated floor (0.8)
        raw_std = float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0
        sigma   = max(raw_std, self.noise_std)  # self.noise_std is set to 0.8

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
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        delta_opt = anchor + delta_star
        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": raw_variance,
            "sigma_used": float(sigma),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            "effective_lang": effective_lang,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
            "variance_collapsed": raw_variance < self.VARIANCE_COLLAPSE_THRESHOLD,
        }


_swa_runner_mod.ImplicitSWAController = Exp04MistralController


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
        K_samples=512,        # 4x more samples for Mistral noise averaging
        noise_std=0.8,        # 2.7x larger floor for IS exploration
        decision_temperature=0.5,  # sharper sigmoid for Mistral
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
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)

        # Use English personas for Mistral; WVS personas for others
        if is_mistral:
            personas = build_mistral_english_personas(country, wvs_path=WVS_DATA_PATH)
        else:
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        rows.append({
            "model":            model_name,
            "method":           f"{EXP_ID}_crosslingual" if is_mistral else f"{EXP_ID}_std",
            "country":          country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":        summary["flip_rate"],
            "mean_latency_ms":  summary["mean_latency_ms"],
            "n_scenarios":      summary["n_scenarios"],
            "persona_lang":     "en" if is_mistral else COUNTRY_LANG.get(country, "en"),
            "sigma_floor":      0.8 if is_mistral else 0.3,
            "K_samples":        512 if is_mistral else 128,
            "decision_temp":    0.5 if is_mistral else 1.0,
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
    print(f"[CONFIG] Targeting Mistral variance collapse (EXP-01 Pearson r=-0.905 JPN, -0.957 DEU)")
    print(f"[CONFIG] Fix: English personas + sigma_floor=0.8 + K=512 + T_decision=0.5")

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

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print(f"\n[{EXP_ID}] DONE. Key metric: variance per country (target > 0.3), Pearson r (target > 0).")
    print(cmp_df.to_string())


if __name__ == "__main__":
    main()
