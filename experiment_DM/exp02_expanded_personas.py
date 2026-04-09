#!/usr/bin/env python3
"""
EXP-02: Expanded Persona Pool — 8-Agent SWA-PTIS
=================================================

**Hypothesis**: The original 4-persona pool (young/middle/older + utilitarian)
under-samples within-country cultural variance. Two orthogonal axes of
heterogeneity are well-documented in cross-cultural psychology:

  (a) Generational axis: young / middle / older (WVS-grounded)
  (b) Urban–rural axis: cosmopolitan city-dwellers vs. rural traditionalists

Crossing these axes yields 6 empirically motivated agents plus the standard
utilitarian anchor and a "global-citizen" metacognitive agent → 8 total.

**Key changes** vs. EXP-01 (kaggle_experiment.py):
  - `ExpandedPersonaController`: subclass of `PaperSWAController` that builds
    8 personas per country (3 age × 2 urban/rural + utilitarian + global)
  - Urban/rural modifier strings applied on top of WVS base persona
  - `lambda_coop` raised to 0.75 (more consensus weight with 8 agents)
  - `K_samples` raised to 256 (denser IS grid for wider agent space)

**Novel contribution**: Empirically tests whether finer-grained demographic
resolution in the persona pool improves AMCE-alignment over the 4-agent
baseline. Directly motivated by WVS-7 urban/rural respondent stratification.

Expected outcome: ≥ +5% additional MIS reduction vs. EXP-01 on Confucian
(CHN/JPN) where urban-rural value gaps are largest (WVS data: Δreligiosity
CHN-rural vs CHN-urban ≈ 0.4 norm units).

Usage on Kaggle
---------------
Option A: Upload this file and run:
    !python exp02_expanded_personas.py

Option B: Clone repo first:
    !git clone https://github.com/trungkiet2005/cultural_alignment.git
    %cd cultural_alignment
    !python experiment/exp02_expanded_personas.py
"""

# ============================================================================
# Step 0: env vars MUST be set before any torch import
# ============================================================================
import os
import sys
import subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

# ============================================================================
# Step 1: bootstrap (Kaggle only)
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
    cmds = [
        "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet --no-deps --force-reinstall pyarrow",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]
    for c in cmds:
        subprocess.run(c, shell=True, check=False)


_REPO_DIR = _ensure_repo()
_install_deps()

# ============================================================================
# Step 2: imports
# ============================================================================
import gc
import shutil
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

from src.config import BaselineConfig, SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG, COUNTRY_FULL_NAMES
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import (
    build_country_personas, load_wvs_profiles,
    generate_wvs_persona, SUPPORTED_COUNTRIES
)
from src.controller import ImplicitSWAController
from src.persona_i18n import COUNTRY_NATIVE_NAME, UTILITARIAN_PERSONA_I18N
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment
from src.baseline_runner import run_baseline_vanilla

# ============================================================================
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-02"
EXP_NAME = "expanded_personas_8agent"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42
SKIP_BASELINE: bool = True   # Baseline from EXP-01 is already done

BASE_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/baseline"
SWA_ROOT:  str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT:  str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: Expanded Persona Builder
# ============================================================================

# Urban/rural modifier phrases per language (applied to WVS-base persona text).
# Each entry: (urban_prefix, rural_prefix)
_URBAN_RURAL: Dict[str, tuple] = {
    "en":    ("You are a city-dwelling, cosmopolitan",
              "You are a rural, traditional"),
    "zh":    ("你是一位居住在大城市、具有国际视野的",
              "你是一位来自农村地区、保有传统价值观的"),
    "ja":    ("あなたは都市部に住む、コスモポリタンな視野を持つ",
              "あなたは農村部に暮らす、伝統的な価値観を持つ"),
    "de":    ("Du bist ein kosmopolitischer Stadtbewohner, der",
              "Du bist ein traditioneller Landbewohner, der"),
    "pt":    ("Você é um morador cosmopolita da cidade que",
              "Você é um morador rural e tradicional que"),
    "ko":    ("당신은 대도시에 사는 코즈모폴리탄적",
              "당신은 농촌 지역에 사는 전통적인"),
    "fr":    ("Vous êtes un habitant cosmopolite de la ville qui",
              "Vous êtes un habitant rural et traditionnel qui"),
    "ru":    ("Вы — житель города с космополитическими взглядами,",
              "Вы — сельский житель с традиционными ценностями,"),
    "hi":    ("आप एक महानगरीय शहरी निवासी हैं जो",
              "आप एक ग्रामीण, परंपरावादी निवासी हैं जो"),
    "ar":    ("أنت ساكن مدينة كوزموبوليتاني",
              "أنت ساكن ريفي تقليدي"),
    "vi":    ("Bạn là một cư dân thành thị, mang tư tưởng quốc tế và",
              "Bạn là một cư dân nông thôn, giữ gìn truyền thống và"),
    "es":    ("Eres un habitante urbano cosmopolita que",
              "Eres un habitante rural y tradicional que"),
    "tr":    ("Kent yaşamlı, kozmopolit bir bireysin ve",
              "Kırsal kesimde yaşayan, geleneksel değerlere sahip birisisin ve"),
    "id":    ("Anda adalah warga kota kosmopolit yang",
              "Anda adalah warga pedesaan dengan nilai-nilai tradisional yang"),
    "uk":    ("Ти міський житель із космополітичними поглядами,",
              "Ти сільський житель із традиційними цінностями,"),
}

# Global-citizen metacognitive agent string (speaks English regardless of country)
_GLOBAL_CITIZEN_AGENT = (
    "You are a globally-minded ethicist who has studied moral philosophy across "
    "cultures. You approach moral dilemmas by first asking: 'What solution would "
    "a reasonable, impartial person from any culture endorse?' You weigh both "
    "utilitarian outcomes and deontological constraints, and you are especially "
    "sensitive to how cultural context changes the moral weight of each option."
)


def build_expanded_personas(country_iso: str, wvs_path: str = "") -> List[str]:
    """
    Build 8 culturally-grounded personas per country:
      P1–P3: WVS young/middle/older base personas (same as EXP-01)
      P4–P6: Urban/rural modulated versions of P1-P3
      P7:    Utilitarian anchor (same as EXP-01)
      P8:    Global-citizen metacognitive agent

    If WVS data is unavailable, falls back to BASE_PERSONAS ×2 + utilitarian + global.
    """
    lang = COUNTRY_LANG.get(country_iso, "en")
    country_name = COUNTRY_FULL_NAMES.get(country_iso, country_iso)
    native_country = COUNTRY_NATIVE_NAME.get(country_iso, country_name)
    urban_prefix, rural_prefix = _URBAN_RURAL.get(lang, _URBAN_RURAL["en"])

    # -- Try WVS-grounded base personas --
    base_personas = build_country_personas(country_iso, wvs_path=wvs_path)
    # base_personas = [P_young, P_middle, P_older, P_utilitarian]

    p_young, p_middle, p_older = base_personas[0], base_personas[1], base_personas[2]
    p_utilitarian = base_personas[3]

    # Modulate young and older with urban prefix, middle with rural prefix
    # (urban younger vs rural older = largest empirical gap in WVS-7)
    p_urban_young  = f"{urban_prefix} {p_young}"
    p_rural_young  = f"{rural_prefix} {p_young}"
    p_urban_older  = f"{urban_prefix} {p_older}"

    personas_8 = [
        p_young,         # P1: WVS young base
        p_middle,        # P2: WVS middle base
        p_older,         # P3: WVS older base
        p_urban_young,   # P4: urban-cosmopolitan young
        p_rural_young,   # P5: rural-traditional young
        p_urban_older,   # P6: urban-cosmopolitan older
        p_utilitarian,   # P7: utilitarian anchor
        _GLOBAL_CITIZEN_AGENT,  # P8: global-citizen meta-agent
    ]
    print(f"[EXP-02] Built {len(personas_8)} personas for {country_iso} (lang={lang})")
    return personas_8


# ============================================================================
# Step 5: EXP-02 SWA Controller — 8 agents, higher K, elevated lambda_coop
# ============================================================================
class Exp02SWAController(ImplicitSWAController):
    """
    8-agent variant of SWA-PTIS.

    Changes vs. PaperSWAController (EXP-01):
      - N=8 persona agents (3 WVS base + 3 urban/rural modulated + 1 utilitarian + 1 global)
      - lambda_coop=0.75 (more weight on consensus; justified by larger N)
      - K_samples=256 (denser IS proposal grid)
      - PT parameters unchanged (alpha=0.88, beta=0.88, kappa=2.25)
    """

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
        # Pass 1
        db1, da1, logit_temp = self._extract_logit_gaps(
            user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        # Linear debias
        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # Adaptive sigma
        sigma = float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else self.noise_std
        sigma = max(sigma, self.noise_std)

        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        v_per_agent = self._pt_value(g_per_agent)
        mean_v      = v_per_agent.mean(dim=1)

        g_cons  = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons  = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        delta_opt = anchor + delta_star
        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0,
            "sigma_used": float(sigma),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "n_agents": len(self.personas),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp02SWAController


# ============================================================================
# Step 6: runner helpers
# ============================================================================
def _dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total / 1e9


def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    roots = [
        os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface",
    ]
    seen = set()
    for root in roots:
        if not root:
            continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target = os.path.join(hub_dir, safe)
        if target in seen or not os.path.isdir(target):
            continue
        seen.add(target)
        try:
            size_gb = _dir_size_gb(target)
            shutil.rmtree(target)
            print(f"[CLEANUP] removed {target}  ({size_gb:.2f} GB freed)")
        except Exception as e:
            print(f"[CLEANUP] failed to remove {target}: {e}")


def _build_swa_config(model_name: str) -> SWAConfig:
    cfg = SWAConfig(
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
        lambda_coop=0.75,     # elevated for 8-agent pool
        K_samples=256,        # denser IS grid
    )
    return cfg


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
    print(f"\n{'#'*70}\n# {EXP_ID} SWA-8agent [{model_name}] -> {out_dir}\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)
        personas = build_expanded_personas(country, wvs_path=WVS_DATA_PATH)
        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        rows.append({
            "model":   model_name,
            "method":  f"{EXP_ID}_8agent_swa",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":        summary["flip_rate"],
            "mean_latency_ms":  summary["mean_latency_ms"],
            "n_scenarios":      summary["n_scenarios"],
            "n_personas":       8,
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 7: main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (BASE_ROOT, SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] models   : {MODELS}")
    print(f"[CONFIG] countries: {TARGET_COUNTRIES}")
    print(f"[CONFIG] n_scenarios: {N_SCENARIOS}")
    print(f"[CONFIG] 8-agent expanded persona pool | lambda_coop=0.75 | K=256")

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

        cmp_path = Path(CMP_ROOT) / "comparison.csv"
        pd.DataFrame(all_rows).to_csv(cmp_path, index=False)
        print(f"[SAVE] partial comparison -> {cmp_path}  ({len(all_rows)} rows)")

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")
    print(cmp_df.to_string())


if __name__ == "__main__":
    main()
