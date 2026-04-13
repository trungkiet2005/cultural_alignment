#!/usr/bin/env python3
"""
EXP-24 — BLEnD Short-Answer QA with SWA-PTIS Cultural Steering (DPBR)
======================================================================
Single-file Kaggle-ready script combining:
  1. BLEnD benchmark evaluation (NeurIPS 2024 Datasets & Benchmarks)
  2. SWA-PTIS candidate-level cultural steering (our method)
  3. EXP-24 Dual-Pass Bootstrap IS Reliability weighting

Model: Phi-4 14B (BF16 full precision, SDPA attention, ~28GB on H100 80GB)
Requires raw logit access for candidate scoring → HF native transformers

Algorithm — SWA-PTIS adapted for QA (NOT MPPI):
  The core Moral Machine method operates on SCALAR logit gaps (delta) between
  A/B binary choices. For M-candidate QA, we reduce to the scalar case:
    1. Generate M diverse candidates via persona-steered decoding
    2. Score candidates under N cultural agents (log-prob matrix L[N+1, M])
    3. Identify top-2 consensus candidates → pairwise scalar delta
    4. Run PTIS IS update (exactly as ImplicitSWAController._is_solve_decision):
       - eps ~ N(0, sigma²), one-step IS
       - Per-agent PT utility v(g_i/sigma) BEFORE averaging (preserves loss aversion)
       - ESS collapse guard (rho_eff threshold)
    5. EXP-24 dual-pass bootstrap: 2 × K_HALF independent IS passes
       - Reliability: r = exp(-(δ*₁-δ*₂)² / VAR_SCALE)
       - δ* = r · (δ*₁+δ*₂)/2
    6. ESS-adaptive anchor blend + hierarchical country prior (EMA)
    7. sigmoid(delta_opt / T_decision) → probability of top-1 over top-2

Pipeline:
  Step 1 — Load Phi-4 14B (BF16, SDPA, H100 80GB)
  Step 2 — For each (country, language, prompt):
            a) Generate M candidate answers via persona-steered decoding
            b) Score candidates under N cultural agents (log-prob matrix)
            c) Reduce to scalar delta (top-2 pairwise gap)
            d) Dual-pass PTIS + DPBR reliability → candidate selection
  Step 3 — Evaluate with Soft Exact Match (SEM-B & SEM-W)

H100 optimizations:
  - BF16 full precision (14B model ~28GB, well within H100 80GB)
  - SDPA attention (PyTorch built-in, dispatches FA kernel on H100 via cuDNN)
  - Batched inference for candidate generation + scoring
  - Language-tool lazy install (only installs what's needed per language)
"""

import os
import sys
import subprocess
import gc
import csv
import json
import re
import signal
import time
import warnings
import unicodedata as ud
from string import punctuation
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================
_ON_KAGGLE = os.path.exists("/kaggle/working")


def _run(cmd: str, verbose: bool = False) -> int:
    """Run a shell command, streaming stdout in real-time when verbose=True."""
    if verbose:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line)
        proc.wait()
        return proc.returncode
    else:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0 and r.stderr:
            print(r.stderr.strip())
        return r.returncode


if _ON_KAGGLE:
    print("[SETUP] Installing dependencies...")

    # -----------------------------------------------------------------------
    # Install order matters — transformers ↔ huggingface_hub version coupling
    # is tight.  Upgrade them TOGETHER so `is_offline_mode` etc. exist.
    # -----------------------------------------------------------------------
    print("[SETUP] (1/8) transformers + huggingface_hub (pinned compatible) ...")
    _run("pip install -q --upgrade 'transformers>=4.46,<4.52' 'huggingface_hub>=0.26,<0.30'", verbose=True)
    for _mod in list(sys.modules):
        if _mod.startswith(("huggingface_hub", "transformers")):
            del sys.modules[_mod]

    print("[SETUP] (2/8) numpy / protobuf / grpcio ...")
    _run('pip install -q "numpy<2.3"', verbose=True)
    _run("pip uninstall -y -q tensorflow tensorflow-cpu tf_keras 2>/dev/null || true")
    _run('pip install -q --upgrade "protobuf>=5.29.6,<6" "grpcio>=1.68" "googleapis-common-protos>=1.66"', verbose=True)

    print("[SETUP] (3/8) scipy tqdm sentencepiece accelerate ...")
    _run("pip install -q scipy tqdm sentencepiece accelerate", verbose=True)

    # (transformers already installed in step 1)
    print("[SETUP] (4/8) transformers — already installed")

    print("[SETUP] (5/8) datasets ...")
    _run('pip install -q "datasets>=3.4.1,<4.4.0"', verbose=True)

    print("[SETUP] (6/8) spacy + en_core_web_sm ...")
    _run("pip install -q spacy", verbose=True)
    _run("python -m spacy download en_core_web_sm", verbose=True)

    print("[SETUP] (7/8) matplotlib seaborn ...")
    _run("pip install -q matplotlib seaborn", verbose=True)

    print("[SETUP] (8/8) konlpy jieba hazm qalsadi nlp-id hausastemmer ...")
    _run("pip install -q konlpy jieba hazm qalsadi nlp-id hausastemmer", verbose=True)

    # Flush stale module caches after all installs (hazm etc. may have touched them)
    for _mod in list(sys.modules):
        if _mod.startswith(("huggingface_hub", "transformers")):
            del sys.modules[_mod]

    print("[SETUP] Installation complete.")


def _ensure_lang_tools(language: str):
    """Install heavy language tools on-demand (not at startup)."""
    if language == "Greek":
        try:
            import cltk  # noqa: F401
        except ImportError:
            print(f"[SETUP] Installing cltk for {language}...")
            _run("pip install -q cltk")

    elif language in ("Spanish", "Amharic"):
        try:
            import sparknlp  # noqa: F401
        except ImportError:
            print(f"[SETUP] Installing spark-nlp for {language} (may take a few minutes)...")
            _run("pip install -q 'pyspark==3.5.3' 'spark-nlp==5.3.3'")

    elif language == "Azerbaijani":
        repo = "/kaggle/working/stemmer" if _ON_KAGGLE else "stemmer"
        if not os.path.exists(repo):
            print(f"[SETUP] Cloning Azerbaijani stemmer...")
            base = "/kaggle/working" if _ON_KAGGLE else "."
            _run(f"cd {base} && git clone https://github.com/aznlp-disc/stemmer.git")

    elif language == "Assamese":
        lib = "/kaggle/working/indic_nlp_library" if _ON_KAGGLE else "indic_nlp_library"
        if not os.path.exists(lib):
            print(f"[SETUP] Cloning indic_nlp_library for {language}...")
            base = "/kaggle/working" if _ON_KAGGLE else "."
            _run(f"cd {base} && git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git")
        if lib not in sys.path:
            sys.path.insert(0, lib)
        res_dir = "/kaggle/working/indic_nlp_resources" if _ON_KAGGLE else "indic_nlp_resources"
        if not os.path.exists(res_dir):
            print(f"[SETUP] Cloning indic_nlp_resources for {language}...")
            base = "/kaggle/working" if _ON_KAGGLE else "."
            _run(f"cd {base} && git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git")

    elif language == "Sundanese":
        repo = "/kaggle/working/SUSTEM" if _ON_KAGGLE else "SUSTEM"
        if not os.path.exists(repo):
            print(f"[SETUP] Cloning SUSTEM for {language}...")
            base = "/kaggle/working" if _ON_KAGGLE else "."
            _run(f"cd {base} && git clone https://github.com/andhikaprima/SUSTEM.git")


# Phi-4 uses PyTorch built-in SDPA (dispatches FA kernel on H100 via cuDNN)
os.environ.setdefault("HF_ATTN_IMPLEMENTATION", "sdpa")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Performance knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ============================================================================
# 1. CONFIG
# ============================================================================

# ---------------------------------------------------------------------------
# EXP-24 toggle — set False for vanilla greedy baseline
# ---------------------------------------------------------------------------
USE_EXP24 = True

# WVS data path for persona construction (set "" to use fallback personas)
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    if _ON_KAGGLE else ""
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if _ON_KAGGLE:
    _INPUT_DATA = "/kaggle/input/blend-data/data"
    _WORKING_DATA = "/kaggle/working/blend_data/data"
    if os.path.exists(os.path.join(_INPUT_DATA, "questions")):
        DATA_ROOT = _INPUT_DATA
    else:
        DATA_ROOT = _WORKING_DATA
    WORK_DIR = Path("/kaggle/working/blend_results")
else:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "blend_results"


def _download_blend_from_hf():
    """Download BLEnD dataset from HuggingFace into a writable directory."""
    print("[DATA] Data not found. Downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        dl_dir = os.path.dirname(DATA_ROOT)
        os.makedirs(dl_dir, exist_ok=True)
        snapshot_download(
            repo_id="nayeon212/BLEnD",
            repo_type="dataset",
            local_dir=dl_dir,
            allow_patterns=["data/**"],
        )
        if os.path.exists(os.path.join(DATA_ROOT, "questions")):
            print("[DATA] Download complete.")
        else:
            if os.path.exists(os.path.join(dl_dir, "questions")):
                import shutil
                os.makedirs(DATA_ROOT, exist_ok=True)
                for sub in ["questions", "annotations", "prompts"]:
                    src = os.path.join(dl_dir, sub)
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(DATA_ROOT, sub))
                print("[DATA] Download complete (reorganized).")
            else:
                raise FileNotFoundError("Downloaded but expected directories not found")
    except ImportError:
        print("[DATA] huggingface_hub not installed. Run: pip install huggingface_hub")
        raise SystemExit(1)
    except Exception as e:
        print(f"[DATA] Download failed: {e}")
        print("[DATA] Please add BLEnD as a Kaggle dataset input, or run:")
        print("       git clone https://huggingface.co/datasets/nayeon212/BLEnD")
        raise SystemExit(1)


# Auto-download if data is missing
if not os.path.exists(os.path.join(DATA_ROOT, "questions")):
    _download_blend_from_hf()

QUESTIONS_DIR = os.path.join(DATA_ROOT, "questions")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
PROMPTS_DIR = os.path.join(DATA_ROOT, "prompts")

RESULTS_DIR = WORK_DIR / "model_inference_results"
EVAL_DIR = WORK_DIR / "evaluation_results"
for d in [RESULTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model — Phi-4 14B (BF16 full precision, ~28GB VRAM on H100)
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/phi-4"
MODEL_NAME = "Phi-4-14B"
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0
LOAD_TIMEOUT_MINUTES = int(os.environ.get("BLEND_LOAD_TIMEOUT_MINUTES", "15"))

# ---------------------------------------------------------------------------
# Countries & languages — exact 16 from the paper
# ---------------------------------------------------------------------------
COUNTRY_LANG: Dict[str, str] = {
    # "UK": "English", 
    # "US": "English",
    "South_Korea": "Korean",
    "Algeria": "Arabic",
    "China": "Chinese", 
    "Indonesia": "Indonesian",
    "Spain": "Spanish", 
    "Iran": "Persian", 
    "Mexico": "Spanish",
    "Assam": "Assamese", 
    "Greece": "Greek", 
    "Ethiopia": "Amharic",
    "Northern_Nigeria": "Hausa", 
    "Azerbaijan": "Azerbaijani",
    "North_Korea": "Korean", 
    "West_Java": "Sundanese",
}

PROMPT_NOS: List[str] = ["inst-4"]


# ============================================================================
# 2. EXP-24 SWA-PTIS ENGINE (DPBR) — adapted for QA
#
#    Mirrors src/controller.py::ImplicitSWAController (SWA-PTIS)
#    + experiment_DM/exp24_dpbr_core.py::Exp24DualPassController (DPBR)
#
#    NOT MPPI — this is Prospect-Theory Importance Sampling:
#      - IS in 1D scalar space (pairwise logit gap between top-2 candidates)
#      - PT value function applied PER-AGENT before averaging
#      - ESS collapse guard (K_eff/K < rho_eff → delta_star = 0)
#      - Self-attenuating at consensus (no correction when agents agree)
# ============================================================================

# ---------------------------------------------------------------------------
# EXP-24 Config
# ---------------------------------------------------------------------------
# EXP-09 hierarchical-prior hyperparameters (unchanged in EXP-24)
N_WARMUP = 50
DECAY_TAU = 100
BETA_EMA = 0.10

# EXP-24 dual-pass IS — override via env before importing
K_HALF = int(os.environ.get("EXP24_K_HALF", "64"))   # samples per pass (2 × K_HALF total)
VAR_SCALE = float(os.environ.get("EXP24_VAR_SCALE", "0.04"))  # r = exp(-bvar / s)


@dataclass
class EXP24QAConfig:
    """Hyperparameters for EXP-24 SWA-PTIS Cultural QA with DPBR."""
    # SWA-PTIS Core (matches ImplicitSWAController defaults)
    lambda_coop: float = 0.70       # balance private vs social reward
    rho_eff: float = 0.10           # K_eff/K threshold for IS collapse guard
    K_half: int = K_HALF            # samples per IS pass (2 × K_half = total K)
    noise_std: float = 0.3          # Gaussian perturbation floor std
    temperature: float = 0.5        # IS softmax temperature (eta/beta)
    logit_temperature: float = 1.0  # candidate scoring softmax temperature
    decision_temperature: float = 1.0  # sigmoid temperature for final selection

    # Prospect Theory (Kahneman & Tversky, 1979) — same as src/controller.py
    pt_alpha: float = 0.88          # gain curvature
    pt_beta: float = 0.88           # loss curvature
    pt_kappa: float = 2.25          # loss aversion coefficient

    # EXP-24 Dual-Pass Bootstrap IS Reliability
    var_scale: float = VAR_SCALE    # r = exp(-(δ*₁-δ*₂)² / var_scale)

    # ESS-adaptive anchor blend (EXP-05 / paper §Limitations)
    use_ess_anchor_reg: bool = True

    # Candidate generation
    M_candidates: int = 8           # target diverse candidates
    max_new_tokens: int = 64        # max tokens per candidate
    gen_temperature: float = 0.7    # temperature for diversity sampling

    # WVS data path
    wvs_data_path: str = ""

    # BLEnD country → WVS ISO mapping
    BLEND_TO_WVS: Dict[str, str] = field(default_factory=lambda: {
        "UK": "GBR", "US": "USA", "South_Korea": "KOR", "China": "CHN",
        "Mexico": "MEX", "Northern_Nigeria": "NGA", "Azerbaijan": "AZE",
        "Algeria": "DZA", "Indonesia": "IDN", "Spain": "ESP", "Iran": "IRN",
        "Assam": "IND", "Greece": "GRC", "Ethiopia": "ETH",
        "North_Korea": "KOR", "West_Java": "IDN",
    })

    BLEND_COUNTRY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        "UK": "the United Kingdom", "US": "the United States",
        "South_Korea": "South Korea", "Algeria": "Algeria",
        "China": "China", "Indonesia": "Indonesia",
        "Spain": "Spain", "Iran": "Iran", "Mexico": "Mexico",
        "Assam": "Assam, India", "Greece": "Greece",
        "Ethiopia": "Ethiopia", "Northern_Nigeria": "Northern Nigeria",
        "Azerbaijan": "Azerbaijan", "North_Korea": "North Korea",
        "West_Java": "West Java, Indonesia",
    })


# ---------------------------------------------------------------------------
# Hierarchical Country Prior (EXP-09, unchanged in EXP-24)
# ---------------------------------------------------------------------------
class BootstrapPriorState:
    """Minimal EXP-09 country prior (scalar EMA + annealed blend)."""

    def __init__(self) -> None:
        self.delta_country = 0.0
        self.step = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < N_WARMUP:
            return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


PRIOR_STATE: Dict[str, BootstrapPriorState] = {}


def dpbr_reliability_weight(delta_star_1: float, delta_star_2: float,
                            var_scale: float = VAR_SCALE) -> float:
    """Paper-facing: r = exp(-(δ*₁-δ*₂)² / s)."""
    bv = (delta_star_1 - delta_star_2) ** 2
    return float(np.exp(-bv / var_scale))


def ess_anchor_blend_alpha(ess_min: float, rho_eff: float) -> float:
    """α = clip(ess_min, ρ, 1). See exp24_dpbr_core.py."""
    return float(min(1.0, max(float(rho_eff), float(ess_min))))


# ---------------------------------------------------------------------------
# WVS-Based Cultural Persona Generation
# ---------------------------------------------------------------------------
_WVS_DIMS = {
    "gender_equality": (["Q58P", "Q59P", "Q60P"], "gender egalitarianism"),
    "religion":        (["Q6P"],                   "religious importance"),
    "trust":           (["Q43P"],                  "interpersonal trust"),
    "moral_permissiveness": (["Q50", "Q52P", "Q54P"], "moral permissiveness"),
    "work_importance": (["Q5P"],                   "work centrality"),
    "family":          (["Q1P"],                   "family importance"),
    "autonomy":        (["Q39P"],                  "personal autonomy"),
    "meritocracy":     (["Q40P"],                  "meritocratic orientation"),
}

_WVS_PROFILES_CACHE: Dict[str, Dict] = {}


def _load_wvs_profiles(wvs_csv_path: str, target_countries: List[str]) -> Dict[str, Dict]:
    """Load and compute WVS value profiles per country per age group."""
    global _WVS_PROFILES_CACHE
    if _WVS_PROFILES_CACHE:
        return _WVS_PROFILES_CACHE

    import csv as _csv

    all_vars = set()
    for vars_list, _ in _WVS_DIMS.values():
        all_vars.update(vars_list)
    all_vars.add("Q261")
    all_vars.add("A_YEAR")

    def _age_group(birth_year, survey_year):
        age = survey_year - birth_year
        if age < 36: return "young"
        if age < 56: return "middle"
        return "older"

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    try:
        with open(wvs_csv_path, "r") as f:
            reader = _csv.reader(f)
            header = next(reader)
            cidx = header.index("B_COUNTRY_ALPHA")
            var_idx = {v: header.index(v) for v in all_vars if v in header}

            for row in reader:
                country = row[cidx]
                if country not in target_countries:
                    continue
                try:
                    birth = float(row[var_idx["Q261"]])
                    syear = float(row[var_idx["A_YEAR"]])
                    if birth < 1900 or birth > 2010 or syear < 2015:
                        continue
                except (ValueError, KeyError):
                    continue
                ag = _age_group(birth, syear)

                for var in all_vars:
                    if var in ("Q261", "A_YEAR"):
                        continue
                    try:
                        val = float(row[var_idx[var]])
                        if val > 0:
                            data[country][ag][var].append(val)
                            data[country]["all"][var].append(val)
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        print(f"[WARN] WVS data not found: {wvs_csv_path}")
        return {}

    profiles = {}
    for c in target_countries:
        profiles[c] = {}
        for ag in ["young", "middle", "older", "all"]:
            dim_means = {}
            for dim_name, (vars_list, _) in _WVS_DIMS.items():
                vals = []
                for v in vars_list:
                    vals.extend(data[c][ag][v])
                dim_means[dim_name] = round(sum(vals) / len(vals), 2) if vals else 0
            profiles[c][ag] = dim_means

    n_loaded = sum(1 for c in profiles if profiles[c].get("all", {}).get("religion", 0) > 0)
    print(f"[WVS] Loaded profiles for {n_loaded}/{len(target_countries)} countries")
    _WVS_PROFILES_CACHE = profiles
    return profiles


def _describe_value(dim_name: str, value: float, scale_max: float = 4.0) -> str:
    """Convert a WVS dimension mean into a natural language descriptor."""
    ratio = value / scale_max
    if dim_name == "religion":
        if ratio > 0.85: return "deeply religious"
        if ratio > 0.70: return "moderately religious"
        if ratio > 0.55: return "somewhat secular"
        return "highly secular"
    elif dim_name == "gender_equality":
        if ratio > 0.85: return "strongly gender-egalitarian"
        if ratio > 0.75: return "moderately gender-egalitarian"
        if ratio > 0.65: return "somewhat traditional on gender"
        return "traditional on gender roles"
    elif dim_name == "trust":
        if ratio > 0.55: return "high interpersonal trust"
        if ratio > 0.45: return "moderate trust"
        return "low interpersonal trust"
    elif dim_name == "moral_permissiveness":
        if value > 3.5: return "morally permissive"
        if value > 3.0: return "moderately permissive"
        if value > 2.5: return "morally conservative"
        return "morally strict"
    elif dim_name == "autonomy":
        if ratio > 0.90: return "strongly values personal autonomy"
        if ratio > 0.80: return "values personal autonomy"
        return "moderate on personal autonomy"
    elif dim_name == "meritocracy":
        if ratio > 0.95: return "strongly meritocratic"
        if ratio > 0.85: return "meritocratic"
        return "egalitarian on income"
    elif dim_name == "work_importance":
        if ratio > 0.90: return "work is central to identity"
        if ratio > 0.80: return "values work highly"
        return "moderate work orientation"
    elif dim_name == "family":
        return "family is paramount"
    return ""


def _generate_qa_persona(country_iso: str, age_group: str,
                         profile: Dict[str, float],
                         country_name: str) -> str:
    """Generate a single QA-adapted persona from WVS profile."""
    age_desc = {
        "young": ("young adult", "in your 20s-30s"),
        "middle": ("middle-aged adult", "in your 40s-50s"),
        "older": ("senior citizen", "over 60"),
        "all": ("citizen", ""),
    }
    role, age_range = age_desc.get(age_group, ("citizen", ""))

    traits = []
    for dim_name in ["religion", "gender_equality", "trust", "moral_permissiveness",
                     "autonomy", "meritocracy", "work_importance"]:
        val = profile.get(dim_name, 0)
        if val > 0:
            desc = _describe_value(dim_name, val)
            if desc:
                traits.append(desc)

    traits_str = ", ".join(traits[:5])

    return (
        f"You are a {role} from {country_name}"
        f"{' ' + age_range if age_range else ''}. "
        f"You have deep knowledge of everyday life, food, customs, holidays, "
        f"sports, education, and family traditions in {country_name}. "
        f"Based on the cultural values of your society, you are {traits_str}. "
        f"Answer questions from your authentic personal experience as someone "
        f"who grew up in {country_name}."
    )


# Fallback personas for countries without WVS data
_BASE_QA_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        "You are a young American from a coastal city. You have deep knowledge of American everyday life, food, holidays like Thanksgiving and 4th of July, sports like football and basketball, and school culture. Answer from your authentic personal experience.",
        "You are a middle-aged American from the Midwest. You know well about American BBQ traditions, county fairs, school cafeteria food, family gatherings, and local sports culture. Answer from your authentic experience.",
        "You are an elderly American who has lived in the US your entire life. You have decades of experience with American traditions, holidays, food culture, education system, and family customs. Answer from your authentic experience.",
        "You are an American cultural studies professor. You have expert knowledge of American everyday customs, regional food variations, holiday traditions, education system, sports culture, and family dynamics across different communities.",
    ],
    "GBR": [
        "You are a young British person from London. You know everyday British life well — pub culture, fish and chips, the Premier League, school uniforms, bank holidays, and tea time. Answer from your authentic experience.",
        "You are a middle-aged British person from Northern England. You have deep knowledge of British traditions, Sunday roasts, Guy Fawkes Night, cricket, the NHS, and school life. Answer from your authentic experience.",
        "You are an elderly British citizen who has lived in the UK your entire life. You know decades of British customs, food, sports, royal traditions, and everyday life. Answer from your authentic experience.",
        "You are a British cultural historian. You have expert knowledge of British everyday customs, food traditions, holidays, education system, sports, and family life across regions and classes.",
    ],
    "KOR": [
        "당신은 한국에서 나고 자란 젊은 한국인입니다. 한국의 일상생활, 음식문화, 명절, 학교생활, 스포츠 등에 대해 잘 알고 있습니다. 진실된 경험을 바탕으로 답변해 주세요.",
        "당신은 한국의 중년 직장인입니다. 한국의 직장문화, 가정생활, 전통 명절, 음식, 교육 시스템에 대해 깊은 지식을 가지고 있습니다. 경험을 바탕으로 답변해 주세요.",
        "당신은 한국의 어르신입니다. 수십 년간의 한국 생활 경험으로 전통 문화, 음식, 명절, 가족 관습에 대해 잘 알고 있습니다. 진실된 경험을 바탕으로 답변해 주세요.",
        "당신은 한국 문화를 연구하는 교수입니다. 한국의 일상, 음식, 명절, 교육, 스포츠, 가족문화에 대한 전문 지식을 가지고 있습니다.",
    ],
    "CHN": [
        "你是一位在中国长大的年轻人。你对中国的日常生活、饮食文化、传统节日、学校生活和运动非常了解。请根据你的真实经历回答问题。",
        "你是一位中国的中年人。你深入了解中国的工作文化、家庭生活、传统节日、美食和教育体系。请根据你的经验回答问题。",
        "你是一位中国的长者。你拥有数十年的中国生活经验，对传统文化、饮食、节日和家族习俗非常熟悉。请根据你的真实经历回答。",
        "你是一位中国文化研究教授。你对中国的日常习俗、饮食传统、节日、教育制度、体育和家庭文化有专业知识。",
    ],
    "IRN": [
        "You are a young Iranian who grew up in Iran. You know everyday Iranian life well — food like ghormeh sabzi and tahdig, Nowruz celebrations, school life, football culture, and family traditions. Answer from authentic experience.",
        "You are a middle-aged Iranian. You have deep knowledge of Iranian customs, cuisine, religious holidays, education system, and family gatherings. Answer from your authentic experience.",
        "You are an elderly Iranian who has lived in Iran your entire life. You know decades of Iranian traditions, food culture, Nowruz customs, and family life. Answer from your authentic experience.",
        "You are an Iranian cultural studies expert. You have deep knowledge of Persian customs, food traditions, holidays like Nowruz and Yalda, education, sports, and family dynamics.",
    ],
    "DZA": [
        "You are a young Algerian who grew up in Algeria. You know everyday Algerian life — couscous, Ramadan traditions, school life, football culture, and family gatherings. Answer from your authentic experience.",
        "You are a middle-aged Algerian. You have deep knowledge of Algerian customs, cuisine, religious practices, education system, and family traditions. Answer from your authentic experience.",
        "You are an elderly Algerian who has lived in Algeria your entire life. You know decades of Algerian traditions, food culture, customs, and family life. Answer from your authentic experience.",
        "You are an Algerian cultural studies expert. You have deep knowledge of Algerian customs, food traditions, holidays, education, sports, and family dynamics.",
    ],
}


def build_qa_personas(country: str, config: EXP24QAConfig) -> List[str]:
    """Build 4 cultural personas for a BLEnD country."""
    wvs_iso = config.BLEND_TO_WVS.get(country, "")
    country_name = config.BLEND_COUNTRY_NAMES.get(country, country.replace("_", " "))

    # Try WVS-based generation
    if config.wvs_data_path and os.path.exists(config.wvs_data_path) and wvs_iso:
        target_isos = list(set(config.BLEND_TO_WVS.values()))
        profiles = _load_wvs_profiles(config.wvs_data_path, target_isos)
        country_profile = profiles.get(wvs_iso, {})

        if country_profile and country_profile.get("all", {}).get("religion", 0) > 0:
            personas = []
            for ag in ["young", "middle", "older"]:
                p = country_profile.get(ag, country_profile["all"])
                if p.get("religion", 0) > 0:
                    personas.append(_generate_qa_persona(wvs_iso, ag, p, country_name))

            personas.append(
                f"You are a cultural studies expert specializing in {country_name}. "
                f"You have comprehensive knowledge of everyday customs, traditional "
                f"and modern food culture, holidays, education system, popular sports, "
                f"work life, and family traditions in {country_name}. "
                f"Answer questions with detailed cultural accuracy."
            )
            while len(personas) < 4:
                personas.append(_generate_qa_persona(
                    wvs_iso, "all", country_profile["all"], country_name
                ))
            print(f"[EXP24] WVS personas for {country} ({wvs_iso}): {len(personas[:4])}")
            return personas[:4]

    # Fallback: base personas by WVS ISO
    if wvs_iso in _BASE_QA_PERSONAS:
        return list(_BASE_QA_PERSONAS[wvs_iso])

    # Generic fallback
    return [
        f"You are a young adult from {country_name}. You have deep knowledge of "
        f"everyday life, food, customs, holidays, sports, and education in {country_name}. "
        f"Answer from your authentic personal experience.",

        f"You are a middle-aged person from {country_name}. You have extensive experience "
        f"with daily life, family traditions, work culture, and local customs in {country_name}. "
        f"Answer from your authentic personal experience.",

        f"You are an elderly person who has lived in {country_name} your entire life. "
        f"You have decades of experience with traditions, food culture, holidays, "
        f"and family customs. Answer from your authentic personal experience.",

        f"You are a cultural studies expert specializing in {country_name}. "
        f"You have comprehensive knowledge of everyday customs, food culture, holidays, "
        f"education, sports, and family traditions. Answer with cultural accuracy.",
    ]


# ---------------------------------------------------------------------------
# EXP24QAController — SWA-PTIS + Dual-Pass Bootstrap adapted for QA
#
# QA adaptation of the scalar PTIS from src/controller.py:
#   In Moral Machine: delta = logit(B) - logit(A) is a scalar per agent.
#   In QA: we reduce M candidates to a SCALAR by computing the pairwise
#   log-prob gap between the top-2 consensus candidates:
#     delta_base = L[base, j1] - L[base, j2]
#     delta_i    = L[persona_i, j1] - L[persona_i, j2]
#   Then the IS math is IDENTICAL to the binary case.
# ---------------------------------------------------------------------------
class EXP24QAController:
    """
    EXP-24 SWA-PTIS for Short-Answer QA with Dual-Pass Bootstrap IS Reliability.

    Pipeline per question:
      1. Generate M diverse candidates (persona-steered + temperature sampling)
      2. Score candidates under base + N personas → L[N+1, M] log-prob matrix
      3. Identify top-2 by cultural consensus → pairwise scalar delta
      4. Dual-pass IS (2 × K_HALF):
           Pass k: eps ~ N(0, sigma²), IS weights via PT utility
           → delta_star_k
      5. DPBR reliability: r = exp(-(δ*₁-δ*₂)² / VAR_SCALE)
         δ* = r · (δ*₁+δ*₂)/2
      6. ESS-adaptive anchor blend: δ_micro = α·anchor + (1-α)·δ_base + δ*
      7. Hierarchical country prior: δ_opt = α_h·δ_country + (1-α_h)·δ_micro
      8. sigmoid(δ_opt / T_decision) → p(choose j1 over j2)
    """

    def __init__(self, model, tokenizer, personas: List[str],
                 country: str,
                 config: Optional[EXP24QAConfig] = None):
        self.config = config or EXP24QAConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.personas = personas
        self.country = country
        self.N = len(personas)
        self.M = self.config.M_candidates
        self.device = next(model.parameters()).device

        # SWA-PTIS parameters (match ImplicitSWAController)
        self.lambda_coop = self.config.lambda_coop
        self.rho_eff = self.config.rho_eff
        self.K_half = self.config.K_half
        self.noise_std = self.config.noise_std
        self.beta = self.config.temperature          # IS softmax temperature (eta)
        self.logit_temp = self.config.logit_temperature
        self.decision_temperature = self.config.decision_temperature

        # Prospect Theory (same as src/controller.py)
        self.pt_alpha = self.config.pt_alpha
        self.pt_beta = self.config.pt_beta
        self.pt_kappa = self.config.pt_kappa

        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        self._build_persona_prefixes()

    def _get_prior(self) -> BootstrapPriorState:
        if self.country not in PRIOR_STATE:
            PRIOR_STATE[self.country] = BootstrapPriorState()
        return PRIOR_STATE[self.country]

    @torch.no_grad()
    def _build_persona_prefixes(self):
        """Tokenize persona system prompts for prefix-caching."""
        print(f"[EXP24] Building prefixes for {self.N} personas + 1 base...")
        t0 = time.time()

        self.persona_prefix_ids = []
        for persona_text in self.personas:
            prefix = f"<|system|>\n{persona_text}<|end|>\n<|user|>\n"
            ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
            self.persona_prefix_ids.append(ids)

        base_prefix = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n"
        self.base_prefix_ids = self.tokenizer(
            base_prefix, return_tensors="pt"
        ).input_ids.to(self.device)

        print(f"[EXP24] Prefix tokenisation: {time.time() - t0:.2f}s")

    # ── Phase 1: Candidate Generation ─────────────────────────────────
    @torch.no_grad()
    def _generate_single(self, prefix_ids: torch.Tensor, query_text: str,
                         do_sample: bool = False, temperature: float = 1.0) -> str:
        """Generate one answer given a prefix and query."""
        suffix = query_text + "<|end|>\n<|assistant|>\n"
        query_ids = self.tokenizer(suffix, return_tensors="pt").input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        input_ids = torch.cat([prefix_ids, query_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.pad_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        output_ids = self.model.generate(**gen_kwargs)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _generate_batched(self, query_text: str) -> List[str]:
        """Batched greedy: base + all personas in ONE forward pass."""
        suffix = query_text + "<|end|>\n<|assistant|>\n"
        query_ids = self.tokenizer(suffix, return_tensors="pt").input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        seqs = [torch.cat([p, query_ids], dim=1) for p in all_prefixes]
        max_len = max(s.shape[1] for s in seqs)

        batch_ids, batch_mask = [], []
        for s in seqs:
            pad_len = max_len - s.shape[1]
            batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
            batch_mask.append(F.pad(
                torch.ones(1, s.shape[1], dtype=torch.long, device=self.device),
                (pad_len, 0), value=0,
            ))

        batch_ids = torch.cat(batch_ids, dim=0)
        batch_mask = torch.cat(batch_mask, dim=0)

        output_ids = self.model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.pad_id,
            do_sample=False,
            use_cache=True,
        )

        results = []
        for i in range(len(all_prefixes)):
            new_tokens = output_ids[i][max_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(text)

        return results

    @torch.no_grad()
    def _generate_candidates(self, query_text: str) -> List[str]:
        """Generate M diverse candidates: batched greedy + optional sampling."""
        seen = set()
        candidates = []

        def _add(text: str):
            normalized = text.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(text.strip())

        for text in self._generate_batched(query_text):
            _add(text)

        max_extra = self.M * 2
        attempts = 0
        while len(candidates) < self.M and attempts < max_extra:
            _add(self._generate_single(
                self.base_prefix_ids, query_text,
                do_sample=True, temperature=self.config.gen_temperature
            ))
            attempts += 1

        return candidates

    # ── Phase 2: Candidate Scoring → Scalar Delta Extraction ──────────
    @torch.no_grad()
    def _score_candidates(self, query_text: str,
                          candidates: List[str]) -> torch.Tensor:
        """
        Score all candidates under base + N personas.
        Returns L[n_agents, M] mean log-prob matrix.
        """
        M = len(candidates)
        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids  
        n_agents = len(all_prefixes)

        suffix = query_text + "<|end|>\n<|assistant|>\n"
        query_ids = self.tokenizer(suffix, return_tensors="pt").input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        cand_ids_list = []
        for c in candidates:
            ids = self.tokenizer(
                c, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            cand_ids_list.append(ids)

        L = torch.zeros(n_agents, M, device=self.device)

        for i, prefix in enumerate(all_prefixes):
            seqs = []
            answer_starts = []
            answer_lens = []
            for j in range(M):
                seq = torch.cat([prefix, query_ids, cand_ids_list[j]], dim=1)
                seqs.append(seq)
                answer_starts.append(prefix.shape[1] + query_ids.shape[1])
                answer_lens.append(cand_ids_list[j].shape[1])

            max_len = max(s.shape[1] for s in seqs)

            batch_ids, batch_mask = [], []
            for s in seqs:
                pad_len = max_len - s.shape[1]
                batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
                batch_mask.append(F.pad(
                    torch.ones(1, s.shape[1], dtype=torch.long, device=self.device),
                    (pad_len, 0), value=0,
                ))

            batch_ids = torch.cat(batch_ids, dim=0)
            batch_mask = torch.cat(batch_mask, dim=0)

            outputs = self.model(
                input_ids=batch_ids, attention_mask=batch_mask, use_cache=False,
            )
            logits = outputs.logits

            for j in range(M):
                a_len = answer_lens[j]
                if a_len == 0:
                    L[i, j] = -100.0
                    continue
                pad_len = max_len - seqs[j].shape[1]
                a_start = pad_len + answer_starts[j]
                pred_logits = logits[j, a_start - 1: a_start + a_len - 1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                target = cand_ids_list[j][0]
                token_lp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                L[i, j] = token_lp.mean()

        return L

    def _extract_pairwise_deltas(self, L: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Reduce M-candidate log-prob matrix to scalar pairwise deltas.

        Identifies the top-2 candidates by cultural consensus (mean of persona
        softmax probs) and computes:
          delta_base = L[base, j1] - L[base, j2]     (scalar)
          delta_agents[i] = L[persona_i, j1] - L[persona_i, j2]  (N,)

        These are exactly analogous to the A/B logit gaps in Moral Machine.
        """
        L_agents = L[1:]   # (N, M)
        p_agents = F.softmax(L_agents / self.logit_temp, dim=1)
        p_consensus = p_agents.mean(dim=0)   # (M,)

        # Top-2 by consensus
        top2 = torch.topk(p_consensus, min(2, p_consensus.shape[0]))
        j1 = int(top2.indices[0].item())
        j2 = int(top2.indices[1].item()) if top2.indices.shape[0] > 1 else j1

        # Pairwise scalar deltas (exactly like logit(B) - logit(A))
        delta_base = L[0, j1] - L[0, j2]        # scalar
        delta_agents = L_agents[:, j1] - L_agents[:, j2]  # (N,)

        return delta_base, delta_agents, j1, j2

    # ── Phase 3: PTIS IS Update (mirrors _is_solve_decision exactly) ──
    def _prospect_value(self, x: torch.Tensor) -> torch.Tensor:
        """Prospect Theory value function (Kahneman & Tversky, 1979).
        v(x) =  x^α           if x ≥ 0
        v(x) = -κ · |x|^β     if x < 0
        """
        return torch.where(
            x >= 0,
            x.abs().pow(self.pt_alpha),
            -self.pt_kappa * x.abs().pow(self.pt_beta),
        )

    def _adaptive_noise_std(self, delta_agents: torch.Tensor) -> float:
        """Per-scenario proposal std, floored at self.noise_std.
        Matches ImplicitSWAController._adaptive_noise_std.
        """
        if delta_agents.numel() < 2:
            return self.noise_std
        std = float(delta_agents.std(unbiased=True).item())
        return max(std, self.noise_std)

    @torch.no_grad()
    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        K: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Single-pass PTIS update — mirrors Exp24DualPassController._single_is_pass
        and ImplicitSWAController._is_solve_decision exactly.

        Math (paper Eqs. 5-8):
          delta_tilde_k = anchor + eps_k,    eps_k ~ N(0, sigma²)
          g_{i,k}   = |delta_base - delta_i| - |delta_tilde_k - delta_i|
          g_cons_k  = |delta_base - anchor| - |delta_tilde_k - anchor|
          U(eps_k)  = (1-λ) · mean_i v(g_{i,k}/σ) + λ · v(g_cons_k/σ)
          w_k       = softmax(U / η)
          delta_star = Σ w_k · eps_k

        Returns (delta_star, ess_ratio).
        """
        device = self.device
        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps                                    # (K,)

        # Per-agent gain in logit-gap units
        dist_base_to_i = (delta_base - delta_agents).abs()            # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1)
                          - delta_agents.unsqueeze(0)).abs()          # (K, N)
        g_per_agent = dist_base_to_i.unsqueeze(0) - dist_cand_to_i    # (K, N)

        # Sigma-normalise then apply PT per-agent before averaging
        v_per_agent = self._prospect_value(g_per_agent / sigma)       # (K, N)
        mean_v = v_per_agent.mean(dim=1)                              # (K,)

        # Consensus-target gain
        g_cons = ((delta_base - anchor).abs()
                  - (delta_tilde - anchor).abs())                     # (K,)
        v_cons = self._prospect_value(g_cons / sigma)                 # (K,)

        # Collective utility (paper Eq. 7; no control-cost term)
        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons

        w = F.softmax(U / self.beta, dim=0)

        # ESS collapse guard
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_r = float(k_eff.item()) / K

        delta_star = (
            torch.sum(w * eps) if ess_r >= self.rho_eff
            else torch.zeros((), device=device)
        )
        return delta_star, ess_r

    # ── Main Prediction: Dual-Pass Bootstrap PTIS ─────────────────────
    @torch.no_grad()
    def predict_qa(self, query_text: str) -> Dict:
        """
        EXP-24 DPBR prediction for short-answer QA.

        Full pipeline:
          1. Generate candidates
          2. Score → log-prob matrix
          3. Extract pairwise deltas (top-2)
          4. Dual-pass IS (2 × K_HALF) → δ*₁, δ*₂
          5. DPBR reliability: r = exp(-(δ*₁-δ*₂)²/s)
          6. ESS anchor blend + hierarchical prior
          7. sigmoid → selection probability
        """
        candidates = self._generate_candidates(query_text)

        if len(candidates) == 0:
            return {"answer": "", "candidates": [], "selected_idx": -1,
                    "j1": -1, "j2": -1, "p_j1": 0.0,
                    "reliability_r": 0.0, "bootstrap_var": 0.0,
                    "delta_opt": 0.0, "ess_pass1": 0.0, "ess_pass2": 0.0,
                    "variance": 0.0, "sigma_used": 0.0,
                    "delta_country": 0.0, "alpha_h": 0.0}

        if len(candidates) == 1:
            return {"answer": candidates[0], "candidates": candidates,
                    "selected_idx": 0, "j1": 0, "j2": 0, "p_j1": 1.0,
                    "reliability_r": 1.0, "bootstrap_var": 0.0,
                    "delta_opt": 0.0, "ess_pass1": 1.0, "ess_pass2": 1.0,
                    "variance": 0.0, "sigma_used": 0.0,
                    "delta_country": 0.0, "alpha_h": 0.0}

        # Score candidates
        L = self._score_candidates(query_text, candidates)

        # Reduce to scalar deltas (top-2 pairwise gap)
        delta_base, delta_agents, j1, j2 = self._extract_pairwise_deltas(L)

        # Adaptive noise std (floored at self.noise_std)
        sigma = self._adaptive_noise_std(delta_agents)
        anchor = delta_agents.mean()

        # ── EXP-24 Dual-Pass Bootstrap IS ──
        ds1, ess1 = self._single_is_pass(
            delta_base, delta_agents, anchor, sigma, self.K_half)
        ds2, ess2 = self._single_is_pass(
            delta_base, delta_agents, anchor, sigma, self.K_half)

        # DPBR reliability weight
        bootstrap_var = float((ds1 - ds2).pow(2).item())
        reliability_r = dpbr_reliability_weight(
            float(ds1.item()), float(ds2.item()), self.config.var_scale)
        delta_star = reliability_r * (ds1 + ds2) / 2.0

        # ESS-adaptive anchor blend (EXP-05 / paper)
        ess_min = min(ess1, ess2)
        if self.config.use_ess_anchor_reg:
            alpha_reg = ess_anchor_blend_alpha(ess_min, self.rho_eff)
            delta_opt_micro = float(
                (alpha_reg * anchor + (1.0 - alpha_reg) * delta_base + delta_star).item()
            )
        else:
            alpha_reg = 1.0
            delta_opt_micro = float((anchor + delta_star).item())

        # Hierarchical country prior
        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        ps = prior.stats

        # Selection: sigmoid(delta_opt / T_decision)
        # delta > 0 → prefer j1 (top consensus candidate)
        p_j1 = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()

        selected_idx = j1 if p_j1 >= 0.5 else j2

        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "answer": candidates[selected_idx],
            "candidates": candidates,
            "selected_idx": selected_idx,
            "j1": j1,
            "j2": j2,
            "p_j1": p_j1,
            # DPBR diagnostics
            "reliability_r": reliability_r,
            "bootstrap_var": bootstrap_var,
            "delta_star_1": float(ds1.item()),
            "delta_star_2": float(ds2.item()),
            "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            # IS diagnostics
            "ess_pass1": ess1,
            "ess_pass2": ess2,
            "ess_anchor_alpha": alpha_reg,
            "variance": variance,
            "sigma_used": sigma,
            "delta_consensus": float(anchor.item()),
            # Prior diagnostics
            "delta_country": ps["delta_country"],
            "alpha_h": ps["alpha_h"],
            "prior_step": ps["step"],
            # Agent diagnostics
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "L_scores": L.tolist(),
        }


# ============================================================================
# 3. UTILITY FUNCTIONS (BLEnD)
# ============================================================================

def write_csv_row(values: list, filename: str):
    """Append a single row to a CSV file."""
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def replace_country_name(s: str, country: str) -> str:
    return s.replace("your country", country)


def load_questions(country: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(QUESTIONS_DIR, f"{country}_questions.csv"), encoding="utf-8")


def load_prompts(country: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROMPTS_DIR, f"{country}_prompts.csv"), encoding="utf-8")


def load_annotations(country: str) -> dict:
    with open(os.path.join(ANNOTATIONS_DIR, f"{country}_data.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(question: str, prompt_no: str, language: str,
                country: str, prompt_sheet: pd.DataFrame) -> str:
    row = prompt_sheet[prompt_sheet["id"] == prompt_no]
    if language == "English":
        template = row["English"].values[0]
    else:
        template = row["Translation"].values[0]
    return template.replace("{q}", question)


# ============================================================================
# 4. MODEL LOADING — HF Transformers (raw logit access for EXP-24)
# ============================================================================

MODEL_LOAD_TIMEOUT_SEC = 900


def _model_load_watchdog(timeout: int):
    """Daemon timer that aborts after timeout seconds."""
    import threading

    def _handler():
        print(
            f"\n[FATAL] Model load exceeded {timeout // 60} min ({timeout}s). "
            "Aborting to free Kaggle GPU resources.",
            flush=True,
        )
        os.abort()

    t = threading.Timer(timeout, _handler)
    t.daemon = True
    return t


def load_model():
    """
    Load Phi-4 14B via HF Transformers (BF16, SDPA attention).
    HF native required for raw logit access in SWA-PTIS candidate scoring.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_impl = os.environ.get("HF_ATTN_IMPLEMENTATION", "sdpa")
    print(f"[MODEL] Loading {MODEL_ID} (BF16, attn={attn_impl}) ...")

    wd = _model_load_watchdog(MODEL_LOAD_TIMEOUT_SEC)
    wd.start()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        )
    except Exception:
        wd.cancel()
        raise
    else:
        wd.cancel()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[MODEL] Loaded. GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB)")
    return model, tokenizer


class _LoadTimeout(Exception):
    pass


def _load_model_with_timeout(timeout_minutes: int = LOAD_TIMEOUT_MINUTES):
    """Load model with hard wall-clock timeout (SIGALRM on Linux, plain on Windows)."""
    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        print(f"[LOAD] SIGALRM unavailable on {sys.platform} — loading without timeout")
        return load_model()

    def _handler(signum, frame):
        raise _LoadTimeout(
            f"Model load exceeded {timeout_minutes} minute(s). "
            "Check VRAM availability or increase BLEND_LOAD_TIMEOUT_MINUTES."
        )

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_minutes * 60)
    print(f"[LOAD] Timeout set to {timeout_minutes} minute(s)")
    try:
        result = load_model()
        signal.alarm(0)
        return result
    except _LoadTimeout as exc:
        signal.alarm(0)
        print(f"\n[LOAD][ERROR] {exc}")
        if _ON_KAGGLE:
            raise SystemExit("[LOAD] Stopping Kaggle run to avoid wasting GPU.") from exc
        raise RuntimeError(str(exc)) from exc
    finally:
        signal.signal(signal.SIGALRM, prev_handler)


# ============================================================================
# 5. INFERENCE
# ============================================================================

def generate_response(model, tokenizer, prompt: str) -> str:
    """Vanilla greedy decoding (baseline)."""
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=None,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _init_exp24_controller(model, tokenizer, country: str):
    """Initialize EXP-24 PTIS controller for a given country."""
    config = EXP24QAConfig(wvs_data_path=WVS_DATA_PATH)
    personas = build_qa_personas(country, config)
    return EXP24QAController(
        model=model, tokenizer=tokenizer, personas=personas,
        country=country, config=config,
    )


def run_inference_for_country(
    model, tokenizer,
    country: str, language: str, prompt_no: str
) -> str:
    """
    Run inference on all questions for one (country, language, prompt) combo.
    Uses EXP-24 DPBR PTIS when USE_EXP24=True, vanilla greedy otherwise.
    """
    q_df = load_questions(country)
    prompt_sheet = load_prompts(country)

    local_lang = COUNTRY_LANG[country]
    q_col = "Question" if language == local_lang else "Translation"
    replace_country_flag = (language == "English" and local_lang != "English")

    method_tag = "EXP24-DPBR" if USE_EXP24 else MODEL_NAME
    output_filename = str(
        RESULTS_DIR / f"{method_tag}-{country}_{language}_{prompt_no}_result.csv"
    )

    # Resume support
    done_ids = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename, encoding="utf-8")
        done_ids = set(already["ID"])
        print(f"  Resuming: {len(done_ids)} already done")
    else:
        header = ["ID", q_col, "prompt", "response", "prompt_no"]
        if USE_EXP24:
            header += ["reliability_r", "bootstrap_var", "delta_opt",
                        "ess_pass1", "ess_pass2", "variance", "sigma_used",
                        "n_candidates", "selected_idx", "j1", "j2", "p_j1",
                        "delta_country", "alpha_h", "candidates"]
        write_csv_row(header, output_filename)

    # Initialize EXP-24 controller if needed
    exp24_controller = None
    if USE_EXP24:
        # Reset country prior for this country
        PRIOR_STATE[country] = BootstrapPriorState()
        exp24_controller = _init_exp24_controller(model, tokenizer, country)

    pb = tqdm(q_df.iterrows(), desc=f"{country}/{language}/{prompt_no}",
              total=len(q_df))

    for _, row in pb:
        qid = row["ID"]
        question = row[q_col]
        pb.set_postfix({"ID": qid})

        if qid in done_ids:
            continue

        if replace_country_flag:
            question = replace_country_name(question, country.replace("_", " "))

        full_prompt = make_prompt(question, prompt_no, language, country, prompt_sheet)

        if USE_EXP24 and exp24_controller is not None:
            result = exp24_controller.predict_qa(full_prompt)
            response = result["answer"]
            extra = [
                f"{result['reliability_r']:.6f}",
                f"{result['bootstrap_var']:.6f}",
                f"{result['delta_opt']:.6f}",
                f"{result['ess_pass1']:.4f}",
                f"{result['ess_pass2']:.4f}",
                f"{result['variance']:.6f}",
                f"{result['sigma_used']:.6f}",
                len(result["candidates"]),
                result["selected_idx"],
                result["j1"],
                result["j2"],
                f"{result['p_j1']:.6f}",
                f"{result['delta_country']:.6f}",
                f"{result['alpha_h']:.6f}",
                json.dumps(result["candidates"], ensure_ascii=False),
            ]
            write_csv_row(
                [qid, question, full_prompt, response, prompt_no] + extra,
                output_filename,
            )
        else:
            response = generate_response(model, tokenizer, full_prompt)
            write_csv_row(
                [qid, question, full_prompt, response, prompt_no],
                output_filename,
            )

    return output_filename


# ============================================================================
# 6. EVALUATION — Soft Exact Match (SEM-B & SEM-W)
# ============================================================================

def delete_prompt_from_answer(text: str, prompt: str) -> str:
    """Remove prompt/prefix artifacts from LLM response."""
    text = text.replace(prompt, "").replace("：", ":").replace("、", ",")\
               .replace("，", ",").replace("。", ".").lower()
    prompt = prompt.replace("：", ":").replace("、", ",")\
                   .replace("，", ",").replace("。", ".").lower()
    match = re.findall(r"^(\w+:)\s", text)
    extracted = ""
    for m in match:
        if len(m) > len(extracted) and m.replace(":", "") in prompt:
            extracted = m
    if match:
        return text.replace(extracted, "").strip()
    return text.strip()


def get_llm_response_by_id(res_df: pd.DataFrame, qid: str,
                           id_col: str, r_col: str) -> Optional[str]:
    if qid not in set(res_df[id_col]):
        return None
    try:
        llm_response = res_df[res_df[id_col] == qid][r_col].values[-1]
        prompt = res_df[res_df[id_col] == qid]["prompt"].values[-1]
        llm_response = str(llm_response)
        prompt = str(prompt)
        llm_response = delete_prompt_from_answer(llm_response, prompt)
        llm_response = llm_response.strip(".").lower()
        return llm_response
    except Exception:
        return None


def _load_lemma_tools(language: str):
    """Load language-specific NLP pipeline for lemmatization."""
    _ensure_lang_tools(language)

    if language == "Spanish":
        import sparknlp
        from sparknlp.base import DocumentAssembler, Pipeline
        from sparknlp.annotator import Tokenizer, LemmatizerModel
        from sparknlp.base import LightPipeline
        spark = sparknlp.start()
        doc = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tok = Tokenizer().setInputCols(["document"]).setOutputCol("token")
        lem = LemmatizerModel.pretrained("lemma", "es")\
                             .setInputCols(["token"]).setOutputCol("lemma")
        pipeline = Pipeline(stages=[doc, tok, lem])
        return LightPipeline(pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

    elif language == "Amharic":
        import sparknlp
        from sparknlp.base import DocumentAssembler, Pipeline
        from sparknlp.annotator import Tokenizer, LemmatizerModel
        from sparknlp.base import LightPipeline
        spark = sparknlp.start()
        doc = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tok = Tokenizer().setInputCols(["document"]).setOutputCol("token")
        lem = LemmatizerModel.pretrained("lemma", "am")\
                             .setInputCols(["token"]).setOutputCol("lemma")
        pipeline = Pipeline(stages=[doc, tok, lem])
        return LightPipeline(pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

    elif language == "English":
        import spacy
        return spacy.load("en_core_web_sm")

    return None


def lemma_check(answer: str, llm_response: str,
                nlp_pipeline, language: str = "English") -> bool:
    """Language-aware answer matching with lemmatization."""
    if answer in llm_response:
        return True
    if answer.replace("-", " ") in llm_response:
        return True
    if answer.replace(" ", "-") in llm_response:
        return True

    try:
        if language == "Korean":
            from konlpy.tag import Okt
            okt = Okt()
            answer_tokens = okt.morphs(
                " ".join([w for w, p in okt.pos(answer) if p != "Josa"]),
                stem=True)
            llm_tokens = okt.morphs(
                " ".join([w for w, p in okt.pos(llm_response) if p != "Josa"]),
                stem=True)

        elif language == "Hausa":
            import hausastemmer
            answer_tokens = [hausastemmer.stem(t.strip("-")) for t in answer.split()]
            llm_tokens = [hausastemmer.stem(t.strip("-")) for t in llm_response.split()]

        elif language == "Amharic":
            answer_tokens = [
                token.result if lemma.result.startswith("_") else lemma.result
                for token, lemma in zip(
                    nlp_pipeline.fullAnnotate(answer)[0]["lemma"],
                    nlp_pipeline.fullAnnotate(answer)[0]["token"])
            ]
            llm_tokens = [
                token.result if lemma.result.startswith("_") else lemma.result
                for token, lemma in zip(
                    nlp_pipeline.fullAnnotate(llm_response)[0]["lemma"],
                    nlp_pipeline.fullAnnotate(llm_response)[0]["token"])
            ]

        elif language == "Azerbaijani":
            from stemmer.stemmer import Stemmer as AZStemmer
            my_stemmer = AZStemmer()

            def stem_az(text):
                text = text.replace("\u0130", "I").replace("\u201c", "")\
                           .replace("\u201d", "").replace("\u2018", "")\
                           .replace('"', "")
                words = ["".join(c for c in w if c not in punctuation or c == "-")
                         for w in text.split()]
                return my_stemmer.stem_words(words)

            answer_tokens = stem_az(answer)
            llm_tokens = stem_az(llm_response)

        elif language == "Indonesian":
            from nlp_id.lemmatizer import Lemmatizer as IDLemmatizer
            lem = IDLemmatizer()
            answer_tokens = lem.lemmatize(answer).split()
            llm_tokens = lem.lemmatize(llm_response).split()

        elif language == "Persian":
            from hazm import Lemmatizer as PRLemmatizer
            lem = PRLemmatizer()
            answer_tokens = [lem.lemmatize(t) for t in answer.split()]
            llm_tokens = [lem.lemmatize(t) for t in llm_response.split()]

        elif language == "Arabic":
            from qalsadi.lemmatizer import Lemmatizer as ARLemmatizer
            lem = ARLemmatizer()
            answer_tokens = lem.lemmatize(answer)
            llm_tokens = lem.lemmatize(llm_response)

        elif language == "Greek":
            from cltk import NLP
            cltk_nlp = NLP(language="grc", suppress_banner=True)
            answer_tokens = cltk_nlp.analyze(text=answer).lemmata
            llm_tokens = cltk_nlp.analyze(text=llm_response).lemmata

        elif language == "Spanish":
            answer_tokens = [l.result for l in nlp_pipeline.fullAnnotate(answer)[0]["lemma"]]
            llm_tokens = [l.result for l in nlp_pipeline.fullAnnotate(llm_response)[0]["lemma"]]

        elif language == "Sundanese":
            from SUSTEM.SUSTEM_S import EcsStemmer
            stemmer = EcsStemmer()
            answer_tokens = [stemmer.stemmingProcess(w.replace("(", "").replace(")", ""))
                             for w in answer.split()]
            llm_tokens = [stemmer.stemmingProcess(w.replace("(", "").replace(")", ""))
                          for w in llm_response.split()]

        elif language == "English":
            answer_tokens = [t.lemma_ for t in nlp_pipeline(answer)]
            llm_tokens = [t.lemma_ for t in nlp_pipeline(llm_response)]

        elif language == "Chinese":
            import jieba
            answer_tokens = list(jieba.cut(answer))
            llm_tokens = list(jieba.cut(llm_response))

        elif language == "Assamese":
            from indicnlp import common as indic_common
            from indicnlp import loader as indic_loader
            from indicnlp.tokenize import indic_tokenize
            indic_resources = os.path.join(
                "/kaggle/working" if _ON_KAGGLE else ".", "indic_nlp_resources")
            indic_common.set_resources_path(indic_resources)
            indic_loader.load()
            answer_tokens = indic_tokenize.trivial_tokenize(answer)
            llm_tokens = indic_tokenize.trivial_tokenize(llm_response)

        else:
            answer_tokens = answer.split()
            llm_tokens = llm_response.split()

    except Exception as e:
        print(f"  [WARN] lemma_check failed for {language}: {e}")
        answer_tokens = answer.split()
        llm_tokens = llm_response.split()

    d = {ord("\N{COMBINING ACUTE ACCENT}"): None}
    answer_tokens = [ud.normalize("NFD", t).translate(d).lower()
                     for t in answer_tokens if t not in punctuation and t != ""]
    llm_tokens = [ud.normalize("NFD", t).translate(d).lower()
                  for t in llm_tokens if t not in punctuation and t != ""]

    return all(a in llm_tokens for a in answer_tokens)


def soft_exact_match(
    country: str, language: str,
    annotation_dict: dict, response_df: pd.DataFrame,
    id_col: str = "ID", r_col: str = "response",
    annotations_key: str = "annotations"
) -> Tuple[float, float, pd.DataFrame]:
    """Paper's SEM-B (binary) and SEM-W (weighted) metrics."""
    nlp_pipeline = _load_lemma_tools(language)

    import spacy
    en_lemmatizer = spacy.load("en_core_web_sm")

    binary_score = 0
    weight_score = 0
    valid_count = 0

    response_df["binary_score"] = None
    response_df["weight_score"] = None

    pb = tqdm(annotation_dict.items(), total=len(annotation_dict), desc="Evaluating")

    for qid, data in pb:
        pb.set_description(qid)

        idks = data.get("idks", {})
        if (idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3
                or idks.get("idk", 0) >= 5
                or len(data.get(annotations_key, [])) == 0):
            continue

        valid_count += 1

        llm_response = get_llm_response_by_id(response_df, qid, id_col, r_col)
        flag = False

        if llm_response and data.get(annotations_key):
            max_vote = data[annotations_key][0]["count"]

            for agg_ans in data[annotations_key]:
                if language != "English":
                    for a in agg_ans["answers"]:
                        if lemma_check(a, llm_response, nlp_pipeline, language):
                            binary_score += 1
                            weight_score += agg_ans["count"] / max_vote
                            flag = True
                            break
                if not flag:
                    for a in agg_ans["en_answers"]:
                        if lemma_check(a, llm_response, en_lemmatizer, "English"):
                            binary_score += 1
                            weight_score += agg_ans["count"] / max_vote
                            flag = True
                            break
                if flag:
                    break

        if flag:
            response_df.loc[response_df[id_col] == qid, "binary_score"] = 1
            response_df.loc[response_df[id_col] == qid, "weight_score"] = \
                agg_ans["count"] / max_vote
        else:
            response_df.loc[response_df[id_col] == qid, "binary_score"] = 0
            response_df.loc[response_df[id_col] == qid, "weight_score"] = 0

        if valid_count > 0:
            pb.set_postfix({
                "SEM-B": f"{binary_score / valid_count * 100:.1f}",
                "SEM-W": f"{weight_score / valid_count * 100:.1f}",
            })

    if valid_count == 0:
        return 0.0, 0.0, response_df

    sem_b = binary_score / valid_count * 100
    sem_w = weight_score / valid_count * 100

    return sem_b, sem_w, response_df


# ============================================================================
# 7. EVALUATION ORCHESTRATION
# ============================================================================

def evaluate_single(country: str, language: str, prompt_no: str,
                    result_file: str, eval_result_file: str):
    if not os.path.exists(result_file):
        return None, None

    res_df = pd.read_csv(result_file, encoding="utf-8")
    annotations = load_annotations(country)

    sem_b, sem_w, scored_df = soft_exact_match(
        country=country, language=language,
        annotation_dict=annotations, response_df=res_df,
        id_col="ID", r_col="response", annotations_key="annotations",
    )

    method_tag = "EXP24-DPBR" if USE_EXP24 else MODEL_NAME
    print(f"    >>> [{method_tag}] SEM-B: {sem_b:.2f}%  |  SEM-W: {sem_w:.2f}%")

    write_csv_row([method_tag, country, language, prompt_no, "SEM-B", sem_b],
                  eval_result_file)
    write_csv_row([method_tag, country, language, prompt_no, "SEM-W", sem_w],
                  eval_result_file)

    scored_df.to_csv(
        str(RESULTS_DIR / f"{method_tag}_{country}_{language}_{prompt_no}_response_score.csv"),
        index=False, encoding="utf-8",
    )

    # Print DPBR diagnostics if available
    if USE_EXP24 and "reliability_r" in res_df.columns:
        r_mean = res_df["reliability_r"].astype(float).mean()
        bv_mean = res_df["bootstrap_var"].astype(float).mean()
        print(f"    >>> [DPBR] reliability_r={r_mean:.4f}  "
              f"bootstrap_var={bv_mean:.6f}")

    return sem_b, sem_w


def run_all_inference_and_eval(model, tokenizer) -> pd.DataFrame:
    """Run inference + evaluation for each (country, language, prompt) combo."""
    eval_result_file = str(EVAL_DIR / "evaluation_results.csv")
    if not os.path.exists(eval_result_file):
        write_csv_row(
            ["model", "country", "language", "prompt_no", "eval_method", "score"],
            eval_result_file,
        )

    total_combos = []
    for country, local_lang in COUNTRY_LANG.items():
        for prompt_no in PROMPT_NOS:
            total_combos.append((country, local_lang, prompt_no))

    print(f"\n{'='*60}")
    print(f"INFERENCE + EVAL: {len(total_combos)} combos "
          f"({len(COUNTRY_LANG)} countries × {len(PROMPT_NOS)} prompts × languages)")
    print(f"{'='*60}\n")

    for i, (country, language, prompt_no) in enumerate(total_combos):
        print(f"\n[{i+1}/{len(total_combos)}] {country} / {language} / {prompt_no}")

        result_file = run_inference_for_country(
            model, tokenizer, country, language, prompt_no)

        evaluate_single(country, language, prompt_no, result_file, eval_result_file)

        gc.collect()
        torch.cuda.empty_cache()

    df = pd.read_csv(eval_result_file, encoding="utf-8")
    df.drop_duplicates(
        subset=["model", "country", "language", "prompt_no", "eval_method"],
        keep="last", inplace=True,
    )
    df.to_csv(eval_result_file, index=False, encoding="utf-8")
    return df


def print_summary(df: pd.DataFrame):
    """Print final summary tables by language and country."""
    eval_result_file = str(EVAL_DIR / "evaluation_results.csv")

    method_tag = "EXP24-DPBR" if USE_EXP24 else MODEL_NAME
    print(f"\n{'='*60}")
    print(f"RESULTS [{method_tag}] saved to: {eval_result_file}")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    print("SUMMARY BY LANGUAGE (averaged over countries & prompts)")
    print(f"{'='*60}")
    for metric in ["SEM-B", "SEM-W"]:
        metric_df = df[df["eval_method"] == metric]
        lang_avg = metric_df.groupby("language")["score"].mean().sort_values(ascending=False)
        print(f"\n  {metric}:")
        for lang, score in lang_avg.items():
            print(f"    {lang:<15s} {score:6.2f}%")
        print(f"    {'─'*25}")
        print(f"    {'OVERALL':<15s} {lang_avg.mean():6.2f}%")

    print(f"\n{'='*60}")
    print("SUMMARY BY COUNTRY (averaged over languages & prompts)")
    print(f"{'='*60}")
    for metric in ["SEM-B", "SEM-W"]:
        metric_df = df[df["eval_method"] == metric]
        country_avg = metric_df.groupby("country")["score"].mean().sort_values(ascending=False)
        print(f"\n  {metric}:")
        for ctry, score in country_avg.items():
            print(f"    {ctry:<20s} {score:6.2f}%")
        print(f"    {'─'*30}")
        print(f"    {'OVERALL':<20s} {country_avg.mean():6.2f}%")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    method = "EXP-24 DPBR (SWA-PTIS)" if USE_EXP24 else "Vanilla Greedy"
    print("=" * 60)
    print("BLEnD Short-Answer QA — EXP-24 DPBR")
    print(f"Method: {method}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {PROMPT_NOS}")
    print(f"Countries: {len(COUNTRY_LANG)}")
    print(f"Data: {DATA_ROOT}")
    print(f"Output: {WORK_DIR}")
    if USE_EXP24:
        cfg = EXP24QAConfig()
        print(f"[THEORY] K_half={cfg.K_half}×2={cfg.K_half*2} total  |  VAR_SCALE={cfg.var_scale}")
        print(f"[THEORY] r = exp(-(δ*₁-δ*₂)² / {cfg.var_scale})  →  δ* = r·(δ*₁+δ*₂)/2")
        print(f"[THEORY] ESS anchor blend: "
              f"{'ON (δ_micro = α·anchor + (1-α)·δ_base + δ*)' if cfg.use_ess_anchor_reg else 'OFF'}")
        print(f"[CONFIG] lambda_coop={cfg.lambda_coop}  rho_eff={cfg.rho_eff}  "
              f"M_candidates={cfg.M_candidates}")
        print(f"[CONFIG] PT: α={cfg.pt_alpha} β={cfg.pt_beta} κ={cfg.pt_kappa}")
    print("=" * 60)

    for subdir in [QUESTIONS_DIR, ANNOTATIONS_DIR, PROMPTS_DIR]:
        if not os.path.exists(subdir):
            print(f"[ERROR] Data directory not found: {subdir}")
            print("Please upload the BLEnD data/ folder as a Kaggle dataset")
            sys.exit(1)

    # Step 1: Load model
    model, tokenizer = _load_model_with_timeout()

    # Step 2: Run inference + evaluate per combo
    results_df = run_all_inference_and_eval(model, tokenizer)

    # Step 3: Free GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Print summaries
    print_summary(results_df)

    print("\n[DONE] EXP-24 DPBR pipeline complete.")


if __name__ == "__main__":
    main()
