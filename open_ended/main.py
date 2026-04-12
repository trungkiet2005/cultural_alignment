#!/usr/bin/env python3
"""
BLEnD Short-Answer QA with SWA-MPPI Cultural Steering
======================================================
Single-file Kaggle-ready script combining:
  1. BLEnD benchmark evaluation (NeurIPS 2024 Datasets & Benchmarks)
  2. SWA-MPPI candidate-level cultural steering (our method)

Pipeline:
  Step 1 — Load Llama 3.1 70B (4-bit, H100 80GB)
  Step 2 — For each (country, language, prompt):
            a) Generate M candidate answers via persona-steered decoding
            b) Score candidates under N cultural agents (log-prob matrix)
            c) MPPI optimization with Prospect Theory + social cooperation
            d) Select culturally-optimal answer
  Step 3 — Evaluate with Soft Exact Match (SEM-B & SEM-W)

Usage on Kaggle:
  1. Upload BLEnD dataset as Kaggle dataset input
  2. Set HF_TOKEN in Kaggle secrets
  3. Select GPU H100 80GB runtime
  4. Run this script
"""

import os
import sys
import subprocess
import gc
import csv
import json
import re
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
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose and r.stdout:
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.strip())
    return r.returncode


if _ON_KAGGLE:
    print("[SETUP] Installing dependencies...")
    # Core ML
    _run("pip install -q bitsandbytes scipy tqdm matplotlib seaborn")
    _run("pip install --upgrade --no-deps unsloth")
    _run("pip install -q unsloth_zoo")
    _run("pip install --quiet --no-deps --force-reinstall pyarrow")
    _run("pip install --quiet 'datasets>=3.4.1,<4.4.0'")

    # Evaluation — lemmatizers per language
    _run("pip install -q spacy")
    _run("python -m spacy download en_core_web_sm")
    _run("pip install -q konlpy")          # Korean
    _run("pip install -q jieba")           # Chinese
    _run("pip install -q hazm")            # Persian
    _run("pip install -q qalsadi")         # Arabic
    # Uncomment below when running full 16 countries:
    # _run("pip install -q nlp-id")        # Indonesian
    # _run("pip install -q hausastemmer")   # Hausa
    # _run("pip install -q cltk")          # Greek
    # _run("pip install -q pyspark spark-nlp")  # Spanish, Amharic

    for repo, url in [
        ("stemmer", "https://github.com/aznlp-disc/stemmer.git"),
        ("indic_nlp_library", "https://github.com/anoopkunchukuttan/indic_nlp_library.git"),
        ("SUSTEM", "https://github.com/andhikaprima/SUSTEM.git"),
    ]:
        if not os.path.exists(f"/kaggle/working/{repo}"):
            _run(f"cd /kaggle/working && git clone {url}")

    print("[SETUP] Installation complete.")

# CRITICAL: import unsloth BEFORE transformers
try:
    import unsloth  # noqa: F401
except Exception:
    pass

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
# SWA-MPPI toggle — set False for vanilla greedy baseline
# ---------------------------------------------------------------------------
USE_SWA_MPPI = True

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
        raise SystemExit(1)


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
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
MODEL_NAME = "Llama-3.1-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0

# ---------------------------------------------------------------------------
# Countries & languages — exact 16 from the paper
# ---------------------------------------------------------------------------
## Full 16 countries (uncomment when ready for full run):
# COUNTRY_LANG: Dict[str, str] = {
#     "UK": "English", "US": "English", "South_Korea": "Korean",
#     "Algeria": "Arabic", "China": "Chinese", "Indonesia": "Indonesian",
#     "Spain": "Spanish", "Iran": "Persian", "Mexico": "Spanish",
#     "Assam": "Assamese", "Greece": "Greek", "Ethiopia": "Amharic",
#     "Northern_Nigeria": "Hausa", "Azerbaijan": "Azerbaijani",
#     "North_Korea": "Korean", "West_Java": "Sundanese",
# }

# 5 representative countries (mix of high/low resource, diverse regions)
COUNTRY_LANG: Dict[str, str] = {
    "UK": "English",
    "South_Korea": "Korean",
    "China": "Chinese",
    "Iran": "Persian",
    "Algeria": "Arabic",
}

PROMPT_NOS: List[str] = ["inst-4"]


# ============================================================================
# 2. SWA-MPPI CULTURAL STEERING ENGINE
# ============================================================================

# ---------------------------------------------------------------------------
# SWA-MPPI Config
# ---------------------------------------------------------------------------
@dataclass
class CulturalQAConfig:
    """Hyperparameters for SWA-MPPI Cultural QA."""
    # SWA-MPPI Core
    lambda_coop: float = 0.7        # balance private vs social reward
    alpha_kl: float = 0.05          # KL divergence penalty weight
    K_samples: int = 128            # MPPI perturbation samples
    noise_std: float = 0.3          # Gaussian perturbation std
    temperature: float = 0.5        # MPPI softmax temperature (beta)
    tau_conflict: float = 0.001     # variance threshold for MPPI trigger
    logit_temperature: float = 1.0  # candidate scoring softmax temperature

    # Prospect Theory (Kahneman & Tversky, 1979)
    pt_alpha: float = 0.88          # gain curvature
    pt_beta: float = 0.88           # loss curvature
    pt_kappa: float = 2.25          # loss aversion coefficient

    # Candidate generation
    M_candidates: int = 8           # target diverse candidates
    max_new_tokens: int = 64        # max tokens per candidate
    gen_temperature: float = 0.7    # temperature for diversity sampling

    # Adaptive tau calibration
    tau_target_trigger_rate: float = 0.35
    tau_calibration_n: int = 50

    # WVS data path
    wvs_data_path: str = ""

    # BLEnD country → WVS ISO mapping
    BLEND_TO_WVS: Dict[str, str] = field(default_factory=lambda: {
        "UK": "GBR", "US": "USA", "South_Korea": "KOR", "China": "CHN",
        "Mexico": "MEX", "Northern_Nigeria": "NGA", "Azerbaijan": "AZE",
        "Algeria": "DZA", "Indonesia": "IDN", "Spain": "ESP", "Iran": "IRN",
        "Assam": "IND", "Greece": "GRC", "Ethiopia": "ETH",
        "North_Korea": "KOR",
        "West_Java": "IDN",
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
}


def build_qa_personas(country: str, config: CulturalQAConfig) -> List[str]:
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
            print(f"[SWA-QA] WVS personas for {country} ({wvs_iso}): {len(personas[:4])}")
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
# CulturalQAController — SWA-MPPI adapted for candidate-level QA
# ---------------------------------------------------------------------------
class CulturalQAController:
    """
    Socially-Weighted Alignment (SWA) via MPPI for Short Question Answering.

    Instead of binary LEFT/RIGHT logits (Moral Machine), operates in
    M-dimensional candidate space.

    Pipeline:
      1. Generate M diverse candidates via persona-steered decoding
      2. Score each candidate under each cultural agent (log-prob matrix)
      3. Detect inter-agent conflict via variance threshold (tau)
      4. If conflict: MPPI optimization with Prospect Theory + social cooperation
      5. Select answer from optimal candidate distribution
    """

    def __init__(self, model, tokenizer, personas: List[str],
                 config: Optional[CulturalQAConfig] = None):
        self.config = config or CulturalQAConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.personas = personas
        self.N = len(personas)
        self.M = self.config.M_candidates
        self.device = next(model.parameters()).device

        # MPPI parameters
        self.lambda_coop = self.config.lambda_coop
        self.alpha_kl = self.config.alpha_kl
        self.K = self.config.K_samples
        self.noise_std = self.config.noise_std
        self.beta = self.config.temperature
        self.tau_conflict = self.config.tau_conflict
        self.logit_temp = self.config.logit_temperature

        # Prospect Theory
        self.pt_alpha = self.config.pt_alpha
        self.pt_beta = self.config.pt_beta
        self.pt_kappa = self.config.pt_kappa

        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        self._build_persona_prefixes()

    @torch.no_grad()
    def _build_persona_prefixes(self):
        """Tokenize persona system prompts for prefix-caching."""
        print(f"[SWA-QA] Building prefixes for {self.N} personas + 1 base...")
        t0 = time.time()

        self.persona_prefix_ids = []
        for persona_text in self.personas:
            prefix = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{persona_text}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
            )
            ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
            self.persona_prefix_ids.append(ids)

        base_prefix = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        self.base_prefix_ids = self.tokenizer(
            base_prefix, return_tensors="pt"
        ).input_ids.to(self.device)

        print(f"[SWA-QA] Prefix tokenisation: {time.time() - t0:.2f}s")

    # ── Phase 1: Candidate Generation (BATCHED) ────────────────────────
    @torch.no_grad()
    def _generate_single(self, prefix_ids: torch.Tensor, query_text: str,
                         do_sample: bool = False, temperature: float = 1.0) -> str:
        """Generate one answer given a prefix and query (fallback for sampling)."""
        suffix = query_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
        """Batched greedy generation: base + all personas in ONE forward pass."""
        suffix = query_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        query_ids = self.tokenizer(suffix, return_tensors="pt").input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        seqs = [torch.cat([p, query_ids], dim=1) for p in all_prefixes]
        max_len = max(s.shape[1] for s in seqs)

        # Left-pad to same length for batching
        batch_ids, batch_mask = [], []
        for s in seqs:
            pad_len = max_len - s.shape[1]
            batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
            batch_mask.append(F.pad(
                torch.ones(1, s.shape[1], dtype=torch.long, device=self.device),
                (pad_len, 0), value=0,
            ))

        batch_ids = torch.cat(batch_ids, dim=0)   # (N+1, max_len)
        batch_mask = torch.cat(batch_mask, dim=0)  # (N+1, max_len)

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

        # Batched greedy: base + all personas in ONE call (~5x speedup)
        for text in self._generate_batched(query_text):
            _add(text)

        # Temperature sampling for extra diversity if needed
        max_extra = self.M * 2
        attempts = 0
        while len(candidates) < self.M and attempts < max_extra:
            _add(self._generate_single(
                self.base_prefix_ids, query_text,
                do_sample=True, temperature=self.config.gen_temperature
            ))
            attempts += 1

        return candidates

    # ── Phase 2: Candidate Scoring (BATCHED) ─────────────────────────
    @torch.no_grad()
    def _score_candidates(self, query_text: str,
                          candidates: List[str]) -> torch.Tensor:
        """
        Batched scoring: all (agent, candidate) pairs in mini-batches.

        Instead of 25 individual forward passes, runs ~5 batched passes
        (one per agent, each scoring all candidates at once).
        """
        M = len(candidates)
        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        n_agents = len(all_prefixes)

        suffix = query_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        query_ids = self.tokenizer(suffix, return_tensors="pt").input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        # Pre-tokenize all candidate answers
        cand_ids_list = []
        for c in candidates:
            ids = self.tokenizer(
                c, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            cand_ids_list.append(ids)

        L = torch.zeros(n_agents, M, device=self.device)

        # For each agent, batch all candidates together
        for i, prefix in enumerate(all_prefixes):
            # Build sequences: prefix + query + candidate_j for each j
            seqs = []
            answer_starts = []
            answer_lens = []
            for j in range(M):
                seq = torch.cat([prefix, query_ids, cand_ids_list[j]], dim=1)
                seqs.append(seq)
                answer_starts.append(prefix.shape[1] + query_ids.shape[1])
                answer_lens.append(cand_ids_list[j].shape[1])

            max_len = max(s.shape[1] for s in seqs)

            # Left-pad for batching
            batch_ids, batch_mask = [], []
            for s in seqs:
                pad_len = max_len - s.shape[1]
                batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
                batch_mask.append(F.pad(
                    torch.ones(1, s.shape[1], dtype=torch.long, device=self.device),
                    (pad_len, 0), value=0,
                ))

            batch_ids = torch.cat(batch_ids, dim=0)   # (M, max_len)
            batch_mask = torch.cat(batch_mask, dim=0)  # (M, max_len)

            outputs = self.model(
                input_ids=batch_ids, attention_mask=batch_mask, use_cache=False,
            )
            logits = outputs.logits  # (M, max_len, vocab_size)

            # Extract per-candidate log-probs
            for j in range(M):
                a_len = answer_lens[j]
                if a_len == 0:
                    L[i, j] = -100.0
                    continue
                # Account for left-padding offset
                pad_len = max_len - seqs[j].shape[1]
                a_start = pad_len + answer_starts[j]
                pred_logits = logits[j, a_start - 1: a_start + a_len - 1, :]
                log_probs = F.log_softmax(pred_logits, dim=-1)
                target = cand_ids_list[j][0]
                token_lp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                L[i, j] = token_lp.mean()

        return L

    # ── Phase 3: Reward Computation + MPPI ───────────────────────────
    def _compute_qa_rewards(self, L: torch.Tensor
                            ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Contrastive rewards and conflict signal from log-prob matrix."""
        L_base = L[0:1]
        L_agents = L[1:]

        p_base = F.softmax(L_base / self.logit_temp, dim=1)
        p_agents = F.softmax(L_agents / self.logit_temp, dim=1)

        r_agents = p_agents - p_base
        p_consensus = p_agents.mean(dim=0)
        variance = torch.var(p_agents, dim=0).sum().item()

        return r_agents, variance, p_consensus

    def _prospect_value(self, x: torch.Tensor) -> torch.Tensor:
        """Prospect Theory value function (Kahneman & Tversky, 1979)."""
        return torch.where(
            x >= 0,
            x.abs().pow(self.pt_alpha),
            -self.pt_kappa * x.abs().pow(self.pt_beta),
        )

    @torch.no_grad()
    def _mppi_solve_qa(self, p_consensus: torch.Tensor,
                       r_agents: torch.Tensor) -> torch.Tensor:
        """MPPI optimization in M-dimensional candidate space."""
        M = p_consensus.shape[0]
        N = r_agents.shape[0]

        log_consensus = torch.log(p_consensus.clamp(min=1e-8))
        epsilon = torch.randn(self.K, M, device=self.device) * self.noise_std
        p_pert = F.softmax(log_consensus.unsqueeze(0) + epsilon, dim=1)

        kl_penalty = (p_pert * torch.log(
            p_pert / p_consensus.unsqueeze(0).clamp(min=1e-8)
        )).sum(dim=1)

        U_total = torch.zeros(self.K, device=self.device)

        for i in range(N):
            r_i_k = (r_agents[i].unsqueeze(0) * p_pert).sum(dim=1)
            if N > 1:
                r_others_vec = (r_agents.sum(0) - r_agents[i]) / (N - 1)
            else:
                r_others_vec = r_agents[i]
            r_others_k = (r_others_vec.unsqueeze(0) * p_pert).sum(dim=1)

            u_private = self._prospect_value(r_i_k)
            u_social = self._prospect_value(r_others_k)
            u_i = (1 - self.lambda_coop) * u_private + self.lambda_coop * u_social
            U_total += u_i

        U_total /= N
        U_total -= self.alpha_kl * kl_penalty

        weights = F.softmax(U_total / self.beta, dim=0)
        p_star = (weights.unsqueeze(1) * p_pert).sum(dim=0)

        return p_star

    # ── Adaptive Tau Calibration ─────────────────────────────────────
    @torch.no_grad()
    def calibrate_tau(self, sample_prompts: List[str],
                      target_trigger_rate: Optional[float] = None) -> float:
        """Calibrate tau so MPPI fires on ~target_trigger_rate of questions."""
        if target_trigger_rate is None:
            target_trigger_rate = self.config.tau_target_trigger_rate

        n_calib = min(len(sample_prompts), self.config.tau_calibration_n)
        variances = []

        print(f"[SWA-QA] Calibrating tau on {n_calib} samples "
              f"(target trigger: {target_trigger_rate:.0%})...")

        for prompt in tqdm(sample_prompts[:n_calib], desc="Calibrating tau"):
            candidates = self._generate_candidates(prompt)
            if len(candidates) < 2:
                continue
            L = self._score_candidates(prompt, candidates)
            _, variance, _ = self._compute_qa_rewards(L)
            variances.append(variance)

        if not variances:
            print("[SWA-QA] No valid calibration samples, keeping default tau")
            return self.tau_conflict

        percentile = (1.0 - target_trigger_rate) * 100.0
        tau_calibrated = float(np.percentile(variances, percentile))
        self.tau_conflict = tau_calibrated

        print(f"[SWA-QA] Calibrated tau = {tau_calibrated:.6f} "
              f"(percentile: {percentile:.0f}th of {len(variances)} samples)")
        return tau_calibrated

    # ── Main Prediction ──────────────────────────────────────────────
    @torch.no_grad()
    def predict_qa(self, query_text: str) -> Dict:
        """
        Run SWA-MPPI prediction for a short-answer question.

        Returns dict with: answer, candidates, p_star, mppi_triggered,
        variance, selected_idx, r_agents, L_scores
        """
        candidates = self._generate_candidates(query_text)

        if len(candidates) == 0:
            return {"answer": "", "candidates": [], "p_star": [],
                    "mppi_triggered": False, "variance": 0.0, "selected_idx": -1}

        if len(candidates) == 1:
            return {"answer": candidates[0], "candidates": candidates,
                    "p_star": [1.0], "mppi_triggered": False,
                    "variance": 0.0, "selected_idx": 0}

        L = self._score_candidates(query_text, candidates)
        r_agents, variance, p_consensus = self._compute_qa_rewards(L)

        mppi_triggered = variance >= self.tau_conflict

        if mppi_triggered:
            p_star = self._mppi_solve_qa(p_consensus, r_agents)
            selected_idx = torch.argmax(p_star).item()
        else:
            p_star = p_consensus
            selected_idx = torch.argmax(p_consensus).item()

        return {
            "answer": candidates[selected_idx],
            "candidates": candidates,
            "p_star": p_star.tolist(),
            "p_consensus": p_consensus.tolist(),
            "mppi_triggered": mppi_triggered,
            "variance": variance,
            "selected_idx": selected_idx,
            "r_agents": r_agents.tolist(),
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
# 4. MODEL LOADING
# ============================================================================

def load_model():
    """Load Llama 3.1 70B Instruct (4-bit, fits H100 80GB)."""
    print(f"[MODEL] Loading {MODEL_ID} via HF transformers ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )
    model.eval()
    model.generation_config.max_length = None
    print(f"[MODEL] Loaded (4-bit). Device: {model.device}")
    return model, tokenizer, "hf"


# ============================================================================
# 5. INFERENCE
# ============================================================================

def generate_response(model, tokenizer, backend: str, prompt: str) -> str:
    """Vanilla greedy decoding (paper baseline)."""
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


def _init_swa_controller(model, tokenizer, country: str):
    """Initialize SWA-MPPI controller for a given country."""
    config = CulturalQAConfig(wvs_data_path=WVS_DATA_PATH)
    personas = build_qa_personas(country, config)
    return CulturalQAController(
        model=model, tokenizer=tokenizer, personas=personas, config=config,
    )


def run_inference_for_country(
    model, tokenizer, backend: str,
    country: str, language: str, prompt_no: str
) -> str:
    """Run inference on all questions for one (country, language, prompt) combo."""
    q_df = load_questions(country)
    prompt_sheet = load_prompts(country)

    local_lang = COUNTRY_LANG[country]
    q_col = "Question" if language == local_lang else "Translation"
    replace_country_flag = (language == "English" and local_lang != "English")

    method_tag = "SWA-MPPI" if USE_SWA_MPPI else MODEL_NAME
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
        if USE_SWA_MPPI:
            header += ["mppi_triggered", "variance", "n_candidates", "candidates"]
        write_csv_row(header, output_filename)

    # Initialize SWA-MPPI controller if needed
    swa_controller = None
    if USE_SWA_MPPI:
        swa_controller = _init_swa_controller(model, tokenizer, country)

        # Calibrate tau on first N questions
        calib_prompts = []
        for _, row in q_df.head(swa_controller.config.tau_calibration_n).iterrows():
            q = row[q_col]
            if replace_country_flag:
                q = replace_country_name(q, country.replace("_", " "))
            calib_prompts.append(make_prompt(q, prompt_no, language, country, prompt_sheet))
        swa_controller.calibrate_tau(calib_prompts)

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

        if USE_SWA_MPPI and swa_controller is not None:
            result = swa_controller.predict_qa(full_prompt)
            response = result["answer"]
            extra = [
                result["mppi_triggered"],
                f"{result['variance']:.6f}",
                len(result["candidates"]),
                json.dumps(result["candidates"], ensure_ascii=False),
            ]
            write_csv_row(
                [qid, question, full_prompt, response, prompt_no] + extra,
                output_filename,
            )
        else:
            response = generate_response(model, tokenizer, backend, full_prompt)
            write_csv_row(
                [qid, question, full_prompt, response, prompt_no],
                output_filename,
            )

    return output_filename


# ============================================================================
# 6. EVALUATION — Soft Exact Match (SEM-B & SEM-W)
# ============================================================================

def delete_prompt_from_answer(text: str, prompt: str) -> str:
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

    method_tag = "SWA-MPPI" if USE_SWA_MPPI else MODEL_NAME
    print(f"    >>> [{method_tag}] SEM-B: {sem_b:.2f}%  |  SEM-W: {sem_w:.2f}%")

    write_csv_row([method_tag, country, language, prompt_no, "SEM-B", sem_b],
                  eval_result_file)
    write_csv_row([method_tag, country, language, prompt_no, "SEM-W", sem_w],
                  eval_result_file)

    scored_df.to_csv(
        str(RESULTS_DIR / f"{method_tag}_{country}_{language}_{prompt_no}_response_score.csv"),
        index=False, encoding="utf-8",
    )

    return sem_b, sem_w


def run_all_inference_and_eval(model, tokenizer, backend: str) -> pd.DataFrame:
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
          f"({len(COUNTRY_LANG)} countries x {len(PROMPT_NOS)} prompts)")
    print(f"{'='*60}\n")

    for i, (country, language, prompt_no) in enumerate(total_combos):
        print(f"\n[{i+1}/{len(total_combos)}] {country} / {language} / {prompt_no}")

        result_file = run_inference_for_country(
            model, tokenizer, backend, country, language, prompt_no)

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
    eval_result_file = str(EVAL_DIR / "evaluation_results.csv")

    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {eval_result_file}")
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
    method = "SWA-MPPI Cultural Steering" if USE_SWA_MPPI else "Vanilla Greedy"
    print("=" * 60)
    print("BLEnD Short-Answer QA with SWA-MPPI")
    print(f"Method: {method}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {PROMPT_NOS}")
    print(f"Countries: {len(COUNTRY_LANG)}")
    print(f"Data: {DATA_ROOT}")
    print(f"Output: {WORK_DIR}")
    print("=" * 60)

    for subdir in [QUESTIONS_DIR, ANNOTATIONS_DIR, PROMPTS_DIR]:
        if not os.path.exists(subdir):
            print(f"[ERROR] Data directory not found: {subdir}")
            print("Please upload the BLEnD data/ folder as a Kaggle dataset")
            sys.exit(1)

    # Step 1: Load model
    model, tokenizer, backend = load_model()

    # Step 2: Run inference + evaluate per combo
    results_df = run_all_inference_and_eval(model, tokenizer, backend)

    # Step 3: Free GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Print summaries
    print_summary(results_df)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
