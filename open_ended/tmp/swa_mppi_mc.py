#!/usr/bin/env python3
"""
SWA-MPPI for Culturally-Aware Multiple-Choice Question Answering
=================================================================
Adapts the Socially-Weighted Alignment via Model Predictive Path Integral
(SWA-MPPI) framework from binary ethical decisions to 4-choice MC questions.

Core innovation: 4-Choice Logit-Space MPPI
  - Extract logits at choice tokens A/B/C/D (generalized from binary LEFT/RIGHT)
  - MPPI optimization in 4-dimensional probability simplex
  - Positional debiasing via cyclic rotations (generalized from LEFT↔RIGHT swap)
  - Cultural personas from World Values Survey (WVS) Wave 7

Theoretical contributions preserved:
  - Prospect Theory value function (Kahneman & Tversky, 1979)
  - Social cooperation via lambda_coop weighting
  - KL-regularized MPPI optimal control
  - Adaptive tau conflict detection

Pipeline per MC question:
  Phase 1 — Evaluate base + N persona agents (single batched forward pass)
  Phase 2 — Compute contrastive rewards in 4-choice space
  Phase 3 — Detect inter-agent conflict via variance threshold
  Phase 4 — If conflict: MPPI optimization → optimal choice distribution
  Phase 5 — Positional debiasing via 4 cyclic rotations, average
"""

# ============================================================================
# KAGGLE ENVIRONMENT SETUP
# ============================================================================
import sys
import os
import subprocess
from pathlib import Path


def _run(cmd: str, verbose: bool = False) -> int:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose and r.stdout:
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.strip())
    return r.returncode


_ON_KAGGLE = os.path.exists("/kaggle/working")
if _ON_KAGGLE:
    print("[SETUP] Installing dependencies...")
    # Core ML
    _run("pip install -q bitsandbytes scipy tqdm matplotlib seaborn")
    _run("pip install --upgrade --no-deps unsloth")
    _run("pip install -q unsloth_zoo")
    _run("pip install --quiet --no-deps --force-reinstall pyarrow")
    _run("pip install --quiet 'datasets>=3.4.1,<4.4.0'")
    # HuggingFace Hub (for auto-downloading dataset)
    _run("pip install -q huggingface_hub")
    print("[SETUP] Done")

# CRITICAL: import unsloth BEFORE transformers
try:
    import unsloth  # noqa: F401
except Exception:
    pass

import gc
import csv
import json
import re
import time
import math
import itertools
import warnings
from dataclasses import dataclass, field, replace
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Performance knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class CulturalMCConfig:
    """Hyperparameters for SWA-MPPI on Multiple-Choice QA."""

    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

    # SWA-MPPI Core
    lambda_coop: float = 0.7        # balance private vs social reward
    alpha_kl: float = 0.05          # KL divergence penalty weight
    K_samples: int = 128            # number of MPPI perturbation samples
    noise_std: float = 0.3          # Gaussian perturbation std
    temperature: float = 0.5        # MPPI softmax temperature (beta)
    tau_conflict: float = -1.0     # Always-on MPPI (bypasses variance check)
    logit_temperature: float = 1.0  # temperature for choice logit softmax

    # Prospect Theory (Kahneman & Tversky, 1979)
    pt_alpha: float = 0.88          # gain curvature
    pt_beta: float = 0.88           # loss curvature
    pt_kappa: float = 2.25          # loss aversion coefficient

    # Positional debiasing
    n_rotations: int = 4            # number of cyclic rotations for debiasing

    # Adaptive tau calibration
    tau_target_trigger_rate: float = 0.35
    tau_calibration_n: int = 50

    # Data sampling
    n_per_country: int = 500        # same as baseline; uses all unique IDs per country

    # WVS data path (World Values Survey Wave 7)
    wvs_data_path: str = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"

    # BLEnD country → WVS ISO mapping
    BLEND_TO_WVS: Dict[str, str] = field(default_factory=lambda: {
        "UK": "GBR", "US": "USA", "South_Korea": "KOR", "China": "CHN",
        "Mexico": "MEX", "Northern_Nigeria": "NGA", "Azerbaijan": "AZE",
        "Algeria": "DZA", "Indonesia": "IDN", "Spain": "ESP", "Iran": "IRN",
        "Assam": "IND", "Greece": "GRC", "Ethiopia": "ETH",
        "North_Korea": "KOR", "West_Java": "IDN",
    })

    # BLEnD country full names (for persona text)
    BLEND_COUNTRY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        "UK": "the United Kingdom",
        "US": "the United States",
        "South_Korea": "South Korea",
        "Algeria": "Algeria",
        "China": "China",
        "Indonesia": "Indonesia",
        "Spain": "Spain",
        "Iran": "Iran",
        "Mexico": "Mexico",
        "Assam": "Assam, India",
        "Greece": "Greece",
        "Ethiopia": "Ethiopia",
        "Northern_Nigeria": "Northern Nigeria",
        "Azerbaijan": "Azerbaijan",
        "North_Korea": "North Korea",
        "West_Java": "West Java, Indonesia",
    })


# ============================================================================
# PATHS
# ============================================================================
def _find_mc_data_dir_kaggle():
    """Auto-discover MC data directory under /kaggle/input/ by scanning all datasets."""
    search_roots = []
    if os.path.isdir("/kaggle/input"):
        for name in os.listdir("/kaggle/input"):
            search_roots.append(os.path.join("/kaggle/input", name))
    search_roots.append("/kaggle/working/blend_data")
    search_roots.append("/kaggle/working")

    sub_paths = [
        os.path.join("evaluation", "mc_data", "v1.1"),
        os.path.join("mc_data", "v1.1"),
        os.path.join("evaluation", "mc_data"),
        "mc_data",
    ]
    for root in search_roots:
        for sp in sub_paths:
            candidate = os.path.join(root, sp)
            if os.path.isdir(candidate):
                return candidate, root
    return None, search_roots[0] if search_roots else "/kaggle/working"


if _ON_KAGGLE:
    _found_mc, _DATA_BASE = _find_mc_data_dir_kaggle()
    WORK_DIR = Path("/kaggle/working/swa_mppi_mc_results")
else:
    _found_mc = None
    _DATA_BASE = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "swa_mppi_mc_results"

# MC data directory
if _found_mc:
    MC_DATA_DIR = _found_mc
else:
    MC_DATA_DIR = os.path.join(_DATA_BASE, "evaluation", "mc_data", "v1.1")
    if not os.path.exists(MC_DATA_DIR):
        for alt in [
            os.path.join(_DATA_BASE, "mc_data", "v1.1"),
            os.path.join(_DATA_BASE, "evaluation", "mc_data"),
            os.path.join(_DATA_BASE, "mc_data"),
        ]:
            if os.path.exists(alt):
                MC_DATA_DIR = alt
                break


def _download_blend_mc_from_hf():
    """Download BLEnD MC data from HuggingFace into a writable directory."""
    global MC_DATA_DIR
    print("[DATA] MC data not found. Downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        if _ON_KAGGLE:
            dl_dir = "/kaggle/working/blend_data"
        else:
            dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_data")
        os.makedirs(dl_dir, exist_ok=True)
        snapshot_download(
            repo_id="nayeon212/BLEnD",
            repo_type="dataset",
            local_dir=dl_dir,
            allow_patterns=["evaluation/mc_data/**"],
        )
        for candidate in [
            os.path.join(dl_dir, "evaluation", "mc_data", "v1.1"),
            os.path.join(dl_dir, "evaluation", "mc_data"),
            os.path.join(dl_dir, "mc_data", "v1.1"),
            os.path.join(dl_dir, "mc_data"),
        ]:
            if os.path.isdir(candidate):
                MC_DATA_DIR = candidate
                print(f"[DATA] Download complete. MC data at: {MC_DATA_DIR}")
                return
        raise FileNotFoundError("Downloaded but MC data directories not found")
    except ImportError:
        print("[DATA] huggingface_hub not installed. Run: pip install huggingface_hub")
        raise SystemExit(1)
    except Exception as e:
        print(f"[DATA] Download failed: {e}")
        print("[DATA] Please add BLEnD as a Kaggle dataset input, or run:")
        print("       git clone https://huggingface.co/datasets/nayeon212/BLEnD")
        raise SystemExit(1)


# Auto-download if MC data is missing
if not os.path.exists(MC_DATA_DIR):
    _download_blend_mc_from_hf()

MC_FILES = ["mc_questions_file-1.csv", "mc_questions_file-2.csv"]
MODEL_NAME = "Llama-3.1-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")

RESULTS_DIR = WORK_DIR / "results"
EVAL_DIR = WORK_DIR / "evaluation"
for d in [RESULTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 5 representative countries (same as baseline_mc.py)
MC_COUNTRIES = ["UK", "South_Korea", "China", "Iran", "Algeria"]
SAMPLE_SEED = 42

# Country → language code mapping for native-language personas
_BLEND_LANG: Dict[str, str] = {
    "UK": "en", "US": "en", "Northern_Nigeria": "en",
    "South_Korea": "ko", "North_Korea": "ko",
    "China": "zh",
    "Iran": "fa",
    "Algeria": "ar",
    "Mexico": "es", "Spain": "es",
    "Indonesia": "id", "West_Java": "id",
    "Azerbaijan": "az",
    "Greece": "el",
    "Ethiopia": "am",
    "Assam": "hi",
}


# ============================================================================
# WVS-BASED CULTURAL PERSONA GENERATION
# (Adapted from swa_mppi_qa.py for self-contained Kaggle execution)
# ============================================================================
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


def _load_wvs_profiles(wvs_csv_path: str,
                       target_countries: List[str]) -> Dict[str, Dict]:
    """Load and compute WVS value profiles per country per age group."""
    global _WVS_PROFILES_CACHE
    if _WVS_PROFILES_CACHE:
        return _WVS_PROFILES_CACHE

    import csv as _csv

    all_vars = set()
    for vars_list, _ in _WVS_DIMS.values():
        all_vars.update(vars_list)
    all_vars.add("Q261")   # Birth year
    all_vars.add("A_YEAR")  # Survey year

    def _age_group(birth_year, survey_year):
        age = survey_year - birth_year
        if age < 36:
            return "young"
        if age < 56:
            return "middle"
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
                dim_means[dim_name] = (
                    round(sum(vals) / len(vals), 2) if vals else 0
                )
            profiles[c][ag] = dim_means

    n_loaded = sum(
        1 for c in profiles
        if profiles[c].get("all", {}).get("religion", 0) > 0
    )
    print(f"[WVS-MC] Loaded profiles for {n_loaded}/{len(target_countries)} countries")
    _WVS_PROFILES_CACHE = profiles
    return profiles


def _describe_value(dim_name: str, value: float,
                    scale_max: float = 4.0) -> str:
    """Convert a WVS dimension mean into a natural language descriptor."""
    ratio = value / scale_max
    if dim_name == "religion":
        if ratio > 0.85:
            return "deeply religious"
        if ratio > 0.70:
            return "moderately religious"
        if ratio > 0.55:
            return "somewhat secular"
        return "highly secular"
    elif dim_name == "gender_equality":
        if ratio > 0.85:
            return "strongly gender-egalitarian"
        if ratio > 0.75:
            return "moderately gender-egalitarian"
        if ratio > 0.65:
            return "somewhat traditional on gender"
        return "traditional on gender roles"
    elif dim_name == "trust":
        if ratio > 0.55:
            return "high interpersonal trust"
        if ratio > 0.45:
            return "moderate trust"
        return "low interpersonal trust"
    elif dim_name == "moral_permissiveness":
        if value > 3.5:
            return "morally permissive"
        if value > 3.0:
            return "moderately permissive"
        if value > 2.5:
            return "morally conservative"
        return "morally strict"
    elif dim_name == "autonomy":
        if ratio > 0.90:
            return "strongly values personal autonomy"
        if ratio > 0.80:
            return "values personal autonomy"
        return "moderate on personal autonomy"
    elif dim_name == "meritocracy":
        if ratio > 0.95:
            return "strongly meritocratic"
        if ratio > 0.85:
            return "meritocratic"
        return "egalitarian on income"
    elif dim_name == "work_importance":
        if ratio > 0.90:
            return "work is central to identity"
        if ratio > 0.80:
            return "values work highly"
        return "moderate work orientation"
    elif dim_name == "family":
        return "family is paramount"
    return ""


# Native-language WVS persona templates: {role}, {country_name}, {age_range}, {traits_str}
_WVS_TEMPLATES: Dict[str, Dict] = {
    "ko": {
        "roles": {"young": ("청년", "20-30대"), "middle": ("중년", "40-50대"),
                  "older": ("어르신", "60대 이상"), "all": ("시민", "")},
        "tpl": "당신은 {country_name}의 {role}입니다{age_desc}. "
               "당신은 {country_name}의 일상생활, 음식, 관습, 명절, 스포츠, 교육, "
               "가족 전통에 대해 깊이 알고 있습니다. "
               "사회의 문화적 가치관에 기반하여, 당신은 {traits_str}. "
               "{country_name}에서 자란 사람으로서 진정한 경험을 바탕으로 질문에 답하세요.",
    },
    "zh": {
        "roles": {"young": ("年轻人", "二三十岁"), "middle": ("中年人", "四五十岁"),
                  "older": ("老年人", "六十岁以上"), "all": ("公民", "")},
        "tpl": "你是{country_name}的一位{role}{age_desc}。"
               "你对{country_name}的日常生活、饮食、习俗、节日、体育、教育和"
               "家庭传统有深入的了解。"
               "基于你所在社会的文化价值观，你{traits_str}。"
               "请以在{country_name}长大的人的真实经历来回答问题。",
    },
    "fa": {
        "roles": {"young": ("جوان", "۲۰ تا ۳۰ ساله"), "middle": ("میانسال", "۴۰ تا ۵۰ ساله"),
                  "older": ("سالمند", "بالای ۶۰ سال"), "all": ("شهروند", "")},
        "tpl": "شما یک {role} از {country_name} هستید{age_desc}. "
               "شما درباره زندگی روزمره، غذا، آداب و رسوم، تعطیلات، ورزش، آموزش و "
               "سنت‌های خانوادگی در {country_name} دانش عمیقی دارید. "
               "بر اساس ارزش‌های فرهنگی جامعه‌تان، شما {traits_str}. "
               "به عنوان کسی که در {country_name} بزرگ شده‌اید، از تجربه واقعی خود پاسخ دهید.",
    },
    "ar": {
        "roles": {"young": ("شاب", "في العشرينات والثلاثينات"), "middle": ("كهل", "في الأربعينات والخمسينات"),
                  "older": ("مسن", "فوق الستين"), "all": ("مواطن", "")},
        "tpl": "أنت {role} من {country_name}{age_desc}. "
               "لديك معرفة عميقة بالحياة اليومية والطعام والعادات والأعياد والرياضة والتعليم "
               "والتقاليد العائلية في {country_name}. "
               "بناءً على القيم الثقافية لمجتمعك، أنت {traits_str}. "
               "أجب على الأسئلة من تجربتك الشخصية الحقيقية كشخص نشأ في {country_name}.",
    },
    "es": {
        "roles": {"young": ("joven", "de 20-30 años"), "middle": ("adulto de mediana edad", "de 40-50 años"),
                  "older": ("persona mayor", "mayor de 60 años"), "all": ("ciudadano", "")},
        "tpl": "Eres un {role} de {country_name}{age_desc}. "
               "Tienes un conocimiento profundo de la vida cotidiana, la comida, las costumbres, "
               "los días festivos, los deportes, la educación y las tradiciones familiares en {country_name}. "
               "Según los valores culturales de tu sociedad, eres {traits_str}. "
               "Responde las preguntas desde tu experiencia auténtica como alguien que creció en {country_name}.",
    },
    "id": {
        "roles": {"young": ("pemuda", "berusia 20-30 tahun"), "middle": ("orang paruh baya", "berusia 40-50 tahun"),
                  "older": ("lansia", "berusia di atas 60 tahun"), "all": ("warga negara", "")},
        "tpl": "Anda adalah seorang {role} dari {country_name}{age_desc}. "
               "Anda memiliki pengetahuan mendalam tentang kehidupan sehari-hari, makanan, adat istiadat, "
               "hari libur, olahraga, pendidikan, dan tradisi keluarga di {country_name}. "
               "Berdasarkan nilai-nilai budaya masyarakat Anda, Anda {traits_str}. "
               "Jawablah pertanyaan dari pengalaman otentik Anda sebagai orang yang tumbuh di {country_name}.",
    },
    "az": {
        "roles": {"young": ("gənc", "20-30 yaş arası"), "middle": ("orta yaşlı", "40-50 yaş arası"),
                  "older": ("yaşlı", "60 yaşdan yuxarı"), "all": ("vətəndaş", "")},
        "tpl": "Siz {country_name}dan olan bir {role}sunuz{age_desc}. "
               "Siz {country_name}ın gündəlik həyatı, yeməkləri, adət-ənənələri, bayramları, idmanı, "
               "təhsili və ailə ənənələri haqqında dərin biliyə sahibsiniz. "
               "Cəmiyyətinizin mədəni dəyərlərinə əsasən, siz {traits_str}. "
               "{country_name}da böyümüş biri kimi həqiqi təcrübənizə əsaslanaraq cavab verin.",
    },
    "el": {
        "roles": {"young": ("νέος", "20-30 ετών"), "middle": ("μεσήλικας", "40-50 ετών"),
                  "older": ("ηλικιωμένος", "άνω των 60 ετών"), "all": ("πολίτης", "")},
        "tpl": "Είσαι ένας {role} από {country_name}{age_desc}. "
               "Έχεις βαθιά γνώση της καθημερινής ζωής, του φαγητού, των εθίμων, των αργιών, "
               "του αθλητισμού, της εκπαίδευσης και των οικογενειακών παραδόσεων στην {country_name}. "
               "Με βάση τις πολιτιστικές αξίες της κοινωνίας σου, είσαι {traits_str}. "
               "Απάντησε στις ερωτήσεις από την αυθεντική προσωπική σου εμπειρία.",
    },
    "am": {
        "roles": {"young": ("ወጣት", "ከ20-30 ዓመት"), "middle": ("አዋቂ", "ከ40-50 ዓመት"),
                  "older": ("አዛውንት", "ከ60 ዓመት በላይ"), "all": ("ዜጋ", "")},
        "tpl": "እርስዎ ከ{country_name} {role} ነዎት{age_desc}። "
               "በ{country_name} ውስጥ ስለ ዕለተ ዕለት ሕይወት፣ ምግብ፣ ልማዶች፣ በዓላት፣ ስፖርት፣ "
               "ትምህርት እና የቤተሰብ ባህሎች ጥልቅ ዕውቀት አለዎት። "
               "በማህበረሰብዎ ባህላዊ እሴቶች ላይ ተመስርተው፣ {traits_str}። "
               "በ{country_name} ያደጉ ሰው ሆነው ከእውነተኛ ልምድዎ መልስ ይስጡ።",
    },
    "hi": {
        "roles": {"young": ("युवा", "20-30 वर्ष"), "middle": ("मध्यम आयु", "40-50 वर्ष"),
                  "older": ("वृद्ध", "60 वर्ष से अधिक"), "all": ("नागरिक", "")},
        "tpl": "आप {country_name} के एक {role} हैं{age_desc}। "
               "आपको {country_name} के दैनिक जीवन, भोजन, रीति-रिवाज़, त्योहार, खेल, शिक्षा "
               "और पारिवारिक परंपराओं की गहरी जानकारी है। "
               "अपने समाज के सांस्कृतिक मूल्यों के आधार पर, आप {traits_str}। "
               "{country_name} में पले-बढ़े व्यक्ति के रूप में अपने प्रामाणिक अनुभव से उत्तर दें।",
    },
}


def _generate_mc_persona(country_iso: str, age_group: str,
                         profile: Dict[str, float],
                         country_name: str,
                         lang: str = "en") -> str:
    """Generate a single MC-adapted persona from WVS profile in native language."""
    # Get language-specific template data
    lang_data = _WVS_TEMPLATES.get(lang)

    # Collect trait descriptions (kept in English — model understands in context)
    traits = []
    for dim_name in [
        "religion", "gender_equality", "trust", "moral_permissiveness",
        "autonomy", "meritocracy", "work_importance",
    ]:
        val = profile.get(dim_name, 0)
        if val > 0:
            desc = _describe_value(dim_name, val)
            if desc:
                traits.append(desc)
    traits_str = ", ".join(traits[:5])

    # Use native template if available
    if lang_data:
        roles = lang_data["roles"]
        role, age_range = roles.get(age_group, roles["all"])
        age_desc = f" ({age_range})" if age_range else ""
        return lang_data["tpl"].format(
            role=role, country_name=country_name,
            age_desc=age_desc, traits_str=traits_str,
        )

    # Default English template
    age_desc_en = {
        "young": ("young adult", "in your 20s-30s"),
        "middle": ("middle-aged adult", "in your 40s-50s"),
        "older": ("senior citizen", "over 60"),
        "all": ("citizen", ""),
    }
    role, age_range = age_desc_en.get(age_group, ("citizen", ""))
    return (
        f"You are a {role} from {country_name}"
        f"{' ' + age_range if age_range else ''}. "
        f"You have deep knowledge of everyday life, food, customs, holidays, "
        f"sports, education, and family traditions in {country_name}. "
        f"Based on the cultural values of your society, you are {traits_str}. "
        f"Answer questions from your authentic personal experience as someone "
        f"who grew up in {country_name}."
    )


# Fallback personas for countries without WVS data — in NATIVE LANGUAGE
_BASE_MC_PERSONAS: Dict[str, List[str]] = {
    "GBR": [
        "You are a young British person from London. You know everyday British life well — pub culture, fish and chips, the Premier League, school uniforms, bank holidays, and tea time. Answer from your authentic experience.",
        "You are a middle-aged British person from Northern England. You have deep knowledge of British traditions, Sunday roasts, Guy Fawkes Night, cricket, the NHS, and school life. Answer from your authentic experience.",
        "You are an elderly British citizen who has lived in the UK your entire life. You know decades of British customs, food, sports, royal traditions, and everyday life. Answer from your authentic experience.",
        "You are a British cultural historian. You have expert knowledge of British everyday customs, food traditions, holidays, education system, sports, and family life across regions and classes.",
    ],
    "KOR": [
        "당신은 한국에서 자란 젊은 한국인입니다. 김치, 학교 급식, 추석, 야구, K-pop 문화, 가족 전통 등 한국의 일상생활을 잘 알고 있습니다. 진정한 경험을 바탕으로 답해주세요.",
        "당신은 한국의 중년 직장인입니다. 한국의 직장 문화, 가족 모임, 설날, 음식 문화, 교육 제도에 대해 깊이 알고 있습니다. 진정한 경험을 바탕으로 답해주세요.",
        "당신은 평생 한국에서 살아온 어르신입니다. 수십 년간의 한국 전통, 계절 음식, 명절, 가족 관습을 잘 알고 있습니다. 진정한 경험을 바탕으로 답해주세요.",
        "당신은 한국 문화 연구 교수입니다. 한국의 일상 관습, 음식 전통, 명절, 교육, 스포츠, 가족 문화에 대한 전문 지식을 갖고 있습니다.",
    ],
    "CHN": [
        "你是一个在中国长大的年轻人。你非常了解中国的日常生活——饮食文化、春节、学校生活、体育运动和家庭传统。请根据你的真实经历回答。",
        "你是一个中年中国人。你对中国的风俗习惯、美食、中秋节、教育制度和家庭聚会有深入的了解。请根据你的真实经历回答。",
        "你是一位在中国生活了一辈子的老年人。你了解数十年来的中国传统、饮食文化、节日和家庭生活。请根据你的真实经历回答。",
        "你是一位中国文化研究专家。你对中国的日常习俗、饮食传统、节日、教育、体育和家庭关系有深入的了解。",
    ],
    "IRN": [
        "شما یک جوان ایرانی هستید که در ایران بزرگ شده‌اید. شما زندگی روزمره ایرانی را به خوبی می‌شناسید — غذاهایی مثل قورمه‌سبزی و ته‌دیگ، جشن‌های نوروز، زندگی مدرسه، فرهنگ فوتبال و سنت‌های خانوادگی. از تجربه واقعی خود پاسخ دهید.",
        "شما یک ایرانی میانسال هستید. شما درباره آداب و رسوم ایرانی، آشپزی، تعطیلات مذهبی، نظام آموزشی و مهمانی‌های خانوادگی دانش عمیقی دارید. از تجربه واقعی خود پاسخ دهید.",
        "شما یک سالمند ایرانی هستید که تمام عمر خود را در ایران زندگی کرده‌اید. شما دهه‌ها تجربه از سنت‌های ایرانی، فرهنگ غذایی، آداب نوروز و زندگی خانوادگی دارید. از تجربه واقعی خود پاسخ دهید.",
        "شما کارشناس مطالعات فرهنگی ایران هستید. شما دانش عمیقی درباره آداب و رسوم ایرانی، سنت‌های غذایی، جشن‌هایی مانند نوروز و یلدا، آموزش، ورزش و پویایی خانواده دارید.",
    ],
    "DZA": [
        "أنت شاب جزائري نشأت في الجزائر. أنت تعرف الحياة اليومية الجزائرية جيداً — الكسكس، الشاي بالنعناع، تقاليد رمضان، كرة القدم، الحياة المدرسية، والعادات العائلية. أجب من تجربتك الحقيقية.",
        "أنت جزائري في منتصف العمر. لديك معرفة عميقة بالعادات والتقاليد الجزائرية، المطبخ، الأعياد الدينية، النظام التعليمي، والتجمعات العائلية. أجب من تجربتك الحقيقية.",
        "أنت مسن جزائري عشت في الجزائر طوال حياتك. أنت تعرف عقوداً من التقاليد الجزائرية والثقافة الغذائية والعادات العائلية. أجب من تجربتك الحقيقية.",
        "أنت خبير في الدراسات الثقافية الجزائرية. لديك معرفة عميقة بالعادات الجزائرية وتقاليد الطعام والأعياد والتعليم والرياضة وديناميكيات الأسرة.",
    ],
}


def build_mc_personas(country: str, config: CulturalMCConfig) -> List[str]:
    """
    Build 4 cultural personas for a BLEnD country in native language.
    Priority: WVS data (3 age-cohort + 1 expert) → fallback base personas.
    """
    wvs_iso = config.BLEND_TO_WVS.get(country, "")
    country_name = config.BLEND_COUNTRY_NAMES.get(
        country, country.replace("_", " "))
    lang = _BLEND_LANG.get(country, "en")

    # Native-language expert persona templates
    _EXPERT_TPL = {
        "ko": "당신은 {cn} 문화 전문가입니다. {cn}의 일상 관습, 전통 및 현대 음식 문화, 명절, 교육 제도, 인기 스포츠, 직장 생활, 가족 전통에 대한 포괄적인 지식을 갖고 있습니다. 문화적 정확성을 바탕으로 답하세요.",
        "zh": "你是专门研究{cn}的文化专家。你对{cn}的日常习俗、传统与现代饮食文化、节日、教育制度、热门运动、工作生活和家庭传统有全面的了解。请以文化准确性回答问题。",
        "fa": "شما متخصص مطالعات فرهنگی {cn} هستید. شما دانش جامعی درباره آداب و رسوم روزمره، فرهنگ غذایی سنتی و مدرن، تعطیلات، نظام آموزشی، ورزش‌های محبوب، زندگی کاری و سنت‌های خانوادگی در {cn} دارید. با دقت فرهنگی پاسخ دهید.",
        "ar": "أنت خبير في الدراسات الثقافية متخصص في {cn}. لديك معرفة شاملة بالعادات اليومية وثقافة الطعام التقليدية والحديثة والأعياد ونظام التعليم والرياضات الشعبية وحياة العمل والتقاليد العائلية في {cn}. أجب بدقة ثقافية.",
        "es": "Eres un experto en estudios culturales especializado en {cn}. Tienes un conocimiento integral de las costumbres cotidianas, la cultura culinaria tradicional y moderna, los días festivos, el sistema educativo, los deportes populares, la vida laboral y las tradiciones familiares en {cn}. Responde con precisión cultural.",
        "id": "Anda adalah ahli studi budaya yang mengkhususkan diri pada {cn}. Anda memiliki pengetahuan komprehensif tentang adat istiadat sehari-hari, budaya makanan tradisional dan modern, hari libur, sistem pendidikan, olahraga populer, kehidupan kerja, dan tradisi keluarga di {cn}. Jawablah dengan akurasi budaya.",
        "az": "Siz {cn} mədəniyyəti üzrə ixtisaslaşmış mədəniyyət tədqiqatçısısınız. {cn}ın gündəlik adətləri, ənənəvi və müasir yemək mədəniyyəti, bayramlar, təhsil sistemi, populyar idman növləri, iş həyatı və ailə ənənələri haqqında hərtərəfli biliyiniz var. Mədəni dəqiqliklə cavab verin.",
        "el": "Είσαι ειδικός πολιτιστικών μελετών με ειδίκευση στην {cn}. Έχεις ολοκληρωμένη γνώση των καθημερινών εθίμων, της παραδοσιακής και σύγχρονης κουλτούρας φαγητού, των αργιών, του εκπαιδευτικού συστήματος, των δημοφιλών αθλημάτων, της εργασιακής ζωής και των οικογενειακών παραδόσεων. Απάντησε με πολιτιστική ακρίβεια.",
        "am": "እርስዎ በ{cn} ላይ የተካኑ የባህል ጥናት ባለሙያ ነዎት። ስለ ዕለተ ዕለት ልማዶች፣ ባህላዊ እና ዘመናዊ የምግብ ባህል፣ በዓላት፣ የትምህርት ሥርዓት፣ ታዋቂ ስፖርቶች፣ የሥራ ሕይወት እና የቤተሰብ ባህሎች ሁሉን አቀፍ ዕውቀት አለዎት። በባህላዊ ትክክለኛነት መልስ ይስጡ።",
        "hi": "आप {cn} में विशेषज्ञता रखने वाले सांस्कृतिक अध्ययन विशेषज्ञ हैं। आपको {cn} के दैनिक रीति-रिवाज़, पारंपरिक और आधुनिक खान-पान संस्कृति, त्योहार, शिक्षा प्रणाली, लोकप्रिय खेल, कार्य जीवन और पारिवारिक परंपराओं की व्यापक जानकारी है। सांस्कृतिक सटीकता के साथ उत्तर दें।",
    }

    # Try WVS-based generation
    if (config.wvs_data_path
            and os.path.exists(config.wvs_data_path) and wvs_iso):
        target_isos = list(set(config.BLEND_TO_WVS.values()))
        profiles = _load_wvs_profiles(config.wvs_data_path, target_isos)
        country_profile = profiles.get(wvs_iso, {})

        if (country_profile
                and country_profile.get("all", {}).get("religion", 0) > 0):
            personas = []
            for ag in ["young", "middle", "older"]:
                p = country_profile.get(ag, country_profile["all"])
                if p.get("religion", 0) > 0:
                    personas.append(_generate_mc_persona(
                        wvs_iso, ag, p, country_name, lang=lang))

            # 4th persona: cultural expert in native language
            expert_tpl = _EXPERT_TPL.get(lang)
            if expert_tpl:
                personas.append(expert_tpl.format(cn=country_name))
            else:
                personas.append(
                    f"You are a cultural studies expert specializing in "
                    f"{country_name}. You have comprehensive knowledge of everyday "
                    f"customs, traditional and modern food culture, holidays, "
                    f"education system, popular sports, work life, and family "
                    f"traditions in {country_name}. Answer questions with detailed "
                    f"cultural accuracy."
                )

            while len(personas) < 4:
                personas.append(_generate_mc_persona(
                    wvs_iso, "all", country_profile["all"], country_name,
                    lang=lang))

            print(f"[SWA-MC] WVS personas for {country} ({wvs_iso}, "
                  f"lang={lang}): {len(personas[:4])}")
            return personas[:4]

    # Fallback: check base personas by WVS ISO
    if wvs_iso in _BASE_MC_PERSONAS:
        return list(_BASE_MC_PERSONAS[wvs_iso])

    # Generic fallback in native language
    lang_data = _WVS_TEMPLATES.get(lang)
    if lang_data:
        roles = lang_data["roles"]
        expert_tpl = _EXPERT_TPL.get(lang, "")
        personas = []
        for ag in ["young", "middle", "older"]:
            role, age_range = roles.get(ag, roles["all"])
            age_desc = f" ({age_range})" if age_range else ""
            personas.append(lang_data["tpl"].format(
                role=role, country_name=country_name,
                age_desc=age_desc, traits_str="culturally knowledgeable",
            ))
        personas.append(
            expert_tpl.format(cn=country_name) if expert_tpl
            else f"You are a cultural studies expert specializing in "
                 f"{country_name}. Answer with cultural accuracy."
        )
        return personas

    # English fallback
    return [
        f"You are a young adult from {country_name}. You have deep knowledge "
        f"of everyday life, food, customs, holidays, sports, and education in "
        f"{country_name}. Answer from your authentic personal experience.",

        f"You are a middle-aged person from {country_name}. You have extensive "
        f"experience with daily life, family traditions, work culture, and "
        f"local customs in {country_name}. Answer from your authentic experience.",

        f"You are an elderly person who has lived in {country_name} your "
        f"entire life. You have decades of experience with traditions, food "
        f"culture, holidays, and family customs. Answer from your authentic "
        f"personal experience.",

        f"You are a cultural studies expert specializing in {country_name}. "
        f"You have comprehensive knowledge of everyday customs, food culture, "
        f"holidays, education, sports, and family traditions. Answer with "
        f"cultural accuracy.",
    ]


# ============================================================================
# UTILITY FUNCTIONS (shared with baseline_mc.py, self-contained here)
# ============================================================================

def write_csv_row(values: list, filename: str):
    """Append a single row to a CSV file."""
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def get_json_str(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response text."""
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip internal newlines before regex matching (matches original utils.py)
    cleaned = text.replace("\n", "")
    patterns = [
        r'\{[^{}]*"answer_choice"\s*:\s*"[^"]*"[^{}]*\}',
        r'\{[^{}]*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for m in matches:
            try:
                obj = json.loads(m)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue
    return text


def wilson_ci(n_correct: int, n_total: int,
              z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
        / denom
    )
    return max(0, center - margin) * 100, min(1, center + margin) * 100


def load_mc_questions(
    mc_dir: str,
    mc_files: List[str],
    countries: List[str],
    n_per_country: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Load and sample MC questions from large CSV files (chunked)."""
    print(f"[DATA] Loading MC questions from {mc_dir}")
    print(f"[DATA] Target countries: {countries}")
    print(f"[DATA] Sampling {n_per_country} per country")

    country_set = set(countries)
    chunks = []

    for mc_file in mc_files:
        filepath = os.path.join(mc_dir, mc_file)
        if not os.path.exists(filepath):
            print(f"[WARN] MC file not found: {filepath}")
            continue

        print(f"[DATA] Reading {mc_file} (chunked)...")
        n_chunks = 0
        for chunk in pd.read_csv(filepath, encoding="utf-8", chunksize=50000):
            filtered = chunk[chunk["country"].isin(country_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
            n_chunks += 1
            if n_chunks % 20 == 0:
                n_loaded = sum(len(c) for c in chunks)
                print(f"  ... {n_chunks} chunks, {n_loaded} matching rows")

    if not chunks:
        raise FileNotFoundError(
            f"No MC questions found in {mc_dir} for countries {countries}")

    all_data = pd.concat(chunks, ignore_index=True)
    print(f"[DATA] Total matching rows: {len(all_data)}")

    rng = np.random.RandomState(seed)
    sampled_parts = []

    for country in countries:
        country_df = all_data[all_data["country"] == country]
        if len(country_df) == 0:
            print(f"[WARN] No MC questions for {country}")
            continue

        n_available = len(country_df)
        n_sample = min(n_per_country, n_available)
        unique_ids = country_df["ID"].unique()

        if len(unique_ids) <= n_sample:
            sampled_ids = unique_ids
        else:
            sampled_ids = rng.choice(unique_ids, size=n_sample, replace=False)

        # For each sampled ID, pick one random MC variant
        sampled = (
            country_df[country_df["ID"].isin(sampled_ids)]
            .groupby("ID")
            .apply(lambda g: g.sample(n=1, random_state=seed))
            .reset_index(drop=True)
        )

        sampled_parts.append(sampled.head(n_sample))
        print(f"  {country}: {n_sample} questions "
              f"(from {n_available}, {len(unique_ids)} unique IDs)")

    result = pd.concat(sampled_parts, ignore_index=True)
    print(f"[DATA] Final MC dataset: {len(result)} questions")
    return result


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: Optional[str] = None):
    """Load Llama 3.1 70B Instruct with 4-bit quantization."""
    model_id = model_name or CulturalMCConfig().model_name
    print(f"[MODEL] Loading {model_id} via HF transformers ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN,
    )
    model.eval()
    model.generation_config.max_length = None
    print(f"[MODEL] Loaded (4-bit). Device: {model.device}")
    return model, tokenizer


# ============================================================================
# CORE: CulturalMCController — SWA-MPPI for 4-Choice MC
# ============================================================================
class CulturalMCController:
    """
    Socially-Weighted Alignment (SWA) via MPPI for Multiple-Choice QA.

    Operates in 4-dimensional choice space {A, B, C, D} instead of binary
    LEFT/RIGHT. Each dimension represents a choice option's selection logit.

    Pipeline per question:
      1. Batched forward pass: base + N persona agents → 4-choice logits
      2. Compute contrastive rewards in 4-dim probability space
      3. Detect inter-agent conflict via variance threshold (tau)
      4. If conflict: MPPI optimization with Prospect Theory + social cooperation
      5. Positional debiasing: run 4 cyclic rotations, average per-choice probs
      6. Select answer: argmax of debiased distribution
    """

    def __init__(
        self,
        model,
        tokenizer,
        personas: List[str],
        config: Optional[CulturalMCConfig] = None,
    ):
        self.config = config or CulturalMCConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.personas = personas
        self.N = len(personas)
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

        self._resolve_choice_token_ids()
        self._build_persona_prefixes()

    # ------------------------------------------------------------------
    # Token ID resolution for A/B/C/D
    # ------------------------------------------------------------------
    def _resolve_choice_token_ids(self):
        """
        Resolve token IDs for choice letters A, B, C, D.

        The prompt ends with "\\nAnswer:" so the model's next token should be
        " A" (with space). We resolve both " A" and "A" and pick the variant
        that the tokenizer prefers.
        """
        self.choice_letters = ["A", "B", "C", "D"]
        choice_ids = []

        for letter in self.choice_letters:
            # Try with leading space first (more common after "Answer:")
            spaced = f" {letter}"
            ids_spaced = self.tokenizer.encode(
                spaced, add_special_tokens=False)
            ids_bare = self.tokenizer.encode(
                letter, add_special_tokens=False)

            # Use the last token (in case of multi-token encoding)
            # Prefer spaced variant as it matches typical generation context
            if len(ids_spaced) > 0:
                choice_ids.append(ids_spaced[-1])
            elif len(ids_bare) > 0:
                choice_ids.append(ids_bare[-1])
            else:
                raise ValueError(
                    f"Tokenizer could not encode '{letter}' — "
                    f"check model vocabulary.")

        self.choice_ids = torch.tensor(choice_ids, device=self.device)
        print(f"[SWA-MC] Choice token IDs — "
              f"A: {choice_ids[0]}, B: {choice_ids[1]}, "
              f"C: {choice_ids[2]}, D: {choice_ids[3]}")

    # ------------------------------------------------------------------
    # Persona prefix construction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_persona_prefixes(self):
        """Tokenize persona system prompts for prefix-caching."""
        print(f"[SWA-MC] Building prefixes for {self.N} personas + 1 base...")
        t0 = time.time()

        self.persona_prefix_ids = []
        for persona_text in self.personas:
            prefix = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{persona_text}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
            )
            ids = self.tokenizer(
                prefix, return_tensors="pt"
            ).input_ids.to(self.device)
            self.persona_prefix_ids.append(ids)

        base_prefix = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        self.base_prefix_ids = self.tokenizer(
            base_prefix, return_tensors="pt"
        ).input_ids.to(self.device)

        elapsed = time.time() - t0
        print(f"[SWA-MC] Prefix tokenisation: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Core: batched forward pass for base + N personas
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_all_agents_mc(
        self,
        query_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single batched forward pass for base + N persona agents.
        Extract logits at the 4 choice token positions (A/B/C/D).

        Args:
            query_ids: (1, seq_len) tokenized MC prompt

        Returns:
            z_base:   (1, 4) temperature-scaled logits for base model
            z_agents: (N, 4) temperature-scaled logits for persona agents
        """
        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        seqs = [torch.cat([p, query_ids], dim=1) for p in all_prefixes]
        max_len = max(s.shape[1] for s in seqs)

        # Left-pad to same length for batched inference
        batch_ids, batch_mask = [], []
        for s in seqs:
            pad_len = max_len - s.shape[1]
            batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
            batch_mask.append(F.pad(
                torch.ones(1, s.shape[1], dtype=torch.long,
                           device=self.device),
                (pad_len, 0), value=0,
            ))

        batch_ids = torch.cat(batch_ids, dim=0)   # (N+1, max_len)
        batch_mask = torch.cat(batch_mask, dim=0)

        try:
            out = self.model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                use_cache=False,
            )
            last_logits = out.logits[:, -1, :]  # (N+1, vocab_size)
        except torch.cuda.OutOfMemoryError:
            # Fallback: sequential evaluation if OOM on batch
            print("[WARN] OOM on batched forward, falling back to sequential")
            torch.cuda.empty_cache()
            last_logits = []
            for i, s in enumerate(seqs):
                mask = torch.ones(1, s.shape[1], dtype=torch.long,
                                  device=self.device)
                out = self.model(
                    input_ids=s, attention_mask=mask, use_cache=False)
                last_logits.append(out.logits[:, -1, :])
            last_logits = torch.cat(last_logits, dim=0)

        # Extract logits at choice token positions and apply temperature
        z_choices = (
            last_logits[:, self.choice_ids].clamp(-100, 100)
            / self.logit_temp
        )  # (N+1, 4)

        z_base = z_choices[0:1]    # (1, 4)
        z_agents = z_choices[1:]   # (N, 4)
        return z_base, z_agents

    # ------------------------------------------------------------------
    # Reward computation in 4-choice space
    # ------------------------------------------------------------------
    def _compute_mc_rewards(
        self,
        z_base: torch.Tensor,
        z_agents: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Compute contrastive rewards in 4-choice probability space.

        Generalizes the binary delta = z_right - z_left to the full simplex.

        Args:
            z_base:   (1, 4) base model logits
            z_agents: (N, 4) persona agent logits

        Returns:
            r_agents:    (N, 4) contrastive reward vectors
            variance:    scalar conflict signal (sum of per-choice variances)
            p_consensus: (4,) consensus distribution over choices
        """
        p_base = F.softmax(z_base, dim=1)       # (1, 4)
        p_agents = F.softmax(z_agents, dim=1)   # (N, 4)

        # Contrastive reward: how each agent differs from base
        r_agents = p_agents - p_base             # (N, 4)

        # Consensus: mean agent distribution
        p_consensus = p_agents.mean(dim=0)       # (4,)

        # Conflict signal: total variance across agents
        # Generalizes var(delta) from binary to sum of per-choice variances
        variance = torch.var(p_agents, dim=0).sum().item()

        return r_agents, variance, p_consensus

    # ------------------------------------------------------------------
    # Prospect Theory value function
    # ------------------------------------------------------------------
    def _prospect_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prospect Theory value function (Kahneman & Tversky, 1979).

        v(x) =  x^alpha           if x >= 0   (diminishing gains)
        v(x) = -kappa * |x|^beta  if x < 0    (loss aversion)
        """
        return torch.where(
            x >= 0,
            x.abs().pow(self.pt_alpha),
            -self.pt_kappa * x.abs().pow(self.pt_beta),
        )

    # ------------------------------------------------------------------
    # MPPI optimization in 4-dim simplex
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _mppi_solve_mc(
        self,
        p_consensus: torch.Tensor,
        r_agents: torch.Tensor,
    ) -> torch.Tensor:
        """
        MPPI optimization over the 4-choice probability simplex.

        Finds optimal p* over {A, B, C, D} by:
          1. Sample K perturbations in log-space around consensus
          2. Project to simplex via softmax
          3. Evaluate socially-weighted Prospect Theory utility per agent
          4. Compute MPPI weighted average

        Args:
            p_consensus: (4,) consensus distribution
            r_agents:    (N, 4) contrastive reward matrix

        Returns:
            p_star: (4,) MPPI-optimal distribution over choices
        """
        C = p_consensus.shape[0]  # 4
        N = r_agents.shape[0]

        # Sample K perturbations in log-probability space
        log_consensus = torch.log(p_consensus.clamp(min=1e-8))  # (4,)
        epsilon = (
            torch.randn(self.K, C, device=self.device) * self.noise_std
        )  # (K, 4)

        # Project onto probability simplex via softmax
        p_pert = F.softmax(
            log_consensus.unsqueeze(0) + epsilon, dim=1
        )  # (K, 4)

        # KL divergence penalty: D_KL(p_pert || p_consensus)
        kl_penalty = (
            p_pert * torch.log(
                p_pert / p_consensus.unsqueeze(0).clamp(min=1e-8)
            )
        ).sum(dim=1)  # (K,)

        # Compute socially-weighted utility for each perturbation
        U_total = torch.zeros(self.K, device=self.device)

        for i in range(N):
            # Agent i's expected contrastive reward under perturbation
            r_i_k = (r_agents[i].unsqueeze(0) * p_pert).sum(dim=1)  # (K,)

            # Other agents' average reward
            if N > 1:
                r_others = (r_agents.sum(0) - r_agents[i]) / (N - 1)
            else:
                r_others = r_agents[i]
            r_others_k = (
                r_others.unsqueeze(0) * p_pert
            ).sum(dim=1)  # (K,)

            # Apply Prospect Theory value function
            u_private = self._prospect_value(r_i_k)
            u_social = self._prospect_value(r_others_k)

            # Social cooperation weighting
            u_i = ((1 - self.lambda_coop) * u_private
                   + self.lambda_coop * u_social)
            U_total += u_i

        U_total /= N
        U_total -= self.alpha_kl * kl_penalty

        # MPPI weighted average
        weights = F.softmax(U_total / self.beta, dim=0)  # (K,)
        p_star = (weights.unsqueeze(1) * p_pert).sum(dim=0)  # (4,)

        return p_star

    # ------------------------------------------------------------------
    # Adaptive tau calibration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_tau(
        self,
        sample_prompts: List[str],
        target_trigger_rate: Optional[float] = None,
    ) -> float:
        """Disabled: MPPI is now always-on."""
        return self.tau_conflict

    # ------------------------------------------------------------------
    # Prompt tokenization helper
    # ------------------------------------------------------------------
    def _tokenize_mc_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize an MC prompt for the Llama chat template."""
        formatted = (
            prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        query_ids = self.tokenizer(
            formatted, return_tensors="pt"
        ).input_ids.to(self.device)
        # Strip BOS if tokenizer added one (prefix already has it)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]
        return query_ids

    # ------------------------------------------------------------------
    # Prompt permutation for positional debiasing
    # ------------------------------------------------------------------
    @staticmethod
    def _build_permuted_prompt(
        prompt: str,
        choices: Dict[str, str],
        permutation: List[int],
    ) -> Tuple[str, Dict[int, int]]:
        """
        Rebuild MC prompt with choices in a new order.

        Args:
            prompt: original MC prompt string
            choices: {"A": "text1", "B": "text2", "C": "text3", "D": "text4"}
            permutation: e.g. [1, 2, 3, 0] means position 0 gets original
                         choice B, position 1 gets C, etc.

        Returns:
            (permuted_prompt, position_to_original_map)
            position_to_original_map: {0: 1, 1: 2, 2: 3, 3: 0} means
              new position 0 came from original index 1 (choice B)
        """
        original_letters = sorted(choices.keys())  # ['A', 'B', 'C', 'D']
        original_texts = [choices[k] for k in original_letters]

        # Split prompt: find the choice lines and question stem
        # Typical format: "Question text...\n\nA. text\nB. text\nC. text\nD. text\n\nAnswer:"
        lines = prompt.split("\n")

        # Find choice line indices — only match lines that start with
        # exactly "A. ", "B. ", etc. (with period+space) to avoid false
        # positives like "According to B. Smith..."
        choice_line_indices = []
        choice_line_texts = {}
        for idx, line in enumerate(lines):
            stripped = line.strip()
            for letter in original_letters:
                prefix = f"{letter}. "
                if stripped.startswith(prefix) or stripped == f"{letter}.":
                    choice_line_indices.append(idx)
                    choice_line_texts[letter] = stripped[len(f"{letter}."):]
                    break

        if len(choice_line_indices) < len(original_letters):
            # Could not parse choice lines — return original prompt
            identity_map = {i: i for i in range(len(original_letters))}
            return prompt, identity_map

        # Build position-to-original map
        pos_to_orig = {}
        for new_pos, orig_idx in enumerate(permutation):
            pos_to_orig[new_pos] = orig_idx

        # Rebuild choice lines in permuted order
        new_letters = ["A", "B", "C", "D"]
        for new_pos, orig_idx in enumerate(permutation):
            orig_letter = original_letters[orig_idx]
            orig_text = choice_line_texts.get(
                orig_letter,
                original_texts[orig_idx] if orig_idx < len(original_texts) else ""
            )
            new_line = f"{new_letters[new_pos]}.{orig_text}"
            line_idx = choice_line_indices[new_pos]
            lines[line_idx] = new_line

        permuted_prompt = "\n".join(lines)
        return permuted_prompt, pos_to_orig

    # ------------------------------------------------------------------
    # Single-pass prediction (one rotation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_single_pass(
        self,
        prompt: str,
    ) -> Dict:
        """
        Single forward pass for an MC prompt. Returns prediction dict.

        Returns:
            dict with p_choices (4,), variance, mppi_triggered, etc.
        """
        query_ids = self._tokenize_mc_prompt(prompt)
        z_base, z_agents = self._evaluate_all_agents_mc(query_ids)
        r_agents, variance, p_consensus = self._compute_mc_rewards(
            z_base, z_agents)

        mppi_triggered = True  # Forced always-on
        p_star = self._mppi_solve_mc(p_consensus, r_agents)

        selected_idx = torch.argmax(p_star).item()

        # Check if MPPI flipped the answer
        consensus_choice = torch.argmax(p_consensus).item()
        mppi_flipped = mppi_triggered and (selected_idx != consensus_choice)

        return {
            "p_choices": p_star,           # (4,) tensor
            "p_consensus": p_consensus,    # (4,) tensor
            "variance": variance,
            "mppi_triggered": mppi_triggered,
            "mppi_flipped": mppi_flipped,
            "selected_idx": selected_idx,
            "z_base": z_base,
            "z_agents": z_agents,
        }

    # ------------------------------------------------------------------
    # Full prediction with positional debiasing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_mc(
        self,
        prompt: str,
        choices_json: str,
    ) -> Dict:
        """
        Run SWA-MPPI with positional debiasing for MC.

        Runs P cyclic rotations of choice order, maps probabilities back
        to original choices, and averages to cancel positional bias.

        Args:
            prompt: the full MC prompt string
            choices_json: JSON string of choices {"A": "...", ...}

        Returns:
            dict with predicted_answer, p_debiased, diagnostics
        """
        try:
            choices = (
                json.loads(choices_json)
                if isinstance(choices_json, str)
                else choices_json
            )
        except (json.JSONDecodeError, TypeError):
            # Cannot parse choices — run single pass without debiasing
            result = self._predict_single_pass(prompt)
            letter = self.choice_letters[result["selected_idx"]]
            return {
                "predicted_answer": letter,
                "p_debiased": result["p_choices"].tolist(),
                "selected_idx": result["selected_idx"],
                "mppi_triggered": result["mppi_triggered"],
                "mppi_flipped": result["mppi_flipped"],
                "variance": result["variance"],
                "positional_bias": 0.0,
                "n_rotations": 1,
            }

        n_choices = len(choices)

        # Generate cyclic rotations
        rotations = []
        for shift in range(min(self.config.n_rotations, n_choices)):
            perm = [(i + shift) % n_choices for i in range(n_choices)]
            rotations.append(perm)

        # Accumulate per-original-choice probabilities
        accumulated_p = torch.zeros(n_choices, device=self.device)
        any_triggered = False
        any_flipped = False
        all_variances = []
        all_mapped_p = []  # per-rotation probs mapped to original choice space

        for perm in rotations:
            permuted_prompt, pos_to_orig = self._build_permuted_prompt(
                prompt, choices, perm)

            result = self._predict_single_pass(permuted_prompt)

            p_choices = result["p_choices"]  # (4,) over positions in permuted

            # Map position probabilities back to original choice indices
            mapped_p = torch.zeros(n_choices, device=self.device)
            for new_pos in range(n_choices):
                orig_idx = pos_to_orig[new_pos]
                mapped_p[orig_idx] = p_choices[new_pos]
                accumulated_p[orig_idx] += p_choices[new_pos]

            all_mapped_p.append(mapped_p)

            if result["mppi_triggered"]:
                any_triggered = True
            if result["mppi_flipped"]:
                any_flipped = True
            all_variances.append(result["variance"])

        # Average over rotations
        p_debiased = accumulated_p / len(rotations)
        selected_idx = torch.argmax(p_debiased).item()
        choice_letters = sorted(choices.keys())
        predicted_answer = (
            choice_letters[selected_idx]
            if selected_idx < len(choice_letters)
            else "A"
        )

        # Compute positional bias: variance of per-original-choice probs
        # across rotations (high = model is sensitive to option position)
        if len(all_mapped_p) > 1:
            stacked = torch.stack(all_mapped_p)  # (n_rot, n_choices)
            positional_bias = torch.var(stacked, dim=0).sum().item()
        else:
            positional_bias = 0.0

        return {
            "predicted_answer": predicted_answer,
            "p_debiased": p_debiased.tolist(),
            "selected_idx": selected_idx,
            "mppi_triggered": any_triggered,
            "mppi_flipped": any_flipped,
            "variance": float(np.mean(all_variances)),
            "positional_bias": positional_bias,
            "n_rotations": len(rotations),
        }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_country_experiment(
    model,
    tokenizer,
    country: str,
    mc_df: pd.DataFrame,
    config: CulturalMCConfig,
) -> Dict:
    """
    Run full SWA-MPPI MC experiment for one country.

    Steps:
      1. Build cultural personas
      2. Create CulturalMCController
      3. Calibrate tau on subset
      4. Run prediction on all MC questions
      5. Evaluate accuracy
    """
    print(f"\n{'=' * 60}")
    print(f"  COUNTRY: {country}")
    print(f"{'=' * 60}")

    country_df = mc_df[mc_df["country"] == country].copy()
    if len(country_df) == 0:
        print(f"[WARN] No questions for {country}")
        return {"country": country, "accuracy": 0.0, "n_total": 0}

    # --- Step 1: Build personas ---
    personas = build_mc_personas(country, config)
    print(f"[SWA-MC] {len(personas)} personas for {country}")
    for i, p in enumerate(personas):
        print(f"  Persona {i}: {p[:80]}...")

    # --- Step 2: Create controller ---
    controller = CulturalMCController(
        model=model,
        tokenizer=tokenizer,
        personas=personas,
        config=config,
    )

    # --- Step 3: MPPI Trigger (Always-On) ---
    tau = config.tau_conflict
    print(f"[SWA-MC] Using always-on MPPI (tau={tau})")

    # --- Step 4: Run predictions ---
    output_filename = str(
        RESULTS_DIR / f"swa_mppi_mc_{country}_results.csv"
    )

    # Resume support
    done_ids = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename, encoding="utf-8")
        done_ids = set(already["MCQID"])
        print(f"  Resuming: {len(done_ids)} already done")
    else:
        write_csv_row(
            ["MCQID", "ID", "country", "answer_idx", "predicted_answer",
             "correct", "p_A", "p_B", "p_C", "p_D",
             "variance", "mppi_triggered", "mppi_flipped",
             "positional_bias", "n_rotations"],
            output_filename,
        )

    n_correct = 0
    n_done = 0
    n_triggered = 0
    n_flipped = 0

    pb = tqdm(country_df.iterrows(), desc=f"SWA-MC/{country}",
              total=len(country_df))

    for _, row in pb:
        mcqid = row["MCQID"]

        if mcqid in done_ids:
            continue  # don't count skipped rows in running accuracy

        prompt = row["prompt"]
        choices_json = row["choices"]
        answer_idx = str(row["answer_idx"]).strip()

        # Run SWA-MPPI prediction
        result = controller.predict_mc(prompt, choices_json)

        predicted = result["predicted_answer"]
        correct = int(predicted == answer_idx)

        n_done += 1
        n_correct += correct
        if result["mppi_triggered"]:
            n_triggered += 1
        if result["mppi_flipped"]:
            n_flipped += 1

        # Extract per-choice probabilities
        p_list = result["p_debiased"]
        p_A = p_list[0] if len(p_list) > 0 else 0.0
        p_B = p_list[1] if len(p_list) > 1 else 0.0
        p_C = p_list[2] if len(p_list) > 2 else 0.0
        p_D = p_list[3] if len(p_list) > 3 else 0.0

        # Save row
        write_csv_row(
            [mcqid, row["ID"], country, answer_idx, predicted,
             correct, f"{p_A:.6f}", f"{p_B:.6f}", f"{p_C:.6f}", f"{p_D:.6f}",
             f"{result['variance']:.6f}",
             int(result["mppi_triggered"]),
             int(result["mppi_flipped"]),
             f"{result['positional_bias']:.6f}",
             result["n_rotations"]],
            output_filename,
        )

        if n_done > 0:
            pb.set_postfix({
                "acc": f"{n_correct / n_done * 100:.1f}%",
                "trig": f"{n_triggered / n_done * 100:.0f}%",
            })

    # --- Step 5: Evaluate ---
    accuracy = n_correct / n_done * 100 if n_done > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_correct, n_done)
    trigger_rate = n_triggered / n_done * 100 if n_done > 0 else 0.0
    flip_rate = n_flipped / n_done * 100 if n_done > 0 else 0.0

    print(f"\n  >>> {country} Results:")
    print(f"      Accuracy:     {accuracy:.2f}% [{ci_lo:.1f}%, {ci_hi:.1f}%]")
    print(f"      Trigger rate: {trigger_rate:.1f}%")
    print(f"      Flip rate:    {flip_rate:.1f}%")
    print(f"      Tau:          {tau:.6f}")

    return {
        "country": country,
        "accuracy": accuracy,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_total": n_done,
        "n_correct": n_correct,
        "mppi_trigger_rate": trigger_rate,
        "mppi_flip_rate": flip_rate,
        "tau_calibrated": tau,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def print_mc_summary(results: List[Dict], config: CulturalMCConfig):
    """Print final summary table."""
    print(f"\n{'=' * 80}")
    print(f"{'SWA-MPPI MULTIPLE-CHOICE EVALUATION RESULTS':^80}")
    print(f"{'=' * 80}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Method:      SWA-MPPI (lambda={config.lambda_coop}, "
          f"K={config.K_samples}, rotations={config.n_rotations})")
    print(f"  Q/country:   {config.n_per_country}")
    print(f"{'─' * 80}")
    print(f"  {'Country':<15s} {'Acc':>7s} {'95% CI':>16s} "
          f"{'N':>5s} {'Trig%':>6s} {'Flip%':>6s} {'Tau':>10s}")
    print(f"  {'─' * 15} {'─' * 7} {'─' * 16} "
          f"{'─' * 5} {'─' * 6} {'─' * 6} {'─' * 10}")

    total_n = 0
    total_correct = 0
    all_acc = []

    for r in results:
        print(f"  {r['country']:<15s} {r['accuracy']:>6.2f}% "
              f"[{r['ci_lower']:>5.1f}%, {r['ci_upper']:>5.1f}%] "
              f"{r['n_total']:>5d} "
              f"{r['mppi_trigger_rate']:>5.1f}% "
              f"{r['mppi_flip_rate']:>5.1f}% "
              f"{r.get('tau_calibrated', 0):>10.6f}")
        all_acc.append(r["accuracy"])
        total_n += r["n_total"]
        total_correct += r["n_correct"]

    overall = total_correct / total_n * 100 if total_n > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(total_correct, total_n)

    print(f"  {'─' * 80}")
    print(f"  {'OVERALL':<15s} {overall:>6.2f}% "
          f"[{ci_lo:>5.1f}%, {ci_hi:>5.1f}%] "
          f"{total_n:>5d}")
    print(f"  {'Mean (macro)':<15s} {np.mean(all_acc):>6.2f}%")
    print(f"{'=' * 80}")

    print(f"\n  Random baseline (4 choices): 25.00%")
    print(f"  Model advantage over random: {overall - 25.0:+.2f}pp")


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

TUNE_COUNTRY = "UK"
TUNE_SEED = 42

TUNE_DIR = WORK_DIR / "tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)
TUNE_LOG = TUNE_DIR / "tune_results.csv"

# Search ranges — sampled randomly (log-uniform for noise_std & temperature)
PARAM_RANGES = {
    "noise_std":       (0.005, 0.5,   "log"),
    "lambda_coop":     (0.0,   0.9,   "linear"),
    "temperature":     (0.01,  1.0,   "log"),
}

PHASE1_N_SAMPLES = 200
PHASE1_N_QUESTIONS = 50
PHASE2_N_QUESTIONS = 100


def _sample_random_configs(n: int, seed: int = 42) -> list:
    """Sample n random configs from param ranges."""
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n):
        params = {}
        for name, (lo, hi, scale) in PARAM_RANGES.items():
            if scale == "log":
                val = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            else:
                val = rng.uniform(lo, hi)
            params[name] = round(float(val), 4)
        configs.append(params)
    return configs


def _fine_grid_around(best: dict) -> dict:
    """Phase 2: fine grid +/- small step around Phase 1 winner."""
    ns = best["noise_std"]
    lc = best["lambda_coop"]
    t = best["temperature"]

    ns_step = max(0.002, ns * 0.3)
    t_step = max(0.005, t * 0.3)
    lc_step = 0.05

    return {
        "noise_std": sorted(set([
            round(max(0.001, ns - ns_step), 4),
            round(ns, 4),
            round(ns + ns_step, 4),
        ])),
        "lambda_coop": sorted(set([
            round(max(0.0, lc - lc_step), 4),
            round(lc, 4),
            round(min(0.95, lc + lc_step), 4),
        ])),
        "temperature": sorted(set([
            round(max(0.005, t - t_step), 4),
            round(t, 4),
            round(t + t_step, 4),
        ])),
    }


def _evaluate_tune_config(model, tokenizer, mc_df, config, personas,
                          config_name=""):
    """Quick evaluate a single config on tune set (no CSV, no tqdm)."""
    controller = CulturalMCController(
        model=model, tokenizer=tokenizer,
        personas=personas, config=config,
    )
    n_correct = 0
    n_total = 0
    t0 = time.time()

    for _, row in mc_df.iterrows():
        result = controller.predict_mc(row["prompt"], row["choices"])
        if result["predicted_answer"] == str(row["answer_idx"]).strip():
            n_correct += 1
        n_total += 1

    elapsed = time.time() - t0
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_correct, n_total)

    print(f"  [{config_name:>25s}] "
          f"acc={accuracy:5.1f}% [{ci_lo:.1f}-{ci_hi:.1f}] "
          f"t={elapsed:.0f}s ({elapsed/max(n_total,1):.2f}s/q)")

    return {
        "config_name": config_name,
        "accuracy": accuracy, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "n_correct": n_correct, "n_total": n_total,
        "elapsed_sec": elapsed, "sec_per_q": elapsed / max(n_total, 1),
    }


_TUNE_LOG_INITIALIZED = False

def _log_tune_result(result: dict):
    global _TUNE_LOG_INITIALIZED
    # Overwrite on first call per run, then append
    mode = "w" if not _TUNE_LOG_INITIALIZED else "a"
    write_header = not _TUNE_LOG_INITIALIZED
    with open(TUNE_LOG, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    _TUNE_LOG_INITIALIZED = True


def run_random_search(model, tokenizer, mc_df, personas, base_config):
    """Phase 1: random search over full param ranges."""
    configs = _sample_random_configs(PHASE1_N_SAMPLES, seed=TUNE_SEED)

    print(f"\n{'='*60}")
    print(f"  Phase 1: Random search")
    print(f"  {PHASE1_N_SAMPLES} configs x {len(mc_df)} questions")
    est_min = PHASE1_N_SAMPLES * len(mc_df) * 1.1 / 60
    print(f"  Estimated: ~{est_min:.0f} min")
    for name, (lo, hi, scale) in PARAM_RANGES.items():
        print(f"    {name}: [{lo}, {hi}] ({scale})")
    print(f"{'='*60}")

    results = []
    for i, params in enumerate(configs):
        label = (f"{i+1:>3d}/{PHASE1_N_SAMPLES} "
                 f"ns={params['noise_std']:.4f}|"
                 f"lc={params['lambda_coop']:.2f}|"
                 f"t={params['temperature']:.4f}")

        config = replace(base_config, **params)
        r = _evaluate_tune_config(
            model, tokenizer, mc_df, config, personas, label)
        r.update(params)
        r["phase"] = "phase1_random"
        results.append(r)
        _log_tune_result(r)

        if (i + 1) % 20 == 0:
            best_so_far = max(results, key=lambda x: x["accuracy"])
            print(f"  --- Best so far ({i+1}/{PHASE1_N_SAMPLES}): "
                  f"{best_so_far['accuracy']:.1f}% "
                  f"ns={best_so_far['noise_std']} "
                  f"lc={best_so_far['lambda_coop']} "
                  f"t={best_so_far['temperature']} ---")

    results.sort(key=lambda x: (-x["accuracy"], x["elapsed_sec"]))
    return results


def run_fine_grid(model, tokenizer, mc_df, personas, base_config, grid):
    """Phase 2: fine grid around best from Phase 1."""
    param_names = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in param_names)))

    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine grid")
    print(f"  {len(combos)} configs x {len(mc_df)} questions")
    est_min = len(combos) * len(mc_df) * 1.1 / 60
    print(f"  Estimated: ~{est_min:.0f} min")
    print(f"  Grid: {grid}")
    print(f"{'='*60}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        label = (f"ns={params['noise_std']}|"
                 f"lc={params['lambda_coop']}|"
                 f"t={params['temperature']}")

        config = replace(base_config, **params)
        r = _evaluate_tune_config(
            model, tokenizer, mc_df, config, personas, label)
        r.update(params)
        r["phase"] = "phase2_fine"
        results.append(r)
        _log_tune_result(r)

    results.sort(key=lambda x: (-x["accuracy"], x["elapsed_sec"]))
    return results


def run_full_evaluation(model, tokenizer, best_params: dict):
    """Phase 3: full eval on all countries with tuned config."""
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Full Evaluation with Tuned Hyperparameters")
    print(f"{'='*70}")
    print(f"  noise_std:   {best_params['noise_std']}")
    print(f"  lambda_coop: {best_params['lambda_coop']}")
    print(f"  temperature: {best_params['temperature']}")
    print(f"  Countries:   {MC_COUNTRIES}")
    print(f"  Q/country:   500")
    print(f"{'='*70}")

    config = CulturalMCConfig(
        noise_std=best_params["noise_std"],
        lambda_coop=best_params["lambda_coop"],
        temperature=best_params["temperature"],
    )

    # Clean old per-country result CSVs to avoid resume mixing old+new configs
    for country in MC_COUNTRIES:
        old_csv = RESULTS_DIR / f"swa_mppi_mc_{country}_results.csv"
        if old_csv.exists():
            old_csv.unlink()
            print(f"  Removed old results: {old_csv.name}")

    print("\n  Loading full MC dataset...")
    available_files = [f for f in MC_FILES
                       if os.path.exists(os.path.join(MC_DATA_DIR, f))]
    mc_df = load_mc_questions(
        mc_dir=MC_DATA_DIR,
        mc_files=available_files,
        countries=MC_COUNTRIES,
        n_per_country=config.n_per_country,
        seed=SAMPLE_SEED,
    )

    eval_result_file = str(EVAL_DIR / "swa_mppi_mc_tuned_evaluation.csv")
    with open(eval_result_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "method", "country", "accuracy", "ci_lower", "ci_upper",
             "n_total", "n_correct", "mppi_trigger_rate", "mppi_flip_rate",
             "tau_calibrated",
             "noise_std", "lambda_coop", "temperature"]
        )

    results = []
    t0_full = time.time()

    for i, country in enumerate(MC_COUNTRIES):
        print(f"\n  [{i+1}/{len(MC_COUNTRIES)}] {country}...")
        result = run_country_experiment(
            model, tokenizer, country, mc_df, config)
        results.append(result)

        write_csv_row(
            [MODEL_NAME, "SWA-MPPI-tuned", result["country"],
             result["accuracy"], result["ci_lower"], result["ci_upper"],
             result["n_total"], result["n_correct"],
             result["mppi_trigger_rate"], result["mppi_flip_rate"],
             result.get("tau_calibrated", 0),
             best_params["noise_std"], best_params["lambda_coop"],
             best_params["temperature"]],
            eval_result_file,
        )

        gc.collect()
        torch.cuda.empty_cache()

    elapsed_full = time.time() - t0_full
    print_mc_summary(results, config)
    print(f"\n  Full evaluation time: {elapsed_full/60:.1f} min")
    print(f"  Results saved to: {eval_result_file}")
    return results


# ============================================================================
# MAIN — Tune + Full Eval
# ============================================================================

def main():
    base_config = CulturalMCConfig()
    t0_total = time.time()

    print("=" * 70)
    print("SWA-MPPI Hyperparameter Tuning + Full Evaluation")
    print("=" * 70)
    print(f"  Phase 1: Random search {PHASE1_N_SAMPLES} configs x {PHASE1_N_QUESTIONS}q")
    print(f"  Phase 2: Fine grid ~27 configs x {PHASE2_N_QUESTIONS}q")
    print(f"  Phase 3: Full eval {len(MC_COUNTRIES)} countries x 500q")
    est_total = (PHASE1_N_SAMPLES * PHASE1_N_QUESTIONS * 1.1
                 + 27 * PHASE2_N_QUESTIONS * 1.1
                 + 2500 * 1.1) / 3600
    print(f"  Estimated total: ~{est_total:.1f} hours")
    print(f"  MC data: {MC_DATA_DIR}")
    print("=" * 70)

    # Verify MC data
    if not os.path.exists(MC_DATA_DIR):
        print(f"[ERROR] MC data directory not found: {MC_DATA_DIR}")
        sys.exit(1)
    available_files = [f for f in MC_FILES
                       if os.path.exists(os.path.join(MC_DATA_DIR, f))]
    if not available_files:
        print(f"[ERROR] No MC question files found in {MC_DATA_DIR}")
        sys.exit(1)

    # ---- Load tune data ----
    print("\n[1/6] Loading tune data...")
    full_val_df = load_mc_questions(
        mc_dir=MC_DATA_DIR,
        mc_files=available_files,
        countries=[TUNE_COUNTRY],
        n_per_country=PHASE2_N_QUESTIONS,
        seed=TUNE_SEED,
    )
    phase1_df = full_val_df.head(PHASE1_N_QUESTIONS)
    phase2_df = full_val_df

    # ---- Load model ONCE ----
    print("\n[2/6] Loading model...")
    model, tokenizer = load_model(base_config.model_name)

    # ---- Build personas ONCE ----
    print("\n[3/6] Building personas...")
    personas = build_mc_personas(TUNE_COUNTRY, base_config)

    # ---- Phase 1: Random search ----
    print("\n[4/6] Phase 1: Random search...")
    p1_results = run_random_search(
        model, tokenizer, phase1_df, personas, base_config)

    print(f"\n  Top 10 Phase 1:")
    for i, r in enumerate(p1_results[:10]):
        print(f"    #{i+1} acc={r['accuracy']:.1f}%  "
              f"ns={r['noise_std']:.4f} lc={r['lambda_coop']:.2f} "
              f"t={r['temperature']:.4f}")

    best_p1 = {
        "noise_std": p1_results[0]["noise_std"],
        "lambda_coop": p1_results[0]["lambda_coop"],
        "temperature": p1_results[0]["temperature"],
    }
    print(f"\n  Phase 1 best: {best_p1}")

    # ---- Phase 2: Fine grid ----
    print("\n[5/6] Phase 2: Fine grid...")
    grid2 = _fine_grid_around(best_p1)
    p2_results = run_fine_grid(
        model, tokenizer, phase2_df, personas, base_config, grid2)

    print(f"\n  Top 5 Phase 2:")
    for i, r in enumerate(p2_results[:5]):
        print(f"    #{i+1} acc={r['accuracy']:.1f}%  "
              f"ns={r['noise_std']} lc={r['lambda_coop']} "
              f"t={r['temperature']}")

    best_final = {
        "noise_std": p2_results[0]["noise_std"],
        "lambda_coop": p2_results[0]["lambda_coop"],
        "temperature": p2_results[0]["temperature"],
    }

    config_out = TUNE_DIR / "best_config.json"
    with open(config_out, "w") as f:
        json.dump(best_final, f, indent=2)

    tune_elapsed = time.time() - t0_total
    print(f"\n  Tuning done in {tune_elapsed/60:.1f} min")
    print(f"  Best: {best_final}")

    # ---- Phase 3: Full eval ----
    print("\n[6/6] Phase 3: Full evaluation...")
    run_full_evaluation(model, tokenizer, best_final)

    # ---- Cleanup ----
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    total_elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {total_elapsed/60:.1f} min total")
    print(f"{'='*70}")
    print(f"  Best params:  {best_final}")
    print(f"  Tune log:     {TUNE_LOG}")
    print(f"  Best config:  {config_out}")
    print(f"  Full results: {EVAL_DIR}")


if __name__ == "__main__":
    main()
