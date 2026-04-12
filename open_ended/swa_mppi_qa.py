#!/usr/bin/env python3
"""
SWA-MPPI for Culturally-Aware Short Question Answering
=======================================================
Adapts the Socially-Weighted Alignment via Model Predictive Path Integral
(SWA-MPPI) framework from binary ethical decisions to open-ended short QA.

Core idea: Candidate-Level MPPI
  Phase 1 — Generate M candidate answers via persona-steered decoding
  Phase 2 — Score candidates under all N cultural agents (log-prob matrix)
  Phase 3 — MPPI optimization in M-dim candidate space → select best answer

Theoretical contributions preserved:
  - Prospect Theory value function (Kahneman & Tversky, 1979)
  - Social cooperation via lambda_coop weighting
  - KL-regularized MPPI optimal control
  - Adaptive tau conflict detection
"""

import os
import gc
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Performance knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class CulturalQAConfig:
    """Hyperparameters for SWA-MPPI Cultural QA."""
    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

    # SWA-MPPI Core (same defaults as moral machine version)
    lambda_coop: float = 0.7        # balance private vs social reward
    alpha_kl: float = 0.05          # KL divergence penalty weight
    K_samples: int = 128            # number of MPPI perturbation samples
    noise_std: float = 0.3          # Gaussian perturbation std
    temperature: float = 0.5        # MPPI softmax temperature (beta)
    tau_conflict: float = 0.001     # variance threshold for MPPI trigger
    logit_temperature: float = 1.0  # temperature for candidate scoring softmax

    # Prospect Theory (Kahneman & Tversky, 1979)
    pt_alpha: float = 0.88          # gain curvature
    pt_beta: float = 0.88           # loss curvature
    pt_kappa: float = 2.25          # loss aversion coefficient

    # Candidate generation
    M_candidates: int = 8           # target number of diverse candidates
    max_new_tokens: int = 64        # max tokens per candidate answer
    gen_temperature: float = 0.7    # temperature for diversity sampling

    # Adaptive tau calibration
    tau_target_trigger_rate: float = 0.35
    tau_calibration_n: int = 50

    # Scoring batch size (adjust for GPU memory)
    score_batch_size: int = 8

    # WVS data path
    wvs_data_path: str = ""

    # BLEnD country → WVS ISO mapping
    BLEND_TO_WVS: Dict[str, str] = field(default_factory=lambda: {
        "UK": "GBR", "US": "USA", "South_Korea": "KOR", "China": "CHN",
        "Mexico": "MEX", "Northern_Nigeria": "NGA", "Azerbaijan": "AZE",
        "Algeria": "DZA", "Indonesia": "IDN", "Spain": "ESP", "Iran": "IRN",
        "Assam": "IND", "Greece": "GRC", "Ethiopia": "ETH",
        "North_Korea": "KOR",   # fallback to South Korea WVS
        "West_Java": "IDN",     # use Indonesia
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
# WVS-BASED CULTURAL PERSONA GENERATION
# ============================================================================
# WVS dimension definitions (reused from swa_mppi.py)
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
    from collections import defaultdict

    all_vars = set()
    for vars_list, _ in _WVS_DIMS.values():
        all_vars.update(vars_list)
    all_vars.add("Q261")   # Birth year
    all_vars.add("A_YEAR") # Survey year

    def _age_group(birth_year, survey_year):
        age = survey_year - birth_year
        if age < 36: return "young"
        if age < 56: return "middle"
        return "older"

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    try:
        with open(wvs_csv_path, 'r') as f:
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
    print(f"[WVS-QA] Loaded profiles for {n_loaded}/{len(target_countries)} countries")
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

    persona = (
        f"You are a {role} from {country_name}"
        f"{' ' + age_range if age_range else ''}. "
        f"You have deep knowledge of everyday life, food, customs, holidays, "
        f"sports, education, and family traditions in {country_name}. "
        f"Based on the cultural values of your society, you are {traits_str}. "
        f"Answer questions from your authentic personal experience as someone "
        f"who grew up in {country_name}."
    )
    return persona


# Fallback personas for countries without WVS data or as base
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
    """
    Build 4 cultural personas for a BLEnD country.

    Priority: WVS data (3 age-cohort + 1 cultural expert) → fallback base personas.
    """
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
                    personas.append(_generate_qa_persona(
                        wvs_iso, ag, p, country_name
                    ))

            # 4th persona: cultural expert
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

    # Fallback: check base personas by WVS ISO
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


# ============================================================================
# CORE: CulturalQAController — SWA-MPPI adapted for candidate-level QA
# ============================================================================
class CulturalQAController:
    """
    Socially-Weighted Alignment (SWA) via MPPI for Short Question Answering.

    Instead of binary LEFT/RIGHT logits, operates in M-dimensional candidate
    space where each dimension represents a candidate answer's selection weight.

    Pipeline:
      1. Generate M diverse candidates via persona-steered decoding
      2. Score each candidate under each cultural agent (log-prob matrix)
      3. Detect inter-agent conflict via variance threshold
      4. If conflict: MPPI optimization with Prospect Theory + social cooperation
      5. Select answer from optimal candidate distribution
    """

    def __init__(
        self,
        model,
        tokenizer,
        personas: List[str],
        config: Optional[CulturalQAConfig] = None,
    ):
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

    # ------------------------------------------------------------------
    # Prefix construction
    # ------------------------------------------------------------------
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

        elapsed = time.time() - t0
        print(f"[SWA-QA] Prefix tokenisation: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Phase 1: Candidate generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_single(
        self,
        prefix_ids: torch.Tensor,
        query_text: str,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """Generate one answer given a prefix and query."""
        suffix = (
            query_text
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        query_ids = self.tokenizer(
            suffix, return_tensors="pt"
        ).input_ids.to(self.device)

        # Strip BOS if tokenizer added one (prefix already has it)
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
    def _generate_candidates(self, query_text: str) -> List[str]:
        """
        Generate M diverse candidate answers.

        Strategy:
          1. Greedy from base model (vanilla baseline answer)
          2. Greedy from each persona (culturally-steered answers)
          3. Temperature-sampled from base if more diversity needed
        """
        seen = set()
        candidates = []

        def _add(text: str):
            normalized = text.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(text.strip())

        # Base model greedy
        _add(self._generate_single(self.base_prefix_ids, query_text, do_sample=False))

        # Persona-steered greedy
        for prefix in self.persona_prefix_ids:
            _add(self._generate_single(prefix, query_text, do_sample=False))

        # Temperature sampling for extra diversity if needed
        max_extra_attempts = self.M * 2  # avoid infinite loop
        attempts = 0
        while len(candidates) < self.M and attempts < max_extra_attempts:
            _add(self._generate_single(
                self.base_prefix_ids, query_text,
                do_sample=True, temperature=self.config.gen_temperature
            ))
            attempts += 1

        return candidates

    # ------------------------------------------------------------------
    # Phase 2: Candidate scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_log_prob(
        self,
        prefix_ids: torch.Tensor,
        query_text: str,
        answer_text: str,
    ) -> float:
        """
        Compute log p(answer | prefix, query) by teacher-forcing.

        Returns the mean log-probability per token of the answer.
        """
        suffix = (
            query_text
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        query_ids = self.tokenizer(
            suffix, return_tensors="pt"
        ).input_ids.to(self.device)
        if query_ids[0, 0] == self.tokenizer.bos_token_id:
            query_ids = query_ids[:, 1:]

        answer_ids = self.tokenizer(
            answer_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        # Full sequence: prefix + query + answer
        input_ids = torch.cat([prefix_ids, query_ids, answer_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Extract log-probs for answer tokens only
        # Answer tokens start at position (prefix_len + query_len)
        # The logit at position t predicts token at position t+1
        answer_start = prefix_ids.shape[1] + query_ids.shape[1]
        answer_len = answer_ids.shape[1]

        if answer_len == 0:
            return -100.0  # degenerate empty answer

        # Logits predicting answer tokens are at positions [answer_start-1, answer_start+answer_len-2]
        pred_logits = logits[0, answer_start - 1: answer_start + answer_len - 1, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)

        # Gather log-probs of actual answer tokens
        target_ids = answer_ids[0]  # (answer_len,)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        # Mean log-prob (length-normalized to avoid bias toward short answers)
        return token_log_probs.mean().item()

    @torch.no_grad()
    def _score_candidates(
        self,
        query_text: str,
        candidates: List[str],
    ) -> torch.Tensor:
        """
        Score all candidates under all agents.

        Returns:
            L: tensor of shape (N+1, M) where L[i, j] = log p(candidate_j | agent_i, query)
               Row 0 = base model, rows 1..N = persona agents.
        """
        M = len(candidates)
        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        n_agents = len(all_prefixes)

        L = torch.zeros(n_agents, M, device=self.device)

        for i, prefix in enumerate(all_prefixes):
            for j, candidate in enumerate(candidates):
                L[i, j] = self._compute_log_prob(prefix, query_text, candidate)

        return L

    # ------------------------------------------------------------------
    # Phase 3: Reward computation + MPPI optimization
    # ------------------------------------------------------------------
    def _compute_qa_rewards(
        self,
        L: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Compute contrastive rewards and conflict signal.

        Args:
            L: (N+1, M) log-prob matrix

        Returns:
            r_agents: (N, M) contrastive reward matrix
            variance: scalar conflict signal
            p_consensus: (M,) consensus candidate distribution
        """
        L_base = L[0:1]    # (1, M)
        L_agents = L[1:]   # (N, M)

        # Normalize to probability distributions over candidates
        p_base = F.softmax(L_base / self.logit_temp, dim=1)      # (1, M)
        p_agents = F.softmax(L_agents / self.logit_temp, dim=1)   # (N, M)

        # Contrastive reward: shift from base distribution
        r_agents = p_agents - p_base  # (N, M)

        # Consensus: average agent distribution
        p_consensus = p_agents.mean(dim=0)  # (M,)

        # Conflict: total variance of agent distributions across candidates
        variance = torch.var(p_agents, dim=0).sum().item()

        return r_agents, variance, p_consensus

    def _prospect_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prospect Theory value function (Kahneman & Tversky, 1979).

        v(x) =  x^alpha           if x >= 0   (diminishing sensitivity to gains)
        v(x) = -kappa * |x|^beta  if x < 0    (loss aversion)
        """
        return torch.where(
            x >= 0,
            x.abs().pow(self.pt_alpha),
            -self.pt_kappa * x.abs().pow(self.pt_beta),
        )

    @torch.no_grad()
    def _mppi_solve_qa(
        self,
        p_consensus: torch.Tensor,
        r_agents: torch.Tensor,
    ) -> torch.Tensor:
        """
        MPPI optimization in M-dimensional candidate space.

        Finds the optimal candidate distribution p* by sampling K perturbations,
        evaluating socially-weighted utility with Prospect Theory, and computing
        a weighted average.

        Args:
            p_consensus: (M,) consensus distribution over candidates
            r_agents: (N, M) contrastive reward matrix

        Returns:
            p_star: (M,) MPPI-optimal distribution over candidates
        """
        M = p_consensus.shape[0]
        N = r_agents.shape[0]

        # Sample K perturbations in log-probability space
        log_consensus = torch.log(p_consensus.clamp(min=1e-8))  # (M,)
        epsilon = torch.randn(self.K, M, device=self.device) * self.noise_std  # (K, M)

        # Project onto probability simplex via softmax
        p_pert = F.softmax(log_consensus.unsqueeze(0) + epsilon, dim=1)  # (K, M)

        # KL penalty: D_KL(p_pert || p_consensus)
        kl_penalty = (p_pert * torch.log(
            p_pert / p_consensus.unsqueeze(0).clamp(min=1e-8)
        )).sum(dim=1)  # (K,)

        # Compute socially-weighted utility for each perturbation
        U_total = torch.zeros(self.K, device=self.device)

        for i in range(N):
            # Agent i's expected reward under each perturbation
            r_i_k = (r_agents[i].unsqueeze(0) * p_pert).sum(dim=1)  # (K,)

            # Other agents' average expected reward
            if N > 1:
                r_others_vec = (r_agents.sum(0) - r_agents[i]) / (N - 1)  # (M,)
            else:
                r_others_vec = r_agents[i]  # fallback
            r_others_k = (r_others_vec.unsqueeze(0) * p_pert).sum(dim=1)  # (K,)

            # Apply Prospect Theory value function
            u_private = self._prospect_value(r_i_k)
            u_social = self._prospect_value(r_others_k)

            # Social cooperation weighting
            u_i = (1 - self.lambda_coop) * u_private + self.lambda_coop * u_social
            U_total += u_i

        U_total /= N
        U_total -= self.alpha_kl * kl_penalty

        # MPPI weighted average
        weights = F.softmax(U_total / self.beta, dim=0)  # (K,)
        p_star = (weights.unsqueeze(1) * p_pert).sum(dim=0)  # (M,)

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
        """
        Calibrate tau_conflict so MPPI fires on ~target_trigger_rate of questions.

        Args:
            sample_prompts: list of question prompts to calibrate on
            target_trigger_rate: fraction of questions that should trigger MPPI

        Returns:
            calibrated tau value
        """
        if target_trigger_rate is None:
            target_trigger_rate = self.config.tau_target_trigger_rate

        n_calib = min(len(sample_prompts), self.config.tau_calibration_n)
        variances = []

        print(f"[SWA-QA] Calibrating tau on {n_calib} samples (target trigger: {target_trigger_rate:.0%})...")

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

        print(
            f"[SWA-QA] Calibrated tau = {tau_calibrated:.6f} "
            f"(percentile: {percentile:.0f}th of {len(variances)} samples)"
        )
        return tau_calibrated

    # ------------------------------------------------------------------
    # Main prediction entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_qa(self, query_text: str) -> Dict:
        """
        Run SWA-MPPI prediction for a short-answer question.

        Pipeline:
          1. Generate M diverse candidate answers
          2. Score candidates under all cultural agents
          3. Detect conflict via variance threshold
          4. If conflict → MPPI optimization; else → use consensus
          5. Select best candidate

        Args:
            query_text: the full question prompt (already formatted with template)

        Returns:
            dict with keys:
              - answer: selected answer string
              - candidates: list of all candidate answers
              - p_star: optimal distribution over candidates
              - mppi_triggered: whether MPPI was activated
              - variance: inter-agent conflict signal
              - selected_idx: index of chosen candidate
        """
        # Phase 1: Generate candidates
        candidates = self._generate_candidates(query_text)

        if len(candidates) == 0:
            return {
                "answer": "",
                "candidates": [],
                "p_star": [],
                "mppi_triggered": False,
                "variance": 0.0,
                "selected_idx": -1,
            }

        if len(candidates) == 1:
            return {
                "answer": candidates[0],
                "candidates": candidates,
                "p_star": [1.0],
                "mppi_triggered": False,
                "variance": 0.0,
                "selected_idx": 0,
            }

        # Phase 2: Score candidates
        L = self._score_candidates(query_text, candidates)

        # Phase 3: Compute rewards and detect conflict
        r_agents, variance, p_consensus = self._compute_qa_rewards(L)

        mppi_triggered = variance >= self.tau_conflict

        if mppi_triggered:
            # Phase 3b: MPPI optimization
            p_star = self._mppi_solve_qa(p_consensus, r_agents)
            selected_idx = torch.argmax(p_star).item()
        else:
            # Use consensus directly
            p_star = p_consensus
            selected_idx = torch.argmax(p_consensus).item()

        answer = candidates[selected_idx]

        return {
            "answer": answer,
            "candidates": candidates,
            "p_star": p_star.tolist(),
            "p_consensus": p_consensus.tolist(),
            "mppi_triggered": mppi_triggered,
            "variance": variance,
            "selected_idx": selected_idx,
            "r_agents": r_agents.tolist(),
            "L_scores": L.tolist(),
        }
