#!/usr/bin/env python3
"""
EXP-00: BLEnD Open-Ended Evaluation — Vanilla + SWA-PTIS on Kaggle H100
=========================================================================
Extends the open-ended evaluation from:
  "BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures
   and Languages" (NeurIPS 2024 Datasets and Benchmarks Track)

Bootstraps the cultural_alignment repo via git clone (like kaggle_experiment.py)
to reuse WVS-based personas from src/personas.py.

Two evaluation modes:
  1. Vanilla baseline — direct model response, scored by LLM judge
  2. SWA-PTIS — WVS persona-augmented responses with PT-IS selection

SWA-PTIS adaptation for open-ended (vs binary choice):
  Binary SWA:    logit_gap(A,B) -> IS on continuous scalars -> probability
  Open-ended:    judge_score(response) -> PT-IS weighted response selection

  For each question:
    1. Generate vanilla response (base/anchor)
    2. Generate N persona responses (WVS cultural agents via src/personas.py)
    3. Score all N+1 with LLM judge against human annotations (1-5 Likert)
    4. Per-agent gain:  g_i = (s_persona_i - s_vanilla) / sigma
       Consensus gain:  g_cons_i = (s_i - s_mean) / sigma
    5. PT utility: U_i = (1-lambda)*v(g_i) + lambda*v(g_cons_i)
    6. IS weights: w_i = softmax(U_i / beta)
    7. ESS gate: if k_eff/N < rho_eff -> reject IS, keep vanilla
    8. Select response with highest IS weight

Countries: 7 BLEnD countries that overlap with WVS Wave 7 data:
  US(USA), UK(GBR), South_Korea(KOR), China(CHN),
  Indonesia(IDN), Iran(IRN), Mexico(MEX)

Usage on Kaggle:
  1. Upload the BLEnD dataset (data/ folder) as a Kaggle dataset
  2. Select GPU H100 80GB runtime
  3. !python experiment_open_ended/experiment_00.py
"""

# ============================================================================
# 0. ENVIRONMENT SETUP — env vars MUST be set before any torch import
# ============================================================================
import os
import sys
import subprocess

# H100: let torch.compile work (fast CUDA graphs). Disable only for local dev.
if not os.path.exists("/kaggle/working"):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

# ============================================================================
# 1. BOOTSTRAP — git clone repo + pip install (same as kaggle_experiment.py)
# ============================================================================
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    """Make sure we are inside the cultural_alignment repo."""
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError(
            "Not running on Kaggle and not inside the cultural_alignment repo. "
            "cd into the repo or run on Kaggle."
        )
    if not os.path.isdir(REPO_DIR_KAGGLE):
        print(f"[BOOTSTRAP] git clone {REPO_URL} -> {REPO_DIR_KAGGLE}")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE],
            check=True,
        )
    else:
        print(f"[BOOTSTRAP] Repo already at {REPO_DIR_KAGGLE}")
    os.chdir(REPO_DIR_KAGGLE)
    if REPO_DIR_KAGGLE not in sys.path:
        sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    print("[BOOTSTRAP] Installing dependencies ...")
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
# 2. IMPORTS — now safe to import torch + project code
# ============================================================================
import gc
import csv
import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

# Suppress noisy HF deprecation warnings (AttentionMaskConverter, max_length)
warnings.filterwarnings("ignore", message=".*attention mask API.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Also suppress HF logger warnings (these use logging, not warnings module)
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import unsloth  # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

if not _on_kaggle():
    try:
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

import torch.nn.functional as F
from tqdm.auto import tqdm

# Project imports (require _ensure_repo())
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.model import setup_seeds

# ============================================================================
# 3. CONFIG
# ============================================================================

# ---------------------------------------------------------------------------
# Model — Llama 3.1 8B Instruct (4-bit via unsloth, ~5GB on H100)
# ---------------------------------------------------------------------------
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODEL_NAME = "Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 512
SEED = 42

# H100 optimization
BATCH_SIZE = 8  # 8B 4-bit uses ~5GB, leaves 75GB for batch KV cache

# ---------------------------------------------------------------------------
# SWA-PTIS hyperparameters
# ---------------------------------------------------------------------------
PT_ALPHA = 0.88
PT_BETA_EXP = 0.88
PT_KAPPA = 2.25
LAMBDA_COOP = 0.7
IS_TEMPERATURE = 0.1
RHO_EFF = 0.3
SIGMA_FLOOR = 0.5

# ---------------------------------------------------------------------------
# BLEnD x WVS overlap — countries present in BOTH datasets
# BLEnD name -> (ISO code for WVS, language name for BLEnD prompts)
# ---------------------------------------------------------------------------
# Full 7 countries (uncomment when ready):
# BLEND_WVS_COUNTRIES: Dict[str, Tuple[str, str]] = {
#     "US":          ("USA", "English"),
#     "UK":          ("GBR", "English"),
#     "South_Korea": ("KOR", "Korean"),
#     "China":       ("CHN", "Chinese"),
#     "Indonesia":   ("IDN", "Indonesian"),
#     "Iran":        ("IRN", "Persian"),
#     "Mexico":      ("MEX", "Spanish"),
# }

# 5 countries for faster runs
BLEND_WVS_COUNTRIES: Dict[str, Tuple[str, str]] = {
    "US":          ("USA", "English"),
    "UK":          ("GBR", "English"),
    "South_Korea": ("KOR", "Korean"),
    "China":       ("CHN", "Chinese"),
    "Iran":        ("IRN", "Persian"),
}

PROMPT_NOS: List[str] = ["inst-4"]

# ---------------------------------------------------------------------------
# Paths — BLEnD data + WVS data
# ---------------------------------------------------------------------------
if _on_kaggle():
    _INPUT_DATA = "/kaggle/input/blend-data/data"
    _WORKING_DATA = "/kaggle/working/blend_data/data"
    DATA_ROOT = (
        _INPUT_DATA
        if os.path.exists(os.path.join(_INPUT_DATA, "questions"))
        else _WORKING_DATA
    )
    WORK_DIR = Path("/kaggle/working/blend_open_ended_results")
    WVS_DATA_PATH = (
        "/kaggle/input/datasets/trungkiet/mutltitp-data/"
        "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    )
else:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "blend_open_ended_results"
    WVS_DATA_PATH = ""


def _download_blend_from_hf():
    """Download BLEnD dataset from HuggingFace."""
    print("[DATA] Downloading BLEnD from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        dl_dir = os.path.dirname(DATA_ROOT)
        os.makedirs(dl_dir, exist_ok=True)
        snapshot_download(
            repo_id="nayeon212/BLEnD", repo_type="dataset",
            local_dir=dl_dir, allow_patterns=["data/**"],
        )
        if os.path.exists(os.path.join(DATA_ROOT, "questions")):
            print("[DATA] Download complete.")
        else:
            import shutil
            if os.path.exists(os.path.join(dl_dir, "questions")):
                os.makedirs(DATA_ROOT, exist_ok=True)
                for sub in ["questions", "annotations", "prompts"]:
                    src = os.path.join(dl_dir, sub)
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(DATA_ROOT, sub))
    except Exception as e:
        print(f"[DATA] Download failed: {e}")
        raise SystemExit(1)


if not os.path.exists(os.path.join(DATA_ROOT, "questions")):
    _download_blend_from_hf()

QUESTIONS_DIR = os.path.join(DATA_ROOT, "questions")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
PROMPTS_DIR = os.path.join(DATA_ROOT, "prompts")

VANILLA_DIR = WORK_DIR / "vanilla_results"
SWA_DIR = WORK_DIR / "swa_results"
CMP_DIR = WORK_DIR / "comparison"
for d in [VANILLA_DIR, SWA_DIR, CMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 4. UTILITIES
# ============================================================================


def write_csv_row(values: list, filename: str):
    with open(filename, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(values)


def load_questions(country: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(QUESTIONS_DIR, f"{country}_questions.csv"), encoding="utf-8"
    )


def load_prompts(country: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PROMPTS_DIR, f"{country}_prompts.csv"), encoding="utf-8"
    )


def load_annotations(country: str) -> dict:
    with open(
        os.path.join(ANNOTATIONS_DIR, f"{country}_data.json"), "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def make_prompt(
    question: str, prompt_no: str, language: str,
    country: str, prompt_sheet: pd.DataFrame,
) -> str:
    row = prompt_sheet[prompt_sheet["id"] == prompt_no]
    template = (
        row["English"].values[0]
        if language == "English"
        else row["Translation"].values[0]
    )
    return template.replace("{q}", question)


def _format_references(annotations: list) -> str:
    lines = []
    for ann in annotations:
        answers = ann.get("answers", []) or ann.get("en_answers", [])
        count = ann.get("count", 1)
        lines.append(f"  [{count} annotator(s)]: {'; '.join(answers)}")
    return "\n".join(lines) if lines else "  (no references)"


def _skip_question(data: dict) -> bool:
    idks = data.get("idks", {})
    return (
        idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3
        or idks.get("idk", 0) >= 5
        or len(data.get("annotations", [])) == 0
    )


# ============================================================================
# 5. MODEL LOADING — unsloth, optimized for H100
# ============================================================================


def load_model():
    """
    Load Llama 3.1 8B via unsloth, optimized for H100 80GB.
    - bfloat16: native H100 dtype, 2x throughput vs float32
    - 4-bit: ~5GB VRAM, leaves 75GB for batch KV cache
    - unsloth for_inference: FlashAttention-2 + fused RoPE
    """
    print(f"[MODEL] Loading {MODEL_ID} via unsloth (bfloat16 + 4bit) ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    # Remove max_length from generation_config to prevent the
    # "Both max_new_tokens and max_length seem to have been set" spam.
    # We always pass max_new_tokens explicitly in generate() calls.
    model.generation_config.max_length = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"[MODEL] GPU: {torch.cuda.get_device_properties(0).name} "
              f"| Free: {free_mem:.1f} GB")
    print(f"[MODEL] Loaded. Device: {model.device}")
    return model, tokenizer


# ============================================================================
# 6. GENERATION — single + batched (H100 optimized)
# ============================================================================


def _generate_single(model, tokenizer, messages: list) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_batch(
    model, tokenizer, messages_list: list, max_new_tokens: int = MAX_NEW_TOKENS,
) -> List[str]:
    """Batched generation — multiple prompts in one forward pass on H100."""
    if len(messages_list) == 1:
        return [_generate_single(model, tokenizer, messages_list[0])]
    try:
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        results = []
        for i in range(len(messages_list)):
            input_len = int(encoded["attention_mask"][i].sum().item())
            results.append(
                tokenizer.decode(out[i][input_len:], skip_special_tokens=True).strip()
            )
        return results
    except Exception as e:
        print(f"  [WARN] Batch failed ({e}), sequential fallback")
        return [_generate_single(model, tokenizer, m) for m in messages_list]


def generate_response(model, tokenizer, prompt: str) -> str:
    return _generate_single(model, tokenizer, [{"role": "user", "content": prompt}])


def generate_all_persona_responses(
    model, tokenizer, prompt: str, personas: List[str],
) -> List[str]:
    """Batch-generate all persona responses. 5 personas fits in 1 batch on H100."""
    msgs = [
        [{"role": "system", "content": p}, {"role": "user", "content": prompt}]
        for p in personas
    ]
    results = []
    for i in range(0, len(msgs), BATCH_SIZE):
        results.extend(generate_batch(model, tokenizer, msgs[i:i + BATCH_SIZE]))
    return results


# ============================================================================
# 7. LLM JUDGE — same model scores cultural accuracy 1-5
# ============================================================================

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of cultural knowledge. You will be given:
1. A question about everyday cultural knowledge for a specific country/region.
2. Human-annotated reference answers from people of that culture.
3. An AI model's response to the question.

Score the AI response on cultural accuracy from 1 to 5:
  5 = Fully correct, matches reference answers.
  4 = Mostly correct, minor omissions.
  3 = Partially correct, notable gaps.
  2 = Mostly incorrect, significant errors.
  1 = Incorrect or irrelevant.

Respond with ONLY: {"score": <int>, "reason": "<brief>"}"""

JUDGE_USER_TEMPLATE = """\
Country: {country}
Question: {question}

Reference answers from local annotators:
{references}

AI model response:
{response}

Score (1-5). Respond with only JSON."""


def _parse_judge_output(raw: str) -> Tuple[int, str]:
    try:
        parsed = json.loads(raw)
        return max(1, min(5, int(parsed["score"]))), parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    match = re.search(r'"score"\s*:\s*(\d)', raw)
    if match:
        return int(match.group(1)), raw
    match = re.search(r'\b([1-5])\b', raw)
    if match:
        return int(match.group(1)), raw
    return 0, f"PARSE_FAILED: {raw[:200]}"


def call_judge(model, tokenizer, country, question, references, response):
    user_msg = JUDGE_USER_TEMPLATE.format(
        country=country.replace("_", " "),
        question=question, references=references, response=response,
    )
    raw = _generate_single(model, tokenizer, [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    return _parse_judge_output(raw)


def batch_judge(model, tokenizer, country, question, references, responses):
    """Judge multiple responses in one batched pass."""
    msgs = []
    for resp in responses:
        user_msg = JUDGE_USER_TEMPLATE.format(
            country=country.replace("_", " "),
            question=question, references=references, response=resp,
        )
        msgs.append([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    raw_list = []
    for i in range(0, len(msgs), BATCH_SIZE):
        raw_list.extend(generate_batch(model, tokenizer, msgs[i:i + BATCH_SIZE], max_new_tokens=128))
    return [_parse_judge_output(r) for r in raw_list]


# ============================================================================
# 8. SWA-PTIS ENGINE — score-based IS with PT utility
# ============================================================================


def _pt_value(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x >= 0,
        x.abs().pow(PT_ALPHA),
        -PT_KAPPA * x.abs().pow(PT_BETA_EXP),
    )


def swa_select(vanilla_score: float, persona_scores: List[float]) -> Tuple[int, float, dict]:
    """PT-IS weighted selection. Returns (best_idx, score, diagnostics)."""
    N = len(persona_scores)
    scores = torch.tensor(persona_scores, dtype=torch.float32)
    sigma = max(scores.std().item(), SIGMA_FLOOR) if N > 1 else SIGMA_FLOOR

    g_ind = (scores - vanilla_score) / sigma
    g_cons = (scores - scores.mean()) / sigma
    U = (1.0 - LAMBDA_COOP) * _pt_value(g_ind) + LAMBDA_COOP * _pt_value(g_cons)
    w = F.softmax(U / IS_TEMPERATURE, dim=0)

    k_eff = 1.0 / (w * w).sum().clamp_min(1e-12).item()
    ess_ratio = k_eff / N
    best_idx = int(w.argmax().item())
    best_score = persona_scores[best_idx]

    diag = {
        "is_weights": w.tolist(), "ess_ratio": ess_ratio,
        "sigma": sigma, "persona_scores": persona_scores,
    }

    if ess_ratio < RHO_EFF:
        if best_score > vanilla_score:
            diag["gate"] = "ess_low_but_improved"
            return best_idx, best_score, diag
        diag["gate"] = "ess_reject"
        return -1, vanilla_score, diag

    if best_score >= vanilla_score:
        diag["gate"] = "is_accept"
        return best_idx, best_score, diag

    diag["gate"] = "is_no_improve"
    return -1, vanilla_score, diag


# ============================================================================
# 9. VANILLA PIPELINE
# ============================================================================


def run_vanilla(model, tokenizer, blend_country, language, prompt_no):
    q_df = load_questions(blend_country)
    prompts = load_prompts(blend_country)
    annotations = load_annotations(blend_country)
    local_lang = BLEND_WVS_COUNTRIES[blend_country][1]
    q_col = "Question" if language == local_lang else "Translation"
    replace_flag = language == "English" and local_lang != "English"

    out_file = str(VANILLA_DIR / f"{MODEL_NAME}-{blend_country}_{prompt_no}_vanilla.csv")

    done = {}
    if os.path.exists(out_file):
        prev = pd.read_csv(out_file, encoding="utf-8")
        for _, r in prev.iterrows():
            done[r["ID"]] = {
                "response": str(r["response"]),
                "score": int(r.get("judge_score", 0)),
            }
        print(f"  Resume: {len(done)} done")
    else:
        write_csv_row(["ID", q_col, "prompt", "response", "prompt_no",
                        "judge_score", "judge_reason"], out_file)

    pb = tqdm(q_df.iterrows(), desc=f"Vanilla {blend_country}", total=len(q_df))
    for _, row in pb:
        qid = row["ID"]
        if qid in done:
            continue
        question = row[q_col]
        if replace_flag:
            question = question.replace("your country", blend_country.replace("_", " "))
        full_prompt = make_prompt(question, prompt_no, language, blend_country, prompts)

        response = generate_response(model, tokenizer, full_prompt)

        score, reason = 0, "SKIPPED"
        ann = annotations.get(qid)
        if ann and not _skip_question(ann):
            refs = _format_references(ann.get("annotations", []))
            score, reason = call_judge(model, tokenizer, blend_country, question, refs, response)

        done[qid] = {"response": response, "score": score}
        write_csv_row([qid, question, full_prompt, response, prompt_no, score, reason], out_file)
        pb.set_postfix({"score": score})

    return out_file, done


# ============================================================================
# 10. SWA PIPELINE — WVS personas + batched generation + IS selection
# ============================================================================


def run_swa(model, tokenizer, blend_country, iso_code, language, prompt_no, vanilla_results):
    q_df = load_questions(blend_country)
    prompts_sheet = load_prompts(blend_country)
    annotations = load_annotations(blend_country)
    local_lang = BLEND_WVS_COUNTRIES[blend_country][1]
    q_col = "Question" if language == local_lang else "Translation"
    replace_flag = language == "English" and local_lang != "English"

    # Build WVS personas from repo's src/personas.py
    personas = build_country_personas(iso_code, wvs_path=WVS_DATA_PATH)
    print(f"  Built {len(personas)} WVS personas for {iso_code}")

    out_file = str(SWA_DIR / f"{MODEL_NAME}-{blend_country}_{prompt_no}_swa.csv")

    done_ids = set()
    if os.path.exists(out_file):
        prev = pd.read_csv(out_file, encoding="utf-8")
        done_ids = set(prev["ID"])
        print(f"  Resume SWA: {len(done_ids)} done")
    else:
        write_csv_row(
            ["ID", q_col, "prompt", "method", "response",
             "judge_score", "vanilla_score", "selected_persona",
             "ess_ratio", "gate", "persona_scores"],
            out_file,
        )

    pb = tqdm(q_df.iterrows(), desc=f"SWA {blend_country}", total=len(q_df))
    swa_scores = []

    for _, row in pb:
        qid = row["ID"]
        if qid in done_ids:
            continue
        ann = annotations.get(qid)
        if not ann or _skip_question(ann):
            continue

        v_data = vanilla_results.get(qid, {})
        vanilla_score = v_data.get("score", 0)
        if vanilla_score <= 0:
            continue

        question = row[q_col]
        if replace_flag:
            question = question.replace("your country", blend_country.replace("_", " "))
        full_prompt = make_prompt(question, prompt_no, language, blend_country, prompts_sheet)
        refs = _format_references(ann.get("annotations", []))

        # Batched persona generation + batched judging
        persona_responses = generate_all_persona_responses(model, tokenizer, full_prompt, personas)
        judge_results = batch_judge(model, tokenizer, blend_country, question, refs, persona_responses)
        persona_scores = [max(s, 1) for s, _ in judge_results]

        # SWA-PTIS selection
        best_idx, best_score, diag = swa_select(vanilla_score, persona_scores)

        if best_idx >= 0:
            final_response = persona_responses[best_idx]
            method = "swa_persona"
        else:
            final_response = v_data.get("response", "")
            best_score = vanilla_score
            method = "swa_vanilla_fallback"

        swa_scores.append(best_score)
        write_csv_row(
            [qid, question, full_prompt, method, final_response,
             best_score, vanilla_score, best_idx,
             f"{diag['ess_ratio']:.3f}", diag["gate"],
             json.dumps(persona_scores)],
            out_file,
        )
        valid = [s for s in swa_scores if s > 0]
        if valid:
            pb.set_postfix({"swa": f"{np.mean(valid):.2f}", "gate": diag["gate"][:8]})

    return out_file


# ============================================================================
# 11. COMPARISON OUTPUT
# ============================================================================


def print_comparison(vanilla_all, swa_files):
    width = 72
    rows = []
    print(f"\n{'='*width}")
    print(f"  COMPARISON: Vanilla vs SWA-PTIS  |  Judge Score (1-5, higher=better)")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*width}")
    print(f"  {'Country':>15}  {'Vanilla':>10}  {'SWA':>10}  {'Delta':>8}  {'Improve':>10}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    all_v, all_s = [], []
    for blend_country in BLEND_WVS_COUNTRIES:
        v_results = vanilla_all.get(blend_country, {})
        v_scores = [d["score"] for d in v_results.values() if d.get("score", 0) > 0]

        swa_file = swa_files.get(blend_country)
        s_scores = []
        if swa_file and os.path.exists(swa_file):
            sdf = pd.read_csv(swa_file, encoding="utf-8")
            s_scores = [int(s) for s in sdf["judge_score"] if int(s) > 0]

        if not v_scores or not s_scores:
            continue

        vm, sm = float(np.mean(v_scores)), float(np.mean(s_scores))
        d = sm - vm
        imp = (d / vm * 100) if vm > 0 else 0.0
        sign = "+" if d >= 0 else ""
        print(f"  {blend_country:>15}  {vm:10.3f}  {sm:10.3f}  "
              f"{sign}{d:7.3f}  {sign}{imp:9.2f}%")
        all_v.append(vm)
        all_s.append(sm)
        rows.append({"country": blend_country, "vanilla": vm, "swa": sm,
                      "delta": d, "improve_pct": imp})

    if all_v:
        gv, gs = np.mean(all_v), np.mean(all_s)
        gd = gs - gv
        gi = (gd / gv * 100) if gv > 0 else 0.0
        sign = "+" if gd >= 0 else ""
        print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
        print(f"  {'MEAN':>15}  {gv:10.3f}  {gs:10.3f}  "
              f"{sign}{gd:7.3f}  {sign}{gi:9.2f}%")
        wins = sum(1 for r in rows if r["delta"] > 0)
        print(f"  SWA wins: {wins}/{len(rows)}")
    print(f"{'='*width}")

    if rows:
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(CMP_DIR / "vanilla_vs_swa.csv", index=False)
        print(f"[SAVE] {CMP_DIR / 'vanilla_vs_swa.csv'}")


# ============================================================================
# 12. MAIN
# ============================================================================


def main():
    setup_seeds(SEED)

    print("=" * 60)
    print("BLEnD Open-Ended: Vanilla + SWA-PTIS")
    print(f"Model: {MODEL_NAME}")
    print(f"Countries: {list(BLEND_WVS_COUNTRIES.keys())} (BLEnD x WVS overlap)")
    print(f"Prompts: {PROMPT_NOS}")
    print(f"Output: {WORK_DIR}")
    print("=" * 60)

    for subdir in [QUESTIONS_DIR, ANNOTATIONS_DIR, PROMPTS_DIR]:
        if not os.path.exists(subdir):
            print(f"[ERROR] Not found: {subdir}")
            sys.exit(1)

    model, tokenizer = load_model()

    vanilla_all = {}
    swa_files = {}

    for prompt_no in PROMPT_NOS:
        for ci, (blend_country, (iso_code, language)) in enumerate(BLEND_WVS_COUNTRIES.items()):
            print(f"\n{'#'*60}")
            print(f"  [{ci+1}/{len(BLEND_WVS_COUNTRIES)}] {blend_country} "
                  f"({iso_code}) / {language} / {prompt_no}")
            print(f"{'#'*60}")

            # Verify this country exists in BLEnD data
            q_path = os.path.join(QUESTIONS_DIR, f"{blend_country}_questions.csv")
            if not os.path.exists(q_path):
                print(f"  [SKIP] No BLEnD data for {blend_country}")
                continue

            # Verify WVS support
            if iso_code not in SUPPORTED_COUNTRIES:
                print(f"  [SKIP] {iso_code} not in SUPPORTED_COUNTRIES")
                continue

            # Phase 1: Vanilla
            print(f"\n--- Phase 1: Vanilla ---")
            _, v_results = run_vanilla(model, tokenizer, blend_country, language, prompt_no)
            vanilla_all[blend_country] = v_results
            valid = [d["score"] for d in v_results.values() if d["score"] > 0]
            if valid:
                print(f"  Vanilla mean: {np.mean(valid):.3f}/5 ({len(valid)} questions)")

            # Phase 2: SWA-PTIS with WVS personas
            print(f"\n--- Phase 2: SWA-PTIS (WVS personas for {iso_code}) ---")
            swa_file = run_swa(
                model, tokenizer, blend_country, iso_code,
                language, prompt_no, v_results,
            )
            swa_files[blend_country] = swa_file

            gc.collect()
            torch.cuda.empty_cache()

    # Phase 3: Comparison
    print_comparison(vanilla_all, swa_files)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n[DONE] Results under {WORK_DIR}")


if __name__ == "__main__":
    main()
