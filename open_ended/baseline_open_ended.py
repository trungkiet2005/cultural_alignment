#!/usr/bin/env python3
"""
BLEnD Short-Answer Question Evaluation — Phi-4 14B on Kaggle H100
==================================================================
Reproduces the exact methodology from:
  "BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures
   and Languages" (NeurIPS 2024 Datasets and Benchmarks Track)

Pipeline:
  Step 1 — Install dependencies & setup paths
  Step 2 — Load Phi-4 14B (BF16, SDPA attention, fits H100 80GB)
  Step 3 — Run inference on all 16 countries × 2 prompts (inst-4, pers-3)
            × 2 languages (local + English) — exactly like the paper
  Step 4 — Evaluate with Soft Exact Match (SEM-B & SEM-W)

H100 optimizations:
  - BF16 full precision (14B model ~28GB, well within H100 80GB)
  - SDPA attention (PyTorch built-in, dispatches FA kernel on H100 via cuDNN)
  - Batched inference (batch_size=4) for throughput
  - Language-tool lazy install (only installs what's needed per language)

"""

import os
import sys
import subprocess
import gc
import csv
import json
import re
import time
import unicodedata as ud
from string import punctuation
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================
_ON_KAGGLE = os.path.exists("/kaggle/working")


def _run(cmd: str, verbose: bool = False) -> int:
    """Run a shell command, streaming stdout in real-time when verbose=True."""
    if verbose:
        # Stream output line-by-line so long installs show progress
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
    print("[SETUP] (1/7) scipy tqdm matplotlib seaborn ...")
    _run("pip install -q scipy tqdm matplotlib seaborn", verbose=True)
    print("[SETUP] (2/7) pyarrow ...")
    _run("pip install --quiet --no-deps --force-reinstall pyarrow", verbose=True)
    print("[SETUP] (3/7) datasets ...")
    _run("pip install --quiet 'datasets>=3.4.1,<4.4.0'", verbose=True)
    print("[SETUP] (4/7) huggingface_hub + transformers + accelerate ...")
    # --upgrade --force-reinstall overwrites the system-installed packages in
    # /usr/local/lib/python3.12/dist-packages/ so Python picks up the new version.
    # Needed because huggingface_hub 0.21+ removed is_offline_mode which the
    # pre-installed transformers still imports.
    _run("pip install -q --upgrade --force-reinstall "
         "'huggingface_hub>=0.26.0' 'transformers>=4.47.0' accelerate", verbose=True)
    # Purge any cached module state from this Jupyter session so the
    # freshly installed versions are picked up on next import.
    import importlib, sys
    for _mod in list(sys.modules.keys()):
        if _mod == "transformers" or _mod.startswith("transformers.") \
                or _mod == "huggingface_hub" or _mod.startswith("huggingface_hub."):
            sys.modules.pop(_mod, None)
    importlib.invalidate_caches()
    print("[SETUP] (5/7) spacy ...")
    _run("pip install -q spacy", verbose=True)
    print("[SETUP] (6/7) spacy en_core_web_sm model ...")
    _run("python -m spacy download en_core_web_sm -q", verbose=True)
    print("[SETUP] (7/7) konlpy jieba hazm qalsadi nlp-id hausastemmer ...")
    _run("pip install -q konlpy jieba hazm qalsadi nlp-id hausastemmer", verbose=True)
    # NOTE: cltk, pyspark, spark-nlp are heavy and slow — installed lazily
    #       per language in _ensure_lang_tools() below, only when needed.
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
            _run("pip install -q pyspark spark-nlp")

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

    elif language == "Sundanese":
        repo = "/kaggle/working/SUSTEM" if _ON_KAGGLE else "SUSTEM"
        if not os.path.exists(repo):
            print(f"[SETUP] Cloning SUSTEM for {language}...")
            base = "/kaggle/working" if _ON_KAGGLE else "."
            _run(f"cd {base} && git clone https://github.com/andhikaprima/SUSTEM.git")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
# ============================================================================
# 1. CONFIG
# ============================================================================

# ---------------------------------------------------------------------------
# Paths — on Kaggle: try /kaggle/input first, else download to /kaggle/working
# ---------------------------------------------------------------------------
if _ON_KAGGLE:
    # Check if user uploaded dataset as Kaggle input (read-only)
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
        # Always download to a writable location
        dl_dir = os.path.dirname(DATA_ROOT)
        os.makedirs(dl_dir, exist_ok=True)
        snapshot_download(
            repo_id="nayeon212/BLEnD",
            repo_type="dataset",
            local_dir=dl_dir,
            allow_patterns=["data/**"],
        )
        # Verify download
        if os.path.exists(os.path.join(DATA_ROOT, "questions")):
            print("[DATA] Download complete.")
        else:
            # HF might put files flat — check and reorganize
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
# H100 optimizations: SDPA attention, batched inference, torch.compile
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/phi-4"
MODEL_NAME = "Phi-4-14B"
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 512       # same as paper (max_tokens=512)
TEMPERATURE = 0.0          # paper: temperature=0 (deterministic)
TOP_P = 1.0                # paper: top_p=1
BATCH_SIZE = 4             # H100 80GB, conservative for 14B BF16 (~28GB)

# ---------------------------------------------------------------------------
# Countries & languages — exact 16 from the paper
# ---------------------------------------------------------------------------
# Full 16 countries from the paper
COUNTRY_LANG: Dict[str, str] = {
    "UK": "English", "US": "English", "South_Korea": "Korean",
    "Algeria": "Arabic", "China": "Chinese", "Indonesia": "Indonesian",
    "Spain": "Spanish", "Iran": "Persian", "Mexico": "Spanish",
    "Assam": "Assamese", "Greece": "Greek", "Ethiopia": "Amharic",
    "Northern_Nigeria": "Hausa", "Azerbaijan": "Azerbaijani",
    "North_Korea": "Korean", "West_Java": "Sundanese",
}

# Paper uses inst-4 and pers-3 for final evaluation; add "pers-3" to run both
PROMPT_NOS: List[str] = ["inst-4"]

# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def write_csv_row(values: list, filename: str):
    """Append a single row to a CSV file."""
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def replace_country_name(s: str, country: str) -> str:
    """Replace 'your country' with actual country name (paper method)."""
    return s.replace("your country", country)


def load_questions(country: str) -> pd.DataFrame:
    """Load questions CSV for a given country."""
    filepath = os.path.join(QUESTIONS_DIR, f"{country}_questions.csv")
    return pd.read_csv(filepath, encoding="utf-8")


def load_prompts(country: str) -> pd.DataFrame:
    """Load prompt templates CSV for a given country."""
    filepath = os.path.join(PROMPTS_DIR, f"{country}_prompts.csv")
    return pd.read_csv(filepath, encoding="utf-8")


def load_annotations(country: str) -> dict:
    """Load human annotations JSON for a given country."""
    filepath = os.path.join(ANNOTATIONS_DIR, f"{country}_data.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(question: str, prompt_no: str, language: str,
                country: str, prompt_sheet: pd.DataFrame) -> str:
    """
    Construct the final prompt by substituting {q} into the template.
    Exactly matches paper's model_inference.py:40-47.
    """
    row = prompt_sheet[prompt_sheet["id"] == prompt_no]
    if language == "English":
        template = row["English"].values[0]
    else:
        template = row["Translation"].values[0]
    return template.replace("{q}", question)


# ============================================================================
# 3. MODEL LOADING
# ============================================================================

def load_model():
    """
    Load Phi-4 14B with H100-optimized settings.
    - BF16 full precision (14B model ~28GB, well within H100 80GB)
    - SDPA attention (PyTorch built-in, dispatches FA kernel on H100 via cuDNN)
    - torch.compile() for graph fusion (first batch slower, rest faster)
    - Padding side left for correct batched generation
    """
    print(f"[MODEL] Loading {MODEL_ID} (BF16 + SDPA) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # PyTorch built-in, dispatches FA on H100
        token=HF_TOKEN if HF_TOKEN else None,
        trust_remote_code=True,      # required for Phi-4
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN if HF_TOKEN else None,
        padding_side="left",         # required for batched generation
        trust_remote_code=True,
    )
    # Phi-4 uses eos as pad; ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    model.generation_config.max_length = None

    # torch.compile: mode="reduce-overhead" best for repeated same-shape batches
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("[MODEL] torch.compile() enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"[MODEL] torch.compile() skipped: {e}")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[MODEL] Loaded (BF16 + SDPA). GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB)")
    return model, tokenizer


# ============================================================================
# 4. INFERENCE — exactly follows paper methodology
# ============================================================================

def generate_batch(model, tokenizer, prompts: List[str]) -> List[str]:
    """
    Batched generation for H100 throughput.
    Left-pads all prompts to same length, generates in one forward pass.
    """
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    # apply_chat_template with padding
    encoded = tokenizer.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

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

    # Decode only new tokens for each item in batch
    responses = []
    for out in output_ids:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def generate_response(model, tokenizer, prompt: str) -> str:
    """Single-prompt wrapper around generate_batch (for compatibility)."""
    return generate_batch(model, tokenizer, [prompt])[0]


def run_inference_for_country(
    model, tokenizer,
    country: str, language: str, prompt_no: str
) -> str:
    """
    Run inference on all questions for one (country, language, prompt) combo.
    Returns the output CSV filepath.
    Uses batched generation (BATCH_SIZE) for H100 throughput.
    """
    q_df = load_questions(country)
    prompt_sheet = load_prompts(country)

    local_lang = COUNTRY_LANG[country]
    q_col = "Question" if language == local_lang else "Translation"
    replace_country_flag = (language == "English" and local_lang != "English")

    output_filename = str(
        RESULTS_DIR / f"{MODEL_NAME}-{country}_{language}_{prompt_no}_result.csv"
    )

    # Resume support — skip already processed questions
    done_ids: set = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename, encoding="utf-8")
        done_ids = set(already["ID"])
        print(f"  Resuming: {len(done_ids)} already done")
    else:
        write_csv_row(["ID", q_col, "prompt", "response", "prompt_no"], output_filename)

    # Build list of pending rows
    pending = []
    for _, row in q_df.iterrows():
        qid = row["ID"]
        if qid in done_ids:
            continue
        question = row[q_col]
        if replace_country_flag:
            question = replace_country_name(question, country.replace("_", " "))
        full_prompt = make_prompt(question, prompt_no, language, country, prompt_sheet)
        pending.append((qid, question, full_prompt))

    # Batched inference
    pb = tqdm(range(0, len(pending), BATCH_SIZE),
              desc=f"{country}/{language}/{prompt_no}",
              total=(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE)

    for start in pb:
        batch = pending[start: start + BATCH_SIZE]
        prompts = [item[2] for item in batch]
        responses = generate_batch(model, tokenizer, prompts)
        for (qid, question, full_prompt), response in zip(batch, responses):
            write_csv_row([qid, question, full_prompt, response, prompt_no],
                          output_filename)
        pb.set_postfix({"done": start + len(batch)})

    return output_filename


def evaluate_single(country: str, language: str, prompt_no: str,
                    result_file: str, eval_result_file: str):
    """Evaluate a single (country, language, prompt) combo right after inference."""
    if not os.path.exists(result_file):
        return None, None

    res_df = pd.read_csv(result_file, encoding="utf-8")
    annotations = load_annotations(country)

    sem_b, sem_w, scored_df = soft_exact_match(
        country=country,
        language=language,
        annotation_dict=annotations,
        response_df=res_df,
        id_col="ID",
        r_col="response",
        annotations_key="annotations",
    )

    print(f"    >>> SEM-B: {sem_b:.2f}%  |  SEM-W: {sem_w:.2f}%")

    write_csv_row([MODEL_NAME, country, language, prompt_no, "SEM-B", sem_b],
                  eval_result_file)
    write_csv_row([MODEL_NAME, country, language, prompt_no, "SEM-W", sem_w],
                  eval_result_file)

    scored_df.to_csv(
        str(RESULTS_DIR / f"{MODEL_NAME}_{country}_{language}_{prompt_no}_response_score.csv"),
        index=False, encoding="utf-8",
    )

    return sem_b, sem_w


def run_all_inference_and_eval(model, tokenizer) -> pd.DataFrame:
    """
    Run inference + immediate evaluation for each (country, language, prompt) combo.
    Paper runs: 16 countries × 2 prompts (inst-4, pers-3) × 2 languages.
    """
    eval_result_file = str(EVAL_DIR / "evaluation_results.csv")
    if not os.path.exists(eval_result_file):
        write_csv_row(
            ["model", "country", "language", "prompt_no", "eval_method", "score"],
            eval_result_file,
        )

    total_combos = []
    for country, local_lang in COUNTRY_LANG.items():
        for prompt_no in PROMPT_NOS:
            # local language
            total_combos.append((country, local_lang, prompt_no))
            # English (skip if local is already English)
            if local_lang != "English":
                total_combos.append((country, "English", prompt_no))

    print(f"\n{'='*60}")
    print(f"INFERENCE + EVAL: {len(total_combos)} combos "
          f"({len(COUNTRY_LANG)} countries × {len(PROMPT_NOS)} prompts × languages)")
    print(f"{'='*60}\n")

    for i, (country, language, prompt_no) in enumerate(total_combos):
        print(f"\n[{i+1}/{len(total_combos)}] {country} / {language} / {prompt_no}")

        # --- Inference ---
        result_file = run_inference_for_country(
            model, tokenizer, country, language, prompt_no)

        # --- Evaluate immediately ---
        evaluate_single(country, language, prompt_no, result_file, eval_result_file)

        gc.collect()
        torch.cuda.empty_cache()

    # De-duplicate & return final results
    df = pd.read_csv(eval_result_file, encoding="utf-8")
    df.drop_duplicates(
        subset=["model", "country", "language", "prompt_no", "eval_method"],
        keep="last", inplace=True,
    )
    df.to_csv(eval_result_file, index=False, encoding="utf-8")
    return df


# ============================================================================
# 5. EVALUATION — Soft Exact Match (SEM-B & SEM-W)
#    Exactly reproduces evaluation/exact_match.py from the paper
# ============================================================================

def delete_prompt_from_answer(text: str, prompt: str) -> str:
    """Remove prompt/prefix artifacts from LLM response. (paper method)"""
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
    """Extract and clean LLM response for a given question ID."""
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
    """Load language-specific NLP pipeline for lemmatization.
    Heavy packages (spark-nlp, cltk, git repos) are installed on-demand.
    """
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
    """
    Language-aware answer matching with lemmatization.
    Exactly reproduces evaluation/exact_match.py:54-147.
    """
    # Direct match shortcuts
    if answer in llm_response:
        return True
    if answer.replace("-", " ") in llm_response:
        return True
    if answer.replace(" ", "-") in llm_response:
        return True

    # Language-specific tokenization + lemmatization
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
            # Fallback: simple whitespace split
            answer_tokens = answer.split()
            llm_tokens = llm_response.split()

    except Exception as e:
        print(f"  [WARN] lemma_check failed for {language}: {e}")
        answer_tokens = answer.split()
        llm_tokens = llm_response.split()

    # Normalize: remove accents, lowercase, filter punctuation
    d = {ord("\N{COMBINING ACUTE ACCENT}"): None}
    answer_tokens = [ud.normalize("NFD", t).translate(d).lower()
                     for t in answer_tokens if t not in punctuation and t != ""]
    llm_tokens = [ud.normalize("NFD", t).translate(d).lower()
                  for t in llm_tokens if t not in punctuation and t != ""]

    # Check: all answer tokens must appear in LLM response tokens
    return all(a in llm_tokens for a in answer_tokens)


def soft_exact_match(
    country: str, language: str,
    annotation_dict: dict, response_df: pd.DataFrame,
    id_col: str = "ID", r_col: str = "response",
    annotations_key: str = "annotations"
) -> Tuple[float, float, pd.DataFrame]:
    """
    Paper's SEM-B (binary) and SEM-W (weighted) metrics.
    Exactly reproduces evaluation/exact_match.py:173-269.
    """
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

        # Skip questions with too many "idk" / "no-answer" / "not-applicable"
        # (exact paper logic)
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
                # Try local language first
                if language != "English":
                    for a in agg_ans["answers"]:
                        if lemma_check(a, llm_response, nlp_pipeline, language):
                            binary_score += 1
                            weight_score += agg_ans["count"] / max_vote
                            flag = True
                            break
                # Fallback to English
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


def print_summary(df: pd.DataFrame):
    """Print final summary tables by language and country."""
    eval_result_file = str(EVAL_DIR / "evaluation_results.csv")

    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {eval_result_file}")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # --- Summary per language ---
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

    # --- Summary per country ---
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
# 6. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("BLEnD Short-Answer Question Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {PROMPT_NOS}")
    print(f"Countries: {len(COUNTRY_LANG)}")
    print(f"Data: {DATA_ROOT}")
    print(f"Output: {WORK_DIR}")
    print("=" * 60)

    # Verify data exists
    for subdir in [QUESTIONS_DIR, ANNOTATIONS_DIR, PROMPTS_DIR]:
        if not os.path.exists(subdir):
            print(f"[ERROR] Data directory not found: {subdir}")
            print("Please upload the BLEnD data/ folder as a Kaggle dataset")
            print("and adjust DATA_ROOT in this script.")
            sys.exit(1)

    # --- Step 1: Load model ---
    model, tokenizer = load_model()

    # --- Step 2: Run inference + evaluate per combo ---
    results_df = run_all_inference_and_eval(model, tokenizer)

    # --- Step 3: Free GPU memory ---
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 4: Print summaries ---
    print_summary(results_df)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
