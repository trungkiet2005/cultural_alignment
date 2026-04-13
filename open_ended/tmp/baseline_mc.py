#!/usr/bin/env python3
"""
BLEnD Multiple-Choice Question Evaluation — Llama 3.1 70B on Kaggle H100
=========================================================================
Reproduces the MC evaluation methodology from:
  "BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures
   and Languages" (NeurIPS 2024 Datasets and Benchmarks Track)

Pipeline:
  Step 1 — Install dependencies & setup paths
  Step 2 — Load Llama 3.1 70B (4-bit quantized, fits H100 80GB)
  Step 3 — Load & sample MC questions (3M+ rows → manageable subset)
  Step 4 — Run inference on all countries
  Step 5 — Evaluate with simple accuracy + 95% CI

Usage on Kaggle:
  1. Upload the BLEnD dataset (data/ + evaluation/mc_data/) as a Kaggle dataset
  2. Set HF_TOKEN in Kaggle secrets
  3. Select GPU H100 80GB runtime
  4. Run this notebook/script
"""

import os
import sys
import subprocess
import gc
import csv
import json
import re
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
    # HuggingFace Hub (for auto-downloading dataset)
    _run("pip install -q huggingface_hub")
    print("[SETUP] Installation complete.")

# CRITICAL: import unsloth BEFORE transformers
try:
    import unsloth  # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ============================================================================
# 1. CONFIG
# ============================================================================

# --- Paths ---
def _find_mc_data_dir_kaggle():
    """Auto-discover MC data directory under /kaggle/input/ by scanning all datasets."""
    search_roots = []
    # Collect all dataset dirs under /kaggle/input/
    if os.path.isdir("/kaggle/input"):
        for name in os.listdir("/kaggle/input"):
            search_roots.append(os.path.join("/kaggle/input", name))
    search_roots.append("/kaggle/working/blend_data")
    search_roots.append("/kaggle/working")

    # Candidate sub-paths, ordered by specificity
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
    WORK_DIR = Path("/kaggle/working/blend_mc_results")
else:
    _found_mc = None
    _DATA_BASE = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "blend_mc_results"

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
        # Download to a writable location
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
        # Find the downloaded MC data
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

RESULTS_DIR = WORK_DIR / "model_inference_results"
EVAL_DIR = WORK_DIR / "evaluation_results"
for d in [RESULTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Model ---
MODEL_ID = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
MODEL_NAME = "Llama-3.1-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 128  # MC answers are short (JSON with single letter)
TEMPERATURE = 0.0
TOP_P = 1.0

# --- Countries & sampling ---
# Full 16 countries (uncomment when ready):
# MC_COUNTRIES = [
#     "UK", "US", "South_Korea", "Algeria", "China", "Indonesia",
#     "Spain", "Iran", "Mexico", "Assam", "Greece", "Ethiopia",
#     "Northern_Nigeria", "Azerbaijan", "North_Korea", "West_Java",
# ]

# 5 representative countries
MC_COUNTRIES = ["UK", "South_Korea", "China", "Iran", "Algeria"]

N_PER_COUNTRY = 500  # number of MC questions to sample per country
SAMPLE_SEED = 42


# ============================================================================
# 2. UTILITY FUNCTIONS
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
    # Try direct parse first
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


def extract_mc_answer(full_response: str, prompt: str,
                      choices_json: str) -> str:
    """
    Extract A/B/C/D answer from model response.

    Priority:
      1. Parse JSON {"answer_choice": "X"} → extract letter
      2. If answer_choice is text (not letter), reverse-lookup in choices
      3. Regex fallback: first A-D letter in response
      4. Last resort: return raw response (will score as incorrect)
    """
    json_res = get_json_str(full_response)

    if isinstance(json_res, dict) and "answer_choice" in json_res:
        answer_val = str(json_res["answer_choice"]).strip()

        # Try direct letter extraction
        letters = re.findall(r"[A-D]", answer_val)
        if letters:
            letter = letters[0]
            # Verify this letter exists as a choice in the prompt
            if f"{letter}." in prompt:
                return letter

        # Reverse lookup: answer_choice might be the choice text (exact match)
        try:
            choices = json.loads(choices_json) if isinstance(choices_json, str) else choices_json
            for k, v in choices.items():
                if str(v).strip() == answer_val:
                    return str(k)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Still return first letter found
        if letters:
            return letters[0]

        # answer_choice was in dict but no valid letter — return raw response
        return str(full_response)[:50]

    # Fallback (json_res is NOT a dict with answer_choice):
    # regex for any A-D letter in raw response
    if not isinstance(json_res, dict):
        raw = str(json_res) if json_res is not None else str(full_response)
        try:
            return re.findall(r"[A-D]", raw)[0]
        except (IndexError, TypeError):
            pass

    return str(full_response)[:50]  # truncate for readability


def wilson_ci(n_correct: int, n_total: int,
              z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0, center - margin) * 100, min(1, center + margin) * 100


# ============================================================================
# 3. MODEL LOADING
# ============================================================================

def load_model():
    """
    Load Llama 3.1 70B Instruct with 4-bit quantization.
    Uses pure HF transformers. Fits on H100 80GB (~38GB weights + KV cache).
    """
    print(f"[MODEL] Loading {MODEL_ID} via HF transformers ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.eval()
    model.generation_config.max_length = None
    print(f"[MODEL] Loaded (4-bit). Device: {model.device}")
    return model, tokenizer


# ============================================================================
# 4. MC DATA LOADING
# ============================================================================

def load_mc_questions(
    mc_dir: str,
    mc_files: List[str],
    countries: List[str],
    n_per_country: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and sample MC questions from large CSV files.

    The MC dataset has 3M+ rows per file. This function:
      1. Reads in chunks (50k rows) to avoid memory issues
      2. Filters to target countries only
      3. Samples n_per_country rows per country (stratified by question ID)

    Returns:
        DataFrame with columns: MCQID, ID, country, prompt, choices,
                                choice_countries, answer_idx
    """
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
            # Filter to target countries
            filtered = chunk[chunk["country"].isin(country_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
            n_chunks += 1
            if n_chunks % 20 == 0:
                n_loaded = sum(len(c) for c in chunks)
                print(f"  ... processed {n_chunks} chunks, {n_loaded} matching rows")

    if not chunks:
        raise FileNotFoundError(
            f"No MC questions found in {mc_dir} for countries {countries}. "
            f"Available files: {os.listdir(mc_dir) if os.path.exists(mc_dir) else 'dir not found'}"
        )

    all_data = pd.concat(chunks, ignore_index=True)
    print(f"[DATA] Total matching rows: {len(all_data)}")

    # Stratified sampling per country
    rng = np.random.RandomState(seed)
    sampled_parts = []

    for country in countries:
        country_df = all_data[all_data["country"] == country]
        if len(country_df) == 0:
            print(f"[WARN] No MC questions for {country}")
            continue

        n_available = len(country_df)
        n_sample = min(n_per_country, n_available)

        # Stratified by base question ID to ensure question diversity
        unique_ids = country_df["ID"].unique()
        if len(unique_ids) <= n_sample:
            # Fewer unique questions than sample size: one variant per ID
            sampled_ids = unique_ids
        else:
            # Sample question IDs first
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
              f"(from {n_available} available, {len(unique_ids)} unique IDs)")

    result = pd.concat(sampled_parts, ignore_index=True)
    print(f"[DATA] Final MC dataset: {len(result)} questions")
    return result


# ============================================================================
# 5. INFERENCE
# ============================================================================

def generate_mc_response(
    model,
    tokenizer,
    prompt: str,
) -> str:
    """
    Generate a response for an MC question.

    Args:
        model: The loaded LLM
        tokenizer: The tokenizer
        prompt: The full MC prompt (includes question + choices + JSON instruction)

    Returns:
        full_response string
    """
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
    full_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return full_response


def run_mc_inference_for_country(
    model,
    tokenizer,
    country: str,
    mc_df: pd.DataFrame,
) -> str:
    """
    Run MC inference on all questions for a given country.
    Returns the output CSV filepath.
    """
    country_df = mc_df[mc_df["country"] == country].copy()

    output_filename = str(
        RESULTS_DIR / f"{MODEL_NAME}-MC-{country}_result.csv"
    )

    # Resume support
    done_ids = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename, encoding="utf-8")
        done_ids = set(already["MCQID"])
        print(f"  Resuming: {len(done_ids)} already done")
    else:
        write_csv_row(
            ["MCQID", "ID", "country", "answer_idx",
             "full_response", "predicted_answer"],
            output_filename,
        )

    pb = tqdm(
        country_df.iterrows(),
        desc=f"MC/{country}",
        total=len(country_df),
    )

    n_correct = 0
    n_done = 0

    for _, row in pb:
        mcqid = row["MCQID"]
        pb.set_postfix({"ID": mcqid})

        if mcqid in done_ids:
            continue  # don't count skipped rows in running accuracy

        prompt = row["prompt"]
        choices_json = row["choices"]
        answer_idx = str(row["answer_idx"])

        # Generate response
        full_response = generate_mc_response(model, tokenizer, prompt)

        # Extract answer
        predicted = extract_mc_answer(full_response, prompt, choices_json)

        # Track running accuracy
        n_done += 1
        if predicted == answer_idx:
            n_correct += 1
        if n_done > 0:
            pb.set_postfix({
                "acc": f"{n_correct / n_done * 100:.1f}%",
                "n": n_done,
            })

        # Save row
        write_csv_row(
            [mcqid, row["ID"], country, answer_idx,
             full_response, predicted],
            output_filename,
        )

    return output_filename


# ============================================================================
# 6. EVALUATION
# ============================================================================

def evaluate_mc_country(country: str, result_file: str) -> Dict:
    """
    Compute accuracy and confidence intervals for a country's MC results.
    """
    if not os.path.exists(result_file):
        return {
            "country": country,
            "accuracy": 0.0,
            "n_total": 0,
            "n_correct": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    df = pd.read_csv(result_file, encoding="utf-8")
    df = df[df["country"] == country] if "country" in df.columns else df

    df["correct"] = (
        df["answer_idx"].astype(str).str.strip()
        == df["predicted_answer"].astype(str).str.strip()
    ).astype(int)

    n_total = len(df)
    n_correct = int(df["correct"].sum())
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0

    ci_lower, ci_upper = wilson_ci(n_correct, n_total)

    # Per-question-ID breakdown
    per_qid = df.groupby("ID")["correct"].mean()

    return {
        "country": country,
        "accuracy": accuracy,
        "n_total": n_total,
        "n_correct": n_correct,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "per_qid_mean": per_qid.mean() * 100 if len(per_qid) > 0 else 0.0,
        "n_unique_questions": len(per_qid),
    }


def run_all_mc_inference_and_eval(
    model,
    tokenizer,
    mc_df: pd.DataFrame,
) -> List[Dict]:
    """
    Run MC inference + evaluation for all countries.
    """
    eval_result_file = str(EVAL_DIR / "mc_evaluation_results.csv")
    if not os.path.exists(eval_result_file):
        write_csv_row(
            ["model", "country", "accuracy", "ci_lower", "ci_upper",
             "n_total", "n_correct", "n_unique_questions"],
            eval_result_file,
        )

    results = []

    for i, country in enumerate(MC_COUNTRIES):
        print(f"\n[{i + 1}/{len(MC_COUNTRIES)}] {country}")

        # --- Inference ---
        result_file = run_mc_inference_for_country(
            model, tokenizer, country, mc_df)

        # --- Evaluate ---
        eval_result = evaluate_mc_country(country, result_file)
        results.append(eval_result)

        print(f"    >>> Accuracy: {eval_result['accuracy']:.2f}% "
              f"[{eval_result['ci_lower']:.1f}%, {eval_result['ci_upper']:.1f}%] "
              f"({eval_result['n_correct']}/{eval_result['n_total']})")

        # Save to summary CSV
        write_csv_row(
            [MODEL_NAME, country, eval_result["accuracy"],
             eval_result["ci_lower"], eval_result["ci_upper"],
             eval_result["n_total"], eval_result["n_correct"],
             eval_result["n_unique_questions"]],
            eval_result_file,
        )

        gc.collect()
        torch.cuda.empty_cache()

    # Deduplicate evaluation results (handles re-runs)
    if os.path.exists(eval_result_file):
        df = pd.read_csv(eval_result_file, encoding="utf-8")
        df.drop_duplicates(
            subset=["model", "country"],
            keep="last", inplace=True,
        )
        df.to_csv(eval_result_file, index=False, encoding="utf-8")

    return results


# ============================================================================
# 7. SUMMARY
# ============================================================================

def print_mc_summary(results: List[Dict]):
    """Print final summary table of MC evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"{'MULTIPLE-CHOICE EVALUATION RESULTS':^70}")
    print(f"{'=' * 70}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Questions per country: {N_PER_COUNTRY}")
    print(f"{'─' * 70}")
    print(f"  {'Country':<20s} {'Accuracy':>10s} {'95% CI':>18s} "
          f"{'N':>6s} {'Correct':>8s}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 18} {'─' * 6} {'─' * 8}")

    all_acc = []
    total_n = 0
    total_correct = 0

    for r in results:
        print(f"  {r['country']:<20s} {r['accuracy']:>9.2f}% "
              f"[{r['ci_lower']:>5.1f}%, {r['ci_upper']:>5.1f}%] "
              f"{r['n_total']:>6d} {r['n_correct']:>8d}")
        all_acc.append(r["accuracy"])
        total_n += r["n_total"]
        total_correct += r["n_correct"]

    overall = total_correct / total_n * 100 if total_n > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(total_correct, total_n)

    print(f"  {'─' * 70}")
    print(f"  {'OVERALL':<20s} {overall:>9.2f}% "
          f"[{ci_lo:>5.1f}%, {ci_hi:>5.1f}%] "
          f"{total_n:>6d} {total_correct:>8d}")
    print(f"  {'Mean (macro)':<20s} {np.mean(all_acc):>9.2f}%")
    print(f"{'=' * 70}")

    # Random baseline comparison
    print(f"\n  Random baseline (4 choices): 25.00%")
    print(f"  Model advantage over random: {overall - 25.0:+.2f}pp")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BLEnD Multiple-Choice Question Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Countries: {MC_COUNTRIES}")
    print(f"Questions per country: {N_PER_COUNTRY}")
    print(f"MC data: {MC_DATA_DIR}")
    print(f"Output: {WORK_DIR}")
    print("=" * 70)

    # Verify MC data exists
    if not os.path.exists(MC_DATA_DIR):
        print(f"[ERROR] MC data directory not found: {MC_DATA_DIR}")
        print("Please ensure the MC questions are available at:")
        print(f"  {MC_DATA_DIR}")
        sys.exit(1)

    available_files = [f for f in MC_FILES if os.path.exists(
        os.path.join(MC_DATA_DIR, f))]
    if not available_files:
        print(f"[ERROR] No MC question files found in {MC_DATA_DIR}")
        print(f"Expected: {MC_FILES}")
        print(f"Available: {os.listdir(MC_DATA_DIR)}")
        sys.exit(1)

    # --- Step 1: Load & sample MC questions ---
    print("\n[STEP 1] Loading MC questions...")
    mc_df = load_mc_questions(
        mc_dir=MC_DATA_DIR,
        mc_files=available_files,
        countries=MC_COUNTRIES,
        n_per_country=N_PER_COUNTRY,
        seed=SAMPLE_SEED,
    )

    # --- Step 2: Load model ---
    print("\n[STEP 2] Loading model...")
    model, tokenizer = load_model()

    # --- Step 3: Run inference + evaluate ---
    print("\n[STEP 3] Running MC inference + evaluation...")
    results = run_all_mc_inference_and_eval(model, tokenizer, mc_df)

    # --- Step 4: Free GPU ---
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 5: Print summary ---
    print_mc_summary(results)

    print(f"\n[DONE] Results saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
