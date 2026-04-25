#!/usr/bin/env python3
"""
Paper sweep — Open-Ended SWA-DPBR — Qwen2.5-7B actor + Qwen2.5-72B-GPTQ-Int4 judge
====================================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

Same DPBR math as ``exp_paper/exp_paper_qwen25_7b.py``, but the actor (Qwen2.5-7B)
produces free-form reasoning text instead of emitting an A/B next token. A separate
72B GPTQ-Int4 judge LLM parses each generation into ``{choice, confidence}``, which
is mapped to a scalar pseudo-logit-gap and fed into an offline DPBR controller so
PT-IS, dual-pass bootstrap reliability, ESS anchor blend, and the hierarchical
country prior run unchanged.

Two-stage pipeline (memory-safe on Kaggle):
    Stage 1  load Qwen2.5-7B  ->  generate all (country x scenario x agent x debias)
             texts to JSONL, release actor.
    Stage 2  load Qwen2.5-72B-GPTQ-Int4 judge  ->  parse each text, assemble debiased
             pseudo-deltas, run offline DPBR, compute AMCE + alignment.

Both stages are resume-safe (Stage 1 dedupes JSONL rows; Stage 2 caches judge verdicts).

Setup (same pattern as exp_paper_ablation_qwen25_7b.py):
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input
    3. Add Qwen2.5-72B-Instruct-GPTQ-Int4 as Kaggle Model input
    4. Add multitp-data dataset
    5. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_openended_with_judge_llm.py

Environment variables:
    OPENENDED_STAGE            "1" | "2" | "both"     (default: both)
    OPENENDED_COUNTRIES        comma-separated ISO3   (default: PAPER_20_COUNTRIES)
    OPENENDED_N_SCENARIOS      integer                (default: 500)
    OPENENDED_MAX_NEW_TOKENS_ACTOR  integer           (default: 400)
    OPENENDED_MAX_NEW_TOKENS_JUDGE  integer           (default: 64)
    OPENENDED_JSONL_DIR        path                   (default: results/openended/stage1)
    OPENENDED_RESULTS_BASE     path                   (default: results/openended)
    ACTOR_LOAD_4BIT            "1" | "0"              (default: 0 — bf16 native)
    JUDGE_LOAD_4BIT            "1" | "0"              (default: 0 — GPTQ-Int4 already quantised)
    EXP24_VAR_SCALE / EXP24_K_HALF / EXP24_ESS_ANCHOR_REG  (pass-through)
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. KAGGLE OFFLINE BOOTSTRAP                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT_DATASET_DIR = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
ACTOR_MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
JUDGE_MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct-gptq-int4/1"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WORK_DIR = "/kaggle/working/cultural_alignment"

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_AUTO_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Backend: HF native bf16 (matches main Qwen2.5 experiment)
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")
# ESS anchor regularisation ON (matches paper §4.2)
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    """Copy project from read-only input to writable working dir."""
    if _on_kaggle():
        project_src = None
        for cand in (PROJECT_DATASET_DIR, PROJECT_DATASET_DIR_ALT):
            if os.path.isdir(cand):
                project_src = cand
                break
        if os.path.isdir(WORK_DIR) and os.path.isfile(
            os.path.join(WORK_DIR, "src", "controller.py")
        ):
            print(f"[SETUP] Working dir exists: {WORK_DIR}")
        else:
            if project_src is None:
                raise RuntimeError(
                    "Project dataset not found. Checked: "
                    f"{PROJECT_DATASET_DIR} and {PROJECT_DATASET_DIR_ALT}"
                )
            print(f"[SETUP] Copying project from {project_src} → {WORK_DIR} ...")
            shutil.copytree(project_src, WORK_DIR, dirs_exist_ok=True)
        os.chdir(WORK_DIR)
        sys.path.insert(0, WORK_DIR)
        return WORK_DIR
    else:
        here = os.getcwd()
        if os.path.isfile(os.path.join(here, "src", "controller.py")):
            sys.path.insert(0, here)
            return here
        raise RuntimeError("Not on Kaggle and not inside repo root.")


def _resolve_model_path(base_path: str, label: str) -> str:
    """Resolve local model weights path; search subdirs for config.json if needed."""
    if not _on_kaggle():
        return os.environ.get(f"{label}_MODEL_PATH", base_path)
    if os.path.isdir(base_path) and os.path.isfile(os.path.join(base_path, "config.json")):
        return base_path
    for sub in Path(base_path).rglob("config.json"):
        return str(sub.parent)
    candidates = [
        f"{base_path}/transformers/default/1",
        f"{base_path}/pytorch/default/1",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise RuntimeError(f"{label} model weights not found at {base_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. BOOTSTRAP                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  KAGGLE OFFLINE — Open-Ended SWA-DPBR")
print("  Actor: Qwen2.5-7B-Instruct (bf16) | Judge: Qwen2.5-72B-Instruct-GPTQ-Int4")
print("=" * 70)

_setup_project()

ACTOR_MODEL_PATH = _resolve_model_path(ACTOR_MODEL_LOCAL_PATH, "ACTOR")
JUDGE_MODEL_PATH = _resolve_model_path(JUDGE_MODEL_LOCAL_PATH, "JUDGE")
print(f"[SETUP] Actor model path: {ACTOR_MODEL_PATH}")
print(f"[SETUP] Judge model path: {JUDGE_MODEL_PATH}")

# Offline dep fallback (uses Kaggle's pre-cached wheels; --no-index avoids network)
# auto-gptq / optimum are needed for the 72B GPTQ-Int4 judge; if the Kaggle base
# image doesn't ship them this is a no-op and the GPTQ load will fail loudly.
if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf "
        "auto-gptq optimum 2>/dev/null || true",
        shell=True, check=False,
    )

from exp_paper.openended.stage1_actor_phi4 import Stage1Config, run_stage1  # noqa: E402
from exp_paper.openended.stage2_judge_qwen72b import Stage2Config, run_stage2  # noqa: E402


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name, "").strip()
    if not v:
        return list(default)
    return [s.strip() for s in v.split(",") if s.strip()]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HARDCODED RUN CONFIG — 3 countries × full 500 scenarios                   ║
# ║  (Western / SE Asia / Western — diverse cultural sample)                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
RUN_COUNTRIES: list[str] = ["USA", "VNM", "DEU"]
RUN_N_SCENARIOS: int = 500
RUN_STAGE: str = "both"  # "1" | "2" | "both"
RUN_MAX_NEW_TOKENS_ACTOR: int = 400
RUN_MAX_NEW_TOKENS_JUDGE: int = 64


def main() -> None:
    # Env vars still override hardcoded defaults (handy for resume / partial reruns).
    stage = os.environ.get("OPENENDED_STAGE", RUN_STAGE).strip().lower()
    if stage not in ("1", "2", "both"):
        raise ValueError(f"OPENENDED_STAGE must be 1|2|both, got {stage!r}")

    countries = _env_list("OPENENDED_COUNTRIES", RUN_COUNTRIES)
    n_scenarios = int(os.environ.get("OPENENDED_N_SCENARIOS", str(RUN_N_SCENARIOS)))

    results_base_default = (
        f"{WORK_DIR}/results/openended" if _on_kaggle() else
        str(Path(__file__).parent.parent / "results" / "openended")
    )
    results_base = os.environ.get("OPENENDED_RESULTS_BASE", results_base_default)
    jsonl_dir = os.environ.get("OPENENDED_JSONL_DIR", f"{results_base}/stage1")

    print(f"\n[OPENENDED] stage={stage}  countries={countries}")
    print(f"[OPENENDED] actor={ACTOR_MODEL_PATH}")
    print(f"[OPENENDED] judge={JUDGE_MODEL_PATH}")
    print(f"[OPENENDED] results_base={results_base}  jsonl_dir={jsonl_dir}")

    if stage in ("1", "both"):
        s1 = Stage1Config(
            model_name=ACTOR_MODEL_PATH,
            out_jsonl_dir=jsonl_dir,
            multitp_data_path=MULTITP_DATA_PATH,
            wvs_data_path=WVS_DATA_PATH,
            use_real_data=os.path.isdir(MULTITP_DATA_PATH),
            n_scenarios=n_scenarios,
            max_new_tokens=int(os.environ.get("OPENENDED_MAX_NEW_TOKENS_ACTOR", "400")),
            load_in_4bit=_env_bool("ACTOR_LOAD_4BIT", False),
            countries=countries,
        )
        run_stage1(s1)
        gc.collect()

    if stage == "both":
        # Bridge between stages: drop any lingering CUDA allocations before
        # loading the 72B judge. `run_stage1` already frees the actor, this is
        # belt-and-braces.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if stage in ("2", "both"):
        judge_4bit = _env_bool("JUDGE_LOAD_4BIT", False)
        if judge_4bit and "gptq" in JUDGE_MODEL_PATH.lower():
            print(
                "[WARN] Judge weights are GPTQ-Int4 already; forcing JUDGE_LOAD_4BIT=0 "
                "(BitsAndBytes 4bit on top of GPTQ would fail to load)."
            )
            judge_4bit = False
        s2 = Stage2Config(
            judge_model_name=JUDGE_MODEL_PATH,
            stage1_jsonl_dir=jsonl_dir,
            results_base=results_base,
            human_amce_path=HUMAN_AMCE_PATH,
            load_in_4bit=judge_4bit,
            max_new_tokens=int(os.environ.get("OPENENDED_MAX_NEW_TOKENS_JUDGE", "64")),
            countries=countries,
        )
        run_stage2(s2)


if __name__ == "__main__":
    main()
