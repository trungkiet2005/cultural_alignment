#!/usr/bin/env python3
"""
Paper sweep — Open-Ended SWA-DPBR — Phi-4 actor + Qwen2.5-72B judge
====================================================================

Same DPBR math as ``exp_paper/models/exp_paper_phi_4.py``, but the actor (Phi-4)
produces free-form reasoning text instead of emitting an A/B next token.
A separate 70B+ judge LLM (Qwen2.5-72B-Instruct by default) parses each
generation into ``{choice, confidence}``, which is mapped to a scalar
pseudo-logit-gap and fed into an offline DPBR controller so PT-IS, dual-pass
bootstrap reliability, ESS anchor blend, and the hierarchical country prior
run unchanged.

Two-stage pipeline (memory-safe on Kaggle):
    Stage 1  load Phi-4  ->  generate all (country x scenario x agent x debias)
             texts to JSONL, release Phi-4.
    Stage 2  load Qwen-72B judge  ->  parse each text, assemble debiased
             pseudo-deltas, run offline DPBR, compute AMCE + alignment.

Stage 1 alone is ~33 h across all 20 countries; run per-country subsets with
``OPENENDED_COUNTRIES=USA,GBR,...`` to fit Kaggle's 9 h limit. Both stages are
resume-safe (Stage 1 dedupes JSONL rows; Stage 2 caches judge verdicts).

Environment variables:
    OPENENDED_STAGE            "1" | "2" | "both"     (default: both)
    OPENENDED_COUNTRIES        comma-separated ISO3   (default: PAPER_20_COUNTRIES)
    OPENENDED_N_SCENARIOS      integer                (default: 500)
    OPENENDED_MAX_NEW_TOKENS   integer                (default: 400 actor, 64 judge)
    OPENENDED_JSONL_DIR        path                   (default: results/openended/stage1)
    OPENENDED_RESULTS_BASE     path                   (default: results/openended)
    ACTOR_MODEL_NAME           HF id                  (default: microsoft/phi-4)
    JUDGE_MODEL_NAME           HF id                  (default: Qwen/Qwen2.5-72B-Instruct)
    ACTOR_LOAD_4BIT            "1" | "0"              (default: 1)
    JUDGE_LOAD_4BIT            "1" | "0"              (default: 1)
    EXP24_VAR_SCALE / EXP24_K_HALF / EXP24_ESS_ANCHOR_REG  (pass-through)

Kaggle:
    !python exp_paper/openended/exp_paper_openended_phi4_judge_qwen72b.py
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        sys.path.insert(0, here)
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


_ensure_repo()

from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()


from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from exp_paper.openended.stage1_actor_phi4 import Stage1Config, run_stage1  # noqa: E402
from exp_paper.openended.stage2_judge_qwen72b import Stage2Config, run_stage2  # noqa: E402


_KAGGLE_ROOT = "/kaggle/working/cultural_alignment"
_DEFAULT_MULTITP = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
_DEFAULT_WVS = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
_DEFAULT_HUMAN_AMCE = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name, "").strip()
    if not v:
        return list(default)
    return [s.strip() for s in v.split(",") if s.strip()]


def main() -> None:
    stage = os.environ.get("OPENENDED_STAGE", "both").strip().lower()
    if stage not in ("1", "2", "both"):
        raise ValueError(f"OPENENDED_STAGE must be 1|2|both, got {stage!r}")

    countries = _env_list("OPENENDED_COUNTRIES", PAPER_20_COUNTRIES)
    n_scenarios = int(os.environ.get("OPENENDED_N_SCENARIOS", "500"))
    actor_model = os.environ.get("ACTOR_MODEL_NAME", "microsoft/phi-4")
    judge_model = os.environ.get("JUDGE_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    results_base = os.environ.get(
        "OPENENDED_RESULTS_BASE", f"{_KAGGLE_ROOT}/results/openended"
    )
    jsonl_dir = os.environ.get(
        "OPENENDED_JSONL_DIR", f"{results_base}/stage1"
    )
    human_amce_path = os.environ.get("HUMAN_AMCE_PATH", _DEFAULT_HUMAN_AMCE)
    multitp_data_path = os.environ.get("MULTITP_DATA_PATH", _DEFAULT_MULTITP)
    wvs_data_path = os.environ.get("WVS_DATA_PATH", _DEFAULT_WVS)

    print(f"\n[OPENENDED] stage={stage}  countries={countries}")
    print(f"[OPENENDED] actor={actor_model}  judge={judge_model}")
    print(f"[OPENENDED] results_base={results_base}  jsonl_dir={jsonl_dir}")

    if stage in ("1", "both"):
        s1 = Stage1Config(
            model_name=actor_model,
            out_jsonl_dir=jsonl_dir,
            multitp_data_path=multitp_data_path,
            wvs_data_path=wvs_data_path,
            use_real_data=os.path.isdir(multitp_data_path),
            n_scenarios=n_scenarios,
            max_new_tokens=int(os.environ.get("OPENENDED_MAX_NEW_TOKENS_ACTOR", "400")),
            load_in_4bit=_env_bool("ACTOR_LOAD_4BIT", True),
            countries=countries,
        )
        run_stage1(s1)
        gc.collect()

    if stage in ("1", "both") and stage != "1":
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
        s2 = Stage2Config(
            judge_model_name=judge_model,
            stage1_jsonl_dir=jsonl_dir,
            results_base=results_base,
            human_amce_path=human_amce_path,
            load_in_4bit=_env_bool("JUDGE_LOAD_4BIT", True),
            max_new_tokens=int(os.environ.get("OPENENDED_MAX_NEW_TOKENS_JUDGE", "64")),
            countries=countries,
        )
        run_stage2(s2)


if __name__ == "__main__":
    main()
