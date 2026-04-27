#!/usr/bin/env python3
"""
Paper sweep — Open-Ended SWA-DPBR — Qwen2.5-7B actor + Qwen2.5-7B self-judge
============================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

Same DPBR math as ``exp_paper/exp_paper_qwen25_7b.py``, but the actor (Qwen2.5-7B)
produces free-form reasoning ending in ``ANSWER: A/B`` and a judge (same
Qwen2.5-7B weights, self-judge BF16) parses it into ``{choice, confidence}``.
The pseudo-logit-gap is fed into an offline DPBR controller so PT-IS, dual-pass
bootstrap reliability, ESS anchor blend, and the hierarchical country prior run
unchanged.

UNIFIED single-pass pipeline (2026-04-27): actor and judge are merged into one
loop in :mod:`exp_paper.openended.unified_actor_judge`. Since both use the same
Qwen2.5-7B BF16 weights, the model is loaded ONCE and each (country, scenario,
agent) is generated → judged → scored in-line. Cuts wall-time roughly in half
vs. the legacy two-stage path and makes the code easier to modify (one file,
one loop, no JSONL handoff between stages).

Resume-safe: each (country, sid, agent_role) row is appended to
``{out}/combined/{country}.jsonl`` with both actor_text AND judge fields. Re-runs
read existing keys and skip them. DPBR/AMCE re-runs from cached rows on every
invocation.

Setup:
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input (used for actor AND judge)
    3. Add multitp-data dataset
    4. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_openended_with_DISCA.py

Configuration: edit the RUN CONFIG block near the top of this file (RUN_BATCH,
PLAN_B_LOOSEN_DPBR_RELIABILITY, etc.). No environment variables. Set
RUN_BATCH="all" to do all 20 paper countries in one session (may exceed Kaggle's
12h limit at 500 scenarios — split into "b1"/"b2" if needed).
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
# Self-judge: same model as actor (Qwen2.5-7B BF16). Avoids 72B GPTQ deps
# (optimum / auto-gptq) and fits a single 96 GB GPU comfortably.
JUDGE_MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RUN CONFIG — edit here, no env vars needed.                                ║
# ║                                                                             ║
# ║  Unified pipeline: model loaded ONCE; actor+judge run in a single loop. Set ║
# ║  RUN_BATCH="all" to do all 20 paper countries in one session, or "b1"/"b2"  ║
# ║  to split across two sessions if 12h Kaggle limit is tight.                 ║
# ║                                                                             ║
# ║  Resume-safe: ``{out}/combined/{country}.jsonl`` is keyed by               ║
# ║  (country, sid, role). Re-running picks up where it left off automatically. ║
# ║                                                                             ║
# ║  Plan A (disable hierarchical EMA prior) is ALWAYS ON for the open-ended    ║
# ║  pipeline — hardcoded in exp_paper/openended/dpbr_offline.py. Reason: with  ║
# ║  pseudo-deltas the EMA accumulates judge bias instead of cultural signal    ║
# ║  and tugs every scenario the same way regardless of preferred_on_right.    ║
# ║                                                                             ║
# ║  Plan B — loosen DPBR reliability gate. Default VAR_SCALE=0.04 was tuned    ║
# ║  for logit gaps (~±2); pseudo-delta range is ~±4.6 with σ=0.30, making     ║
# ║  (ds1-ds2)² ≈ 0.1-0.5 → r = exp(-bv/0.04) collapses to ~0 and kills δ*.    ║
# ║  Set 1.0 to keep r useful. Toggle for ablation.                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
RUN_BATCH: str = "all"          # "b1" (countries 1-10) | "b2" (11-20) | "all" (all 20)
RUN_N_SCENARIOS: int = 500
# Free-form reasoning: actor writes 2-4 sentences then "ANSWER: A/B". Need
# enough budget for reasoning + the final answer line; 384 gives headroom for
# verbose generators and non-English tokenizers (VI/ZH/TH eat ~1.5x tokens per
# char vs EN) without bloating wall-time. Drop to 8 only if reverting to the
# constrained A/B-only mode.
RUN_MAX_NEW_TOKENS_ACTOR: int = 384
RUN_MAX_NEW_TOKENS_JUDGE: int = 64
LOAD_4BIT: bool = False         # single shared model — one flag for actor+judge

PLAN_B_LOOSEN_DPBR_RELIABILITY: bool = False  # True -> VAR_SCALE=1.0 (else 0.04)

# Echo Plan B into env BEFORE the unified module imports experiment_DM.exp24_dpbr_core
# (which reads EXP24_VAR_SCALE at module-import time, not per-call).
os.environ["EXP24_VAR_SCALE"] = "1.0" if PLAN_B_LOOSEN_DPBR_RELIABILITY else "0.04"


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
print("  Actor: Qwen2.5-7B-Instruct (bf16) | Judge: Qwen2.5-7B-Instruct (bf16, self-judge)")
print("=" * 70)

_setup_project()

ACTOR_MODEL_PATH = _resolve_model_path(ACTOR_MODEL_LOCAL_PATH, "ACTOR")
JUDGE_MODEL_PATH = _resolve_model_path(JUDGE_MODEL_LOCAL_PATH, "JUDGE")
print(f"[SETUP] Actor model path: {ACTOR_MODEL_PATH}")
print(f"[SETUP] Judge model path: {JUDGE_MODEL_PATH}")

# Offline dep fallback (uses Kaggle's pre-cached wheels; --no-index avoids network).
# Self-judge reuses the actor's Qwen2.5-7B BF16 weights — no GPTQ deps required.
if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf "
        "2>/dev/null || true",
        shell=True, check=False,
    )

from exp_paper.openended.unified_actor_judge import UnifiedConfig, run_unified  # noqa: E402
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402


_PAPER_20: list[str] = list(PAPER_20_COUNTRIES)
_BATCHES: dict[str, list[str]] = {
    "all": _PAPER_20,           # 20 countries — may exceed Kaggle's 12h session
    "b1": _PAPER_20[:10],       # USA GBR DEU ARG BRA MEX COL VNM MMR THA
    "b2": _PAPER_20[10:],       # MYS IDN CHN JPN BGD IRN SRB ROU KGZ ETH
}


def main() -> None:
    if RUN_BATCH not in _BATCHES:
        raise ValueError(f"RUN_BATCH must be one of {sorted(_BATCHES)}, got {RUN_BATCH!r}")

    countries = list(_BATCHES[RUN_BATCH])
    results_base = (
        f"{WORK_DIR}/results/openended" if _on_kaggle() else
        str(Path(__file__).parent.parent / "results" / "openended")
    )

    print(f"\n[OPENENDED] batch={RUN_BATCH}  countries={countries}  n={RUN_N_SCENARIOS}")
    print(f"[OPENENDED] model={ACTOR_MODEL_PATH}  (actor + self-judge, shared weights)")
    print(f"[OPENENDED] results_base={results_base}")
    print(f"[OPENENDED] max_new_tokens_actor={RUN_MAX_NEW_TOKENS_ACTOR}  "
          f"max_new_tokens_judge={RUN_MAX_NEW_TOKENS_JUDGE}  4bit={LOAD_4BIT}")
    print(f"[OPENENDED] plan_b={PLAN_B_LOOSEN_DPBR_RELIABILITY}  "
          f"EXP24_VAR_SCALE={os.environ['EXP24_VAR_SCALE']}  "
          f"hierarchical_prior=DISABLED (Plan A always-on)")

    cfg = UnifiedConfig(
        model_name=ACTOR_MODEL_PATH,
        out_dir=results_base,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        use_real_data=os.path.isdir(MULTITP_DATA_PATH),
        n_scenarios=RUN_N_SCENARIOS,
        max_new_tokens_actor=RUN_MAX_NEW_TOKENS_ACTOR,
        max_new_tokens_judge=RUN_MAX_NEW_TOKENS_JUDGE,
        load_in_4bit=LOAD_4BIT,
        countries=countries,
        model_label="qwen25_7b_openended_disca",
    )
    run_unified(cfg)
    gc.collect()


if __name__ == "__main__":
    main()
