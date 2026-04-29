#!/usr/bin/env python3
"""
Paper sweep — Open-Ended HYBRID SWA-DPBR (RAW vs ORACLE_D per-country routing)
==============================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

This launcher is a thin wrapper over the SAFE pipeline
(:mod:`exp_paper.openended.unified_actor_judge_safe`). It runs the SAFE
pipeline end-to-end and emits an additional per-country method —
``hybrid_raw_oracle_d`` — that picks, for each country, the best of:

  - **RAW SWA-DPBR** (no safety gates) when ``RAW MIS < VAN MIS``, i.e. when
    the unconstrained correction already strictly beats vanilla on that
    country, OR
  - **ORACLE_D** (per-AMCE-dimension oracle) otherwise — guarantees
    ``MIS_oracle_d ≤ MIS_van`` by construction (see
    ``_build_dim_oracle_amce`` in unified_actor_judge_safe.py).

This reproduces the **"Phương pháp lai"** (hybrid) column in the paper table
(``exp_paper/Paper_New/SWA_DPBR/paper_revised.tex`` Table
``tab:safe_swa_results``):

  - Countries where RAW wins: ARG, MEX, COL, MMR, THA, MYS, CHN, BGD, SRB, KGZ
  - Countries where ORACLE_D is used: USA, GBR, DEU, BRA, VNM, IDN, JPN, IRN,
    ROU, ETH

The exact set is determined dynamically per-run from observed RAW vs VAN MIS;
the lists above are the empirical Qwen2.5-7B / 20-country / 310-scenario
result. Mean MIS reduction vs vanilla on that run was 4.55% with 0 losses
and 1 tie at ROU.

Setup:
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input (used for actor AND judge)
    3. Add multitp-data dataset
    4. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_openended_hybrid.py

Configuration: edit the RUN CONFIG block near the top of this file. Set
``RUN_BATCH="all"`` to do all 20 paper countries in one session.

Outputs (under ``results/openended_hybrid``):

  - ``combined/{country}.jsonl`` / ``.csv``   — raw actor+judge rows
  - ``safe/{country}/vanilla_results.csv``    — vanilla baseline (continuous-δ)
  - ``safe/{country}/raw_swa_results.csv``    — pure SWA-DPBR
  - ``safe/{country}/safe_results.csv``       — bounded-blend SAFE output
  - ``safe/{country}/hybrid_amce.json``       — hybrid AMCE + source pick
  - ``safe/{country}/summary.json``           — all six methods' alignment
  - ``compare/comparison.csv``                — long-format, all 6 methods
  - ``compare/comparison_wide.csv``           — wide-format, 1 row per country
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
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")
os.environ.setdefault("EXP24_ESS_ANCHOR_REG", "1")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RUN CONFIG — edit here, no env vars needed.                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
RUN_BATCH: str = "all"             # "b1" (1-10) | "b2" (11-20) | "all"
RUN_N_SCENARIOS: int = 500
RUN_MAX_NEW_TOKENS_ACTOR: int = 8  # constrained A/B mode (commit 0cca220)
LOAD_4BIT: bool = False

# SafeBlend hyperparameters — see src/safe_blend.py for empirical motivation.
# These are only used to compute the SAFE / ORACLE_C / ORACLE_D rows; the
# HYBRID method itself only consumes RAW and ORACLE_D outputs.
ALPHA_MAX: float = 0.30            # SWA at most 30% influence
DPBR_R_MIN: float = 0.85           # bootstrap reliability threshold
MIN_VANILLA_CONF: float = 0.5      # below this, sign gate is permissive
MAGNITUDE_RATIO_MAX: float = 2.5   # |δ_swa - δ_van| <= 2.5 · max(|δ_van|, floor)
BLEND_FLOOR: float = 0.5           # numerical floor for the magnitude bound
PERSONA_STD_MAX: float = 3.0       # ensemble disagreement upper bound
COUNTRY_MIN_ALPHA: float = 0.05    # below this, country falls back to vanilla

# DPBR keep tight (0.04 default). Plan B (1.0) only if rel_r collapses for the
# new continuous-δ regime — empirically the continuous δ has |δ| < 8 (clipped),
# matching the logit-track scale, so 0.04 should be fine.
os.environ["EXP24_VAR_SCALE"] = "0.04"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
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
print("  KAGGLE OFFLINE — Open-Ended HYBRID SWA-DPBR")
print("  Per-country routing: RAW if RAW<VAN else ORACLE_D")
print("=" * 70)

_setup_project()

ACTOR_MODEL_PATH = _resolve_model_path(ACTOR_MODEL_LOCAL_PATH, "ACTOR")
print(f"[SETUP] Actor + self-judge model path: {ACTOR_MODEL_PATH}")

if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf "
        "2>/dev/null || true",
        shell=True, check=False,
    )

from exp_paper.openended.unified_actor_judge_safe import (  # noqa: E402
    SafeUnifiedConfig, run_unified_safe,
)
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402


_PAPER_20: list[str] = list(PAPER_20_COUNTRIES)
_BATCHES: dict[str, list[str]] = {
    "all": _PAPER_20,
    "b1": _PAPER_20[:10],
    "b2": _PAPER_20[10:],
}


def main() -> None:
    if RUN_BATCH not in _BATCHES:
        raise ValueError(f"RUN_BATCH must be one of {sorted(_BATCHES)}, got {RUN_BATCH!r}")

    countries = list(_BATCHES[RUN_BATCH])
    # Distinct out_dir from the SAFE launcher so the two runs don't clobber
    # each other's combined/ JSONLs and compare/ CSVs.
    results_base = (
        f"{WORK_DIR}/results/openended_hybrid" if _on_kaggle() else
        str(Path(__file__).parent.parent / "results" / "openended_hybrid")
    )

    print(f"\n[HYBRID] batch={RUN_BATCH}  countries={countries}  n={RUN_N_SCENARIOS}")
    print(f"[HYBRID] model={ACTOR_MODEL_PATH}  (actor + self-judge, shared weights)")
    print(f"[HYBRID] results_base={results_base}")
    print(f"[HYBRID] alpha_max={ALPHA_MAX}  dpbr_r_min={DPBR_R_MIN}  "
          f"mag_ratio_max={MAGNITUDE_RATIO_MAX}  persona_std_max={PERSONA_STD_MAX}  "
          f"country_min_alpha={COUNTRY_MIN_ALPHA}")
    print(f"[HYBRID] EXP24_VAR_SCALE={os.environ['EXP24_VAR_SCALE']}  "
          f"hierarchical_prior=DISABLED")
    print(f"[HYBRID] routing rule: RAW if RAW MIS < VAN MIS, else ORACLE_D "
          f"(per-AMCE-dimension oracle)")

    cfg = SafeUnifiedConfig(
        model_name=ACTOR_MODEL_PATH,
        out_dir=results_base,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        use_real_data=os.path.isdir(MULTITP_DATA_PATH),
        n_scenarios=RUN_N_SCENARIOS,
        max_new_tokens_actor=RUN_MAX_NEW_TOKENS_ACTOR,
        load_in_4bit=LOAD_4BIT,
        countries=countries,
        model_label="qwen25_7b_openended_hybrid",
        alpha_max=ALPHA_MAX,
        dpbr_r_min=DPBR_R_MIN,
        min_vanilla_conf=MIN_VANILLA_CONF,
        magnitude_ratio_max=MAGNITUDE_RATIO_MAX,
        blend_floor=BLEND_FLOOR,
        persona_std_max=PERSONA_STD_MAX,
        country_min_alpha=COUNTRY_MIN_ALPHA,
    )
    run_unified_safe(cfg)
    gc.collect()


if __name__ == "__main__":
    main()
