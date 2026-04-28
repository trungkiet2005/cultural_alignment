#!/usr/bin/env python3
"""
Paper sweep — Open-Ended SAFE SWA-DPBR — guaranteed-not-worse-than-vanilla
==========================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

Three architectural fixes vs the vanilla unified pipeline (see
``exp_paper/openended/unified_actor_judge_safe.py`` module docstring for
detailed rationale):

  1. **Continuous-δ from judge logits** — the judge runs ONE forward step on a
     decisive prompt and we read logits at the A/B token positions. Replaces
     pseudo-δ parsing through a 3-5-bucket bottleneck.
  2. **Per-scenario vanilla-anchored bounded blend** — α ≤ 0.30 with four hard
     safety gates. When ANY gate fails, α=0 ⇒ pure vanilla output.
  3. **Country-level abstain** — if mean α across the country is below
     ``country_min_alpha``, revert the entire country to vanilla.

Outputs three method labels per country in the comparison CSV:
  - ``vanilla_continuous``: vanilla baseline using continuous judge δ
  - ``raw_swa_dpbr``: pure SWA-DPBR output (for ablation)
  - ``safe_swa_blend``: blended + country-level-fallback output (RECOMMENDED)

Compute budget: ~3-5h on Kaggle RTX 6000 96GB for 20 countries × 310 scenarios
× 5 personas × (1 actor gen 8 tokens + 1 judge 1-step forward). Comfortably
inside the 12h session limit.

Setup:
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input (used for actor AND judge)
    3. Add multitp-data dataset
    4. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_openended_safe.py

Configuration: edit the RUN CONFIG block near the top of this file. Set
``RUN_BATCH="all"`` to do all 20 paper countries in one session.
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
RUN_BATCH: str = "all"            # "b1" (1-10) | "b2" (11-20) | "all"
RUN_N_SCENARIOS: int = 500
RUN_MAX_NEW_TOKENS_ACTOR: int = 8  # constrained A/B mode (commit 0cca220)
LOAD_4BIT: bool = False

# SafeBlend hyperparameters — see src/safe_blend.py for empirical motivation.
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
print("  KAGGLE OFFLINE — Open-Ended SAFE SWA-DPBR")
print("  Continuous-δ + bounded blend + country-level abstain")
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
    results_base = (
        f"{WORK_DIR}/results/openended_safe" if _on_kaggle() else
        str(Path(__file__).parent.parent / "results" / "openended_safe")
    )

    print(f"\n[SAFE] batch={RUN_BATCH}  countries={countries}  n={RUN_N_SCENARIOS}")
    print(f"[SAFE] model={ACTOR_MODEL_PATH}  (actor + self-judge, shared weights)")
    print(f"[SAFE] results_base={results_base}")
    print(f"[SAFE] alpha_max={ALPHA_MAX}  dpbr_r_min={DPBR_R_MIN}  "
          f"mag_ratio_max={MAGNITUDE_RATIO_MAX}  persona_std_max={PERSONA_STD_MAX}  "
          f"country_min_alpha={COUNTRY_MIN_ALPHA}")
    print(f"[SAFE] EXP24_VAR_SCALE={os.environ['EXP24_VAR_SCALE']}  "
          f"hierarchical_prior=DISABLED")

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
        model_label="qwen25_7b_openended_safe",
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
