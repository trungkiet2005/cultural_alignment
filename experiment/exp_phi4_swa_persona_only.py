#!/usr/bin/env python3
"""
SWA persona-mean only (no PT-IS) + vanilla baseline
===================================================

Chạy **chỉ** pipeline SWA với N persona (forward batch base + agents), **không**
chạy importance sampling / Prospect-Theory (_is_solve_decision): quyết định =
mean(debiased persona logit gaps), tức ``delta_consensus`` sau debias A↔B.

So sánh với **vanilla** (một forward, không persona) cùng scenario/country.

Trên **Kaggle**, nếu đã thêm Models *Magistral Small 2509* (Transformers), mặc định
load từ ``/kaggle/input/models/mistral-ai/.../1`` khi thư mục đó tồn tại; không thì
Hub ``unsloth/Magistral-Small-2509-unsloth-bnb-4bit``. Với **đường dẫn thư mục local**,
dùng ``load_model_hf_native`` (Unsloth hay lỗi Mistral3 → nhánh vision / image processor).
Override: ``SWA_PERSONA_ONLY_MODEL``, ``SWA_PERSONA_ONLY_KAGGLE_WEIGHTS``, ``SWA_PERSONA_ONLY_SHORT``.
Kaggle cài ``transformers>=5.5`` (Mistral3 tokenizer). Nếu bundle thiếu tiktoken, tokenizer tải từ
Hub ``mistralai/Magistral-Small-2509`` (cần mạng / ``HF_TOKEN``). Ghi đè: ``MORAL_TOKENIZER_HUB_ID``.

Kaggle::

    !python experiment/exp_phi4_swa_persona_only.py

Local / Jupyter: đặt ``MULTITP_DATA_PATH``, ``WVS_DATA_PATH``, ``HUMAN_AMCE_PATH``
nếu cần; repo root được đoán từ ``cwd`` khi ``__file__`` không có (notebook).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

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
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE],
            check=True,
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    for cmd in [
        "pip install -q bitsandbytes scipy tqdm",
        "pip install sentencepiece protobuf \"datasets==4.3.0\" \"huggingface_hub>=0.34.0\" hf_transfer",
        "pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth",
        # Magistral / Mistral3: need TOKENIZER_MAPPING + tokenizer stack newer than 4.56; Kaggle Models
        # bundle often lacks tiktoken blob → tokenizer loaded from Hub while weights stay local.
        "pip install \"transformers>=5.5.0,<6.0\"",
        "pip install --no-deps trl==0.22.2",
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

import torch

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import pandas as pd

from experiment_DM.exp_reporting import (
    CompareSpec,
    print_alignment_table,
    print_metric_comparison,
)
from src.baseline_runner import run_baseline_vanilla
from src.config import SWAConfig, model_slug
from src.constants import COUNTRY_LANG
from src.controller import ImplicitSWAController
from src.data import load_multitp_dataset
from src.model import load_model, load_model_hf_native, setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.scenarios import generate_multitp_scenarios
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment

# ---------------------------------------------------------------------------
# Controller: giữ debias 2-pass + mean persona; bỏ toàn bộ PT-IS (delta* = 0)
# ---------------------------------------------------------------------------


class PersonaMeanOnlyController(ImplicitSWAController):
    """SWA forward + persona pooling only; no K-sample prospect IS."""

    @torch.no_grad()
    def _is_solve_decision(
        self,
        delta_base_scalar: torch.Tensor,
        delta_agents: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        return torch.zeros((), device=self.device)


_KAGGLE_MAGISTRAL_WEIGHTS = os.environ.get(
    "SWA_PERSONA_ONLY_KAGGLE_WEIGHTS",
    "/kaggle/input/models/mistral-ai/magistral-small-2509/transformers/magistral-small-2509/1",
).strip()
_MAGISTRAL_HUB_DEFAULT = "unsloth/Magistral-Small-2509-unsloth-bnb-4bit"


def _default_swa_persona_model() -> str:
    v = os.environ.get("SWA_PERSONA_ONLY_MODEL", "").strip()
    if v:
        return v
    if _on_kaggle() and os.path.isdir(_KAGGLE_MAGISTRAL_WEIGHTS):
        return _KAGGLE_MAGISTRAL_WEIGHTS
    return _MAGISTRAL_HUB_DEFAULT


MODEL_NAME = _default_swa_persona_model()
MODEL_SHORT = os.environ.get("SWA_PERSONA_ONLY_SHORT", "magistral_small_2509").strip() or "magistral_small_2509"

TARGET_COUNTRIES: List[str] = [
    c.strip()
    for c in os.environ.get(
        "SWA_PERSONA_ONLY_COUNTRIES",
        "USA,CHN,JPN,DEU,BRA",
    ).split(",")
    if c.strip()
]

N_SCENARIOS = int(os.environ.get("SWA_PERSONA_ONLY_N_SCENARIOS", "500"))
BATCH_SIZE = int(os.environ.get("SWA_PERSONA_ONLY_BATCH_SIZE", "1"))
SEED = int(os.environ.get("SWA_PERSONA_ONLY_SEED", "42"))
LAMBDA_COOP = float(os.environ.get("SWA_PERSONA_ONLY_LAMBDA_COOP", "0.70"))

_KAGGLE_MULTI = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
_KAGGLE_WVS = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
_KAGGLE_AMCE = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


def _repo_root() -> Path:
    """Repo root for local paths. ``__file__`` is undefined in Jupyter/IPython."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        cwd = Path.cwd().resolve()
        if (cwd / "src" / "controller.py").is_file():
            return cwd
        exp = cwd / "experiment"
        if (exp.parent / "src" / "controller.py").is_file():
            return exp.parent
        return cwd


_LOCAL_ROOT = _repo_root()
_LOCAL_MULTI = str(_LOCAL_ROOT / "data" / "data")
_LOCAL_WVS = str(_LOCAL_ROOT / "data" / "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv")
_LOCAL_AMCE = str(_LOCAL_ROOT / "data" / "data" / "country_specific_ACME.csv")


def _path_or_env(env_name: str, kaggle_p: str, local_p: str) -> str:
    v = os.environ.get(env_name, "").strip()
    if v:
        return v
    return kaggle_p if _on_kaggle() else local_p


MULTITP_DATA_PATH = _path_or_env("MULTITP_DATA_PATH", _KAGGLE_MULTI, _LOCAL_MULTI)
WVS_DATA_PATH = _path_or_env("WVS_DATA_PATH", _KAGGLE_WVS, _LOCAL_WVS)
HUMAN_AMCE_PATH = _path_or_env("HUMAN_AMCE_PATH", _KAGGLE_AMCE, _LOCAL_AMCE)

RESULTS_BASE = os.environ.get(
    "SWA_PERSONA_ONLY_RESULTS",
    (
        "/kaggle/working/cultural_alignment/results/exp_phi4_swa_persona_only"
        if _on_kaggle()
        else str(_LOCAL_ROOT / "results" / "exp_phi4_swa_persona_only")
    ),
)


def _build_cfg(model_name: str, swa_root: str) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=swa_root,
        lambda_coop=LAMBDA_COOP,
        K_samples=128,
    )


def _load_swa_persona_model() -> tuple:
    """Unsloth mis-routes Mistral3 (Magistral) text checkpoints to vision → use HF native for local dirs."""
    force_hf = os.environ.get("SWA_PERSONA_ONLY_HF_NATIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_hf or os.path.isdir(MODEL_NAME):
        return load_model_hf_native(MODEL_NAME, max_seq_length=2048, load_in_4bit=True)
    return load_model(MODEL_NAME, max_seq_length=2048, load_in_4bit=True)


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path,
            lang=lang,
            translator=cfg.multitp_translator,
            suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def main() -> None:
    setup_seeds(SEED)

    swa_root = f"{RESULTS_BASE}/{MODEL_SHORT}/swa"
    cmp_root = f"{RESULTS_BASE}/{MODEL_SHORT}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(MODEL_NAME, swa_root)
    # Local Kaggle bundle paths often end in "/1" → model_slug would be useless.
    out_nested = MODEL_SHORT if os.path.isdir(MODEL_NAME) else model_slug(MODEL_NAME)
    out_dir = Path(swa_root) / out_nested
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    exp_tag = "SWA_persona_mean_only"
    print(f"\n{'='*70}\n  {exp_tag}  |  {MODEL_NAME}\n{'='*70}")
    print(f"[CONFIG] seed={SEED}  countries={TARGET_COUNTRIES}  n_scen={N_SCENARIOS}")
    print(f"[CONFIG] PT-IS disabled (persona debiased mean only)")
    print(f"[OUT] {out_dir}")

    model, tokenizer = _load_swa_persona_model()

    _swa_runner_mod.ImplicitSWAController = PersonaMeanOnlyController

    rows: List[dict] = []
    try:
        for ci, country in enumerate(TARGET_COUNTRIES):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue
            print(f"\n[{ci+1}/{len(TARGET_COUNTRIES)}] {country}")

            scen = _load_scen(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

            print("  [VANILLA] …")
            bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
            bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
            rows.append(
                {
                    "model": MODEL_NAME,
                    "method": "baseline_vanilla",
                    "country": country,
                    **{f"align_{k}": v for k, v in bl["alignment"].items()},
                    "n_scenarios": len(bl["results_df"]),
                }
            )

            print(f"  [{exp_tag}] …")
            results_df, summary = run_country_experiment(
                model, tokenizer, country, personas, scen, cfg
            )
            results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
            rows.append(
                {
                    "model": MODEL_NAME,
                    "method": exp_tag,
                    "country": country,
                    **{f"align_{k}": v for k, v in summary["alignment"].items()},
                    "flip_rate": summary["flip_rate"],
                    "n_scenarios": summary["n_scenarios"],
                }
            )

            torch.cuda.empty_cache()
            gc.collect()

    finally:
        _swa_runner_mod.ImplicitSWAController = ImplicitSWAController
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(Path(cmp_root) / "comparison.csv", index=False)

    van = cmp_df[cmp_df["method"] == "baseline_vanilla"].copy()
    swa = cmp_df[cmp_df["method"] == exp_tag].copy()

    print(f"\n{'#'*70}\n# {exp_tag} — summary\n{'#'*70}")
    if not van.empty:
        print_alignment_table(van, title="Vanilla (no personas)")
    if not swa.empty:
        print_alignment_table(swa, title="SWA persona-mean only (no PT-IS)")
    if not van.empty and not swa.empty:
        print_metric_comparison(
            van,
            swa,
            title="Persona-mean vs Vanilla (MIS)",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="baseline_vanilla",
                cur_method=exp_tag,
            ),
        )
    print(f"\n[{exp_tag}] DONE — {cmp_root}")


if __name__ == "__main__":
    main()
