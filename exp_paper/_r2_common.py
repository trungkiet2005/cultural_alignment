"""Shared Kaggle bootstrap + country-loop helpers for all Round-2 scripts.

Every ``exp_paper/exp_r2_*.py`` script imports from this module to avoid
duplicating:
    * the repo-clone/cwd bootstrap
    * HF credentials / Kaggle dep install
    * SWAConfig construction against the Kaggle dataset paths
    * a single-model load with a load-timeout guard
    * a per-country loop that writes a tidy CSV

Nothing here is SWA-specific -- the per-country work is supplied by the caller
as a ``runner_fn(model, tokenizer, scenario_df, country, cfg) -> dict`` with
the same shape as :func:`src.baseline_runner.run_baseline_vanilla`.
"""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def ensure_repo() -> str:
    """Clone the repo into Kaggle working dir if not present and put it on
    ``sys.path``. No-op when already inside the repo locally."""
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        if here not in sys.path:
            sys.path.insert(0, here)
        return here
    if not on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def default_kaggle_paths() -> Dict[str, str]:
    return {
        "multitp":    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data",
        "wvs":        "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv",
        "human_amce": "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv",
    }


def build_cfg(
    model_name: str,
    results_base: str,
    target_countries: List[str],
    *,
    n_scenarios: int = 500,
    load_in_4bit: bool = False,
    lambda_coop: float = 0.70,
    k_samples: int = 128,
):
    from src.config import SWAConfig  # deferred (after ensure_repo)

    p = default_kaggle_paths()
    return SWAConfig(
        model_name=model_name,
        n_scenarios=n_scenarios,
        batch_size=1,
        target_countries=list(target_countries),
        load_in_4bit=load_in_4bit,
        use_real_data=True,
        multitp_data_path=p["multitp"],
        wvs_data_path=p["wvs"],
        human_amce_path=p["human_amce"],
        output_dir=results_base,
        lambda_coop=lambda_coop,
        K_samples=k_samples,
    )


def load_scenarios(cfg, country: str):
    """Load scenarios for one country with the country's language."""
    from src.constants import COUNTRY_LANG
    from src.data import load_multitp_dataset

    lang = COUNTRY_LANG.get(country, "en")
    df = load_multitp_dataset(
        data_base_path=cfg.multitp_data_path,
        lang=lang,
        translator=cfg.multitp_translator,
        suffix=cfg.multitp_suffix,
        n_scenarios=cfg.n_scenarios,
    )
    df = df.copy()
    df["lang"] = lang
    return df


class _LoadTimeout(Exception):
    pass


def _ensure_vllm_installed() -> None:
    """Safety net: install vLLM on the fly if the user asked for it but the
    paper_runtime install step didn't run (or ran with the wrong backend
    because MORAL_MODEL_BACKEND was set too late). Triggered only when
    ``import vllm`` fails.
    """
    import importlib
    try:
        importlib.import_module("vllm")
        return
    except ImportError:
        pass
    print("[R2] vllm not importable — installing on the fly (one-time).")
    os.environ["MORAL_MODEL_BACKEND"] = "vllm"
    try:
        from exp_paper.paper_runtime import install_paper_kaggle_deps
        install_paper_kaggle_deps()
    except Exception as exc:
        print(f"[R2] paper_runtime install failed ({exc}); pip-installing vllm directly.")
        subprocess.run("pip install -q vllm", shell=True, check=False)
    # Last-chance smoke-test
    try:
        importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError(
            "Failed to install vLLM even after the on-the-fly fix. "
            "Set R2_BACKEND=hf_native or install vllm manually."
        ) from exc


def load_model_timed(
    model_name: str,
    *,
    backend: str = "vllm",
    load_in_4bit: bool = False,
    timeout_minutes: int = 15,
    max_seq_length: int = 2048,
) -> Tuple[object, object]:
    """Load a model via the chosen backend with a SIGALRM timeout on Linux.

    ``backend`` ∈ {"vllm", "hf_native", "unsloth"}. ``MORAL_MODEL_BACKEND`` is
    respected by the internal loaders; we also export it here so downstream
    wrappers pick the same branch. If the vLLM wheel is missing when backend
    == "vllm", this function triggers a one-time install before loading.
    """
    from src.model import load_model, load_model_hf_native  # deferred

    os.environ["MORAL_MODEL_BACKEND"] = backend
    if backend == "vllm":
        _ensure_vllm_installed()

    def _do_load():
        if backend == "vllm":
            return load_model(model_name, max_seq_length=max_seq_length, load_in_4bit=False)
        if backend == "hf_native":
            return load_model_hf_native(model_name, max_seq_length=max_seq_length, load_in_4bit=False)
        return load_model(model_name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit)

    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        return _do_load()

    def _handler(signum, frame):
        raise _LoadTimeout(
            f"Model load exceeded {timeout_minutes} min (override via env)."
        )

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_minutes * 60)
    try:
        m, t = _do_load()
        signal.alarm(0)
        return m, t
    finally:
        signal.signal(signal.SIGALRM, prev)


def _flatten_alignment(align: Dict[str, float], prefix: str = "align_") -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in (align or {}).items()}


def run_country_loop(
    *,
    model,
    tokenizer,
    cfg,
    countries: List[str],
    runner_fn: Callable,
    method_tag: str,
    out_dir: Path,
    row_extras_fn: Optional[Callable[[dict], Dict]] = None,
) -> List[Dict]:
    """Apply ``runner_fn`` to each country, write per-country CSV and return
    a flat list of summary rows.

    ``runner_fn`` is called as ``runner_fn(model, tokenizer, scenario_df,
    country, cfg)`` and must return ``{"results_df": DataFrame, "alignment":
    Dict, ...}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for ci, country in enumerate(countries):
        print(f"\n[{ci+1}/{len(countries)}] {method_tag} | {country}")
        scen = load_scenarios(cfg, country)
        t0 = time.time()
        out = runner_fn(model, tokenizer, scen, country, cfg)
        dt = time.time() - t0

        results_df = out["results_df"]
        results_df.to_csv(out_dir / f"{method_tag}_results_{country}.csv", index=False)

        row = {
            "method":      method_tag,
            "country":     country,
            "n_scenarios": len(results_df),
            "elapsed_sec": dt,
        }
        row.update(_flatten_alignment(out.get("alignment", {})))
        if row_extras_fn is not None:
            row.update(row_extras_fn(out))
        rows.append(row)

        a = out.get("alignment", {})
        print(f"  ✓ {method_tag} | {country}  "
              f"MIS={a.get('mis', float('nan')):.4f}  "
              f"r={a.get('pearson_r', float('nan')):+.3f}  "
              f"JSD={a.get('jsd', float('nan')):.4f}  ({dt:.0f}s)")

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return rows


def save_summary(rows: List[Dict], out_dir: Path, filename: str) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"\n[SAVED] {path}  ({len(df)} rows)")
    return path
