"""
EXP-24 Dual-Pass Bootstrap IS — shared base (DPBR)
====================================================
Core algorithm + runner reused by every per-model file.
NOT self-contained: env bootstrap must have run before this is imported.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
    patch_swa_runner_controller,
)
from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    print_tracker_ready_report,
    try_load_reference_comparison,
)
from src.baseline_runner import run_baseline_vanilla
from src.config import SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.data import load_multitp_dataset
from src.model import load_model, load_model_hf_native, setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.scenarios import generate_multitp_scenarios
from src.swa_runner import run_country_experiment

# ─── Shared hyperparameters (EXP-09 base + EXP-24 dual-pass) ─────────────────
EXP_BASE      = "EXP-24"
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]

N_SCENARIOS  = 500
BATCH_SIZE   = 1
SEED         = 42
LAMBDA_COOP  = 0.70

# K_HALF, VAR_SCALE, prior hyperparams: experiment_DM.exp24_dpbr_core

RESULTS_BASE = "/kaggle/working/cultural_alignment/results/exp24_model_sweep"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
PREFLIGHT_TIMEOUT_MINUTES = int(os.environ.get("EXP24_PREFLIGHT_TIMEOUT_MINUTES", "15"))


def _exp24_seed() -> int:
    return int(os.environ.get("EXP24_SEED", str(SEED)))


def _lambda_coop_from_env() -> float:
    v = os.environ.get("EXP24_LAMBDA_COOP", "").strip()
    return float(v) if v else LAMBDA_COOP


# DPBR controller + PRIOR_STATE: experiment_DM.exp24_dpbr_core (same as exp24_dual_pass_bootstrap.py)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]:
        if not root:
            continue
        hub = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target = os.path.join(hub, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as exc:
                print(f"[CLEANUP] error: {exc}")


def _build_cfg(
    model_name: str,
    swa_root: str,
    *,
    load_in_4bit: bool = True,
    target_countries: List[str],
) -> SWAConfig:
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(target_countries), load_in_4bit=load_in_4bit, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=swa_root,
        lambda_coop=_lambda_coop_from_env(), K_samples=128,
    )


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _write_preflight_error_flag(flag_path: Path, message: str) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(message + "\n", encoding="utf-8")
    print(f"[PREFLIGHT] wrote error flag: {flag_path}")


def _preflight_model_load(
    model_name: str,
    timeout_minutes: int,
    *,
    load_in_4bit: bool = True,
    use_hf_native: bool = False,
    use_vllm: bool = False,
) -> tuple[bool, str]:
    timeout_seconds = max(1, timeout_minutes) * 60
    _lb = "True" if load_in_4bit else "False"
    _mn = json.dumps(model_name)
    if use_vllm:
        _load_block = (
            "from src.vllm_causal import load_model_vllm\n"
            f"model, tokenizer = load_model_vllm({_mn}, max_seq_length=2048, load_in_4bit={_lb})\n"
        )
    elif use_hf_native:
        _load_block = (
            "from src.model import load_model_hf_native\n"
            f"model, tokenizer = load_model_hf_native({_mn}, max_seq_length=2048, load_in_4bit={_lb})\n"
        )
    else:
        _load_block = (
            "from src.model import load_model\n"
            f"model, tokenizer = load_model({_mn}, max_seq_length=2048, load_in_4bit={_lb})\n"
        )
    code = f"""
import gc
import os
import torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
{_load_block}
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("PREFLIGHT_OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Model pre-flight timeout after {timeout_minutes} minute(s)."

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-1500:]
        return False, f"Model pre-flight failed (rc={result.returncode}). Details:\n{tail}"

    if "PREFLIGHT_OK" not in (result.stdout or ""):
        return False, "Model pre-flight did not confirm successful load."
    return True, "ok"


def _abort_kaggle_run(reason: str) -> None:
    print(f"[PREFLIGHT][ERROR] {reason}")
    if _on_kaggle():
        raise SystemExit("[PREFLIGHT] Stopping Kaggle run to avoid wasting GPU.")
    raise RuntimeError(reason)


# ─── Core run function ─────────────────────────────────────────────────────────
def run_for_model(
    model_name: str,
    model_short: str,
    *,
    load_in_4bit: bool = True,
    target_countries: Optional[List[str]] = None,
    results_base: Optional[str] = None,
    use_hf_native: bool = False,
    use_vllm: bool = False,
) -> None:
    """
    Full EXP-24 run for a single model.
    Called directly from each per-model entry script.

    Parameters
    ----------
    load_in_4bit
        If True (default), load 4-bit quantized weights (bitsandbytes).
        If False, load full bf16 (needs enough VRAM — ~16–20 GB for 7–9B).
    use_hf_native
        If True, load with Hugging Face ``AutoModelForCausalLM`` / ``AutoTokenizer`` only
        (no Unsloth). Used by ``exp_24/hf_full/*`` upstream-weight runs.
    use_vllm
        If True, load with ``vllm.LLM`` and score via ``generate`` with per-language A/B ``allowed_token_ids``.
        Mutually exclusive with ``use_hf_native``. Used by ``exp_24/hf_vllm/*``.
    """
    if use_hf_native and use_vllm:
        raise ValueError("use_hf_native and use_vllm cannot both be True")

    _seed = _exp24_seed()
    setup_seeds(_seed)

    countries: List[str] = list(target_countries) if target_countries is not None else list(TARGET_COUNTRIES)
    rb = results_base if results_base is not None else RESULTS_BASE

    exp_id   = f"{EXP_BASE}-{model_short.upper()}"
    swa_root = f"{rb}/{model_short}/swa"
    cmp_root = f"{rb}/{model_short}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {exp_id}: Dual-Pass Bootstrap IS [{model_name}]")
    print(f"{'='*70}")
    print(f"[THEORY] K_half={K_HALF}×2={K_HALF*2} total  |  VAR_SCALE={VAR_SCALE}")
    print(f"[THEORY] r = exp(-(δ*₁-δ*₂)² / {VAR_SCALE})  →  δ* = r·(δ*₁+δ*₂)/2")
    _ear = os.environ.get("EXP24_ESS_ANCHOR_REG", "1").strip().lower()
    _ess_on = _ear not in ("0", "false", "no", "off")
    print(f"[THEORY] ESS anchor blend: {'ON (δ_micro = α·anchor + (1-α)·δ_base + δ*)' if _ess_on else 'OFF (legacy δ_micro = anchor + δ*)'}")
    _q = "4-bit" if load_in_4bit else "bf16 (full, no quant)"
    if use_vllm:
        _bk = "vLLM"
    elif use_hf_native:
        _bk = "HF native"
    else:
        _bk = "Unsloth"
    print(f"[CONFIG] seed={_seed}  lambda_coop={_lambda_coop_from_env()}  weights={_q}  loader={_bk}")

    cfg     = _build_cfg(
        model_name, swa_root, load_in_4bit=load_in_4bit, target_countries=countries
    )
    out_dir = Path(swa_root) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    preflight_flag = out_dir / "preflight_error.flag"

    skip_vllm_pf = use_vllm and os.environ.get("EXP24_VLLM_PREFLIGHT", "").strip() != "1"
    if skip_vllm_pf:
        print(
            "[PREFLIGHT] skipping subprocess vLLM load (set EXP24_VLLM_PREFLIGHT=1 to run full preflight)"
        )
        ok, message = True, "ok"
    else:
        print(f"[PREFLIGHT] checking model load with timeout={PREFLIGHT_TIMEOUT_MINUTES} minute(s)")
        ok, message = _preflight_model_load(
            model_name,
            PREFLIGHT_TIMEOUT_MINUTES,
            load_in_4bit=load_in_4bit,
            use_hf_native=use_hf_native,
            use_vllm=use_vllm,
        )
    if not ok:
        _write_preflight_error_flag(preflight_flag, message)
        _abort_kaggle_run(message)
    if preflight_flag.exists():
        preflight_flag.unlink(missing_ok=True)
    print("[PREFLIGHT] PASS")

    if use_vllm:
        from src.vllm_causal import load_model_vllm as _load
    elif use_hf_native:
        _load = load_model_hf_native
    else:
        _load = load_model
    model, tokenizer = _load(model_name, max_seq_length=2048, load_in_4bit=load_in_4bit)

    rows: List[dict] = []
    dp_method = f"{exp_id}_dual_pass"
    try:
        for ci, country in enumerate(countries):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue
            print(f"\n[{ci+1}/{len(countries)}] {exp_id} | {country}")

            scen     = _load_scen(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

            # ── Vanilla baseline (same scenarios, no personas / no SWA-IS) ──
            print(f"  [VANILLA] Token-logit baseline …")
            bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
            bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
            rows.append({
                "model": model_name,
                "method": "baseline_vanilla",
                "country": country,
                **{f"align_{k}": v for k, v in bl["alignment"].items()},
                "flip_rate": float("nan"),
                "n_scenarios": len(bl["results_df"]),
                "final_delta_country": float("nan"),
                "final_alpha_h": float("nan"),
                "mean_reliability_r": float("nan"),
                "mean_bootstrap_var": float("nan"),
                "mean_ess_pass1": float("nan"),
                "mean_ess_pass2": float("nan"),
                "mean_ess_anchor_alpha": float("nan"),
                "std_reliability_r": float("nan"),
                "std_bootstrap_var": float("nan"),
                "mean_positional_bias": float("nan"),
                "utilitarianism_slope_hat": float("nan"),
                "utilitarianism_slope_n": 0,
            })

            # ── DPBR (EXP-24) ──
            PRIOR_STATE.clear()
            PRIOR_STATE[country] = BootstrapPriorState()
            print(f"  [DPBR] Dual-pass bootstrap IS …")
            patch_swa_runner_controller()

            results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)

            results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
            append_rows_csv(
                str(Path(cmp_root) / "per_dim_breakdown.csv"),
                flatten_per_dim_alignment(
                    summary.get("per_dimension_alignment", {}),
                    model=model_name, method=dp_method, country=country,
                ),
            )
            ps  = PRIOR_STATE.get(country, BootstrapPriorState()).stats
            mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

            def _std(col: str) -> float:
                if col not in results_df.columns or len(results_df) < 2:
                    return float("nan")
                return float(results_df[col].std(ddof=1))

            us = summary.get("utilitarianism_slope") or {}
            rows.append({
                "model": model_name, "method": dp_method, "country": country,
                **{f"align_{k}": v for k, v in summary["alignment"].items()},
                "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
                "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
                "mean_reliability_r": mea("reliability_r"),
                "mean_bootstrap_var": mea("bootstrap_var"),
                "mean_ess_pass1": mea("ess_pass1"), "mean_ess_pass2": mea("ess_pass2"),
                "mean_ess_anchor_alpha": mea("ess_anchor_alpha"),
                "std_reliability_r": _std("reliability_r"),
                "std_bootstrap_var": _std("bootstrap_var"),
                "mean_positional_bias": mea("positional_bias"),
                "utilitarianism_slope_hat": float(us.get("slope_hat", float("nan"))),
                "utilitarianism_slope_n": int(us.get("n_obs", 0) or 0),
            })

            pda = summary.get("per_dimension_alignment", {})
            if pda:
                print(f"\n  ┌── Per-Dimension ({country}) ──")
                for dk, dd in sorted(pda.items()):
                    hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                    print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
                print(f"  └── MIS={summary['alignment']['mis']:.4f}  "
                      f"r={summary['alignment']['pearson_r']:+.3f}  "
                      f"Flip={summary['flip_rate']:.1%}  "
                      f"rel_r(avg)={mea('reliability_r'):.3f}")

            torch.cuda.empty_cache()
            gc.collect()

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _free_model_cache(model_name)

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(Path(cmp_root) / "comparison.csv", index=False)

    vanilla_df = cmp_df[cmp_df["method"] == "baseline_vanilla"].copy()
    dpbr_df    = cmp_df[cmp_df["method"] == dp_method].copy()

    print(f"\n{'#'*70}\n# {exp_id} FINAL REPORT\n{'#'*70}")
    if not vanilla_df.empty:
        print_alignment_table(vanilla_df, title=f"{exp_id} VANILLA (no personas / no IS)")
    print_alignment_table(dpbr_df, title=f"{exp_id} DPBR")
    if not vanilla_df.empty and not dpbr_df.empty:
        print_metric_comparison(
            vanilla_df, dpbr_df,
            title=f"{exp_id} vs Vanilla (MIS)",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="baseline_vanilla",
                cur_method=dp_method,
            ),
        )
    if not dpbr_df.empty:
        _rr = dpbr_df["mean_reliability_r"]
        rel_mean = float(_rr.mean()) if _rr.notna().any() else float("nan")
        print(
            f"\n  DPBR MEAN MIS={dpbr_df['align_mis'].mean():.4f}  "
            f"r={dpbr_df['align_pearson_r'].mean():+.3f}  "
            f"Flip={dpbr_df['flip_rate'].mean():.1%}  "
            f"rel_r={rel_mean:.3f}"
        )
        print(f"  (EXP-09 SOTA: 0.3975  |  EXP-24 multi-model ref: 0.3969)")
    ref = try_load_reference_comparison()
    if ref is not None and not dpbr_df.empty:
        print_metric_comparison(
            ref, dpbr_df, title=f"{exp_id} vs EXP-01 SWA-PTIS (MIS)",
            spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                             cur_method=dp_method),
        )
    print_tracker_ready_report(
        cmp_df, exp_id=exp_id, cur_method=dp_method,
        per_dim_csv_path=str(Path(cmp_root) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{exp_id}] DONE — {cmp_root}")
