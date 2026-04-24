#!/usr/bin/env python3
"""Cross-model logit-conditioning vs SWA-DPBR MIS-gain correlation.

Generalises ``exp_r2_logit_conditioning.py`` (Phi-4 only) to all six
paper backbones, then computes the per-(model, country) correlation
between vanilla decision-margin and SWA-DPBR's relative MIS reduction.

Output: a single 6-models x 20-countries scatter dataset plus an
aggregated correlation table (Pearson r, Spearman rho, p-values) at
two granularities:
  • per-model:  N=20  (one (margin, gain) per country)
  • pooled:     N=120 (all model x country cells together)

The architectural-failure claim from \\S\\ref{sec:discussion}-that
poorly-conditioned decision logits cap SWA-DPBR's headroom-currently
rests on a single backbone. This script extends it to a 6-backbone
panel.

Two-step run (each step is independent):

  Step 1 (GPU, ~30 min on H100 per model): vanilla logit-conditioning
    !python exp_paper/round3/posthoc/exp_r3_logit_conditioning_cross_model.py
  This walks every model and writes ``logit_cond_<model>.csv``.

  Step 2 (CPU, post-hoc): aggregate against SWA-DPBR per-country MIS gains
    !python exp_paper/round3/posthoc/exp_r3_logit_conditioning_cross_model.py --aggregate-only
  Reads the step-1 CSVs + the existing per-model swa/baseline summaries,
  produces the cross-model correlation table.

Env overrides:
    R2_MODELS         comma list of model HF ids (default: 6 paper models)
    R2_COUNTRIES      comma ISO3 list (default: all 20)
    R2_N_SCENARIOS    per-country scenario count (default: 300)
    R2_BACKEND        vllm (default) | hf_native
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _r2_bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))

import argparse
import gc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()

# Six paper models. (display_name, HF id, dir_short, slug).
#
# Llama-3.3-70B uses the Unsloth bnb-4bit weights (~40 GB) because the BF16
# checkpoint (~140 GB) cannot fit Kaggle's ~20 GB working disk. All other
# models comfortably fit as BF16 via vLLM. Per-model backend/dtype overrides
# are applied below in ``_run_one_model``.
DEFAULT_MODELS: List[Tuple[str, str, str, str]] = [
    ("Llama-3.3-70B",        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
     "llama_3_3_70b",        "meta-llama-3.3-70b-instruct"),
    ("Magistral-Small-2509", "mistralai/Magistral-Small-2509",
     "magistral_small_2509", "magistral-small-2509"),
    ("Phi-4",                "microsoft/phi-4",
     "phi_4",                "phi-4"),
    ("Qwen3-VL-8B",          "Qwen/Qwen3-VL-8B-Instruct",
     "qwen3_vl_8b",          "qwen3-vl-8b-instruct"),
    ("Qwen2.5-7B",           "Qwen/Qwen2.5-7B-Instruct",
     "qwen2_5_7b",           "qwen2.5-7b-instruct"),
    ("Phi-3.5-mini",         "microsoft/Phi-3.5-mini-instruct",
     "phi_3_5_mini",         "phi-3.5-mini-instruct"),
]

# Models that must use Unsloth 4-bit (too large for Kaggle disk as BF16).
_UNSLOTH_4BIT_MODELS = {"unsloth/Llama-3.3-70B-Instruct-bnb-4bit"}

# Rough on-disk size estimate per HF id, in GB. Used only for the pre-flight
# disk check; being off by 20% is fine. BF16 ≈ 2 bytes/param, 4-bit ≈ 0.55
# bytes/param (nf4 + absmax scales + tokenizer/config).
_MODEL_DISK_GB: Dict[str, float] = {
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit": 42.0,
    "mistralai/Magistral-Small-2509":           48.0,
    "microsoft/phi-4":                          28.0,
    "Qwen/Qwen3-VL-8B-Instruct":                17.0,
    "Qwen/Qwen2.5-7B-Instruct":                 15.0,
    "microsoft/Phi-3.5-mini-instruct":          8.0,
}

# Disk-headroom margin on top of the estimated checkpoint size (GB). Accounts
# for HF's 2x footprint during the .incomplete → final rename.
_DISK_SAFETY_GB = 4.0


def _free_disk_gb(path: str = "/") -> float:
    import shutil
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return float("inf")


def _purge_all_hf_cache() -> None:
    """Nuke every HF hub cache root. Last-resort disk reclaim."""
    import shutil
    roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
        "/kaggle/working/.cache/huggingface",
    ]
    for root in roots:
        if not root:
            continue
        hub = os.path.join(root, "hub")
        for target in (hub, root):
            if os.path.isdir(target):
                try:
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"  [cache] swept {target}")
                except Exception:
                    pass

if "R2_MODELS" in os.environ:
    MODELS = []
    for tok in os.environ["R2_MODELS"].split(";"):
        parts = [s.strip() for s in tok.split(",")]
        if len(parts) >= 4:
            MODELS.append(tuple(parts[:4]))
else:
    MODELS = DEFAULT_MODELS

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402

N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "300"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round3/logit_conditioning_cross_model"
    if on_kaggle()
    else "results/exp24_round3/logit_conditioning_cross_model"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_paper_20c"
    if on_kaggle()
    else "results/exp24_paper_20c"
)


# ─── per-model logit-conditioning sweep ──────────────────────────────────────
def _run_one_model(display_name: str, hf_id: str, dir_short: str) -> None:
    """Vanilla forward pass + per-scenario margin/entropy across countries."""
    # NOTE: we do NOT call install_paper_kaggle_deps() here. It's called once
    # at the top of main() for the initial backend (vLLM). Re-calling it per
    # model with a different backend installs unsloth/transformers 5.x which
    # breaks the already-imported pyarrow ABI (IpcReadOptions size mismatch)
    # and kills every subsequent model load.
    from src.logit_conditioning import diagnose_country  # noqa: E402
    from src.model import setup_seeds  # noqa: E402

    setup_seeds(42)
    # Per-model backend override: models that are too large for Kaggle disk
    # as BF16 fall back to Unsloth 4-bit; everything else uses vLLM BF16.
    if hf_id in _UNSLOTH_4BIT_MODELS:
        # Unsloth path requires the unsloth package. If it isn't available
        # in this process, skip rather than poison the env by pip-installing
        # unsloth mid-run (which drags in transformers 5.x and breaks
        # pyarrow for every later vLLM model).
        try:
            import unsloth  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"Skipping {hf_id}: unsloth not installed and installing it "
                f"mid-run would break pyarrow/transformers for the other "
                f"models. Run this model in a separate Kaggle notebook with "
                f"MORAL_MODEL_BACKEND=unsloth set BEFORE the first import."
            ) from exc
        backend, use_4bit = "unsloth", True
        print(f"  [backend] forcing unsloth 4-bit for {hf_id} (disk headroom)")
    else:
        backend = "vllm"
        use_4bit = False

    # Pre-flight disk check. If free disk < est model size + safety margin,
    # first try a global cache sweep; if still short, raise so main() logs
    # and moves on.
    need = _MODEL_DISK_GB.get(hf_id, 20.0) + _DISK_SAFETY_GB
    free = _free_disk_gb("/")
    if free < need:
        print(f"  [disk] free={free:.1f} GB < need≈{need:.1f} GB — sweeping HF cache")
        _purge_all_hf_cache()
        free = _free_disk_gb("/")
    if free < need:
        raise RuntimeError(
            f"Insufficient disk: free={free:.1f} GB, need≈{need:.1f} GB for {hf_id}. "
            f"Skipping (run this model in its own session, or enable Kaggle persistent disk)."
        )
    print(f"  [disk] free={free:.1f} GB OK for {hf_id} (need≈{need:.1f} GB)")

    cfg = build_cfg(hf_id, str(OUT_DIR), COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=use_4bit)
    model, tokenizer = load_model_timed(hf_id, backend=backend, load_in_4bit=use_4bit)

    rows: List[Dict] = []
    for country in COUNTRIES:
        try:
            scen = load_scenarios(cfg, country)
            agg, _ = diagnose_country(model, tokenizer, country, scen, cfg)
            rows.append({
                "model":          display_name,
                "model_short":    dir_short,
                "country":        country,
                **agg,
            })
            print(f"  ✓ {display_name} {country}  margin={agg.get('mean_margin', float('nan')):.3f}  "
                  f"entropy={agg.get('mean_entropy', float('nan')):.3f}")
        except Exception as exc:
            print(f"[error] {display_name} {country}: {exc}")
            rows.append({"model": display_name, "country": country,
                         "error": str(exc)[:500]})

    out_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")

    # cleanup: free GPU/RAM, then wipe HF cache for this model so the next
    # model has disk headroom (Kaggle /root is ~20GB, a single 70B BF16
    # checkpoint is ~140GB — we must evict between models).
    del model, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _purge_hf_cache(hf_id)


def _purge_hf_cache(hf_id: str) -> None:
    """Delete the HF hub cache entry for ``hf_id`` to reclaim disk."""
    import shutil
    cache_roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
        "/kaggle/working/.cache/huggingface",
    ]
    # HF converts "org/name" -> "models--org--name" on disk.
    dir_name = "models--" + hf_id.replace("/", "--")
    for root in cache_roots:
        if not root:
            continue
        for sub in ("hub", ""):
            cand = os.path.join(root, sub, dir_name) if sub else os.path.join(root, dir_name)
            if os.path.isdir(cand):
                try:
                    shutil.rmtree(cand, ignore_errors=True)
                    print(f"  [cache] purged {cand}")
                except Exception as exc:
                    print(f"  [cache] failed to purge {cand}: {exc}")


# ─── post-hoc aggregation against SWA-DPBR per-country MIS gains ─────────────
def _load_per_country_mis(dir_short: str, slug: str) -> Optional[pd.DataFrame]:
    """Find vanilla-vs-SWA-DPBR per-country MIS for this model."""
    base = PAPER_RESULTS_BASE
    candidates = [
        f"{base}/{dir_short}/comparison.csv",
        f"{base}/**/{dir_short}/**/comparison.csv",
    ]
    if on_kaggle():
        for d in sorted(glob.glob("/kaggle/input/*")):
            candidates.extend([
                f"{d}/results/exp24_paper_20c/{dir_short}/comparison.csv",
                f"{d}/**/{dir_short}/**/comparison.csv",
            ])
    for pat in candidates:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return pd.read_csv(hits[0])
    return None


def _aggregate() -> None:
    """Aggregate per-model logit_cond_*.csv against per-country MIS gains."""
    from scipy.stats import pearsonr, spearmanr

    pooled_rows: List[Dict] = []
    per_model_rows: List[Dict] = []

    for (display_name, _, dir_short, slug) in MODELS:
        cond_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
        if not cond_path.exists():
            print(f"[skip] {display_name}: no {cond_path}")
            continue
        cond = pd.read_csv(cond_path)
        cmp = _load_per_country_mis(dir_short, slug)
        if cmp is None:
            print(f"[skip] {display_name}: no comparison.csv (re-run main {dir_short} first)")
            continue
        # comparison.csv format: method, country, align_mis, ...
        van = cmp[cmp["method"] == "baseline_vanilla"][["country", "align_mis"]].rename(
            columns={"align_mis": "van_mis"})
        swa = cmp[cmp["method"].str.contains("dual_pass", na=False)][["country", "align_mis"]].rename(
            columns={"align_mis": "swa_mis"})
        mis = van.merge(swa, on="country")
        mis["mis_gain"]     = mis["van_mis"] - mis["swa_mis"]
        mis["mis_gain_pct"] = 100.0 * mis["mis_gain"] / mis["van_mis"]

        merged = cond.merge(mis, on="country")
        merged["model"] = display_name
        pooled_rows.extend(merged.to_dict("records"))

        # Per-model correlation
        if len(merged) >= 5:
            r,  p  = pearsonr(merged["mean_margin"],  merged["mis_gain_pct"])
            rh, ph = spearmanr(merged["mean_margin"], merged["mis_gain_pct"])
            per_model_rows.append({
                "model":      display_name,
                "n":          len(merged),
                "pearson_r":  round(r, 3),
                "pearson_p":  round(p, 4),
                "spearman_rho": round(rh, 3),
                "spearman_p":   round(ph, 4),
            })
            print(f"  {display_name:<22s}  N={len(merged):>2d}  "
                  f"r={r:+.3f} (p={p:.3f})  rho={rh:+.3f}")

    if not pooled_rows:
        raise SystemExit("[error] No (logit_cond, comparison) pairs found.")

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT_DIR / "logit_cond_cross_model_pooled.csv", index=False)
    print(f"\n[saved] {OUT_DIR / 'logit_cond_cross_model_pooled.csv'}  (N={len(pooled)} cells)")

    # Pooled correlation
    rp, pp = pearsonr(pooled["mean_margin"],  pooled["mis_gain_pct"])
    rsp, psp = spearmanr(pooled["mean_margin"], pooled["mis_gain_pct"])
    per_model_rows.append({
        "model":      "POOLED (6 x 20)",
        "n":          len(pooled),
        "pearson_r":  round(rp, 3),
        "pearson_p":  round(pp, 4),
        "spearman_rho": round(rsp, 3),
        "spearman_p":   round(psp, 4),
    })
    summary = pd.DataFrame(per_model_rows)
    summary.to_csv(OUT_DIR / "logit_cond_cross_model_correlations.csv", index=False)
    print(f"[saved] {OUT_DIR / 'logit_cond_cross_model_correlations.csv'}")
    print("\n" + summary.to_string(index=False))

    # LaTeX table
    lines = [
        r"\begin{tabular}{lrrr}\toprule",
        r"Model & $N$ & Pearson $r$ & Spearman $\rho$ \\\midrule",
    ]
    for _, row in summary.iterrows():
        sig = "$^{*}$" if row["pearson_p"] < 0.05 else ""
        lines.append(
            f"{row['model']} & {int(row['n'])} & "
            f"{row['pearson_r']:+.3f}{sig} & {row['spearman_rho']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "logit_cond_cross_model_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {OUT_DIR / 'logit_cond_cross_model_table.tex'}")

    _zip_outputs(OUT_DIR, "round3_posthoc_logit_cond_cross_model")


def _zip_outputs(out_dir: Path, label: str) -> None:
    import shutil
    dest_base = (
        Path("/kaggle/working")
        if os.path.isdir("/kaggle/input")
        else out_dir.parent.parent / "download"
    )
    dest_base.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(dest_base / label), "zip",
        root_dir=str(out_dir.parent),
        base_dir=out_dir.name,
    )
    print(f"[zip] {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip GPU sweep, only aggregate existing logit_cond_*.csv")
    # Use parse_known_args so Jupyter/Colab kernel args (e.g. -f <json>) don't
    # abort the run. Also allow env-var override for notebook usage.
    args, _unknown = parser.parse_known_args()
    if os.environ.get("R2_AGGREGATE_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        args.aggregate_only = True

    if not args.aggregate_only:
        # Install deps ONCE for the initial backend (vLLM). Never re-install
        # per model — switching to unsloth mid-run upgrades transformers/
        # pyarrow and breaks ABI for every later load.
        install_paper_kaggle_deps()

        # Optional: run only a single model this session (R2_MODEL_INDEX=0..5)
        # or a subset (R2_MODEL_INDEX=0,2,4). Useful on Kaggle where the
        # 20 GB working disk can't hold multiple >30 GB BF16 checkpoints
        # even with cache purging between them. Default = all models.
        idx_env = os.environ.get("R2_MODEL_INDEX", "").strip()
        if idx_env:
            keep = {int(x) for x in idx_env.split(",") if x.strip().isdigit()}
            run_list = [m for i, m in enumerate(MODELS) if i in keep]
            print(f"[R2_MODEL_INDEX={idx_env}] running {[m[0] for m in run_list]}")
        else:
            run_list = list(MODELS)

        for (display_name, hf_id, dir_short, _) in run_list:
            print(f"\n{'#' * 72}\n# {display_name}  ({hf_id})\n{'#' * 72}")
            print(f"  [disk] free before load: {_free_disk_gb('/'):.1f} GB")
            try:
                _run_one_model(display_name, hf_id, dir_short)
            except Exception as exc:
                print(f"[error] {display_name}: {exc}")
                # Aggressive cleanup so a failed model doesn't starve the next:
                # drop any loaded weights from GPU, purge this model's HF cache,
                # and if disk is still tight, sweep the whole hub cache.
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                _purge_hf_cache(hf_id)
                if _free_disk_gb("/") < 10.0:
                    print("  [disk] <10 GB free after per-model purge — full sweep")
                    _purge_all_hf_cache()
                print(f"  [disk] free after cleanup: {_free_disk_gb('/'):.1f} GB")

    print(f"\n{'=' * 72}\nAggregating cross-model correlations\n{'=' * 72}")
    _aggregate()


if __name__ == "__main__":
    main()
