#!/usr/bin/env python3
"""Oral-level baseline: PRISM-style cultural prompting (Reviewer comparison gap).

The reviewer noted that inference-time cultural prompting (e.g., "Answer as
someone from [country]") is a natural baseline that is *not* compared against
SWA-DPBR anywhere in the paper. This script fills that gap.

PRISM baseline approach (cultural framing prefix):
    We prepend a short cultural-context sentence to every scenario:
        "{country_name_native} perspective: please evaluate this scenario
         as a typical person from {country_name} would."
    The model then produces A/B logits from this culturally-prefixed prompt.
    No logit correction is applied — we measure how much the prompt alone
    shifts the model toward country-specific human AMCEs.

This tests the hypothesis: "is SWA-DPBR just doing what a good cultural
prompt already does, or does the logit-space IS correction add value?"

Expected finding: PRISM prompting gives modest MIS reduction (2-5%) while
SWA-DPBR achieves 19-24%, confirming the necessity of the IS correction.

Kaggle (~2–3 h on H100 for Phi-4 × 20 countries × 300 scenarios):
    !python exp_paper/round2/phase7_oral/exp_r2_prism_baseline.py

Env overrides:
    R2_MODEL           HF id (default: microsoft/phi-4)
    R2_COUNTRIES       comma-separated ISO3 (default: 20 paper countries)
    R2_N_SCENARIOS     per-country (default: 300)
    R2_BACKEND         vllm (default) | hf_native
    R2_PRISM_STRENGTH  'short' (default) | 'long'  -- length of cultural prefix
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap
# ─────────────────────────────────────────────────────────────────────────────
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
        raise RuntimeError(
            "Not on Kaggle and not inside the repo root. "
            "Either cd into the cultural_alignment repo first, or run on Kaggle."
        )
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))

import gc
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()

from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import pandas as pd
import torch
import torch.nn.functional as F

from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.baseline_runner import resolve_decision_tokens_for_lang
from src.constants import COUNTRY_LANG, COUNTRY_FULL_NAMES
from src.i18n import PROMPT_FRAME_I18N
from src.model import ChatTemplateHelper, gather_last_logits_one_row, setup_seeds, text_tokenizer

# ─── config ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN     = int(os.environ.get("R2_N_SCENARIOS", "300"))
COUNTRIES  = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)
STRENGTH = os.environ.get("R2_PRISM_STRENGTH", "short")

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/prism_baseline"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "prism_baseline")
)

HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    if on_kaggle()
    else "WVS_data/country_specific_ACME.csv"
)

# ─── cultural prefix templates ───────────────────────────────────────────────
# "short": one sentence (minimal cultural framing)
# "long":  three sentences (PRISM-style detailed framing)
_PRISM_PREFIX_SHORT = (
    "You are evaluating the following moral dilemma from the perspective of "
    "a typical person from {country_name}. Please choose as most people "
    "in {country_name} would choose."
)
_PRISM_PREFIX_LONG = (
    "You are taking the viewpoint of a person raised and living in "
    "{country_name}, with the cultural values, social norms, and moral "
    "priorities typical of that country. "
    "When making the judgment below, reflect the preferences that most "
    "people in {country_name} would express, not your general defaults. "
    "Choose A or B as a representative person from {country_name} would."
)


def _prism_prefix(country: str, strength: str = "short") -> str:
    name = COUNTRY_FULL_NAMES.get(country, country)
    tpl  = _PRISM_PREFIX_LONG if strength == "long" else _PRISM_PREFIX_SHORT
    return tpl.format(country_name=name)


# ─── single-scenario PRISM inference ─────────────────────────────────────────
def _prism_p_spare(
    model,
    tokenizer,
    chat_helper: ChatTemplateHelper,
    scenario_text: str,
    country: str,
    lang: str,
    a_id: int,
    b_id: int,
    strength: str = "short",
) -> float:
    """Run the PRISM-prefixed prompt and return P(spare preferred side)."""
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    cultural_prefix = _prism_prefix(country, strength)
    # Prepend cultural context to the scenario user turn.
    user_content = cultural_prefix + "\n\n" + frame.format(scenario=scenario_text)

    formatted = chat_helper.decode_query_suffix_str_for_ab_probe(user_content)
    tt = text_tokenizer(tokenizer)
    input_ids = tt.encode(formatted, add_special_tokens=False, return_tensors="pt")
    if hasattr(input_ids, "to"):
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

    with torch.no_grad():
        out    = model(input_ids=input_ids, use_cache=False)
        logits = gather_last_logits_one_row(out)
        pair   = torch.stack([logits[a_id], logits[b_id]])
        pair   = torch.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)
        probs  = F.softmax(pair, dim=-1)
        probs  = torch.nan_to_num(probs, nan=0.5)
    return float(probs[1].item())  # prob of B (= spare preferred when preferred_on_right=1)


# ─── per-country PRISM run ────────────────────────────────────────────────────
def _run_prism_country(
    model, tokenizer, cfg, country: str, scen_df: pd.DataFrame
) -> Dict:
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)

    records = []
    t0 = time.time()
    for _, row in scen_df.iterrows():
        # Column convention from src/data.py is "Prompt" (capital P);
        # earlier drafts of this script looked up scenario_text /
        # verbalized_scenario which silently returned "" and dropped every row.
        scenario_text = row.get(
            "Prompt",
            row.get("prompt",
                    row.get("scenario_text",
                            row.get("verbalized_scenario", ""))))
        if not scenario_text:
            continue
        pref_right = int(row.get("preferred_on_right", 0))
        try:
            p_b = _prism_p_spare(
                model, tokenizer, chat_helper,
                scenario_text, country, lang,
                a_id, b_id, strength=STRENGTH,
            )
            # P(spare preferred) = P(B) if preferred is on right, else 1-P(B).
            p_spare = p_b if pref_right == 1 else (1.0 - p_b)
        except Exception as exc:
            print(f"  [WARN] {country} row {row.name}: {exc}")
            p_spare = 0.5

        rec = {
            "country":              country,
            "phenomenon_category":  row.get("phenomenon_category", ""),
            "preferred_on_right":   pref_right,
            "p_spare_preferred":    p_spare,
        }
        for col in ("n_left", "n_right"):
            if col in row.index:
                rec[col] = row[col]
        records.append(rec)

    elapsed = time.time() - t0
    res_df = pd.DataFrame(records)

    # Alignment metrics
    model_amce = compute_amce_from_preferences(res_df)
    human_amce = load_human_amce(HUMAN_AMCE_PATH, country)
    alignment  = compute_alignment_metrics(model_amce, human_amce)

    out_path = Path(RESULTS_BASE) / f"prism_results_{country}.csv"
    res_df.to_csv(out_path, index=False)

    return {
        "country":     country,
        "n_scenarios": len(records),
        "elapsed_sec": elapsed,
        "mis":         alignment.get("mis", float("nan")),
        "jsd":         alignment.get("jsd", float("nan")),
        "pearson_r":   alignment.get("pearson_r", float("nan")),
        "prism_strength": STRENGTH,
    }


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    scen_cache: Dict[str, pd.DataFrame] = {}
    for c in COUNTRIES:
        scen_cache[c] = load_scenarios(cfg, c)

    rows = []
    for country in COUNTRIES:
        scen = scen_cache.get(country)
        if scen is None or scen.empty:
            print(f"[SKIP] {country} — no scenarios")
            continue
        print(f"\n{'#'*70}\n# PRISM [{STRENGTH}] — {country}  (n={len(scen)})\n{'#'*70}")
        try:
            row = _run_prism_country(model, tokenizer, cfg, country, scen)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "prism_partial.csv", index=False)
            print(f"  ✓ {country}  MIS={row['mis']:.4f}  r={row['pearson_r']:.3f}  "
                  f"({row['elapsed_sec']:.0f}s)")
        except Exception as exc:
            print(f"[ERROR] {country}: {exc}")
            rows.append({"country": country, "error": str(exc)[:500]})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary(rows, out_dir, "prism_summary.csv")
    _zip_outputs(out_dir, "round2_phase7_prism_baseline")


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
    print(f"[ZIP] {zip_path}")


if __name__ == "__main__":
    main()
