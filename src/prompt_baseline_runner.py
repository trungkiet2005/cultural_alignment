"""Runner for prompt-only baselines (B1, B2, B3-short, B3-long, B4).

This is a thin wrapper around the vanilla ``baseline_runner`` that
prepends a cultural-framing prefix (see ``src/prompt_baselines.py``) to
every scenario, then runs the same A/B logit-gap inference.  Because all
prompt baselines share the vanilla decision pipeline, we expose a single
``run_prompt_baseline_country`` entry point that dispatches by name.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import pandas as pd
import torch

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.baseline_runner import (
    BASE_ASSISTANT_I18N,
    PROMPT_FRAME_I18N,
    logit_fallback_p_spare,
    resolve_decision_tokens_for_lang,
)
from src.constants import COUNTRY_LANG
from src.model import ChatTemplateHelper, gather_last_logits_one_row, text_tokenizer
from src.prompt_baselines import wrap_scenario, load_wvs_vector


def _run_one_scenario(model, tokenizer, chat_helper, base_ids, a_id, b_id,
                      country: str, baseline: str, wvs_vec: Optional[Dict[str, float]],
                      scenario_prompt: str, decision_temp: float) -> float:
    """Return p(spare-preferred) for one scenario under the chosen baseline."""
    user_content = wrap_scenario(scenario_prompt, country, baseline, wvs_vec=wvs_vec)
    device = next(model.parameters()).device
    query_ids = chat_helper.encode_query_suffix(user_content, device)
    full_ids = torch.cat([base_ids, query_ids], dim=1)
    with torch.no_grad():
        # Reuse the same logit_fallback the vanilla runner uses so we share
        # numerical conventions (NaN guard, decision temperature).
        p_spare = logit_fallback_p_spare(
            model, full_ids, a_id, b_id,
            preferred_on_right=True,  # caller corrects below
            temperature=decision_temp,
        )
    return float(p_spare)


def run_prompt_baseline_country(
    model, tokenizer, scenario_df: pd.DataFrame, country: str, cfg,
    baseline: str, wvs_csv_path: Optional[str] = None,
    human_amce_path: Optional[str] = None,
) -> Dict:
    """Evaluate one prompt-only baseline (B1 / B2 / B3_short / B3_long / B4) on
    the per-country scenario slice.

    Returns a single-row summary dict identical in shape to the vanilla
    summary so it slots into existing aggregator scripts.
    """
    device = next(model.parameters()).device
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)
    if hasattr(model, "set_decision_tokens"):
        model.set_decision_tokens(int(a_id), int(b_id))
    setattr(tokenizer, "_moral_vllm_ab", (int(a_id), int(b_id)))

    wvs_vec = None
    if wvs_csv_path and baseline in ("B2", "B4"):
        wvs_vec = load_wvs_vector(wvs_csv_path, country)
        if wvs_vec is None:
            print(f"  [warn] {country}: no WVS vector → {baseline} degrades to B1")

    decision_temp = float(getattr(cfg, "decision_temperature", 0.5))
    records = []
    t0 = time.time()
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        pref_right = int(row.get("preferred_on_right", 1))
        try:
            p_b = _run_one_scenario(
                model, tokenizer, chat_helper, base_ids, a_id, b_id,
                country, baseline, wvs_vec, prompt, decision_temp,
            )
            p_spare = p_b if pref_right == 1 else (1.0 - p_b)
        except Exception as exc:
            print(f"  [warn] {country} row {row.name}: {exc}")
            p_spare = 0.5
        rec = {
            "country":             country,
            "baseline":            baseline,
            "phenomenon_category": row.get("phenomenon_category", ""),
            "preferred_on_right":  pref_right,
            "p_spare_preferred":   p_spare,
        }
        for col in ("n_left", "n_right"):
            if col in row.index:
                rec[col] = row[col]
        records.append(rec)
    elapsed = time.time() - t0
    res_df = pd.DataFrame(records)

    summary = {
        "method":      f"prompt_{baseline}",
        "country":     country,
        "n_scenarios": len(records),
        "elapsed_sec": elapsed,
    }
    if human_amce_path:
        try:
            model_amce = compute_amce_from_preferences(res_df)
            human_amce = load_human_amce(human_amce_path, country)
            alignment = compute_alignment_metrics(model_amce, human_amce)
            summary["mis"]       = alignment.get("mis", float("nan"))
            summary["jsd"]       = alignment.get("jsd", float("nan"))
            summary["pearson_r"] = alignment.get("pearson_r", float("nan"))
        except Exception as exc:
            print(f"  [warn] alignment metrics failed for {country}: {exc}")
    return {"summary": summary, "results_df": res_df}
