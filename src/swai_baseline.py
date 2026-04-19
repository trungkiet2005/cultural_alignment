"""SWAI-style logit-bias baseline (binary-decision adaptation).

Following the spirit of "Steering Language Models Before They Speak:
Logit-Level Interventions" (An et al., 2026 / arXiv:2601.10960), this
baseline estimates a *static* per-country token-bias from a small WVS
profile corpus and adds it to the decision-token logits at inference:

        z'_a = z_a + b_a(country)
        z'_b = z_b + b_b(country)
        b_a(c) = log E_corpus[ p(A | s, c) ]
        b_b(c) = log E_corpus[ p(B | s, c) ]

where ``s`` ranges over the country's WVS profile sentences (the same
ones SWA-DPBR feeds to its persona panel).

The result is a *constant* logit shift per country -- the strict
contrast to SWA-DPBR's *per-scenario* persona-disagreement signal. The
baseline thus isolates "what a static, corpus-informed logit bias can
buy you" without any scenario-specific adaptation.

Hard requirement: any backend (vLLM works, since we only need a single
forward pass per profile sentence; no hooks).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

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
from src.constants import COUNTRY_FULL_NAMES, COUNTRY_LANG
from src.model import ChatTemplateHelper, gather_last_logits_one_row
from src.prompt_baselines import _quartile_descriptor, load_wvs_vector


def _build_corpus_sentences(country: str,
                            wvs_vec: Optional[Dict[str, float]] = None
                            ) -> List[str]:
    """Generate a small WVS-grounded corpus for ``country``.

    Each sentence is a single-attribute statement matching the country's
    WVS pole on one dimension.  This mirrors the persona descriptors but
    one-attribute-per-sentence (so the bias estimate is a finer-grained
    average over independent statements).
    """
    name = COUNTRY_FULL_NAMES.get(country, country)
    if not wvs_vec:
        # Neutral fallback corpus: still country-conditioned but no WVS pole.
        return [f"This statement is from {name}."]
    feats = ["religiosity", "moral_acceptability", "social_trust",
             "national_pride", "gender_equality", "tolerance_diversity",
             "child_rearing", "happiness", "materialism_orientation",
             "political_participation"]
    sentences = []
    for feat in feats:
        if feat not in wvs_vec or wvs_vec[feat] != wvs_vec[feat]:
            continue
        descriptor = _quartile_descriptor(feat, wvs_vec[feat])
        sentences.append(
            f"In {name}, the typical adult is {descriptor}."
        )
    if not sentences:
        sentences = [f"This statement is from {name}."]
    return sentences


@torch.no_grad()
def estimate_token_bias(model, tokenizer, country: str, lang: str,
                        wvs_csv_path: Optional[str] = None) -> Dict[str, float]:
    """Estimate per-country (b_a, b_b) bias from the WVS profile corpus.

    Runs a single forward pass per profile sentence, gathers the
    last-position logits at the A and B decision tokens, averages
    log-probabilities over the corpus, and returns the gap.
    """
    device = next(model.parameters()).device
    chat_helper = ChatTemplateHelper(tokenizer)
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)

    wvs_vec = load_wvs_vector(wvs_csv_path, country) if wvs_csv_path else None
    sentences = _build_corpus_sentences(country, wvs_vec=wvs_vec)
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)

    log_p_a, log_p_b = [], []
    for s in sentences:
        # Wrap the corpus sentence as if it were a scenario, asking for A/B.
        frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
        user = frame.format(scenario=s)
        query_ids = chat_helper.encode_query_suffix(user, device)
        full_ids = torch.cat([base_ids, query_ids], dim=1)
        out = model(input_ids=full_ids, use_cache=False)
        logits = gather_last_logits_one_row(out)
        pair = torch.stack([logits[a_id], logits[b_id]])
        pair = torch.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)
        logp = F.log_softmax(pair, dim=-1)
        log_p_a.append(float(logp[0].item()))
        log_p_b.append(float(logp[1].item()))

    if not log_p_a:
        return {"b_a": 0.0, "b_b": 0.0, "n_sentences": 0}
    mean_a = sum(log_p_a) / len(log_p_a)
    mean_b = sum(log_p_b) / len(log_p_b)
    # Centre to zero-mean: only the gap matters at decision time.
    return {
        "b_a":         mean_a - 0.5 * (mean_a + mean_b),
        "b_b":         mean_b - 0.5 * (mean_a + mean_b),
        "n_sentences": len(log_p_a),
    }


def run_swai_country(model, tokenizer, scenario_df: pd.DataFrame,
                     country: str, cfg,
                     wvs_csv_path: Optional[str] = None,
                     human_amce_path: Optional[str] = None) -> Dict:
    """Apply a static per-country logit bias and run the binary baseline."""
    device = next(model.parameters()).device
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)
    if hasattr(model, "set_decision_tokens"):
        model.set_decision_tokens(int(a_id), int(b_id))
    setattr(tokenizer, "_moral_vllm_ab", (int(a_id), int(b_id)))

    print(f"  [swai] {country}: estimating token bias from WVS corpus …")
    bias = estimate_token_bias(model, tokenizer, country, lang,
                               wvs_csv_path=wvs_csv_path)
    print(f"  [swai] {country}: b_a={bias['b_a']:+.3f}, b_b={bias['b_b']:+.3f} "
          f"(n_sentences={bias['n_sentences']})")

    decision_temp = float(getattr(cfg, "decision_temperature", 0.5))
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])

    records: List[Dict] = []
    t0 = time.time()
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        pref_right = int(row.get("preferred_on_right", 1))
        try:
            user = frame.format(scenario=prompt)
            query_ids = chat_helper.encode_query_suffix(user, device)
            full_ids = torch.cat([base_ids, query_ids], dim=1)
            with torch.no_grad():
                out = model(input_ids=full_ids, use_cache=False)
                logits = gather_last_logits_one_row(out)
                # Apply static bias to A and B logits.
                z_a = float(logits[a_id].item()) + bias["b_a"]
                z_b = float(logits[b_id].item()) + bias["b_b"]
                # Binary softmax with decision temperature.
                gap = (z_b - z_a) / max(decision_temp, 1e-8)
                p_b = float(torch.sigmoid(torch.tensor(gap)).item())
            p_spare = p_b if pref_right == 1 else (1.0 - p_b)
        except Exception as exc:
            print(f"  [warn] {country} row {row.name}: {exc}")
            p_spare = 0.5
        rec = {
            "country":             country,
            "method":              "swai",
            "phenomenon_category": row.get("phenomenon_category", ""),
            "preferred_on_right":  pref_right,
            "p_spare_preferred":   p_spare,
            "bias_a":              bias["b_a"],
            "bias_b":              bias["b_b"],
        }
        for col in ("n_left", "n_right"):
            if col in row.index:
                rec[col] = row[col]
        records.append(rec)
    elapsed = time.time() - t0
    res_df = pd.DataFrame(records)

    summary = {
        "method":      "swai",
        "country":     country,
        "n_scenarios": len(records),
        "elapsed_sec": elapsed,
        "bias_a":      bias["b_a"],
        "bias_b":      bias["b_b"],
        "n_corpus_sentences": bias["n_sentences"],
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
