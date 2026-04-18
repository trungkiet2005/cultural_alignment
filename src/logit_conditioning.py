"""Per-scenario decision-logit conditioning diagnostics.

Round 2 reviewer W10 asks:

    "You attribute some failures to poorly conditioned decision logits. Can
    you provide a diagnostic metric (e.g., decision-gap entropy or margin
    statistics) and a scatter showing how improvements vary with that metric
    across countries/scenarios?"

This module computes:

    1. ``decision_gap``      := z_B - z_A  (raw vanilla logit gap)
    2. ``decision_entropy``  := H(p_A, p_B) with p = softmax((z_A, z_B))
                                 (in nats; 0 = sharp, ln2 ≈ 0.693 = uniform)
    3. ``decision_margin``   := |p_B - p_A| = |sigmoid(z_B-z_A) - sigmoid(z_A-z_B)|
                                 (∈ [0,1]; 0 = uninformative, 1 = decisive)

It is computed over the *vanilla* forward pass only (no personas, no IS), so
it reflects the intrinsic conditioning of the base model's decision head for
each scenario + country. Downstream aggregations give per-country and
per-category means.

Entry point: :func:`diagnose_country` -- same signature as
:func:`src.baseline_runner.run_baseline_vanilla`, returns per-scenario rows.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.baseline_runner import resolve_decision_tokens_for_lang
from src.constants import COUNTRY_LANG
from src.i18n import BASE_ASSISTANT_I18N, PROMPT_FRAME_I18N
from src.model import ChatTemplateHelper, gather_last_logits


def _entropy_nats(p_a: float, p_b: float) -> float:
    """Shannon entropy in nats for a 2-class distribution. Defensive against
    p_a + p_b != 1 due to numerical issues (renormalise first)."""
    s = p_a + p_b
    if s <= 0 or not math.isfinite(s):
        return 0.0
    pa, pb = p_a / s, p_b / s
    h = 0.0
    for p in (pa, pb):
        if p > 1e-12:
            h -= p * math.log(p)
    return float(h)


def diagnose_country(
    model, tokenizer, scenario_df: pd.DataFrame, country: str, cfg,
    *,
    batch_size: int = 8,
) -> dict:
    """Run one vanilla forward pass per scenario and compute conditioning stats.

    Returns a dict with:
        * ``results_df``: per-scenario columns
              [phenomenon_category, preferred_on_right, raw_logit_gap,
               p_left, p_right, decision_entropy, decision_margin]
        * ``summary``: aggregated per-category and overall means, plus a
          conditioning "score" = overall mean decision_margin (higher = better
          conditioned).
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

    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    pending: List[tuple] = []
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        user_content = frame.format(scenario=prompt)
        query_ids = chat_helper.encode_query_suffix(user_content, device)
        full_ids = torch.cat([base_ids, query_ids], dim=1)[0]
        pending.append((row.to_dict(), full_ids))

    def _pad(seqs):
        lens = torch.tensor([s.size(0) for s in seqs], dtype=torch.long, device=device)
        m = int(lens.max().item())
        out_ids = torch.full((len(seqs), m), pad_id, dtype=seqs[0].dtype, device=device)
        for i, s in enumerate(seqs):
            out_ids[i, : s.size(0)] = s
        return out_ids, lens

    rows_out: List[Dict] = []
    for start in tqdm(range(0, len(pending), batch_size),
                      desc=f"LogitCond [{country}]"):
        chunk = pending[start:start + batch_size]
        ids, lens = _pad([c[1] for c in chunk])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            batch_idx = torch.arange(ids.size(0), device=device)
            last = gather_last_logits(out, batch_idx, lens)
            za = last[:, a_id].float().cpu().numpy()
            zb = last[:, b_id].float().cpu().numpy()
            pair = torch.stack([last[:, a_id], last[:, b_id]], dim=-1)
            probs = F.softmax(pair.float(), dim=-1).cpu().numpy()
        for (row, _), z_a, z_b, p in zip(chunk, za, zb, probs):
            p_a, p_b = float(p[0]), float(p[1])
            gap = float(z_b - z_a)
            rows_out.append({
                "country":             country,
                "phenomenon_category": row.get("phenomenon_category", "Unknown"),
                "preferred_on_right":  int(row.get("preferred_on_right", 1)),
                "z_a":                 float(z_a),
                "z_b":                 float(z_b),
                "raw_logit_gap":       gap,
                "p_left":              p_a,
                "p_right":             p_b,
                "decision_entropy":    _entropy_nats(p_a, p_b),
                "decision_margin":     abs(p_b - p_a),
            })

    res_df = pd.DataFrame(rows_out)

    # Summary: per-category means + overall.
    by_cat = (res_df.groupby("phenomenon_category")
                     [["decision_entropy", "decision_margin", "raw_logit_gap"]]
                     .mean()
                     .to_dict(orient="index"))

    overall = {
        "mean_entropy":  float(res_df["decision_entropy"].mean()),
        "mean_margin":   float(res_df["decision_margin"].mean()),
        "median_margin": float(res_df["decision_margin"].median()),
        "std_margin":    float(res_df["decision_margin"].std(ddof=1))
                          if len(res_df) >= 2 else float("nan"),
        "mean_abs_gap":  float(res_df["raw_logit_gap"].abs().mean()),
        "frac_margin_lt_0.1": float((res_df["decision_margin"] < 0.1).mean()),
        "frac_margin_gt_0.5": float((res_df["decision_margin"] > 0.5).mean()),
        "by_category": by_cat,
    }
    return {"results_df": res_df, "summary": overall}
