"""MC-Dropout inference-time calibration baseline.

Motivation (Round 2 reviewer W1a):
----------------------------------
Dropout calibration inflates *generic* epistemic uncertainty on top of the
frozen model's logits by running a small number of stochastic forward passes
with dropout enabled at inference time, then averaging the resulting A/B
probabilities. Unlike SWA-DPBR, it has no country-conditioned signal --- it
simply damps overconfident decisions toward uniform.

What this baseline is (and is not):
  * It *is* a direct head-to-head test of "does making the model less confident
    help cross-cultural alignment?" Under the MIS metric this predicts a
    smoothing of AMCE toward 50%, which can help countries whose human AMCE is
    near the center and hurt countries with sharp directional preferences.
  * It is *not* a country-conditioned method. The same dropout samples are used
    for every country, so it cannot explain between-country variation in human
    preferences.

Implementation:
  * Enable .train()-mode dropout on all transformer dropout modules
    (attention, residual, mlp). KV cache is disabled to force re-computation.
  * Run T Monte-Carlo passes; average probabilities.
  * Works with the HF-native backend (vLLM does not expose dropout hooks).

Exposed entry point:
    :func:`run_baseline_mc_dropout` with the same signature as
    :func:`src.baseline_runner.run_baseline_vanilla`.

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation" (ICML).
Cross-cultural adaptation follows the "moral uncertainty inflation" template
discussed in the Round 2 review (Kwon et al. 2025 style).
"""

from __future__ import annotations

import gc
import math
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.baseline_runner import resolve_decision_tokens_for_lang
from src.constants import COUNTRY_LANG
from src.i18n import BASE_ASSISTANT_I18N, PROMPT_FRAME_I18N
from src.model import ChatTemplateHelper, gather_last_logits


def _enable_dropout(model: torch.nn.Module, rate: Optional[float] = None) -> int:
    """Switch every Dropout* module into training mode so it samples at inference.

    If ``rate`` is given, overwrite each module's ``p`` --- useful because some
    HF checkpoints ship with p=0 for distillation-era configs.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            if rate is not None:
                m.p = float(rate)
            n += 1
    return n


def _disable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()


def run_baseline_mc_dropout(
    model,
    tokenizer,
    scenario_df: pd.DataFrame,
    country: str,
    cfg,
    *,
    T: int = 8,
    dropout_p: Optional[float] = 0.1,
) -> dict:
    """Run MC-Dropout calibration baseline for a single country.

    Returns the same-shaped dict as :func:`run_baseline_vanilla`.
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
    temperature = max(float(cfg.decision_temperature), 1e-10)

    model.eval()
    n_dropout = _enable_dropout(model, rate=dropout_p)
    if n_dropout == 0:
        print(f"[MC-Dropout] WARN: no Dropout modules in model; "
              f"behaves identically to vanilla (T={T} is wasted).")
    else:
        print(f"[MC-Dropout] enabled {n_dropout} dropout modules at p={dropout_p}")
        print(f"[MC-Dropout] running T={T} stochastic passes per scenario")

    results = []
    try:
        pbar = tqdm(scenario_df.iterrows(), total=len(scenario_df),
                    desc=f"MC-Dropout [{country}] T={T}")
        for _, row in pbar:
            prompt = row.get("Prompt", row.get("prompt", ""))
            if not prompt:
                continue
            user_content = frame.format(scenario=prompt)
            query_ids = chat_helper.encode_query_suffix(user_content, device)
            full_ids = torch.cat([base_ids, query_ids], dim=1)

            p_accum = torch.zeros(2, device=device, dtype=torch.float32)
            with torch.no_grad():
                for _t in range(T):
                    out = model(input_ids=full_ids, use_cache=False)
                    logits = gather_last_logits(
                        out,
                        torch.tensor([0], device=device),
                        torch.tensor([full_ids.shape[1]], device=device),
                    )[0]
                    pair = torch.stack([logits[a_id], logits[b_id]])
                    pair = torch.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)
                    probs = F.softmax(pair / temperature, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                    s = float(probs.sum().item())
                    if s <= 0 or math.isnan(s):
                        probs = torch.tensor([0.5, 0.5], device=device, dtype=probs.dtype)
                    else:
                        probs = probs / probs.sum()
                    p_accum = p_accum + probs.float()

            p_mean = (p_accum / max(1, T)).cpu().numpy()
            p_l, p_r = float(p_mean[0]), float(p_mean[1])
            pref_right = bool(row.get("preferred_on_right", 1))
            p_spare = p_r if pref_right else p_l

            results.append({
                "phenomenon_category": row.get("phenomenon_category", "Unknown"),
                "this_group_name":     row.get("this_group_name",     "Unknown"),
                "n_left":  int(row.get("n_left", 1)),
                "n_right": int(row.get("n_right", 1)),
                "preferred_on_right": int(pref_right),
                "p_left":  p_l,
                "p_right": p_r,
                "p_spare_preferred": p_spare,
                "Prompt":  prompt,
                "mc_T":    T,
                "mc_dropout_p": float(dropout_p) if dropout_p is not None else float("nan"),
            })
    finally:
        _disable_dropout(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[MC-Dropout {country}] {len(results)} scenarios scored")
    temp_df = pd.DataFrame(results)
    temp_df["country"] = country
    model_amce = compute_amce_from_preferences(temp_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment = compute_alignment_metrics(model_amce, human_amce)
    return {"model_amce": model_amce, "human_amce": human_amce,
            "alignment": alignment, "results_df": temp_df}
