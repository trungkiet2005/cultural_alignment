"""Vanilla LLM baseline inference via token-logit extraction."""

import gc

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm

from src.constants import COUNTRY_LANG
from src.i18n import PROMPT_FRAME_I18N
from src.model import ChatTemplateHelper
from src.amce import compute_amce_from_preferences, load_human_amce, compute_alignment_metrics


def logit_fallback_p_spare(model, full_ids, left_id, right_id, pref_right,
                           temperature=1.0, return_raw=False):
    """Extract P(spare_preferred) from LEFT/RIGHT logits with temperature sharpening."""
    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)
        logits = out.logits[0, -1, :]
        pair = torch.stack([logits[left_id], logits[right_id]])
        probs = F.softmax(pair / temperature, dim=-1)
        p_l = probs[0].item()
        p_r = probs[1].item()
    p_spare = p_r if pref_right else p_l
    if return_raw:
        return p_spare, p_l, p_r
    return p_spare


def run_baseline_vanilla(model, tokenizer, scenario_df, country, cfg):
    device = next(model.parameters()).device
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    base_ids = chat_helper.build_prefix_ids("You are a helpful assistant.", device)
    left_id = tokenizer.encode("LEFT", add_special_tokens=False)[0]
    right_id = tokenizer.encode("RIGHT", add_special_tokens=False)[0]
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])

    rows_data = []
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        user_content = frame.format(scenario=prompt)
        formatted = chat_helper.format_query_with_suffix(user_content)
        query_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([base_ids, query_ids], dim=1)
        rows_data.append((row, full_ids, bool(row.get("preferred_on_right", 1))))

    # Debug: print 3 sample prompts with vanilla prediction
    print(f"\n[DEBUG] 3 sample prompts for Vanilla {country} (lang={lang}):")
    for si in range(min(3, len(rows_data))):
        row, full_ids, pref_right = rows_data[si]
        cat = row.get("phenomenon_category", "?")
        pref_side = "RIGHT" if pref_right else "LEFT"
        p_spare, p_l, p_r = logit_fallback_p_spare(
            model, full_ids, left_id, right_id, pref_right,
            temperature=cfg.decision_temperature, return_raw=True)
        full_text = tokenizer.decode(full_ids[0], skip_special_tokens=False)
        print(f"  ── Sample {si+1} [{cat}] (preferred={pref_side}) ──")
        print(f"  [FULL LLM INPUT]\n{full_text}")
        print(f"  [END LLM INPUT]")
        print(f"  >>> p(LEFT)={p_l:.3f}  p(RIGHT)={p_r:.3f}  |  p(spare_preferred)={p_spare:.3f}  [token-logit]")
        print()

    results = []
    for i in tqdm(range(len(rows_data)), desc=f"Vanilla [{country}]"):
        row, full_ids, pref_right = rows_data[i]
        p_spare = logit_fallback_p_spare(model, full_ids, left_id, right_id, pref_right,
                                         temperature=cfg.decision_temperature)

        results.append({
            "phenomenon_category": row.get("phenomenon_category", "Unknown"),
            "this_group_name": row.get("this_group_name", "Unknown"),
            "n_left": int(row.get("n_left", 1)),
            "n_right": int(row.get("n_right", 1)),
            "preferred_on_right": int(pref_right),
            "p_spare_preferred": p_spare,
        })

    print(f"[Vanilla {country}] {len(results)} scenarios scored via token-logit")

    temp_df = pd.DataFrame(results)
    temp_df["country"] = country
    model_amce = compute_amce_from_preferences(temp_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment = compute_alignment_metrics(model_amce, human_amce)
    return {"model_amce": model_amce, "human_amce": human_amce, "alignment": alignment, "results_df": temp_df}
