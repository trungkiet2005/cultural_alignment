"""Vanilla LLM baseline inference via token-logit extraction."""

import gc

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm

from src.constants import COUNTRY_LANG
from src.i18n import BASE_ASSISTANT_I18N, PROMPT_FRAME_I18N
from src.model import (
    ChatTemplateHelper,
    encode_text_to_tensor,
    gather_last_logits,
    gather_last_logits_one_row,
    text_tokenizer,
)
from src.amce import compute_amce_from_preferences, load_human_amce, compute_alignment_metrics


def resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang):
    """Resolve the (A_token_id, B_token_id) the model would emit at the answer
    position for `lang`'s prompt template.

    See `controller._resolve_decision_tokens_for_lang` for the rationale —
    BPE tokenizers (Llama 3, Qwen 2.5, GPT-OSS) merge the answer letter with
    the preceding character, so the naive `tokenizer.encode("A")[0]` returns
    the wrong token id for any language whose prompt does not end with a CJK
    fullwidth character. We MUST resolve this per-language using the actual
    chat-templated formatted prompt.
    """
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    user_content = frame.format(scenario="DUMMY_SCENARIO_FOR_TOKEN_RESOLUTION")
    formatted = chat_helper.format_query_with_suffix(user_content)

    tt = text_tokenizer(tokenizer)
    base_ids = tt.encode(formatted, add_special_tokens=False)
    a_full = tt.encode(formatted + "A", add_special_tokens=False)
    b_full = tt.encode(formatted + "B", add_special_tokens=False)

    def _first_diff(base, full):
        n = min(len(base), len(full))
        for i in range(n):
            if base[i] != full[i]:
                return i
        return n

    a_pos = _first_diff(base_ids, a_full)
    b_pos = _first_diff(base_ids, b_full)
    if a_pos >= len(a_full) or b_pos >= len(b_full):
        raise RuntimeError(
            f"Failed to resolve A/B answer tokens for lang={lang}: "
            f"prompt+'A'/'B' produced no new token."
        )
    a_id, b_id = a_full[a_pos], b_full[b_pos]
    a_str = tt.decode([a_id])
    b_str = tt.decode([b_id])
    print(f"[Baseline] Decision tokens for lang={lang}: "
          f"A={a_id}({a_str!r})  B={b_id}({b_str!r})")
    return a_id, b_id


def logit_fallback_p_spare(model, full_ids, a_id, b_id, pref_right,
                           temperature=1.0, return_raw=False):
    """Extract P(spare_preferred) from A/B option logits with temperature sharpening.

    `a_id` / `b_id` are language-specific answer-token ids resolved via
    `resolve_decision_tokens_for_lang`.
    """
    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)
        logits = gather_last_logits_one_row(out)
        pair = torch.stack([logits[a_id], logits[b_id]])
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
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)
    # Per-language A/B answer tokens (CRITICAL: do NOT use naive
    # `tokenizer.encode("A")[0]` — see resolve_decision_tokens_for_lang).
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)
    if hasattr(model, "set_decision_tokens"):
        model.set_decision_tokens(int(a_id), int(b_id))
    # vLLM shim (``VllmCausalLogitModel`` / ``MORAL_MODEL_BACKEND=vllm``) reads this before forward.
    setattr(tokenizer, "_moral_vllm_ab", (int(a_id), int(b_id)))
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])

    rows_data = []
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        user_content = frame.format(scenario=prompt)
        formatted = chat_helper.format_query_with_suffix(user_content)
        query_ids = encode_text_to_tensor(
            tokenizer, formatted, device, add_special_tokens=False
        )
        full_ids = torch.cat([base_ids, query_ids], dim=1)
        rows_data.append((row, full_ids, bool(row.get("preferred_on_right", 1))))

    # Debug: print 3 sample prompts with vanilla prediction
    print(f"\n[DEBUG] 3 sample prompts for Vanilla {country} (lang={lang}):")
    for si in range(min(3, len(rows_data))):
        row, full_ids, pref_right = rows_data[si]
        cat = row.get("phenomenon_category", "?")
        pref_side = "B" if pref_right else "A"
        p_spare, p_l, p_r = logit_fallback_p_spare(
            model, full_ids, a_id, b_id, pref_right,
            temperature=cfg.decision_temperature, return_raw=True)
        full_text = text_tokenizer(tokenizer).decode(
            full_ids[0], skip_special_tokens=False
        )
        print(f"  ── Sample {si+1} [{cat}] (preferred={pref_side}) ──")
        print(f"  [FULL LLM INPUT]\n{full_text}")
        print(f"  [END LLM INPUT]")
        print(f"  >>> p(A)={p_l:.3f}  p(B)={p_r:.3f}  |  p(spare_preferred)={p_spare:.3f}  [token-logit]")
        print()

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _pad_batch(seqs):
        """Right-pad batch, return (ids, lengths). NO attention_mask.

        Unsloth's patched attention for Gemma2 / GPT-OSS mishandles 2D masks
        under Transformers >=5.2. With causal attention + right-pad, logits at
        the last real token position depend only on real tokens, so no mask
        is required — we gather at `lengths[i] - 1` instead of `-1`.
        """
        lens = torch.tensor([s.size(0) for s in seqs],
                            dtype=torch.long, device=device)
        max_len = int(lens.max().item())
        ids = torch.full((len(seqs), max_len), pad_id,
                         dtype=seqs[0].dtype, device=device)
        for j, s in enumerate(seqs):
            ids[j, : s.size(0)] = s
        return ids, lens

    def _auto_batch_size():
        """Empirically pick the largest batch that fits in 80% of free VRAM."""
        if not torch.cuda.is_available():
            return 8
        # Sort by length descending so probe uses worst-case (longest) prompts
        sorted_seqs = sorted([t[1][0] for t in rows_data],
                             key=lambda s: -s.size(0))
        free_before, total = torch.cuda.mem_get_info(device)
        budget = int(0.80 * free_before)
        bs = 1
        last_ok = 1
        while bs <= min(1024, len(sorted_seqs)):
            try:
                torch.cuda.empty_cache()
                ids, _lens = _pad_batch(sorted_seqs[:bs])
                with torch.no_grad():
                    _ = model(input_ids=ids, use_cache=False)
                free_after, _ = torch.cuda.mem_get_info(device)
                used = free_before - free_after
                if used > budget:
                    break
                last_ok = bs
                # Stop early if we already use >40% — doubling would risk OOM
                if used > 0.65 * free_before:
                    break
                bs *= 2
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        torch.cuda.empty_cache()
        # Back off one step for safety margin on shorter batches that pad less
        return max(1, last_ok)

    requested = int(getattr(cfg, "batch_size", 32))
    if requested <= 0:
        batch_size = _auto_batch_size()
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9 if torch.cuda.is_available() else 0
        print(f"[AUTO-BATCH] Picked batch_size={batch_size} "
              f"(free VRAM {free_gb:.1f} GB)")
    else:
        batch_size = requested

    results = []
    temperature = cfg.decision_temperature
    for start in tqdm(range(0, len(rows_data), batch_size),
                      desc=f"Vanilla [{country}]"):
        chunk = rows_data[start:start + batch_size]
        seqs = [t[1][0] for t in chunk]                 # 1D tensors
        # Right-pad and gather at the last real token position for each row.
        input_ids, lens = _pad_batch(seqs)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            last = gather_last_logits(out, batch_idx, lens)
            pair = torch.stack([last[:, a_id], last[:, b_id]], dim=-1)
            probs = F.softmax(pair / temperature, dim=-1).float().cpu().numpy()

        for j, (row, _ids, pref_right) in enumerate(chunk):
            p_l, p_r = float(probs[j, 0]), float(probs[j, 1])
            p_spare = p_r if pref_right else p_l
            results.append({
                "phenomenon_category": row.get("phenomenon_category", "Unknown"),
                "this_group_name": row.get("this_group_name", "Unknown"),
                "n_left": int(row.get("n_left", 1)),
                "n_right": int(row.get("n_right", 1)),
                "preferred_on_right": int(pref_right),
                "p_left": p_l,
                "p_right": p_r,
                "p_spare_preferred": p_spare,
                "Prompt": row.get("Prompt", row.get("prompt", "")),
            })

    print(f"[Vanilla {country}] {len(results)} scenarios scored via token-logit")

    temp_df = pd.DataFrame(results)
    temp_df["country"] = country
    model_amce = compute_amce_from_preferences(temp_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment = compute_alignment_metrics(model_amce, human_amce)
    return {"model_amce": model_amce, "human_amce": human_amce, "alignment": alignment, "results_df": temp_df}
