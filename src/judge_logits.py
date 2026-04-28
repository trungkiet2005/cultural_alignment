"""Continuous-δ extractor from judge logits — replaces pseudo-δ parsing.

Standard open-ended pipeline runs the judge in *generation* mode and parses the
emitted JSON into a quantized ``{choice, confidence}`` pair, then maps to a
pseudo-δ. That bottleneck collapses the continuous signal PT-IS expects (see
SWA-PTIS in src/controller.py:_is_solve_decision) into 3-5 discrete values.

This module bypasses generation entirely: forward the judge ONE step on a
"decisive" closing prompt and read the next-token logits at the A/B token
positions. The continuous δ = logit(B) − logit(A) restores the smooth signal
that PT-IS softmax weighting was designed for.

Usage:
    a_id, b_id = resolve_ab_token_ids(tokenizer)
    delta = judge_logit_delta(
        model, tokenizer, scenario_en, actor_text, a_id, b_id, device
    )
    # delta is real-valued; sigmoid(delta / T_DECISION) = p(B|context)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


JUDGE_DECISIVE_SYSTEM = "You are a careful evaluator."

JUDGE_DECISIVE_USER_TEMPLATE = (
    "A participant answered a moral dilemma.\n\n"
    "DILEMMA (English canonical form):\n{scenario_en}\n\n"
    "PARTICIPANT RESPONSE (may be in any language, may include reasoning):\n"
    "{actor_text}\n\n"
    "Identify which option (A or B) the participant ultimately chose."
)

# Priming string appended INSIDE the assistant turn, after add_generation_prompt.
# This forces the next-token position to be the immediate continuation of
# "...chose option ", which is overwhelmingly likely to be " A" or " B" — so
# logit(B) − logit(A) at this position carries genuine preference signal
# instead of being dominated by generic continuation tokens like "The", "I",
# or "Based on...". Without this priming, the model's first generated token is
# unconstrained and the A/B logit gap is meaningless.
JUDGE_PRIMING = "The participant chose option"


def _try_token_id(tokenizer, candidate: str) -> Optional[int]:
    """Encode a candidate string; return the token id only if it tokenizes to ONE id."""
    try:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
    except Exception:
        return None
    return int(ids[0]) if len(ids) == 1 else None


def resolve_ab_token_ids(tokenizer) -> Tuple[int, int]:
    """Find a stable single-token id for ' A' and ' B' in the judge tokenizer.

    The judge prompt ends with priming "The participant chose option " (note
    trailing space implicitly from BPE re-tokenization). After this, the
    natural continuation is " A" or " B" with leading space, OR "A" / "B"
    if BPE merges the space into the priming. Tries both forms and prefers
    the one that resolves to a single id.
    """
    # Order matters. For actor's first-token logit (after chat template's
    # add_generation_prompt), Qwen models typically emit "A" / "B" with NO
    # leading space — the assistant header already ended with '\n'. We try
    # the no-space form first, then leading-space variants as fallbacks.
    a_candidates = ("A", " A", "ĠA", "a", " a")
    b_candidates = ("B", " B", "ĠB", "b", " b")

    a_id: Optional[int] = None
    a_form: str = ""
    for c in a_candidates:
        tid = _try_token_id(tokenizer, c)
        if tid is not None:
            a_id, a_form = tid, c
            break

    b_id: Optional[int] = None
    b_form: str = ""
    for c in b_candidates:
        tid = _try_token_id(tokenizer, c)
        if tid is not None:
            b_id, b_form = tid, c
            break

    if a_id is None:
        a_id = int(tokenizer.encode(" A", add_special_tokens=False)[0])
        a_form = " A (multi-token fallback first)"
    if b_id is None:
        b_id = int(tokenizer.encode(" B", add_special_tokens=False)[0])
        b_form = " B (multi-token fallback first)"

    if a_id == b_id:
        raise RuntimeError(
            f"Resolved A and B to same token id={a_id} — judge cannot distinguish."
        )
    print(f"[judge_logits] A token id={a_id} (form={a_form!r})  "
          f"B token id={b_id} (form={b_form!r})")
    return a_id, b_id


def _build_judge_input_ids(
    tokenizer, scenario_en: str, actor_text: str, device: torch.device,
) -> torch.Tensor:
    """Build judge input ids = chat-template(messages) + priming string.

    The priming "The participant chose option" is appended as the START of the
    assistant's response (after add_generation_prompt). The model sees this as
    its own partial response and the next-token distribution is heavily
    concentrated on " A" / " B" — making the logit gap at that position a
    meaningful preference signal.
    """
    user = JUDGE_DECISIVE_USER_TEMPLATE.format(
        scenario_en=scenario_en,
        actor_text=actor_text if len(actor_text) <= 4000 else actor_text[:4000] + "\n[...]",
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": JUDGE_DECISIVE_SYSTEM},
            {"role": "user", "content": user},
        ]
        templated = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        if isinstance(templated, torch.Tensor):
            base_ids = templated.to(device)
        else:
            base_ids = templated["input_ids"].to(device)
    else:
        prompt = f"{JUDGE_DECISIVE_SYSTEM}\n\n{user}\n"
        base_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Append priming so next-token logits resolve over " A" / " B"
    priming_ids = tokenizer.encode(
        JUDGE_PRIMING, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    return torch.cat([base_ids, priming_ids], dim=-1)


@torch.no_grad()
def actor_logit_delta(
    model,
    tokenizer,
    helper,
    persona_text: str,
    user_content: str,
    a_token_id: int,
    b_token_id: int,
    device: torch.device,
    *,
    max_new_tokens: int = 8,
    delta_clip: float = 8.0,
) -> Tuple[str, int, float, float]:
    """Generate actor text AND capture first-token logit gap in one pass.

    Returns:
        text: decoded actor response (free-form, 8 tokens)
        n_new_tokens: number of generated tokens
        seconds: wall time
        delta: clipped logit(B) − logit(A) at the FIRST generated position.
               This is the actor's decision logit, conditioned on persona.

    Why use actor logits instead of judge logits for the open-ended track:
    in 8-token actor mode all 5 personas typically emit the same letter
    (just "A" or "B"). The judge's logit gap on these similar texts is also
    similar — DPBR has no leverage. The actor's OWN first-token logit gap,
    however, is directly persona-conditioned (different system prompts produce
    different decision logits) and gives DPBR the per-persona variance it
    needs. Same continuous signal regime as the logit-track pipeline that
    achieves 19-24% MIS on the moral-machine track.
    """
    prefix_ids = helper.build_prefix_ids(persona_text, device)
    query_ids = helper.encode_query_suffix(user_content, device)
    input_ids = torch.cat([prefix_ids, query_ids], dim=1)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    import time as _time
    t0 = _time.time()
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=pad_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    elapsed = _time.time() - t0
    sequences = out.sequences
    new_ids = sequences[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Logit gap at the FIRST generated position (the decision moment).
    # out.scores is a tuple of length max_new_tokens; scores[0] has shape (1, vocab).
    if out.scores is None or len(out.scores) == 0:
        raise RuntimeError("model.generate did not return scores; cannot extract logit gap")
    first_logits = out.scores[0][0]  # (vocab_size,)
    raw_delta = float(first_logits[b_token_id].item() - first_logits[a_token_id].item())
    delta = float(max(-delta_clip, min(delta_clip, raw_delta)))
    return text, int(new_ids.shape[0]), elapsed, delta


@torch.no_grad()
def judge_logit_delta(
    model,
    tokenizer,
    scenario_en: str,
    actor_text: str,
    a_token_id: int,
    b_token_id: int,
    device: torch.device,
    *,
    delta_clip: float = 8.0,
) -> Tuple[float, str, float]:
    """Forward the judge ONE step and return (delta, choice, confidence).

    delta = logit(B) − logit(A) at the next-token position. Continuous, signed,
    on the same scale as raw model logit gaps. Clipped to ±delta_clip to stay
    within the empirical pseudo-δ range (~±4.6 at conf=0.99).

    Returns:
        delta: continuous logit gap (B − A). Positive ⇒ judge thinks B.
        choice: "B" if delta > 0 else "A" if delta < 0 else "UNCERTAIN".
        confidence: sigmoid(|delta|), the implied probability of the chosen side.
    """
    input_ids = _build_judge_input_ids(tokenizer, scenario_en, actor_text, device)
    outputs = model(input_ids=input_ids, use_cache=False)
    next_logits = outputs.logits[0, -1, :]
    raw_delta = float(next_logits[b_token_id].item() - next_logits[a_token_id].item())
    delta = float(max(-delta_clip, min(delta_clip, raw_delta)))

    if delta > 0.0:
        choice = "B"
        conf = float(1.0 / (1.0 + math.exp(-delta)))
    elif delta < 0.0:
        choice = "A"
        conf = float(1.0 / (1.0 + math.exp(delta)))
    else:
        choice = "UNCERTAIN"
        conf = 0.5
    return delta, choice, conf
