"""Judge-output -> pseudo-delta mapping for the open-ended SWA-DPBR variant.

The logit-based SWA-DPBR pipeline operates on scalar decision gaps
    delta = logit(P(B)) - logit(P(A))
produced by an actor forward pass. In the open-ended variant the actor emits
free-form reasoning text and a separate judge LLM classifies that text into
``{A, B, UNCERTAIN}`` with a confidence score. This module converts the judge's
output into a scalar that is on the SAME SCALE as a genuine logit gap, so
``Exp24DualPassController``'s IS / bootstrap / prior math can run unchanged.

Round-trip invariant:
    sigmoid(pseudo_delta_from_judge(choice, conf) / T_DECISION) ==
        (clipped confidence if choice == "B" else 1 - clipped confidence)

This mirrors the final-step mapping in ``src.controller.ImplicitSWAController``:
    p_right = sigmoid(delta_opt / decision_temperature)

Sign convention (matches controller.py:520): delta > 0 -> prefer B / right.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Match ImplicitSWAController default (src/controller.py:78, :520).
T_DECISION: float = 1.0
# Clip confidence so |delta| is bounded (~4.6 at 0.99). Prevents infinities
# when the judge returns 1.0 and keeps the pseudo-delta in the empirical
# range produced by Phi-4's real logit gaps.
CONF_CLIP: float = 0.99
UNCERTAIN_DELTA: float = 0.0


def pseudo_delta_from_judge(
    choice: str,
    confidence: float,
    *,
    t_decision: float = T_DECISION,
    conf_clip: float = CONF_CLIP,
) -> float:
    """Map a judge verdict {choice, confidence} to a scalar pseudo-logit-gap.

    Args:
        choice: one of ``"A"``, ``"B"``, ``"UNCERTAIN"``.
        confidence: float in [0, 1]; interpreted as P(chosen option).
        t_decision: sigmoid temperature to round-trip with (default 1.0).
        conf_clip: upper bound on |confidence| to keep |delta| finite.

    Returns:
        float delta such that ``sigmoid(delta / t_decision)`` equals the
        judge's implied P(right). ``UNCERTAIN`` returns 0.0 (p_right = 0.5).
    """
    if choice == "UNCERTAIN":
        return UNCERTAIN_DELTA
    if choice not in ("A", "B"):
        return UNCERTAIN_DELTA
    c = float(np.clip(float(confidence), 1.0 - conf_clip, conf_clip))
    p_right = c if choice == "B" else (1.0 - c)
    return float(t_decision) * math.log(p_right / (1.0 - p_right))


def pseudo_p_right_from_delta(
    delta: float, *, t_decision: float = T_DECISION
) -> float:
    """Inverse helper: sigmoid(delta / T) -> P(right). Used for vanilla-from-base."""
    return float(1.0 / (1.0 + math.exp(-float(delta) / float(t_decision))))
