"""Open-ended prompt frames for the generate-then-judge SWA-DPBR variant.

**Constrained generation mode (2026-04-26)**: the actor is asked to emit ONLY a
single letter ("A" or "B") with no reasoning chain. This brings the open-ended
pipeline as close as possible to the logit-based one: the persona system prompt
drives the choice directly without the reasoning chain "talking the model out
of" the persona prior. Pair with ``max_new_tokens ≈ 8`` in the actor caller —
just enough room for tokens like " A" / "B\\n" / "Option A" without bloating.

English is authoritative; all other locales currently fall back to English via
``OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])``. Per-language
translations can be added incrementally. The judge still sees the canonical
English scenario for stability.
"""

from __future__ import annotations

from typing import Dict

OPENENDED_FRAME_I18N: Dict[str, str] = {
    "en": (
        "The following is a moral dilemma:\n{scenario}\n\n"
        "Based on your cultural and ethical perspective, which option do you choose? "
        "Output ONLY a single letter — either A or B. No explanation, no extra text.\n\n"
        "Your answer (A or B):"
    ),
}


def build_openended_prompt(scenario: str, lang: str = "en") -> str:
    """Format a scenario into the open-ended actor prompt for ``lang``.

    Falls back to English if ``lang`` is not in :data:`OPENENDED_FRAME_I18N`.
    """
    frame = OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])
    return frame.format(scenario=scenario)
