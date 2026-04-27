"""Open-ended prompt frames for the generate-then-judge SWA-DPBR variant.

**Open-ended mode (2026-04-27)**: the actor sees the dilemma and is asked to
choose between A and B. No reasoning structure is imposed — the model may
think out loud, refuse, hedge, or just emit a letter. Whatever it produces is
handed to the judge, which is responsible for extracting the committed choice
(or returning UNCERTAIN if none is clear).

Pair with ``max_new_tokens ≈ 256`` in the actor caller — enough headroom for
the model to reason if it wants to, without bloating wall-time. Drop to ~8 to
revert to a constrained A/B-only mode (no judge needed in that limit).

English is authoritative; all other locales currently fall back to English via
``OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])``. Per-language
translations can be added incrementally. The judge still sees the canonical
English scenario for stability.
"""

from __future__ import annotations

from typing import Dict

OPENENDED_FRAME_I18N: Dict[str, str] = {
    "en": "Moral dilemma:\n{scenario}",
}


def build_openended_prompt(scenario: str, lang: str = "en") -> str:
    """Format a scenario into the open-ended actor prompt for ``lang``.

    Falls back to English if ``lang`` is not in :data:`OPENENDED_FRAME_I18N`.
    """
    frame = OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])
    return frame.format(scenario=scenario)
