"""Open-ended prompt frames for the generate-then-judge SWA-DPBR variant.

Parallel to :data:`src.i18n.PROMPT_FRAME_I18N` but asks the actor to produce a
short reasoning paragraph followed by a committed final answer ("ANSWER: A" or
"ANSWER: B") instead of emitting a single A/B token. The final-answer suffix is
strict so a downstream judge LLM (or even a regex fallback) can reliably locate
the chosen option in the generated text.

English is authoritative; all other locales currently fall back to English via
``OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])``. Phi-4 follows
English instructions even when reasoning in another language, and the judge
always sees the canonical English scenario for stability. Per-language
translations can be added incrementally.
"""

from __future__ import annotations

from typing import Dict

OPENENDED_FRAME_I18N: Dict[str, str] = {
    "en": (
        "The following is a moral dilemma:\n{scenario}\n\n"
        "Think through the trade-offs step by step from your cultural and ethical "
        "perspective. You MUST commit to either option A or option B — do not refuse "
        "or pick \"neither\".\n\n"
        "Respond in exactly this format:\n"
        "REASONING: <2-5 sentences explaining your moral reasoning>\n"
        "ANSWER: A   (or)   ANSWER: B\n\n"
        "Your final line must be literally \"ANSWER: A\" or \"ANSWER: B\" with no extra text."
    ),
}


def build_openended_prompt(scenario: str, lang: str = "en") -> str:
    """Format a scenario into the open-ended actor prompt for ``lang``.

    Falls back to English if ``lang`` is not in :data:`OPENENDED_FRAME_I18N`.
    """
    frame = OPENENDED_FRAME_I18N.get(lang, OPENENDED_FRAME_I18N["en"])
    return frame.format(scenario=scenario)
