"""Inference-time prompt-only baselines for cultural alignment.

Implements four prompt-prefix baselines that share the same inference
machinery as the vanilla baseline runner (see ``baseline_runner.py``):

  • B1  Country-Tailored Prompt
  • B2  WVS Profile Prompting
  • B3  PRISM-Style Prompting (short / long variants)
  • B4  Country + WVS Profile  (B1 ∪ B2 — strongest prompt-only baseline)

Each baseline prepends a cultural framing string to the standard MultiTP
scenario prompt, then re-uses the vanilla A/B logit-gap pipeline.  No
weights are touched and no per-country supervision is consumed.

These baselines isolate "what a careful prompt alone can do," the natural
counterpart to the persona-disagreement signal SWA-DPBR exploits.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from src.constants import COUNTRY_FULL_NAMES, COUNTRY_LANG
from src.i18n import PROMPT_FRAME_I18N

# ─────────────────────────────────────────────────────────────────────────────
# Cultural-prefix templates.  All variants are written in English so they work
# uniformly across model families; the underlying scenario stays in the
# country's native language via ``PROMPT_FRAME_I18N`` (see baseline_runner.py).
# ─────────────────────────────────────────────────────────────────────────────

_B1_COUNTRY_TAILORED = (
    "You are evaluating the following moral dilemma. "
    "Please choose A or B as a typical adult resident of {country_name} "
    "would choose."
)

_B3_PRISM_SHORT = (
    "You are evaluating the following moral dilemma from the perspective of "
    "a typical person from {country_name}. Please choose as most people "
    "in {country_name} would choose."
)

_B3_PRISM_LONG = (
    "You are taking the viewpoint of a person raised and living in "
    "{country_name}, with the cultural values, social norms, and moral "
    "priorities typical of that country. "
    "When making the judgment below, reflect the preferences that most "
    "people in {country_name} would express, not your general defaults. "
    "Choose A or B as a representative person from {country_name} would."
)

# B2: WVS profile sentence template.  We summarise the country's normalised
# 10-dim WVS-7 vector into one sentence using qualitative descriptors so the
# baseline is comparable to the persona panel SWA-DPBR uses internally
# (Appendix A.5 of the paper).
_B2_WVS_PROFILE_HEADER = (
    "Cultural context for {country_name} (World Values Survey, Wave 7): "
)


# ─── descriptor mapping (mirrors persona quartile cuts) ─────────────────────
_QUARTILE_CUTS = {
    "religiosity":             ("deeply religious", "moderately religious",
                                "somewhat secular", "highly secular"),
    "child_rearing":           ("strongly values obedience and faith",
                                "leans toward obedience", "leans toward independence",
                                "strongly values independence and imagination"),
    "moral_acceptability":     ("very permissive on contested issues",
                                "moderately permissive", "moderately conservative",
                                "strictly conservative"),
    "social_trust":            ("very high social trust", "moderate social trust",
                                "low social trust", "very low social trust"),
    "political_participation": ("active political participant",
                                "occasional participant", "infrequent participant",
                                "non-participant"),
    "national_pride":          ("very proud of nation", "moderately proud",
                                "lukewarm on nation", "not proud"),
    "happiness":               ("very happy", "moderately happy",
                                "somewhat unhappy", "unhappy"),
    "gender_equality":         ("strongly egalitarian on gender",
                                "moderately egalitarian", "moderately traditional",
                                "strongly traditional"),
    "materialism_orientation": ("post-materialist values", "leans post-materialist",
                                "leans materialist", "strongly materialist"),
    "tolerance_diversity":     ("very tolerant of diversity", "moderately tolerant",
                                "somewhat intolerant", "intolerant"),
}


def _quartile_descriptor(feature: str, value01: float) -> str:
    """Map a [0,1] WVS feature score to a 4-level natural-language descriptor."""
    if value01 != value01 or value01 is None:  # NaN check
        return "(unknown)"
    cuts = _QUARTILE_CUTS.get(feature)
    if cuts is None:
        return f"{value01:.2f}"
    if value01 >= 0.75:
        return cuts[0]
    if value01 >= 0.50:
        return cuts[1]
    if value01 >= 0.25:
        return cuts[2]
    return cuts[3]


def _build_wvs_summary_sentence(country: str, wvs_vec: Dict[str, float]) -> str:
    """One sentence summarising the country's 10-dim WVS profile.

    Used by B2 (WVS Profile Prompting) and B4 (Country + WVS).
    """
    name = COUNTRY_FULL_NAMES.get(country, country)
    parts = []
    for feat in (
        "religiosity", "moral_acceptability", "social_trust",
        "national_pride", "gender_equality", "tolerance_diversity",
    ):
        if feat in wvs_vec:
            parts.append(_quartile_descriptor(feat, wvs_vec[feat]))
    if not parts:
        return _B2_WVS_PROFILE_HEADER.format(country_name=name) + "(no data available)."
    summary = ", ".join(parts)
    return _B2_WVS_PROFILE_HEADER.format(country_name=name) + summary + "."


# ─── prefix builders ─────────────────────────────────────────────────────────
def _country_name(country: str) -> str:
    return COUNTRY_FULL_NAMES.get(country, country)


def b1_prefix(country: str, **_) -> str:
    return _B1_COUNTRY_TAILORED.format(country_name=_country_name(country))


def b2_prefix(country: str, wvs_vec: Optional[Dict[str, float]] = None, **_) -> str:
    if not wvs_vec:
        # Degrade gracefully: equivalent to B1 if WVS vector is unavailable.
        return b1_prefix(country)
    return _build_wvs_summary_sentence(country, wvs_vec)


def b3_prefix(country: str, strength: str = "short", **_) -> str:
    name = _country_name(country)
    tpl = _B3_PRISM_LONG if strength == "long" else _B3_PRISM_SHORT
    return tpl.format(country_name=name)


def b4_prefix(country: str, wvs_vec: Optional[Dict[str, float]] = None, **_) -> str:
    """B1 + B2 — the strongest prompt-only baseline."""
    return b1_prefix(country) + " " + b2_prefix(country, wvs_vec=wvs_vec)


# Public registry — used by the runner to dispatch by name.
PREFIX_BUILDERS: Dict[str, Callable[..., str]] = {
    "B1": b1_prefix,
    "B2": b2_prefix,
    "B3_short": lambda c, **kw: b3_prefix(c, strength="short"),
    "B3_long":  lambda c, **kw: b3_prefix(c, strength="long"),
    "B4": b4_prefix,
}


# ─── prompt wrapper used by the runner ──────────────────────────────────────
def wrap_scenario(prompt: str, country: str, baseline: str,
                  wvs_vec: Optional[Dict[str, float]] = None) -> str:
    """Prepend the chosen cultural prefix to the scenario user-content.

    The output is what gets fed into ``ChatTemplateHelper.encode_query_suffix``
    inside ``baseline_runner.py``-style inference loops.
    """
    builder = PREFIX_BUILDERS.get(baseline)
    if builder is None:
        raise ValueError(
            f"Unknown prompt-baseline {baseline!r}. "
            f"Choose from: {sorted(PREFIX_BUILDERS)}"
        )
    prefix = builder(country, wvs_vec=wvs_vec)
    lang = COUNTRY_LANG.get(country, "en")
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    return prefix + "\n\n" + frame.format(scenario=prompt)


# ─── WVS vector loader ──────────────────────────────────────────────────────
def load_wvs_vector(wvs_csv_path: str, country: str) -> Optional[Dict[str, float]]:
    """Return the 10-dim normalised WVS vector for ``country`` (or None).

    Mirrors the loader used by ``personas.py`` so B2/B4 stay aligned with the
    SWA-DPBR persona panel.
    """
    try:
        from src.personas import load_wvs_profiles
        profiles = load_wvs_profiles(wvs_csv_path, [country])
        return profiles.get(country, {}).get("normalised", profiles.get(country))
    except Exception:
        return None
