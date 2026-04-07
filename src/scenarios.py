"""Scenario generation and verbalization for moral dilemma prompts."""

import random as _rng
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

from src.constants import CHARACTERS, CATEGORY_POOLS, PHENOMENON_GROUP
from src.i18n import SCENARIO_FRAME_I18N, CHARACTERS_I18N, SCENARIO_STARTS_I18N


def make_scenario_prompt(context, left_desc, right_desc,
                         left_legality="", right_legality="",
                         is_pedped=True, lang="en"):
    """Build a scenario prompt in the given language."""
    sf = SCENARIO_FRAME_I18N.get(lang, SCENARIO_FRAME_I18N["en"])
    left_note = f" ({left_legality})" if left_legality else ""
    right_note = f" ({right_legality})" if right_legality else ""
    if is_pedped:
        gl, gr = sf["group_a"], sf["group_b"]
    else:
        gl, gr = sf["passengers"], sf["pedestrians"]
    return (
        f"{context}\n\n"
        f"{sf['left_lane']} — {gl}: {left_desc}{left_note}\n"
        f"{sf['right_lane']} — {gr}: {right_desc}{right_note}\n\n"
        f"{sf['closing']}"
    )


def verbalize_group_lang(char_list: List[str], lang: str = "en") -> str:
    """Verbalize a character list in the given language."""
    chars_i18n = CHARACTERS_I18N.get(lang, CHARACTERS_I18N["en"])
    counts = Counter(char_list)
    parts = []
    for char_type, cnt in counts.items():
        if char_type not in chars_i18n:
            # Fall back to English
            singular, plural = CHARACTERS.get(char_type, (char_type, char_type + "s"))
        else:
            singular, plural = chars_i18n[char_type]
        if cnt == 1:
            if lang == "en":
                article = "an" if singular[0] in "aeiou" else "a"
                parts.append(f"{article} {singular}")
            elif lang in ("zh", "ja", "ko", "ar", "vi", "hi"):
                parts.append(f"1名{singular}" if lang in ("zh", "ja", "ko") else f"{cnt} {singular}")
            else:
                parts.append(f"1 {singular}")
        else:
            parts.append(f"{cnt} {plural}")
    # Language-specific conjunctions
    _CONJUNCTIONS = {
        "zh": ("、", "和"), "ja": ("、", "と"), "ko": ("、", "그리고 "),
        "de": (", ", " und "), "fr": (", ", " et "), "pt": (", ", " e "),
        "ar": ("، ", " و"), "vi": (", ", " và "), "hi": (", ", " और "),
        "ru": (", ", " и "), "es": (", ", " y "),
    }
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        if lang == "zh":
            sep = "和"
        elif lang == "ja":
            sep = "と"
        elif lang == "ko":
            # Korean: 과 after consonant-ending, 와 after vowel-ending
            last_char = parts[0][-1] if parts[0] else ''
            sep = "과 " if (last_char and ord(last_char) >= 0xAC00 and (ord(last_char) - 0xAC00) % 28 != 0) else "와 "
        else:
            sep = " and "
        return f"{parts[0]}{sep}{parts[1]}"
    else:
        list_sep, final_conj = _CONJUNCTIONS.get(lang, (", ", ", and "))
        return list_sep.join(parts[:-1]) + final_conj + parts[-1]


def generate_multitp_scenarios(n_scenarios: int = 500, seed: int = 42,
                                max_chars_per_group: int = 5,
                                lang: str = "en") -> pd.DataFrame:
    """Generate synthetic MultiTP scenarios in the given language."""
    _rng.seed(seed)
    np.random.seed(seed)
    rows = []
    phenomena = list(CATEGORY_POOLS.keys())
    per_phenom = max(n_scenarios // len(phenomena), 10)

    scenario_starts = SCENARIO_STARTS_I18N.get(lang, SCENARIO_STARTS_I18N["en"])

    for phenom in phenomena:
        non_pref_pool, pref_pool = CATEGORY_POOLS[phenom]
        group_name = PHENOMENON_GROUP[phenom]

        for i in range(per_phenom):
            ctx = _rng.choice(scenario_starts)
            if phenom == "Utilitarianism":
                n_non_pref = _rng.randint(1, 2)
                n_pref = n_non_pref + _rng.randint(1, 3)
            else:
                n_both = _rng.randint(1, min(3, max_chars_per_group))
                n_non_pref = n_both
                n_pref = n_both

            non_pref_chars = [_rng.choice(non_pref_pool) for _ in range(n_non_pref)]
            pref_chars = [_rng.choice(pref_pool) for _ in range(n_pref)]
            non_pref_desc = verbalize_group_lang(non_pref_chars, lang)
            pref_desc = verbalize_group_lang(pref_chars, lang)

            is_pedped = True
            preferred_on_right = _rng.random() < 0.5
            if preferred_on_right:
                left_chars_desc, right_chars_desc = non_pref_desc, pref_desc
            else:
                left_chars_desc, right_chars_desc = pref_desc, non_pref_desc

            prompt = make_scenario_prompt(ctx, left_chars_desc, right_chars_desc,
                                          is_pedped=is_pedped, lang=lang)
            rows.append({
                "Prompt": prompt,
                "phenomenon_category": phenom,
                "this_group_name": group_name,
                "two_choices_unordered_set": f"{{{non_pref_desc}}} vs {{{pref_desc}}}",
                "preferred_on_right": int(preferred_on_right),
                "n_left": n_non_pref if preferred_on_right else n_pref,
                "n_right": n_pref if preferred_on_right else n_non_pref,
                "lang": lang,
            })

    _rng.shuffle(rows)
    rows = rows[:n_scenarios]
    return pd.DataFrame(rows)
