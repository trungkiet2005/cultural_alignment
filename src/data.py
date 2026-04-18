"""Data loading and dataset management for MultiTP scenarios."""

import os
import ast
import hashlib
import random as _rng

import numpy as np
import pandas as pd

from src.constants import (
    MULTITP_VALID_CATEGORIES, UTILITARIANISM_QUALITY_ROLES,
    MAX_SCENARIOS_PER_CATEGORY,
)


def find_multitp_csv(data_base_path, lang, translator, suffix):
    csv_name = f"dataset_{lang}+{translator}{suffix}.csv"
    csv_path = os.path.join(data_base_path, "datasets", csv_name)
    if os.path.exists(csv_path):
        return csv_path
    datasets_dir = os.path.join(data_base_path, "datasets")
    if os.path.isdir(datasets_dir):
        available = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
        if available:
            print(f"[DATA] Exact file not found, using: {available[0]}")
            return os.path.join(datasets_dir, available[0])
        raise FileNotFoundError(f"No dataset CSVs in {datasets_dir}.")
    available = sorted(
        f for f in os.listdir(data_base_path)
        if f.startswith("dataset_") and f.endswith(".csv")
    )
    if available:
        print(f"[DATA] Found dataset at root: {available[0]}")
        return os.path.join(data_base_path, available[0])
    raise FileNotFoundError(f"No MultiTP dataset CSVs found in {data_base_path}.")


def parse_left_right(row, sub1, sub2, g1, g2):
    paraphrase = str(row.get("paraphrase_choice", ""))
    if f"first {sub1}" in paraphrase and f"then {sub2}" in paraphrase:
        return g1, g2, sub1, sub2, False
    if f"first {sub2}" in paraphrase and f"then {sub1}" in paraphrase:
        return g2, g1, sub2, sub1, False
    first_idx = paraphrase.find("first ")
    if first_idx >= 0:
        after_first = paraphrase[first_idx + 6:]
        if after_first.startswith(sub1):
            return g1, g2, sub1, sub2, False
        if after_first.startswith(sub2):
            return g2, g1, sub2, sub1, False
    # Deterministic fallback: use hashlib for cross-session reproducibility
    h = int(hashlib.sha256(f"{sub1}|{sub2}|{g1}|{g2}".encode()).hexdigest(), 16) % 2
    if h == 0:
        return g1, g2, sub1, sub2, True
    return g2, g1, sub2, sub1, True


def is_utilitarianism_quality(g1, g2):
    if len(g1) != len(g2):
        return False
    return set(g1) | set(g2) <= UTILITARIANISM_QUALITY_ROLES


def load_multitp_dataset(data_base_path, lang="en", translator="google",
                          suffix="", n_scenarios=500, seed=42,
                          max_per_category=MAX_SCENARIOS_PER_CATEGORY,
                          *,
                          cap_per_category: bool = True,
                          dump_ids_path: str = ""):
    """Load + balance MultiTP scenarios.

    Round-2 preprocessing flags (reviewer W5):
        * ``cap_per_category`` — if False, skip the per-category sub-sampling
          cap. Used by :mod:`exp_paper.exp_r2_no_oversampling` to show that
          the headline numbers are not an artefact of the cap.
        * ``dump_ids_path`` — if non-empty, write the exact ``Prompt`` list
          selected for this (lang, seed) run to a CSV at the given path so
          reviewers can reproduce the scenario slice bit-for-bit.
    """
    # Lazy imports to avoid circular dependencies and keep module lightweight
    from src.constants import PHENOMENON_GROUP, SCENARIO_STARTS
    from src.i18n import SCENARIO_STARTS_I18N
    from src.scenarios import make_scenario_prompt, verbalize_group_lang

    csv_path = find_multitp_csv(data_base_path, lang, translator, suffix)
    print(f"[DATA] Loading MultiTP dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[DATA] Raw MultiTP rows: {len(df)}")
    if "which_paraphrase" in df.columns:
        df = df[df["which_paraphrase"] == 0].copy()
        print(f"[DATA] After dedup (paraphrase=0): {len(df)} rows")

    _rng.seed(seed)
    np.random.seed(seed)
    rows = []
    n_quality_filtered = 0
    n_fallback = 0

    for _, row in df.iterrows():
        cat = row.get("phenomenon_category", "")
        if cat not in MULTITP_VALID_CATEGORIES:
            continue
        sub1 = str(row.get("sub1", ""))
        sub2 = str(row.get("sub2", ""))
        try:
            g1 = ast.literal_eval(str(row.get("group1", "[]")))
            g2 = ast.literal_eval(str(row.get("group2", "[]")))
        except (ValueError, SyntaxError):
            g1, g2 = ["Person"], ["Person"]
        if not isinstance(g1, list):
            g1 = [str(g1)]
        if not isinstance(g2, list):
            g2 = [str(g2)]
        if cat == "Utilitarianism" and is_utilitarianism_quality(g1, g2):
            n_quality_filtered += 1
            continue
        mapped_cat = cat
        preferred_sub = PHENOMENON_GROUP[cat]
        preferred_group = PHENOMENON_GROUP.get(mapped_cat, preferred_sub)
        left_group, right_group, left_sub, right_sub, used_fallback = (
            parse_left_right(row, sub1, sub2, g1, g2)
        )
        if used_fallback:
            n_fallback += 1
        preferred_on_right = int(preferred_sub == right_sub)
        left_desc = verbalize_group_lang(left_group, lang)
        right_desc = verbalize_group_lang(right_group, lang)
        context = _rng.choice(SCENARIO_STARTS_I18N.get(lang, SCENARIO_STARTS))
        prompt = make_scenario_prompt(context, left_desc, right_desc, is_pedped=True, lang=lang)
        rows.append({
            "Prompt": prompt,
            "phenomenon_category": mapped_cat,
            "this_group_name": preferred_group,
            "preferred_on_right": preferred_on_right,
            "n_left": len(left_group),
            "n_right": len(right_group),
            "source": "multitp",
        })

    real_df = pd.DataFrame(rows)
    print(f"[DATA] Utilitarianism quality rows filtered: {n_quality_filtered}")
    if n_fallback > 0:
        pct = n_fallback / len(rows) * 100 if rows else 0
        print(f"[WARN] paraphrase_choice fallback: {n_fallback} rows ({pct:.1f}%)")
        if pct > 5:
            print(f"[WARN] Fallback rate >{5}% — check MultiTP CSV format!")

    balanced_parts = []
    for cat in real_df["phenomenon_category"].unique():
        cat_df = real_df[real_df["phenomenon_category"] == cat]
        if cap_per_category and len(cat_df) > max_per_category:
            cat_df = cat_df.sample(n=max_per_category, random_state=seed)
        balanced_parts.append(cat_df)

    result_df = pd.concat(balanced_parts, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if dump_ids_path:
        import os as _os  # local to avoid top-level churn
        _os.makedirs(_os.path.dirname(dump_ids_path) or ".", exist_ok=True)
        result_df[["Prompt", "phenomenon_category", "preferred_on_right",
                   "n_left", "n_right"]].to_csv(dump_ids_path, index=False)
        print(f"[DATA] Dumped scenario ids → {dump_ids_path}")

    side_pct = result_df["preferred_on_right"].mean()
    if side_pct < 0.3 or side_pct > 0.7:
        print(f"[WARN] Side balance skewed: {side_pct:.1%} preferred on RIGHT")

    n_cats = result_df["phenomenon_category"].nunique()
    print(f"[DATA] Final dataset: {len(result_df)} scenarios ({n_cats} categories)")
    print(f"[DATA] Side balance: {side_pct:.1%} preferred on RIGHT")
    print(f"[DATA] Category distribution:")
    for cat, cnt in result_df["phenomenon_category"].value_counts().sort_index().items():
        print(f"  {cat:20s}: {cnt:4d}")
    return result_df


def load_scenario_dataset(path: str, n_scenarios: int = 500) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.head(n_scenarios)
    print(f"[DATA] Loaded {len(df)} scenarios from {path}")
    return df


