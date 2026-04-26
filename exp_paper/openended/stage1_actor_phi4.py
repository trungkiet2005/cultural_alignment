"""Stage 1 of the open-ended DISCA variant — actor text generation (single-pass).

Iterates (country, scenario, agent) and asks the actor model to produce a
reasoning paragraph followed by a committed "ANSWER: A"/"ANSWER: B" line, given
a persona-prepended system prompt. All generations are appended to
``{out_jsonl_dir}/{country}.jsonl`` so Stage 2 can read them, map each to a
pseudo-delta via :func:`src.pseudo_delta.pseudo_delta_from_judge`, and feed the
result to :class:`exp_paper.openended.dpbr_offline.Exp24DualPassControllerOffline`.

Note (2026-04-26): The A↔B positional debiasing pass (pass2) was removed.
In free-text greedy decoding the swapped prompt does not produce a clean mirror
of the original — semantic content dominates, so ``(δ₁ − δ₂) / 2`` injects noise
instead of cancelling positional bias. Empirically this drove SWA r below VAN r
on USA/VNM/DEU. We now use single-pass pseudo-deltas directly. ``DEBIAS_VARIANTS``
remains exposed as ``("pass1",)`` so legacy callers / Stage 2 keep importing it.

Resume-safe: before generating a row the script hashes
``(country, scenario_id, agent_role, "pass1")`` into a set of already-written
keys. Generations are ``greedy``; `do_sample=False` gives reproducible text
across re-runs of the same (prompt, persona) pair.

Persona set matches the logit-based runner (``exp_model/_base_dpbr.py``): one
``base`` agent using :data:`BASE_ASSISTANT_I18N` and 4 WVS-derived personas
from :func:`build_country_personas`.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import torch

from src.constants import COUNTRY_LANG
from src.data import load_multitp_dataset
from src.i18n import BASE_ASSISTANT_I18N
from src.model import ChatTemplateHelper, load_model_hf_native, setup_seeds
from src.openended_prompts import build_openended_prompt
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.scenarios import generate_multitp_scenarios


AGENT_ROLES: Tuple[str, ...] = ("base", "persona_0", "persona_1", "persona_2", "persona_3")
# Single-pass: pass2 (A↔B swap) was disabled — see module docstring.
DEBIAS_VARIANTS: Tuple[str, ...] = ("pass1",)
SCHEMA_VERSION: int = 1

# Subset of columns mirrored to ``{country}.csv`` for human inspection.
# `persona_text` and `prompt` are skipped — they're long and repeat per agent;
# the full record is still in ``{country}.jsonl``.
CSV_FIELDS: Tuple[str, ...] = (
    "country", "scenario_id", "phenomenon_category", "this_group_name",
    "preferred_on_right", "n_left", "n_right", "lang",
    "agent_role", "debias_variant",
    "gen_tokens", "gen_seconds",
    "scenario_native", "scenario_en", "actor_text",
)


@dataclass
class Stage1Config:
    model_name: str = "microsoft/phi-4"
    out_jsonl_dir: str = "/kaggle/working/cultural_alignment/results/openended/stage1"
    multitp_data_path: str = ""
    wvs_data_path: str = ""
    use_real_data: bool = True
    n_scenarios: int = 500
    max_new_tokens: int = 400
    load_in_4bit: bool = True
    seed: int = 42
    flush_every: int = 20
    countries: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _load_scenarios_bilingual(
    cfg: Stage1Config, country: str
) -> pd.DataFrame:
    """Return a DataFrame with both native-language and English prompts.

    Same seed + same source CSV => same row selection + same ordering in both
    languages, so we can zip them by scenario index safely.
    """
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data and cfg.multitp_data_path and os.path.isdir(cfg.multitp_data_path):
        df_native = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            n_scenarios=cfg.n_scenarios, seed=cfg.seed,
        ).reset_index(drop=True)
        df_en = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang="en",
            n_scenarios=cfg.n_scenarios, seed=cfg.seed,
        ).reset_index(drop=True)
    else:
        df_native = generate_multitp_scenarios(
            cfg.n_scenarios, seed=cfg.seed, lang=lang
        ).reset_index(drop=True)
        df_en = generate_multitp_scenarios(
            cfg.n_scenarios, seed=cfg.seed, lang="en"
        ).reset_index(drop=True)

    n = min(len(df_native), len(df_en))
    df_native = df_native.iloc[:n].copy()
    df_en = df_en.iloc[:n].copy()
    df_native["scenario_en"] = df_en["Prompt"].values
    df_native["lang"] = lang
    df_native.rename(columns={"Prompt": "scenario_native"}, inplace=True)
    return df_native


def _backfill_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> int:
    """Replay JSONL rows into CSV when the mirror is missing/empty.

    Skips legacy ``debias_variant != "pass1"`` rows (the A↔B swap was disabled).
    Returns the number of rows written. No-op when JSONL is missing.
    """
    if not jsonl_path.exists():
        return 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return 0
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as jh, \
         csv_path.open("w", encoding="utf-8", newline="") as ch:
        writer = csv.DictWriter(
            ch, fieldnames=list(CSV_FIELDS),
            quoting=csv.QUOTE_ALL, lineterminator="\n",
        )
        writer.writeheader()
        for line in jh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("debias_variant", "pass1")) != "pass1":
                continue
            writer.writerow({k: rec.get(k, "") for k in CSV_FIELDS})
            n += 1
    return n


def _read_existing_keys(jsonl_path: Path) -> Set[Tuple[str, int, str, str]]:
    keys: Set[Tuple[str, int, str, str]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((
                str(rec.get("country", "")),
                int(rec.get("scenario_id", -1)),
                str(rec.get("agent_role", "")),
                str(rec.get("debias_variant", "")),
            ))
    return keys


def _build_persona_texts(country: str, wvs_path: str, lang: str) -> List[str]:
    """Return 5 system-prompt strings: [base, persona_0, ..., persona_3]."""
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    cultural = build_country_personas(country, wvs_path=wvs_path)
    if len(cultural) < 4:
        # build_country_personas enforces 4, but defensively pad.
        cultural = list(cultural) + [cultural[-1]] * (4 - len(cultural))
    return [base_text] + list(cultural[:4])


@torch.no_grad()
def _generate_one(
    model, tokenizer, helper: ChatTemplateHelper,
    persona_text: str, user_content: str,
    max_new_tokens: int, device: torch.device,
) -> Tuple[str, int, float]:
    """Greedy decode one response. Returns (decoded_text, n_new_tokens, seconds)."""
    prefix_ids = helper.build_prefix_ids(persona_text, device)
    query_ids = helper.encode_query_suffix(user_content, device)
    input_ids = torch.cat([prefix_ids, query_ids], dim=1)

    t0 = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    elapsed = time.time() - t0
    prompt_len = input_ids.shape[1]
    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0]), elapsed


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_stage1(cfg: Stage1Config) -> None:
    if not cfg.countries:
        raise ValueError("Stage1Config.countries is empty")
    setup_seeds(cfg.seed)

    out_dir = Path(cfg.out_jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED Stage 1 (SWA-DPBR) — actor=[{cfg.model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] out_dir={out_dir}  countries={cfg.countries}  n_scenarios={cfg.n_scenarios}")
    print(f"[CFG] max_new_tokens={cfg.max_new_tokens}  4bit={cfg.load_in_4bit}  seed={cfg.seed}")
    print(f"[CFG] use_real_data={cfg.use_real_data}  flush_every={cfg.flush_every}")

    model, tokenizer = load_model_hf_native(
        cfg.model_name, max_seq_length=2048, load_in_4bit=cfg.load_in_4bit,
    )
    helper = ChatTemplateHelper(tokenizer)
    device = next(model.parameters()).device

    try:
        for ci, country in enumerate(cfg.countries):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue
            jsonl_path = out_dir / f"{country}.jsonl"
            seen = _read_existing_keys(jsonl_path)
            print(f"\n[{ci+1}/{len(cfg.countries)}] {country}  "
                  f"existing={len(seen)} rows in {jsonl_path.name}")

            scen = _load_scenarios_bilingual(cfg, country)
            lang = COUNTRY_LANG.get(country, "en")
            personas = _build_persona_texts(country, cfg.wvs_data_path, lang)
            assert len(personas) == len(AGENT_ROLES), (
                f"Expected {len(AGENT_ROLES)} personas, got {len(personas)}"
            )

            csv_path = out_dir / f"{country}.csv"
            backfilled = _backfill_csv_from_jsonl(jsonl_path, csv_path)
            if backfilled:
                print(f"  [CSV] backfilled {backfilled} pass1 rows from {jsonl_path.name} "
                      f"-> {csv_path.name}")
            csv_needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0

            rows_since_flush = 0
            with jsonl_path.open("a", encoding="utf-8") as fh, \
                 csv_path.open("a", encoding="utf-8", newline="") as csv_fh:
                csv_writer = csv.DictWriter(
                    csv_fh, fieldnames=list(CSV_FIELDS),
                    quoting=csv.QUOTE_ALL, lineterminator="\n",
                )
                if csv_needs_header:
                    csv_writer.writeheader()

                for sid, row in scen.iterrows():
                    sid = int(sid)
                    scenario_native = str(row["scenario_native"])
                    scenario_en = str(row["scenario_en"])
                    phenom_cat = str(row.get("phenomenon_category", "default"))
                    group_name = str(row.get("this_group_name", ""))
                    pref_on_right = int(row.get("preferred_on_right", 1))
                    n_left = int(row.get("n_left", 0))
                    n_right = int(row.get("n_right", 0))

                    for agent_idx, agent_role in enumerate(AGENT_ROLES):
                        persona_text = personas[agent_idx]
                        key = (country, sid, agent_role, "pass1")
                        if key in seen:
                            continue
                        user_content = build_openended_prompt(scenario_native, lang)
                        actor_text, n_new, secs = _generate_one(
                            model, tokenizer, helper,
                            persona_text, user_content,
                            cfg.max_new_tokens, device,
                        )
                        rec = {
                            "country": country,
                            "scenario_id": sid,
                            "phenomenon_category": phenom_cat,
                            "this_group_name": group_name,
                            "preferred_on_right": pref_on_right,
                            "n_left": n_left,
                            "n_right": n_right,
                            "lang": lang,
                            "scenario_en": scenario_en,
                            "scenario_native": scenario_native,
                            "agent_role": agent_role,
                            "persona_text": persona_text,
                            "debias_variant": "pass1",
                            "swap_changed": False,
                            "prompt": user_content,
                            "actor_text": actor_text,
                            "gen_tokens": n_new,
                            "gen_seconds": secs,
                            "schema_version": SCHEMA_VERSION,
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        csv_writer.writerow({k: rec[k] for k in CSV_FIELDS})
                        rows_since_flush += 1
                        if rows_since_flush >= cfg.flush_every:
                            fh.flush()
                            csv_fh.flush()
                            rows_since_flush = 0
                fh.flush()
                csv_fh.flush()

            print(f"  [OK] {country} -> {jsonl_path}")
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Stage 1] DONE")
