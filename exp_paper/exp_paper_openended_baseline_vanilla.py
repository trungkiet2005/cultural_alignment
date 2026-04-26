#!/usr/bin/env python3
"""
Paper sweep — Open-Ended VANILLA BASELINE — Qwen2.5-7B actor + Qwen2.5-7B self-judge
====================================================================================
Kaggle OFFLINE version — no Internet, no git clone, no pip install.

This is the **vanilla baseline** counterpart of
``exp_paper_openended_with_judge_llm.py``. Uses Qwen2.5-7B for BOTH actor and
judge (self-judge), BF16, no GPTQ. No SWA-DPBR (no PT-IS, no dual-pass bootstrap
reliability, no ESS anchor blend, no hierarchical prior, no persona ensemble,
no positional debiasing).

Per scenario:
    Stage 1  Qwen2.5-7B generates ONE free-form answer using only the base
             (utilitarian-neutral) persona on the original prompt (pass1, no swap).
    Stage 2  Qwen2.5-7B (same model, reloaded) judges that answer into
             {A, B, UNCERTAIN, conf}.
             Pseudo-delta = pseudo_delta_from_judge(choice, conf).
             p_right = sigmoid(pseudo_delta / T_DECISION).
             AMCE + alignment computed against human AMCE.

Workload is ~10× lighter than the SWA-DPBR variant (1 generation/scenario instead
of ~10 generation × debias variants). Self-judge avoids the 72B GPTQ dependency
chain (optimum / auto-gptq) and fits comfortably on a single 96 GB GPU.

Setup:
    1. Upload cultural_alignment as Kaggle Dataset
    2. Add Qwen2.5-7B-Instruct as Kaggle Model input (used for actor AND judge)
    3. Add multitp-data dataset
    4. Run with Internet OFF

Usage:
    !python /kaggle/input/cultural-alignment/exp_paper/exp_paper_openended_baseline_vanilla.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. KAGGLE OFFLINE BOOTSTRAP                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT_DATASET_DIR = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
ACTOR_MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
# Self-judge: same model as actor (Qwen2.5-7B BF16). Avoids 72B GPTQ deps
# (optimum / auto-gptq) and fits a single 96 GB GPU comfortably.
JUDGE_MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WORK_DIR = "/kaggle/working/cultural_alignment"

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_AUTO_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MORAL_MODEL_BACKEND", "hf_native")


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    if _on_kaggle():
        project_src = None
        for cand in (PROJECT_DATASET_DIR, PROJECT_DATASET_DIR_ALT):
            if os.path.isdir(cand):
                project_src = cand
                break
        if os.path.isdir(WORK_DIR) and os.path.isfile(
            os.path.join(WORK_DIR, "src", "controller.py")
        ):
            print(f"[SETUP] Working dir exists: {WORK_DIR}")
        else:
            if project_src is None:
                raise RuntimeError(
                    "Project dataset not found. Checked: "
                    f"{PROJECT_DATASET_DIR} and {PROJECT_DATASET_DIR_ALT}"
                )
            print(f"[SETUP] Copying project from {project_src} → {WORK_DIR} ...")
            shutil.copytree(project_src, WORK_DIR, dirs_exist_ok=True)
        os.chdir(WORK_DIR)
        sys.path.insert(0, WORK_DIR)
        return WORK_DIR
    else:
        here = os.getcwd()
        if os.path.isfile(os.path.join(here, "src", "controller.py")):
            sys.path.insert(0, here)
            return here
        raise RuntimeError("Not on Kaggle and not inside repo root.")


def _resolve_model_path(base_path: str, label: str) -> str:
    if not _on_kaggle():
        return os.environ.get(f"{label}_MODEL_PATH", base_path)
    if os.path.isdir(base_path) and os.path.isfile(os.path.join(base_path, "config.json")):
        return base_path
    for sub in Path(base_path).rglob("config.json"):
        return str(sub.parent)
    candidates = [
        f"{base_path}/transformers/default/1",
        f"{base_path}/pytorch/default/1",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise RuntimeError(f"{label} model weights not found at {base_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. BOOTSTRAP                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  KAGGLE OFFLINE — Open-Ended VANILLA BASELINE (no SWA-DPBR)")
print("  Actor: Qwen2.5-7B-Instruct (bf16) | Judge: Qwen2.5-7B-Instruct (bf16, self-judge)")
print("=" * 70)

_setup_project()

ACTOR_MODEL_PATH = _resolve_model_path(ACTOR_MODEL_LOCAL_PATH, "ACTOR")
JUDGE_MODEL_PATH = _resolve_model_path(JUDGE_MODEL_LOCAL_PATH, "JUDGE")
print(f"[SETUP] Actor model path: {ACTOR_MODEL_PATH}")
print(f"[SETUP] Judge model path: {JUDGE_MODEL_PATH}")

if _on_kaggle():
    subprocess.run(
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf "
        "2>/dev/null || true",
        shell=True, check=False,
    )

import math  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.amce import (  # noqa: E402
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    load_human_amce,
)
from src.constants import COUNTRY_LANG  # noqa: E402
from src.data import load_multitp_dataset  # noqa: E402
from src.i18n import BASE_ASSISTANT_I18N  # noqa: E402
from src.judge_prompts import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_output,
)
from src.model import ChatTemplateHelper, load_model_hf_native, setup_seeds  # noqa: E402
from src.openended_prompts import build_openended_prompt  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES  # noqa: E402
from src.pseudo_delta import (  # noqa: E402
    T_DECISION,
    pseudo_delta_from_judge,
    pseudo_p_right_from_delta,
)
from src.scenarios import generate_multitp_scenarios  # noqa: E402

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HARDCODED RUN CONFIG — 20 paper countries × full 500 scenarios            ║
# ║  (PAPER_20_COUNTRIES: 5 continents × 6 language families)                  ║
# ║                                                                            ║
# ║  Batch presets (resume-safe — Stage1 dedupes by scenario_id, Stage2 caches ║
# ║  by sha1(scenario_en + actor_text)):                                       ║
# ║    BATCH = "all"  -> all 20 countries (one session, ~3h20m on GPU)        ║
# ║    BATCH = "b1"   -> first 10 countries                                    ║
# ║    BATCH = "b2"   -> last 10 countries                                     ║
# ║  Override via env var:  OPENENDED_BATCH=b1 | b2 | all                      ║
# ║  Or pass explicit list: OPENENDED_COUNTRIES=USA,VNM,DEU                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
RUN_BATCH: str = "all"  # "all" | "b1" | "b2"

_PAPER_20: List[str] = list(PAPER_20_COUNTRIES)
_BATCHES: Dict[str, List[str]] = {
    "all": _PAPER_20,
    "b1": _PAPER_20[:10],   # USA GBR DEU ARG BRA MEX COL VNM MMR THA
    "b2": _PAPER_20[10:],   # MYS IDN CHN JPN BGD IRN SRB ROU KGZ ETH
}


def _resolve_run_countries() -> List[str]:
    explicit = os.environ.get("OPENENDED_COUNTRIES", "").strip()
    if explicit:
        return [s.strip() for s in explicit.split(",") if s.strip()]
    batch = os.environ.get("OPENENDED_BATCH", RUN_BATCH).strip().lower()
    if batch not in _BATCHES:
        raise ValueError(
            f"OPENENDED_BATCH must be one of {sorted(_BATCHES)}, got {batch!r}"
        )
    return list(_BATCHES[batch])


RUN_COUNTRIES: List[str] = _resolve_run_countries()
RUN_N_SCENARIOS: int = 500
RUN_STAGE: str = "both"  # "1" | "2" | "both"
RUN_MAX_NEW_TOKENS_ACTOR: int = 400
RUN_MAX_NEW_TOKENS_JUDGE: int = 64
RUN_SEED: int = 42
RUN_FLUSH_EVERY: int = 20
RUN_MAX_PARSE_FAIL_PCT: float = 5.0


@dataclass
class BaselineConfig:
    actor_model_name: str
    judge_model_name: str
    out_jsonl_dir: str
    results_base: str
    multitp_data_path: str
    wvs_data_path: str
    human_amce_path: str
    countries: List[str]
    n_scenarios: int = 500
    max_new_tokens_actor: int = 400
    max_new_tokens_judge: int = 64
    seed: int = 42
    flush_every: int = 20
    max_parse_fail_pct: float = 5.0
    use_real_data: bool = True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. STAGE 1 — actor generates 1 vanilla answer per scenario                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _load_scenarios_bilingual(cfg: BaselineConfig, country: str) -> pd.DataFrame:
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


def _read_existing_scenario_ids(jsonl_path: Path) -> set[int]:
    seen: set[int] = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("scenario_id")
            if isinstance(sid, int):
                seen.add(sid)
    return seen


@torch.no_grad()
def _generate_one(
    model, tokenizer, helper: ChatTemplateHelper,
    persona_text: str, user_content: str,
    max_new_tokens: int, device: torch.device,
) -> Tuple[str, int, float]:
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
    new_ids = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0]), elapsed


def run_stage1(cfg: BaselineConfig) -> None:
    setup_seeds(cfg.seed)
    out_dir = Path(cfg.out_jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED Stage 1 (VANILLA BASELINE) — actor=[{cfg.actor_model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] out_dir={out_dir}  countries={cfg.countries}  n_scenarios={cfg.n_scenarios}")
    print(f"[CFG] max_new_tokens={cfg.max_new_tokens_actor}  4bit=False  seed={cfg.seed}")
    print(f"[CFG] use_real_data={cfg.use_real_data}  flush_every={cfg.flush_every}")

    model, tokenizer = load_model_hf_native(
        cfg.actor_model_name, max_seq_length=2048, load_in_4bit=False,
    )
    helper = ChatTemplateHelper(tokenizer)
    device = next(model.parameters()).device

    try:
        for ci, country in enumerate(cfg.countries):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue
            jsonl_path = out_dir / f"{country}.jsonl"
            seen = _read_existing_scenario_ids(jsonl_path)
            print(f"\n[{ci+1}/{len(cfg.countries)}] {country}  "
                  f"existing={len(seen)} rows in {jsonl_path.name}")

            scen = _load_scenarios_bilingual(cfg, country)
            lang = COUNTRY_LANG.get(country, "en")
            base_persona = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])

            rows_since_flush = 0
            with jsonl_path.open("a", encoding="utf-8") as fh:
                for sid, row in scen.iterrows():
                    sid = int(sid)
                    if sid in seen:
                        continue
                    scenario_native = str(row["scenario_native"])
                    scenario_en = str(row["scenario_en"])
                    user_content = build_openended_prompt(scenario_native, lang)
                    actor_text, n_new, secs = _generate_one(
                        model, tokenizer, helper,
                        base_persona, user_content,
                        cfg.max_new_tokens_actor, device,
                    )
                    rec = {
                        "country": country,
                        "scenario_id": sid,
                        "phenomenon_category": str(row.get("phenomenon_category", "default")),
                        "this_group_name": str(row.get("this_group_name", "")),
                        "preferred_on_right": int(row.get("preferred_on_right", 1)),
                        "n_left": int(row.get("n_left", 0)),
                        "n_right": int(row.get("n_right", 0)),
                        "lang": lang,
                        "scenario_en": scenario_en,
                        "scenario_native": scenario_native,
                        "prompt": user_content,
                        "actor_text": actor_text,
                        "gen_tokens": n_new,
                        "gen_seconds": secs,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_since_flush += 1
                    if rows_since_flush >= cfg.flush_every:
                        fh.flush()
                        rows_since_flush = 0
                fh.flush()

            print(f"  [OK] {country} -> {jsonl_path}")
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n[Stage 1] DONE")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. STAGE 2 — judge each text, sigmoid → p_right, no SWA-DPBR              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _record_hash(scenario_en: str, actor_text: str) -> str:
    h = hashlib.sha1()
    h.update(scenario_en.encode("utf-8"))
    h.update(b"\x00")
    h.update(actor_text.encode("utf-8"))
    return h.hexdigest()


def _load_judge_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get("hash")
            if k:
                cache[k] = obj
    return cache


@torch.no_grad()
def _judge_generate(
    judge_model, judge_tokenizer,
    scenario_en: str, actor_text: str,
    max_new_tokens: int, device: torch.device,
) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(scenario_en, actor_text)},
    ]
    if hasattr(judge_tokenizer, "apply_chat_template"):
        templated = judge_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        # Newer Transformers may return a BatchEncoding dict instead of a Tensor.
        if isinstance(templated, torch.Tensor):
            input_ids = templated.to(device)
        else:
            input_ids = templated["input_ids"].to(device)
    else:
        prompt = (
            f"{JUDGE_SYSTEM_PROMPT}\n\n"
            f"{build_judge_prompt(scenario_en, actor_text)}\n"
        )
        input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    out = judge_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=(
            judge_tokenizer.pad_token_id
            if judge_tokenizer.pad_token_id is not None
            else judge_tokenizer.eos_token_id
        ),
    )
    new_ids = out[0, input_ids.shape[1]:]
    return judge_tokenizer.decode(new_ids, skip_special_tokens=True)


def _summarize(results_df: pd.DataFrame, human_amce: Dict[str, float]) -> Dict[str, Any]:
    model_amce = compute_amce_from_preferences(results_df)
    alignment = compute_alignment_metrics(model_amce, human_amce) if human_amce else {}
    per_dim = (
        compute_per_dimension_alignment(model_amce, human_amce) if human_amce else {}
    )
    return {
        "model_amce": model_amce,
        "alignment": alignment,
        "per_dimension_alignment": per_dim,
        "n_scenarios": int(len(results_df)),
    }


def run_stage2(cfg: BaselineConfig) -> None:
    setup_seeds(cfg.seed)

    stage1_dir = Path(cfg.out_jsonl_dir)
    results_base = Path(cfg.results_base)
    van_root = results_base / "vanilla"
    cmp_root = results_base / "compare"
    judge_cache_dir = results_base / "judge_cache"
    for d in (van_root, cmp_root, judge_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED Stage 2 (VANILLA BASELINE) — judge=[{cfg.judge_model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] stage1={stage1_dir}  out={results_base}  countries={cfg.countries}")
    print(f"[CFG] max_new_tokens={cfg.max_new_tokens_judge}  T_DECISION={T_DECISION}  "
          f"max_parse_fail_pct={cfg.max_parse_fail_pct}")

    judge_model, judge_tokenizer = load_model_hf_native(
        cfg.judge_model_name, max_seq_length=4096, load_in_4bit=False,
    )
    device = next(judge_model.parameters()).device

    compare_rows: List[Dict[str, Any]] = []
    try:
        for ci, country in enumerate(cfg.countries):
            jsonl_path = stage1_dir / f"{country}.jsonl"
            if not jsonl_path.exists():
                print(f"[SKIP] {country}: no Stage-1 JSONL at {jsonl_path}")
                continue
            print(f"\n[{ci+1}/{len(cfg.countries)}] judging {country}")

            rows: List[Dict[str, Any]] = []
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not rows:
                print(f"[SKIP] {country}: empty JSONL")
                continue

            cache_path = judge_cache_dir / f"{country}_judged.jsonl"
            cache = _load_judge_cache(cache_path)
            stats = {"judged": 0, "cached": 0, "parse_fail": 0}
            t0 = time.time()

            van_rows: List[Dict[str, Any]] = []
            with cache_path.open("a", encoding="utf-8") as cache_fh:
                for r in rows:
                    scenario_en = str(r.get("scenario_en", ""))
                    actor_text = str(r.get("actor_text", ""))
                    key = _record_hash(scenario_en, actor_text)

                    if key in cache:
                        hit = cache[key]
                        raw = hit.get("raw", "")
                        parsed = {
                            "choice": hit.get("choice", "UNCERTAIN"),
                            "confidence": float(hit.get("confidence", 0.0)),
                            "parse_ok": bool(hit.get("parse_ok", False)),
                        }
                        stats["cached"] += 1
                    else:
                        raw = _judge_generate(
                            judge_model, judge_tokenizer,
                            scenario_en, actor_text,
                            cfg.max_new_tokens_judge, device,
                        )
                        parsed = parse_judge_output(raw)
                        cache_fh.write(json.dumps({
                            "hash": key,
                            "raw": raw,
                            "choice": parsed["choice"],
                            "confidence": parsed["confidence"],
                            "parse_ok": parsed["parse_ok"],
                        }, ensure_ascii=False) + "\n")
                        cache_fh.flush()
                        stats["judged"] += 1

                    if not parsed["parse_ok"]:
                        stats["parse_fail"] += 1

                    delta = pseudo_delta_from_judge(
                        parsed["choice"], float(parsed["confidence"])
                    )
                    p_right = pseudo_p_right_from_delta(delta, t_decision=T_DECISION)
                    pref_on_right = int(r.get("preferred_on_right", 1))
                    p_pref = p_right if pref_on_right else 1.0 - p_right

                    van_rows.append({
                        "scenario_id": int(r.get("scenario_id", -1)),
                        "phenomenon_category": str(r.get("phenomenon_category", "default")),
                        "this_group_name": str(r.get("this_group_name", "")),
                        "preferred_on_right": pref_on_right,
                        "n_left": int(r.get("n_left", 0)),
                        "n_right": int(r.get("n_right", 0)),
                        "lang": str(r.get("lang", "en")),
                        "p_left": 1.0 - p_right,
                        "p_right": p_right,
                        "p_spare_preferred": p_pref,
                        "delta": float(delta),
                        "judge_choice": parsed["choice"],
                        "judge_confidence": float(parsed["confidence"]),
                        "judge_parse_ok": bool(parsed["parse_ok"]),
                    })

            dt = time.time() - t0
            n_total = max(1, stats["judged"] + stats["cached"])
            fail_pct = 100.0 * stats["parse_fail"] / n_total
            print(f"  judged={stats['judged']}  cached={stats['cached']}  "
                  f"parse_fail%={fail_pct:.1f}  t={dt:.1f}s")
            if fail_pct >= cfg.max_parse_fail_pct:
                print(f"[ERROR] {country} parse-fail rate {fail_pct:.1f}% "
                      f">= {cfg.max_parse_fail_pct:.1f}% — aborting this country")
                continue

            van_df = pd.DataFrame(van_rows)
            country_dir = van_root / country
            country_dir.mkdir(parents=True, exist_ok=True)
            van_df.to_csv(country_dir / "vanilla_results.csv", index=False)

            human_amce = (
                load_human_amce(cfg.human_amce_path, country)
                if cfg.human_amce_path else {}
            )
            van_summary = _summarize(van_df, human_amce)
            (country_dir / "summary.json").write_text(
                json.dumps({
                    "country": country,
                    "vanilla": van_summary,
                    "judge_stats": stats,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            compare_rows.append({
                "model": "qwen25_7b_openended_baseline_selfjudge",
                "judge": cfg.judge_model_name,
                "method": "vanilla_baseline_selfjudge",
                "country": country,
                **{f"align_{k}": v for k, v in van_summary.get("alignment", {}).items()},
                "n_scenarios": van_summary["n_scenarios"],
                "judge_parse_fail_pct": fail_pct,
            })

            van_align = van_summary.get("alignment", {})
            print(
                f"  [OK] {country}  "
                f"VAN MIS={van_align.get('mis', float('nan')):.4f}  "
                f"r={van_align.get('pearson_r', float('nan')):+.3f}  "
                f"JSD={van_align.get('jsd', float('nan')):.4f}  "
                f"n={van_summary['n_scenarios']}  parse_fail%={fail_pct:.1f}"
            )
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        del judge_model, judge_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv(cmp_root / "comparison.csv", index=False)
    print(f"\n[Stage 2] DONE — comparison at {cmp_root/'comparison.csv'}")
    if not cmp_df.empty:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY — VANILLA BASELINE  ({len(cmp_df)} country rows)")
        print(f"{'='*70}")
        cols = [c for c in (
            "country", "align_mis", "align_pearson_r", "align_jsd",
            "n_scenarios", "judge_parse_fail_pct",
        ) if c in cmp_df.columns]
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(cmp_df[cols].to_string(index=False))
        if "align_mis" in cmp_df.columns:
            print(
                f"\n[MEAN] MIS={cmp_df['align_mis'].mean():.4f}  "
                f"r={cmp_df['align_pearson_r'].mean():+.3f}  "
                f"JSD={cmp_df['align_jsd'].mean():.4f}"
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    stage = RUN_STAGE.strip().lower()
    if stage not in ("1", "2", "both"):
        raise ValueError(f"RUN_STAGE must be 1|2|both, got {stage!r}")

    results_base = (
        f"{WORK_DIR}/results/openended_baseline" if _on_kaggle() else
        str(Path(__file__).parent.parent / "results" / "openended_baseline")
    )
    jsonl_dir = f"{results_base}/stage1"

    cfg = BaselineConfig(
        actor_model_name=ACTOR_MODEL_PATH,
        judge_model_name=JUDGE_MODEL_PATH,
        out_jsonl_dir=jsonl_dir,
        results_base=results_base,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        countries=RUN_COUNTRIES,
        n_scenarios=RUN_N_SCENARIOS,
        max_new_tokens_actor=RUN_MAX_NEW_TOKENS_ACTOR,
        max_new_tokens_judge=RUN_MAX_NEW_TOKENS_JUDGE,
        seed=RUN_SEED,
        flush_every=RUN_FLUSH_EVERY,
        max_parse_fail_pct=RUN_MAX_PARSE_FAIL_PCT,
        use_real_data=os.path.isdir(MULTITP_DATA_PATH),
    )

    print(f"\n[BASELINE] stage={stage}  countries={cfg.countries}  n={cfg.n_scenarios}")
    print(f"[BASELINE] actor={cfg.actor_model_name}")
    print(f"[BASELINE] judge={cfg.judge_model_name}")
    print(f"[BASELINE] results_base={cfg.results_base}  jsonl_dir={cfg.out_jsonl_dir}")
    print(f"[BASELINE] max_new_tokens_actor={cfg.max_new_tokens_actor}  "
          f"max_new_tokens_judge={cfg.max_new_tokens_judge}  seed={cfg.seed}")

    if stage in ("1", "both"):
        run_stage1(cfg)
        gc.collect()

    if stage == "both":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if stage in ("2", "both"):
        run_stage2(cfg)


if __name__ == "__main__":
    main()
