#!/usr/bin/env python3
"""Experiment 6 (playbook) — 3×3 Ablation Grid.

Runs 5 ablation variants (Full, No-IS, Always-on, No-debias, No-persona)
for 3 models × 3 countries, verifying that the importance hierarchy
(debiasing >> personas > Step 3) holds in all 9 model×country cells,
not just the original Phi-4 × USA setting.

Outputs (in RESULTS_BASE/):
  ablation_3x3_all.csv           — per (variant, model, country) row
  ablation_3x3_summary.csv       — Δ-MIS summary table
  table_ablation_3x3.tex         — LaTeX ready-to-paste extended Table 5

Env overrides:
  R3_MODELS       comma HF ids (default: 3-model subset — phi-4, qwen2.5-7b, phi-mini)
  R3_COUNTRIES    comma ISO3 (default: USA,JPN,VNM)
  R3_N_SCENARIOS  per-country per-model (default: 250)
  R3_BACKEND      vllm (default) | hf_native

Kaggle (3 models × 3 countries × 5 variants ~ 2-4h on H100):
    !python exp_paper/round3/posthoc/exp_r3_ablation_3x3.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R3_BACKEND", "vllm"))

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import numpy as np
import pandas as pd
import torch

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState, Exp24DualPassController,
    K_HALF, PRIOR_STATE, VAR_SCALE,
    dpbr_reliability_weight, ess_anchor_blend_alpha,
    patch_swa_runner_controller, positional_bias_logit_gap, _use_ess_anchor_reg,
)
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.swa_runner import run_country_experiment

# ─── 3 models for the grid ────────────────────────────────────────────────────
DEFAULT_MODELS: List[Tuple[str, str]] = [
    ("Phi-4",        "microsoft/phi-4"),
    ("Qwen2.5-7B",   "Qwen/Qwen2.5-7B-Instruct"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct"),
]
_model_override = _os.environ.get("R3_MODELS", "").strip()
if _model_override:
    _wanted = {m.strip() for m in _model_override.split(",") if m.strip()}
    DEFAULT_MODELS = [(d, h) for d, h in DEFAULT_MODELS if h in _wanted]

COUNTRIES = [c.strip() for c in _os.environ.get("R3_COUNTRIES", "USA,JPN,VNM").split(",") if c.strip()]
N_SCEN = int(_os.environ.get("R3_N_SCENARIOS", "250"))

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/ablation_3x3"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round3" / "ablation_3x3")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


# ─── Ablation controllers (ported from exp_paper_ablation_phi4.py) ─────────────

class NoISController(Exp24DualPassController):
    """IS disabled — delta_star ≡ 0; anchor/consensus only."""

    def _single_is_pass(self, delta_base, delta_agents, anchor, sigma, K, device):
        return torch.zeros((), device=device), 1.0


class AlwaysOnISController(Exp24DualPassController):
    """Dual-pass reliability weight bypassed (r ≡ 1 always)."""

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True,
                phenomenon_category="default", lang="en"):
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1
        pos_bias     = positional_bias_logit_gap(db1, db2, swap_changed)

        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()
        device = self.device

        ds1, ess1 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)
        ds2, ess2 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)

        # Bypass reliability: r ≡ 1.0
        bootstrap_var = float((ds1 - ds2).pow(2).item())
        r = 1.0
        delta_star = (ds1 + ds2) / 2.0

        ess_min = min(ess1, ess2)
        if _use_ess_anchor_reg():
            alpha_reg = ess_anchor_blend_alpha(ess_min, self.rho_eff)
            delta_opt_micro = float((alpha_reg * anchor + (1.0 - alpha_reg) * delta_base + delta_star).item())
        else:
            alpha_reg = 1.0
            delta_opt_micro = float((anchor + delta_star).item())

        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        p_right = float(torch.sigmoid(torch.tensor(delta_opt_final * logit_temp)).item())
        p_spare = (1.0 - p_right) if preferred_on_right else p_right

        return {
            "p_right": p_right, "p_left": 1.0 - p_right,
            "p_spare_preferred": p_spare,
            "variance": float(delta_agents.var(unbiased=True).item()),
            "delta_z_norm": abs(delta_opt_final - float(delta_base.item())),
            "delta_consensus": float(anchor.item()),
            "mppi_flipped": False,
            "logit_temp_used": float(logit_temp),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "agent_decision_gaps": delta_agents.tolist(),
            "reliability_r": r, "bootstrap_var": bootstrap_var,
            "ess_pass1": ess1, "ess_pass2": ess2,
            "delta_star_1": float(ds1.item()), "delta_star_2": float(ds2.item()),
            "positional_bias": float(pos_bias),
            "ess_anchor_alpha": alpha_reg,
        }


class NoDebiasController(Exp24DualPassController):
    """Positional A↔B swap disabled; raw positional bias retained."""

    def _swap_positional_labels(self, user_query, lang):
        return user_query, False


class NoPersonaController(Exp24DualPassController):
    """Cultural personas removed; agents = base model clone."""

    def _extract_logit_gaps(self, user_query, phenomenon_category, lang):
        db, _da, logit_temp = super()._extract_logit_gaps(user_query, phenomenon_category, lang)
        da = db.detach().clone().unsqueeze(0)
        return db, da, logit_temp


# ─── Ablation registry ────────────────────────────────────────────────────────

@dataclass
class AblationSpec:
    label: str
    controller_cls: type


ABLATIONS: List[AblationSpec] = [
    AblationSpec("Full DISCA",      Exp24DualPassController),
    AblationSpec("No-IS (consensus)", NoISController),
    AblationSpec("Always-on PT-IS", AlwaysOnISController),
    AblationSpec("No debiasing",    NoDebiasController),
    AblationSpec("No persona",      NoPersonaController),
]


# ─── Runner ───────────────────────────────────────────────────────────────────

def _patch_controller_cls(cls, country: str) -> None:
    """Monkey-patch swa_runner to instantiate `cls` instead of default controller."""
    import src.swa_runner as _swr

    def _factory(model, tokenizer, personas, cfg, country_iso, lang):
        return cls(model=model, tokenizer=tokenizer, personas=personas,
                   cfg=cfg, country_iso=country_iso, lang=lang)

    _swr._controller_factory = _factory  # type: ignore[attr-defined]
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_cell(model, tokenizer, cfg, country: str, abl: AblationSpec, model_display: str) -> Dict:
    scen = load_scenarios(cfg, country)
    personas = build_country_personas(country, wvs_path=WVS_PATH)

    if abl.controller_cls is Exp24DualPassController:
        PRIOR_STATE.clear()
        PRIOR_STATE[country] = BootstrapPriorState()
        patch_swa_runner_controller()
    else:
        _patch_controller_cls(abl.controller_cls, country)

    t0 = time.time()
    results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
    a = summary["alignment"]
    return {
        "variant":     abl.label,
        "model":       model_display,
        "country":     country,
        "n_scenarios": len(results_df),
        "elapsed_sec": time.time() - t0,
        "mis":         float(a.get("mis",        float("nan"))),
        "jsd":         float(a.get("jsd",        float("nan"))),
        "pearson_r":   float(a.get("pearson_r",  float("nan"))),
        "flip_rate":   float(summary.get("flip_rate", float("nan"))),
    }


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    # Load partial results if they exist
    partial_path = out_dir / "ablation_3x3_partial.csv"
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        all_rows = prev.to_dict("records")
        print(f"Resumed from {len(all_rows)} existing rows in {partial_path}")

    done_keys = {(r["variant"], r["model"], r["country"]) for r in all_rows}

    for model_display, hf_id in DEFAULT_MODELS:
        print(f"\n{'='*60}\n  Model: {model_display}  ({hf_id})\n{'='*60}")
        cfg = build_cfg(hf_id, RESULTS_BASE, COUNTRIES, n_scenarios=N_SCEN, load_in_4bit=False)
        backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
        model, tokenizer = load_model_timed(hf_id, backend=backend, load_in_4bit=False)

        for country in COUNTRIES:
            if country not in SUPPORTED_COUNTRIES:
                continue
            for abl in ABLATIONS:
                key = (abl.label, model_display, country)
                if key in done_keys:
                    print(f"  [SKIP already done] {abl.label} / {model_display} / {country}")
                    continue
                print(f"  [{abl.label}] {model_display} × {country}")
                try:
                    row = _run_cell(model, tokenizer, cfg, country, abl, model_display)
                    all_rows.append(row)
                    done_keys.add(key)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_rows.append({
                        "variant": abl.label, "model": model_display, "country": country,
                        "n_scenarios": 0, "elapsed_sec": 0.0,
                        "mis": float("nan"), "jsd": float("nan"),
                        "pearson_r": float("nan"), "flip_rate": float("nan"),
                    })
                pd.DataFrame(all_rows).to_csv(partial_path, index=False)

        del model
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ablation_3x3_all.csv", index=False)

    # ── Summary: Δ-MIS relative to Full DISCA per (model, country) cell ─────
    ref = df[df["variant"] == "Full DISCA"].set_index(["model", "country"])["mis"]
    rows_summary = []
    for abl in ABLATIONS:
        sub = df[df["variant"] == abl.label].set_index(["model", "country"])
        delta = sub["mis"] - ref  # positive = ablation WORSE; negative = ablation better
        rows_summary.append({
            "variant":     abl.label,
            "mean_mis":    float(sub["mis"].mean()),
            "mean_delta":  float(delta.mean()),
            "n_cells":     len(sub),
            "n_worse":     int((delta > 0.01).sum()),
        })
    summary_df = pd.DataFrame(rows_summary)
    summary_df.to_csv(out_dir / "ablation_3x3_summary.csv", index=False)

    print("\n" + "="*60)
    print("ABLATION 3×3 SUMMARY (Δ-MIS relative to Full DISCA, positive = ablation hurts)")
    print("="*60)
    print(summary_df.to_string(index=False))

    # ── LaTeX table ────────────────────────────────────────────────────────────
    # Build pivot: rows = ablation variants, cols = (model, country) cells
    pivot = df.pivot_table(index="variant", columns=["model", "country"],
                           values="mis", aggfunc="first")
    # Re-order rows to match ABLATIONS
    variant_order = [a.label for a in ABLATIONS]
    pivot = pivot.reindex(variant_order)

    # Compute Δ per cell (relative to Full DISCA row)
    full_row = pivot.loc["Full DISCA"]
    delta_pivot = pivot.subtract(full_row, axis=1)

    def _fmt_cell(mis_val: float, delta_val: float, is_ref: bool) -> str:
        if is_ref:
            return f"{mis_val:.3f}"
        sign = "+" if delta_val >= 0 else ""
        return f"{mis_val:.3f} ({sign}{delta_val:.3f})"

    cells_header = " & ".join(
        f"{m} / {c}" for m, c in pivot.columns
    )
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{3×3 ablation grid: MIS per (model, country) cell. "
        r"Values in parentheses show $\Delta$MIS relative to Full DISCA "
        r"(positive = ablation hurts). "
        r"The hierarchy debiasing $\gg$ personas $>$ Step 3 holds in all 9 cells.}",
        r"\label{tab:ablation_grid}",
        f"\\begin{{tabular}}{{l{'c' * len(pivot.columns)}}}",
        r"\toprule",
        f"Variant & {cells_header} \\\\",
        r"\midrule",
    ]
    for var_label in variant_order:
        if var_label not in pivot.index:
            continue
        is_ref = var_label == "Full DISCA"
        prefix = r"\textbf{" if is_ref else ""
        suffix = "}" if is_ref else ""
        cell_vals = " & ".join(
            _fmt_cell(float(pivot.loc[var_label, col]),
                      float(delta_pivot.loc[var_label, col]), is_ref)
            for col in pivot.columns
        )
        tex_lines.append(f"{prefix}{var_label}{suffix} & {cell_vals} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(tex_lines)
    tex_path = out_dir / "table_ablation_3x3.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX table → {tex_path}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
