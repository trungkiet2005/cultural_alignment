"""
EXP-24 Dual-Pass Bootstrap IS — shared base (DPBR)
====================================================
Core algorithm + runner reused by every per-model file.
NOT self-contained: env bootstrap must have run before this is imported.
"""

import gc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    print_tracker_ready_report,
    try_load_reference_comparison,
)
from src.config import SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.controller import ImplicitSWAController
from src.data import load_multitp_dataset
from src.model import load_model, setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.scenarios import generate_multitp_scenarios
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment

# ─── Shared hyperparameters (EXP-09 base + EXP-24 dual-pass) ─────────────────
EXP_BASE      = "EXP-24"
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]

N_SCENARIOS  = 500
BATCH_SIZE   = 1
SEED         = 42
LAMBDA_COOP  = 0.70

N_WARMUP     = 50
DECAY_TAU    = 100
BETA_EMA     = 0.10

K_HALF       = 64       # K1=K2=64 → total K=128 (same as EXP-09)
VAR_SCALE    = 0.04     # r = exp(-bootstrap_var / VAR_SCALE)

RESULTS_BASE = "/kaggle/working/cultural_alignment/results/exp24_model_sweep"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ─── Prior state (reset per country run) ──────────────────────────────────────
class BootstrapPriorState:
    def __init__(self) -> None:
        self.delta_country = 0.0
        self.step = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < N_WARMUP:
            return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, BootstrapPriorState] = {}


# ─── Dual-Pass Bootstrap IS Controller ────────────────────────────────────────
class Exp24DualPassController(ImplicitSWAController):
    """EXP-09 + Dual-Pass Bootstrap IS Reliability Filter."""

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> BootstrapPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = BootstrapPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        k_samples: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        eps          = torch.randn(k_samples, device=device) * sigma
        delta_tilde  = anchor + eps
        dist_base    = (delta_base - delta_agents).abs()
        dist_cand    = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_agents     = (dist_base.unsqueeze(0) - dist_cand) / sigma
        mean_v       = self._pt_value(g_agents).mean(dim=1)
        g_cons       = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        u            = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * self._pt_value(g_cons)
        w            = F.softmax(u / self.beta, dim=0)
        k_eff        = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_r        = float(k_eff.item()) / k_samples
        delta_star   = torch.sum(w * eps) if ess_r >= self.rho_eff else torch.zeros((), device=device)
        return delta_star, ess_r

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1
        sigma  = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()
        device = self.device

        ds1, ess1     = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)
        ds2, ess2     = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)
        bvar          = float((ds1 - ds2).pow(2).item())
        rel_r         = float(np.exp(-bvar / VAR_SCALE))
        delta_star    = rel_r * (ds1 + ds2) / 2.0

        delta_opt_micro = float((anchor + delta_star).item())
        prior           = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_star_1": float(ds1.item()), "delta_star_2": float(ds2.item()),
            "bootstrap_var": bvar, "reliability_r": rel_r,
            "ess_pass1": ess1, "ess_pass2": ess2,
            "delta_country": st["delta_country"], "alpha_h": st["alpha_h"],
            "prior_step": st["step"], "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp24DualPassController


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]:
        if not root:
            continue
        hub = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target = os.path.join(hub, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as exc:
                print(f"[CLEANUP] error: {exc}")


def _build_cfg(model_name: str, swa_root: str) -> SWAConfig:
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=swa_root,
        lambda_coop=LAMBDA_COOP, K_samples=128,
    )


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


# ─── Core run function ─────────────────────────────────────────────────────────
def run_for_model(model_name: str, model_short: str) -> None:
    """
    Full EXP-24 run for a single model.
    Called directly from each per-model entry script.
    """
    setup_seeds(SEED)

    exp_id   = f"{EXP_BASE}-{model_short.upper()}"
    swa_root = f"{RESULTS_BASE}/{model_short}/swa"
    cmp_root = f"{RESULTS_BASE}/{model_short}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {exp_id}: Dual-Pass Bootstrap IS [{model_name}]")
    print(f"{'='*70}")
    print(f"[THEORY] K_half={K_HALF}×2={K_HALF*2} total  |  VAR_SCALE={VAR_SCALE}")
    print(f"[THEORY] r = exp(-(δ*₁-δ*₂)² / {VAR_SCALE})  →  δ* = r·(δ*₁+δ*₂)/2")

    cfg     = _build_cfg(model_name, swa_root)
    out_dir = Path(swa_root) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)

    rows: List[dict] = []
    try:
        for ci, country in enumerate(TARGET_COUNTRIES):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue
            _PRIOR_STATE.clear()
            _PRIOR_STATE[country] = BootstrapPriorState()
            print(f"\n[{ci+1}/{len(TARGET_COUNTRIES)}] {exp_id} | {country}")

            scen     = _load_scen(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

            orig_init = Exp24DualPassController.__init__
            def patched_init(self, *a, country=country, **kw):
                orig_init(self, *a, country=country, **kw)
            Exp24DualPassController.__init__ = patched_init
            _swa_runner_mod.ImplicitSWAController = Exp24DualPassController

            results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
            Exp24DualPassController.__init__ = orig_init

            results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
            append_rows_csv(
                str(Path(cmp_root) / "per_dim_breakdown.csv"),
                flatten_per_dim_alignment(
                    summary.get("per_dimension_alignment", {}),
                    model=model_name, method=f"{exp_id}_dual_pass", country=country,
                ),
            )
            ps  = _PRIOR_STATE.get(country, BootstrapPriorState()).stats
            mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")
            rows.append({
                "model": model_name, "method": f"{exp_id}_dual_pass", "country": country,
                **{f"align_{k}": v for k, v in summary["alignment"].items()},
                "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
                "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
                "mean_reliability_r": mea("reliability_r"),
                "mean_bootstrap_var": mea("bootstrap_var"),
                "mean_ess_pass1": mea("ess_pass1"), "mean_ess_pass2": mea("ess_pass2"),
            })

            pda = summary.get("per_dimension_alignment", {})
            if pda:
                print(f"\n  ┌── Per-Dimension ({country}) ──")
                for dk, dd in sorted(pda.items()):
                    hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                    print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
                print(f"  └── MIS={summary['alignment']['mis']:.4f}  "
                      f"r={summary['alignment']['pearson_r']:+.3f}  "
                      f"Flip={summary['flip_rate']:.1%}  "
                      f"rel_r(avg)={mea('reliability_r'):.3f}")

            torch.cuda.empty_cache()
            gc.collect()

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _free_model_cache(model_name)

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(Path(cmp_root) / "comparison.csv", index=False)

    print(f"\n{'#'*70}\n# {exp_id} FINAL REPORT\n{'#'*70}")
    print_alignment_table(cmp_df, title=f"{exp_id} RESULTS")
    if not cmp_df.empty:
        print(
            f"  MEAN MIS={cmp_df['align_mis'].mean():.4f}  "
            f"r={cmp_df['align_pearson_r'].mean():+.3f}  "
            f"Flip={cmp_df['flip_rate'].mean():.1%}  "
            f"rel_r={cmp_df['mean_reliability_r'].mean():.3f}"
        )
        print(f"  (EXP-09 SOTA: 0.3975  |  EXP-24 SOTA: 0.3969)")
    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref, cmp_df, title=f"{exp_id} vs EXP-01",
            spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                             cur_method=f"{exp_id}_dual_pass"),
        )
    print_tracker_ready_report(
        cmp_df, exp_id=exp_id,
        per_dim_csv_path=str(Path(cmp_root) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{exp_id}] DONE — {cmp_root}")
