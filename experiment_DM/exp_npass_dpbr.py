"""
EXP-24 N-pass ablation: generalise dual-pass DPBR to N independent IS passes
============================================================================

Two pieces:

1. ``LoggingDPBRController`` — runs the standard EXP-24 dual-pass DPBR
   (so its in-process numbers match the canonical Phi-3.5 run exactly), but
   *additionally* records per-scenario debiased logit gaps to an in-memory
   list. The driver flushes that list to a parquet logit-cache.

2. ``replay_country_npass`` — pure-numpy replay. Given the cached
   debiased logit gaps for one country, runs N independent IS passes per
   scenario and combines via the generalised DPBR rule:

       bootstrap_var = 2 * Var_ddof1(delta_star_1, ..., delta_star_N)
       r             = exp(-bootstrap_var / VAR_SCALE)
       delta_star    = r * mean(delta_star_1, ..., delta_star_N)

   At N=2 this reduces *exactly* to the paper:
   2 * Var_ddof1(a, b) = (a-b)^2.
   At N=1 there is no bootstrap variance: r = 1.

The cache lets a single GPU sweep across 20 countries feed every
N variant (N=1..N_max) at near-zero additional cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    Exp24DualPassController,
    K_HALF,
    PERSONA_FLOOR,
    PRIOR_STATE,
    VAR_SCALE,
    _use_ess_anchor_reg,
    dpbr_reliability_weight,
    ess_anchor_blend_alpha,
    positional_bias_logit_gap,
)


# ─── Generalised DPBR rule ───────────────────────────────────────────────────
def npass_bootstrap_var(delta_stars: Sequence[float]) -> float:
    """Bootstrap variance for N IS passes.

    Formula: ``2 * sample_var_ddof1(delta_stars)``. Equals ``(a-b)^2`` at N=2,
    matching the paper's dual-pass definition. Returns 0.0 for N<2.
    """
    n = len(delta_stars)
    if n < 2:
        return 0.0
    arr = np.asarray(delta_stars, dtype=np.float64)
    return 2.0 * float(np.var(arr, ddof=1))


def npass_reliability_weight(
    delta_stars: Sequence[float],
    var_scale: Optional[float] = None,
) -> float:
    """``r = exp(-bootstrap_var / VAR_SCALE)``. For N<2 returns 1.0 (no
    bootstrap defined → trust the single IS pass)."""
    if len(delta_stars) < 2:
        return 1.0
    s = float(var_scale if var_scale is not None else VAR_SCALE)
    return float(np.exp(-npass_bootstrap_var(delta_stars) / s))


# ─── Logit cache I/O ─────────────────────────────────────────────────────────
def cache_path_for(out_dir: Path, country: str) -> Path:
    return Path(out_dir) / f"logit_cache_{country}.parquet"


def save_logit_cache(out_dir: Path, country: str, df: pd.DataFrame) -> None:
    p = cache_path_for(out_dir, country)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def load_logit_cache(out_dir: Path, country: str) -> Optional[pd.DataFrame]:
    p = cache_path_for(out_dir, country)
    return pd.read_parquet(p) if p.is_file() else None


# ─── Logging controller (caches debiased logit gaps during normal DPBR run) ─
class LoggingDPBRController(Exp24DualPassController):
    """Standard EXP-24 dual-pass DPBR + per-scenario logit recording.

    The recorded values are the *debiased* logit gaps used by the IS sampler:
    ``delta_base, delta_agents, positional_bias, logit_temp``. These are
    deterministic functions of the model's two forward passes (original +
    A↔B-swapped prompt), so replay across any N reuses them exactly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_records: List[Dict] = []
        self._scenario_meta: Dict = {}

    def attach_meta(self, meta: Dict) -> None:
        """Set scenario-level metadata stamped onto the next predict() record."""
        self._scenario_meta = dict(meta)

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # --- Extraction (identical to Exp24DualPassController.predict) ---
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1
        delta_base = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1
        pb = positional_bias_logit_gap(db1, db2, swap_changed)

        # --- CACHE: record post-debias logit gaps (one entry per predict call) ---
        rec = {
            "delta_base": float(delta_base.item()),
            "delta_agents": json.dumps([float(x) for x in delta_agents.tolist()]),
            "positional_bias": float(pb),
            "logit_temp": float(logit_temp),
            **(dict(self._scenario_meta) if self._scenario_meta else {}),
        }
        self._cache_records.append(rec)

        # --- Standard EXP-24 dual-pass IS (re-uses the already-extracted tensors) ---
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()
        device = self.device
        ds1, ess1 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)
        ds2, ess2 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)

        bootstrap_var = float((ds1 - ds2).pow(2).item())
        r = dpbr_reliability_weight(float(ds1.item()), float(ds2.item()))
        delta_star = r * (ds1 + ds2) / 2.0

        ess_min = min(ess1, ess2)
        if _use_ess_anchor_reg():
            alpha_reg = ess_anchor_blend_alpha(ess_min, self.rho_eff)
            delta_opt_micro = float(
                (alpha_reg * anchor + (1.0 - alpha_reg) * delta_base + delta_star).item()
            )
        else:
            alpha_reg = 1.0
            delta_opt_micro = float((anchor + delta_star).item())

        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": delta_opt_final, "delta_opt_micro": delta_opt_micro,
            "delta_star_1": float(ds1.item()), "delta_star_2": float(ds2.item()),
            "bootstrap_var": bootstrap_var, "reliability_r": r,
            "ess_pass1": ess1, "ess_pass2": ess2,
            "ess_anchor_alpha": alpha_reg, "ess_anchor_reg_enabled": _use_ess_anchor_reg(),
            "delta_country": st["delta_country"], "alpha_h": st["alpha_h"],
            "prior_step": st["step"], "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "positional_bias": float(pb),
        }


def patch_swa_runner_with_logger() -> "LoggingDPBRController":
    """Install ``LoggingDPBRController`` for ``swa_runner.run_country_experiment``.

    Returns the class object so callers can later read ``_cache_records`` from
    the live controller instance via ``swa_runner``-internal hooks if needed.
    In practice, the runner constructs the controller from the class object,
    so we capture the cache by inspecting the *constructed* instance via a
    weak reference set in ``__init__``."""
    import src.swa_runner as _swa_runner_mod
    _swa_runner_mod.ImplicitSWAController = LoggingDPBRController
    return LoggingDPBRController


# ─── Replay (no model, pure numpy) ───────────────────────────────────────────
def _is_pass_numpy(
    delta_base: float,
    delta_agents: np.ndarray,
    anchor: float,
    sigma: float,
    K: int,
    *,
    lambda_coop: float,
    pt_alpha: float,
    pt_beta: float,
    pt_kappa: float,
    beta_temp: float,
    rho_eff: float,
    persona_floor: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Numpy translation of ``Exp24DualPassController._single_is_pass``."""
    eps = rng.standard_normal(K) * sigma
    delta_tilde = anchor + eps                                         # (K,)

    dist_base_to_i = np.abs(delta_base - delta_agents)                 # (N,)
    dist_cand_to_i = np.abs(delta_tilde[:, None] - delta_agents[None, :])  # (K,N)
    g_per_agent = (dist_base_to_i[None, :] - dist_cand_to_i) / sigma

    def _pt(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, np.abs(x) ** pt_alpha,
                        -pt_kappa * np.abs(x) ** pt_beta)

    v = _pt(g_per_agent)
    if persona_floor > 0:
        v = np.maximum(v, -persona_floor)
    mean_v = v.mean(axis=1)

    g_cons = (np.abs(delta_base - anchor) - np.abs(delta_tilde - anchor)) / sigma
    v_cons = _pt(g_cons)
    U = (1.0 - lambda_coop) * mean_v + lambda_coop * v_cons
    U_shift = U - U.max()
    w = np.exp(U_shift / beta_temp)
    w = w / w.sum()
    k_eff = 1.0 / max(float(np.sum(w * w)), 1e-12)
    ess_r = k_eff / K
    ds = float(np.sum(w * eps)) if ess_r >= rho_eff else 0.0
    return ds, ess_r


def replay_country_npass(
    cache_df: pd.DataFrame,
    country_iso: str,
    n_passes: int,
    *,
    K_per_pass: int = K_HALF,
    var_scale: float = VAR_SCALE,
    lambda_coop: float = 0.70,
    pt_alpha: float = 0.88,
    pt_beta: float = 0.88,
    pt_kappa: float = 2.25,
    beta_temp: float = 0.5,
    rho_eff: float = 0.1,
    decision_temperature: float = 1.0,
    noise_std_floor: float = 0.3,
    persona_floor: float = PERSONA_FLOOR,
    use_ess_anchor_reg: Optional[bool] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Replay N IS passes for a single country from cached debiased logits.

    Returns a results_df with the columns ``compute_amce_from_preferences``
    and the surrounding swa_runner reporting code expect, so it can be fed
    straight into the same metrics + per-dimension breakdown pipeline.
    """
    if use_ess_anchor_reg is None:
        use_ess_anchor_reg = _use_ess_anchor_reg()

    # Per-country deterministic seed (independent across countries, reproducible
    # across Python sessions — Python's built-in hash() is randomised by default).
    country_digest = int.from_bytes(
        hashlib.sha256(country_iso.encode("utf-8")).digest()[:4], "little"
    )
    seed_country = (seed * 1_000_003 + country_digest) % (2**32 - 1)
    rng = np.random.default_rng(seed_country)

    # Fresh hierarchical-prior state for this country / N value.
    PRIOR_STATE.pop(country_iso, None)
    PRIOR_STATE[country_iso] = BootstrapPriorState()
    prior = PRIOR_STATE[country_iso]

    rows = []
    for ridx, row in cache_df.iterrows():
        delta_base = float(row["delta_base"])
        delta_agents = np.asarray(json.loads(row["delta_agents"]), dtype=np.float64)
        if delta_agents.size >= 2:
            sigma = max(float(np.std(delta_agents, ddof=1)), noise_std_floor)
        else:
            sigma = noise_std_floor
        anchor = float(np.mean(delta_agents))

        ds_list: List[float] = []
        ess_list: List[float] = []
        for _ in range(n_passes):
            ds, ess_r = _is_pass_numpy(
                delta_base, delta_agents, anchor, sigma, K_per_pass,
                lambda_coop=lambda_coop, pt_alpha=pt_alpha, pt_beta=pt_beta,
                pt_kappa=pt_kappa, beta_temp=beta_temp, rho_eff=rho_eff,
                persona_floor=persona_floor, rng=rng,
            )
            ds_list.append(ds)
            ess_list.append(ess_r)

        bv = npass_bootstrap_var(ds_list)
        r = npass_reliability_weight(ds_list, var_scale=var_scale)
        delta_star = r * float(np.mean(ds_list)) if ds_list else 0.0
        ess_min = float(min(ess_list)) if ess_list else 0.0

        if use_ess_anchor_reg:
            alpha_reg = float(min(1.0, max(rho_eff, ess_min)))
            delta_opt_micro = alpha_reg * anchor + (1.0 - alpha_reg) * delta_base + delta_star
        else:
            alpha_reg = 1.0
            delta_opt_micro = anchor + delta_star

        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)

        p_right = float(1.0 / (1.0 + np.exp(-delta_opt_final / decision_temperature)))
        pref_right = bool(int(row.get("preferred_on_right", 1)))
        p_pref = p_right if pref_right else 1.0 - p_right

        rows.append({
            "country": country_iso,
            "scenario_idx": int(row.get("scenario_idx", ridx)),
            "Prompt": row.get("Prompt", ""),
            "phenomenon_category": row.get("phenomenon_category", "default"),
            "this_group_name": row.get("this_group_name", "Unknown"),
            "preferred_on_right": int(pref_right),
            "n_left": int(row.get("n_left", 1)),
            "n_right": int(row.get("n_right", 1)),
            "p_left": 1.0 - p_right,
            "p_right": p_right,
            "p_spare_preferred": p_pref,
            "delta_consensus": anchor,
            "delta_opt": float(delta_opt_final),
            "delta_opt_micro": float(delta_opt_micro),
            "reliability_r": r,
            "bootstrap_var": bv,
            "n_passes": int(n_passes),
            "ess_min": ess_min,
            "ess_anchor_alpha": alpha_reg,
            "logit_temp_used": float(row.get("logit_temp", 1.0)),
            "positional_bias": float(row.get("positional_bias", 0.0)),
            "mppi_flipped": (anchor > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - anchor),
            "mppi_variance": float(np.var(delta_agents, ddof=1)) if delta_agents.size > 1 else 0.0,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
        })
    return pd.DataFrame(rows)
