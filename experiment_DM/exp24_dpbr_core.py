"""
EXP-24 Dual-Pass Bootstrap IS — single source of truth
======================================================
Shared by:
  - experiment_DM/exp24_dual_pass_bootstrap.py  (multi-model Kaggle runner)
  - exp_model/_base_dpbr.py                   (per-model sweep + vanilla)

Do not duplicate this controller elsewhere; import from here.

Ablations: set ``EXP24_VAR_SCALE`` / ``EXP24_K_HALF`` in the environment **before**
importing this module (see ``docs/exp24_reproducibility.md``).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.controller import ImplicitSWAController

# EXP-09 hierarchical-prior hyperparameters (unchanged in EXP-24)
N_WARMUP = 50
DECAY_TAU = 100
BETA_EMA = 0.10

# EXP-24 dual-pass IS — override for ablations via env (before first import of this module)
K_HALF = int(os.environ.get("EXP24_K_HALF", "64"))  # samples per pass (2 × K_HALF = EXP-09 K)
VAR_SCALE = float(os.environ.get("EXP24_VAR_SCALE", "0.04"))  # r = exp(-bootstrap_var / VAR_SCALE)


def dpbr_reliability_weight(
    delta_star_1: float,
    delta_star_2: float,
    var_scale: Optional[float] = None,
) -> float:
    """Paper-facing helper: r = exp(-(δ*₁-δ*₂)² / s) with bootstrap_var = (δ*₁-δ*₂)²."""
    s = float(var_scale if var_scale is not None else VAR_SCALE)
    bv = (delta_star_1 - delta_star_2) ** 2
    return float(np.exp(-bv / s))


class BootstrapPriorState:
    """Minimal EXP-09 country prior (scalar EMA + annealed blend)."""

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


PRIOR_STATE: Dict[str, BootstrapPriorState] = {}


class Exp24DualPassController(ImplicitSWAController):
    """
    EXP-09 with Dual-Pass Bootstrap IS Reliability Filter.

    Two changes vs EXP-09:
    1. IS split into two independent passes (K_HALF each, same total K=128).
    2. Soft reliability: r = exp(-bootstrap_var / VAR_SCALE),
       delta_star = r · (delta_star_1 + delta_star_2) / 2
    """

    def __init__(self, *args, country_iso: str = "UNKNOWN", **kwargs):
        super().__init__(*args, country_iso=country_iso, **kwargs)
        self.country = country_iso  # prior-state key (EXP-09 hierarchical prior)

    def _get_prior(self) -> BootstrapPriorState:
        if self.country not in PRIOR_STATE:
            PRIOR_STATE[self.country] = BootstrapPriorState()
        return PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        K: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent = self._pt_value(g_per_agent)
        mean_v = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)
        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_r = float(k_eff.item()) / K

        delta_star = (
            torch.sum(w * eps) if ess_r >= self.rho_eff else torch.zeros((), device=device)
        )
        return delta_star, ess_r

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1
        delta_base = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

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

        delta_opt_micro = float((anchor + delta_star).item())
        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        # Diagnostics: micro-only decisions from each IS pass (no hierarchical prior). For analysis / ablations.
        def _p_pref_micro(d_s: torch.Tensor) -> float:
            dm = float((anchor + d_s).item())
            pr = torch.sigmoid(torch.tensor(dm / self.decision_temperature)).item()
            return pr if preferred_on_right else 1.0 - pr

        p_pref_micro_1 = _p_pref_micro(ds1)
        p_pref_micro_2 = _p_pref_micro(ds2)

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_star_1": float(ds1.item()),
            "delta_star_2": float(ds2.item()),
            "bootstrap_var": bootstrap_var,
            "reliability_r": r,
            "ess_pass1": ess1,
            "ess_pass2": ess2,
            "delta_country": st["delta_country"],
            "alpha_h": st["alpha_h"],
            "prior_step": st["step"],
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_is_pass1_micro": p_pref_micro_1,
            "p_spare_preferred_is_pass2_micro": p_pref_micro_2,
            "positional_bias": 0.0,
        }


def patch_swa_runner_controller() -> None:
    """Install Exp24DualPassController as ImplicitSWAController for swa_runner."""
    import src.swa_runner as _swa_runner_mod

    _swa_runner_mod.ImplicitSWAController = Exp24DualPassController
