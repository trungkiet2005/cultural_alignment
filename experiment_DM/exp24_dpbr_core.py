"""
EXP-24 Dual-Pass Bootstrap IS — single source of truth
======================================================
Shared by:
  - experiment_DM/exp24_dual_pass_bootstrap.py  (multi-model Kaggle runner)
  - exp_model/_base_dpbr.py                   (per-model sweep + vanilla)

Do not duplicate this controller elsewhere; import from here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.controller import ImplicitSWAController

# EXP-09 hierarchical-prior hyperparameters (unchanged in EXP-24)
N_WARMUP = 50
DECAY_TAU = 100
BETA_EMA = 0.10

# EXP-24 dual-pass IS
K_HALF = 64  # samples per pass (2 × K_HALF = 128 = EXP-09 total K)
VAR_SCALE = 0.04  # r = exp(-bootstrap_var / VAR_SCALE)


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

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

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
        r = float(np.exp(-bootstrap_var / VAR_SCALE))
        delta_star = r * (ds1 + ds2) / 2.0

        delta_opt_micro = float((anchor + delta_star).item())
        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

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
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


def patch_swa_runner_controller() -> None:
    """Install Exp24DualPassController as ImplicitSWAController for swa_runner."""
    import src.swa_runner as _swa_runner_mod

    _swa_runner_mod.ImplicitSWAController = Exp24DualPassController
