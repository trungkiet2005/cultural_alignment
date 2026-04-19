"""Offline DPBR controller that accepts pre-computed pseudo-deltas.

In the logit-based pipeline, :class:`experiment_DM.exp24_dpbr_core.Exp24DualPassController`
runs two forward passes of the actor (pass1 + A-B-swapped pass2), extracts
logit gaps, debiases, and then runs the dual-pass bootstrap IS + hierarchical
prior math. In the open-ended variant those two passes have already happened
OUT OF PROCESS (Stage 1) and a judge has converted each generation to a
pseudo-logit-gap via :func:`src.pseudo_delta.pseudo_delta_from_judge`.

This subclass bypasses the actor-forward half and exposes
:meth:`predict_from_deltas`, which mirrors
:meth:`Exp24DualPassController.predict` from line 179 onward (sigma, anchor,
``_single_is_pass`` x2, reliability, ess-anchor blend, prior apply/update).
Output schema matches the parent's ``predict`` dict so the rest of the
reporting pipeline (CSV writers, AMCE, alignment metrics) can reuse the same
column names.

Invariant: when fed with true debiased logit gaps and the same RNG state,
``predict_from_deltas`` returns a dict numerically identical to the parent's
``predict``. See ``exp_paper/openended/test_offline_parity.py`` (if present)
for the parity test spec.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    Exp24DualPassController,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
    _use_ess_anchor_reg,
    dpbr_reliability_weight,
    ess_anchor_blend_alpha,
)


class Exp24DualPassControllerOffline(Exp24DualPassController):
    """DPBR controller that consumes pre-computed (debiased) scalar deltas.

    Does NOT load a model or tokenizer. Holds only the scalar hyper-parameters
    used by :meth:`Exp24DualPassController._single_is_pass` /
    :meth:`Exp24DualPassController._pt_value` and the country's prior state.
    """

    def __init__(
        self,
        country_iso: str,
        *,
        lambda_coop: float = 0.70,
        decision_temperature: float = 1.0,
        noise_std: float = 0.30,
        beta: float = 0.50,
        rho_eff: float = 0.10,
        pt_alpha: float = 0.88,
        pt_beta: float = 0.88,
        pt_kappa: float = 2.25,
        device: Optional[torch.device] = None,
    ):
        # Skip the full ImplicitSWAController.__init__ chain (model/tokenizer/
        # persona prefixes / decision-token resolution) — none of it is used by
        # the pure-math path we execute here.
        self.country = country_iso
        self.country_iso = country_iso
        self.lambda_coop = float(lambda_coop)
        self.decision_temperature = float(decision_temperature)
        self.noise_std = float(noise_std)
        self.beta = float(beta)
        self.rho_eff = float(rho_eff)
        self.pt_alpha = float(pt_alpha)
        self.pt_beta = float(pt_beta)
        self.pt_kappa = float(pt_kappa)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        # Diagnostics parity with the online controller.
        self.logit_temperature = 1.0
        self.category_logit_temperatures: Dict[str, float] = {}

        if self.country not in PRIOR_STATE:
            PRIOR_STATE[self.country] = BootstrapPriorState()

    @torch.no_grad()
    def predict_from_deltas(
        self,
        delta_base_deb: float,
        delta_agents_deb,  # iterable or tensor of length N
        *,
        preferred_on_right: int = 1,
        phenomenon_category: str = "default",
        positional_bias: float = 0.0,
        swap_changed: bool = True,
    ) -> Dict:
        """Run DPBR given pre-computed debiased deltas. Mirrors parent predict()."""
        device = self.device
        delta_base = torch.tensor(float(delta_base_deb), dtype=torch.float32, device=device)
        if isinstance(delta_agents_deb, torch.Tensor):
            delta_agents = delta_agents_deb.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            delta_agents = torch.tensor(
                list(delta_agents_deb), dtype=torch.float32, device=device
            )

        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()

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

        p_right = float(
            torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        )
        p_pref = p_right if preferred_on_right else 1.0 - p_right
        variance = (
            float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0
        )

        def _p_pref_micro(d_s: torch.Tensor) -> float:
            dm = float((anchor + d_s).item())
            pr = float(torch.sigmoid(torch.tensor(dm / self.decision_temperature)).item())
            return pr if preferred_on_right else 1.0 - pr

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
            "ess_anchor_alpha": alpha_reg,
            "ess_anchor_reg_enabled": _use_ess_anchor_reg(),
            "delta_country": st["delta_country"],
            "alpha_h": st["alpha_h"],
            "prior_step": st["step"],
            "logit_temp_used": self.logit_temperature,
            "n_personas": int(delta_agents.numel()),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_is_pass1_micro": _p_pref_micro(ds1),
            "p_spare_preferred_is_pass2_micro": _p_pref_micro(ds2),
            "positional_bias": float(positional_bias),
            "swap_changed": bool(swap_changed),
        }
