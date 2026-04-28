"""Vanilla-anchored bounded blend with hard safety gates.

Wraps a candidate SWA-DPBR correction (δ_swa) around a vanilla baseline (δ_van)
so that the per-scenario decision can never drift far from vanilla and can
never flip sign on confidently-vanilla scenarios.

Final scalar:
    δ_final = (1 − α) · δ_van + α · δ_swa,    α ∈ [0, ALPHA_MAX]

α is forced to 0 (⇒ identical to vanilla) if ANY of four safety gates fail:
    1. Sign agreement on confident vanilla:
       sign(δ_swa) == sign(δ_van)  OR  |δ_van| < min_vanilla_conf
    2. DPBR reliability:
       reliability_r >= dpbr_r_min
    3. Magnitude bound:
       |δ_swa − δ_van| <= magnitude_ratio_max · max(|δ_van|, floor)
    4. Persona consensus:
       std(persona_deltas) <= persona_std_max

When all 4 gates pass, α scales smoothly with reliability:
    α = alpha_max · clip(reliability_r, 0, 1)

Per-country roll-up: if mean α across the country's scenarios is below
country_min_alpha, the gates fired on too many scenarios — there is no SWA
signal worth committing, so the country reverts entirely to vanilla. This is
the third (country-level) safety net.

Empirically motivated bounds (from tracker_open_ended.md cross-scale data):
- ALPHA_MAX=0.30: SWA at most 30% influence; vanilla anchor dominates.
- DPBR_R_MIN=0.85: above the threshold where 7B DEU was destroyed
  (Δr=−0.922 happened with rel_r=0.943, so we also need other gates).
- MAGNITUDE_RATIO_MAX=2.5: cuts off catastrophic VNM-3B style flips
  (|δ_swa − δ_van| ≈ 4.0 vs |δ_van| ≈ 0.06 — ratio 67×, easily blocked).
- PERSONA_STD_MAX=3.0: above this the ensemble is in disagreement and
  PT-IS softmax weights have no clean winner.
- COUNTRY_MIN_ALPHA=0.05: requires SWA to actually take effect on >~17% of
  scenarios (since alpha_max=0.30 → mean α=0.05 ≈ 1/6 active rate). Below
  this, the ensemble had no leverage on the country and we report vanilla.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SafeBlendConfig:
    """Conservative defaults; see module docstring for empirical motivation."""
    alpha_max: float = 0.30
    dpbr_r_min: float = 0.85
    min_vanilla_conf: float = 0.5
    magnitude_ratio_max: float = 2.5
    floor: float = 0.5
    persona_std_max: float = 3.0
    country_min_alpha: float = 0.05


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def safe_blend_scalar(
    delta_van: float,
    delta_swa: float,
    diag: Dict[str, float],
    cfg: SafeBlendConfig = SafeBlendConfig(),
) -> Tuple[float, float, Dict[str, bool]]:
    """Apply vanilla-anchored bounded blend with hard safety gates.

    Args:
        delta_van: vanilla baseline scalar (continuous logit gap from base persona).
        delta_swa: candidate SWA-DPBR corrected scalar.
        diag: per-scenario diagnostics; expected keys include:
              - "reliability_r": DPBR bootstrap reliability (0..1)
              - "persona_std": std of the per-persona deltas
        cfg: SafeBlendConfig; defaults are conservative.

    Returns:
        delta_final: blended scalar, on the same scale as delta_van.
        alpha_used: blend weight actually applied (0 ⇒ pure vanilla).
        gate_flags: dict of {"sign", "dpbr", "magnitude", "consensus"} → pass(True)/fail(False).
    """
    flags = {"sign": True, "dpbr": True, "magnitude": True, "consensus": True}

    sign_van = _sign(delta_van)
    sign_swa = _sign(delta_swa)
    if abs(delta_van) > cfg.min_vanilla_conf and sign_van != 0 and sign_swa != 0 and sign_swa != sign_van:
        flags["sign"] = False

    rel_r = float(diag.get("reliability_r", 0.0))
    if rel_r < cfg.dpbr_r_min:
        flags["dpbr"] = False

    bound = cfg.magnitude_ratio_max * max(abs(delta_van), cfg.floor)
    if abs(delta_swa - delta_van) > bound:
        flags["magnitude"] = False

    persona_std = float(diag.get("persona_std", 0.0))
    if persona_std > cfg.persona_std_max:
        flags["consensus"] = False

    if not all(flags.values()):
        return float(delta_van), 0.0, flags

    rel_r_clamped = max(0.0, min(1.0, rel_r))
    alpha = float(cfg.alpha_max * rel_r_clamped)
    delta_final = float((1.0 - alpha) * delta_van + alpha * delta_swa)
    return delta_final, alpha, flags


def country_level_decision(
    alphas: List[float], cfg: SafeBlendConfig = SafeBlendConfig(),
) -> Tuple[bool, float]:
    """Decide whether the country should commit blended SWA or revert to vanilla.

    Args:
        alphas: list of per-scenario alpha_used values (0 ⇒ that scenario was vanilla).
        cfg: SafeBlendConfig.

    Returns:
        commit_swa: True if mean α >= country_min_alpha (commit blended), else False.
        mean_alpha: the mean α observed.
    """
    if not alphas:
        return False, 0.0
    mean_alpha = float(sum(alphas) / len(alphas))
    return (mean_alpha >= cfg.country_min_alpha), mean_alpha
