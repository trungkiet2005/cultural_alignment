"""
Unit tests for EXP-24 Dual-Pass Bootstrap IS (DPBR) core math.

Run from repo root:
    pip install -r requirements-dev.txt
    pytest tests/test_exp24_dpbr.py -q
"""

from __future__ import annotations

import importlib
import math
import os

import numpy as np
import pytest


def test_dpbr_reliability_perfect_agreement():
    from experiment_DM import exp24_dpbr_core as m

    r = m.dpbr_reliability_weight(0.1, 0.1, var_scale=0.04)
    assert r == pytest.approx(1.0)


def test_dpbr_reliability_disagreement():
    from experiment_DM import exp24_dpbr_core as m

    s = 0.04
    d1, d2 = 0.0, 0.2
    bv = (d1 - d2) ** 2
    r = m.dpbr_reliability_weight(d1, d2, var_scale=s)
    assert r == pytest.approx(math.exp(-bv / s))


def test_bootstrap_prior_warmup_and_anneal():
    from experiment_DM.exp24_dpbr_core import DECAY_TAU, N_WARMUP, BootstrapPriorState

    p = BootstrapPriorState()
    assert p.alpha_h() == 0.0
    # Through warmup: alpha_h stays 0 while step < N_WARMUP
    for _ in range(N_WARMUP - 1):
        p.update(0.1)
    assert p.alpha_h() == 0.0
    p.update(0.1)
    assert p.step == N_WARMUP
    # At step == N_WARMUP, alpha_h = 1 - exp(0) = 0; one more step → > 0
    p.update(0.1)
    assert p.step == N_WARMUP + 1
    assert p.alpha_h() > 0.0
    assert p.alpha_h() == pytest.approx(1.0 - math.exp(-1.0 / DECAY_TAU), rel=1e-5)


def test_apply_prior_identity_before_warmup():
    from experiment_DM.exp24_dpbr_core import BootstrapPriorState

    p = BootstrapPriorState()
    x = 0.33
    assert p.apply_prior(x) == pytest.approx(x)


def test_env_var_scale_requires_import_order():
    """EXP24_VAR_SCALE is read at module import; reload picks up new env."""
    os.environ["EXP24_VAR_SCALE"] = "0.08"
    import experiment_DM.exp24_dpbr_core as m

    importlib.reload(m)
    try:
        assert m.VAR_SCALE == 0.08
        assert m.dpbr_reliability_weight(0.0, 0.2, var_scale=None) == pytest.approx(
            math.exp(-0.04 / 0.08)
        )
    finally:
        os.environ.pop("EXP24_VAR_SCALE", None)
        importlib.reload(m)


def test_total_is_budget_matches_exp09():
    from experiment_DM.exp24_dpbr_core import K_HALF

    assert K_HALF * 2 == 128
