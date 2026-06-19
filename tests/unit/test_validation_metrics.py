"""
Unit tests for the PD discrimination and calibration metrics
(src.validation.discrimination / src.validation.calibration).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.calibration import brier_score, hosmer_lemeshow
from src.validation.discrimination import auc_roc, gini, ks_statistic

# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #


def test_perfect_separation_metrics():
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    score = y.astype(float)  # score == label -> perfect ranking
    assert auc_roc(y, score) == pytest.approx(1.0)
    assert gini(y, score) == pytest.approx(1.0)
    assert ks_statistic(y, score) == pytest.approx(1.0)


def test_random_scores_have_near_zero_gini():
    rng = np.random.default_rng(42)
    n = 6000
    y = (rng.random(n) < 0.2).astype(int)
    score = rng.random(n)  # independent of y
    assert abs(gini(y, score)) < 0.10
    assert ks_statistic(y, score) < 0.10


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(7)
    n = 2000
    y = (rng.random(n) < 0.3).astype(int)
    # a noisy-but-informative score
    score = 0.5 * y + rng.normal(0, 1, n)
    assert gini(y, score) == pytest.approx(2.0 * auc_roc(y, score) - 1.0)


def test_discrimination_requires_both_classes():
    y = np.ones(100, dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        gini(y, np.linspace(0, 1, 100))


def test_discrimination_rejects_nonfinite_scores():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    score = np.r_[np.full(50, np.nan), np.ones(50)]
    with pytest.raises(ValueError, match="non-finite"):
        auc_roc(y, score)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_hl_well_calibrated_not_rejected():
    rng = np.random.default_rng(42)
    n = 5000
    p = rng.uniform(0.02, 0.6, n)  # predicted probabilities
    y = (rng.random(n) < p).astype(int)  # outcomes drawn from those same probs
    result = hosmer_lemeshow(y, p, n_groups=10)
    assert result.dof == result.n_groups - 2
    assert result.p_value > 0.05  # well calibrated -> fail to reject


def test_hl_miscalibrated_rejected():
    rng = np.random.default_rng(42)
    n = 5000
    p_true = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p_true).astype(int)
    p_biased = np.clip(p_true * 0.5, 1e-6, 1 - 1e-6)  # systematically under-predicts
    result = hosmer_lemeshow(y, p_biased, n_groups=10)
    assert result.p_value < 0.05  # poorly calibrated -> reject


def test_hl_rejects_too_concentrated_probs():
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    p = np.full(200, 0.5)  # no spread -> cannot form bins
    with pytest.raises(ValueError, match="concentrated|bins"):
        hosmer_lemeshow(y, p, n_groups=10)


def test_brier_score_bounds():
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)  # perfect
    assert brier_score(y, np.full(200, 0.5)) == pytest.approx(0.25)  # uninformative


def test_brier_rejects_out_of_range_prob():
    y = np.r_[np.zeros(10), np.ones(10)].astype(int)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score(y, np.r_[np.full(10, 1.5), np.zeros(10)])
