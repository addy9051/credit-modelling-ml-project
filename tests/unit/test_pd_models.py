"""
Unit tests for the PD evaluation foundation and the scorecard baseline:

  * src.models.splits.oot_split          — out-of-time partitioning
  * src.models.quality                   — metric bundle + gate enforcement
  * src.models.pd.scorecard.PDScorecard  — WOE + logistic baseline
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.pd.scorecard import PDScorecard
from src.models.quality import (
    ModelQualityError,
    PDMetrics,
    enforce_pd_gates,
    evaluate_pd,
)
from src.models.splits import oot_split

# ``credit_data`` is provided by tests/conftest.py (shared with the tree rungs).

CONTINUOUS = ["cibil_score", "revolving_utilisation", "dpd_90_count_24m"]
CATEGORICAL = ["borrower_segment"]


# --------------------------------------------------------------------------- #
# OOT split
# --------------------------------------------------------------------------- #


def test_oot_split_partitions_by_date(credit_data):
    _, _, dates = credit_data
    split = oot_split(dates, "2023-03-31", "2024-03-31", "2024-12-31")

    # masks are disjoint and each fold is populated
    assert not (split.train & split.validation).any()
    assert not (split.validation & split.test).any()
    assert all(v > 0 for v in split.counts.values())

    # every train row is on/before the train cutoff; every test row after validation
    assert (dates[split.train] <= pd.Timestamp("2023-03-31")).all()
    assert (dates[split.test] > pd.Timestamp("2024-03-31")).all()


def test_oot_split_rejects_unordered_cutoffs(credit_data):
    _, _, dates = credit_data
    with pytest.raises(ValueError, match="increasing"):
        oot_split(dates, "2024-03-31", "2023-03-31", "2024-12-31")


def test_oot_split_empty_test_raises(credit_data):
    _, _, dates = credit_data
    # all cutoffs far in the future of the data -> test fold empty
    with pytest.raises(ValueError, match="Test fold is empty"):
        oot_split(dates, "2030-01-01", "2030-06-01", "2030-12-31")


# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #


def test_evaluate_pd_bundle(credit_data):
    X, y, _ = credit_data
    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X, y)
    metrics = evaluate_pd(y, model.predict_proba(X))
    assert isinstance(metrics, PDMetrics)
    assert 0.0 <= metrics.auc <= 1.0
    assert metrics.gini == pytest.approx(2 * metrics.auc - 1)
    assert metrics.n == len(y)
    assert metrics.n_default == int(y.sum())


def test_evaluate_pd_rejects_bad_hl_groups(credit_data):
    X, y, _ = credit_data
    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X, y)
    # hl_groups < 3 is a configuration error: fail fast, do not silently NaN it.
    with pytest.raises(ValueError, match="hl_groups"):
        evaluate_pd(y, model.predict_proba(X), hl_groups=2)


def test_enforce_gates_pass():
    metrics = PDMetrics(
        gini=0.55,
        ks=0.42,
        auc=0.775,
        brier=0.08,
        hl_statistic=6.0,
        hl_pvalue=0.65,
        n=1000,
        n_default=120,
    )
    # all thresholds satisfied -> no raise
    enforce_pd_gates(metrics, min_gini=0.45, min_ks=0.35, min_hl_pvalue=0.05)


def test_enforce_gates_fail_discrimination():
    metrics = PDMetrics(
        gini=0.30,
        ks=0.20,
        auc=0.65,
        brier=0.12,
        hl_statistic=6.0,
        hl_pvalue=0.65,
        n=1000,
        n_default=120,
    )
    with pytest.raises(ModelQualityError, match=r"Gini.*min_gini"):
        enforce_pd_gates(metrics, min_gini=0.45, min_ks=0.35)


def test_enforce_gates_nan_hl_pvalue_is_failure():
    metrics = PDMetrics(
        gini=0.55,
        ks=0.42,
        auc=0.775,
        brier=0.08,
        hl_statistic=math.nan,
        hl_pvalue=math.nan,
        n=1000,
        n_default=120,
    )
    # NaN HL p-value must fail the calibration gate, not silently pass.
    with pytest.raises(ModelQualityError, match="Hosmer-Lemeshow"):
        enforce_pd_gates(metrics, min_gini=0.45, min_ks=0.35, min_hl_pvalue=0.05)


# --------------------------------------------------------------------------- #
# PDScorecard
# --------------------------------------------------------------------------- #


def test_scorecard_discriminates_out_of_time(credit_data):
    X, y, dates = credit_data
    split = oot_split(dates, "2023-03-31", "2024-03-31", "2024-12-31")

    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X[split.train], y[split.train])
    metrics = model.evaluate(X[split.test], y[split.test])

    # strong synthetic signal should clear the base PD gate on the OOT test fold
    enforce_pd_gates(metrics, min_gini=0.40, min_ks=0.30)
    assert metrics.gini > 0.40


def test_scorecard_predict_proba_is_probability(credit_data):
    X, y, _ = credit_data
    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_scorecard_points_rank_inversely_to_risk(credit_data):
    X, y, _ = credit_data
    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X, y)
    p = model.predict_proba(X)
    points = model.score_points(X)
    # points are a strictly decreasing (logit) function of default probability:
    # ordering rows by ascending p yields non-increasing points.
    points_by_risk = points[np.argsort(p)]
    assert (np.diff(points_by_risk) <= 1e-9).all()


def test_scorecard_coefficients_and_iv(credit_data):
    X, y, _ = credit_data
    model = PDScorecard(CONTINUOUS, CATEGORICAL).fit(X, y)

    coefs = model.coefficients()
    assert "intercept" in set(coefs["feature"])
    # one coefficient per WOE feature + intercept
    assert len(coefs) == len(CONTINUOUS) + len(CATEGORICAL) + 1

    iv = model.iv_summary()
    cibil_iv = iv.loc[iv["feature"] == "cibil_score", "iv"].iloc[0]
    assert cibil_iv > 0.10  # CIBIL is strongly predictive here


def test_scorecard_blocks_forbidden_feature(credit_data):
    X, y, _ = credit_data
    X2 = X.copy()
    X2["state_code"] = "MH"
    # state_code routed as a feature must trip the PD forbidden-feature guard
    model = PDScorecard(CONTINUOUS, [*CATEGORICAL, "state_code"])
    with pytest.raises(ValueError, match=r"[Ff]orbidden"):
        model.fit(X2, y)


def test_scorecard_predict_before_fit_raises(credit_data):
    X, _, _ = credit_data
    with pytest.raises(RuntimeError, match="not fitted"):
        PDScorecard(CONTINUOUS, CATEGORICAL).predict_proba(X)
