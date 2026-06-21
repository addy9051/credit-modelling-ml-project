"""
Unit tests for the XGBoost PD rung (src.models.pd.xgboost_model).

The ``credit_data`` fixture is shared via tests/conftest.py. Tests use a small
booster / few Optuna trials to stay fast without changing behaviour under test.
"""

from __future__ import annotations

import pytest

from src.models.pd.xgboost_model import PDXGBoost
from src.models.quality import enforce_pd_gates
from src.models.splits import oot_split

CONTINUOUS = ["cibil_score", "revolving_utilisation", "dpd_90_count_24m"]
CATEGORICAL = ["borrower_segment"]

# Lean booster keeps the suite fast; the signal is strong enough to clear gates.
FAST = {"params": {"n_estimators": 80, "max_depth": 4}, "n_jobs": 1}
CUTOFFS = ("2023-03-31", "2024-03-31", "2024-12-31")


def test_xgboost_discriminates_out_of_time(credit_data):
    X, y, dates = credit_data
    split = oot_split(dates, *CUTOFFS)

    model = PDXGBoost(CONTINUOUS, CATEGORICAL, **FAST).fit(
        X[split.train],
        y[split.train],
        eval_X=X[split.validation],
        eval_y=y[split.validation],
    )
    metrics = model.evaluate(X[split.test], y[split.test])

    enforce_pd_gates(metrics, min_gini=0.40, min_ks=0.30)
    assert metrics.gini > 0.40


def test_xgboost_predict_proba_is_probability(credit_data):
    """
    Verify that predict_proba outputs valid probability values.
    
    Checks that predictions are a 1D array of the correct length and all values
    are within the valid probability range [0, 1].
    """
    X, y, _ = credit_data
    model = PDXGBoost(CONTINUOUS, CATEGORICAL, **FAST).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_xgboost_feature_importances(credit_data):
    X, y, _ = credit_data
    model = PDXGBoost(CONTINUOUS, CATEGORICAL, **FAST).fit(X, y)

    imp = model.feature_importances()
    assert list(imp.columns) == ["feature", "importance"]
    assert set(imp["feature"]) == set(CONTINUOUS + CATEGORICAL)
    assert imp["importance"].is_monotonic_decreasing
    # cibil_score carries the dominant signal here -> a top-ranked feature
    assert "cibil_score" in set(imp["feature"].head(2))


def test_xgboost_tune_optimises_on_oot_validation(credit_data):
    X, y, dates = credit_data
    split = oot_split(dates, *CUTOFFS)

    model = PDXGBoost(CONTINUOUS, CATEGORICAL, n_jobs=1)
    best = model.tune(
        X[split.train],
        y[split.train],
        X[split.validation],
        y[split.validation],
        n_trials=5,
    )
    # search space is reflected in the returned best params
    assert {"max_depth", "learning_rate", "n_estimators"} <= set(best)
    assert model.study_ is not None
    assert len(model.study_.trials) == 5

    # a model fit with the tuned params still clears the base gate out-of-time
    tuned = PDXGBoost(CONTINUOUS, CATEGORICAL, params=best, n_jobs=1).fit(
        X[split.train], y[split.train]
    )
    metrics = tuned.evaluate(X[split.test], y[split.test])
    assert metrics.gini > 0.40


def test_xgboost_blocks_forbidden_feature(credit_data):
    X, y, _ = credit_data
    X2 = X.copy()
    X2["state_code"] = "MH"
    model = PDXGBoost(CONTINUOUS, [*CATEGORICAL, "state_code"], **FAST)
    with pytest.raises(ValueError, match=r"[Ff]orbidden"):
        model.fit(X2, y)


def test_xgboost_predict_before_fit_raises(credit_data):
    X, _, _ = credit_data
    with pytest.raises(RuntimeError, match="not fitted"):
        PDXGBoost(CONTINUOUS, CATEGORICAL, **FAST).predict_proba(X)


def test_xgboost_without_scale_pos_weight_still_ranks(credit_data):
    X, y, dates = credit_data
    split = oot_split(dates, *CUTOFFS)

    model = PDXGBoost(CONTINUOUS, CATEGORICAL, use_scale_pos_weight=False, **FAST).fit(
        X[split.train], y[split.train]
    )
    metrics = model.evaluate(X[split.test], y[split.test])
    assert metrics.gini > 0.40
