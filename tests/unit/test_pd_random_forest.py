"""
Unit tests for the Random Forest PD rung (src.models.pd.random_forest).

The ``credit_data`` fixture is shared via tests/conftest.py.
"""

from __future__ import annotations

import pytest

from src.models.pd.random_forest import PDRandomForest
from src.models.quality import enforce_pd_gates
from src.models.splits import oot_split

CONTINUOUS = ["cibil_score", "revolving_utilisation", "dpd_90_count_24m"]
CATEGORICAL = ["borrower_segment"]

# A lean forest keeps the unit tests fast without changing behaviour under test.
FAST = {"n_estimators": 120, "n_jobs": 1}


def test_random_forest_discriminates_out_of_time(credit_data):
    X, y, dates = credit_data
    split = oot_split(dates, "2023-03-31", "2024-03-31", "2024-12-31")

    model = PDRandomForest(CONTINUOUS, CATEGORICAL, **FAST).fit(X[split.train], y[split.train])
    metrics = model.evaluate(X[split.test], y[split.test])

    # strong synthetic signal should clear the base PD gate on the OOT test fold
    enforce_pd_gates(metrics, min_gini=0.40, min_ks=0.30)
    assert metrics.gini > 0.40


def test_random_forest_predict_proba_is_probability(credit_data):
    X, y, _ = credit_data
    model = PDRandomForest(CONTINUOUS, CATEGORICAL, **FAST).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_random_forest_smote_applies_to_fit_only(credit_data):
    """SMOTE is a sampler: it must resample at fit but be bypassed at predict."""
    X, y, dates = credit_data
    split = oot_split(dates, "2023-03-31", "2024-03-31", "2024-12-31")

    model = PDRandomForest(CONTINUOUS, CATEGORICAL, use_smote=True, **FAST)
    model.fit(X[split.train], y[split.train])

    # one probability per input row on the untouched test fold (no oversampling)
    n_test = int(split.test.sum())
    p = model.predict_proba(X[split.test])
    assert p.shape == (n_test,)


def test_random_forest_feature_importances(credit_data):
    X, y, _ = credit_data
    model = PDRandomForest(CONTINUOUS, CATEGORICAL, **FAST).fit(X, y)

    imp = model.feature_importances()
    assert list(imp.columns) == ["feature", "importance"]
    assert set(imp["feature"]) == set(CONTINUOUS + CATEGORICAL)
    assert imp["importance"].sum() == pytest.approx(1.0, abs=1e-6)
    # sorted descending
    assert imp["importance"].is_monotonic_decreasing
    # cibil_score carries the dominant signal here -> a top-ranked feature
    assert "cibil_score" in set(imp["feature"].head(2))


def test_random_forest_blocks_forbidden_feature(credit_data):
    X, y, _ = credit_data
    X2 = X.copy()
    X2["state_code"] = "MH"
    # state_code routed as a feature must trip the PD forbidden-feature guard
    model = PDRandomForest(CONTINUOUS, [*CATEGORICAL, "state_code"], **FAST)
    with pytest.raises(ValueError, match=r"[Ff]orbidden"):
        model.fit(X2, y)


def test_random_forest_predict_before_fit_raises(credit_data):
    X, _, _ = credit_data
    with pytest.raises(RuntimeError, match="not fitted"):
        PDRandomForest(CONTINUOUS, CATEGORICAL, **FAST).predict_proba(X)


def test_random_forest_without_smote_also_discriminates(credit_data):
    """The class_weight path (no SMOTE) is a valid configuration and still ranks."""
    X, y, dates = credit_data
    split = oot_split(dates, "2023-03-31", "2024-03-31", "2024-12-31")

    model = PDRandomForest(
        CONTINUOUS, CATEGORICAL, use_smote=False, class_weight="balanced_subsample", **FAST
    ).fit(X[split.train], y[split.train])
    metrics = model.evaluate(X[split.test], y[split.test])
    assert metrics.gini > 0.40
