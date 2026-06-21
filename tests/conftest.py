"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def credit_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Predictive synthetic PD data with observation dates spanning the OOT cutoffs.

    Default risk falls with CIBIL and rises with utilisation and delinquency, so
    any competent PD model should discriminate strongly. The signal is deliberately
    strong (and the default rate higher than a real book) so model *mechanics* —
    not calibration realism — are what the unit tests exercise. Shared by the
    scorecard and tree-rung test modules.
    """
    rng = np.random.default_rng(42)
    n = 5000
    cibil = rng.integers(300, 900, n).astype(float)
    util = np.clip(rng.beta(2, 5, n) + rng.normal(0, 0.05, n), 0, 1.2)
    dpd = rng.poisson(0.3, n).astype(float)
    segment = rng.choice(["salaried", "self_employed", "msme"], n, p=[0.5, 0.3, 0.2])

    lp = -0.011 * (cibil - 600) + 2.2 * util + 0.6 * dpd - 1.2
    p = 1.0 / (1.0 + np.exp(-lp))
    y = (rng.random(n) < p).astype(int)

    X = pd.DataFrame(
        {
            "cibil_score": cibil,
            "revolving_utilisation": util,
            "dpd_90_count_24m": dpd,
            "borrower_segment": segment,
        }
    )
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")
    return X, y, pd.Series(dates, name="observation_date")
