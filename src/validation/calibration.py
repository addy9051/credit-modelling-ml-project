"""
Calibration diagnostics for PD models: the Hosmer-Lemeshow goodness-of-fit test
and the Brier score.

The HL test underpins the production calibration gate (``HL p > 0.05`` — a *high*
p-value means we fail to reject "predicted == observed", i.e. the model is well
calibrated). Reliability diagrams and the binomial/normal tests in the module's
Phase-5 remit are out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Outcome of a Hosmer-Lemeshow test. ``p_value`` high == well calibrated."""

    statistic: float
    p_value: float
    dof: int
    n_groups: int


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-dimensional.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_prob length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    if not np.isfinite(yp).all():
        raise ValueError("y_prob contains non-finite values.")
    if (yp < 0).any() or (yp > 1).any():
        raise ValueError("y_prob must lie in [0, 1].")
    if yt.dtype == bool:
        yt = yt.astype(int)
    classes = set(np.unique(yt).tolist())
    if not classes.issubset({0, 1}):
        raise ValueError(f"y_true must be binary 0/1; got {sorted(classes)}")
    return yt.astype(int), yp


def hosmer_lemeshow(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> HosmerLemeshowResult:
    """
    Hosmer-Lemeshow goodness-of-fit test, grouping by predicted-probability deciles.

    Observations are sorted into ``n_groups`` equal-frequency bins; the statistic
    compares observed vs expected defaults per bin. ``dof = n_groups - 2`` and the
    p-value is the upper-tail chi-square probability. A p-value **above** the gate
    threshold (e.g. 0.05) indicates the model is well calibrated.

    Ties in ``y_prob`` can collapse bins, so the realised group count may be lower
    than ``n_groups``; ``dof`` and the returned ``n_groups`` reflect the realised
    count.
    """
    yt, yp = _validate(y_true, y_prob)
    if n_groups < 3:
        raise ValueError(f"n_groups must be >= 3 for a meaningful HL test; got {n_groups}")

    # Equal-frequency bins on predicted probability; drop duplicate edges from ties.
    bins = pd.qcut(yp, q=n_groups, duplicates="drop")
    frame = pd.DataFrame({"y": yt, "p": yp, "bin": bins})

    realised_groups = frame["bin"].cat.categories.size
    if realised_groups < 3:
        raise ValueError(
            "Predicted probabilities are too concentrated to form >=3 HL bins; "
            "calibration cannot be assessed."
        )

    stat = 0.0
    for _, g in frame.groupby("bin", observed=True):
        n = len(g)
        observed_1 = float(g["y"].sum())
        expected_1 = float(g["p"].sum())
        observed_0 = n - observed_1
        expected_0 = n - expected_1
        # Guard empty expected cells (a fully 0/1-predicted bin contributes 0).
        if expected_1 > 0:
            stat += (observed_1 - expected_1) ** 2 / expected_1
        if expected_0 > 0:
            stat += (observed_0 - expected_0) ** 2 / expected_0

    dof = realised_groups - 2
    p_value = float(chi2.sf(stat, dof))
    return HosmerLemeshowResult(
        statistic=float(stat), p_value=p_value, dof=dof, n_groups=realised_groups
    )


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the predicted default probabilities (lower is better)."""
    yt, yp = _validate(y_true, y_prob)
    return float(np.mean((yp - yt) ** 2))


__all__ = ["HosmerLemeshowResult", "hosmer_lemeshow", "brier_score"]
