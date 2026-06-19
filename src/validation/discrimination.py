"""
Discrimination metrics for PD models: AUC-ROC, Gini, and the KS statistic.

These are the rank-ordering metrics the PD quality gates are written against
(``models.pd.min_gini`` / ``min_ks``). All take the predicted probability of the
**positive (default) class** — i.e. ``predict_proba(X)[:, 1]`` — so orientation
is consistent regardless of any score-point transform applied downstream.

Scope: the bootstrap confidence intervals, CAP curves and decile tables noted in
the module's Phase-5 remit are intentionally out of scope here; this provides the
point estimates the Phase-3 trainer's gates need.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError(f"y_true and y_score length mismatch: {yt.shape[0]} vs {ys.shape[0]}")
    if yt.ndim != 1 or ys.ndim != 1:
        raise ValueError("y_true and y_score must be 1-dimensional.")
    if not np.isfinite(ys).all():
        raise ValueError("y_score contains non-finite values.")
    if yt.dtype == bool:
        yt = yt.astype(int)
    classes = set(np.unique(yt).tolist())
    if not classes.issubset({0, 1}):
        raise ValueError(f"y_true must be binary 0/1; got {sorted(classes)}")
    if classes != {0, 1}:
        raise ValueError("y_true must contain both classes (0 and 1) to score discrimination.")
    return yt.astype(int), ys


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve for the predicted default probability."""
    yt, ys = _validate(y_true, y_score)
    return float(roc_auc_score(yt, ys))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini coefficient (accuracy ratio) = ``2 * AUC - 1``.

    Ranges from 0 (no discrimination) to 1 (perfect ranking). This is the headline
    PD discrimination metric in the quality gates.
    """
    return 2.0 * auc_roc(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: the maximum separation between the predicted-
    score distributions of defaulters and non-defaulters (range 0-1).
    """
    yt, ys = _validate(y_true, y_score)
    return float(ks_2samp(ys[yt == 1], ys[yt == 0]).statistic)


__all__ = ["auc_roc", "gini", "ks_statistic"]
