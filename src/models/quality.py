"""
PD model quality gates.

Bundles the discrimination and calibration metrics into one ``PDMetrics`` record
and enforces the configured thresholds, raising :class:`ModelQualityError` when a
gate fails — the hard stop the project's governance rules require of every PD
trainer (``models.pd.min_gini`` / ``min_ks`` and, in production, ``HL p > 0.05``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from src.validation.calibration import brier_score, hosmer_lemeshow
from src.validation.discrimination import auc_roc, gini, ks_statistic


class ModelQualityError(RuntimeError):
    """Raised when a model fails one or more configured quality gates."""


@dataclass(frozen=True)
class PDMetrics:
    """Discrimination + calibration metrics for one evaluation fold."""

    gini: float
    ks: float
    auc: float
    brier: float
    hl_statistic: float
    hl_pvalue: float
    n: int
    n_default: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_pd(y_true: np.ndarray, y_prob: np.ndarray, *, hl_groups: int = 10) -> PDMetrics:
    """
    Compute the full PD metric bundle for one fold.

    The Hosmer-Lemeshow test can be undefined when predicted probabilities are too
    concentrated to bin; in that case ``hl_statistic`` / ``hl_pvalue`` are NaN
    (and the calibration gate, if armed, will fail rather than pass silently).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    try:
        hl = hosmer_lemeshow(yt, yp, n_groups=hl_groups)
        hl_stat, hl_p = hl.statistic, hl.p_value
    except ValueError:
        hl_stat, hl_p = math.nan, math.nan

    return PDMetrics(
        gini=gini(yt, yp),
        ks=ks_statistic(yt, yp),
        auc=auc_roc(yt, yp),
        brier=brier_score(yt, yp),
        hl_statistic=hl_stat,
        hl_pvalue=hl_p,
        n=int(yt.shape[0]),
        n_default=int(np.asarray(yt).astype(int).sum()),
    )


def enforce_pd_gates(
    metrics: PDMetrics,
    *,
    min_gini: float,
    min_ks: float,
    min_hl_pvalue: float | None = None,
) -> None:
    """
    Raise :class:`ModelQualityError` if any gate fails; return ``None`` if all pass.

    Parameters
    ----------
    metrics : PDMetrics
        Output of :func:`evaluate_pd`, typically on the out-of-time test fold.
    min_gini, min_ks : float
        Discrimination floors (e.g. base 0.40 / 0.30, production 0.45 / 0.35).
    min_hl_pvalue : float | None
        Calibration floor on the HL p-value (production sets 0.05). When set, a
        NaN p-value (HL undefined) is treated as a failure, never a pass.
    """
    failures: list[str] = []
    if metrics.gini < min_gini:
        failures.append(f"Gini {metrics.gini:.4f} < min_gini {min_gini}")
    if metrics.ks < min_ks:
        failures.append(f"KS {metrics.ks:.4f} < min_ks {min_ks}")
    if min_hl_pvalue is not None and not (metrics.hl_pvalue >= min_hl_pvalue):
        # `not (>=)` makes a NaN p-value (HL undefined) a failure, not a pass.
        failures.append(
            f"Hosmer-Lemeshow p-value {metrics.hl_pvalue:.4f} < min {min_hl_pvalue} "
            "(model not adequately calibrated)"
        )
    if failures:
        raise ModelQualityError("PD quality gates failed:\n  - " + "\n  - ".join(failures))


__all__ = ["ModelQualityError", "PDMetrics", "evaluate_pd", "enforce_pd_gates"]
