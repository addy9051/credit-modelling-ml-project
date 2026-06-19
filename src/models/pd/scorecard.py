"""
Logistic-regression WOE scorecard — the regulatory-interpretable PD baseline.

This is the first rung of the PD ladder and the model the others are benchmarked
against. It chains the Phase-2 scorecard preprocessing
(``build_pd_feature_pipeline("scorecard", ...)`` — WOE encoding with the
forbidden-feature guard armed) into a plain :class:`~sklearn.linear_model.LogisticRegression`,
which on WOE inputs *is* the classic points-based scorecard.

Beyond ``predict_proba`` it exposes the regulatory artefacts examiners expect:
per-feature WOE Information Values, the fitted logistic coefficients, and the
standard PDO points transform. Evaluation produces a :class:`PDMetrics` bundle;
gate enforcement is left to the caller (the trainer), which holds the config
thresholds.

Probabilities are well calibrated by construction (unweighted logistic), so no
SMOTE / class-weighting is applied here — those belong to the imbalanced tree
rungs, on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.feature_engineering.pipeline import PD_FORBIDDEN_FEATURES, build_pd_feature_pipeline
from src.models.quality import PDMetrics, evaluate_pd

# Clip probabilities away from 0/1 before the log-odds points transform.
_PROB_EPS = 1e-6


class PDScorecard:
    """
    WOE + logistic-regression PD scorecard.

    Parameters
    ----------
    continuous_features, categorical_features : list[str]
        Column splits, typically ``PDFeatureMatrix.continuous`` / ``.categorical``.
    n_bins : int
        Quantile bins for the WOE encoder.
    C : float
        Inverse L2 regularisation strength for the logistic regression.
    max_iter : int
        Solver iteration cap.
    random_state : int
        Seed (project default 42) for solver reproducibility.
    forbidden_features : Sequence[str]
        Regulatory guardrail passed through to the WOE pipeline. Defaults to
        :data:`PD_FORBIDDEN_FEATURES`; the fit raises if any are present.
    """

    def __init__(
        self,
        continuous_features: list[str],
        categorical_features: list[str],
        *,
        n_bins: int = 10,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        forbidden_features=PD_FORBIDDEN_FEATURES,
    ) -> None:
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        self.n_bins = n_bins
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.forbidden_features = forbidden_features
        self.pipeline_: Pipeline | None = None

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> PDScorecard:
        """Fit the WOE encoder and logistic regression on the training fold."""
        woe_pipe = build_pd_feature_pipeline(
            "scorecard",
            self.continuous_features,
            self.categorical_features,
            forbidden_features=self.forbidden_features,
            n_bins=self.n_bins,
        )
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        # woe_pipe is Pipeline([("woe", WOEEncoder)]); extend it with the classifier.
        self.pipeline_ = Pipeline([*woe_pipe.steps, ("clf", clf)])
        self.pipeline_.fit(X, np.asarray(y))
        return self

    def _check_fitted(self) -> Pipeline:
        if self.pipeline_ is None:
            raise RuntimeError("PDScorecard is not fitted; call fit() first.")
        return self.pipeline_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the 1-D predicted probability of **default** (positive class)."""
        pipe = self._check_fitted()
        return pipe.predict_proba(X)[:, 1]

    def score_points(
        self,
        X: pd.DataFrame,
        *,
        pdo: int = 20,
        base_score: int = 600,
        base_odds: float = 50.0,
    ) -> np.ndarray:
        """
        Map predictions to scorecard points (higher points == lower risk).

        Uses the standard ``points = offset + factor·ln(odds_good)`` transform,
        where ``factor = pdo/ln(2)`` so every ``pdo`` points doubles the good:bad
        odds, calibrated so ``base_odds`` good:bad maps to ``base_score``.
        """
        p = np.clip(self.predict_proba(X), _PROB_EPS, 1.0 - _PROB_EPS)
        factor = pdo / np.log(2.0)
        offset = base_score - factor * np.log(base_odds)
        odds_good = (1.0 - p) / p
        return offset + factor * np.log(odds_good)

    # ------------------------------------------------------------------ #
    # Interpretability artefacts
    # ------------------------------------------------------------------ #

    def iv_summary(self) -> pd.DataFrame:
        """Per-feature WOE Information Value table (from the fitted encoder)."""
        pipe = self._check_fitted()
        return pipe.named_steps["woe"].get_iv_summary()

    def coefficients(self) -> pd.DataFrame:
        """Fitted logistic coefficients per WOE feature, plus the intercept."""
        pipe = self._check_fitted()
        woe = pipe.named_steps["woe"]
        clf: LogisticRegression = pipe.named_steps["clf"]
        names = list(woe.get_feature_names_out())
        rows = [{"feature": "intercept", "coefficient": float(clf.intercept_[0])}]
        rows += [
            {"feature": name, "coefficient": float(coef)}
            for name, coef in zip(names, clf.coef_[0], strict=True)
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self, X: pd.DataFrame, y: np.ndarray | pd.Series, *, hl_groups: int = 10
    ) -> PDMetrics:
        """Evaluate discrimination + calibration on a fold (use the OOT test fold)."""
        return evaluate_pd(np.asarray(y), self.predict_proba(X), hl_groups=hl_groups)


__all__ = ["PDScorecard"]
