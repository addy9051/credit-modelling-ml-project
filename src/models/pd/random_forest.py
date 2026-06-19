"""
Random Forest PD model — the first tree rung of the PD ladder.

A non-linear challenger to the WOE scorecard baseline. It chains the Phase-2
*tree* preprocessing (``build_pd_feature_pipeline("tree", ...)`` — median /
most-frequent imputation + ordinal encoding, with the forbidden-feature guard
armed) into a :class:`~sklearn.ensemble.RandomForestClassifier`.

Class imbalance is handled with **SMOTE applied to the training fold only**. The
resampler is embedded as a step in an :class:`imblearn.pipeline.Pipeline`, which
runs samplers during ``fit`` *but bypasses them during ``predict``* — so SMOTE
can never leak into the validation/test folds by construction (the project's hard
guardrail). SMOTE is on by default here because this is a tree rung; it is *not*
used in the scorecard, which stays unweighted to keep its probabilities
calibrated.

Note on calibration: SMOTE-balanced training inflates the predicted default rate,
so the raw ``predict_proba`` is optimistic. That is fine for this rung — it is
selected on **discrimination** (Gini / KS, which are invariant to monotone
miscalibration); probability calibration is a downstream concern handled by
``calibrator.py`` before any production calibration gate (HL p > 0.05) applies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering.pipeline import PD_FORBIDDEN_FEATURES, build_pd_feature_pipeline
from src.models.quality import PDMetrics, evaluate_pd


class PDRandomForest:
    """
    Random Forest PD model with leakage-safe SMOTE.

    Parameters
    ----------
    continuous_features, categorical_features : Sequence[str]
        Column splits, typically ``PDFeatureMatrix.continuous`` / ``.categorical``.
    n_estimators : int
        Number of trees.
    max_depth : int | None
        Maximum tree depth (``None`` = grow until ``min_samples_leaf``).
    min_samples_leaf : int
        Minimum samples per leaf — the main overfitting control; kept high so
        leaf default rates are stable on an out-of-time book.
    max_features : str | int | float
        Features considered per split (sklearn semantics; ``"sqrt"`` default).
    class_weight : str | dict | None
        Passed to the forest. Left ``None`` when ``use_smote`` is True so
        imbalance is not corrected twice.
    use_smote : bool
        Apply SMOTE to the training fold (default True for this tree rung).
    smote_k_neighbors : int
        SMOTE neighbourhood size (needs > this many minority samples to fit).
    random_state : int
        Seed (project default 42); shared by SMOTE and the forest.
    n_jobs : int
        Forest parallelism (``-1`` = all cores).
    forbidden_features : Sequence[str]
        Regulatory guardrail passed through to the tree pipeline; the fit raises
        if any are present. Defaults to :data:`PD_FORBIDDEN_FEATURES`.
    """

    def __init__(
        self,
        continuous_features: Sequence[str],
        categorical_features: Sequence[str],
        *,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 50,
        max_features: str | int | float = "sqrt",
        class_weight: str | dict | None = None,
        use_smote: bool = True,
        smote_k_neighbors: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        forbidden_features: Sequence[str] = PD_FORBIDDEN_FEATURES,
    ) -> None:
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.forbidden_features = forbidden_features
        self.pipeline_: ImbPipeline | None = None

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> PDRandomForest:
        """Fit the preprocessing, (optional) SMOTE, and the forest on the training fold."""
        tree_pipe = build_pd_feature_pipeline(
            "tree",
            self.continuous_features,
            self.categorical_features,
            forbidden_features=self.forbidden_features,
        )
        # tree_pipe is Pipeline([("preprocess", ColumnTransformer)]). Build an
        # imblearn pipeline so SMOTE (a sampler) fires only at fit, never predict.
        steps = list(tree_pipe.steps)
        if self.use_smote:
            steps.append(
                ("smote", SMOTE(random_state=self.random_state, k_neighbors=self.smote_k_neighbors))
            )
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
            )
        )
        self.pipeline_ = ImbPipeline(steps)
        self.pipeline_.fit(X, np.asarray(y))
        return self

    def _check_fitted(self) -> ImbPipeline:
        if self.pipeline_ is None:
            raise RuntimeError("PDRandomForest is not fitted; call fit() first.")
        return self.pipeline_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the 1-D predicted probability of **default** (positive class)."""
        pipe = self._check_fitted()
        return pipe.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Interpretability
    # ------------------------------------------------------------------ #

    def feature_importances(self) -> pd.DataFrame:
        """
        Impurity-based feature importances, sorted descending.

        A cheap proxy for the SHAP ranking the governance sanity-check uses
        (``cibil_score`` / ``dpd_90_count_24m`` / ``revolving_utilisation`` should
        rank highly); SHAP itself lands in the Phase-5 validation work.
        """
        pipe = self._check_fitted()
        clf: RandomForestClassifier = pipe.named_steps["clf"]
        importances = clf.feature_importances_
        # The tree ColumnTransformer emits continuous columns then one ordinal
        # column per categorical, so names line up 1:1 in that order.
        names: list[str] = [*self.continuous_features, *self.categorical_features]
        if len(names) != len(importances):
            names = [
                str(n).split("__", 1)[-1]
                for n in pipe.named_steps["preprocess"].get_feature_names_out()
            ]
        return pd.DataFrame({"feature": names, "importance": importances}).sort_values(
            "importance", ascending=False, ignore_index=True
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self, X: pd.DataFrame, y: np.ndarray | pd.Series, *, hl_groups: int = 10
    ) -> PDMetrics:
        """Evaluate discrimination + calibration on a fold (use the OOT test fold)."""
        return evaluate_pd(np.asarray(y), self.predict_proba(X), hl_groups=hl_groups)


__all__ = ["PDRandomForest"]
