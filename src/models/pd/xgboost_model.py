"""
XGBoost PD model — the gradient-boosted rung of the PD ladder.

A stronger non-linear challenger than the Random Forest. It reuses the Phase-2
*tree* preprocessing (``build_pd_feature_pipeline("tree", ...)`` — impute +
ordinal encode, forbidden-feature guard armed) and fits an
:class:`xgboost.XGBClassifier`, mirroring the ``PDScorecard`` / ``PDRandomForest``
API (``fit`` / ``predict_proba`` / ``evaluate`` / ``feature_importances``).

Class imbalance is handled with XGBoost's native ``scale_pos_weight`` (= neg/pos
on the training fold) rather than SMOTE — boosting responds well to it and it
keeps the data pipeline simpler. The preprocessing is fit on the **training fold
only** and reused to transform the validation/test folds, so there is no leakage.

Hyperparameter search (``tune``) uses **Optuna TPE maximising the Gini on the
out-of-time validation fold** — never random cross-validation, per the project's
PD evaluation rule. The 100-trial sweep is long-running; drive it with ``/loop``.

Scope: this rung deliberately stays narrow. Probability calibration is handled
uniformly across all rungs by ``calibrator.py`` (isotonic / Platt / LRA), and
MLflow logging + champion selection by ``trainer.py`` — not embedded here, so the
rungs stay swappable and consistently responsibility-scoped.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.feature_engineering.pipeline import PD_FORBIDDEN_FEATURES, build_pd_feature_pipeline
from src.models.quality import PDMetrics, evaluate_pd
from src.validation.discrimination import gini

# Tunable hyperparameters and their sensible non-tuned defaults. Optuna's best
# params (or any caller overrides) are merged over these via the ``params`` arg.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# Fixed (non-tuned) estimator settings shared by fit and the Optuna objective.
_FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
}


class PDXGBoost:
    """
    Gradient-boosted PD model with Optuna tuning on the out-of-time fold.

    Parameters
    ----------
    continuous_features, categorical_features : Sequence[str]
        Column splits, typically ``PDFeatureMatrix.continuous`` / ``.categorical``.
    params : dict | None
        Overrides merged over :data:`_DEFAULT_PARAMS` — e.g. the output of
        :meth:`tune`. ``scale_pos_weight`` is *not* set here; it is derived from
        the training fold at fit time.
    early_stopping_rounds : int
        Early-stopping patience; only active when an eval fold is passed to
        :meth:`fit` (and inside :meth:`tune`).
    use_scale_pos_weight : bool
        Set ``scale_pos_weight = neg/pos`` on the training fold (default True).
    random_state : int
        Seed (project default 42); shared by the booster and the TPE sampler.
    n_jobs : int
        Booster parallelism (``-1`` = all cores).
    forbidden_features : Sequence[str]
        Regulatory guardrail passed through to the tree pipeline; the fit raises
        if any are present. Defaults to :data:`PD_FORBIDDEN_FEATURES`.
    """

    def __init__(
        self,
        continuous_features: Sequence[str],
        categorical_features: Sequence[str],
        *,
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 50,
        use_scale_pos_weight: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        forbidden_features: Sequence[str] = PD_FORBIDDEN_FEATURES,
    ) -> None:
        """
        Initialize a PDXGBoost model with feature configuration and training settings.
        
        Parameters:
            continuous_features (Sequence[str]): Column names of continuous features.
            categorical_features (Sequence[str]): Column names of categorical features.
            params (dict[str, Any] | None): User-provided hyperparameters, merged over defaults. Defaults to None.
            early_stopping_rounds (int): Number of rounds for early stopping during training. Defaults to 50.
            use_scale_pos_weight (bool): Whether to apply class-imbalance weighting. Defaults to True.
            random_state (int): Random seed for reproducibility. Defaults to 42.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
            forbidden_features (Sequence[str]): Feature names to exclude from preprocessing. Defaults to PD_FORBIDDEN_FEATURES.
        """
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.use_scale_pos_weight = use_scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.forbidden_features = forbidden_features
        self.preprocess_: Pipeline | None = None
        self.clf_: XGBClassifier | None = None
        self.best_params_: dict[str, Any] | None = None
        self.study_: Any = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Computes the class imbalance weight for XGBoost training.
        
        If class weighting is disabled, returns 1.0. Otherwise, returns the ratio of negative to positive labels in the target vector, or 1.0 if no positive labels exist.
        
        Returns:
            float: The class imbalance weight.
        """
        if not self.use_scale_pos_weight:
            return 1.0
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return neg / pos if pos > 0 else 1.0

    def _make_clf(
        self, params: dict[str, Any], scale_pos_weight: float, early_stopping_rounds: int | None
    ) -> XGBClassifier:
        """
        Create an XGBoost classifier with merged hyperparameters and instance configuration.
        
        Parameters:
            params (dict[str, Any]): Hyperparameters to merge with default values.
            scale_pos_weight (float): Class weight to address data imbalance.
            early_stopping_rounds (int | None): Early stopping threshold; None disables it.
        
        Returns:
            XGBClassifier: The configured gradient booster instance.
        """
        return XGBClassifier(
            **params,
            **_FIXED_PARAMS,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )

    def _new_preprocess(self) -> Pipeline:
        """
        Create a Phase-2 preprocessing pipeline for tree-based feature engineering.
        
        Returns:
            Pipeline: A scikit-learn Pipeline configured with continuous and categorical feature transformations, respecting forbidden features constraints.
        """
        return build_pd_feature_pipeline(
            "tree",
            self.continuous_features,
            self.categorical_features,
            forbidden_features=self.forbidden_features,
        )

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        eval_X: pd.DataFrame | None = None,
        eval_y: np.ndarray | pd.Series | None = None,
    ) -> PDXGBoost:
        """
        Fit the preprocessing (training fold only) and the booster.

        If ``eval_X`` / ``eval_y`` (the out-of-time validation fold) are supplied,
        early stopping is enabled against them; the eval fold is transformed with
        the train-fitted preprocessing, so no statistics leak from it.
        """
        self.preprocess_ = self._new_preprocess()
        xt = self.preprocess_.fit_transform(X)
        yt = np.asarray(y)
        spw = self._scale_pos_weight(yt)

        if eval_X is not None and eval_y is not None:
            xvt = self.preprocess_.transform(eval_X)
            clf = self._make_clf(self.params, spw, self.early_stopping_rounds)
            clf.fit(xt, yt, eval_set=[(xvt, np.asarray(eval_y))], verbose=False)
        else:
            clf = self._make_clf(self.params, spw, None)
            clf.fit(xt, yt)
        self.clf_ = clf
        return self

    def _check_fitted(self) -> tuple[Pipeline, XGBClassifier]:
        """
        Verify that the model has been fitted.
        
        Returns:
            tuple[Pipeline, XGBClassifier]: The fitted preprocessing pipeline and classifier.
        
        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.preprocess_ is None or self.clf_ is None:
            raise RuntimeError("PDXGBoost is not fitted; call fit() first.")
        return self.preprocess_, self.clf_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the predicted probability of default for each sample.
        
        Returns:
            np.ndarray: 1-D array of probabilities for the positive class (default).
        """
        preprocess, clf = self._check_fitted()
        return clf.predict_proba(preprocess.transform(X))[:, 1]

    # ------------------------------------------------------------------ #
    # Tuning (Optuna TPE on the OOT validation fold)
    # ------------------------------------------------------------------ #

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame,
        y_val: np.ndarray | pd.Series,
        *,
        n_trials: int = 100,
        timeout: float | None = None,
        show_progress_bar: bool = False,
    ) -> dict[str, Any]:
        """
        Optuna TPE search maximising **validation-fold Gini**.

        The objective trains on the training fold and scores Gini on the
        out-of-time validation fold — never random CV — honouring the PD
        evaluation rule. The preprocessing is fit once on the training fold and
        reused across trials (no per-trial leakage). Returns the best params
        (also stored on ``best_params_``); the full study is on ``study_``.

        The 100-trial default is long-running — run it under ``/loop``.
        """
        import optuna

        preprocess = self._new_preprocess()
        xt = preprocess.fit_transform(X_train)
        xvt = preprocess.transform(X_val)
        ytr = np.asarray(y_train)
        yvl = np.asarray(y_val)
        spw = self._scale_pos_weight(ytr)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            clf = self._make_clf(params, spw, self.early_stopping_rounds)
            clf.fit(xt, ytr, eval_set=[(xvt, yvl)], verbose=False)
            return gini(yvl, clf.predict_proba(xvt)[:, 1])

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress_bar
        )
        self.study_ = study
        self.best_params_ = dict(study.best_params)
        return self.best_params_

    # ------------------------------------------------------------------ #
    # Interpretability / evaluation
    # ------------------------------------------------------------------ #

    def feature_importances(self) -> pd.DataFrame:
        """
        Return feature importances from the fitted model.
        
        Returns:
        	pd.DataFrame: A DataFrame with columns 'feature' (feature name) and 'importance' (importance value), sorted by importance in descending order.
        """
        preprocess, clf = self._check_fitted()
        importances = clf.feature_importances_
        names: list[str] = [*self.continuous_features, *self.categorical_features]
        if len(names) != len(importances):
            names = [
                str(n).split("__", 1)[-1]
                for n in preprocess.named_steps["preprocess"].get_feature_names_out()
            ]
        return pd.DataFrame({"feature": names, "importance": importances}).sort_values(
            "importance", ascending=False, ignore_index=True
        )

    def evaluate(
        self, X: pd.DataFrame, y: np.ndarray | pd.Series, *, hl_groups: int = 10
    ) -> PDMetrics:
        """
        Compute discrimination and calibration metrics on a provided fold.
        
        Parameters:
            hl_groups (int): Number of groups for calibration analysis. Defaults to 10.
        
        Returns:
            PDMetrics: Object containing discrimination and calibration metrics.
        """
        return evaluate_pd(np.asarray(y), self.predict_proba(X), hl_groups=hl_groups)


__all__ = ["PDXGBoost"]
