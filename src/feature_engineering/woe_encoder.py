"""
Weight-of-Evidence (WOE) / Information Value (IV) encoder.

WOEEncoder is a scikit-learn compatible transformer for the regulatory
logistic-regression scorecard. It replaces raw feature values with their
Weight-of-Evidence, the standard interpretable encoding in credit scoring.

Convention (Siddiqi):
    For a bin/category i, with the binary target y where 1 == default ("bad"):
        dist_good_i = non_events_i / total_non_events
        dist_bad_i  = events_i     / total_events
        WOE_i       = ln(dist_good_i / dist_bad_i)
        IV          = Σ_i (dist_good_i - dist_bad_i) * WOE_i

    Higher WOE  => relatively more "goods" => lower default risk.
    IV is always non-negative and measures a feature's univariate predictive
    power. Interpretation bands (get_iv_summary):
        IV < 0.02         useless
        0.02 <= IV < 0.10 weak
        0.10 <= IV < 0.30 medium
        IV >= 0.30        strong  (flagged — may indicate leakage)

Design notes:
  - Continuous features are auto-binned with quantile cut points; values outside
    the fitted range are clamped into the edge bins at transform time.
  - Categorical features are encoded per category; rare categories are merged
    into an "other" group and unseen categories at transform time map to it.
  - Missing values (NaN) always get their own WOE bin.
  - WOE is Laplace-smoothed (so empty bins never produce ±inf) and clipped to
    [-woe_clip, woe_clip] to bound the influence of outlier bins.
  - Serialisable with joblib for MLflow artifact logging.
  - Returns a float ndarray with column order == feature_names_in_; pair with
    ``set_output(transform="pandas")`` for a named DataFrame (SHAP-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_OTHER = "__OTHER__"


@dataclass
class _FeatureWOE:
    """Fitted WOE state for a single feature."""

    kind: str  # "continuous" | "categorical"
    iv: float
    nan_woe: float
    # continuous
    cut_points: np.ndarray | None = None  # internal bin boundaries
    bin_woe: np.ndarray | None = None  # WOE indexed by bin
    # categorical
    category_woe: dict[object, float] = field(default_factory=dict)
    other_woe: float = 0.0


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight-of-Evidence encoder.

    Parameters
    ----------
    n_bins : int, default=10
        Target number of quantile bins for continuous features. Bins with
        duplicate quantile edges are merged, so the realised count may be lower.
    categorical_features : list[str] | None, default=None
        Columns to treat as categorical. If None, columns are inferred from
        dtype (object / category / bool -> categorical, else continuous).
    continuous_features : list[str] | None, default=None
        Columns to treat as continuous. If None, inferred from dtype.
    woe_clip : float, default=3.0
        WOE values are clipped to [-woe_clip, woe_clip].
    rare_threshold : float, default=0.02
        Categorical levels with frequency below this fraction of rows are
        merged into the "other" group.
    regularization : float, default=0.5
        Laplace smoothing added to event / non-event counts per bin to avoid
        division by zero and infinite WOE on pure bins.
    forbidden_features : list[str] | None, default=None
        Columns that are not permitted as inputs for this model (regulatory
        guardrail). If any appear in X during fit, fit raises ValueError. Leave
        None where the columns are allowed — e.g. state_code is forbidden for PD
        but permitted for LGD, so only the PD pipeline passes it.

    Attributes
    ----------
    feature_names_in_ : np.ndarray
        Column names seen during fit.
    n_features_in_ : int
    iv_ : dict[str, float]
        Information Value per feature.
    feature_woe_ : dict[str, _FeatureWOE]
        Fitted per-feature WOE state.
    """

    def __init__(
        self,
        n_bins: int = 10,
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        woe_clip: float = 3.0,
        rare_threshold: float = 0.02,
        regularization: float = 0.5,
        forbidden_features: list[str] | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.woe_clip = woe_clip
        self.rare_threshold = rare_threshold
        self.regularization = regularization
        self.forbidden_features = forbidden_features

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> WOEEncoder:
        """Fit per-feature WOE/IV mappings from the binary target.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix. DataFrame column names are preserved; an ndarray
            gets generated names ``x0, x1, ...``. Column dtype drives automatic
            continuous/categorical detection unless overridden via the
            ``categorical_features`` / ``continuous_features`` parameters.
        y : np.ndarray | pd.Series
            Binary target where ``1`` == event (default) and ``0`` == non-event.
            Boolean targets are accepted; any other values raise ``ValueError``.

        Returns
        -------
        WOEEncoder
            The fitted encoder (``self``), to support method chaining.
        """
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {self.n_bins}")
        if not 0.0 <= self.rare_threshold < 1.0:
            raise ValueError(f"rare_threshold must be in [0, 1); got {self.rare_threshold}")

        X = self._to_frame(X)
        y_arr = self._validate_target(y, n_rows=len(X))

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        if self.forbidden_features:
            present = [c for c in self.forbidden_features if c in set(X.columns)]
            if present:
                raise ValueError(
                    f"Forbidden features present for this model: {present}. "
                    "Per regulatory guardrails (governance.forbidden_features), "
                    "these columns must not be used as model inputs. Drop them, or "
                    "omit them from forbidden_features if they are permitted here "
                    "(e.g. state_code is allowed for LGD but forbidden for PD)."
                )

        cat_cols, cont_cols = self._resolve_feature_types(X)
        self._categorical_features_: list[str] = cat_cols
        self._continuous_features_: list[str] = cont_cols

        total_events = float(y_arr.sum())
        total_non_events = float(len(y_arr) - total_events)
        if total_events == 0 or total_non_events == 0:
            raise ValueError(
                "Target must contain both classes (0 and 1); "
                f"got events={int(total_events)}, non_events={int(total_non_events)}"
            )

        self.feature_woe_: dict[str, _FeatureWOE] = {}
        self.iv_: dict[str, float] = {}

        cat_set = set(cat_cols)
        for col in X.columns:
            series = X[col]
            if col in cat_set:
                fw = self._fit_categorical(series, y_arr, total_events, total_non_events)
            else:
                fw = self._fit_continuous(series, y_arr, total_events, total_non_events)
            self.feature_woe_[col] = fw
            self.iv_[col] = fw.iv

        return self

    def _fit_continuous(
        self,
        series: pd.Series,
        y: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> _FeatureWOE:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        not_nan = ~np.isnan(values)

        # Quantile cut points from non-null values; merge duplicate edges.
        cut_points = np.array([], dtype=float)
        if not_nan.sum() > 0:
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.unique(np.nanquantile(values[not_nan], quantiles))
            # keep internal boundaries only (drop outermost min/max)
            cut_points = edges[1:-1] if edges.size >= 2 else np.array([], dtype=float)

        n_bins_eff = len(cut_points) + 1
        events = np.zeros(n_bins_eff, dtype=float)
        non_events = np.zeros(n_bins_eff, dtype=float)
        if not_nan.any():
            bin_idx = np.digitize(values[not_nan], cut_points, right=False)
            y_valid = y[not_nan]
            for b in range(n_bins_eff):
                in_bin = bin_idx == b
                events[b] = y_valid[in_bin].sum()
                non_events[b] = in_bin.sum() - events[b]

        bin_woe, bin_iv = self._woe_iv(events, non_events, total_events, total_non_events)
        nan_woe, nan_iv = self._nan_woe(not_nan, y, total_events, total_non_events)

        return _FeatureWOE(
            kind="continuous",
            iv=float(bin_iv + nan_iv),
            nan_woe=nan_woe,
            cut_points=cut_points,
            bin_woe=bin_woe,
        )

    def _fit_categorical(
        self,
        series: pd.Series,
        y: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> _FeatureWOE:
        values = series.to_numpy(dtype=object)
        is_nan = pd.isna(series).to_numpy()

        # Map categories; merge rare levels into _OTHER.
        non_null = series[~is_nan]
        counts = non_null.value_counts()
        min_count = self.rare_threshold * len(series)
        keep = [c for c in counts.index if counts[c] >= min_count]

        groups: list[object] = [*keep, _OTHER]
        group_index = {g: i for i, g in enumerate(groups)}
        events = np.zeros(len(groups), dtype=float)
        non_events = np.zeros(len(groups), dtype=float)

        for i in range(len(series)):
            if is_nan[i]:
                continue
            gi = group_index.get(values[i], group_index[_OTHER])
            if y[i] == 1:
                events[gi] += 1
            else:
                non_events[gi] += 1

        group_woe, group_iv = self._woe_iv(events, non_events, total_events, total_non_events)

        category_woe = {g: float(group_woe[group_index[g]]) for g in keep}
        other_woe = float(group_woe[group_index[_OTHER]])
        nan_woe, nan_iv = self._nan_woe(~is_nan, y, total_events, total_non_events)

        return _FeatureWOE(
            kind="categorical",
            iv=float(group_iv + nan_iv),
            nan_woe=nan_woe,
            category_woe=category_woe,
            other_woe=other_woe,
        )

    def _nan_woe(
        self,
        not_nan: np.ndarray,
        y: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> tuple[float, float]:
        """WOE and IV contribution for the NaN bin (0.0 / 0.0 if no NaNs)."""
        nan_mask = ~not_nan
        if nan_mask.sum() == 0:
            return 0.0, 0.0
        nan_events = float(y[nan_mask].sum())
        nan_non_events = float(nan_mask.sum() - nan_events)
        woe, iv = self._woe_iv(
            np.array([nan_events]),
            np.array([nan_non_events]),
            total_events,
            total_non_events,
        )
        return float(woe[0]), float(iv)

    def _woe_iv(
        self,
        events: np.ndarray,
        non_events: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> tuple[np.ndarray, float]:
        """Laplace-smoothed, clipped WOE per bin and the summed IV."""
        reg = self.regularization
        n = len(events)
        dist_bad = (events + reg) / (total_events + reg * n)
        dist_good = (non_events + reg) / (total_non_events + reg * n)
        # With reg=0 a pure bin (no events or no non-events) yields ±inf, and a
        # fully empty bin yields 0/0 -> nan. Clip bounds the ±inf to ±woe_clip;
        # nan (degenerate empty bin) maps to a neutral WOE of 0.0.
        with np.errstate(divide="ignore", invalid="ignore"):
            woe = np.log(dist_good / dist_bad)
        woe = np.clip(woe, -self.woe_clip, self.woe_clip)
        woe = np.nan_to_num(woe, nan=0.0)
        iv = float(np.sum((dist_good - dist_bad) * woe))
        return woe, iv

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Replace raw values with their fitted Weight-of-Evidence.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Data containing the columns seen during fit. Unseen categories map
            to the "other" WOE, out-of-range continuous values clamp to the edge
            bins, and NaNs map to the dedicated NaN WOE.

        Returns
        -------
        np.ndarray
            Float array of shape ``(n_samples, n_features_in_)`` with column
            order equal to ``feature_names_in_``. With
            ``set_output(transform="pandas")`` a named DataFrame is returned.
        """
        check_is_fitted(self, "feature_woe_")
        X = self._to_frame(X)
        self._check_columns(X)

        out = np.empty((len(X), self.n_features_in_), dtype=float)
        for j, col in enumerate(self.feature_names_in_):
            fw = self.feature_woe_[col]
            if fw.kind == "continuous":
                out[:, j] = self._transform_continuous(X[col], fw)
            else:
                out[:, j] = self._transform_categorical(X[col], fw)
        return out

    def _transform_continuous(self, series: pd.Series, fw: _FeatureWOE) -> np.ndarray:
        if fw.cut_points is None or fw.bin_woe is None:
            raise RuntimeError("Continuous feature state is missing; was fit() called?")
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        not_nan = ~np.isnan(values)
        result = np.full(len(values), fw.nan_woe, dtype=float)
        if not_nan.any():
            bin_idx = np.digitize(values[not_nan], fw.cut_points, right=False)
            result[not_nan] = fw.bin_woe[bin_idx]
        return result

    def _transform_categorical(self, series: pd.Series, fw: _FeatureWOE) -> np.ndarray:
        is_nan = pd.isna(series).to_numpy()
        values = series.to_numpy(dtype=object)
        result = np.empty(len(values), dtype=float)
        for i in range(len(values)):
            if is_nan[i]:
                result[i] = fw.nan_woe
            else:
                result[i] = fw.category_woe.get(values[i], fw.other_woe)
        return result

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_iv_summary(self) -> pd.DataFrame:
        """
        Return a per-feature IV table sorted by descending IV.

        Columns: feature, iv, strength, leakage_warning, kind.
        """
        check_is_fitted(self, "feature_woe_")
        rows = []
        for col in self.feature_names_in_:
            iv = self.iv_[col]
            rows.append(
                {
                    "feature": col,
                    "iv": iv,
                    "strength": self._iv_strength(iv),
                    "leakage_warning": iv >= 0.30,
                    "kind": self.feature_woe_[col].kind,
                }
            )
        summary = pd.DataFrame(rows)
        return summary.sort_values("iv", ascending=False, ignore_index=True)

    @staticmethod
    def _iv_strength(iv: float) -> str:
        if iv < 0.02:
            return "useless"
        if iv < 0.10:
            return "weak"
        if iv < 0.30:
            return "medium"
        return "strong"

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the output feature names, one per input column ("<col>_woe").

        Parameters
        ----------
        input_features : ignored
            Accepted only for scikit-learn API compatibility; the encoder uses
            the column names captured during fit.

        Returns
        -------
        np.ndarray
            Array of output feature names aligned with ``feature_names_in_``.
        """
        check_is_fitted(self, "feature_woe_")
        return np.asarray([f"{name}_woe" for name in self.feature_names_in_], dtype=object)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"x{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    @staticmethod
    def _validate_target(y: np.ndarray | pd.Series, n_rows: int) -> np.ndarray:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-dimensional.")
        if len(y_arr) != n_rows:
            raise ValueError(f"X has {n_rows} rows but y has {len(y_arr)}.")
        if y_arr.dtype == bool:
            y_arr = y_arr.astype(int)
        unique = set(np.unique(y_arr).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"y must be binary (0/1); got values {sorted(unique)}")
        return y_arr.astype(int)

    def _resolve_feature_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self.categorical_features is not None or self.continuous_features is not None:
            cat = list(self.categorical_features or [])
            cont = list(self.continuous_features or [])
            listed = set(cat) | set(cont)
            for col in X.columns:
                if col in listed:
                    continue
                if self._is_categorical_dtype(X[col]):
                    cat.append(col)
                else:
                    cont.append(col)
            return cat, cont
        cat = [c for c in X.columns if self._is_categorical_dtype(X[c])]
        cont = [c for c in X.columns if c not in set(cat)]
        return cat, cont

    @staticmethod
    def _is_categorical_dtype(series: pd.Series) -> bool:
        return (
            series.dtype == object
            or isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == bool
        )

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing:
            raise ValueError(f"transform is missing columns seen during fit: {missing}")
