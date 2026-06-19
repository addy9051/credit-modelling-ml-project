"""
PD feature pipeline: assembly of the family feature builders into a single
model-ready matrix, and the model-type-specific preprocessing pipelines.

Two public entry points:

* :func:`assemble_pd_feature_matrix` — wires the loaders and the per-family
  builders together: loan + bureau + account + GST features joined by borrower,
  with macro state attached point-in-time. It strips outcome-leakage columns and
  (for PD) the regulator-forbidden ``state_code`` / ``gender`` / ``religion``,
  and returns the feature frame, the aligned target, and the continuous /
  categorical column split.

* :func:`build_pd_feature_pipeline` — returns an sklearn ``Pipeline`` for one of
  ``{"scorecard", "tree", "neural"}``:

    ===========  =================================================================
    model_type   preprocessing
    ===========  =================================================================
    scorecard    WOE encoding (NaN -> own bin; ``forbidden_features`` guard armed)
    tree         median / most-frequent imputation + ordinal encoding (no scaling)
    neural       median imputation + IQR clipping + standardisation; one-hot cats
    ===========  =================================================================

  All pipelines are joblib-serialisable for MLflow artifact logging.

The forbidden-feature guardrail is enforced in both places: assembly drops the
columns, and :func:`build_pd_feature_pipeline` raises if a caller still slips
one into the feature lists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.feature_engineering import account_features as af
from src.feature_engineering import bureau_features as bf
from src.feature_engineering import gst_features as gf
from src.feature_engineering import loan_features as lf
from src.feature_engineering import macro_features as mf
from src.feature_engineering.loaders import BureauLoader, TabularSource
from src.feature_engineering.woe_encoder import WOEEncoder

#: Regulator-forbidden inputs for the PD model (governance.forbidden_features).
#: ``state_code`` is permitted for LGD, so it is excluded only here.
PD_FORBIDDEN_FEATURES: tuple[str, ...] = ("state_code", "gender", "religion")

_MODEL_TYPES = ("scorecard", "tree", "neural")
_META_COLUMNS = ["loan_id", "borrower_id", "observation_date"]


# --------------------------------------------------------------------------- #
# Assembled matrix container
# --------------------------------------------------------------------------- #


@dataclass
class PDFeatureMatrix:
    """Result of :func:`assemble_pd_feature_matrix`."""

    X: pd.DataFrame
    y: pd.Series
    continuous: list[str]
    categorical: list[str]
    meta: pd.DataFrame = field(repr=False)

    @property
    def feature_names(self) -> list[str]:
        return [*self.continuous, *self.categorical]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble_pd_feature_matrix(
    tabular_source: TabularSource,
    bureau_loader: BureauLoader | None = None,
    *,
    target_column: str = "default_flag_12m",
    forbidden_features: Sequence[str] = PD_FORBIDDEN_FEATURES,
    drop_forbidden: bool = True,
) -> PDFeatureMatrix:
    """
    Build the PD feature matrix from the loaders and family builders.

    The loan portfolio is the backbone (one row per loan). Bureau, account and
    GST features are left-joined by ``borrower_id`` (so borrowers outside a
    segment simply carry NaNs the pipeline imputers handle); macro state is
    attached point-in-time on ``observation_date``.

    Parameters
    ----------
    tabular_source : TabularSource
        Source for the non-bureau tables (see :mod:`.loaders`).
    bureau_loader : BureauLoader | None
        Source for bureau features. If ``None``, bureau features are omitted
        (useful for tests / non-bureau ablations).
    target_column : str
        Binary PD label column in the loan portfolio.
    forbidden_features : Sequence[str]
        Columns to drop for PD when ``drop_forbidden`` is True.
    drop_forbidden : bool
        Drop ``forbidden_features`` from the assembled matrix (default True).

    Returns
    -------
    PDFeatureMatrix
        ``X`` (features only), ``y`` (aligned target), the continuous /
        categorical column split, and ``meta`` (loan/borrower/observation keys).
    """
    loans = tabular_source.loans()
    if target_column not in loans.columns:
        raise ValueError(f"Loan portfolio has no target column {target_column!r}.")

    base = lf.build_loan_features(loans)
    continuous = list(lf.LOAN_CONTINUOUS)
    categorical = list(lf.LOAN_CATEGORICAL)

    borrower_ids = base["borrower_id"].tolist()

    if bureau_loader is not None:
        bureau_raw = bureau_loader.load(borrower_ids)
        if not bureau_raw.empty:
            bureau_feats = bf.build_bureau_features(bureau_raw)
            base = _join_by_borrower(base, bureau_feats)
            continuous += bf.BUREAU_CONTINUOUS
            categorical += bf.BUREAU_CATEGORICAL

    transactions = tabular_source.transactions()
    if not transactions.empty:
        account_feats = af.build_account_features(transactions)
        base = _join_by_borrower(base, account_feats)
        continuous += af.ACCOUNT_CONTINUOUS
        categorical += af.ACCOUNT_CATEGORICAL

    gst = tabular_source.gst()
    if not gst.empty:
        gst_feats = gf.build_gst_features(gst)
        base = _join_by_borrower(base, gst_feats)
        continuous += gf.GST_CONTINUOUS
        categorical += gf.GST_CATEGORICAL

    macro = tabular_source.macro()
    if not macro.empty:
        macro_feats = mf.build_macro_features(macro)
        base = mf.attach_macro_features(base, macro_feats, on="observation_date")
        continuous += mf.MACRO_CONTINUOUS

    # Align the target to the assembled (left-join-preserved) loan order by id.
    target_by_id = loans.set_index(loans["loan_id"].astype(str))[target_column]
    y = base["loan_id"].astype(str).map(target_by_id).astype(int)
    y.name = target_column

    if drop_forbidden:
        forbidden_set = set(forbidden_features)
        continuous = [c for c in continuous if c not in forbidden_set]
        categorical = [c for c in categorical if c not in forbidden_set]

    # Keep only columns that actually materialised (segment-specific families may
    # be entirely absent for a given draw).
    continuous = [c for c in continuous if c in base.columns]
    categorical = [c for c in categorical if c in base.columns]

    meta = base[_META_COLUMNS].reset_index(drop=True)
    X = base[[*continuous, *categorical]].reset_index(drop=True)
    return PDFeatureMatrix(
        X=X,
        y=y.reset_index(drop=True),
        continuous=continuous,
        categorical=categorical,
        meta=meta,
    )


def _join_by_borrower(base: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join a family feature frame onto the backbone by ``borrower_id``.

    The family's own key columns (``borrower_id`` is the join key; its
    ``observation_date`` duplicates the backbone's) are dropped from the right
    side so only engineered feature columns are added.
    """
    right = family.drop(columns=["observation_date"], errors="ignore")
    if right["borrower_id"].duplicated().any():
        raise ValueError(
            "Family feature frame has duplicate borrower_id rows; expected one row "
            "per borrower for a point-in-time join."
        )
    return base.merge(right, on="borrower_id", how="left")


# --------------------------------------------------------------------------- #
# Preprocessing pipelines
# --------------------------------------------------------------------------- #


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Winsorise continuous features to ``[Q1 - k·IQR, Q3 + k·IQR]`` per column.

    Bounds are learned at fit time and applied at transform time, so the same
    clip limits travel with the serialised pipeline (no leakage from transform
    data). Intended to sit after imputation and before standardisation in the
    neural pipeline; trees and WOE are outlier-robust and don't need it.
    """

    def __init__(self, k: float = 1.5) -> None:
        self.k = k

    def fit(self, X, y=None) -> IQRClipper:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def build_pd_feature_pipeline(
    model_type: str,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    forbidden_features: Sequence[str] | None = None,
    n_bins: int = 10,
    iqr_k: float = 1.5,
) -> Pipeline:
    """
    Build the preprocessing ``Pipeline`` for a PD model family.

    Parameters
    ----------
    model_type : {"scorecard", "tree", "neural"}
        Selects the encoding strategy (see module table).
    continuous_features, categorical_features : Sequence[str]
        Column names to route through numeric / categorical preprocessing.
        Typically ``PDFeatureMatrix.continuous`` / ``.categorical``.
    forbidden_features : Sequence[str] | None
        Regulatory guardrail. If any appear in the feature lists, raises
        ``ValueError``. For the scorecard it is also handed to the WOEEncoder as
        a defence-in-depth fit-time guard. Pass :data:`PD_FORBIDDEN_FEATURES`
        for PD; leave ``None`` for LGD (where ``state_code`` is allowed).
    n_bins : int
        Quantile bins for the scorecard WOE encoder.
    iqr_k : float
        IQR multiplier for the neural pipeline's clipper.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A joblib-serialisable preprocessing pipeline.
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_MODEL_TYPES}; got {model_type!r}")

    continuous = list(continuous_features)
    categorical = list(categorical_features)
    _guard_forbidden(continuous, categorical, forbidden_features)

    if model_type == "scorecard":
        woe = WOEEncoder(
            n_bins=n_bins,
            categorical_features=categorical,
            continuous_features=continuous,
            forbidden_features=list(forbidden_features) if forbidden_features else None,
        )
        return Pipeline([("woe", woe)])

    if model_type == "tree":
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        preprocess = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), continuous),
                ("cat", categorical_pipe, categorical),
            ],
            remainder="drop",
        )
        return Pipeline([("preprocess", preprocess)])

    # neural
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clip", IQRClipper(k=iqr_k)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, continuous),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess)])


def _guard_forbidden(
    continuous: Sequence[str],
    categorical: Sequence[str],
    forbidden_features: Sequence[str] | None,
) -> None:
    if not forbidden_features:
        return
    present = sorted(set(forbidden_features) & (set(continuous) | set(categorical)))
    if present:
        raise ValueError(
            f"Forbidden features present in the PD feature lists: {present}. "
            "Per governance.forbidden_features these must not drive the PD model. "
            "Drop them before building the pipeline (assemble_pd_feature_matrix "
            "does this by default)."
        )


__all__ = [
    "PDFeatureMatrix",
    "assemble_pd_feature_matrix",
    "build_pd_feature_pipeline",
    "IQRClipper",
    "PD_FORBIDDEN_FEATURES",
]
