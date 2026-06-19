"""
Bureau-derived features for the PD model.

Consumes the normalised bureau table produced by a
``src.feature_engineering.loaders.BureauLoader`` (columns ==
``loaders.BUREAU_FEATURE_COLUMNS``) and turns it into model-ready features:
the raw bureau signals plus a few interpretable derivations (CIBIL band,
new-to-credit / thin-file flags, any-delinquency flag).

These are the strongest univariate PD predictors — ``cibil_score``,
``dpd_90_count_24m`` and ``revolving_utilisation`` are the
``governance.required_top_features`` that must rank in the top-10 SHAP
importance. None of these features are forbidden; bureau data carries no
``state_code`` / ``gender`` / ``religion``.

Pure transform: input frame in, feature frame out. No I/O.
"""

from __future__ import annotations

import pandas as pd

# Key columns carried through for joining; not model inputs themselves.
KEY_COLUMNS = ["borrower_id", "observation_date"]

#: Continuous bureau features (raw signals; missing values get their own WOE bin).
BUREAU_CONTINUOUS = [
    "cibil_score",
    "dpd_30_count_24m",
    "dpd_60_count_24m",
    "dpd_90_count_24m",
    "months_since_last_delinquency",
    "revolving_utilisation",
    "open_trade_count",
    "oldest_trade_months",
    "enquiry_count_6m",
]

#: Categorical / binary bureau features (explicit levels for WOE / one-hot).
BUREAU_CATEGORICAL = [
    "cibil_band",
    "is_new_to_credit",
    "is_thin_file",
    "any_delinquency_24m",
]

# CIBIL score bands (right-open). NTC borrowers (no score) are labelled "ntc".
_CIBIL_BAND_EDGES = [300, 650, 700, 750, 800, 901]
_CIBIL_BAND_LABELS = ["poor", "fair", "good", "very_good", "excellent"]


def build_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer PD bureau features from a normalised bureau frame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``loaders.BUREAU_FEATURE_COLUMNS``. ``cibil_score`` and the
        optional-count fields may carry NaN / None for new-to-credit borrowers.

    Returns
    -------
    pd.DataFrame
        ``KEY_COLUMNS`` + :data:`BUREAU_CONTINUOUS` + :data:`BUREAU_CATEGORICAL`.
    """
    out = pd.DataFrame()
    out["borrower_id"] = df["borrower_id"].astype(str)
    out["observation_date"] = df["observation_date"]

    score = pd.to_numeric(df["cibil_score"], errors="coerce")
    out["cibil_score"] = score

    for col in (
        "dpd_30_count_24m",
        "dpd_60_count_24m",
        "dpd_90_count_24m",
        "months_since_last_delinquency",
        "revolving_utilisation",
        "open_trade_count",
        "oldest_trade_months",
        "enquiry_count_6m",
    ):
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # CIBIL band — NTC (missing score) gets its own level rather than a numeric edge.
    band = pd.cut(
        score,
        bins=_CIBIL_BAND_EDGES,
        labels=_CIBIL_BAND_LABELS,
        right=False,
    ).astype(object)
    out["cibil_band"] = band.where(score.notna(), other="ntc")

    out["is_new_to_credit"] = score.isna().astype(int)
    out["is_thin_file"] = (out["open_trade_count"].fillna(0) < 2).astype(int)
    any_dpd = (
        out[["dpd_30_count_24m", "dpd_60_count_24m", "dpd_90_count_24m"]].fillna(0).sum(axis=1)
    )
    out["any_delinquency_24m"] = (any_dpd > 0).astype(int)

    return out


__all__ = [
    "build_bureau_features",
    "BUREAU_CONTINUOUS",
    "BUREAU_CATEGORICAL",
    "KEY_COLUMNS",
]
