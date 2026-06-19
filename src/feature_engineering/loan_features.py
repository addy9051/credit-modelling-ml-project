"""
Loan-level features: LTV, drawn/sanctioned ratio (a CCF precursor), seasoning,
product / collateral structure.

Consumes the loan portfolio frame from ``synthetic_generator`` (or any source
with the same schema).

Leakage guardrail (PD): the raw loan table carries point-in-time *distress*
indicators — ``sma_flag``, ``npa_flag``, ``restructured_flag`` — and the target
``default_flag_12m`` is itself derived from ``npa_flag`` / ``sma_flag``. Surfacing
those as PD features leaks the outcome, so they are **deliberately not emitted**
here. They belong to the label, not the application-time feature set.

Forbidden-feature note: ``state_code`` *is* emitted (this builder is shared with
the LGD pipeline, where state-level legal-enforcement efficiency is a legitimate
driver). For PD it is dropped during assembly and blocked by the scorecard
WOEEncoder's ``forbidden_features`` guard.

Pure transform: input frame in, feature frame out. No I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLUMNS = ["loan_id", "borrower_id", "observation_date"]

#: Continuous loan features.
LOAN_CONTINUOUS = [
    "ltv_at_origination",
    "log_outstanding_principal",
    "log_sanctioned_limit",
    "drawn_to_sanctioned",
    "available_headroom",
    "loan_tenor_months",
    "months_on_book",
    "seasoning_ratio",
]

#: Categorical / binary loan features. ``state_code`` is forbidden for PD (see
#: module docstring) but kept for LGD reuse.
LOAN_CATEGORICAL = [
    "product_type",
    "borrower_segment",
    "collateral_type",
    "is_secured",
    "state_code",
]

# Distress / outcome columns that must never become PD features (leakage).
LEAKAGE_COLUMNS = ["sma_flag", "npa_flag", "restructured_flag", "default_flag_12m"]


def build_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer loan-level features from the loan portfolio frame.

    Parameters
    ----------
    df : pd.DataFrame
        Loan portfolio with the ``synthetic_generator`` schema.

    Returns
    -------
    pd.DataFrame
        ``KEY_COLUMNS`` + :data:`LOAN_CONTINUOUS` + :data:`LOAN_CATEGORICAL`.
        Outcome-derived distress columns are intentionally excluded.
    """
    out = pd.DataFrame()
    out["loan_id"] = df["loan_id"].astype(str)
    out["borrower_id"] = df["borrower_id"].astype(str)
    out["observation_date"] = df["observation_date"]

    outstanding = pd.to_numeric(df["outstanding_principal"], errors="coerce")
    sanctioned = pd.to_numeric(df["sanctioned_limit"], errors="coerce")

    out["ltv_at_origination"] = pd.to_numeric(df["ltv_at_origination"], errors="coerce")
    # log1p tames the heavy right tail of dispersed principal/limit amounts.
    out["log_outstanding_principal"] = np.log1p(outstanding.clip(lower=0))
    out["log_sanctioned_limit"] = np.log1p(sanctioned.clip(lower=0))

    # Drawn fraction of the sanctioned limit — the precursor to the EAD/CCF model.
    # Guard against zero/NaN limits; over-limit (>1) is meaningful, so don't cap.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawn = outstanding / sanctioned.where(sanctioned > 0)
    out["drawn_to_sanctioned"] = drawn
    out["available_headroom"] = 1.0 - drawn

    tenor = pd.to_numeric(df["loan_tenor_months"], errors="coerce")
    mob = pd.to_numeric(df["months_on_book"], errors="coerce")
    out["loan_tenor_months"] = tenor
    out["months_on_book"] = mob
    with np.errstate(divide="ignore", invalid="ignore"):
        out["seasoning_ratio"] = mob / tenor.where(tenor > 0)

    out["product_type"] = df["product_type"].astype(object)
    out["borrower_segment"] = df["borrower_segment"].astype(object)
    out["collateral_type"] = df["collateral_type"].astype(object)
    out["is_secured"] = (df["collateral_type"].astype(object) != "none").astype(int)
    out["state_code"] = df["state_code"].astype(object)

    return out


__all__ = [
    "build_loan_features",
    "LOAN_CONTINUOUS",
    "LOAN_CATEGORICAL",
    "LEAKAGE_COLUMNS",
    "KEY_COLUMNS",
]
