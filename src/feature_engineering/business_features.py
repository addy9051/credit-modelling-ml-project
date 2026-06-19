"""
MSME financial-statement features: debt-service coverage (DSCR), interest
coverage (ICR), current ratio, leverage, and auditor quality.

Source note: there is no synthetic generator for company financials yet — these
come from parsed annual reports / financial statements (Phase 1
``src.data_ingestion.ar_parser``). This builder is therefore a pure transform
over the agreed *financials contract* below, so it is fully unit-testable today
and will wire straight onto the parser's output when it lands.

Financials contract (one row per borrower-observation):
    borrower_id, observation_date,
    cash_flow_available, debt_service,      -> DSCR
    ebit, interest_expense,                 -> ICR
    current_assets, current_liabilities,    -> current ratio
    total_debt, total_equity,               -> debt/equity, net-worth sign
    auditor_category                        -> auditor quality (categorical)

Pure transform: input frame in, feature frame out. No I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLUMNS = ["borrower_id", "observation_date"]

BUSINESS_CONTINUOUS = [
    "dscr",
    "icr",
    "current_ratio",
    "debt_to_equity",
]

BUSINESS_CATEGORICAL = [
    "auditor_quality",
    "dscr_below_1",
    "negative_net_worth",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Elementwise ratio with non-positive (zero or negative) denominators -> NaN.

    Non-positive denominators are nonsensical for these financial ratios (e.g.
    negative interest expense or liabilities), and a negative denominator would
    flip the ratio's sign and read as deceptively healthy. Negative net worth is
    instead surfaced explicitly by the ``negative_net_worth`` flag.
    """
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / denom.where(denom > 0)
    return ratio.replace([np.inf, -np.inf], np.nan)


def build_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer MSME financial-statement features from the financials contract.

    Parameters
    ----------
    df : pd.DataFrame
        Frame matching the financials contract in the module docstring.

    Returns
    -------
    pd.DataFrame
        ``KEY_COLUMNS`` + :data:`BUSINESS_CONTINUOUS` + :data:`BUSINESS_CATEGORICAL`.
    """
    out = pd.DataFrame()
    out["borrower_id"] = df["borrower_id"].astype(str)
    out["observation_date"] = df["observation_date"]

    out["dscr"] = _safe_ratio(df["cash_flow_available"], df["debt_service"])
    out["icr"] = _safe_ratio(df["ebit"], df["interest_expense"])
    out["current_ratio"] = _safe_ratio(df["current_assets"], df["current_liabilities"])
    out["debt_to_equity"] = _safe_ratio(df["total_debt"], df["total_equity"])

    out["auditor_quality"] = df["auditor_category"].astype(object)
    out["dscr_below_1"] = (out["dscr"] < 1.0).astype(int)
    equity = pd.to_numeric(df["total_equity"], errors="coerce")
    out["negative_net_worth"] = (equity < 0).astype(int)

    return out


__all__ = [
    "build_business_features",
    "BUSINESS_CONTINUOUS",
    "BUSINESS_CATEGORICAL",
    "KEY_COLUMNS",
]
