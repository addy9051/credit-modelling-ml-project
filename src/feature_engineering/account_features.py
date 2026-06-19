"""
Account / bank-statement features for the retail segments: average monthly
balance, salary-credit regularity, NACH bounce behaviour, cash-out intensity,
debit/credit pressure.

Consumes the ``account_transactions`` frame (retail_salaried /
retail_self_employed) from ``synthetic_generator``.

Pure transform: input frame in, feature frame out. No I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLUMNS = ["borrower_id", "observation_date"]

ACCOUNT_CONTINUOUS = [
    "log_avg_monthly_balance_3m",
    "salary_credit_months_12m",
    "salary_regularity_12m",
    "nach_bounce_count_12m",
    "cash_withdrawal_ratio",
    "debit_credit_ratio_3m",
]

ACCOUNT_CATEGORICAL = [
    "has_nach_bounce",
    "no_salary_credit",
    "overspending_flag",
]


def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer account-behaviour features from the transaction-signals frame.

    Parameters
    ----------
    df : pd.DataFrame
        ``account_transactions`` schema from ``synthetic_generator``.

    Returns
    -------
    pd.DataFrame
        ``KEY_COLUMNS`` + :data:`ACCOUNT_CONTINUOUS` + :data:`ACCOUNT_CATEGORICAL`.
    """
    out = pd.DataFrame()
    out["borrower_id"] = df["borrower_id"].astype(str)
    out["observation_date"] = df["observation_date"]

    amb = pd.to_numeric(df["avg_monthly_balance_3m"], errors="coerce")
    out["log_avg_monthly_balance_3m"] = np.log1p(amb.clip(lower=0))

    salary_months = pd.to_numeric(df["salary_credit_months_12m"], errors="coerce")
    out["salary_credit_months_12m"] = salary_months
    out["salary_regularity_12m"] = salary_months / 12.0

    bounces = pd.to_numeric(df["nach_bounce_count_12m"], errors="coerce")
    out["nach_bounce_count_12m"] = bounces

    out["cash_withdrawal_ratio"] = pd.to_numeric(df["cash_withdrawal_ratio"], errors="coerce")
    dcr = pd.to_numeric(df["debit_credit_ratio_3m"], errors="coerce")
    out["debit_credit_ratio_3m"] = dcr

    out["has_nach_bounce"] = (bounces.fillna(0) > 0).astype(int)
    # Self-employed borrowers legitimately have 0 salary credits; the flag is a
    # signal, not a defect — WOE will learn its (segment-conditional) weight.
    out["no_salary_credit"] = (salary_months.fillna(0) == 0).astype(int)
    out["overspending_flag"] = (dcr > 1.0).astype(int)

    return out


__all__ = [
    "build_account_features",
    "ACCOUNT_CONTINUOUS",
    "ACCOUNT_CATEGORICAL",
    "KEY_COLUMNS",
]
