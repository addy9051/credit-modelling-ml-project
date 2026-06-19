"""
Macroeconomic features and the point-in-time attachment of macro state to loans.

Two pieces:

* :func:`build_macro_features` derives a few signals from the monthly macro
  series (repo-rate momentum, a real-rate proxy) on top of the raw indicators.
* :func:`attach_macro_features` joins those onto a borrower frame **as of**
  each ``observation_date`` using a backward ``merge_asof``. This is what keeps
  the join point-in-time: a loan observed mid-month picks up the most recent
  macro reading *at or before* that date and never a future one, so no macro
  look-ahead leaks into the PD features.

Macro is one shared table for the whole book (joined by date), so it has no
``borrower_id`` and is not, on its own, keyed per borrower.
"""

from __future__ import annotations

import pandas as pd

# Macro has no borrower key; it joins onto loans by date.
DATE_COLUMN = "date"

MACRO_CONTINUOUS = [
    "rbi_repo_rate_pct",
    "repo_rate_change_3m",
    "repo_rate_change_12m",
    "gdp_growth_yoy_pct",
    "wpi_inflation_pct",
    "cpi_inflation_pct",
    "iip_growth_pct",
    "msme_npa_index",
    "real_repo_rate",
]

MACRO_CATEGORICAL: list[str] = []


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive macro features from the monthly macro series.

    Parameters
    ----------
    df : pd.DataFrame
        ``macroeconomic`` schema from ``synthetic_generator`` (monthly, with a
        ``date`` column).

    Returns
    -------
    pd.DataFrame
        Sorted by ``date`` with :data:`MACRO_CONTINUOUS` plus the ``date`` key.
    """
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN])
    out = out.sort_values(DATE_COLUMN).reset_index(drop=True)

    repo = pd.to_numeric(out["rbi_repo_rate_pct"], errors="coerce")
    # Monthly series -> 3- and 12-month changes capture the rate cycle.
    out["repo_rate_change_3m"] = repo - repo.shift(3)
    out["repo_rate_change_12m"] = repo - repo.shift(12)
    # Real policy rate proxy (nominal repo minus CPI inflation).
    out["real_repo_rate"] = repo - pd.to_numeric(out["cpi_inflation_pct"], errors="coerce")

    return out[[DATE_COLUMN, *MACRO_CONTINUOUS]]


def attach_macro_features(
    frame: pd.DataFrame,
    macro_features: pd.DataFrame,
    on: str = "observation_date",
) -> pd.DataFrame:
    """
    Backward as-of join macro features onto a borrower frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Borrower-level frame carrying the observation-date column ``on``.
    macro_features : pd.DataFrame
        Output of :func:`build_macro_features` (must contain :data:`DATE_COLUMN`).
    on : str
        Observation-date column in ``frame`` to align against the macro date.

    Returns
    -------
    pd.DataFrame
        ``frame`` with :data:`MACRO_CONTINUOUS` appended, each row carrying the
        most recent macro reading at or before its ``on`` date.
    """
    left = frame.copy()
    left[on] = pd.to_datetime(left[on])
    # merge_asof requires both keys sorted ascending; preserve original row order.
    left["_row_order"] = range(len(left))
    left = left.sort_values(on)

    right = macro_features.copy()
    right[DATE_COLUMN] = pd.to_datetime(right[DATE_COLUMN])
    right = right.sort_values(DATE_COLUMN)

    merged = pd.merge_asof(
        left,
        right,
        left_on=on,
        right_on=DATE_COLUMN,
        direction="backward",
    )
    merged = merged.sort_values("_row_order").drop(columns=["_row_order", DATE_COLUMN])
    return merged.reset_index(drop=True)


__all__ = [
    "build_macro_features",
    "attach_macro_features",
    "MACRO_CONTINUOUS",
    "MACRO_CATEGORICAL",
    "DATE_COLUMN",
]
