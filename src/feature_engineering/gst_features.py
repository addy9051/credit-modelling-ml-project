"""
GST features for the MSME segment: revenue level and trend, filing discipline,
and input-tax-credit (ITC) behaviour.

Consumes the ``gst_data`` frame (MSME borrowers only) from
``synthetic_generator``. An inflated ITC-to-output ratio is a known stress /
fraud signal; a declining revenue slope and missed filings precede default.

Pure transform: input frame in, feature frame out. No I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLUMNS = ["borrower_id", "observation_date"]

GST_CONTINUOUS = [
    "log_gstr1_revenue_monthly",
    "itc_to_output_ratio",
    "revenue_trend_slope_3m",
    "revenue_trend_slope_12m",
]

GST_CATEGORICAL = [
    "filing_on_time_flag",
    "itc_anomaly_flag",
    "revenue_declining_flag",
]

# An ITC-to-output ratio above this is anomalous (claiming more credit than
# output tax plausibly supports) — a fraud / stress marker.
_ITC_ANOMALY_THRESHOLD = 0.90


def build_gst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer MSME GST features from the GST-signals frame.

    Parameters
    ----------
    df : pd.DataFrame
        ``gst_data`` schema from ``synthetic_generator``.

    Returns
    -------
    pd.DataFrame
        ``KEY_COLUMNS`` + :data:`GST_CONTINUOUS` + :data:`GST_CATEGORICAL`.
    """
    out = pd.DataFrame()
    out["borrower_id"] = df["borrower_id"].astype(str)
    out["observation_date"] = df["observation_date"]

    revenue = pd.to_numeric(df["gstr1_revenue_monthly"], errors="coerce")
    out["log_gstr1_revenue_monthly"] = np.log1p(revenue.clip(lower=0))

    itc = pd.to_numeric(df["itc_to_output_ratio"], errors="coerce")
    out["itc_to_output_ratio"] = itc

    slope_3m = pd.to_numeric(df["revenue_trend_slope_3m"], errors="coerce")
    slope_12m = pd.to_numeric(df["revenue_trend_slope_12m"], errors="coerce")
    out["revenue_trend_slope_3m"] = slope_3m
    out["revenue_trend_slope_12m"] = slope_12m

    out["filing_on_time_flag"] = df["filing_on_time_flag"].astype(bool).astype(int)
    out["itc_anomaly_flag"] = (itc > _ITC_ANOMALY_THRESHOLD).astype(int)
    out["revenue_declining_flag"] = (slope_12m < 0).astype(int)

    return out


__all__ = [
    "build_gst_features",
    "GST_CONTINUOUS",
    "GST_CATEGORICAL",
    "KEY_COLUMNS",
]
