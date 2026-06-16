"""
Synthetic data generator for the credit-risk platform.

Generates all training data that cannot be sourced from the Decentro bureau
API: the loan portfolio (including the PD target label), GST signals, bank
account transaction signals, and the macroeconomic time series.

Bureau features (CIBIL score, DPD history, enquiry counts, revolving
utilisation) are intentionally EXCLUDED here.  In production those come from
src/data_ingestion/bureau_connector.py via the Decentro credit report API.
For training, they are fetched from the Decentro sandbox and joined onto the
loan portfolio by borrower_id.

Outputs (all written to data/synthetic/ as parquet):
  loan_portfolio.parquet      Loan-level records with default_flag_12m label
  gst_data.parquet            GST signals for the MSME borrower segment
  account_transactions.parquet  UPI / bank statement signals (retail segment)
  macroeconomic.parquet       Monthly macro indicators FY18-FY25

Calibration anchors (from config.data.portfolio_stats):
  ICICI Bank Q2 FY25:     GNPA 1.97%  (corporate / large-ticket default rate)
  Bajaj Finance FY24:     GNPA 0.90%  (consumer / MSME default rate)
  BSR-1 sector mix:       mortgages 31%, SME 14%, commercial 13%
  CRILC SMA patterns:     SMA-0/1/2 reclassification timing

Usage:
  uv run python -m src.data_ingestion.synthetic_generator --config config/development.yaml
  uv run python -m src.data_ingestion.synthetic_generator --config config/development.yaml --n-loans 10000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

_PRODUCT_TYPES = [
    "home_loan",
    "lap",
    "auto",
    "personal_loan",
    "credit_card",
    "working_capital",
    "term_loan_sme",
    "term_loan_corporate",
]

# Bajaj Finance AUM mix (Dec-2024 disclosure); maps broad product groups to weights
_PRODUCT_WEIGHTS = {
    "home_loan": 0.20,
    "lap": 0.11,  # mortgages 31%
    "personal_loan": 0.15,
    "credit_card": 0.10,  # consumer 25%
    "term_loan_sme": 0.09,
    "working_capital": 0.05,  # SME 14%
    "term_loan_corporate": 0.13,  # commercial 13%
    "auto": 0.17,  # other 17%
}

# Base annual default rates from published disclosures (to monthly flag probability)
_DEFAULT_RATE_BY_PRODUCT = {
    "home_loan": 0.0090,  # Bajaj Finance mortgage book
    "lap": 0.0090,
    "auto": 0.0090,
    "personal_loan": 0.0090,
    "credit_card": 0.0090,
    "working_capital": 0.0090,
    "term_loan_sme": 0.0090,
    "term_loan_corporate": 0.0197,  # ICICI Bank corporate GNPA
}

# Indian state codes weighted by BSR-1 regional loan book distribution
_STATE_CODES = [
    "MH",
    "DL",
    "KA",
    "TN",
    "GJ",
    "UP",
    "RJ",
    "AP",
    "WB",
    "TS",
    "MP",
    "HR",
    "KL",
    "PB",
    "BR",
    "OR",
    "UK",
    "HP",
    "CG",
    "JH",
]
_STATE_WEIGHTS = [
    0.18,
    0.13,
    0.09,
    0.08,
    0.08,
    0.07,
    0.05,
    0.05,
    0.04,
    0.04,
    0.03,
    0.03,
    0.03,
    0.03,
    0.02,
    0.02,
    0.01,
    0.01,
    0.01,
    0.01,
]

_BORROWER_SEGMENTS = ["retail_salaried", "retail_self_employed", "msme", "corporate"]
_COLLATERAL_TYPES = ["immovable_property", "movable_assets", "none", "mixed"]


# --------------------------------------------------------------------------- #
# Loan portfolio
# --------------------------------------------------------------------------- #


def _generate_loan_portfolio(n_loans: int, seed: int) -> pd.DataFrame:
    """
    Generate n_loans loan-level records spanning FY18-FY25 (monthly snapshots).

    Key calibration:
      - Product mix anchored to Bajaj Finance AUM (Dec-2024)
      - Default rates anchored to ICICI (corporate) and Bajaj Finance (retail/SME)
      - SMA flag progression follows CRILC reclassification patterns
      - LTV bands follow RBI prudential norms per product
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d loan portfolio records …", n_loans)

    # --- Identifiers ---
    loan_ids = [f"L{i:08d}" for i in range(n_loans)]
    borrower_ids = [f"B{i:08d}" for i in range(n_loans)]

    # --- Product mix ---
    products = rng.choice(
        list(_PRODUCT_WEIGHTS.keys()),
        size=n_loans,
        p=list(_PRODUCT_WEIGHTS.values()),
    )

    # --- Borrower segment (correlated with product) ---
    segment_map = {
        "home_loan": "retail_salaried",
        "lap": "retail_self_employed",
        "auto": "retail_salaried",
        "personal_loan": "retail_salaried",
        "credit_card": "retail_salaried",
        "working_capital": "msme",
        "term_loan_sme": "msme",
        "term_loan_corporate": "corporate",
    }
    segments = np.array([segment_map[p] for p in products])

    # --- Observation date (monthly snapshots FY18-FY25) ---
    start = pd.Timestamp("2017-04-30")
    end = pd.Timestamp("2024-12-31")
    n_months = int((end - start).days / 30) + 1
    obs_dates = pd.to_datetime(
        [start + pd.DateOffset(months=rng.integers(0, n_months)) for _ in range(n_loans)]
    ).normalize()

    # --- Disbursement dates (before observation) ---
    months_on_book = rng.integers(1, 120, size=n_loans)
    disbursement_dates = obs_dates - pd.to_timedelta(months_on_book * 30, unit="D")

    # --- Outstanding principal (log-normal, product-specific scale) ---
    principal_scale = {
        "home_loan": 3_500_000,
        "lap": 2_500_000,
        "auto": 800_000,
        "personal_loan": 300_000,
        "credit_card": 100_000,
        "working_capital": 5_000_000,
        "term_loan_sme": 3_000_000,
        "term_loan_corporate": 50_000_000,
    }
    outstanding_principal = np.array(
        [rng.lognormal(mean=np.log(principal_scale[p] * 0.7), sigma=0.6) for p in products]
    )

    # --- Sanctioned limit (>= outstanding) ---
    drawdown_ratio = rng.uniform(0.5, 1.0, size=n_loans)
    sanctioned_limit = outstanding_principal / np.clip(drawdown_ratio, 0.3, 1.0)

    # --- LTV at origination (product-specific, RBI norms) ---
    ltv_ranges = {
        "home_loan": (0.60, 0.80),
        "lap": (0.50, 0.75),
        "auto": (0.70, 0.90),
        "personal_loan": (0.80, 1.10),
        "credit_card": (0.80, 1.20),
        "working_capital": (0.50, 1.00),
        "term_loan_sme": (0.60, 0.90),
        "term_loan_corporate": (0.40, 0.80),
    }
    ltv_at_origination = np.array([rng.uniform(*ltv_ranges[p]) for p in products])

    # --- Collateral ---
    collateral_map = {
        "home_loan": "immovable_property",
        "lap": "immovable_property",
        "auto": "movable_assets",
        "personal_loan": "none",
        "credit_card": "none",
        "working_capital": "mixed",
        "term_loan_sme": "mixed",
        "term_loan_corporate": "mixed",
    }
    collateral_types = np.array([collateral_map[p] for p in products])

    # --- Tenor ---
    tenor_ranges = {
        "home_loan": (120, 300),
        "lap": (60, 240),
        "auto": (36, 84),
        "personal_loan": (12, 60),
        "credit_card": (1, 1),
        "working_capital": (12, 36),
        "term_loan_sme": (36, 120),
        "term_loan_corporate": (36, 180),
    }
    loan_tenor_months = np.array([rng.integers(*tenor_ranges[p]) for p in products])

    # --- State ---
    state_codes = rng.choice(_STATE_CODES, size=n_loans, p=_STATE_WEIGHTS)

    # --- SMA flags (Markov-chain progression) ---
    # SMA-0 → SMA-1 → SMA-2 → NPA follows CRILC reclassification patterns
    base_sma1_rate = 0.04
    base_sma2_rate = 0.02
    sma_flag = np.zeros(n_loans, dtype=int)
    sma_mask = rng.random(n_loans)
    sma_flag[sma_mask < base_sma1_rate + base_sma2_rate] = 1
    sma_flag[sma_mask < base_sma2_rate] = 2

    npa_flag = sma_flag == 2

    # --- Default flag (12-month outcome label for PD model) ---
    # Calibrated to: corporate 1.97% GNPA (ICICI), consumer/SME 0.90% (Bajaj Finance)
    default_prob = np.array([_DEFAULT_RATE_BY_PRODUCT[p] for p in products])
    # Accounts already in SMA-2/NPA are definitionally defaulted
    default_prob = np.where(npa_flag, 1.0, default_prob)
    # SMA-1 has elevated probability
    default_prob = np.where(sma_flag == 1, default_prob * 4, default_prob)
    default_flag_12m = (rng.random(n_loans) < default_prob).astype(bool)

    # --- Restructured flag (post-COVID RBI restructuring norms) ---
    restructured_flag = (rng.random(n_loans) < 0.015).astype(bool)

    df = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "borrower_id": borrower_ids,
            "observation_date": obs_dates,
            "product_type": products,
            "borrower_segment": segments,
            "outstanding_principal": outstanding_principal.round(2),
            "sanctioned_limit": sanctioned_limit.round(2),
            "loan_tenor_months": loan_tenor_months,
            "disbursement_date": disbursement_dates,
            "months_on_book": months_on_book,
            "collateral_type": collateral_types,
            "ltv_at_origination": ltv_at_origination.round(4),
            "state_code": state_codes,
            "sma_flag": sma_flag,
            "npa_flag": npa_flag,
            "default_flag_12m": default_flag_12m,
            "restructured_flag": restructured_flag,
        }
    )

    actual_default_rate = df["default_flag_12m"].mean()
    logger.info(
        "Loan portfolio generated. Default rate: %.3f%% (target: ~1.2%%)",
        actual_default_rate * 100,
    )
    return df


# --------------------------------------------------------------------------- #
# GST data (MSME segment only)
# --------------------------------------------------------------------------- #


def _generate_gst_data(loan_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    GST signals for MSME borrowers only (borrower_segment == 'msme').

    Defaulted borrowers exhibit: declining revenue trend, missed filings,
    lower ITC-to-output ratio. Calibrated to GSTN filing patterns.
    """
    rng = np.random.default_rng(seed + 1)
    msme = loan_df[loan_df["borrower_segment"] == "msme"][
        ["borrower_id", "observation_date", "default_flag_12m"]
    ].copy()

    if msme.empty:
        return pd.DataFrame()

    n = len(msme)
    is_default = msme["default_flag_12m"].values

    # Revenue: defaulters have lower, more volatile revenues
    base_revenue = rng.lognormal(mean=np.log(800_000), sigma=0.8, size=n)
    base_revenue = np.where(is_default, base_revenue * 0.6, base_revenue)

    # Trend slope: negative for stressed borrowers
    slope_3m = rng.normal(0.02, 0.08, size=n)
    slope_3m = np.where(is_default, rng.normal(-0.05, 0.10, size=n), slope_3m)
    slope_12m = rng.normal(0.05, 0.12, size=n)
    slope_12m = np.where(is_default, rng.normal(-0.08, 0.12, size=n), slope_12m)

    # Filing regularity: defaulters miss more filings
    filing_on_time_prob = np.where(is_default, 0.55, 0.87)
    filing_on_time_flag = rng.random(n) < filing_on_time_prob

    # ITC ratio: defaulters inflate ITC (a fraud/stress signal)
    itc_ratio = rng.uniform(0.55, 0.85, size=n)
    itc_ratio = np.where(is_default, rng.uniform(0.70, 1.05, size=n), itc_ratio)

    return pd.DataFrame(
        {
            "borrower_id": msme["borrower_id"].values,
            "observation_date": msme["observation_date"].values,
            "gstr1_revenue_monthly": base_revenue.round(2),
            "filing_on_time_flag": filing_on_time_flag,
            "itc_to_output_ratio": itc_ratio.round(4),
            "revenue_trend_slope_3m": slope_3m.round(6),
            "revenue_trend_slope_12m": slope_12m.round(6),
        }
    )


# --------------------------------------------------------------------------- #
# Account transactions (retail segments)
# --------------------------------------------------------------------------- #


def _generate_account_transactions(loan_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    UPI / bank-statement signals for retail_salaried and retail_self_employed.

    Signals are correlated with the default label: defaulters show declining
    AMB, higher NACH bounce rates, and lower salary-credit regularity.
    """
    rng = np.random.default_rng(seed + 2)
    retail = loan_df[loan_df["borrower_segment"].isin(["retail_salaried", "retail_self_employed"])][
        ["borrower_id", "observation_date", "borrower_segment", "default_flag_12m"]
    ].copy()

    if retail.empty:
        return pd.DataFrame()

    n = len(retail)
    is_default = retail["default_flag_12m"].values
    is_salaried = (retail["borrower_segment"] == "retail_salaried").values

    # Average monthly balance (3-month): defaulters show lower AMB
    amb_base = np.where(is_salaried, 45_000, 80_000)
    avg_monthly_balance_3m = rng.lognormal(mean=np.log(amb_base), sigma=0.7)
    avg_monthly_balance_3m = np.where(
        is_default, avg_monthly_balance_3m * rng.uniform(0.3, 0.6, n), avg_monthly_balance_3m
    )

    # Salary credits in last 12 months (salaried only; 0 for self-employed)
    salary_credit_months_12m = np.where(
        is_salaried,
        np.where(is_default, rng.integers(4, 10, n), rng.integers(9, 13, n)),
        0,
    ).astype(int)

    # NACH bounce count: defaulters have more bounces
    nach_bounce_mean = np.where(is_default, 3.5, 0.3)
    nach_bounce_count_12m = rng.poisson(nach_bounce_mean).astype(int)

    # Cash withdrawal ratio (cash out / total debit): high = stress signal
    cash_withdrawal_ratio = rng.beta(1.5, 5.0, size=n)
    cash_withdrawal_ratio = np.where(
        is_default, np.clip(cash_withdrawal_ratio * 2.0, 0, 0.9), cash_withdrawal_ratio
    )

    # Debit/credit ratio (3-month): >1 means spending > income
    debit_credit_ratio_3m = rng.uniform(0.5, 0.95, size=n)
    debit_credit_ratio_3m = np.where(
        is_default, rng.uniform(0.85, 1.20, size=n), debit_credit_ratio_3m
    )

    return pd.DataFrame(
        {
            "borrower_id": retail["borrower_id"].values,
            "observation_date": retail["observation_date"].values,
            "avg_monthly_balance_3m": avg_monthly_balance_3m.round(2),
            "salary_credit_months_12m": salary_credit_months_12m,
            "nach_bounce_count_12m": nach_bounce_count_12m,
            "cash_withdrawal_ratio": cash_withdrawal_ratio.round(4),
            "debit_credit_ratio_3m": debit_credit_ratio_3m.round(4),
        }
    )


# --------------------------------------------------------------------------- #
# Macroeconomic time series (FY18–FY25)
# --------------------------------------------------------------------------- #


def _generate_macroeconomic(seed: int) -> pd.DataFrame:
    """
    Monthly macro indicators from April 2017 to December 2024.

    Key anchors (actual RBI / MOSPI data):
      - Repo rate: 6.0% (FY18) → 6.5% (FY20, pre-COVID) → 4.0% (COVID low)
                   → 6.5% (FY23 hiking cycle) → 6.5% (FY25)
      - GDP growth: ~7% pre-COVID → -6.6% (FY21) → recovery
      - MSME NPA index: elevated during COVID and NBFC crisis windows

    Macro is the same table for all borrowers — joined by observation_date
    in the feature pipeline.
    """
    rng = np.random.default_rng(seed + 3)
    dates = pd.date_range(start="2017-04-30", end="2024-12-31", freq="ME")
    n = len(dates)

    # Repo rate path (actual trajectory with noise)
    repo_path = np.array(
        [
            6.00,
            6.00,
            6.00,
            6.00,
            6.00,
            6.00,  # FY18 H1
            6.00,
            6.00,
            6.00,
            6.00,
            6.00,
            6.00,  # FY18 H2
            6.00,
            6.00,
            6.25,
            6.25,
            6.50,
            6.50,  # FY19 H1
            6.50,
            6.50,
            6.50,
            6.25,
            6.00,
            5.75,  # FY19 H2
            5.75,
            5.75,
            5.75,
            5.75,
            5.75,
            5.40,  # FY20 H1
            5.15,
            5.15,
            4.40,
            4.00,
            4.00,
            4.00,  # FY20 H2 + COVID cut
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,  # FY21 H1
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,  # FY21 H2
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,  # FY22 H1
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,
            4.00,  # FY22 H2
            4.00,
            4.00,
            4.40,
            4.90,
            5.40,
            5.90,  # FY23 H1 hiking cycle
            6.25,
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,  # FY23 H2
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,  # FY24 H1
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,  # FY24 H2
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,
            6.50,  # FY25 H1
            6.25,
            6.25,
            6.25,
            6.25,
            6.25,
            6.25,  # FY25 H2 (slight easing)
            6.00,
            6.00,
            6.00,  # partial FY26
        ]
    )
    repo = repo_path[:n] + rng.normal(0, 0.05, n)

    # GDP growth (annual; interpolated monthly)
    gdp_annual = [7.2, 6.1, 4.0, -6.6, 8.7, 7.0, 6.5, 6.5]  # FY18-FY25
    gdp_growth = np.interp(np.linspace(0, 7, n), np.arange(8), gdp_annual)
    gdp_growth += rng.normal(0, 0.3, n)

    # Inflation (WPI)
    wpi = 2.5 + rng.normal(0, 0.8, n)
    wpi[30:50] += 4.0  # COVID supply disruption spike
    wpi[50:60] -= 1.0  # recovery

    # CPI
    cpi = 4.5 + rng.normal(0, 0.5, n)
    cpi[30:45] += 1.5

    # IIP growth (Industrial Production Index)
    iip = 4.0 + rng.normal(0, 1.5, n)
    iip[30:42] -= 12.0  # COVID lockdown shock

    # MSME NPA index (1.0 = baseline; >1 = elevated stress)
    msme_npa_index = np.ones(n)
    msme_npa_index[18:30] = rng.uniform(1.2, 1.5, 12)  # NBFC crisis FY19
    msme_npa_index[36:54] = rng.uniform(1.8, 2.8, 18)  # COVID stress FY21-22
    msme_npa_index[54:66] = rng.uniform(1.1, 1.4, 12)  # gradual recovery

    return pd.DataFrame(
        {
            "date": dates,
            "rbi_repo_rate_pct": np.clip(repo, 3.5, 7.5).round(2),
            "gdp_growth_yoy_pct": gdp_growth.round(2),
            "wpi_inflation_pct": wpi.round(2),
            "cpi_inflation_pct": cpi.round(2),
            "iip_growth_pct": iip.round(2),
            "msme_npa_index": np.clip(msme_npa_index, 0.8, 3.5).round(3),
        }
    )


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


def _write_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    size_mb = path.stat().st_size / 1_048_576
    logger.info("Wrote %s → %s (%.1f MB, %d rows)", name, path, size_mb, len(df))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate(config: dict, n_loans: int | None = None, out_dir: Path | None = None) -> None:
    """
    Generate and write all non-bureau synthetic datasets.

    Args:
        config:   Loaded base.yaml config dict.
        n_loans:  Override the config's synthetic_n_loans (useful in tests).
        out_dir:  Override the config's data.synthetic_dir.
    """
    seed: int = config["project"]["random_seed"]
    n = n_loans or config.get("data", {}).get("synthetic_n_loans", 50_000)
    dest = out_dir or Path(config["data"]["synthetic_dir"])

    logger.info("Generating synthetic non-bureau datasets: n_loans=%d seed=%d", n, seed)

    # Loan portfolio is the backbone; other tables join on borrower_id
    loan_df = _generate_loan_portfolio(n, seed)
    _write_parquet(loan_df, dest / "loan_portfolio.parquet", "loan_portfolio")

    gst_df = _generate_gst_data(loan_df, seed)
    if not gst_df.empty:
        _write_parquet(gst_df, dest / "gst_data.parquet", "gst_data")

    txn_df = _generate_account_transactions(loan_df, seed)
    if not txn_df.empty:
        _write_parquet(txn_df, dest / "account_transactions.parquet", "account_transactions")

    macro_df = _generate_macroeconomic(seed)
    _write_parquet(macro_df, dest / "macroeconomic.parquet", "macroeconomic")

    logger.info("Synthetic generation complete.")
    _print_summary(loan_df, gst_df, txn_df, macro_df)


def _print_summary(loan_df, gst_df, txn_df, macro_df) -> None:
    print("\n── Synthetic dataset summary ──────────────────────────────")
    print(f"  loan_portfolio      : {len(loan_df):>8,} rows")
    print(f"  gst_data (MSME)     : {len(gst_df):>8,} rows")
    print(f"  account_transactions: {len(txn_df):>8,} rows")
    print(f"  macroeconomic       : {len(macro_df):>8,} rows (monthly FY18-FY25)")
    overall_default = loan_df["default_flag_12m"].mean() * 100
    print(f"\n  Overall default rate  : {overall_default:.3f}%")
    by_product = (
        loan_df.groupby("product_type")["default_flag_12m"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "default_rate", "count": "n_loans"})
    )
    by_product["default_rate"] = (by_product["default_rate"] * 100).round(3)
    print(f"\n  Default rate by product:\n{by_product.to_string()}")
    print("\n  NOTE: Bureau features (CIBIL score, DPD counts, enquiries,")
    print("  revolving utilisation) are fetched via src/data_ingestion/")
    print("  bureau_connector.py (Decentro API), not generated here.")
    print("──────────────────────────────────────────────────────────\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic non-bureau training data.")
    parser.add_argument(
        "--config",
        default="config/development.yaml",
        help="Config YAML path (default: config/development.yaml)",
    )
    parser.add_argument(
        "--n-loans", type=int, default=None, help="Override config's synthetic_n_loans"
    )
    parser.add_argument("--out-dir", default=None, help="Override config's data.synthetic_dir")
    args = parser.parse_args()

    # Load base.yaml then overlay the specified config
    base_cfg: dict = {}
    for cfg_path in ("config/base.yaml", args.config):
        p = Path(cfg_path)
        if p.exists():
            with open(p) as f:
                overlay = yaml.safe_load(f) or {}
            _deep_merge(base_cfg, overlay)

    out_dir = Path(args.out_dir) if args.out_dir else None
    generate(base_cfg, n_loans=args.n_loans, out_dir=out_dir)


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


if __name__ == "__main__":
    main()
