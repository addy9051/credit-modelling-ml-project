"""
Unit tests for the Phase 2 feature layer:

  * per-family feature builders (bureau, loan, account, GST, business, macro)
  * the loader abstraction (parquet tabular source, bureau loaders)
  * pipeline assembly + model-type preprocessing pipelines
  * regulatory guardrails: forbidden PD features and outcome leakage

The builders are pure transforms, so most tests feed hand-built frames and
assert on the engineered columns. Assembly / loader tests round-trip through
parquet to exercise the real I/O seam.
"""

from __future__ import annotations

import io
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import account_features as af
from src.feature_engineering import bureau_features as bf
from src.feature_engineering import business_features as busf
from src.feature_engineering import gst_features as gf
from src.feature_engineering import loan_features as lf
from src.feature_engineering import macro_features as mf
from src.feature_engineering.loaders import (
    BUREAU_FEATURE_COLUMNS,
    DecentroBureauLoader,
    ParquetBureauLoader,
    ParquetTabularSource,
)
from src.feature_engineering.pipeline import (
    PD_FORBIDDEN_FEATURES,
    IQRClipper,
    PDFeatureMatrix,
    assemble_pd_feature_matrix,
    build_pd_feature_pipeline,
)

# --------------------------------------------------------------------------- #
# Fixtures — small frames matching the synthetic_generator schemas
# --------------------------------------------------------------------------- #


@pytest.fixture
def loan_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L0", "L1", "L2", "L3"],
            "borrower_id": ["B0", "B1", "B2", "B3"],
            "observation_date": pd.to_datetime(
                ["2022-06-15", "2022-07-20", "2023-01-10", "2023-05-05"]
            ),
            "product_type": ["home_loan", "credit_card", "term_loan_sme", "auto"],
            "borrower_segment": [
                "retail_salaried",
                "retail_salaried",
                "msme",
                "retail_self_employed",
            ],
            "outstanding_principal": [2_000_000.0, 80_000.0, 3_000_000.0, 500_000.0],
            "sanctioned_limit": [2_500_000.0, 100_000.0, 4_000_000.0, 600_000.0],
            "loan_tenor_months": [240, 1, 60, 60],
            "disbursement_date": pd.to_datetime(
                ["2018-01-01", "2022-01-01", "2021-01-01", "2021-01-01"]
            ),
            "months_on_book": [54, 6, 24, 28],
            "collateral_type": ["immovable_property", "none", "mixed", "movable_assets"],
            "ltv_at_origination": [0.72, 1.05, 0.80, 0.85],
            "state_code": ["MH", "DL", "KA", "TN"],
            "sma_flag": [0, 0, 1, 0],
            "npa_flag": [False, False, False, False],
            "default_flag_12m": [False, True, True, False],
            "restructured_flag": [False, False, True, False],
        }
    )


@pytest.fixture
def bureau_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "borrower_id": ["B0", "B1", "B2", "B3"],
            "observation_date": pd.to_datetime(
                ["2022-06-15", "2022-07-20", "2023-01-10", "2023-05-05"]
            ),
            "cibil_score": [780, 540, 690, None],  # B3 is new-to-credit
            "score_type": ["ERS4.0", "ERS4.0", "ERS4.0", None],
            "dpd_30_count_24m": [0, 4, 1, 0],
            "dpd_60_count_24m": [0, 2, 0, 0],
            "dpd_90_count_24m": [0, 1, 0, 0],
            "months_since_last_delinquency": [None, 2, 9, None],
            "revolving_utilisation": [0.15, 0.95, 0.40, None],
            "open_trade_count": [4, 3, 2, 0],
            "oldest_trade_months": [80, 40, 30, None],
            "enquiry_count_6m": [1, 5, 2, 0],
        }
    )


@pytest.fixture
def account_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "borrower_id": ["B0", "B1", "B3"],
            "observation_date": pd.to_datetime(["2022-06-15", "2022-07-20", "2023-05-05"]),
            "avg_monthly_balance_3m": [120_000.0, 8_000.0, 60_000.0],
            "salary_credit_months_12m": [12, 5, 0],
            "nach_bounce_count_12m": [0, 4, 1],
            "cash_withdrawal_ratio": [0.10, 0.65, 0.30],
            "debit_credit_ratio_3m": [0.7, 1.15, 0.9],
        }
    )


@pytest.fixture
def gst_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "borrower_id": ["B2"],
            "observation_date": pd.to_datetime(["2023-01-10"]),
            "gstr1_revenue_monthly": [450_000.0],
            "filing_on_time_flag": [False],
            "itc_to_output_ratio": [0.98],
            "revenue_trend_slope_3m": [-0.05],
            "revenue_trend_slope_12m": [-0.08],
        }
    )


@pytest.fixture
def macro_frame() -> pd.DataFrame:
    dates = pd.date_range("2021-01-31", "2023-12-31", freq="ME")
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "rbi_repo_rate_pct": np.linspace(4.0, 6.5, n),
            "gdp_growth_yoy_pct": np.linspace(8.0, 6.5, n),
            "wpi_inflation_pct": np.full(n, 3.0),
            "cpi_inflation_pct": np.full(n, 5.0),
            "iip_growth_pct": np.full(n, 4.0),
            "msme_npa_index": np.linspace(2.0, 1.2, n),
        }
    )


# --------------------------------------------------------------------------- #
# Bureau features
# --------------------------------------------------------------------------- #


def test_bureau_features_columns_and_ntc(bureau_frame):
    out = bf.build_bureau_features(bureau_frame)
    for col in bf.BUREAU_CONTINUOUS + bf.BUREAU_CATEGORICAL:
        assert col in out.columns

    # B3: no score -> NTC flag set, NaN score, band == "ntc"
    b3 = out.set_index("borrower_id").loc["B3"]
    assert b3["is_new_to_credit"] == 1
    assert pd.isna(b3["cibil_score"])
    assert b3["cibil_band"] == "ntc"
    assert b3["is_thin_file"] == 1  # 0 open trades

    # B1: delinquent -> any_delinquency flag set, "poor" band (540)
    b1 = out.set_index("borrower_id").loc["B1"]
    assert b1["any_delinquency_24m"] == 1
    assert b1["cibil_band"] == "poor"

    # B0: clean -> no delinquency, "very_good"/"excellent" band
    b0 = out.set_index("borrower_id").loc["B0"]
    assert b0["any_delinquency_24m"] == 0
    assert b0["cibil_band"] in {"very_good", "excellent"}


def test_bureau_features_carries_no_audit_or_pii(bureau_frame):
    out = bf.build_bureau_features(bureau_frame)
    for leaked in ("decentro_txn_id", "reference_id", "raw_response_code"):
        assert leaked not in out.columns


# --------------------------------------------------------------------------- #
# Loan features — leakage & derivations
# --------------------------------------------------------------------------- #


def test_loan_features_exclude_outcome_leakage(loan_frame):
    out = lf.build_loan_features(loan_frame)
    # Distress / outcome columns must never surface as PD features.
    for leaked in lf.LEAKAGE_COLUMNS:
        assert leaked not in out.columns


def test_loan_features_derivations(loan_frame):
    out = lf.build_loan_features(loan_frame).set_index("loan_id")

    # drawn_to_sanctioned = outstanding / sanctioned
    assert out.loc["L0", "drawn_to_sanctioned"] == pytest.approx(2_000_000 / 2_500_000)
    assert out.loc["L0", "available_headroom"] == pytest.approx(1 - 2_000_000 / 2_500_000)
    # seasoning_ratio = months_on_book / tenor
    assert out.loc["L0", "seasoning_ratio"] == pytest.approx(54 / 240)
    # secured vs unsecured
    assert out.loc["L0", "is_secured"] == 1
    assert out.loc["L1", "is_secured"] == 0
    # state_code retained (LGD reuse); forbidden-handling happens in assembly
    assert "state_code" in out.columns


# --------------------------------------------------------------------------- #
# Account & GST features
# --------------------------------------------------------------------------- #


def test_account_features(account_frame):
    out = af.build_account_features(account_frame).set_index("borrower_id")
    assert out.loc["B0", "has_nach_bounce"] == 0
    assert out.loc["B1", "has_nach_bounce"] == 1
    assert out.loc["B1", "overspending_flag"] == 1  # dcr 1.15 > 1
    assert out.loc["B3", "no_salary_credit"] == 1  # self-employed, 0 salary months
    assert out.loc["B0", "salary_regularity_12m"] == pytest.approx(1.0)


def test_gst_features(gst_frame):
    out = gf.build_gst_features(gst_frame).set_index("borrower_id")
    assert out.loc["B2", "itc_anomaly_flag"] == 1  # 0.98 > 0.90
    assert out.loc["B2", "revenue_declining_flag"] == 1  # slope_12m < 0
    assert out.loc["B2", "filing_on_time_flag"] == 0


# --------------------------------------------------------------------------- #
# Business features (pure transform over the financials contract)
# --------------------------------------------------------------------------- #


def test_business_features_ratios_and_zero_guard():
    df = pd.DataFrame(
        {
            "borrower_id": ["B2", "B9"],
            "observation_date": pd.to_datetime(["2023-01-10", "2023-02-10"]),
            "cash_flow_available": [1_200_000.0, 500_000.0],
            "debt_service": [1_000_000.0, 0.0],  # zero -> NaN, not inf
            "ebit": [800_000.0, -200_000.0],
            "interest_expense": [400_000.0, 100_000.0],
            "current_assets": [2_000_000.0, 1_000_000.0],
            "current_liabilities": [1_000_000.0, 1_500_000.0],
            "total_debt": [5_000_000.0, 3_000_000.0],
            "total_equity": [2_500_000.0, -500_000.0],
            "auditor_category": ["big4", "local"],
        }
    )
    out = busf.build_business_features(df).set_index("borrower_id")
    assert out.loc["B2", "dscr"] == pytest.approx(1.2)
    assert out.loc["B2", "icr"] == pytest.approx(2.0)
    assert out.loc["B2", "current_ratio"] == pytest.approx(2.0)
    assert out.loc["B2", "dscr_below_1"] == 0
    # zero debt_service -> NaN (no inf)
    assert pd.isna(out.loc["B9", "dscr"])
    assert np.isfinite(out["dscr"].dropna()).all()
    # negative equity flagged
    assert out.loc["B9", "negative_net_worth"] == 1


# --------------------------------------------------------------------------- #
# Macro features & point-in-time join
# --------------------------------------------------------------------------- #


def test_build_macro_features_changes(macro_frame):
    out = mf.build_macro_features(macro_frame)
    assert "repo_rate_change_3m" in out.columns
    assert "real_repo_rate" in out.columns
    # real rate = repo - cpi
    np.testing.assert_allclose(
        out["real_repo_rate"].to_numpy(),
        (out["rbi_repo_rate_pct"] - 5.0).to_numpy(),
    )
    # 3m change is NaN for the first 3 rows, finite after
    assert out["repo_rate_change_3m"].iloc[:3].isna().all()
    assert np.isfinite(out["repo_rate_change_3m"].iloc[3:]).all()


def test_attach_macro_is_point_in_time(loan_frame, macro_frame):
    macro_feats = mf.build_macro_features(macro_frame)
    base = lf.build_loan_features(loan_frame)
    merged = mf.attach_macro_features(base, macro_feats, on="observation_date")

    assert len(merged) == len(base)
    # row order preserved
    assert list(merged["loan_id"]) == list(base["loan_id"])
    # backward join: a 2022-06-15 observation must pick a macro date <= it
    row = merged.set_index("loan_id").loc["L0"]
    # the repo rate attached must equal the most recent month-end <= 2022-06-15
    expected = macro_feats[macro_feats["date"] <= pd.Timestamp("2022-06-15")].iloc[-1]
    assert row["rbi_repo_rate_pct"] == pytest.approx(expected["rbi_repo_rate_pct"])


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _write_synthetic(tmp_path, loan_frame, gst_frame, account_frame, macro_frame):
    loan_frame.to_parquet(tmp_path / "loan_portfolio.parquet")
    gst_frame.to_parquet(tmp_path / "gst_data.parquet")
    account_frame.to_parquet(tmp_path / "account_transactions.parquet")
    macro_frame.to_parquet(tmp_path / "macroeconomic.parquet")


def test_parquet_tabular_source(tmp_path, loan_frame, gst_frame, account_frame, macro_frame):
    _write_synthetic(tmp_path, loan_frame, gst_frame, account_frame, macro_frame)
    src = ParquetTabularSource(tmp_path)
    assert len(src.loans()) == 4
    assert len(src.macro()) == len(macro_frame)
    assert len(src.gst()) == 1


def test_parquet_tabular_source_missing_required_raises(tmp_path):
    src = ParquetTabularSource(tmp_path)
    with pytest.raises(FileNotFoundError):
        src.loans()


def test_parquet_tabular_source_missing_optional_is_empty(tmp_path, loan_frame, macro_frame):
    loan_frame.to_parquet(tmp_path / "loan_portfolio.parquet")
    macro_frame.to_parquet(tmp_path / "macroeconomic.parquet")
    src = ParquetTabularSource(tmp_path)
    assert src.gst().empty
    assert src.transactions().empty


def test_parquet_bureau_loader_roundtrip(tmp_path, bureau_frame):
    path = tmp_path / "bureau.parquet"
    bureau_frame[list(BUREAU_FEATURE_COLUMNS)].to_parquet(path)
    loader = ParquetBureauLoader(path)
    out = loader.load(["B0", "B2", "UNKNOWN"])
    assert list(out["borrower_id"]) == ["B0", "B2"]  # unknown dropped
    assert list(out.columns) == list(BUREAU_FEATURE_COLUMNS)


class _StubDecentroClient:
    """Minimal stand-in for DecentroClient that returns a fixed BureauFeatures."""

    def __init__(self):
        self.calls = []

    def fetch_bureau_features(self, *, borrower_id, mobile, name, **kwargs):
        from src.data_ingestion.bureau_connector import BureauFeatures

        self.calls.append((borrower_id, mobile, name, kwargs))
        return BureauFeatures(
            borrower_id=borrower_id,
            observation_date=kwargs.get("observation_date") or date(2023, 1, 1),
            cibil_score=720,
            score_type="ERS4.0",
            dpd_30_count_24m=0,
            dpd_60_count_24m=0,
            dpd_90_count_24m=0,
            months_since_last_delinquency=None,
            revolving_utilisation=0.2,
            open_trade_count=3,
            oldest_trade_months=50,
            enquiry_count_6m=1,
            decentro_txn_id="TXN",
            reference_id="REF",
            raw_response_code="S00000",
        )


def test_decentro_bureau_loader_skips_unresolved_and_drops_audit():
    pytest.importorskip("requests")  # bureau_connector imports requests (ingest extra)
    identities = pd.DataFrame(
        {
            "borrower_id": ["B0", "B1"],
            "mobile": ["9999999990", "9999999991"],
            "name": ["A", "B"],
        }
    )
    loader = DecentroBureauLoader(_StubDecentroClient(), identities)
    out = loader.load(["B0", "B1", "B_NO_IDENTITY"])
    assert list(out["borrower_id"]) == ["B0", "B1"]  # unresolved skipped
    assert list(out.columns) == list(BUREAU_FEATURE_COLUMNS)
    for audit in ("decentro_txn_id", "reference_id", "raw_response_code"):
        assert audit not in out.columns


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #


@pytest.fixture
def assembled(tmp_path, loan_frame, gst_frame, account_frame, macro_frame, bureau_frame):
    _write_synthetic(tmp_path, loan_frame, gst_frame, account_frame, macro_frame)
    bureau_path = tmp_path / "bureau.parquet"
    bureau_frame[list(BUREAU_FEATURE_COLUMNS)].to_parquet(bureau_path)

    src = ParquetTabularSource(tmp_path)
    bureau_loader = ParquetBureauLoader(bureau_path)
    return assemble_pd_feature_matrix(src, bureau_loader)


def test_assemble_shapes_and_target(assembled, loan_frame):
    assert isinstance(assembled, PDFeatureMatrix)
    assert len(assembled.X) == len(loan_frame)
    assert len(assembled.y) == len(loan_frame)
    # target aligned by loan_id
    assert assembled.y.tolist() == [0, 1, 1, 0]
    # X holds only declared feature columns
    assert set(assembled.X.columns) == set(assembled.feature_names)


def test_assemble_drops_forbidden_features(assembled):
    for forbidden in PD_FORBIDDEN_FEATURES:
        assert forbidden not in assembled.X.columns
        assert forbidden not in assembled.feature_names


def test_assemble_joins_all_families(assembled):
    # one representative continuous column from each present family
    for col in (
        "cibil_score",  # bureau
        "drawn_to_sanctioned",  # loan
        "log_avg_monthly_balance_3m",  # account
        "log_gstr1_revenue_monthly",  # gst
        "rbi_repo_rate_pct",  # macro
    ):
        assert col in assembled.X.columns


def test_assemble_without_bureau(tmp_path, loan_frame, gst_frame, account_frame, macro_frame):
    _write_synthetic(tmp_path, loan_frame, gst_frame, account_frame, macro_frame)
    src = ParquetTabularSource(tmp_path)
    fm = assemble_pd_feature_matrix(src, bureau_loader=None)
    assert "cibil_score" not in fm.X.columns
    assert "drawn_to_sanctioned" in fm.X.columns


# --------------------------------------------------------------------------- #
# Preprocessing pipelines
# --------------------------------------------------------------------------- #


def test_scorecard_pipeline_outputs_finite_woe(assembled):
    pipe = build_pd_feature_pipeline(
        "scorecard",
        assembled.continuous,
        assembled.categorical,
        forbidden_features=PD_FORBIDDEN_FEATURES,
    )
    out = pipe.fit_transform(assembled.X, assembled.y)
    assert out.shape == (len(assembled.X), len(assembled.feature_names))
    assert np.isfinite(out).all()


def test_tree_pipeline_no_nan(assembled):
    pipe = build_pd_feature_pipeline("tree", assembled.continuous, assembled.categorical)
    out = pipe.fit_transform(assembled.X, assembled.y)
    assert out.shape[0] == len(assembled.X)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_neural_pipeline_scaled_and_finite(assembled):
    pipe = build_pd_feature_pipeline("neural", assembled.continuous, assembled.categorical)
    out = pipe.fit_transform(assembled.X, assembled.y)
    assert np.isfinite(np.asarray(out, dtype=float)).all()
    assert out.shape[0] == len(assembled.X)


def test_pipeline_invalid_model_type_raises(assembled):
    with pytest.raises(ValueError, match="model_type"):
        build_pd_feature_pipeline("logistic", assembled.continuous, assembled.categorical)


def test_pipeline_forbidden_feature_in_lists_raises():
    with pytest.raises(ValueError, match="[Ff]orbidden"):
        build_pd_feature_pipeline(
            "tree",
            continuous_features=["ltv_at_origination"],
            categorical_features=["state_code"],
            forbidden_features=PD_FORBIDDEN_FEATURES,
        )


def test_pipeline_joblib_roundtrip(assembled):
    pipe = build_pd_feature_pipeline("scorecard", assembled.continuous, assembled.categorical)
    expected = pipe.fit_transform(assembled.X, assembled.y)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)
    np.testing.assert_allclose(loaded.transform(assembled.X), expected)


# --------------------------------------------------------------------------- #
# IQRClipper
# --------------------------------------------------------------------------- #


def test_iqr_clipper_bounds_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    x[0, 0] = 1e6  # extreme outlier
    clipper = IQRClipper(k=1.5).fit(x)
    out = clipper.transform(x)
    assert out[0, 0] < 1e6  # clamped
    assert (out <= clipper.upper_ + 1e-9).all()
    assert (out >= clipper.lower_ - 1e-9).all()


def test_iqr_clipper_uses_fit_bounds_at_transform():
    train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    clipper = IQRClipper(k=0.0).fit(train)  # k=0 -> clip to [Q1, Q3]
    # transform-time extreme must clamp to the fitted upper bound, not its own
    out = clipper.transform(np.array([[999.0]]))
    assert out[0, 0] == pytest.approx(clipper.upper_[0])
