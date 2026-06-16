"""
Unit tests for src.feature_engineering.woe_encoder.WOEEncoder.

Covers the acceptance criteria in project plan §5.1 / §11.1:
  - WOE distribution sum property and non-negative IV
  - unseen categories map to the "other" bin (no KeyError)
  - fit_transform == fit then transform
  - missing values get a separate WOE bin
  - WOE clipping to [-clip, clip]
  - continuous auto-binning + out-of-range clamping at transform time
  - joblib round-trip serialisation
  - get_feature_names_out / get_iv_summary
  - target validation and leakage flagging
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.woe_encoder import WOEEncoder

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def credit_like_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A small credit-like dataset where risk decreases as score increases."""
    rng = np.random.default_rng(42)
    n = 2000
    cibil = rng.integers(300, 900, size=n)
    # default probability falls with score: ~0.6 at 300 -> ~0.02 at 900
    p_default = np.clip(0.9 - (cibil - 300) / 600 * 0.88, 0.01, 0.95)
    y = (rng.random(n) < p_default).astype(int)

    segment = rng.choice(["salaried", "self_employed", "msme"], size=n, p=[0.5, 0.3, 0.2])
    utilisation = np.clip(rng.beta(2, 5, size=n) + (y * 0.2), 0, 1.2)

    X = pd.DataFrame(
        {
            "cibil_score": cibil.astype(float),
            "revolving_utilisation": utilisation,
            "borrower_segment": segment,
        }
    )
    return X, y


# --------------------------------------------------------------------------- #
# Core WOE / IV properties
# --------------------------------------------------------------------------- #


def test_fit_transform_shape_and_names(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder(n_bins=10)
    out = enc.fit_transform(X, y)

    assert out.shape == (len(X), X.shape[1])
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(
        enc.get_feature_names_out(),
        np.array(
            ["cibil_score_woe", "revolving_utilisation_woe", "borrower_segment_woe"], dtype=object
        ),
    )


def test_iv_non_negative(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)
    for feature, iv in enc.iv_.items():
        assert iv >= 0.0, f"IV for {feature} was negative: {iv}"


def test_woe_distribution_sum_property(credit_like_data):
    """Σ (dist_bad_i - dist_good_i) == 0 over all bins (distributions sum to 1)."""
    X, y = credit_like_data
    enc = WOEEncoder(n_bins=10, regularization=0.0).fit(X, y)

    values = pd.to_numeric(X["cibil_score"]).to_numpy(dtype=float)
    total_events = float(y.sum())
    total_non_events = float(len(y) - total_events)

    fw = enc.feature_woe_["cibil_score"]
    bin_idx = np.digitize(values, fw.cut_points, right=False)
    dist_diff = 0.0
    for b in range(len(fw.bin_woe)):
        in_bin = bin_idx == b
        events = y[in_bin].sum()
        non_events = in_bin.sum() - events
        dist_diff += events / total_events - non_events / total_non_events
    assert abs(dist_diff) < 1e-9


def test_cibil_is_predictive(credit_like_data):
    """A score strongly tied to default should rank as at least 'medium' IV."""
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)
    summary = enc.get_iv_summary()
    cibil_iv = summary.loc[summary["feature"] == "cibil_score", "iv"].iloc[0]
    assert cibil_iv >= 0.10


def test_woe_monotonic_direction(credit_like_data):
    """Higher CIBIL bins (lower risk) should carry higher WOE than low bins."""
    X, y = credit_like_data
    enc = WOEEncoder(n_bins=10).fit(X, y)
    fw = enc.feature_woe_["cibil_score"]
    # not strictly monotonic, but lowest-score bin should have lower WOE than highest
    assert fw.bin_woe[0] < fw.bin_woe[-1]


# --------------------------------------------------------------------------- #
# Categorical handling
# --------------------------------------------------------------------------- #


def test_unseen_category_maps_to_other(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)

    X_new = X.head(5).copy()
    X_new["borrower_segment"] = ["salaried", "self_employed", "msme", "ASTRONAUT", "PIRATE"]
    out = enc.transform(X_new)  # must not raise

    seg_col = list(enc.feature_names_in_).index("borrower_segment")
    other_woe = enc.feature_woe_["borrower_segment"].other_woe
    # unseen categories (rows 3, 4) take the "other" WOE
    assert out[3, seg_col] == pytest.approx(other_woe)
    assert out[4, seg_col] == pytest.approx(other_woe)


def test_rare_categories_merged_into_other():
    rng = np.random.default_rng(0)
    n = 1000
    cats = np.array(["A"] * 480 + ["B"] * 480 + ["RARE"] * 40, dtype=object)
    rng.shuffle(cats)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"cat": cats})

    enc = WOEEncoder(rare_threshold=0.05).fit(X, y)  # RARE is 4% < 5% -> merged
    fw = enc.feature_woe_["cat"]
    assert "RARE" not in fw.category_woe
    assert set(fw.category_woe.keys()) == {"A", "B"}


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #


def test_missing_values_get_separate_bin():
    rng = np.random.default_rng(1)
    n = 1000
    x = rng.normal(size=n)
    x[:100] = np.nan  # 10% missing
    y = (rng.random(n) < 0.25).astype(int)
    # make NaN rows default-heavy so the NaN bin carries signal
    y[:100] = (rng.random(100) < 0.7).astype(int)
    X = pd.DataFrame({"num": x})

    enc = WOEEncoder().fit(X, y)
    fw = enc.feature_woe_["num"]
    # NaN rows are riskier -> NaN WOE should be negative (fewer goods)
    assert fw.nan_woe < 0.0

    out = enc.transform(X)
    assert out[:100, 0] == pytest.approx(fw.nan_woe)
    assert np.isfinite(out).all()


def test_categorical_missing_handled():
    rng = np.random.default_rng(2)
    n = 600
    cats = np.array(["X", "Y", None] * 200, dtype=object)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"cat": cats})
    enc = WOEEncoder().fit(X, y)
    out = enc.transform(X)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Clipping and numerical stability
# --------------------------------------------------------------------------- #


def test_woe_clipping_bounds():
    """Perfectly separating feature would give ±inf WOE without clipping."""
    x = np.r_[np.zeros(200), np.ones(200)]
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)  # x==1 <=> default
    X = pd.DataFrame({"perfect": x})

    enc = WOEEncoder(n_bins=2, woe_clip=3.0).fit(X, y)
    out = enc.transform(X)
    assert np.isfinite(out).all()
    assert out.min() >= -3.0 - 1e-9
    assert out.max() <= 3.0 + 1e-9


def test_perfect_separation_flagged_as_leakage():
    x = np.r_[np.zeros(200), np.ones(200)]
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    X = pd.DataFrame({"perfect": x})
    enc = WOEEncoder(n_bins=2).fit(X, y)
    summary = enc.get_iv_summary()
    assert bool(summary.loc[summary["feature"] == "perfect", "leakage_warning"].iloc[0]) is True
    assert summary.loc[summary["feature"] == "perfect", "strength"].iloc[0] == "strong"


def test_constant_feature_is_useless():
    rng = np.random.default_rng(3)
    n = 500
    X = pd.DataFrame({"const": np.full(n, 7.0)})
    y = (rng.random(n) < 0.3).astype(int)
    enc = WOEEncoder().fit(X, y)
    assert enc.iv_["const"] == pytest.approx(0.0, abs=1e-9)
    assert enc.get_iv_summary().iloc[0]["strength"] == "useless"


# --------------------------------------------------------------------------- #
# Continuous binning / out-of-range
# --------------------------------------------------------------------------- #


def test_out_of_range_values_clamped(credit_like_data):
    """Values beyond the fitted range map to the edge bins, not NaN/error."""
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)

    X_new = X.head(3).copy()
    X_new["cibil_score"] = [-999.0, 99999.0, 600.0]  # below min, above max, in range
    out = enc.transform(X_new)
    col = list(enc.feature_names_in_).index("cibil_score")
    fw = enc.feature_woe_["cibil_score"]
    assert out[0, col] == pytest.approx(fw.bin_woe[0])  # clamped to first bin
    assert out[1, col] == pytest.approx(fw.bin_woe[-1])  # clamped to last bin
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# sklearn contract
# --------------------------------------------------------------------------- #


def test_fit_transform_equals_fit_then_transform(credit_like_data):
    X, y = credit_like_data
    a = WOEEncoder(n_bins=8).fit_transform(X, y)
    enc = WOEEncoder(n_bins=8).fit(X, y)
    b = enc.transform(X)
    np.testing.assert_allclose(a, b)


def test_joblib_roundtrip(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)
    expected = enc.transform(X)

    buffer = io.BytesIO()
    joblib.dump(enc, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)

    np.testing.assert_allclose(loaded.transform(X), expected)
    assert loaded.iv_ == enc.iv_


def test_set_output_pandas(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().set_output(transform="pandas").fit(X, y)
    out = enc.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "cibil_score_woe",
        "revolving_utilisation_woe",
        "borrower_segment_woe",
    ]


def test_transform_with_numpy_input():
    rng = np.random.default_rng(5)
    n = 500
    arr = rng.normal(size=(n, 2))
    y = (rng.random(n) < 0.3).astype(int)
    enc = WOEEncoder()
    out = enc.fit_transform(arr, y)
    assert out.shape == (n, 2)
    assert list(enc.get_feature_names_out()) == ["x0_woe", "x1_woe"]


def test_explicit_feature_types():
    rng = np.random.default_rng(6)
    n = 800
    # numeric column that should be treated as categorical (few levels)
    grade = rng.integers(1, 5, size=n).astype(float)
    amount = rng.lognormal(mean=10, sigma=1, size=n)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"grade": grade, "amount": amount})

    enc = WOEEncoder(categorical_features=["grade"]).fit(X, y)
    assert enc.feature_woe_["grade"].kind == "categorical"
    assert enc.feature_woe_["amount"].kind == "continuous"


# --------------------------------------------------------------------------- #
# Validation / error handling
# --------------------------------------------------------------------------- #


def test_non_binary_target_raises(credit_like_data):
    X, _ = credit_like_data
    y_bad = np.random.default_rng(7).integers(0, 3, size=len(X))  # values 0,1,2
    with pytest.raises(ValueError, match="binary"):
        WOEEncoder().fit(X, y_bad)


def test_single_class_target_raises(credit_like_data):
    X, _ = credit_like_data
    y_all_zero = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        WOEEncoder().fit(X, y_all_zero)


def test_boolean_target_accepted(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y.astype(bool))
    assert all(iv >= 0 for iv in enc.iv_.values())


def test_transform_missing_column_raises(credit_like_data):
    X, y = credit_like_data
    enc = WOEEncoder().fit(X, y)
    with pytest.raises(ValueError, match="missing columns"):
        enc.transform(X.drop(columns=["cibil_score"]))


def test_invalid_n_bins_raises(credit_like_data):
    X, y = credit_like_data
    with pytest.raises(ValueError, match="n_bins"):
        WOEEncoder(n_bins=1).fit(X, y)
