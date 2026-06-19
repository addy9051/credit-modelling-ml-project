"""
Feature data loaders — the single seam between feature logic and data sources.

The feature builders in this package (``loan_features``, ``bureau_features`` …)
are deliberately *pure*: they take a raw ``pd.DataFrame`` and return engineered
columns. They never read files or call APIs. This module owns the I/O so that
*where* the raw data comes from can change without touching any feature logic.

Two source families exist:

* **Non-bureau tables** — the loan portfolio, GST signals, account-transaction
  signals and the macro time series. These are produced by
  ``src.data_ingestion.synthetic_generator`` and read from parquet
  (:class:`ParquetTabularSource`).

* **Bureau features** — the one place where the source genuinely diverges
  between training and serving. Both paths sit behind the :class:`BureauLoader`
  interface so callers depend only on the contract:

    * :class:`DecentroBureauLoader` — pulls live (or sandbox-fixture) reports via
      ``src.data_ingestion.bureau_connector.DecentroClient``. This is the
      real-time scoring path (Phase 6) and small-batch path.
    * :class:`ParquetBureauLoader` — reads a pre-materialised parquet of bureau
      pulls. This is the offline / training path, where pulling hundreds of
      thousands of reports per epoch is neither affordable nor reproducible.

Swapping one for the other never changes a feature builder or the pipeline.

The Decentro client (and therefore ``requests``) lives in the optional
``ingest`` extra, so :class:`DecentroBureauLoader` imports it lazily — importing
this module never requires the extra.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import pandas as pd

if TYPE_CHECKING:  # avoid importing the ingest extra at runtime
    from src.data_ingestion.bureau_connector import DecentroClient

logger = logging.getLogger(__name__)

# Audit fields on BureauFeatures are dropped before features are exposed:
# decentro_txn_id / reference_id / raw_response_code are for the DPDP audit
# trail, not model inputs.
_BUREAU_AUDIT_FIELDS = ("decentro_txn_id", "reference_id", "raw_response_code")

#: Columns a :class:`BureauLoader` must return, in order. Mirrors the
#: feature-relevant fields of ``bureau_connector.BureauFeatures``.
BUREAU_FEATURE_COLUMNS: tuple[str, ...] = (
    "borrower_id",
    "observation_date",
    "cibil_score",
    "score_type",
    "dpd_30_count_24m",
    "dpd_60_count_24m",
    "dpd_90_count_24m",
    "months_since_last_delinquency",
    "revolving_utilisation",
    "open_trade_count",
    "oldest_trade_months",
    "enquiry_count_6m",
)

# PII columns DecentroBureauLoader requires to resolve a borrower to a bureau
# pull. Optional match fields (date_of_birth / pincode / pan) and per-row
# overrides (observation_date / fixture_profile) are read opportunistically.
_REQUIRED_IDENTITY_COLUMNS = ("mobile", "name")


# --------------------------------------------------------------------------- #
# Non-bureau tabular source
# --------------------------------------------------------------------------- #


@runtime_checkable
class TabularSource(Protocol):
    """Read-side contract for the non-bureau synthetic tables."""

    def loans(self) -> pd.DataFrame: ...
    def gst(self) -> pd.DataFrame: ...
    def transactions(self) -> pd.DataFrame: ...
    def macro(self) -> pd.DataFrame: ...


class ParquetTabularSource:
    """
    Read the synthetic non-bureau tables from a parquet directory.

    ``loans`` and ``macro`` are required (the loan portfolio is the backbone and
    macro always exists); a missing one raises ``FileNotFoundError``. ``gst`` and
    ``transactions`` are segment-specific and may legitimately be absent for a
    given draw (e.g. no MSME borrowers) — those return an empty DataFrame.

    Parameters
    ----------
    synthetic_dir : str | Path
        Directory holding ``loan_portfolio.parquet`` etc. — typically
        ``config["data"]["synthetic_dir"]``.
    """

    _REQUIRED = {"loans": "loan_portfolio.parquet", "macro": "macroeconomic.parquet"}
    _OPTIONAL = {"gst": "gst_data.parquet", "transactions": "account_transactions.parquet"}

    def __init__(self, synthetic_dir: str | Path) -> None:
        self.synthetic_dir = Path(synthetic_dir)

    def _read(self, filename: str, *, required: bool) -> pd.DataFrame:
        path = self.synthetic_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required synthetic table {path} not found. Run "
                    "`python -m src.data_ingestion.synthetic_generator` first."
                )
            logger.info("Optional table %s absent; returning empty frame.", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def loans(self) -> pd.DataFrame:
        return self._read(self._REQUIRED["loans"], required=True)

    def macro(self) -> pd.DataFrame:
        return self._read(self._REQUIRED["macro"], required=True)

    def gst(self) -> pd.DataFrame:
        return self._read(self._OPTIONAL["gst"], required=False)

    def transactions(self) -> pd.DataFrame:
        return self._read(self._OPTIONAL["transactions"], required=False)


# --------------------------------------------------------------------------- #
# Bureau loader interface + implementations
# --------------------------------------------------------------------------- #


@runtime_checkable
class BureauLoader(Protocol):
    """
    Source-agnostic contract for fetching bureau features by borrower.

    Implementations return **one row per resolvable ``borrower_id``** with
    exactly the columns in :data:`BUREAU_FEATURE_COLUMNS`. Unresolvable ids
    (no parquet row, no identity to pull) are dropped, not error-filled — the
    feature pipeline's imputers handle the resulting absences downstream.
    """

    def load(self, borrower_ids: Sequence[str]) -> pd.DataFrame: ...


def _empty_bureau_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(BUREAU_FEATURE_COLUMNS))


class ParquetBureauLoader:
    """
    Offline bureau loader — reads pre-materialised pulls from parquet.

    Used for training and back-testing, where bureau reports are pulled once and
    cached so every epoch sees identical, reproducible data. The parquet must
    carry at least :data:`BUREAU_FEATURE_COLUMNS`; extra columns are ignored.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, borrower_ids: Sequence[str]) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Bureau parquet {self.path} not found.")
        df = pd.read_parquet(self.path)
        missing = [c for c in BUREAU_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Bureau parquet {self.path} missing columns: {missing}")
        wanted = set(map(str, borrower_ids))
        out = df.loc[df["borrower_id"].astype(str).isin(wanted), list(BUREAU_FEATURE_COLUMNS)]
        return out.reset_index(drop=True)


class DecentroBureauLoader:
    """
    Live bureau loader — pulls reports through the Decentro API connector.

    This is the real-time scoring path and the small-batch path. Each borrower is
    resolved to its PII via the ``identities`` table (kept in memory only; never
    logged), then a credit report is pulled and reduced to a
    :data:`BUREAU_FEATURE_COLUMNS` row. Borrowers absent from ``identities`` are
    skipped with a warning rather than failing the whole batch.

    Parameters
    ----------
    client : DecentroClient
        A constructed connector (typically ``DecentroClient.from_env()``). Its
        own ``use_fixtures`` flag decides live-vs-sandbox; this loader is
        agnostic to that.
    identities : pd.DataFrame
        Indexed by (or carrying a) ``borrower_id`` with at least ``mobile`` and
        ``name``; optional ``date_of_birth`` / ``pincode`` / ``pan`` improve the
        bureau match rate, and per-row ``observation_date`` / ``fixture_profile``
        override the defaults.
    default_fixture_profile : str
        Profile to request when a row has no ``fixture_profile`` (only meaningful
        when the client runs on fixtures).
    """

    def __init__(
        self,
        client: DecentroClient,
        identities: pd.DataFrame,
        default_fixture_profile: str = "good",
    ) -> None:
        self.client = client
        self.default_fixture_profile = default_fixture_profile
        self._identities = self._index_identities(identities)

    @staticmethod
    def _index_identities(identities: pd.DataFrame) -> pd.DataFrame:
        df = identities.copy()
        if df.index.name != "borrower_id":
            if "borrower_id" not in df.columns:
                raise ValueError("identities must have a 'borrower_id' column or index.")
            df = df.set_index("borrower_id")
        missing = [c for c in _REQUIRED_IDENTITY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"identities missing required PII columns: {missing}")
        # One identity per borrower keeps the .loc lookup row-shaped, not frame-shaped.
        return df[~df.index.duplicated(keep="first")]

    def load(self, borrower_ids: Sequence[str]) -> pd.DataFrame:
        rows: list[dict] = []
        for borrower_id in borrower_ids:
            key = str(borrower_id)
            if key not in self._identities.index:
                logger.warning("No identity for borrower_id=%s; skipping bureau pull.", key)
                continue
            ident = cast("pd.Series", self._identities.loc[key])
            features = self.client.fetch_bureau_features(
                borrower_id=key,
                mobile=str(ident["mobile"]),
                name=str(ident["name"]),
                observation_date=_opt(ident, "observation_date"),
                date_of_birth=_opt_str(ident, "date_of_birth"),
                pincode=_opt_str(ident, "pincode"),
                pan=_opt_str(ident, "pan"),
                fixture_profile=_opt_str(ident, "fixture_profile") or self.default_fixture_profile,
            )
            record = asdict(features)
            for audit_field in _BUREAU_AUDIT_FIELDS:
                record.pop(audit_field, None)
            rows.append(record)

        if not rows:
            return _empty_bureau_frame()
        return pd.DataFrame(rows, columns=list(BUREAU_FEATURE_COLUMNS))


def _opt(row: pd.Series, key: str):
    """Return row[key] if present and not-null, else None."""
    if key in row.index:
        value = row[key]
        if value is not None and not pd.isna(value):
            return value
    return None


def _opt_str(row: pd.Series, key: str) -> str | None:
    value = _opt(row, key)
    return None if value is None else str(value)


# --------------------------------------------------------------------------- #
# Factories — wire a source from config
# --------------------------------------------------------------------------- #


def build_tabular_source(config: dict) -> ParquetTabularSource:
    """Construct the non-bureau source from ``config['data']['synthetic_dir']``."""
    synthetic_dir = config.get("data", {}).get("synthetic_dir", "data/synthetic")
    return ParquetTabularSource(synthetic_dir)


def build_bureau_loader(
    config: dict,
    identities: pd.DataFrame | None = None,
) -> BureauLoader:
    """
    Choose a bureau loader from config.

    Resolution order:

    1. If ``bureau_connector.parquet_path`` is configured **and exists**, use the
       cached :class:`ParquetBureauLoader` (preferred for training — reproducible
       and free).
    2. Otherwise fall back to :class:`DecentroBureauLoader` over a client built
       from the environment. This requires ``identities`` (PII to resolve each
       borrower) and the ``ingest`` extra to be installed.
    """
    bureau_cfg = config.get("bureau_connector", {})
    parquet_path = bureau_cfg.get("parquet_path")
    if parquet_path and Path(parquet_path).exists():
        logger.info("Using cached bureau parquet at %s", parquet_path)
        return ParquetBureauLoader(parquet_path)

    if identities is None:
        raise ValueError(
            "No cached bureau parquet found and no identities provided. Either set "
            "bureau_connector.parquet_path to a materialised pull, or pass an "
            "identities frame so the Decentro loader can resolve borrowers."
        )

    # Lazy import: requests / the connector live in the optional 'ingest' extra.
    from src.data_ingestion.bureau_connector import DecentroClient

    client = DecentroClient.from_env()
    return DecentroBureauLoader(
        client=client,
        identities=identities,
        default_fixture_profile=bureau_cfg.get("default_fixture_profile", "good"),
    )
