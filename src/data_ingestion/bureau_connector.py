"""
Decentro multi-bureau credit API connector.

Two Decentro endpoints are used:

  1. Credit Score  POST /v2/bytes/credit-score
     Lightweight — score value only. Use for quick pre-screening or when a
     full trade-line pull would be wasteful.

  2. Credit Report Summary
     POST /v2/financial_services/credit_bureau/credit_report/summary
     Full report — score + trade lines + month-by-month DPD history + enquiries.
     This path generates all bureau features consumed by
     src/feature_engineering/bureau_features.py.

Authentication (headers on every request):
  client_id      DECENTRO_CLIENT_ID env var
  client_secret  DECENTRO_CLIENT_SECRET env var

Environments:
  Sandbox    https://in.staging.decentro.tech  (DECENTRO_ENV=sandbox, default)
  Production https://in.decentro.tech          (DECENTRO_ENV=production)

Consent:
  Every call passes consent=True with a >20-char purpose string. The
  reference_id UUID is stored for the DPDP Act 2023 audit trail.

Local development without credentials:
  Set DECENTRO_USE_FIXTURES=true to return pre-built fixture responses that
  mirror the real Decentro schema. Three fixture profiles are available:
    "good"      CIBIL 760, clean history
    "stressed"  CIBIL 510, 3× DPD-30, 1× DPD-90 in last 24 months
    "ntc"       CIBIL -1 (new-to-credit, no bureau history)

Output:
  BureauFeatures — dataclass whose fields match the column contract expected
  by src/feature_engineering/bureau_features.py.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_BASE_URLS = {
    "sandbox": "https://in.staging.decentro.tech",
    "production": "https://in.decentro.tech",
}

_EP_CREDIT_SCORE = "/v2/bytes/credit-score"
_EP_CREDIT_REPORT = "/v2/financial_services/credit_bureau/credit_report/summary"

_CONSENT_PURPOSE = "Credit risk assessment for loan origination and portfolio monitoring"

# CIBIL payment-history codes that represent delinquency, grouped by DPD bucket.
# Codes are as returned by Decentro's accountDetails[].paymentHistory array.
# "most recent month first" ordering; each entry = one calendar month.
_DPD_30_CODES = frozenset({"030", "060", "090", "120", "150", "180", "SUB", "DBT", "LSS", "SMA"})
_DPD_60_CODES = frozenset({"060", "090", "120", "150", "180", "SUB", "DBT", "LSS"})
_DPD_90_CODES = frozenset({"090", "120", "150", "180", "SUB", "DBT", "LSS"})

# Decentro accountType codes that represent revolving / limit-based credit.
# Used to compute revolving_utilisation = Σoutstanding / Σlimit.
_REVOLVING_ACCOUNT_TYPES = frozenset({"CC", "OD", "CV", "AL", "RV"})
# CC=credit card, OD=overdraft, CV=consumer credit, AL=auto loan (revolving),
# RV=revolving credit. Term loans (HL, PL, TL, etc.) are excluded.


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class BureauConnectorError(Exception):
    """Base exception for all bureau connector errors."""


class BureauNotFoundError(BureauConnectorError):
    """Raised when the borrower has no bureau record (new-to-credit)."""


class BureauConsentError(BureauConnectorError):
    """Raised when consent is missing or rejected by Decentro."""


class BureauCreditsExhaustedError(BureauConnectorError):
    """Raised when the Decentro account has no remaining API credits."""


class BureauRateLimitError(BureauConnectorError):
    """Raised on HTTP 429 after all retries are exhausted."""


# --------------------------------------------------------------------------- #
# Output schema — consumed by bureau_features.py
# --------------------------------------------------------------------------- #


@dataclass
class BureauFeatures:
    """
    Normalised bureau features extracted from a Decentro credit report.

    All field names match the column contract in src/feature_engineering/bureau_features.py
    and the synthetic schema previously generated for bureau_data.parquet.

    None values indicate missing data (e.g. no bureau history, no revolving trades).
    The feature pipeline in src/feature_engineering/pipeline.py imputes these.
    """

    borrower_id: str
    observation_date: date

    # Score
    cibil_score: int | None  # 300–900; -1 from Decentro → mapped to None (NTC)
    score_type: str | None  # e.g. "ERS4.0"

    # Delinquency counts (last 24 months, across all trade lines)
    dpd_30_count_24m: int
    dpd_60_count_24m: int
    dpd_90_count_24m: int
    months_since_last_delinquency: int | None  # None = never delinquent

    # Utilisation
    revolving_utilisation: float | None  # Σ outstanding / Σ limit (revolving only)

    # Trade line breadth
    open_trade_count: int
    oldest_trade_months: int | None  # months since oldest open_date; None if no trades

    # Enquiry velocity
    enquiry_count_6m: int  # hard enquiries in last 6 months

    # Audit fields
    decentro_txn_id: str = field(repr=False)
    reference_id: str = field(repr=False)
    raw_response_code: str = field(repr=False, default="")


# --------------------------------------------------------------------------- #
# Decentro API client
# --------------------------------------------------------------------------- #


class DecentroClient:
    """
    HTTP client for the Decentro credit bureau API.

    Handles authentication, retry with exponential back-off, response parsing,
    and feature extraction.  All PII (mobile, name) is kept in memory only and
    never written to logs.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        env: str = "sandbox",
        timeout: int = 30,
        max_retries: int = 3,
        use_fixtures: bool = False,
    ) -> None:
        if env not in _BASE_URLS:
            raise ValueError(f"env must be one of {list(_BASE_URLS)}; got {env!r}")
        self._base_url = _BASE_URLS[env]
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._use_fixtures = use_fixtures
        self._session = self._build_session(max_retries)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> DecentroClient:
        """
        Construct a client from environment variables.

          DECENTRO_CLIENT_ID     required
          DECENTRO_CLIENT_SECRET required
          DECENTRO_ENV           sandbox (default) | production
          DECENTRO_TIMEOUT       seconds (default 30)
          DECENTRO_USE_FIXTURES  true | false (default false)
        """
        client_id = os.environ.get("DECENTRO_CLIENT_ID", "")
        client_secret = os.environ.get("DECENTRO_CLIENT_SECRET", "")
        env = os.environ.get("DECENTRO_ENV", "sandbox")
        timeout = int(os.environ.get("DECENTRO_TIMEOUT", "30"))
        use_fixtures_raw = os.environ.get("DECENTRO_USE_FIXTURES", "false").lower()
        use_fixtures = use_fixtures_raw in ("true", "1", "yes")

        if not use_fixtures and (not client_id or not client_secret):
            raise BureauConnectorError(
                "DECENTRO_CLIENT_ID and DECENTRO_CLIENT_SECRET must be set, "
                "or set DECENTRO_USE_FIXTURES=true for local development."
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            env=env,
            timeout=timeout,
            use_fixtures=use_fixtures,
        )

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        """Attach a retry adapter that backs off on 429 and 5xx responses."""
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,  # waits 1s, 2s, 4s between attempts
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"POST"},
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # Low-level request
    # ------------------------------------------------------------------ #

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        logger.debug("POST %s reference_id=%s", url, payload.get("reference_id", ""))
        response = self._session.post(
            url,
            json=payload,
            headers=self._auth_headers,
            timeout=self._timeout,
        )
        if response.status_code == 429:
            raise BureauRateLimitError("Decentro rate limit exceeded after retries.")
        if response.status_code >= 500:
            raise BureauConnectorError(
                f"Decentro server error {response.status_code}: {response.text[:200]}"
            )
        body: dict[str, Any] = response.json()
        self._check_api_error(body)
        return body

    @staticmethod
    def _check_api_error(body: dict[str, Any]) -> None:
        response_key = body.get("responseKey", "")
        if response_key == "error_credits_score_not_found":
            raise BureauNotFoundError("No bureau record found for the provided identity.")
        if response_key == "error_empty_consent":
            raise BureauConsentError("Decentro rejected the request: consent missing.")
        if response_key == "error_module_credits_exhausted":
            raise BureauCreditsExhaustedError("Decentro account has no remaining credits.")
        if body.get("status") not in ("SUCCESS", "success") and response_key.startswith("error_"):
            raise BureauConnectorError(
                f"Decentro error {body.get('responseCode', '?')}: {body.get('message', '')}"
            )

    # ------------------------------------------------------------------ #
    # Public API methods
    # ------------------------------------------------------------------ #

    def fetch_credit_score(self, mobile: str, name: str) -> dict[str, Any]:
        """
        Lightweight score-only call.  Returns the raw Decentro response dict.
        Does NOT return a BureauFeatures object — use fetch_bureau_features()
        for that.  Useful for pre-screening before committing to a full report.

        Args:
            mobile: 10-digit Indian mobile number (starts with 6–9).
            name:   Full name as registered with bureau.
        """
        if self._use_fixtures:
            return _FIXTURES["good"]["score"]
        payload = {"mobile": mobile, "name": name}
        return self._post(_EP_CREDIT_SCORE, payload)

    def fetch_credit_report(
        self,
        mobile: str,
        name: str,
        date_of_birth: str | None = None,
        pincode: str | None = None,
        pan: str | None = None,
        fixture_profile: str = "good",
    ) -> dict[str, Any]:
        """
        Full credit report: score + trade lines + DPD history + enquiries.

        Args:
            mobile:        10-digit Indian mobile.
            name:          Full name as registered with bureau.
            date_of_birth: Optional ISO date string "YYYY-MM-DD".
            pincode:       Optional 6-digit postal code (improves match rate).
            pan:           Optional PAN number (document_id; improves match rate).
            fixture_profile: "good" | "stressed" | "ntc" (only used when
                             use_fixtures=True).

        Returns:
            Raw Decentro response dict.  Pass to _extract_features() or
            use the high-level fetch_bureau_features() method.
        """
        if self._use_fixtures:
            return _FIXTURES.get(fixture_profile, _FIXTURES["good"])["report"]

        ref_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "reference_id": ref_id,
            "consent": True,
            "consent_purpose": _CONSENT_PURPOSE,
            "name": name,
            "mobile": mobile,
            "inquiry_purpose": "CREDIT_RISK_ASSESSMENT",
        }
        if date_of_birth:
            payload["date_of_birth"] = date_of_birth
        if pincode:
            payload["address_type"] = "H"
            payload["pincode"] = pincode
        if pan:
            payload["document_type"] = "PAN"
            payload["document_id"] = pan

        return self._post(_EP_CREDIT_REPORT, payload)

    def fetch_bureau_features(
        self,
        borrower_id: str,
        mobile: str,
        name: str,
        observation_date: date | None = None,
        date_of_birth: str | None = None,
        pincode: str | None = None,
        pan: str | None = None,
        fixture_profile: str = "good",
    ) -> BureauFeatures:
        """
        High-level method: pull a full credit report and return a BureauFeatures
        dataclass ready for use in the PD feature pipeline.

        Logs only the borrower_id and transaction ID — no PII.
        """
        obs_date = observation_date or date.today()
        raw = self.fetch_credit_report(
            mobile=mobile,
            name=name,
            date_of_birth=date_of_birth,
            pincode=pincode,
            pan=pan,
            fixture_profile=fixture_profile,
        )
        features = _extract_features(
            raw=raw,
            borrower_id=borrower_id,
            observation_date=obs_date,
        )
        logger.info(
            "bureau features extracted borrower_id=%s txn_id=%s cibil=%s",
            borrower_id,
            features.decentro_txn_id,
            features.cibil_score,
        )
        return features


# --------------------------------------------------------------------------- #
# Feature extraction — maps raw Decentro response → BureauFeatures
# --------------------------------------------------------------------------- #


def _extract_features(
    raw: dict[str, Any],
    borrower_id: str,
    observation_date: date,
) -> BureauFeatures:
    data = raw.get("data", {})
    score_details = data.get("scoreDetails", [])
    account_details = data.get("accountDetails", [])
    enquiry_details = data.get("enquiryDetails", [])

    cibil_score, score_type = _parse_score(score_details)
    dpd30, dpd60, dpd90, months_since_delinq = _dpd_counts(account_details)
    rev_util = _revolving_utilisation(account_details)
    open_trades = _open_trade_count(account_details)
    oldest_months = _oldest_trade_months(account_details, observation_date)
    enq_6m = _enquiry_count_6m(enquiry_details, observation_date)

    return BureauFeatures(
        borrower_id=borrower_id,
        observation_date=observation_date,
        cibil_score=cibil_score,
        score_type=score_type,
        dpd_30_count_24m=dpd30,
        dpd_60_count_24m=dpd60,
        dpd_90_count_24m=dpd90,
        months_since_last_delinquency=months_since_delinq,
        revolving_utilisation=rev_util,
        open_trade_count=open_trades,
        oldest_trade_months=oldest_months,
        enquiry_count_6m=enq_6m,
        decentro_txn_id=raw.get("decentroTxnId", ""),
        reference_id=raw.get("reference_id", ""),
        raw_response_code=raw.get("responseCode", ""),
    )


def _parse_score(score_details: list[dict]) -> tuple[int | None, str | None]:
    """Extract the numeric CIBIL / Equifax score. Returns (None, None) for NTC."""
    if not score_details:
        return None, None
    first = score_details[0]
    raw_val = first.get("value", "-1")
    score_name = first.get("name")
    try:
        val = int(raw_val)
    except (ValueError, TypeError):
        return None, score_name
    return (None if val < 0 else val), score_name


def _dpd_counts(
    account_details: list[dict],
    lookback_months: int = 24,
) -> tuple[int, int, int, int | None]:
    """
    Aggregate DPD event counts across all trade lines for the last lookback_months.

    Decentro returns paymentHistory as a list of monthly payment codes,
    most recent first. Each entry is one calendar month.
    Cross-account aggregation: total count of (account × month) cells with DPD.

    Returns:
        dpd_30_count, dpd_60_count, dpd_90_count, months_since_last_delinquency
    """
    dpd30 = dpd60 = dpd90 = 0
    earliest_delinq_age: int | None = None  # index from most recent = months ago

    for account in account_details:
        history: list[str] = account.get("paymentHistory", [])
        window = history[:lookback_months]
        for month_idx, code in enumerate(window):
            code = code.strip().upper()
            if code in _DPD_30_CODES:
                dpd30 += 1
                if earliest_delinq_age is None or month_idx < earliest_delinq_age:
                    # "earliest" here means most recent (smallest index)
                    earliest_delinq_age = month_idx
            if code in _DPD_60_CODES:
                dpd60 += 1
            if code in _DPD_90_CODES:
                dpd90 += 1

    return dpd30, dpd60, dpd90, earliest_delinq_age


def _revolving_utilisation(account_details: list[dict]) -> float | None:
    """
    Σ currentBalance / Σ sanctionedAmount across revolving trades only.

    Returns None if the borrower has no revolving accounts.
    Caps at 2.0 (over-limit situations are meaningful but capped to prevent
    extreme values from distorting WOE bins).
    """
    total_limit = total_outstanding = 0.0
    for account in account_details:
        if account.get("accountType", "") not in _REVOLVING_ACCOUNT_TYPES:
            continue
        limit = float(account.get("sanctionedAmount") or 0)
        outstanding = float(account.get("currentBalance") or 0)
        total_limit += limit
        total_outstanding += outstanding

    if total_limit <= 0:
        return None
    return min(total_outstanding / total_limit, 2.0)


def _open_trade_count(account_details: list[dict]) -> int:
    """Count accounts where dateClosed is null/empty (still active)."""
    return sum(1 for a in account_details if not a.get("dateClosed"))


def _oldest_trade_months(
    account_details: list[dict],
    observation_date: date,
) -> int | None:
    """Months between the oldest dateOpened and observation_date."""
    oldest: date | None = None
    for account in account_details:
        raw = account.get("dateOpened", "")
        if not raw:
            continue
        try:
            opened = datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if oldest is None or opened < oldest:
            oldest = opened
    if oldest is None:
        return None
    delta = (observation_date - oldest).days
    return max(0, delta // 30)


def _enquiry_count_6m(
    enquiry_details: list[dict],
    observation_date: date,
) -> int:
    """Count hard enquiries in the 6 months prior to observation_date."""
    cutoff = observation_date - timedelta(days=182)
    count = 0
    for enq in enquiry_details:
        raw = enq.get("enquiryDate", "")
        if not raw:
            continue
        try:
            enq_date = datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if cutoff <= enq_date <= observation_date:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Sandbox fixtures — realistic Decentro-shaped responses for local dev
# --------------------------------------------------------------------------- #

_today_str = date.today().isoformat()
_txn_base = "DCTRO_FIXTURE"

_FIXTURES: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------ #
    # "good" — clean salaried borrower, CIBIL 760, no delinquencies
    # ------------------------------------------------------------------ #
    "good": {
        "score": {
            "decentroTxnId": f"{_txn_base}_GOOD_SCR",
            "status": "SUCCESS",
            "responseCode": "S00000",
            "message": "Credit score fetched successfully.",
            "responseKey": "success_credit_score",
            "data": {
                "scoreDetails": [
                    {
                        "type": "ERS",
                        "version": "4.0",
                        "name": "ERS4.0",
                        "value": "760",
                        "scoringElements": [
                            {
                                "type": "RES",
                                "seq": "1",
                                "code": "P011",
                                "description": "Time since most recent account opening is too short",
                            },
                        ],
                    }
                ]
            },
        },
        "report": {
            "decentroTxnId": f"{_txn_base}_GOOD_RPT",
            "status": "SUCCESS",
            "responseCode": "S00000",
            "message": "Credit report fetched successfully.",
            "responseKey": "success_credit_report",
            "reference_id": "fixture-good-0001",
            "data": {
                "scoreDetails": [
                    {
                        "type": "ERS",
                        "version": "4.0",
                        "name": "ERS4.0",
                        "value": "760",
                        "scoringElements": [],
                    }
                ],
                "personalInfo": {
                    "fullName": "TEST GOOD BORROWER",
                    "dob": "1988-05-12",
                    "gender": "M",
                    "totalIncome": "80000",
                    "occupation": "SAL",
                },
                "accountDetails": [
                    {
                        "accountType": "CC",
                        "subscriberName": "HDFC BANK LTD",
                        "dateOpened": "2018-04-01",
                        "dateClosed": None,
                        "sanctionedAmount": 200000,
                        "currentBalance": 42000,
                        "amountOverdue": 0,
                        "ownership": "Individual",
                        # 24 months of STD = perfectly clean history
                        "paymentHistory": ["STD"] * 24,
                        "dpdHistory": [],
                    },
                    {
                        "accountType": "HL",
                        "subscriberName": "SBI HOME FINANCE",
                        "dateOpened": "2016-11-15",
                        "dateClosed": None,
                        "sanctionedAmount": 3500000,
                        "currentBalance": 2100000,
                        "amountOverdue": 0,
                        "ownership": "Individual",
                        "paymentHistory": ["STD"] * 24,
                        "dpdHistory": [],
                    },
                    {
                        "accountType": "PL",
                        "subscriberName": "BAJAJ FINANCE LTD",
                        "dateOpened": "2021-07-01",
                        "dateClosed": "2023-07-01",
                        "sanctionedAmount": 500000,
                        "currentBalance": 0,
                        "amountOverdue": 0,
                        "ownership": "Individual",
                        "paymentHistory": ["STD"] * 24,
                        "dpdHistory": [],
                    },
                ],
                "enquiryDetails": [
                    {
                        "enquiryDate": (date.today() - timedelta(days=90)).isoformat(),
                        "enquiryPurpose": "Home Loan",
                        "enquiryType": "Hard",
                    },
                ],
                "identityInfo": {"panNumber": "XXXXX0000X"},
            },
        },
    },
    # ------------------------------------------------------------------ #
    # "stressed" — self-employed borrower, CIBIL 510, recent DPD history
    # ------------------------------------------------------------------ #
    "stressed": {
        "score": {
            "decentroTxnId": f"{_txn_base}_STRS_SCR",
            "status": "SUCCESS",
            "responseCode": "S00000",
            "message": "Credit score fetched successfully.",
            "responseKey": "success_credit_score",
            "data": {
                "scoreDetails": [
                    {
                        "type": "ERS",
                        "version": "4.0",
                        "name": "ERS4.0",
                        "value": "510",
                        "scoringElements": [],
                    }
                ]
            },
        },
        "report": {
            "decentroTxnId": f"{_txn_base}_STRS_RPT",
            "status": "SUCCESS",
            "responseCode": "S00000",
            "message": "Credit report fetched successfully.",
            "responseKey": "success_credit_report",
            "reference_id": "fixture-stressed-0001",
            "data": {
                "scoreDetails": [
                    {
                        "type": "ERS",
                        "version": "4.0",
                        "name": "ERS4.0",
                        "value": "510",
                        "scoringElements": [],
                    }
                ],
                "personalInfo": {
                    "fullName": "TEST STRESSED BORROWER",
                    "dob": "1982-09-30",
                    "gender": "M",
                    "totalIncome": "45000",
                    "occupation": "SEL",
                },
                "accountDetails": [
                    {
                        "accountType": "CC",
                        "subscriberName": "AXIS BANK LTD",
                        "dateOpened": "2019-02-01",
                        "dateClosed": None,
                        "sanctionedAmount": 80000,
                        "currentBalance": 76000,  # near-limit → high utilisation
                        "amountOverdue": 5000,
                        "ownership": "Individual",
                        # 3× DPD-30, 1× DPD-90 in last 24 months
                        "paymentHistory": [
                            "STD",
                            "030",
                            "STD",
                            "090",
                            "STD",
                            "STD",
                            "030",
                            "STD",
                            "STD",
                            "STD",
                            "030",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                            "STD",
                        ],
                        "dpdHistory": [
                            {"month": 2, "dpdDays": 38},
                            {"month": 4, "dpdDays": 95},
                            {"month": 7, "dpdDays": 31},
                            {"month": 11, "dpdDays": 33},
                        ],
                    },
                    {
                        "accountType": "OD",
                        "subscriberName": "PUNJAB NATIONAL BANK",
                        "dateOpened": "2020-06-01",
                        "dateClosed": None,
                        "sanctionedAmount": 500000,
                        "currentBalance": 430000,
                        "amountOverdue": 0,
                        "ownership": "Individual",
                        "paymentHistory": ["STD"] * 24,
                        "dpdHistory": [],
                    },
                    {
                        "accountType": "TL",
                        "subscriberName": "ICICI BANK LTD",
                        "dateOpened": "2017-03-01",
                        "dateClosed": None,
                        "sanctionedAmount": 1200000,
                        "currentBalance": 650000,
                        "amountOverdue": 0,
                        "ownership": "Individual",
                        "paymentHistory": ["STD"] * 24,
                        "dpdHistory": [],
                    },
                ],
                "enquiryDetails": [
                    {
                        "enquiryDate": (date.today() - timedelta(days=30)).isoformat(),
                        "enquiryPurpose": "Personal Loan",
                        "enquiryType": "Hard",
                    },
                    {
                        "enquiryDate": (date.today() - timedelta(days=75)).isoformat(),
                        "enquiryPurpose": "Business Loan",
                        "enquiryType": "Hard",
                    },
                    {
                        "enquiryDate": (date.today() - timedelta(days=120)).isoformat(),
                        "enquiryPurpose": "Credit Card",
                        "enquiryType": "Hard",
                    },
                ],
            },
        },
    },
    # ------------------------------------------------------------------ #
    # "ntc" — new-to-credit, no bureau history
    # ------------------------------------------------------------------ #
    "ntc": {
        "score": {
            "decentroTxnId": f"{_txn_base}_NTC_SCR",
            "status": "SUCCESS",
            "responseCode": "E00058",
            "message": "Credit score not found.",
            "responseKey": "error_credits_score_not_found",
            "data": {"scoreDetails": [{"value": "-1", "name": "ERS4.0"}]},
        },
        "report": {
            "decentroTxnId": f"{_txn_base}_NTC_RPT",
            "status": "SUCCESS",
            "responseCode": "S00000",
            "message": "Credit report fetched. No bureau history found.",
            "responseKey": "success_credit_report",
            "reference_id": "fixture-ntc-0001",
            "data": {
                "scoreDetails": [
                    {
                        "type": "ERS",
                        "version": "4.0",
                        "name": "ERS4.0",
                        "value": "-1",
                        "scoringElements": [],
                    }
                ],
                "personalInfo": {
                    "fullName": "TEST NTC BORROWER",
                    "dob": "1998-12-01",
                    "gender": "F",
                    "totalIncome": "25000",
                    "occupation": "SAL",
                },
                "accountDetails": [],
                "enquiryDetails": [],
                "identityInfo": {"panNumber": "XXXXX0001X"},
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# CLI — smoke-test the connector with fixtures
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decentro bureau connector — smoke-test with fixtures or sandbox."
    )
    parser.add_argument(
        "--profile",
        choices=["good", "stressed", "ntc"],
        default="good",
        help="Fixture profile to test (default: good).",
    )
    parser.add_argument(
        "--env",
        choices=["sandbox", "production"],
        default="sandbox",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = DecentroClient.from_env()
    print(f"\nUsing fixtures: {client._use_fixtures}  env: {args.env}")
    print(f"Profile: {args.profile}\n")

    features = client.fetch_bureau_features(
        borrower_id="smoke-test-001",
        mobile="9999999999",
        name="Test Borrower",
        fixture_profile=args.profile,
    )
    print("BureauFeatures:")
    for f_name in features.__dataclass_fields__:
        print(f"  {f_name:35s}: {getattr(features, f_name)}")
