# Credit Risk Modelling Platform

Production-grade **PD / LGD / EAD** credit-risk platform built on real Indian
financial-market structures (CRILC, BSR-1, ICICI/Bajaj Finance/HDFC disclosures)
rather than public benchmark datasets. Targets Basel III IRB, IFRS 9, RBI MRM, and
DPDP Act 2023 compliance.

> **Expected Loss = PD × LGD × EAD** — the portfolio roll-up that drives
> provisioning and regulatory capital.

| Model | Output | Regulatory use |
|-------|--------|----------------|
| **PD** | Default probability 0–1 per borrower | Basel III IRB, IFRS 9 staging |
| **LGD** | Loss fraction 0–1 | Capital, ECL provisioning |
| **EAD** | ₹ exposure at default | RWA, CCF for revolving |

## Quick start

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and
dependency management.

```powershell
# 1. Install the lean base environment + dev tools
uv sync

# 2. (Optional) add the extras a phase needs
uv sync --extra ingest --extra dq          # Phase 1 data ingestion
uv sync --all-extras                        # everything (heavy: torch, airflow, dbt)

# 3. Generate synthetic data to work with
./tasks.ps1 generate-synthetic              # Windows
make generate-synthetic                     # Linux/macOS

# 4. Lint
./tasks.ps1 lint
```

`make` is not installed on Windows by default — use **`./tasks.ps1 <target>`**,
which mirrors every `Makefile` target.

## Dependency extras

The base install is intentionally minimal so `uv sync` is fast and reliable on
Windows. Heavier dependencies are grouped:

| Extra | Pulls in | Needed for |
|-------|----------|-----------|
| `ingest` | pdfplumber, camelot, bs4, openpyxl | Phase 1 parsers/connectors |
| `dq` | great-expectations | Phase 1 data-quality suites |
| `featurestore` | feast, redis | Phase 2/6 feature store |
| `deep` | torch, pytorch-tabnet | Phase 3 TabNet model |
| `explain` | lime | Phase 6 explanations |
| `monitoring` | evidently | Phase 7 drift dashboards |
| `dbt` / `warehouse` | dbt-core, adapters | Warehouse transforms |
| `airflow` | apache-airflow | Orchestration |
| `notebooks` | jupyterlab, papermill | EDA notebooks |

## Repository layout

```
config/        base + dev + prod YAML (deep-merged)
src/
  data_ingestion/    synthetic generator, CRILC/BSR-1/AR parsers, connectors
  data_quality/      Great Expectations suites + Pandera schemas
  feature_engineering/ WOE encoder, feature builders, Feast wrapper, pipeline
  models/{pd,lgd,ead}/ model ladders + trainers
  validation/        discrimination, calibration, stability, stress testing
  explainability/    SHAP, reason codes, model cards
  serving/           FastAPI app + schemas
  monitoring/        drift, performance, retraining triggers
pipelines/{dbt,airflow}/  warehouse transforms + orchestration DAGs
notebooks/     EDA and analysis
tests/{unit,integration}/
docker/ k8s/  containerisation and deployment
```

## Build sequence

Work the phases in order — each phase's output feeds the next (see
`credit_risk_modelling_project_plan.md`).

1. **Phase 1** — synthetic data generator → parsers / connectors
2. **Phase 2** — WOE encoder → feature pipeline
3. **Phase 3** — PD model ladder (scorecard → RF → XGBoost → LightGBM → TabNet)
4. **Phase 4** — two-stage LGD + CCF/EAD
5. **Phase 5** — validation metrics + stress testing
6. **Phase 6** — FastAPI serving + MLOps
7. **Phase 7** — monitoring + retraining

## Data & compliance

All data is **synthetic by default**, calibrated to published portfolio
statistics in `config/base.yaml`. No real PII is committed (`data/` is
git-ignored). `state_code`, `gender`, and `religion` are forbidden PD drivers
(see `governance.forbidden_features`).
