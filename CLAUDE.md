# CLAUDE.md — working notes for the credit-risk platform

Guidance for AI coding sessions on this repo. Keep this current as the codebase grows.

## What this is
End-to-end PD / LGD / EAD credit-risk modelling platform on **synthetic Indian
market data** calibrated to real disclosures. Full spec lives in
`credit_risk_modelling_project_plan.md` — treat it as the source of truth for
file responsibilities and acceptance criteria.

## Environment & tooling
- **uv** manages the env (not Poetry). `requires-python = >=3.11,<3.12`.
- Project is **non-packaged** (`[tool.uv] package = false`) — run modules from
  repo root with `uv run python -m src.<...>`.
- Base install is lean; heavy deps are **optional extras** (see `pyproject.toml`).
  Install only what a phase needs, e.g. `uv sync --extra ingest --extra dq`.
- `make` is unavailable on Windows — use **`./tasks.ps1 <target>`** (mirrors the
  `Makefile`). Both call `uv run`.
- Lint/format/type-check: `ruff` (line length 100) + `mypy`. Run `./tasks.ps1 lint`.

## Conventions
- Source under `src/`, imported as `src.<package>.<module>`.
- Config is layered YAML: `base.yaml` overridden by `development.yaml` /
  `production.yaml` (deep-merge). Every entry point takes `--config <path>`.
- Reproducibility: seed everything from `project.random_seed` (42).
- Data outputs are **parquet** (pyarrow). `data/` is git-ignored (PII) except
  `.gitkeep` files.
- MLflow for experiment tracking + model registry; champion/challenger pattern.
- Tests with `pytest`; mark integration tests `@pytest.mark.integration`.

## Regulatory guardrails (do not violate)
- **Forbidden PD features**: `state_code`, `gender`, `religion`
  (`governance.forbidden_features`). `state_code` IS allowed for LGD.
- **Required top features** (sanity check): `cibil_score`, `dpd_90_count_24m`,
  `revolving_utilisation` should rank in top-10 SHAP importance for PD.
- **Quality gates** (prod, `config/production.yaml`): PD min_gini 0.45 / min_ks
  0.35 / HL p>0.05; LGD R²≥0.40; EAD CCF RMSE≤0.15. Trainers must raise
  `ModelQualityError` when a gate fails.
- Use **out-of-time** splits (train/validation/test cutoffs in config), never
  random CV, for PD evaluation. Apply SMOTE to the training fold only — never
  leak into validation/test.
- No real PII in commits; `detect-private-key` pre-commit hook is active.

## Current state
- Repo **scaffolded**: full directory tree + stub modules (docstrings + TODOs).
  No model logic implemented yet.
- Next up: **Phase 1** — implement `src/data_ingestion/synthetic_generator.py`.

## Helpful Claude tooling for this repo
- `pdf` / `xlsx` skills → annual-report & BSR-1 parsing (Phase 1).
- `docx` / `pptx` skills → validation reports & committee decks (Phase 5).
- `code-review` / `security-review` skills → leakage, fairness, PII/DPDP checks.
- Claude-in-Chrome connector → scrape RBI / investor-presentation tables.
- Add connectors when needed: Postgres (MLflow/Feast), GitHub (CI), Slack (alerts).
