<#
.SYNOPSIS
    Windows task runner — `make` equivalent for the credit-risk platform.
.EXAMPLE
    ./tasks.ps1 install
    ./tasks.ps1 lint
    ./tasks.ps1 test
    ./tasks.ps1 generate-synthetic
#>
param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

switch ($Task) {
    "install" {
        uv sync --all-extras
        uv run pre-commit install
    }
    "lint" {
        uv run ruff check src/ tests/
        uv run mypy src/ --ignore-missing-imports
    }
    "format" {
        uv run ruff format src/ tests/
        uv run ruff check --fix src/ tests/
    }
    "test" {
        uv run pytest tests/unit/ -v --cov=src --cov-report=term-missing
    }
    "test-integration" {
        uv run pytest tests/integration/ -v
    }
    "generate-synthetic" {
        uv run python -m src.data_ingestion.synthetic_generator --config config/development.yaml
    }
    "train-pd" {
        uv run python -m src.models.pd.trainer --config config/development.yaml
    }
    "train-lgd" {
        uv run python -m src.models.lgd.trainer --config config/development.yaml
    }
    "train-ead" {
        uv run python -m src.models.ead.trainer --config config/development.yaml
    }
    "validate-models" {
        uv run python -m src.validation.report_generator --config config/development.yaml
    }
    "serve" {
        uv run uvicorn src.serving.api:app --reload --port 8000
    }
    "mlflow-ui" {
        uv run mlflow ui --port 5000
    }
    "docker-build" {
        docker build -f docker/Dockerfile.api -t credit-risk-api:latest .
    }
    "docker-up" {
        docker compose -f docker/docker-compose.yml up -d
    }
    default {
        Write-Host "Usage: ./tasks.ps1 <task>" -ForegroundColor Cyan
        Write-Host "Tasks: install, lint, format, test, test-integration," -ForegroundColor Gray
        Write-Host "       generate-synthetic, train-pd, train-lgd, train-ead," -ForegroundColor Gray
        Write-Host "       validate-models, serve, mlflow-ui, docker-build, docker-up" -ForegroundColor Gray
    }
}
