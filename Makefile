# Common dev commands (uv-based). On Windows without `make`, use ./tasks.ps1 <target>.
.PHONY: install lint format test test-integration generate-synthetic \
        train-pd train-lgd train-ead validate-models serve mlflow-ui \
        docker-build docker-up clean

install:                ## Create venv and install all dependencies + dev tools
	uv sync --all-extras
	uv run pre-commit install

lint:                   ## Ruff lint + mypy type-check
	uv run ruff check src/ tests/
	uv run mypy src/ --ignore-missing-imports

format:                 ## Auto-format and fix with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:                   ## Unit tests with coverage
	uv run pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration:       ## Integration tests
	uv run pytest tests/integration/ -v

generate-synthetic:     ## Generate the synthetic loan portfolio
	uv run python -m src.data_ingestion.synthetic_generator --config config/development.yaml

train-pd:               ## Train the PD model ladder
	uv run python -m src.models.pd.trainer --config config/development.yaml

train-lgd:              ## Train the two-stage LGD model
	uv run python -m src.models.lgd.trainer --config config/development.yaml

train-ead:              ## Train the EAD / CCF model
	uv run python -m src.models.ead.trainer --config config/development.yaml

validate-models:        ## Generate the model validation report
	uv run python -m src.validation.report_generator --config config/development.yaml

serve:                  ## Run the FastAPI serving app
	uv run uvicorn src.serving.api:app --reload --port 8000

mlflow-ui:              ## Launch the MLflow tracking UI
	uv run mlflow ui --port 5000

docker-build:           ## Build the API serving image
	docker build -f docker/Dockerfile.api -t credit-risk-api:latest .

docker-up:              ## Start the local dev stack
	docker compose -f docker/docker-compose.yml up -d

clean:                  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
