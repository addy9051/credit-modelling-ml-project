"""FastAPI serving application: POST /v1/predict/pd, POST /v1/predict/batch,
GET /v1/health, GET /v1/metrics. Loads the champion model from the MLflow
registry at startup. Phase 6 §9.1."""

# TODO: implement per credit_risk_modelling_project_plan.md
# app = FastAPI(title="credit-risk-api")  # populated during Phase 6
app = None
