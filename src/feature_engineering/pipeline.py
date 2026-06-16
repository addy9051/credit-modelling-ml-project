"""build_pd_feature_pipeline(model_type) -> sklearn.Pipeline. model_type in {scorecard, tree, neural}. Assembles imputation, IQR outlier clipping, encoding (WOE/ordinal/scaler), and feature selection. All pipelines are joblib-serialisable for MLflow artifacts. Phase 2 §5.2."""

# TODO: implement per credit_risk_modelling_project_plan.md
