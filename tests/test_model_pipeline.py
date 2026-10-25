"""
tests/test_model_pipeline.py
Tests for the trained scikit-learn pipeline, leakage prevention, and metrics artifact.
"""

import os
import json
import joblib
import pandas as pd
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService, ORDERED_COLUMNS, NUMERICAL_COLS, CATEGORICAL_COLS


def test_model_artifact_exists_and_loads(app):
    """Verifies champion model artifact exists and can be loaded via joblib."""
    model_path = app.config.get("MODEL_ARTIFACT_PATH")
    assert os.path.exists(model_path), f"Artifact missing at {model_path}"
    pipeline = ModelService.get_pipeline(model_path)
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_leakage_exclusion_pagevalues():
    """Verifies PageValues is strictly excluded from active input features."""
    assert "PageValues" not in ORDERED_COLUMNS
    assert "PageValues" not in NUMERICAL_COLS
    assert "PageValues" not in CATEGORICAL_COLS
    assert "Revenue" not in ORDERED_COLUMNS


def test_pipeline_inference_dimensions(app, valid_session_data):
    """Verifies pipeline infers probabilities with shape (1, 2) summing to 1.0."""
    df_input = PredictionService.format_input_dataframe(valid_session_data)
    pipeline = ModelService.get_pipeline()
    probs = pipeline.predict_proba(df_input)

    assert probs.shape == (1, 2)
    assert abs(probs[0].sum() - 1.0) < 1e-4
    assert 0.0 <= probs[0, 1] <= 1.0


def test_metrics_json_integrity(app):
    """Verifies model_metrics.json has complete evaluation structure."""
    metrics_path = app.config.get("METRICS_PATH")
    assert os.path.exists(metrics_path), f"Metrics JSON missing at {metrics_path}"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "champion_model" in metrics
    assert "cross_validation_comparison" in metrics
    assert "test_evaluations" in metrics
    assert "threshold_analysis" in metrics
    assert len(metrics["cross_validation_comparison"]) >= 3
    assert len(metrics["threshold_analysis"]) >= 10
