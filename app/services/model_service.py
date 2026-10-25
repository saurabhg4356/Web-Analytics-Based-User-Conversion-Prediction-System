"""
app/services/model_service.py
Service for loading, caching, and serving model artifacts and evaluation metrics.
"""

import os
import json
import joblib
from flask import current_app

_CACHED_PIPELINE = None
_CACHED_METRICS = None
_CACHED_EDA = None


class ModelService:
    @staticmethod
    def get_pipeline(model_path: str = None):
        """Loads and caches the trained scikit-learn pipeline."""
        global _CACHED_PIPELINE
        if _CACHED_PIPELINE is not None:
            return _CACHED_PIPELINE

        if model_path is None:
            model_path = current_app.config.get("MODEL_ARTIFACT_PATH")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run 'python scripts/train_model.py' to generate and train the model."
            )

        _CACHED_PIPELINE = joblib.load(model_path)
        return _CACHED_PIPELINE

    @staticmethod
    def get_metrics(metrics_path: str = None) -> dict:
        """Loads the comprehensive model evaluation metrics JSON."""
        global _CACHED_METRICS
        if _CACHED_METRICS is not None:
            return _CACHED_METRICS

        if metrics_path is None:
            metrics_path = current_app.config.get("METRICS_PATH")

        if not os.path.exists(metrics_path):
            return {
                "error": "Metrics artifact not found. Please run 'python scripts/train_model.py'."
            }

        with open(metrics_path, "r") as f:
            _CACHED_METRICS = json.load(f)
        return _CACHED_METRICS

    @staticmethod
    def get_eda(eda_path: str = None) -> dict:
        """Loads the Exploratory Data Analysis summary JSON."""
        global _CACHED_EDA
        if _CACHED_EDA is not None:
            return _CACHED_EDA

        if eda_path is None:
            eda_path = current_app.config.get("EDA_PATH")

        if not os.path.exists(eda_path):
            return {
                "error": "EDA summary not found. Please run 'python scripts/analyze_data.py'."
            }

        with open(eda_path, "r") as f:
            _CACHED_EDA = json.load(f)
        return _CACHED_EDA

    @classmethod
    def reset_cache(cls):
        """Clears memory caches upon model retraining."""
        global _CACHED_PIPELINE, _CACHED_METRICS, _CACHED_EDA
        _CACHED_PIPELINE = None
        _CACHED_METRICS = None
        _CACHED_EDA = None
