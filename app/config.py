"""
app/config.py
Configuration classes for Development, Production, and Testing environments.
"""

import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "prod-web-analytics-secret-key-default-2026")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True

    # Paths
    BASE_DIR = BASE_DIR
    DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "online_shoppers_intention.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_ARTIFACT_PATH = os.path.join(MODEL_DIR, "best_conversion_model.joblib")
    METRICS_PATH = os.path.join(BASE_DIR, "reports", "model_metrics.json")
    EDA_PATH = os.path.join(BASE_DIR, "reports", "eda_summary.json")

    # Threshold default
    DEFAULT_DECISION_THRESHOLD = float(os.environ.get("DEFAULT_DECISION_THRESHOLD", 0.40))


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    ENV = "development"
    instance_dir = os.path.join(BASE_DIR, "instance")
    os.makedirs(instance_dir, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(instance_dir, 'web_analytics.db')}"
    )


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    ENV = "production"
    # In production, require secure secret key
    SECRET_KEY = os.environ.get("SECRET_KEY")
    instance_dir = os.path.join(BASE_DIR, "instance")
    os.makedirs(instance_dir, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(instance_dir, 'web_analytics.db')}"
    )


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
