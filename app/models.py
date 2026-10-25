"""
app/models.py
SQLAlchemy database models for storing prediction history and model metadata.
"""

from datetime import datetime, timezone
import json
from app.extensions import db


class PredictionHistory(db.Model):
    """Stores individual session prediction records and user session attributes."""
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    predicted_class = db.Column(db.Integer, nullable=False, index=True)  # 1 = Purchase, 0 = No Purchase
    purchase_probability = db.Column(db.Float, nullable=False)           # 0.0 to 1.0
    decision_threshold = db.Column(db.Float, nullable=False, default=0.40)
    model_version = db.Column(db.String(64), nullable=False, default="v1.0.0")
    session_features_json = db.Column(db.Text, nullable=False)          # Serialized JSON of all 16 input features
    actual_outcome = db.Column(db.Integer, nullable=True)               # Nullable until verified: 1, 0, or None
    session_summary = db.Column(db.String(255), nullable=True)          # Quick human-readable label
    notes = db.Column(db.String(500), nullable=True)

    @property
    def session_features(self) -> dict:
        """Returns deserialized feature dictionary."""
        try:
            return json.loads(self.session_features_json)
        except Exception:
            return {}

    @session_features.setter
    def session_features(self, features_dict: dict):
        self.session_features_json = json.dumps(features_dict)

    def to_dict(self) -> dict:
        """Serializes record to dictionary for API/templates."""
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "predicted_class": self.predicted_class,
            "predicted_label": "Purchase" if self.predicted_class == 1 else "No Purchase",
            "purchase_probability": round(self.purchase_probability, 4),
            "purchase_probability_pct": f"{self.purchase_probability * 100:.1f}%",
            "decision_threshold": self.decision_threshold,
            "model_version": self.model_version,
            "session_features": self.session_features,
            "actual_outcome": self.actual_outcome,
            "session_summary": self.session_summary,
            "notes": self.notes
        }


class ModelMetadata(db.Model):
    """Tracks deployed classification model versions and their measured evaluation metrics."""
    __tablename__ = "model_metadata"

    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(64), unique=True, nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    trained_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    training_sample_count = db.Column(db.Integer, nullable=False)
    selected_threshold = db.Column(db.Float, nullable=False, default=0.40)
    evaluation_metrics_json = db.Column(db.Text, nullable=False)
    model_artifact_path = db.Column(db.String(255), nullable=False)

    @property
    def evaluation_metrics(self) -> dict:
        try:
            return json.loads(self.evaluation_metrics_json)
        except Exception:
            return {}

    @evaluation_metrics.setter
    def evaluation_metrics(self, metrics_dict: dict):
        self.evaluation_metrics_json = json.dumps(metrics_dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "trained_at": self.trained_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "training_sample_count": self.training_sample_count,
            "selected_threshold": self.selected_threshold,
            "evaluation_metrics": self.evaluation_metrics,
            "model_artifact_path": self.model_artifact_path
        }
