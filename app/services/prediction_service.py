"""
app/services/prediction_service.py
Core inference engine: validates input, applies fitted pipeline, computes probabilities,
generates feature-level explanations, and records predictions in SQLite.
"""

import math
import numpy as np
import pandas as pd
from app.extensions import db
from app.models import PredictionHistory
from app.services.model_service import ModelService

NUMERICAL_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay"
]

CATEGORICAL_COLS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]

ORDERED_COLUMNS = NUMERICAL_COLS + CATEGORICAL_COLS


class PredictionService:
    @staticmethod
    def format_input_dataframe(raw_input: dict) -> pd.DataFrame:
        """Converts raw input dictionary into properly formatted DataFrame matching training schema."""
        row_dict = {}

        # Parse numerical attributes
        for col in NUMERICAL_COLS:
            val = raw_input.get(col, 0.0)
            try:
                num_val = float(val)
                if math.isnan(num_val) or math.isinf(num_val):
                    raise ValueError(f"Feature '{col}' contains non-finite numerical value.")
                row_dict[col] = num_val
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid numerical value for '{col}': {val}") from e

        # Parse categorical attributes (convert to string)
        for col in CATEGORICAL_COLS:
            val = raw_input.get(col, "")
            row_dict[col] = str(val).strip()

        df = pd.DataFrame([row_dict], columns=ORDERED_COLUMNS)
        return df

    @classmethod
    def predict_session(cls, raw_input: dict, threshold: float = 0.40, session_label: str = None) -> dict:
        """Executes model inference, computes probabilities, applies threshold, and logs to database."""
        # Clean threshold
        try:
            threshold = float(threshold)
            threshold = max(0.05, min(0.95, threshold))
        except (ValueError, TypeError):
            threshold = 0.40

        # Load fitted pipeline
        pipeline = ModelService.get_pipeline()
        metrics = ModelService.get_metrics()
        model_version = metrics.get("model_version", "v1.0.0")
        champion_model_name = metrics.get("champion_model", "Random Forest")

        # Format input dataframe
        df_input = cls.format_input_dataframe(raw_input)

        # Generate probability estimate
        prob_array = pipeline.predict_proba(df_input)
        purchase_prob = float(prob_array[0, 1])

        # Apply configurable threshold
        predicted_class = 1 if purchase_prob >= threshold else 0
        predicted_label = "Purchase" if predicted_class == 1 else "No Purchase"

        # Behavioral engagement score (0 - 100 scale)
        product_pages = float(raw_input.get("ProductRelated", 0))
        product_duration = float(raw_input.get("ProductRelated_Duration", 0.0))
        bounce_rate = float(raw_input.get("BounceRates", 0.0))
        exit_rate = float(raw_input.get("ExitRates", 0.0))

        # Engagement heuristic
        page_factor = min(40.0, (product_pages / 25.0) * 40.0)
        time_factor = min(35.0, (product_duration / 600.0) * 35.0)
        retention_factor = max(0.0, (1.0 - (bounce_rate * 0.5 + exit_rate * 0.5)) * 25.0)
        engagement_score = round(min(100.0, page_factor + time_factor + retention_factor), 1)

        # Explainability & Behavioral Feature Impacts
        explanations = cls._generate_feature_explanations(raw_input, purchase_prob)

        # Default label if none provided
        if not session_label:
            session_label = f"{raw_input.get('VisitorType', 'Visitor')} in {raw_input.get('Month', 'Session')}"

        # Persist record in SQLite
        history_record = PredictionHistory(
            predicted_class=predicted_class,
            purchase_probability=round(purchase_prob, 4),
            decision_threshold=round(threshold, 2),
            model_version=f"{model_version} ({champion_model_name})",
            session_summary=session_label[:250],
            notes=f"Engagement Score: {engagement_score}/100"
        )
        history_record.session_features = {k: raw_input[k] for k in ORDERED_COLUMNS if k in raw_input}

        db.session.add(history_record)
        db.session.commit()

        return {
            "prediction_id": history_record.id,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "purchase_probability": round(purchase_prob, 4),
            "purchase_probability_pct": f"{purchase_prob * 100:.1f}%",
            "decision_threshold": round(threshold, 2),
            "threshold_pct": f"{threshold * 100:.1f}%",
            "model_name": champion_model_name,
            "model_version": model_version,
            "engagement_score": engagement_score,
            "session_summary": session_label,
            "feature_explanations": explanations,
            "disclaimer": (
                "Notice: This prediction is a probabilistic estimate generated from historical visitor patterns. "
                "It represents statistical likelihood under current session conditions, not a certainty."
            )
        }

    @staticmethod
    def _generate_feature_explanations(session_features: dict, prob: float) -> list:
        """Generates clear, data-grounded explanations of top behavioral drivers for the session."""
        factors = []
        exit_rate = float(session_features.get("ExitRates", 0.05))
        bounce_rate = float(session_features.get("BounceRates", 0.02))
        product_dur = float(session_features.get("ProductRelated_Duration", 300.0))
        product_pages = int(session_features.get("ProductRelated", 5))
        month = str(session_features.get("Month", "May"))
        visitor_type = str(session_features.get("VisitorType", "Returning_Visitor"))
        admin_pages = int(session_features.get("Administrative", 0))

        # Exit rate impact
        if exit_rate <= 0.02:
            factors.append({
                "feature": "Exit Rate",
                "value": f"{exit_rate:.2%}",
                "impact": "Positive",
                "badge": "badge-success",
                "description": "Very low exit rate indicates steady exploration through checkout and cart flow."
            })
        elif exit_rate >= 0.08:
            factors.append({
                "feature": "Exit Rate",
                "value": f"{exit_rate:.2%}",
                "impact": "Negative",
                "badge": "badge-danger",
                "description": "Elevated exit rate suggests friction or hesitation before completing a purchase."
            })

        # Product duration impact
        if product_dur >= 600.0:
            factors.append({
                "feature": "Product Browsing Time",
                "value": f"{product_dur:.0f}s",
                "impact": "Positive",
                "badge": "badge-success",
                "description": "Substantial time spent evaluating merchandise details correlates strongly with purchase intent."
            })
        elif product_dur < 60.0:
            factors.append({
                "feature": "Product Browsing Time",
                "value": f"{product_dur:.0f}s",
                "impact": "Negative",
                "badge": "badge-warning",
                "description": "Brief product duration reflects superficial browsing with limited evaluation."
            })

        # Product pages count
        if product_pages >= 15:
            factors.append({
                "feature": "Product Pages Viewed",
                "value": f"{product_pages} pages",
                "impact": "Positive",
                "badge": "badge-success",
                "description": "Viewing extensive catalog pages demonstrates active shopping intent."
            })

        # Seasonality impact
        if month in ["Nov", "Dec"]:
            factors.append({
                "feature": "Seasonality (Month)",
                "value": month,
                "impact": "Positive",
                "badge": "badge-success",
                "description": "November and December represent peak holiday sales with historically higher conversion rates (~25%)."
            })
        elif month in ["Feb"]:
            factors.append({
                "feature": "Seasonality (Month)",
                "value": month,
                "impact": "Negative",
                "badge": "badge-secondary",
                "description": "February has historically lower e-commerce conversion rates (~1.8%) across the dataset."
            })

        # Visitor type impact
        if visitor_type == "New_Visitor":
            factors.append({
                "feature": "Visitor Type",
                "value": "New Visitor",
                "impact": "Positive",
                "badge": "badge-info",
                "description": "New visitors exhibit high focused campaign arrival rates with an average 24.9% historical conversion."
            })

        # Administrative / Account pages
        if admin_pages >= 2:
            factors.append({
                "feature": "Administrative Actions",
                "value": f"{admin_pages} pages",
                "impact": "Positive",
                "badge": "badge-success",
                "description": "Accessing administrative or cart settings signifies progression toward final purchase."
            })

        if not factors:
            factors.append({
                "feature": "General Engagement",
                "value": "Baseline",
                "impact": "Neutral",
                "badge": "badge-secondary",
                "description": "Session behavioral attributes fall close to standard historical medians."
            })

        return factors
