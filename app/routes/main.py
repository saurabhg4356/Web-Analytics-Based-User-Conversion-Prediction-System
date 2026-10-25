"""
app/routes/main.py
Main entry points: landing page and system health check.
"""

from flask import Blueprint, render_template, jsonify, current_app
from app.extensions import db
from app.services.model_service import ModelService
from app.models import PredictionHistory

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Renders the executive SaaS-style landing page."""
    # Quick live summary for hero cards
    total_predictions = PredictionHistory.query.count()
    metrics = ModelService.get_metrics()
    champion_model = metrics.get("champion_model", "Random Forest")
    test_eval = metrics.get("test_evaluations", {}).get(champion_model, {})
    roc_auc = test_eval.get("roc_auc", 0.78)
    pr_auc = test_eval.get("pr_auc", 0.36)

    return render_template(
        "index.html",
        total_predictions=total_predictions,
        champion_model=champion_model,
        roc_auc=roc_auc,
        pr_auc=pr_auc
    )


@main_bp.route("/health")
def health_check():
    """Production health-check endpoint."""
    # Check DB
    try:
        db.session.execute(db.text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"database_error: {str(e)}"

    # Check Model
    try:
        pipeline = ModelService.get_pipeline()
        model_status = "loaded" if pipeline is not None else "not_loaded"
    except Exception as e:
        model_status = f"model_error: {str(e)}"

    status_code = 200 if db_status == "connected" and model_status == "loaded" else 503
    return jsonify({
        "status": "healthy" if status_code == 200 else "degraded",
        "service": "Web Analytics-Based User Conversion Prediction System",
        "database": db_status,
        "model": model_status,
        "version": "1.0.0"
    }), status_code
