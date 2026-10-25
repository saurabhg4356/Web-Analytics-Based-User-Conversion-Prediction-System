"""
app/routes/evaluation.py
Model evaluation dashboard displaying measured validation & test metrics,
curves, confusion matrix, feature importance, and threshold trade-offs.
"""

from flask import Blueprint, render_template
from app.services.model_service import ModelService

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/evaluation")
def evaluation():
    """Renders the comprehensive model evaluation and diagnostics page."""
    metrics = ModelService.get_metrics()
    is_missing = "error" in metrics

    return render_template(
        "evaluation.html",
        metrics=metrics,
        is_missing=is_missing
    )
