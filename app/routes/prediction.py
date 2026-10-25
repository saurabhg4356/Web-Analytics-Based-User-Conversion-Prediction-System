"""
app/routes/prediction.py
Conversion prediction routes: form handling, ML inference, and REST API.
"""

from flask import Blueprint, render_template, request, flash, jsonify
from app.forms import SessionPredictionForm
from app.services.prediction_service import PredictionService
from app.extensions import csrf

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """Renders prediction form and processes conversion inference."""
    form = SessionPredictionForm()
    prediction_result = None

    if form.validate_on_submit():
        raw_input = {
            "Administrative": form.administrative.data,
            "Administrative_Duration": form.administrative_duration.data,
            "Informational": form.informational.data,
            "Informational_Duration": form.informational_duration.data,
            "ProductRelated": form.product_related.data,
            "ProductRelated_Duration": form.product_related_duration.data,
            "BounceRates": form.bounce_rates.data,
            "ExitRates": form.exit_rates.data,
            "SpecialDay": form.special_day.data,
            "Month": form.month.data,
            "OperatingSystems": form.operating_systems.data,
            "Browser": form.browser.data,
            "Region": form.region.data,
            "TrafficType": form.traffic_type.data,
            "VisitorType": form.visitor_type.data,
            "Weekend": form.weekend.data
        }

        threshold = form.decision_threshold.data
        session_label = form.session_summary.data

        try:
            prediction_result = PredictionService.predict_session(
                raw_input=raw_input,
                threshold=threshold,
                session_label=session_label
            )
            flash(
                f"Prediction calculated: {prediction_result['predicted_label']} "
                f"({prediction_result['purchase_probability_pct']} probability).",
                "success"
            )
        except Exception as e:
            flash(f"Inference error: {str(e)}", "danger")

    elif request.method == "POST":
        flash("Please correct the validation errors in the form below.", "warning")

    return render_template("predict.html", form=form, prediction_result=prediction_result)


@prediction_bp.route("/api/predict", methods=["POST"])
@csrf.exempt
def api_predict():
    """Programmatic REST API endpoint for real-time web session scoring."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload in request body."}), 400

    threshold = data.get("decision_threshold", 0.40)
    session_label = data.get("session_summary", "API Session")

    try:
        result = PredictionService.predict_session(
            raw_input=data,
            threshold=threshold,
            session_label=session_label
        )
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": f"Validation Error: {str(ve)}"}), 422
    except Exception as e:
        return jsonify({"error": f"Inference Error: {str(e)}"}), 500
