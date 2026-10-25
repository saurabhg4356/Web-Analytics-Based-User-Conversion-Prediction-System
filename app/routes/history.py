"""
app/routes/history.py
Prediction history records: search, pagination, detailed session review,
actual outcome verification, and safe CSV export.
"""

import csv
import io
from flask import (
    Blueprint, render_template, request, flash,
    redirect, url_for, Response, abort
)
from app.extensions import db
from app.models import PredictionHistory

history_bp = Blueprint("history", __name__)


def sanitize_csv_cell(value) -> str:
    """Escapes potential spreadsheet formula injection characters (=, +, -, @)."""
    s = str(value) if value is not None else ""
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{s}"
    return s


@history_bp.route("/history")
def history():
    """Lists saved session predictions with pagination, search, and filtering."""
    page = request.args.get("page", 1, type=int)
    predicted_filter = request.args.get("predicted", "all")
    search_query = request.args.get("q", "").strip()

    query = PredictionHistory.query

    if predicted_filter in ("1", "0"):
        query = query.filter_by(predicted_class=int(predicted_filter))

    if search_query:
        if search_query.isdigit():
            query = query.filter(
                (PredictionHistory.id == int(search_query)) |
                (PredictionHistory.session_summary.ilike(f"%{search_query}%"))
            )
        else:
            query = query.filter(PredictionHistory.session_summary.ilike(f"%{search_query}%"))

    # Order by newest first
    query = query.order_by(PredictionHistory.created_at.desc())
    pagination = query.paginate(page=page, per_page=15, error_out=False)

    return render_template(
        "history.html",
        pagination=pagination,
        predictions=pagination.items,
        predicted_filter=predicted_filter,
        search_query=search_query
    )


@history_bp.route("/history/<int:prediction_id>")
def prediction_detail(prediction_id: int):
    """Detailed view for an individual session prediction."""
    record = db.get_or_404(PredictionHistory, prediction_id)
    return render_template("prediction_detail.html", record=record)


@history_bp.route("/history/<int:prediction_id>/outcome", methods=["POST"])
def update_outcome(prediction_id: int):
    """Updates whether the session actually resulted in a purchase once confirmed."""
    record = db.get_or_404(PredictionHistory, prediction_id)
    outcome_str = request.form.get("actual_outcome", "")

    if outcome_str == "1":
        record.actual_outcome = 1
        flash(f"Actual outcome updated: Session #{prediction_id} resulted in a PURCHASE.", "success")
    elif outcome_str == "0":
        record.actual_outcome = 0
        flash(f"Actual outcome updated: Session #{prediction_id} did NOT purchase.", "info")
    else:
        record.actual_outcome = None
        flash(f"Actual outcome for Session #{prediction_id} reset to UNKNOWN.", "secondary")

    db.session.commit()
    return redirect(request.referrer or url_for("history.history"))


@history_bp.route("/history/export")
def export_csv():
    """Generates RFC 4180 compliant and injection-safe CSV export of prediction history."""
    records = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Header columns
    headers = [
        "Prediction_ID",
        "Timestamp_UTC",
        "Predicted_Class",
        "Predicted_Label",
        "Purchase_Probability",
        "Decision_Threshold",
        "Model_Version",
        "Session_Summary",
        "Actual_Outcome",
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]
    writer.writerow(headers)

    for rec in records:
        feats = rec.session_features
        actual_str = "Unknown" if rec.actual_outcome is None else ("Purchase" if rec.actual_outcome == 1 else "No Purchase")
        row = [
            sanitize_csv_cell(rec.id),
            sanitize_csv_cell(rec.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            sanitize_csv_cell(rec.predicted_class),
            sanitize_csv_cell("Purchase" if rec.predicted_class == 1 else "No Purchase"),
            sanitize_csv_cell(round(rec.purchase_probability, 4)),
            sanitize_csv_cell(round(rec.decision_threshold, 2)),
            sanitize_csv_cell(rec.model_version),
            sanitize_csv_cell(rec.session_summary or ""),
            sanitize_csv_cell(actual_str),
            sanitize_csv_cell(feats.get("Administrative", "")),
            sanitize_csv_cell(feats.get("Administrative_Duration", "")),
            sanitize_csv_cell(feats.get("Informational", "")),
            sanitize_csv_cell(feats.get("Informational_Duration", "")),
            sanitize_csv_cell(feats.get("ProductRelated", "")),
            sanitize_csv_cell(feats.get("ProductRelated_Duration", "")),
            sanitize_csv_cell(feats.get("BounceRates", "")),
            sanitize_csv_cell(feats.get("ExitRates", "")),
            sanitize_csv_cell(feats.get("SpecialDay", "")),
            sanitize_csv_cell(feats.get("Month", "")),
            sanitize_csv_cell(feats.get("OperatingSystems", "")),
            sanitize_csv_cell(feats.get("Browser", "")),
            sanitize_csv_cell(feats.get("Region", "")),
            sanitize_csv_cell(feats.get("TrafficType", "")),
            sanitize_csv_cell(feats.get("VisitorType", "")),
            sanitize_csv_cell(feats.get("Weekend", ""))
        ]
        writer.writerow(row)

    output.seek(0)
    filename = "web_analytics_predictions_export.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@history_bp.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears prediction history with flash message."""
    count = PredictionHistory.query.delete()
    db.session.commit()
    flash(f"Cleared {count} historical prediction records.", "info")
    return redirect(url_for("history.history"))
