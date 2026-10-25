"""
app/routes/dashboard.py
Analytics dashboard and dynamic filtering API.
"""

from flask import Blueprint, render_template, request, jsonify
from app.services.analytics_service import AnalyticsService

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
def dashboard():
    """Renders the executive analytics dashboard."""
    month = request.args.get("month", "all")
    visitor_type = request.args.get("visitor_type", "all")
    traffic_type = request.args.get("traffic_type", "all")
    weekend = request.args.get("weekend", "all")

    data = AnalyticsService.get_dashboard_data(
        month=month,
        visitor_type=visitor_type,
        traffic_type=traffic_type,
        weekend=weekend
    )

    return render_template("dashboard.html", data=data)


@dashboard_bp.route("/api/analytics/summary")
def api_analytics_summary():
    """Asynchronous JSON endpoint for interactive dashboard filtering."""
    month = request.args.get("month", "all")
    visitor_type = request.args.get("visitor_type", "all")
    traffic_type = request.args.get("traffic_type", "all")
    weekend = request.args.get("weekend", "all")

    data = AnalyticsService.get_dashboard_data(
        month=month,
        visitor_type=visitor_type,
        traffic_type=traffic_type,
        weekend=weekend
    )

    return jsonify(data)
