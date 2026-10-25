"""
app/routes/data_explorer.py
Dataset explorer: paginated viewing of raw session records, column metadata,
filtering by segment, and authorized CSV download.
"""

import io
import pandas as pd
from flask import (
    Blueprint, render_template, request, Response
)
from app.services.analytics_service import AnalyticsService

data_explorer_bp = Blueprint("data_explorer", __name__)


@data_explorer_bp.route("/data-explorer")
def data_explorer():
    """Renders the paginated dataset explorer with column summaries and filtering."""
    page = request.args.get("page", 1, type=int)
    per_page = 25
    month_filter = request.args.get("month", "all")
    revenue_filter = request.args.get("revenue", "all")
    visitor_filter = request.args.get("visitor_type", "all")

    df = AnalyticsService.get_dataset()

    # Filter dataframe safely and independently
    filtered_df = df.copy()

    # Month filter
    if month_filter and month_filter.lower() != "all":
        valid_months = [m.lower() for m in ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
        if month_filter.lower() in valid_months:
            filtered_df = filtered_df[filtered_df["Month"].str.lower() == month_filter.lower()]

    # Revenue filter (must be independent of visitor_filter)
    if revenue_filter and revenue_filter.lower() in ("true", "false"):
        is_rev = revenue_filter.lower() == "true"
        filtered_df = filtered_df[filtered_df["Revenue"] == is_rev]

    # Visitor type filter (must be independent of revenue_filter)
    if visitor_filter and visitor_filter.lower() != "all":
        valid_visitors = [v.lower() for v in ["Returning_Visitor", "New_Visitor", "Other"]]
        if visitor_filter.lower() in valid_visitors:
            filtered_df = filtered_df[filtered_df["VisitorType"].str.lower() == visitor_filter.lower()]

    total_filtered_rows = len(filtered_df)
    total_pages = max(1, (total_filtered_rows + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    if total_filtered_rows > 0:
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_filtered_rows)
        page_records = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")
    else:
        page_records = []

    # Column descriptions metadata
    columns_info = [
        {"name": "Administrative", "type": "int", "role": "Feature", "description": "Number of account/admin pages viewed."},
        {"name": "Administrative_Duration", "type": "float (s)", "role": "Feature", "description": "Time spent on admin pages."},
        {"name": "Informational", "type": "int", "role": "Feature", "description": "Number of informational/help pages viewed."},
        {"name": "Informational_Duration", "type": "float (s)", "role": "Feature", "description": "Time spent on info pages."},
        {"name": "ProductRelated", "type": "int", "role": "Feature", "description": "Number of product catalog pages viewed."},
        {"name": "ProductRelated_Duration", "type": "float (s)", "role": "Feature", "description": "Time spent viewing products."},
        {"name": "BounceRates", "type": "float [0-1]", "role": "Feature", "description": "Percentage of single-page visits."},
        {"name": "ExitRates", "type": "float [0-1]", "role": "Feature", "description": "Ratio of session exits from pages viewed."},
        {"name": "PageValues", "type": "float", "role": "Leakage Excluded", "description": "Post-hoc page revenue value (excluded from real-time model)."},
        {"name": "SpecialDay", "type": "float [0-1]", "role": "Feature", "description": "Proximity to holiday/shopping promotion."},
        {"name": "Month", "type": "categorical", "role": "Feature", "description": "Month of session visit."},
        {"name": "OperatingSystems", "type": "categorical ID", "role": "Feature", "description": "Client OS identifier (1-8)."},
        {"name": "Browser", "type": "categorical ID", "role": "Feature", "description": "Client browser identifier (1-13)."},
        {"name": "Region", "type": "categorical ID", "role": "Feature", "description": "Visitor geographic region (1-9)."},
        {"name": "TrafficType", "type": "categorical ID", "role": "Feature", "description": "Traffic channel identifier (1-20)."},
        {"name": "VisitorType", "type": "categorical", "role": "Feature", "description": "New, Returning, or Other visitor."},
        {"name": "Weekend", "type": "bool", "role": "Feature", "description": "Whether visit occurred on Saturday/Sunday."},
        {"name": "Revenue", "type": "bool", "role": "Target", "description": "Ground truth conversion outcome (True = Purchase)."}
    ]

    return render_template(
        "data_explorer.html",
        records=page_records,
        columns_info=columns_info,
        total_rows=len(df),
        filtered_rows=total_filtered_rows,
        current_page=page,
        total_pages=total_pages,
        month_filter=month_filter,
        revenue_filter=revenue_filter,
        visitor_filter=visitor_filter
    )


@data_explorer_bp.route("/data-explorer/download")
def download_sample_csv():
    """Allows downloading the dataset (first 2000 rows or filtered slice)."""
    df = AnalyticsService.get_dataset().copy()
    month_filter = request.args.get("month", "all")
    revenue_filter = request.args.get("revenue", "all")
    visitor_filter = request.args.get("visitor_type", "all")

    if month_filter and month_filter.lower() != "all":
        df = df[df["Month"].str.lower() == month_filter.lower()]
    if revenue_filter and revenue_filter.lower() in ("true", "false"):
        is_rev = revenue_filter.lower() == "true"
        df = df[df["Revenue"] == is_rev]
    if visitor_filter and visitor_filter.lower() != "all":
        df = df[df["VisitorType"].str.lower() == visitor_filter.lower()]

    output = io.StringIO()
    df.head(2000).to_csv(output, index=False)
    output.seek(0)

    filename = f"uci_online_shoppers_sample_{month_filter}_{revenue_filter}_{visitor_filter}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
