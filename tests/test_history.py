"""
tests/test_history.py
Tests for prediction history records, search, outcome audit, and safe CSV export.
"""

from app.models import PredictionHistory
from app.extensions import db


def test_history_empty_state(client):
    """Verifies history page renders empty state cleanly when no records exist."""
    response = client.get("/history")
    assert response.status_code == 200
    assert b"No prediction records found" in response.data


def test_history_with_records(client, valid_session_data):
    """Verifies history displays created records."""
    # Create record
    client.post("/api/predict", json=valid_session_data)

    response = client.get("/history")
    assert response.status_code == 200
    assert b"#1" in response.data
    assert b"Test High-Intent Session" in response.data


def test_history_filter_by_class(client, valid_session_data):
    """Verifies filtering by predicted class."""
    # Create one record
    client.post("/api/predict", json=valid_session_data)

    res_purchase = client.get("/history?predicted=1")
    assert res_purchase.status_code == 200

    res_non_purchase = client.get("/history?predicted=0")
    assert res_non_purchase.status_code == 200


def test_prediction_detail_route(client, valid_session_data):
    """Verifies detailed inspection view shows all attributes."""
    client.post("/api/predict", json=valid_session_data)

    response = client.get("/history/1")
    assert response.status_code == 200
    assert b"Prediction Inspection: Session #1" in response.data
    assert b"ProductRelated" in response.data


def test_update_actual_outcome(client, valid_session_data):
    """Verifies updating actual ground truth outcome."""
    client.post("/api/predict", json=valid_session_data)

    # Post outcome update
    response = client.post("/history/1/outcome", data={"actual_outcome": "1"}, follow_redirects=True)
    assert response.status_code == 200

    rec = db.session.get(PredictionHistory, 1)
    assert rec.actual_outcome == 1


def test_csv_export_format_and_security(client, valid_session_data):
    """Verifies CSV export contains expected headers and handles formula injection safely."""
    suspicious_data = valid_session_data.copy()
    suspicious_data["session_summary"] = "=cmd|' /C calc'!A0"  # Attempt formula injection
    client.post("/api/predict", json=suspicious_data)

    response = client.get("/history/export")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    content = response.data.decode("utf-8")

    # Verify headers
    assert "Prediction_ID" in content
    assert "Purchase_Probability" in content
    assert "ProductRelated" in content

    # Verify formula injection character was escaped with single quote
    assert "'=cmd" in content
