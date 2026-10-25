"""
tests/test_prediction.py
Tests for ML model inference, probability validity, DB persistence,
input validation, edge cases, and REST API error hardening.
"""

from unittest.mock import patch
from app.models import PredictionHistory


def test_predict_page_loads(client):
    """Verifies prediction interface loads with form fields."""
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Session Behavioral Attributes" in response.data
    assert b"Product-Related Pages" in response.data


def test_predict_form_submission_valid(client, valid_session_data):
    """Verifies valid form submission produces calibrated prediction and saves to DB."""
    form_data = {
        "administrative": valid_session_data["Administrative"],
        "administrative_duration": valid_session_data["Administrative_Duration"],
        "informational": valid_session_data["Informational"],
        "informational_duration": valid_session_data["Informational_Duration"],
        "product_related": valid_session_data["ProductRelated"],
        "product_related_duration": valid_session_data["ProductRelated_Duration"],
        "bounce_rates": valid_session_data["BounceRates"],
        "exit_rates": valid_session_data["ExitRates"],
        "special_day": valid_session_data["SpecialDay"],
        "month": valid_session_data["Month"],
        "operating_systems": valid_session_data["OperatingSystems"],
        "browser": valid_session_data["Browser"],
        "region": valid_session_data["Region"],
        "traffic_type": valid_session_data["TrafficType"],
        "visitor_type": valid_session_data["VisitorType"],
        "weekend": valid_session_data["Weekend"],
        "decision_threshold": valid_session_data["decision_threshold"],
        "session_summary": valid_session_data["session_summary"]
    }

    response = client.post("/predict", data=form_data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Estimated Purchase Probability" in response.data
    assert b"Top Session Behavioral Drivers" in response.data

    # Check persistence in database
    records = PredictionHistory.query.all()
    assert len(records) == 1
    rec = records[0]
    assert 0.0 <= rec.purchase_probability <= 1.0
    assert rec.predicted_class in (0, 1)
    assert rec.session_summary == "Test High-Intent Session"
    assert rec.session_features["ProductRelated"] == 20


def test_api_predict_json_valid(client, valid_session_data):
    """Verifies programmatic JSON API prediction returns structured output and 200 status."""
    response = client.post("/api/predict", json=valid_session_data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "purchase_probability" in json_data
    assert "predicted_class" in json_data
    assert "engagement_score" in json_data
    assert 0.0 <= json_data["purchase_probability"] <= 1.0
    assert json_data["predicted_class"] in (0, 1)
    assert json_data["prediction_id"] > 0


def test_api_predict_empty_payload(client):
    """Verifies API rejects empty payload with 400 error."""
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_predict_invalid_content_type(client):
    """Verifies API rejects non-JSON Content-Type with 415 error."""
    response = client.post("/api/predict", data="plain text data", content_type="text/plain")
    assert response.status_code == 415
    assert "Unsupported Media Type" in response.get_json()["error"]


def test_api_predict_missing_feature(client, valid_session_data):
    """Verifies API rejects requests missing required feature columns with 422 error."""
    incomplete_data = valid_session_data.copy()
    del incomplete_data["ProductRelated"]
    response = client.post("/api/predict", json=incomplete_data)
    assert response.status_code == 422
    assert "Missing required" in response.get_json()["error"]


def test_api_predict_nan_rejected(client, valid_session_data):
    """Verifies API rejects NaN values with 422 error."""
    nan_data = valid_session_data.copy()
    nan_data["ExitRates"] = float("nan")
    response = client.post("/api/predict", json=nan_data)
    assert response.status_code == 422
    assert "non-finite" in response.get_json()["error"]


def test_api_predict_negative_counts_rejected(client, valid_session_data):
    """Verifies API rejects negative counts or durations with 422 error."""
    negative_data = valid_session_data.copy()
    negative_data["Administrative"] = -3
    response = client.post("/api/predict", json=negative_data)
    assert response.status_code == 422
    assert "cannot be negative" in response.get_json()["error"]


def test_api_predict_out_of_bounds_rate_rejected(client, valid_session_data):
    """Verifies API rejects rates outside [0.0, 1.0] with 422 error."""
    invalid_rate = valid_session_data.copy()
    invalid_rate["BounceRates"] = 1.25
    response = client.post("/api/predict", json=invalid_rate)
    assert response.status_code == 422
    assert "must be between 0.0 and 1.0" in response.get_json()["error"]


def test_api_predict_unsupported_category_rejected(client, valid_session_data):
    """Verifies API rejects unsupported categorical values with 422 error."""
    invalid_cat = valid_session_data.copy()
    invalid_cat["Month"] = "InvalidMonth"
    response = client.post("/api/predict", json=invalid_cat)
    assert response.status_code == 422
    assert "Unsupported Month" in response.get_json()["error"]


def test_api_predict_invalid_threshold_rejected(client, valid_session_data):
    """Verifies API rejects invalid decision thresholds without silent clamping."""
    invalid_th = valid_session_data.copy()
    invalid_th["decision_threshold"] = 1.85  # Out of bounds (> 0.99)
    response = client.post("/api/predict", json=invalid_th)
    assert response.status_code == 422
    assert "Decision threshold must be between 0.01 and 0.99" in response.get_json()["error"]


def test_api_predict_model_missing_handled_safely(client, valid_session_data):
    """Verifies missing model artifact produces safe 503 response without stack trace."""
    with patch("app.services.model_service.ModelService.get_pipeline", side_effect=FileNotFoundError("Artifact missing")):
        response = client.post("/api/predict", json=valid_session_data)
        assert response.status_code == 503
        assert "Model Service Unavailable" in response.get_json()["error"]
