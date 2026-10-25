"""
tests/test_validation.py
Tests for server-side form validation rules and range constraints.
"""

from werkzeug.datastructures import MultiDict
from app.forms import SessionPredictionForm


def test_valid_form_validation(app, valid_session_data):
    """Verifies valid session attributes pass form validation."""
    with app.test_request_context():
        form_dict = MultiDict({
            "administrative": str(valid_session_data["Administrative"]),
            "administrative_duration": str(valid_session_data["Administrative_Duration"]),
            "informational": str(valid_session_data["Informational"]),
            "informational_duration": str(valid_session_data["Informational_Duration"]),
            "product_related": str(valid_session_data["ProductRelated"]),
            "product_related_duration": str(valid_session_data["ProductRelated_Duration"]),
            "bounce_rates": str(valid_session_data["BounceRates"]),
            "exit_rates": str(valid_session_data["ExitRates"]),
            "special_day": str(valid_session_data["SpecialDay"]),
            "month": str(valid_session_data["Month"]),
            "operating_systems": str(valid_session_data["OperatingSystems"]),
            "browser": str(valid_session_data["Browser"]),
            "region": str(valid_session_data["Region"]),
            "traffic_type": str(valid_session_data["TrafficType"]),
            "visitor_type": str(valid_session_data["VisitorType"]),
            "weekend": str(valid_session_data["Weekend"]),
            "decision_threshold": str(valid_session_data["decision_threshold"])
        })
        form = SessionPredictionForm(formdata=form_dict)
        assert form.validate() is True, f"Form errors: {form.errors}"


def test_invalid_negative_duration(app):
    """Verifies negative durations are rejected by validation."""
    with app.test_request_context():
        form_dict = MultiDict({
            "product_related_duration": "-100.0",
            "bounce_rates": "0.01",
            "exit_rates": "0.02"
        })
        form = SessionPredictionForm(formdata=form_dict)
        assert form.validate() is False
        assert "product_related_duration" in form.errors


def test_invalid_rate_bounds(app):
    """Verifies bounce rates and exit rates > 1.0 are rejected."""
    with app.test_request_context():
        form_dict = MultiDict({
            "bounce_rates": "1.85",
            "exit_rates": "2.50"
        })
        form = SessionPredictionForm(formdata=form_dict)
        assert form.validate() is False
        assert "bounce_rates" in form.errors
        assert "exit_rates" in form.errors


def test_invalid_threshold_bounds(app):
    """Verifies threshold outside [0.05, 0.95] is rejected."""
    with app.test_request_context():
        form_dict = MultiDict({
            "decision_threshold": "1.50"
        })
        form = SessionPredictionForm(formdata=form_dict)
        assert form.validate() is False
        assert "decision_threshold" in form.errors
