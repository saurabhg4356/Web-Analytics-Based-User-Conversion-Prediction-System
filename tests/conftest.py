"""
tests/conftest.py
Pytest fixtures for Flask application testing, in-memory DB, and test client.
"""

import pytest
from app import create_app
from app.extensions import db


@pytest.fixture
def app():
    """Creates Flask application configured for isolated testing."""
    test_app = create_app("testing")

    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Provides test HTTP client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Provides CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture
def valid_session_data():
    """Provides a realistic validated visitor session dictionary."""
    return {
        "Administrative": 2,
        "Administrative_Duration": 60.0,
        "Informational": 1,
        "Informational_Duration": 25.0,
        "ProductRelated": 20,
        "ProductRelated_Duration": 850.0,
        "BounceRates": 0.005,
        "ExitRates": 0.02,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": "2",
        "Browser": "2",
        "Region": "1",
        "TrafficType": "2",
        "VisitorType": "Returning_Visitor",
        "Weekend": "False",
        "decision_threshold": 0.40,
        "session_summary": "Test High-Intent Session"
    }
