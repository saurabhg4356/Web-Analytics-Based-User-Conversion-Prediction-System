"""
tests/test_routes.py
Tests for general application routes, status codes, and error handlers.
"""


def test_index_route(client):
    """Verifies overview landing page returns 200 and executive title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Web Analytics-Based User Conversion Prediction System" in response.data
    assert b"ConvertSense AI" in response.data


def test_health_endpoint(client):
    """Verifies health check endpoint returns 200, healthy status, and connected database."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["database"] == "connected"
    assert json_data["model"] == "loaded"


def test_dashboard_route(client):
    """Verifies analytics dashboard loads successfully with KPIs and charts."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"E-Commerce Analytics Dashboard" in response.data
    assert b"Analyzed Sessions" in response.data


def test_dashboard_api_summary(client):
    """Verifies dashboard asynchronous JSON API returns structured KPIs and charts."""
    response = client.get("/api/analytics/summary?month=Nov&visitor_type=all")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "kpis" in json_data
    assert "charts" in json_data
    assert json_data["kpis"]["total_sessions"] > 0
    assert "monthly_conversion" in json_data["charts"]


def test_evaluation_route(client):
    """Verifies model evaluation route displays cross-validation comparison."""
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert b"Model Diagnostics" in response.data
    assert b"5-Fold Stratified Cross-Validation" in response.data


def test_data_explorer_route(client):
    """Verifies data explorer route loads dataset table and column schema."""
    response = client.get("/data-explorer")
    assert response.status_code == 200
    assert b"Dataset Explorer" in response.data
    assert b"ProductRelated" in response.data


def test_404_error_page(client):
    """Verifies unknown route triggers styled 404 page with 404 status code."""
    response = client.get("/non-existent-page-path-1234")
    assert response.status_code == 404
    assert b"404 Page Not Found" in response.data
