"""
tests/test_data_explorer.py
Regression and integration tests for the dataset explorer filtering logic,
empty dataset handling, invalid parameter safety, and CSV downloads.
"""

import io
import pandas as pd


def test_data_explorer_no_filters(client):
    """Verifies data explorer loads all records with default pagination."""
    response = client.get("/data-explorer")
    assert response.status_code == 200
    assert b"Dataset Explorer" in response.data
    assert b"Active filter match: 12,330" in response.data


def test_data_explorer_visitor_filter_only(client):
    """Verifies filtering by visitor type works independently."""
    response = client.get("/data-explorer?visitor_type=Returning_Visitor&revenue=all&month=all")
    assert response.status_code == 200
    # In dataset, Returning_Visitor has 10,551 sessions
    assert b"Active filter match: 10,551" in response.data
    assert b"Returning Visitor" in response.data


def test_data_explorer_revenue_filter_true_only(client):
    """Verifies filtering by revenue=true works when visitor filter is 'all'."""
    response = client.get("/data-explorer?revenue=true&visitor_type=all&month=all")
    assert response.status_code == 200
    # In dataset, Revenue=True has 1,908 sessions
    assert b"Active filter match: 1,908" in response.data
    assert b"True (Purchase)" in response.data


def test_data_explorer_revenue_filter_false_only(client):
    """Verifies filtering by revenue=false works when visitor filter is 'all'."""
    response = client.get("/data-explorer?revenue=false&visitor_type=all&month=all")
    assert response.status_code == 200
    # In dataset, Revenue=False has 10,422 sessions
    assert b"Active filter match: 10,422" in response.data


def test_data_explorer_both_filters_combined(client):
    """Verifies visitor type and revenue filter operate together without interfering."""
    # New_Visitor with Revenue=True
    response = client.get("/data-explorer?visitor_type=New_Visitor&revenue=true&month=all")
    assert response.status_code == 200
    # In dataset, New_Visitor with Revenue=True has 422 sessions
    assert b"Active filter match: 422" in response.data


def test_data_explorer_month_and_visitor_and_revenue(client):
    """Verifies all three dimensions (month, visitor, revenue) slice dataset accurately."""
    response = client.get("/data-explorer?month=Feb&visitor_type=Returning_Visitor&revenue=true")
    assert response.status_code == 200
    # In dataset, Feb + Returning_Visitor + Revenue=True has only 3 sessions
    assert b"Active filter match: 3" in response.data


def test_data_explorer_empty_results_safe(client):
    """Verifies zero-match filter combinations render graceful empty state without crashing."""
    # Month=Feb + Other visitor + Revenue=True has 0 sessions in UCI dataset
    response = client.get("/data-explorer?month=Feb&visitor_type=Other&revenue=true")
    assert response.status_code == 200
    assert b"Active filter match: 0" in response.data
    assert b"No website sessions match the selected filter combination." in response.data


def test_data_explorer_invalid_filter_values_safe(client):
    """Verifies malformed or unrecognized filter query params are safely handled."""
    response = client.get("/data-explorer?month=bogus_month&revenue=not_a_bool&visitor_type=alien_type")
    assert response.status_code == 200
    # Should safely fallback without 500 error and show total dataset
    assert b"Active filter match: 12,330" in response.data


def test_data_explorer_download_csv_with_filters(client):
    """Verifies download endpoint returns valid CSV adhering to active filters."""
    response = client.get("/data-explorer/download?month=Nov&revenue=true&visitor_type=New_Visitor")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    csv_text = response.data.decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_text))
    assert len(df) > 0
    assert (df["Month"].str.lower() == "nov").all()
    assert (df["Revenue"] == True).all()
    assert (df["VisitorType"].str.lower() == "new_visitor").all()
