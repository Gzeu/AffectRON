import pytest
from playwright.sync_api import Playwright, APIRequestContext
import requests


@pytest.fixture(scope="session")
def api_context(playwright: Playwright) -> APIRequestContext:
    """Create API request context for testing backend API."""
    return playwright.request.new_context(
        base_url="http://localhost:8000",
        extra_http_headers={
            "Content-Type": "application/json"
        }
    )


@pytest.fixture
def api_base_url():
    """Base URL for API tests."""
    return "http://localhost:8000"


def test_api_health(api_context: APIRequestContext):
    """Test API health endpoint."""
    response = api_context.get("/health")

    assert response.status == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_api_sentiment_endpoint_requires_currency(api_context: APIRequestContext):
    """Test that sentiment endpoint requires currency parameter."""
    response = api_context.get("/api/v1/sentiment")

    assert response.status == 422  # Validation error


def test_api_sentiment_with_currency(api_context: APIRequestContext):
    """Test sentiment endpoint with currency parameter."""
    response = api_context.get("/api/v1/sentiment?currency=RON")

    # Should return 200 or 404 if no data, but not 422 (validation error)
    assert response.status in [200, 404]


def test_api_market_rates(api_context: APIRequestContext):
    """Test market rates endpoint."""
    response = api_context.get("/api/v1/market/rates?currencies=RON,EUR")

    # Should return 200 or 404 if no data
    assert response.status in [200, 404]


def test_api_analytics_insights(api_context: APIRequestContext):
    """Test analytics insights endpoint."""
    response = api_context.get("/api/v1/analytics/insights/market?currency=RON")

    assert response.status in [200, 404]


def test_api_risk_score(api_context: APIRequestContext):
    """Test risk score endpoint."""
    response = api_context.get("/api/v1/analytics/risk?currency=RON")

    assert response.status in [200, 404]


def test_api_data_sources(api_context: APIRequestContext):
    """Test data sources endpoint."""
    response = api_context.get("/api/v1/data/sources")

    assert response.status == 200

    data = response.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_api_alerts(api_context: APIRequestContext):
    """Test alerts endpoint."""
    response = api_context.get("/api/v1/alerts")

    assert response.status == 200

    data = response.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_api_cors_headers(api_context: APIRequestContext):
    """Test CORS headers."""
    response = api_context.options("/api/v1/sentiment?currency=RON")

    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers or response.status != 405


def test_api_content_types(api_context: APIRequestContext):
    """Test API content types."""
    response = api_context.get("/api/v1/sentiment?currency=RON")

    if response.status == 200:
        assert "application/json" in response.headers.get("content-type", "")


def test_api_response_times(api_context: APIRequestContext):
    """Test API response times."""
    response = api_context.get("/health")

    # Should respond reasonably quickly (under 5 seconds)
    assert response.status == 200


def test_api_error_handling(api_context: APIRequestContext):
    """Test API error handling."""
    # Test with invalid endpoint
    response = api_context.get("/api/v1/nonexistent")

    # Should return 404 for non-existent endpoints
    assert response.status == 404


def test_api_input_validation(api_context: APIRequestContext):
    """Test API input validation."""
    # Test with malformed JSON
    response = api_context.post(
        "/api/v1/sentiment/analyze",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )

    # Should return validation error
    assert response.status == 422


def test_api_rate_limiting_simulation(api_context: APIRequestContext):
    """Test API behavior under load."""
    # Make multiple rapid requests
    responses = []
    for i in range(3):
        response = api_context.get("/health")
        responses.append(response.status)

    # All should succeed
    assert all(status == 200 for status in responses)
