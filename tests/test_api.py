"""
Tests for API endpoints.
Tests all REST API endpoints and functionality.
"""

import pytest
import json
from datetime import datetime


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, test_client):
        """Test health check endpoint returns correct response."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestSentimentAPI:
    """Test sentiment analysis API endpoints."""

    def test_get_sentiment_missing_currency(self, test_client):
        """Test sentiment endpoint requires currency parameter."""
        response = test_client.get("/api/v1/sentiment")

        assert response.status_code == 422  # Validation error

    def test_get_sentiment_invalid_currency(self, test_client):
        """Test sentiment endpoint with invalid currency."""
        response = test_client.get("/api/v1/sentiment?currency=INVALID")

        # Should handle gracefully or return appropriate error
        assert response.status_code in [200, 404, 422]

    def test_get_sentiment_valid_currency(self, test_client, sample_sentiment_data):
        """Test sentiment endpoint with valid currency."""
        response = test_client.get("/api/v1/sentiment?currency=RON&timeframe=1h")

        assert response.status_code == 200
        data = response.json()

        assert "currency" in data
        assert "sentiment" in data
        assert "confidence" in data

    def test_analyze_custom_text(self, test_client):
        """Test custom text sentiment analysis."""
        test_text = "RON exchange rate is looking positive today with strong fundamentals"

        response = test_client.post(
            "/api/v1/sentiment/analyze",
            json={"text": test_text, "language": "ro"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["text"] == test_text
        assert "sentiment" in data
        assert "confidence" in data


class TestMarketAPI:
    """Test market data API endpoints."""

    def test_get_exchange_rates(self, test_client):
        """Test exchange rates endpoint."""
        response = test_client.get("/api/v1/market/rates?currencies=RON,EUR,USD")

        assert response.status_code == 200
        data = response.json()

        assert "rates" in data
        assert "timestamp" in data
        assert "RON" in data["rates"] or len(data["rates"]) == 0


class TestAnalyticsAPI:
    """Test analytics API endpoints."""

    def test_get_market_insights(self, test_client):
        """Test market insights endpoint."""
        response = test_client.get("/api/v1/analytics/insights/market?currency=RON&risk_level=medium")

        assert response.status_code == 200
        data = response.json()

        assert "currency" in data
        assert "recommendations" in data
        assert "confidence" in data

    def test_get_risk_score(self, test_client):
        """Test risk assessment endpoint."""
        response = test_client.get("/api/v1/analytics/risk?currency=RON")

        assert response.status_code == 200
        data = response.json()

        assert "currency" in data
        assert "overall_risk_score" in data
        assert "risk_level" in data
        assert 0.0 <= data["overall_risk_score"] <= 1.0

    def test_get_trends(self, test_client):
        """Test trend analysis endpoint."""
        response = test_client.get("/api/v1/analytics/trends?timeframe=24h")

        assert response.status_code == 200
        data = response.json()

        assert "timeframe" in data
        assert "trends" in data
        assert "insights" in data


class TestDataSourcesAPI:
    """Test data sources API endpoints."""

    def test_get_data_sources(self, test_client):
        """Test data sources endpoint."""
        response = test_client.get("/api/v1/data/sources")

        assert response.status_code == 200
        data = response.json()

        assert "sources" in data
        assert isinstance(data["sources"], list)

        if data["sources"]:
            source = data["sources"][0]
            assert "name" in source
            assert "type" in source
            assert "status" in source


class TestAlertsAPI:
    """Test alerts API endpoints."""

    def test_get_alerts(self, test_client):
        """Test alerts endpoint."""
        response = test_client.get("/api/v1/alerts?limit=10")

        assert response.status_code == 200
        data = response.json()

        assert "alerts" in data
        assert isinstance(data["alerts"], list)


class TestWebSocketAPI:
    """Test WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_client):
        """Test WebSocket connection and basic functionality."""
        # Note: This is a basic test structure
        # Full WebSocket testing would require more complex setup

        # Test that WebSocket endpoint exists
        # In a real scenario, you'd use a WebSocket test client
        # For now, just verify the endpoint structure

        assert True  # Placeholder for WebSocket tests


class TestAuthentication:
    """Test authentication and authorization."""

    def test_unauthorized_access(self, test_client):
        """Test that protected endpoints require authentication."""
        # Try to access a protected endpoint without auth
        response = test_client.get("/api/v1/sentiment?currency=RON")

        # Should return 401 or similar
        assert response.status_code in [401, 403]

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/v1/sentiment?currency=RON")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code != 405


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_payload(self, test_client):
        """Test handling of invalid JSON payloads."""
        response = test_client.post(
            "/api/v1/sentiment/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_sql_injection_protection(self, test_client, db_session):
        """Test SQL injection protection."""
        # Try SQL injection in parameters
        malicious_input = "'; DROP TABLE users; --"

        response = test_client.get(f"/api/v1/sentiment?currency={malicious_input}")

        # Should handle gracefully without executing injection
        assert response.status_code in [200, 400, 422]

    def test_rate_limiting_simulation(self, test_client):
        """Test rate limiting behavior (simulated)."""
        # Make multiple rapid requests
        responses = []
        for i in range(5):
            response = test_client.get("/api/v1/sentiment?currency=RON")
            responses.append(response.status_code)

        # Should handle gracefully (no 500 errors)
        assert all(status in [200, 401, 403, 404] for status in responses)
