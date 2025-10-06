"""
Tests for analytics modules.
Tests market insights and risk scoring functionality.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.analytics.base import BaseAnalytics, AnalyticsConfig, AnalyticsResult
from src.analytics.market_insights import MarketInsightsAnalytics
from src.analytics.risk_scoring import RiskScoringAnalytics


class TestAnalyticsConfig:
    """Test AnalyticsConfig functionality."""

    def test_config_creation(self):
        """Test creating analytics configuration."""
        config = AnalyticsConfig(
            name="test_analytics",
            update_interval=timedelta(minutes=10),
            lookback_period=timedelta(days=3),
            confidence_threshold=0.8
        )

        assert config.name == "test_analytics"
        assert config.update_interval == timedelta(minutes=10)
        assert config.lookback_period == timedelta(days=3)
        assert config.confidence_threshold == 0.8
        assert config.enabled == True


class TestAnalyticsResult:
    """Test AnalyticsResult model."""

    def test_result_creation(self):
        """Test creating analytics result."""
        insights = {"trend": "upward", "confidence": 0.85}

        result = AnalyticsResult(
            analytics_name="test_analytics",
            result_type="trend_analysis",
            insights=insights,
            confidence=0.9
        )

        assert result.analytics_name == "test_analytics"
        assert result.result_type == "trend_analysis"
        assert result.insights == insights
        assert result.confidence == 0.9


class TestMarketInsightsAnalytics:
    """Test MarketInsightsAnalytics functionality."""

    @pytest.fixture
    def market_insights(self, db_session):
        """Create MarketInsightsAnalytics instance for testing."""
        config = AnalyticsConfig(
            name="test_market_insights",
            update_interval=timedelta(minutes=15),
            lookback_period=timedelta(days=1)
        )
        return MarketInsightsAnalytics(config, db_session)

    def test_calculate_sentiment_aggregate(self, market_insights):
        """Test sentiment aggregation calculation."""
        scores = [0.8, 0.6, -0.2, 0.4, 0.9]

        result = market_insights.calculate_sentiment_aggregate(scores)

        assert result["total_samples"] == 5
        assert abs(result["mean"] - 0.5) < 0.1  # Approximately 0.5
        assert 0.4 <= result["positive_ratio"] <= 0.6  # About 40% positive
        assert result["negative_ratio"] == 0.2  # 20% negative

    def test_analyze_sentiment_trends_empty_data(self, market_insights, db_session):
        """Test sentiment trends with no data."""
        # Ensure no sentiment data exists
        result = market_insights._analyze_sentiment_trends({"sentiment_data": []})

        assert result == {}

    def test_analyze_volume_patterns(self, market_insights, db_session):
        """Test volume pattern analysis."""
        # Mock sentiment data
        sentiment_data = [
            (Mock(), Mock(created_at=datetime.now() - timedelta(hours=i)))
            for i in range(10)
        ]

        result = market_insights._analyze_volume_patterns({"sentiment_data": sentiment_data})

        assert "volume_statistics" in result
        assert "peak_hours" in result
        assert result["volume_statistics"]["total_hours"] >= 1

    def test_calculate_sample_distribution(self, market_insights):
        """Test sample distribution calculation."""
        scores = [0.8, 0.6, -0.2, 0.4, 0.9, -0.7, 0.1]

        distribution = market_insights._calculate_sample_distribution(scores)

        assert "distribution" in distribution
        assert "skewness" in distribution
        assert "kurtosis" in distribution

        # Check distribution has all categories
        dist = distribution["distribution"]
        assert "positive" in dist
        assert "negative" in dist
        assert "neutral" in dist


class TestRiskScoringAnalytics:
    """Test RiskScoringAnalytics functionality."""

    @pytest.fixture
    def risk_analytics(self, db_session):
        """Create RiskScoringAnalytics instance for testing."""
        config = AnalyticsConfig(
            name="test_risk_scoring",
            update_interval=timedelta(minutes=20),
            lookback_period=timedelta(days=2)
        )
        return RiskScoringAnalytics(config, db_session)

    def test_calculate_risk_components_empty_data(self, risk_analytics):
        """Test risk calculation with no data."""
        analysis_data = {"sentiment_data": [], "market_data": []}

        components = risk_analytics._calculate_risk_components(analysis_data)

        assert all(isinstance(score, float) for score in components.values())
        assert all(0.0 <= score <= 1.0 for score in components.values())

    def test_calculate_overall_risk(self, risk_analytics):
        """Test overall risk score calculation."""
        risk_components = {
            "sentiment_volatility": 0.3,
            "market_volatility": 0.5,
            "volume_anomaly": 0.2,
            "entity_diversity": 0.4,
            "time_consistency": 0.1
        }

        overall_risk = risk_analytics._calculate_overall_risk(risk_components)

        assert 0.0 <= overall_risk <= 1.0
        # Should be weighted average around 0.3-0.4
        assert 0.2 <= overall_risk <= 0.6

    def test_categorize_risk_level(self, risk_analytics):
        """Test risk level categorization."""
        assert risk_analytics._categorize_risk_level(0.1) == "low"
        assert risk_analytics._categorize_risk_level(0.4) == "medium"
        assert risk_analytics._categorize_risk_level(0.7) == "high"
        assert risk_analytics._categorize_risk_level(0.9) == "critical"

    def test_generate_risk_recommendations(self, risk_analytics):
        """Test risk-based recommendations generation."""
        # High risk scenario
        recommendations = risk_analytics._generate_risk_recommendations(
            0.8, {"sentiment_volatility": 0.9, "market_volatility": 0.7}
        )

        assert len(recommendations) > 0
        assert any("high" in rec.lower() for rec in recommendations)
        assert any("caution" in rec.lower() or "hedging" in rec.lower() for rec in recommendations)

        # Low risk scenario
        recommendations = risk_analytics._generate_risk_recommendations(
            0.2, {"sentiment_volatility": 0.1, "market_volatility": 0.1}
        )

        assert len(recommendations) > 0
        # Should contain positive or neutral recommendations
        assert any("maintain" in rec.lower() or "normal" in rec.lower() for rec in recommendations)

    def test_calculate_volume_anomaly_risk(self, risk_analytics):
        """Test volume anomaly risk calculation."""
        # Mock analysis data with consistent volume
        analysis_data = {
            "sentiment_data": [
                (Mock(), Mock(created_at=datetime.now() - timedelta(hours=i)))
                for i in range(10)
            ]
        }

        risk = risk_analytics._calculate_volume_anomaly_risk(analysis_data)

        assert 0.0 <= risk <= 1.0

    def test_calculate_entity_diversity_risk(self, risk_analytics):
        """Test entity diversity risk calculation."""
        # Mock sentiment data with entities
        mock_sentiment = Mock()
        mock_sentiment.entities = json.dumps({
            "currencies": [{"text": "RON"}],
            "organizations": [{"text": "BNR"}]
        })

        analysis_data = {
            "sentiment_data": [
                (Mock(), mock_sentiment) for _ in range(5)
            ]
        }

        risk = risk_analytics._calculate_entity_diversity_risk(analysis_data)

        assert 0.0 <= risk <= 1.0


class TestIntegration:
    """Integration tests for analytics modules."""

    @pytest.mark.asyncio
    async def test_analytics_pipeline_integration(self, db_session):
        """Test complete analytics pipeline."""
        # Create market insights analytics
        config = AnalyticsConfig(name="integration_test")
        market_analytics = MarketInsightsAnalytics(config, db_session)

        # Run analysis
        results = await market_analytics.run_analysis()

        # Should return results (may be empty if no data)
        assert isinstance(results, list)

        # Create risk scoring analytics
        risk_analytics = RiskScoringAnalytics(config, db_session)

        # Run risk analysis
        risk_results = await risk_analytics.run_analysis()

        assert isinstance(risk_results, list)

    def test_analytics_data_flow(self, db_session):
        """Test data flow between analytics modules."""
        # Create sample data for testing data flow
        from src.models import ExtractedData, DataSource, SentimentAnalysis

        # Create data source
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        # Create extracted data
        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="RON rate analysis",
            title="RON Analysis",
            is_processed=True
        )
        db_session.add(extracted_data)
        db_session.commit()

        # Create sentiment analysis
        sentiment = SentimentAnalysis(
            data_id=extracted_data.id,
            model_name="test",
            sentiment_label="positive",
            sentiment_score=0.6,
            confidence_score=0.8,
            entities='{"currencies": ["RON"]}'
        )
        db_session.add(sentiment)
        db_session.commit()

        # Test that both analytics can access the data
        config = AnalyticsConfig(name="test")
        market_analytics = MarketInsightsAnalytics(config, db_session)

        # Get analysis data
        analysis_data = market_analytics.get_analysis_data()

        assert analysis_data["record_count"] >= 1
        assert len(analysis_data["sentiment_data"]) >= 1
