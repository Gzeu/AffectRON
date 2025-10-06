"""
Tests for correlation analysis functionality.
Tests sentiment-price correlation analysis and prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analytics.correlation_analysis import CorrelationAnalytics, CorrelationResult


class TestCorrelationAnalytics:
    """Test CorrelationAnalytics functionality."""

    @pytest.fixture
    def correlation_analytics(self, db_session):
        """Create CorrelationAnalytics instance for testing."""
        config = AnalyticsConfig(
            name="test_correlation",
            update_interval=timedelta(minutes=15),
            lookback_period=timedelta(days=1)
        )
        return CorrelationAnalytics(config, db_session)

    def test_get_sentiment_price_data_mock(self, correlation_analytics):
        """Test getting combined sentiment and price data."""
        with patch.object(correlation_analytics, '_get_price_data') as mock_price, \
             patch.object(correlation_analytics, '_get_sentiment_data') as mock_sentiment:

            # Mock price data
            mock_price.return_value = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10, freq='D'),
                'price': [4.5, 4.52, 4.48, 4.55, 4.53, 4.57, 4.54, 4.59, 4.56, 4.61],
                'volume': [1000000] * 10
            })

            # Mock sentiment data
            mock_sentiment.return_value = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10, freq='D'),
                'avg_sentiment': [0.2, 0.1, -0.1, 0.3, 0.0, 0.4, 0.2, 0.5, 0.1, 0.3],
                'sentiment_volume': [50] * 10,
                'positive_ratio': [0.6] * 10,
                'negative_ratio': [0.2] * 10
            })

            data = correlation_analytics.get_sentiment_price_data('EUR/RON', 10)

            assert not data.empty
            assert len(data) == 10
            assert 'price_change' in data.columns
            assert 'sentiment_momentum' in data.columns

    def test_calculate_correlation_sufficient_data(self, correlation_analytics):
        """Test correlation calculation with sufficient data."""
        # Create mock data with known correlation
        np.random.seed(42)
        n = 50

        # Create correlated data (sentiment -> price change)
        sentiment = np.random.normal(0, 0.3, n)
        price_change = 0.5 * sentiment + np.random.normal(0, 0.1, n)  # Correlation ~0.8

        data = pd.DataFrame({
            'avg_sentiment': sentiment,
            'price_change': price_change,
            'date': pd.date_range('2024-01-01', periods=n, freq='D')
        })

        with patch.object(correlation_analytics, 'get_sentiment_price_data', return_value=data):
            result = correlation_analytics.calculate_correlation('EUR/RON', 30)

            assert result.currency_pair == 'EUR/RON'
            assert result.sample_size == n
            assert abs(result.correlation_coefficient) > 0.3  # Should detect correlation
            assert result.p_value < 0.05  # Should be statistically significant

    def test_calculate_correlation_insufficient_data(self, correlation_analytics):
        """Test correlation calculation with insufficient data."""
        data = pd.DataFrame({
            'avg_sentiment': [0.1, 0.2],
            'price_change': [0.01, -0.01],
            'date': pd.date_range('2024-01-01', periods=2, freq='D')
        })

        with patch.object(correlation_analytics, 'get_sentiment_price_data', return_value=data):
            result = correlation_analytics.calculate_correlation('EUR/RON', 30)

            assert result.correlation_strength == 'insufficient_data'
            assert result.sample_size == 2

    def test_calculate_correlation_with_lag(self, correlation_analytics):
        """Test correlation calculation with different lag periods."""
        # Create data where sentiment leads price by 1 day
        sentiment = [0.1, 0.2, 0.3, 0.2, 0.1]
        price_change = [0.0, 0.1, 0.2, 0.3, 0.2]  # Lagged by 1 day

        data = pd.DataFrame({
            'avg_sentiment': sentiment,
            'price_change': price_change,
            'date': pd.date_range('2024-01-01', periods=5, freq='D')
        })

        with patch.object(correlation_analytics, 'get_sentiment_price_data', return_value=data):
            result = correlation_analytics.calculate_correlation('EUR/RON', 30)

            # Should detect lag effect
            assert result.lag_days >= 0
            assert result.sample_size > 0

    def test_classify_correlation_strength(self, correlation_analytics):
        """Test correlation strength classification."""
        assert correlation_analytics._classify_correlation_strength(0.05) == 'very_weak'
        assert correlation_analytics._classify_correlation_strength(0.25) == 'weak'
        assert correlation_analytics._classify_correlation_strength(0.45) == 'moderate'
        assert correlation_analytics._classify_correlation_strength(0.65) == 'strong'
        assert correlation_analytics._classify_correlation_strength(0.85) == 'very_strong'

    def test_calculate_sentiment_impact(self, correlation_analytics):
        """Test sentiment impact score calculation."""
        correlation_result = CorrelationResult(
            currency_pair='EUR/RON',
            correlation_coefficient=0.7,
            p_value=0.01,
            correlation_strength='strong',
            lag_days=1,
            sentiment_impact=0.0,  # Will be calculated
            confidence_interval=(0.5, 0.9),
            sample_size=100
        )

        # Mock data for impact calculation
        data = pd.DataFrame({
            'avg_sentiment': np.random.normal(0, 0.3, 50),
            'price_change': np.random.normal(0, 0.1, 50)
        })

        impact = correlation_analytics._calculate_sentiment_impact(data, correlation_result)

        assert 0.0 <= impact <= 1.0
        assert impact > 0.5  # Should be high due to strong correlation and significance

    def test_analyze_multiple_currencies(self, correlation_analytics):
        """Test analysis of multiple currency pairs."""
        with patch.object(correlation_analytics, 'calculate_correlation') as mock_calc:
            # Mock results for different pairs
            mock_calc.side_effect = [
                CorrelationResult('EUR/RON', 0.6, 0.01, 'strong', 0, 0.8, (0.4, 0.8), 50),
                CorrelationResult('USD/RON', 0.3, 0.05, 'moderate', 1, 0.5, (0.1, 0.5), 45),
                CorrelationResult('GBP/EUR', 0.1, 0.30, 'weak', 0, 0.2, (-0.1, 0.3), 25)
            ]

            results = correlation_analytics.analyze_multiple_currencies(days=30)

            assert len(results) == 3
            assert 'EUR/RON' in results
            assert 'USD/RON' in results
            assert 'GBP/EUR' in results

            # Check that strong correlation has higher impact
            assert results['EUR/RON'].sentiment_impact > results['USD/RON'].sentiment_impact

    def test_generate_correlation_report(self, correlation_analytics):
        """Test correlation report generation."""
        results = {
            'EUR/RON': CorrelationResult('EUR/RON', 0.7, 0.01, 'strong', 0, 0.8, (0.5, 0.9), 100),
            'USD/RON': CorrelationResult('USD/RON', 0.4, 0.03, 'moderate', 1, 0.6, (0.2, 0.6), 80),
            'GBP/EUR': CorrelationResult('GBP/EUR', 0.1, 0.40, 'weak', 0, 0.2, (-0.1, 0.3), 20)
        }

        report = correlation_analytics.generate_correlation_report(results)

        assert report['total_pairs_analyzed'] == 3
        assert report['valid_results'] == 3
        assert report['strong_correlations'] == 1
        assert report['significant_correlations'] == 2  # p < 0.05
        assert report['average_correlation'] > 0.3
        assert report['average_sentiment_impact'] > 0.4

        # Check distribution
        assert report['correlation_distribution']['strong'] == 1
        assert report['correlation_distribution']['moderate'] == 1
        assert report['correlation_distribution']['weak'] == 1

    def test_predict_price_movement(self, correlation_analytics):
        """Test price movement prediction based on sentiment."""
        # Mock correlation calculation
        correlation_result = CorrelationResult(
            currency_pair='EUR/RON',
            correlation_coefficient=0.6,
            p_value=0.01,
            correlation_strength='strong',
            lag_days=0,
            sentiment_impact=0.7,
            confidence_interval=(0.4, 0.8),
            sample_size=50
        )

        with patch.object(correlation_analytics, 'calculate_correlation', return_value=correlation_result), \
             patch.object(correlation_analytics, 'get_sentiment_price_data') as mock_data:

            # Mock historical data for regression
            mock_data.return_value = pd.DataFrame({
                'avg_sentiment': [0.1, 0.2, 0.3, 0.2, 0.4],
                'price_change': [0.005, 0.010, 0.015, 0.010, 0.020],
                'date': pd.date_range('2024-01-01', periods=5, freq='D')
            })

            prediction = correlation_analytics.predict_price_movement('EUR/RON', 0.3)

            assert prediction['prediction'] == 'success'
            assert 'expected_price_change' in prediction
            assert 'expected_direction' in prediction
            assert prediction['confidence'] > 0.5

    def test_predict_price_movement_insufficient_data(self, correlation_analytics):
        """Test prediction with insufficient data."""
        correlation_result = CorrelationResult(
            currency_pair='EUR/RON',
            correlation_coefficient=0.0,
            p_value=1.0,
            correlation_strength='insufficient_data',
            lag_days=0,
            sentiment_impact=0.0,
            confidence_interval=(0.0, 0.0),
            sample_size=5
        )

        with patch.object(correlation_analytics, 'calculate_correlation', return_value=correlation_result):
            prediction = correlation_analytics.predict_price_movement('EUR/RON', 0.3)

            assert prediction['prediction'] == 'insufficient_data'
            assert prediction['confidence'] == 0.0

    def test_calculate_confidence_interval(self, correlation_analytics):
        """Test confidence interval calculation."""
        # Test with known correlation
        correlation = 0.5
        n = 100
        confidence = 0.95

        ci_lower, ci_upper = correlation_analytics._calculate_confidence_interval(correlation, n, confidence)

        assert ci_lower < correlation < ci_upper
        assert ci_upper - ci_lower < 0.5  # Reasonable interval width

    def test_get_supported_currency_pairs(self, correlation_analytics):
        """Test getting supported currency pairs."""
        pairs = correlation_analytics.currency_pairs

        assert 'RON/EUR' in pairs
        assert 'EUR/USD' in pairs
        assert 'BTC/USD' in pairs
        assert len(pairs) > 5

    def test_get_analysis_data(self, correlation_analytics):
        """Test getting analysis data."""
        data = correlation_analytics.get_analysis_data()

        assert 'record_count' in data
        assert 'date_range' in data
        assert 'supported_pairs' in data
        assert data['supported_pairs'] == correlation_analytics.currency_pairs


class TestIntegration:
    """Integration tests for correlation analysis."""

    @pytest.mark.asyncio
    async def test_correlation_analysis_integration(self, db_session):
        """Test complete correlation analysis workflow."""
        config = AnalyticsConfig(name="integration_test")
        analytics = CorrelationAnalytics(config, db_session)

        # Run analysis
        results = await analytics.run_analysis()

        assert isinstance(results, list)

        if results:
            result = results[0]
            assert result.analytics_name == "correlation_analysis"
            assert result.result_type == "correlation_report"
            assert 'insights' in result.result_data

    def test_correlation_with_realistic_data_pattern(self, correlation_analytics):
        """Test correlation analysis with realistic market data pattern."""
        # Create realistic pattern: sentiment often leads price changes
        dates = pd.date_range('2024-01-01', periods=20, freq='D')

        # Simulate sentiment that precedes price movements
        sentiment_base = np.sin(np.arange(20) * 0.3)  # Oscillating sentiment
        price_changes = np.roll(sentiment_base, 1) * 0.3 + np.random.normal(0, 0.05, 20)  # Lagged response

        data = pd.DataFrame({
            'date': dates,
            'avg_sentiment': sentiment_base,
            'price_change': price_changes
        })

        with patch.object(correlation_analytics, 'get_sentiment_price_data', return_value=data):
            result = correlation_analytics.calculate_correlation('EUR/RON', 20)

            # Should detect some correlation due to the lag relationship
            assert result.sample_size == 20
            assert abs(result.correlation_coefficient) > 0.1  # Should detect pattern
            assert result.lag_days >= 0
