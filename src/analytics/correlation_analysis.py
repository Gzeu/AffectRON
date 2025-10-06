"""
Correlation analysis between sentiment and market movements.
Analyzes relationships between sentiment trends and price changes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..base import BaseAnalytics, AnalyticsConfig, AnalyticsResult


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    currency_pair: str
    correlation_coefficient: float
    p_value: float
    correlation_strength: str  # 'weak', 'moderate', 'strong', 'very_strong'
    lag_days: int  # Time lag for strongest correlation
    sentiment_impact: float  # 0-1 score of sentiment influence
    confidence_interval: Tuple[float, float]
    sample_size: int


class CorrelationAnalytics(BaseAnalytics):
    """Analyzes correlation between sentiment and market movements."""

    def __init__(self, config: AnalyticsConfig, db_session):
        super().__init__(config, db_session)

        # Correlation analysis parameters
        self.min_sample_size = 30  # Minimum data points for correlation
        self.max_lag_days = 7  # Maximum lag to analyze
        self.confidence_level = 0.95  # For confidence intervals

        # Supported currency pairs
        self.currency_pairs = [
            'RON/EUR', 'RON/USD', 'EUR/USD', 'EUR/RON', 'USD/RON',
            'BTC/USD', 'ETH/USD', 'GBP/EUR', 'CHF/EUR'
        ]

        self.logger = logging.getLogger(__name__)

    def get_sentiment_price_data(self, currency_pair: str, days: int = 30) -> pd.DataFrame:
        """Get combined sentiment and price data for correlation analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get price data
        price_data = self._get_price_data(currency_pair, start_date, end_date)

        # Get sentiment data
        sentiment_data = self._get_sentiment_data(currency_pair, start_date, end_date)

        if price_data.empty or sentiment_data.empty:
            return pd.DataFrame()

        # Combine data
        combined_data = self._combine_sentiment_price_data(price_data, sentiment_data)

        return combined_data

    def _get_price_data(self, currency_pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get price data for currency pair."""
        # This would query the market_data table
        # For now, return mock data structure
        return pd.DataFrame({
            'date': pd.date_range(start_date, end_date, freq='D'),
            'price': np.random.uniform(4.0, 5.0, (end_date - start_date).days + 1),
            'volume': np.random.uniform(1000000, 5000000, (end_date - start_date).days + 1)
        })

    def _get_sentiment_data(self, currency_pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get sentiment data for currency pair."""
        # This would query sentiment analysis results
        # For now, return mock data structure
        dates = pd.date_range(start_date, end_date, freq='D')

        return pd.DataFrame({
            'date': dates,
            'avg_sentiment': np.random.uniform(-0.5, 0.8, len(dates)),
            'sentiment_volume': np.random.randint(10, 100, len(dates)),
            'positive_ratio': np.random.uniform(0.3, 0.7, len(dates)),
            'negative_ratio': np.random.uniform(0.1, 0.4, len(dates))
        })

    def _combine_sentiment_price_data(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Combine sentiment and price data."""
        # Merge on date
        combined = pd.merge(price_df, sentiment_df, on='date', how='inner')

        if combined.empty:
            return combined

        # Calculate price changes
        combined['price_change'] = combined['price'].pct_change()
        combined['price_change_abs'] = combined['price_change'].abs()

        # Calculate sentiment metrics
        combined['sentiment_momentum'] = combined['avg_sentiment'].diff()

        return combined.dropna()

    def calculate_correlation(self, currency_pair: str, days: int = 30) -> CorrelationResult:
        """Calculate correlation between sentiment and price movements."""
        # Get combined data
        data = self.get_sentiment_price_data(currency_pair, days)

        if data.empty or len(data) < self.min_sample_size:
            return CorrelationResult(
                currency_pair=currency_pair,
                correlation_coefficient=0.0,
                p_value=1.0,
                correlation_strength='insufficient_data',
                lag_days=0,
                sentiment_impact=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size=len(data)
            )

        # Calculate immediate correlation (no lag)
        correlation_result = self._calculate_correlation_with_lag(data, 0)

        # Check different lag periods
        best_lag = 0
        best_correlation = correlation_result

        for lag in range(1, min(self.max_lag_days + 1, len(data) // 2)):
            lagged_result = self._calculate_correlation_with_lag(data, lag)

            if abs(lagged_result.correlation_coefficient) > abs(best_correlation.correlation_coefficient):
                best_correlation = lagged_result
                best_lag = lag

        # Calculate sentiment impact score
        sentiment_impact = self._calculate_sentiment_impact(data, best_correlation)

        return CorrelationResult(
            currency_pair=currency_pair,
            correlation_coefficient=best_correlation.correlation_coefficient,
            p_value=best_correlation.p_value,
            correlation_strength=self._classify_correlation_strength(best_correlation.correlation_coefficient),
            lag_days=best_lag,
            sentiment_impact=sentiment_impact,
            confidence_interval=best_correlation.confidence_interval,
            sample_size=len(data)
        )

    def _calculate_correlation_with_lag(self, data: pd.DataFrame, lag_days: int) -> CorrelationResult:
        """Calculate correlation with specified lag."""
        if lag_days == 0:
            # Immediate correlation
            sentiment_series = data['avg_sentiment']
            price_series = data['price_change']
        else:
            # Lagged correlation
            if lag_days >= len(data):
                return CorrelationResult(
                    currency_pair='',
                    correlation_coefficient=0.0,
                    p_value=1.0,
                    correlation_strength='insufficient_data',
                    lag_days=lag_days,
                    sentiment_impact=0.0,
                    confidence_interval=(0.0, 0.0),
                    sample_size=0
                )

            sentiment_series = data['avg_sentiment'].iloc[:-lag_days]
            price_series = data['price_change'].iloc[lag_days:]

        # Calculate correlation
        if len(sentiment_series) < 3 or len(price_series) < 3:
            return CorrelationResult(
                currency_pair='',
                correlation_coefficient=0.0,
                p_value=1.0,
                correlation_strength='insufficient_data',
                lag_days=lag_days,
                sentiment_impact=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size=0
            )

        correlation_coef, p_value = stats.pearsonr(sentiment_series, price_series)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            correlation_coef, len(sentiment_series), self.confidence_level
        )

        return CorrelationResult(
            currency_pair='',
            correlation_coefficient=correlation_coef,
            p_value=p_value,
            correlation_strength='',
            lag_days=lag_days,
            sentiment_impact=0.0,
            confidence_interval=confidence_interval,
            sample_size=len(sentiment_series)
        )

    def _calculate_confidence_interval(self, correlation: float, n: int, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        # Fisher transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se = 1 / np.sqrt(n - 3)

        # Critical value for confidence level
        if confidence == 0.95:
            z_crit = 1.96
        elif confidence == 0.99:
            z_crit = 2.576
        else:
            z_crit = 1.96  # Default to 95%

        # Confidence interval in Fisher space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se

        # Transform back to correlation space
        corr_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        corr_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        return (corr_lower, corr_upper)

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)

        if abs_corr < 0.1:
            return 'very_weak'
        elif abs_corr < 0.3:
            return 'weak'
        elif abs_corr < 0.5:
            return 'moderate'
        elif abs_corr < 0.7:
            return 'strong'
        else:
            return 'very_strong'

    def _calculate_sentiment_impact(self, data: pd.DataFrame, correlation: CorrelationResult) -> float:
        """Calculate sentiment impact score (0-1)."""
        # Factors that increase impact:
        # 1. Strong correlation
        # 2. Statistical significance (low p-value)
        # 3. Sufficient sample size
        # 4. Consistency over time

        impact_factors = []

        # Correlation strength factor
        correlation_strength = abs(correlation.correlation_coefficient)
        impact_factors.append(correlation_strength)

        # Statistical significance factor (1 - p_value)
        significance_factor = 1 - correlation.p_value
        impact_factors.append(significance_factor)

        # Sample size factor (normalized)
        sample_factor = min(correlation.sample_size / 100, 1.0)  # Cap at 100 samples
        impact_factors.append(sample_factor)

        # Average the factors
        sentiment_impact = np.mean(impact_factors)

        return min(sentiment_impact, 1.0)

    def analyze_multiple_currencies(self, days: int = 30) -> Dict[str, CorrelationResult]:
        """Analyze correlations for all supported currency pairs."""
        results = {}

        for currency_pair in self.currency_pairs:
            try:
                result = self.calculate_correlation(currency_pair, days)
                results[currency_pair] = result
            except Exception as e:
                self.logger.error(f"Error analyzing {currency_pair}: {e}")
                results[currency_pair] = None

        return results

    def generate_correlation_report(self, results: Dict[str, CorrelationResult]) -> Dict[str, Any]:
        """Generate comprehensive correlation analysis report."""
        valid_results = [r for r in results.values() if r and r.sample_size >= self.min_sample_size]

        if not valid_results:
            return {
                'total_pairs_analyzed': len(results),
                'valid_results': 0,
                'summary': 'Insufficient data for correlation analysis'
            }

        # Calculate summary statistics
        correlations = [r.correlation_coefficient for r in valid_results]
        impacts = [r.sentiment_impact for r in valid_results]

        # Classify results
        strong_correlations = [r for r in valid_results if r.correlation_strength in ['strong', 'very_strong']]
        significant_correlations = [r for r in valid_results if r.p_value < 0.05]

        report = {
            'total_pairs_analyzed': len(results),
            'valid_results': len(valid_results),
            'strong_correlations': len(strong_correlations),
            'significant_correlations': len(significant_correlations),
            'average_correlation': np.mean(correlations),
            'average_sentiment_impact': np.mean(impacts),
            'correlation_distribution': {
                'very_weak': len([r for r in valid_results if r.correlation_strength == 'very_weak']),
                'weak': len([r for r in valid_results if r.correlation_strength == 'weak']),
                'moderate': len([r for r in valid_results if r.correlation_strength == 'moderate']),
                'strong': len([r for r in valid_results if r.correlation_strength == 'strong']),
                'very_strong': len([r for r in valid_results if r.correlation_strength == 'very_strong'])
            },
            'top_performers': sorted(
                [r for r in valid_results if r.sentiment_impact > 0.5],
                key=lambda x: x.sentiment_impact,
                reverse=True
            )[:5],  # Top 5 by sentiment impact
            'analysis_timestamp': datetime.now().isoformat()
        }

        return report

    def predict_price_movement(self, currency_pair: str, current_sentiment: float) -> Dict[str, Any]:
        """Predict price movement based on current sentiment."""
        # Get historical correlation
        correlation_result = self.calculate_correlation(currency_pair, days=30)

        if correlation_result.sample_size < self.min_sample_size:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0,
                'expected_direction': 'neutral',
                'correlation_strength': 'unknown'
            }

        # Simple linear model prediction
        data = self.get_sentiment_price_data(currency_pair, days=30)

        if data.empty:
            return {
                'prediction': 'no_data',
                'confidence': 0.0,
                'expected_direction': 'neutral'
            }

        # Train simple linear regression
        X = data['avg_sentiment'].values.reshape(-1, 1)
        y = data['price_change'].values

        if len(X) < 5:  # Need minimum samples for regression
            return {
                'prediction': 'insufficient_samples',
                'confidence': 0.0,
                'expected_direction': 'neutral'
            }

        model = LinearRegression()
        model.fit(X, y)

        # Make prediction
        prediction = model.predict([[current_sentiment]])[0]

        # Determine direction
        if prediction > 0.001:
            expected_direction = 'upward'
        elif prediction < -0.001:
            expected_direction = 'downward'
        else:
            expected_direction = 'neutral'

        return {
            'prediction': 'success',
            'expected_price_change': prediction,
            'expected_direction': expected_direction,
            'confidence': correlation_result.sentiment_impact,
            'correlation_coefficient': correlation_result.correlation_coefficient,
            'model_r2_score': r2_score(y, model.predict(X))
        }

    async def run_analysis(self) -> List[AnalyticsResult]:
        """Run correlation analysis."""
        results = []

        try:
            # Analyze all currency pairs
            correlation_results = self.analyze_multiple_currencies(days=30)

            # Generate report
            report = self.generate_correlation_report(correlation_results)

            # Create analytics result
            result = AnalyticsResult(
                analytics_name="correlation_analysis",
                result_type="correlation_report",
                insights=report,
                confidence=np.mean([r.sentiment_impact for r in correlation_results.values()
                                  if r and r.sample_size >= self.min_sample_size]) if correlation_results else 0.0
            )

            results.append(result)

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")

        return results

    def get_analysis_data(self) -> Dict[str, Any]:
        """Get data for correlation analysis."""
        # This would query recent market and sentiment data
        return {
            'record_count': 0,
            'date_range': 'last_30_days',
            'supported_pairs': self.currency_pairs
        }
