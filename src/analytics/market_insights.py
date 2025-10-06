"""
Market insights analytics for AffectRON.
Generates insights about market trends, sentiment correlations, and trading opportunities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, Counter
import statistics

from .base import BaseAnalytics, AnalyticsConfig, AnalyticsResult


logger = logging.getLogger(__name__)


class MarketInsightsAnalytics(BaseAnalytics):
    """Analytics module for generating market insights."""

    def __init__(self, config: AnalyticsConfig, db_session):
        super().__init__(config, db_session)

    async def analyze(self) -> List[AnalyticsResult]:
        """Generate market insights."""
        analysis_data = self.get_analysis_data()

        if analysis_data['record_count'] == 0:
            return []

        insights = []

        # Generate different types of insights
        sentiment_insights = self._analyze_sentiment_trends(analysis_data)
        if sentiment_insights:
            insights.append(AnalyticsResult(
                analytics_name=self.config.name,
                result_type="sentiment_trends",
                insights=sentiment_insights,
                confidence=0.8
            ))

        market_correlation_insights = self._analyze_market_correlations(analysis_data)
        if market_correlation_insights:
            insights.append(AnalyticsResult(
                analytics_name=self.config.name,
                result_type="market_correlations",
                insights=market_correlation_insights,
                confidence=0.75
            ))

        entity_insights = self._analyze_entity_impact(analysis_data)
        if entity_insights:
            insights.append(AnalyticsResult(
                analytics_name=self.config.name,
                result_type="entity_impact",
                insights=entity_insights,
                confidence=0.7
            ))

        volume_insights = self._analyze_volume_patterns(analysis_data)
        if volume_insights:
            insights.append(AnalyticsResult(
                analytics_name=self.config.name,
                result_type="volume_patterns",
                insights=volume_insights,
                confidence=0.85
            ))

        return insights

    def _analyze_sentiment_trends(self, analysis_data) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        sentiment_scores = []
        timestamps = []

        for data, sentiment in analysis_data['sentiment_data']:
            sentiment_scores.append(sentiment.sentiment_score)
            timestamps.append(sentiment.created_at)

        if not sentiment_scores:
            return {}

        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, sentiment_scores))
        sorted_timestamps, sorted_scores = zip(*sorted_pairs)

        # Calculate trend metrics
        overall_stats = self.calculate_sentiment_aggregate(sentiment_scores)

        # Detect trend direction (simple linear trend)
        if len(sorted_scores) > 5:
            # Use first 30% and last 30% for trend detection
            first_third = sorted_scores[:len(sorted_scores)//3]
            last_third = sorted_scores[-len(sorted_scores)//3:]

            first_mean = statistics.mean(first_third)
            last_mean = statistics.mean(last_third)

            trend_direction = "upward" if last_mean > first_mean else "downward"
            trend_strength = abs(last_mean - first_mean) / abs(first_mean) if first_mean != 0 else 0
        else:
            trend_direction = "insufficient_data"
            trend_strength = 0

        return {
            'overall_sentiment': overall_stats,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'analysis_period': {
                'start': sorted_timestamps[0].isoformat(),
                'end': sorted_timestamps[-1].isoformat(),
                'duration_hours': (sorted_timestamps[-1] - sorted_timestamps[0]).total_seconds() / 3600
            },
            'sample_distribution': self._calculate_sample_distribution(sentiment_scores)
        }

    def _analyze_market_correlations(self, analysis_data) -> Dict[str, Any]:
        """Analyze correlations between sentiment and market data."""
        if not analysis_data['market_data']:
            return {}

        # Group market data by currency pair
        market_by_currency = defaultdict(list)
        for market_data in analysis_data['market_data']:
            market_by_currency[market_data.currency_pair].append(market_data)

        correlations = {}

        for currency_pair, market_points in market_by_currency.items():
            if len(market_points) < 5:
                continue

            # Calculate market volatility (rate changes)
            rates = [point.rate for point in market_points]
            market_volatility = statistics.stdev(rates) if len(rates) > 1 else 0

            # Get sentiment data around market data timestamps
            relevant_sentiment = []
            for market_point in market_points:
                # Find sentiment within 1 hour of market data
                for data, sentiment in analysis_data['sentiment_data']:
                    if abs((sentiment.created_at - market_point.timestamp).total_seconds()) < 3600:
                        relevant_sentiment.append(sentiment.sentiment_score)

            if relevant_sentiment:
                sentiment_volatility = statistics.stdev(relevant_sentiment) if len(relevant_sentiment) > 1 else 0
                sentiment_mean = statistics.mean(relevant_sentiment)

                correlations[currency_pair] = {
                    'market_volatility': market_volatility,
                    'sentiment_volatility': sentiment_volatility,
                    'avg_sentiment': sentiment_mean,
                    'correlation_potential': abs(sentiment_volatility - market_volatility) / max(market_volatility, 0.01),
                    'sample_count': len(relevant_sentiment)
                }

        return correlations

    def _analyze_entity_impact(self, analysis_data) -> Dict[str, Any]:
        """Analyze impact of different entities on sentiment."""
        entity_sentiment = defaultdict(list)

        for data, sentiment in analysis_data['sentiment_data']:
            # Extract entities from metadata
            if data.metadata:
                try:
                    metadata = json.loads(data.metadata) if isinstance(data.metadata, str) else data.metadata

                    # Get entities from sentiment analysis if available
                    if sentiment.entities:
                        entities = json.loads(sentiment.entities) if isinstance(sentiment.entities, str) else sentiment.entities

                        for entity_type, entity_list in entities.items():
                            if isinstance(entity_list, list):
                                for entity in entity_list:
                                    if isinstance(entity, dict) and 'text' in entity:
                                        entity_sentiment[entity['text']].append(sentiment.sentiment_score)

                except (json.JSONDecodeError, KeyError):
                    continue

        # Calculate impact for each entity
        entity_impact = {}
        for entity, scores in entity_sentiment.items():
            if len(scores) >= 3:  # Need minimum samples
                entity_impact[entity] = {
                    'avg_sentiment': statistics.mean(scores),
                    'sentiment_volatility': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'occurrence_count': len(scores),
                    'impact_score': abs(statistics.mean(scores)) * len(scores)  # Weighted by frequency
                }

        # Sort by impact score
        sorted_entities = sorted(entity_impact.items(), key=lambda x: x[1]['impact_score'], reverse=True)

        return {
            'top_entities': dict(sorted_entities[:10]),
            'total_entities_analyzed': len(entity_impact),
            'entity_coverage': len(entity_sentiment) / max(len(analysis_data['sentiment_data']), 1)
        }

    def _analyze_volume_patterns(self, analysis_data) -> Dict[str, Any]:
        """Analyze data volume patterns."""
        # Group sentiment data by hour
        hourly_volume = defaultdict(list)

        for data, sentiment in analysis_data['sentiment_data']:
            hour_key = sentiment.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_volume[hour_key].append(sentiment.sentiment_score)

        if not hourly_volume:
            return {}

        # Calculate volume statistics
        volumes = [len(scores) for scores in hourly_volume.values()]
        volume_stats = {
            'total_hours': len(hourly_volume),
            'avg_hourly_volume': statistics.mean(volumes),
            'max_hourly_volume': max(volumes),
            'min_hourly_volume': min(volumes),
            'volume_std': statistics.stdev(volumes) if len(volumes) > 1 else 0
        }

        # Identify peak hours
        peak_hours = sorted(hourly_volume.items(), key=lambda x: len(x[1]), reverse=True)[:5]

        return {
            'volume_statistics': volume_stats,
            'peak_hours': {hour.isoformat(): len(scores) for hour, scores in peak_hours},
            'hourly_breakdown': {hour.isoformat(): len(scores) for hour, scores in hourly_volume.items()},
            'coverage_ratio': len(hourly_volume) / max((datetime.now() - analysis_data['cutoff_time']).total_seconds() / 3600, 1)
        }

    def _calculate_sample_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate distribution of sentiment scores."""
        if not scores:
            return {}

        # Create bins for distribution
        bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
        bin_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']

        distribution = {label: 0 for label in bin_labels}

        for score in scores:
            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                if bin_start <= score < bin_end:
                    distribution[bin_labels[i]] += 1
                    break

        return {
            'distribution': distribution,
            'skewness': self._calculate_skewness(scores),
            'kurtosis': self._calculate_kurtosis(scores)
        }

    def _calculate_skewness(self, scores: List[float]) -> float:
        """Calculate skewness of sentiment scores."""
        if len(scores) < 3:
            return 0.0

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)
        if std_score == 0:
            return 0.0

        # Simplified skewness calculation
        skewness = sum(((score - mean_score) / std_score) ** 3 for score in scores) / len(scores)
        return skewness

    def _calculate_kurtosis(self, scores: List[float]) -> float:
        """Calculate kurtosis of sentiment scores."""
        if len(scores) < 4:
            return 0.0

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)
        if std_score == 0:
            return 0.0

        # Simplified kurtosis calculation
        kurtosis = sum(((score - mean_score) / std_score) ** 4 for score in scores) / len(scores) - 3
        return kurtosis
