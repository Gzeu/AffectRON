"""
Trend analysis pipeline for financial market trends.
Analyzes temporal patterns, correlations, and market sentiment trends.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict, Counter
import statistics

from .base import BasePipeline, PipelineConfig, PipelineResult
from ..models import ExtractedData, SentimentAnalysis


logger = logging.getLogger(__name__)


class TrendAnalyzer(BasePipeline):
    """Pipeline for analyzing financial trends and patterns."""

    def __init__(self, config: PipelineConfig, db_session):
        super().__init__(config, db_session)

        # Time windows for trend analysis
        self.time_windows = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }

    async def load_model(self):
        """Load trend analysis models (if any)."""
        # Trend analysis is primarily statistical, no ML models needed
        logger.info("Trend analyzer initialized (no models to load)")

    def calculate_sentiment_trends(self, time_window: timedelta) -> Dict[str, Any]:
        """Calculate sentiment trends over a time window."""
        cutoff_time = datetime.now() - time_window

        # Get sentiment data within time window
        sentiment_data = self.db_session.query(SentimentAnalysis).join(ExtractedData).filter(
            SentimentAnalysis.created_at >= cutoff_time,
            ExtractedData.is_processed == True
        ).all()

        if not sentiment_data:
            return {}

        # Group by hour for trend analysis
        hourly_sentiment = defaultdict(list)

        for analysis in sentiment_data:
            hour_key = analysis.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_sentiment[hour_key].append(analysis.sentiment_score)

        # Calculate trends
        trends = {}
        for hour, scores in hourly_sentiment.items():
            trends[hour.isoformat()] = {
                'avg_sentiment': statistics.mean(scores),
                'sentiment_std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'sample_count': len(scores),
                'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
                'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores)
            }

        return trends

    def analyze_volume_trends(self, time_window: timedelta) -> Dict[str, Any]:
        """Analyze data volume trends."""
        cutoff_time = datetime.now() - time_window

        # Get data count by hour
        data_points = self.db_session.query(ExtractedData).filter(
            ExtractedData.created_at >= cutoff_time
        ).all()

        hourly_volume = defaultdict(int)

        for data in data_points:
            hour_key = data.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_volume[hour_key] += 1

        # Calculate volume statistics
        volumes = list(hourly_volume.values())

        if not volumes:
            return {}

        volume_trends = {
            'total_data_points': len(data_points),
            'avg_hourly_volume': statistics.mean(volumes),
            'max_hourly_volume': max(volumes),
            'volume_std': statistics.stdev(volumes) if len(volumes) > 1 else 0,
            'hourly_breakdown': {k.isoformat(): v for k, v in hourly_volume.items()}
        }

        return volume_trends

    def analyze_entity_frequency(self, time_window: timedelta) -> Dict[str, Any]:
        """Analyze frequency of financial entities over time."""
        cutoff_time = datetime.now() - time_window

        # Get processed data with sentiment analysis
        processed_data = self.db_session.query(ExtractedData).filter(
            ExtractedData.created_at >= cutoff_time,
            ExtractedData.is_processed == True
        ).all()

        entity_counts = Counter()
        currency_mentions = Counter()
        source_distribution = Counter()

        for data in processed_data:
            # Count entities from metadata
            if data.metadata:
                import json
                try:
                    metadata = json.loads(data.metadata) if isinstance(data.metadata, str) else data.metadata

                    # Extract entities from sentiment analysis
                    sentiment_analyses = self.db_session.query(SentimentAnalysis).filter_by(data_id=data.id).all()

                    for analysis in sentiment_analyses:
                        if analysis.entities:
                            entities = json.loads(analysis.entities) if isinstance(analysis.entities, str) else analysis.entities

                            for entity_type, entity_list in entities.items():
                                if isinstance(entity_list, list):
                                    for entity in entity_list:
                                        if isinstance(entity, dict) and 'text' in entity:
                                            entity_counts[entity['text']] += 1

                            # Count currencies separately
                            if 'CURRENCY' in entities:
                                for currency_entity in entities['CURRENCY']:
                                    if isinstance(currency_entity, dict) and 'currency' in currency_entity:
                                        currency_mentions[currency_entity['currency']] += 1

                except (json.JSONDecodeError, KeyError):
                    continue

            # Count by source
            source_distribution[data.source_id] += 1

        # Get top entities
        top_entities = dict(entity_counts.most_common(20))
        top_currencies = dict(currency_mentions.most_common(10))

        return {
            'top_entities': top_entities,
            'top_currencies': top_currencies,
            'source_distribution': dict(source_distribution),
            'unique_entities': len(entity_counts),
            'total_entity_mentions': sum(entity_counts.values())
        }

    def calculate_correlation_analysis(self) -> Dict[str, Any]:
        """Calculate correlations between different data sources and sentiment."""
        # Get recent data for correlation analysis
        cutoff_time = datetime.now() - timedelta(days=7)

        data_with_sentiment = self.db_session.query(ExtractedData, SentimentAnalysis).join(
            SentimentAnalysis, ExtractedData.id == SentimentAnalysis.data_id
        ).filter(
            ExtractedData.created_at >= cutoff_time
        ).all()

        if len(data_with_sentiment) < 10:
            return {'error': 'Insufficient data for correlation analysis'}

        # Group by source and calculate average sentiment
        source_sentiment = defaultdict(list)

        for data, sentiment in data_with_sentiment:
            source_sentiment[data.source_id].append(sentiment.sentiment_score)

        # Calculate source-wise sentiment averages
        source_averages = {}
        for source_id, scores in source_sentiment.items():
            if len(scores) > 5:  # Only consider sources with enough data
                source_averages[source_id] = {
                    'avg_sentiment': statistics.mean(scores),
                    'sentiment_std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'sample_count': len(scores)
                }

        return {
            'source_sentiment_analysis': source_averages,
            'total_sources_analyzed': len(source_averages)
        }

    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in sentiment and volume patterns."""
        # Get last 24 hours data
        cutoff_time = datetime.now() - timedelta(hours=24)

        recent_data = self.db_session.query(ExtractedData).filter(
            ExtractedData.created_at >= cutoff_time
        ).all()

        if len(recent_data) < 10:
            return {'error': 'Insufficient data for anomaly detection'}

        # Calculate volume baseline (last 7 days average)
        week_ago = datetime.now() - timedelta(days=7)
        baseline_data = self.db_session.query(ExtractedData).filter(
            ExtractedData.created_at >= week_ago
        ).all()

        hourly_volumes_recent = defaultdict(int)
        hourly_volumes_baseline = defaultdict(int)

        for data in recent_data:
            hour_key = data.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_volumes_recent[hour_key] += 1

        for data in baseline_data:
            hour_key = data.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_volumes_baseline[hour_key] += 1

        # Calculate average hourly volumes
        recent_avg = statistics.mean(hourly_volumes_recent.values()) if hourly_volumes_recent else 0
        baseline_avg = statistics.mean(hourly_volumes_baseline.values()) if hourly_volumes_baseline else 0

        # Detect volume anomalies
        volume_anomalies = []
        for hour, volume in hourly_volumes_recent.items():
            if baseline_avg > 0:
                deviation = abs(volume - baseline_avg) / baseline_avg
                if deviation > 2.0:  # 200% deviation threshold
                    volume_anomalies.append({
                        'hour': hour.isoformat(),
                        'volume': volume,
                        'baseline_avg': baseline_avg,
                        'deviation': deviation
                    })

        # Sentiment anomaly detection
        sentiment_anomalies = []
        recent_sentiment = self.db_session.query(SentimentAnalysis).filter(
            SentimentAnalysis.created_at >= cutoff_time
        ).all()

        if recent_sentiment:
            sentiment_scores = [s.sentiment_score for s in recent_sentiment]
            mean_sentiment = statistics.mean(sentiment_scores)
            std_sentiment = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0

            if std_sentiment > 0:
                for sentiment in recent_sentiment:
                    z_score = abs(sentiment.sentiment_score - mean_sentiment) / std_sentiment
                    if z_score > 2.5:  # 2.5 sigma threshold
                        sentiment_anomalies.append({
                            'timestamp': sentiment.created_at.isoformat(),
                            'sentiment_score': sentiment.sentiment_score,
                            'z_score': z_score
                        })

        return {
            'volume_anomalies': volume_anomalies,
            'sentiment_anomalies': sentiment_anomalies,
            'recent_volume_avg': recent_avg,
            'baseline_volume_avg': baseline_avg,
            'anomaly_detection_threshold': {
                'volume_deviation': 2.0,
                'sentiment_z_score': 2.5
            }
        }

    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process batch for trend analysis."""
        results = []

        try:
            # Perform comprehensive trend analysis
            trend_analysis = {}

            # Analyze different time windows
            for window_name, window_delta in self.time_windows.items():
                trend_analysis[f'{window_name}_sentiment'] = self.calculate_sentiment_trends(window_delta)
                trend_analysis[f'{window_name}_volume'] = self.analyze_volume_trends(window_delta)

            # Overall analysis
            trend_analysis['entity_frequency'] = self.analyze_entity_frequency(self.time_windows['24h'])
            trend_analysis['correlation_analysis'] = self.calculate_correlation_analysis()
            trend_analysis['anomalies'] = self.detect_anomalies()

            # Calculate overall metrics
            overall_metrics = {
                'total_analyzed_data': len(self.db_session.query(ExtractedData).filter(
                    ExtractedData.is_processed == True
                ).all()),
                'analysis_timestamp': datetime.now().isoformat(),
                'time_windows_analyzed': list(self.time_windows.keys())
            }

            result = {
                'trend_analysis': trend_analysis,
                'overall_metrics': overall_metrics,
                'analysis_type': 'comprehensive_trend_analysis'
            }

        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            result = {
                'trend_analysis': {},
                'overall_metrics': {},
                'analysis_type': 'error',
                'error_message': str(e)
            }

        # Return result for each text (trend analysis is global)
        for _ in texts:
            results.append(result)

        return results
