"""
Temporal clustering aggregator for AffectRON.
Groups data points by time windows and analyzes temporal patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict
import statistics

from .base import BaseAggregator, AggregatorConfig, AggregatedResult
from ..models import ExtractedData


logger = logging.getLogger(__name__)


class TemporalClusteringAggregator(BaseAggregator):
    """Aggregator that clusters data by temporal patterns."""

    def __init__(self, config: AggregatorConfig, db_session):
        super().__init__(config, db_session)

        # Define time windows for clustering
        self.time_windows = {
            'minute': timedelta(minutes=1),
            '5_minutes': timedelta(minutes=5),
            '15_minutes': timedelta(minutes=15),
            'hour': timedelta(hours=1),
            '6_hours': timedelta(hours=6),
            'day': timedelta(days=1)
        }

    async def aggregate(self, data_batch: List[ExtractedData]) -> List[AggregatedResult]:
        """Aggregate data by temporal clustering."""
        if not data_batch:
            return []

        # Sort data by timestamp
        sorted_data = sorted(data_batch, key=lambda x: x.published_at or x.created_at)

        # Create temporal clusters for different time windows
        cluster_results = []

        for window_name, window_delta in self.time_windows.items():
            clusters = self._create_temporal_clusters(sorted_data, window_delta)

            for cluster_time, cluster_items in clusters.items():
                if len(cluster_items) < 2:
                    continue  # Skip clusters with single items

                # Analyze the cluster
                cluster_analysis = self._analyze_temporal_cluster(cluster_items, window_name)

                result = AggregatedResult(
                    aggregator_name=self.config.name,
                    aggregation_type=f"temporal_cluster_{window_name}",
                    data_points=[item.id for item in cluster_items],
                    result_data=cluster_analysis,
                    created_at=datetime.now()
                )

                cluster_results.append(result)

        return cluster_results

    def _create_temporal_clusters(self, sorted_data: List[ExtractedData],
                                 window_delta: timedelta) -> Dict[datetime, List[ExtractedData]]:
        """Create temporal clusters based on time windows."""
        clusters = defaultdict(list)

        for item in sorted_data:
            # Use published_at if available, otherwise created_at
            timestamp = item.published_at or item.created_at

            # Round timestamp to window boundary
            window_start = timestamp.replace(
                minute=timestamp.minute - (timestamp.minute % (window_delta.total_seconds() / 60)) if window_delta < timedelta(hours=1) else 0,
                second=0,
                microsecond=0
            )

            if window_delta >= timedelta(hours=1):
                window_start = window_start.replace(minute=0, second=0, microsecond=0)

            if window_delta >= timedelta(days=1):
                window_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)

            clusters[window_start].append(item)

        return clusters

    def _analyze_temporal_cluster(self, cluster_items: List[ExtractedData], window_name: str) -> Dict[str, Any]:
        """Analyze a temporal cluster of data points."""
        if not cluster_items:
            return {}

        # Extract timestamps for analysis
        timestamps = []
        for item in cluster_items:
            timestamp = item.published_at or item.created_at
            timestamps.append(timestamp)

        # Calculate temporal statistics
        if timestamps:
            time_diffs = []
            sorted_timestamps = sorted(timestamps)

            for i in range(1, len(sorted_timestamps)):
                diff = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
                time_diffs.append(diff)

            temporal_stats = {
                'cluster_size': len(cluster_items),
                'time_span_seconds': (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0,
                'avg_time_between_items': statistics.mean(time_diffs) if time_diffs else 0,
                'min_time_between_items': min(time_diffs) if time_diffs else 0,
                'max_time_between_items': max(time_diffs) if time_diffs else 0,
                'time_distribution': self._calculate_time_distribution(timestamps, window_name)
            }
        else:
            temporal_stats = {'cluster_size': len(cluster_items)}

        # Analyze content patterns in the cluster
        content_analysis = self._analyze_cluster_content(cluster_items)

        # Calculate sentiment trends if available
        sentiment_trends = self._analyze_cluster_sentiment(cluster_items)

        return {
            'window_name': window_name,
            'temporal_statistics': temporal_stats,
            'content_analysis': content_analysis,
            'sentiment_trends': sentiment_trends,
            'cluster_summary': {
                'total_items': len(cluster_items),
                'sources': list(set(item.source_id for item in cluster_items)),
                'has_financial_content': any(self._is_financial_content(item) for item in cluster_items)
            }
        }

    def _calculate_time_distribution(self, timestamps: List[datetime], window_name: str) -> Dict[str, Any]:
        """Calculate distribution of items over time within the window."""
        if len(timestamps) < 2:
            return {'distribution': 'single_item'}

        # Group by sub-windows
        sorted_timestamps = sorted(timestamps)

        if window_name == 'minute':
            # Group by 10-second intervals
            sub_windows = defaultdict(int)
            for ts in sorted_timestamps:
                sub_key = ts.replace(second=ts.second - (ts.second % 10))
                sub_windows[sub_key] += 1

        elif window_name == '5_minutes':
            # Group by 1-minute intervals
            sub_windows = defaultdict(int)
            for ts in sorted_timestamps:
                sub_key = ts.replace(second=0)
                sub_windows[sub_key] += 1

        elif window_name == '15_minutes':
            # Group by 5-minute intervals
            sub_windows = defaultdict(int)
            for ts in sorted_timestamps:
                minute_rounded = ts.minute - (ts.minute % 5)
                sub_key = ts.replace(minute=minute_rounded, second=0, microsecond=0)
                sub_windows[sub_key] += 1

        elif window_name == 'hour':
            # Group by 15-minute intervals
            sub_windows = defaultdict(int)
            for ts in sorted_timestamps:
                minute_rounded = ts.minute - (ts.minute % 15)
                sub_key = ts.replace(minute=minute_rounded, second=0, microsecond=0)
                sub_windows[sub_key] += 1

        else:
            # For longer windows, just count distribution
            sub_windows = {ts: 1 for ts in sorted_timestamps}

        return {
            'sub_window_counts': dict(sub_windows),
            'distribution_type': 'uniform' if len(set(sub_windows.values())) == 1 else 'clustered',
            'peak_time': max(sub_windows.items(), key=lambda x: x[1])[0].isoformat() if sub_windows else None
        }

    def _analyze_cluster_content(self, cluster_items: List[ExtractedData]) -> Dict[str, Any]:
        """Analyze content patterns within the cluster."""
        # Extract key terms from all items
        all_text = ""
        for item in cluster_items:
            title = item.title or ""
            content = item.content or ""
            all_text += f" {title} {content}"

        # Simple keyword extraction
        words = all_text.lower().split()
        word_counts = defaultdict(int)

        for word in words:
            if len(word) > 3 and word.isalpha():  # Filter short/common words
                word_counts[word] += 1

        # Get most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Check for financial keywords
        financial_keywords = ['ron', 'eur', 'usd', 'banca', 'burs', 'invest', 'economie', 'finant']
        financial_count = sum(1 for word, count in word_counts.items()
                            if word in financial_keywords and count > 0)

        return {
            'total_words': len(words),
            'unique_words': len(word_counts),
            'most_common_words': dict(common_words),
            'financial_keyword_density': financial_count / len(word_counts) if word_counts else 0,
            'avg_content_length': statistics.mean(len(item.content or "") for item in cluster_items)
        }

    def _analyze_cluster_sentiment(self, cluster_items: List[ExtractedData]) -> Dict[str, Any]:
        """Analyze sentiment patterns within the cluster."""
        # This would integrate with the sentiment analysis results
        # For now, return placeholder structure
        return {
            'has_sentiment_data': False,
            'sentiment_summary': 'Sentiment analysis integration needed'
        }

    def _is_financial_content(self, item: ExtractedData) -> bool:
        """Check if item contains financial content."""
        text_to_check = f"{item.title or ''} {item.content or ''}".lower()

        financial_indicators = [
            'ron', 'eur', 'usd', 'leu', 'euro', 'dolar',
            'banca', 'burs', 'invest', 'economie', 'finant',
            'curs', 'valut', 'piat', 'actiun', 'obligatiun'
        ]

        return any(indicator in text_to_check for indicator in financial_indicators)
