"""
Base analytics module for AffectRON analytics system.
Provides common functionality for all analytics modules.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import statistics

from sqlalchemy.orm import Session

from ..models import ExtractedData, SentimentAnalysis, MarketData, Alert


logger = logging.getLogger(__name__)


class AnalyticsConfig:
    """Configuration for analytics modules."""

    def __init__(self,
                 name: str,
                 update_interval: timedelta = timedelta(minutes=15),
                 lookback_period: timedelta = timedelta(days=7),
                 confidence_threshold: float = 0.7,
                 enabled: bool = True):
        self.name = name
        self.update_interval = update_interval
        self.lookback_period = lookback_period
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled


class AnalyticsResult:
    """Standardized result format for analytics operations."""

    def __init__(self,
                 analytics_name: str,
                 result_type: str,
                 insights: Dict[str, Any],
                 confidence: float,
                 created_at: datetime = None):
        self.analytics_name = analytics_name
        self.result_type = result_type
        self.insights = insights
        self.confidence = confidence
        self.created_at = created_at or datetime.now()


class BaseAnalytics(ABC):
    """Abstract base class for all analytics modules."""

    def __init__(self, config: AnalyticsConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self.last_analysis: Optional[datetime] = None

    @abstractmethod
    async def analyze(self) -> List[AnalyticsResult]:
        """Perform analytics. Must be implemented by subclasses."""
        pass

    def should_analyze(self) -> bool:
        """Check if analysis should run based on update interval."""
        if not self.config.enabled:
            return False

        if self.last_analysis is None:
            return True

        next_analysis = self.last_analysis + self.config.update_interval
        return datetime.now() >= next_analysis

    def get_analysis_data(self, lookback_period: timedelta = None) -> Dict[str, Any]:
        """Get data for analysis within the specified period."""
        if lookback_period is None:
            lookback_period = self.config.lookback_period

        cutoff_time = datetime.now() - lookback_period

        # Get processed data with sentiment analysis
        data_with_sentiment = self.db_session.query(ExtractedData, SentimentAnalysis).join(
            SentimentAnalysis, ExtractedData.id == SentimentAnalysis.data_id
        ).filter(
            ExtractedData.created_at >= cutoff_time,
            ExtractedData.is_processed == True,
            SentimentAnalysis.confidence_score >= self.config.confidence_threshold
        ).all()

        # Get market data
        market_data = self.db_session.query(MarketData).filter(
            MarketData.timestamp >= cutoff_time
        ).all()

        return {
            'sentiment_data': data_with_sentiment,
            'market_data': market_data,
            'cutoff_time': cutoff_time,
            'record_count': len(data_with_sentiment)
        }

    def calculate_sentiment_aggregate(self, sentiment_scores: List[float]) -> Dict[str, Any]:
        """Calculate aggregate sentiment statistics."""
        if not sentiment_scores:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }

        mean_score = statistics.mean(sentiment_scores)
        median_score = statistics.median(sentiment_scores)
        std_score = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0.0

        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count

        return {
            'mean': mean_score,
            'median': median_score,
            'std': std_score,
            'positive_ratio': positive_count / len(sentiment_scores),
            'negative_ratio': negative_count / len(sentiment_scores),
            'neutral_ratio': neutral_count / len(sentiment_scores),
            'total_samples': len(sentiment_scores)
        }

    def create_alert(self, alert_type: str, severity: str, title: str, message: str, data: Dict = None):
        """Create an alert in the database."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=json.dumps(data) if data else None,
            created_at=datetime.now()
        )

        self.db_session.add(alert)
        self.db_session.commit()

        logger.info(f"Created {severity} alert: {title}")

    async def run_analysis(self) -> List[AnalyticsResult]:
        """Run the analytics process."""
        if not self.should_analyze():
            logger.info(f"Skipping analysis for {self.config.name} - too soon")
            return []

        try:
            logger.info(f"Starting analysis for {self.config.name}")
            results = await self.analyze()
            self.last_analysis = datetime.now()
            logger.info(f"Successfully completed analysis for {self.config.name}")
            return results

        except Exception as e:
            logger.error(f"Error during analysis for {self.config.name}: {str(e)}")
            raise
