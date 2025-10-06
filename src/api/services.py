"""
Business logic services for AffectRON API.
Contains the core business logic for data processing and analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from ..models import ExtractedData, SentimentAnalysis, MarketData
from ..analytics import MarketInsightsAnalytics, RiskScoringAnalytics
from ..pipelines import SentimentPipeline, NERPipeline
from ..extractors import NewsExtractor, TwitterExtractor, FXExtractor


logger = logging.getLogger(__name__)


class SentimentService:
    """Service for sentiment analysis operations."""

    def __init__(self):
        self.sentiment_pipeline = None
        self.ner_pipeline = None

    async def get_sentiment_analysis(self, currency: str, timeframe: str, limit: int, db: Session) -> Dict[str, Any]:
        """Get sentiment analysis for a currency."""
        # Parse timeframe
        timeframe_delta = self._parse_timeframe(timeframe)

        # Get data from database
        cutoff_time = datetime.now() - timeframe_delta

        sentiment_data = db.query(SentimentAnalysis).join(ExtractedData).filter(
            ExtractedData.created_at >= cutoff_time,
            SentimentAnalysis.model_name == "finbert_sentiment"
        ).limit(limit).all()

        if not sentiment_data:
            return self._empty_sentiment_response(currency, timeframe)

        # Calculate aggregate sentiment
        scores = [s.sentiment_score for s in sentiment_data]
        avg_sentiment = sum(scores) / len(scores)

        # Determine trend direction
        recent_scores = scores[-10:] if len(scores) >= 10 else scores
        older_scores = scores[:10] if len(scores) >= 20 else scores[:len(scores)//2]

        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        older_avg = sum(older_scores) / len(older_scores) if older_scores else 0

        trend_direction = "upward" if recent_avg > older_avg else "downward"
        trend_strength = abs(recent_avg - older_avg)

        # Generate insights
        insights = self._generate_sentiment_insights(sentiment_data, avg_sentiment)

        return {
            "currency": currency,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "overall_sentiment": {
                "label": self._sentiment_label(avg_sentiment),
                "score": avg_sentiment,
                "confidence": min(0.9, len(sentiment_data) / 50)
            },
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "data_points": len(sentiment_data),
            "insights": insights
        }

    async def get_sentiment_history(self, currency: str, start_date: datetime, end_date: datetime, db: Session) -> List[Dict[str, Any]]:
        """Get historical sentiment data."""
        # This would implement historical data retrieval
        # For now, return a placeholder
        return []

    async def analyze_custom_text(self, text: str, language: str, db: Session) -> Dict[str, Any]:
        """Analyze sentiment of custom text."""
        # Initialize pipelines if needed
        if not self.sentiment_pipeline:
            from ..pipelines import PipelineConfig
            config = PipelineConfig(name="custom_sentiment", model_path="ProsusAI/finbert")
            self.sentiment_pipeline = SentimentPipeline(config, db)

        # Process the text
        results = await self.sentiment_pipeline.process_batch([text])

        if results:
            result = results[0]
            return {
                "text": text,
                "language": language,
                "timestamp": datetime.now(),
                "sentiment": result.get("sentiment", {}),
                "entities": result.get("entities", {}),
                "confidence": result.get("confidence", 0.0)
            }

        return {
            "text": text,
            "language": language,
            "timestamp": datetime.now(),
            "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.0},
            "entities": {},
            "confidence": 0.0
        }

    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string to timedelta."""
        timeframes = {
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        return timeframes.get(timeframe, timedelta(hours=1))

    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

    def _empty_sentiment_response(self, currency: str, timeframe: str) -> Dict[str, Any]:
        """Return empty response structure."""
        return {
            "currency": currency,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "overall_sentiment": {
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.0
            },
            "trend_direction": "stable",
            "trend_strength": 0.0,
            "data_points": 0,
            "insights": ["No data available for the specified period"]
        }

    def _generate_sentiment_insights(self, sentiment_data: List, avg_sentiment: float) -> List[str]:
        """Generate insights based on sentiment data."""
        insights = []

        if avg_sentiment > 0.3:
            insights.append("Strong positive sentiment detected in recent data")
        elif avg_sentiment < -0.3:
            insights.append("Strong negative sentiment detected - caution advised")
        else:
            insights.append("Sentiment is relatively neutral")

        if len(sentiment_data) < 10:
            insights.append("Limited data points - analysis confidence may be low")

        return insights


class MarketService:
    """Service for market data operations."""

    def __init__(self):
        self.fx_extractor = None

    async def get_exchange_rates(self, currencies: List[str], db: Session) -> Dict[str, Any]:
        """Get exchange rates for specified currencies."""
        # Get latest market data from database
        latest_rates = {}

        for currency in currencies:
            # Query for latest rate for each currency pair
            market_data = db.query(MarketData).filter(
                MarketData.currency_pair.like(f"%{currency}%")
            ).order_by(MarketData.timestamp.desc()).first()

            if market_data:
                latest_rates[currency] = {
                    "rate": market_data.rate,
                    "currency_pair": market_data.currency_pair,
                    "source": market_data.source,
                    "timestamp": market_data.timestamp.isoformat(),
                    "metadata": market_data.metadata
                }

        return {
            "timestamp": datetime.now(),
            "rates": latest_rates,
            "requested_currencies": currencies
        }


class AnalyticsService:
    """Service for analytics operations."""

    def __init__(self):
        self.market_analytics = None
        self.risk_analytics = None

    async def get_market_insights(self, currency: str, risk_level: str, db: Session) -> Dict[str, Any]:
        """Get market insights for a currency."""
        # Initialize analytics if needed
        if not self.market_analytics:
            from ..analytics import AnalyticsConfig
            config = AnalyticsConfig(name="market_insights")
            self.market_analytics = MarketInsightsAnalytics(config, db)

        # Run analysis
        results = await self.market_analytics.run_analysis()

        # Extract relevant insights for the currency
        # This is a simplified implementation
        return {
            "currency": currency,
            "risk_level": risk_level,
            "timestamp": datetime.now(),
            "market_sentiment": "neutral",
            "price_direction": "sideways",
            "confidence": 0.7,
            "recommendations": [
                "Monitor sentiment trends closely",
                "Consider position sizing based on risk tolerance"
            ],
            "supporting_data": {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points_analyzed": 0
            }
        }

    async def get_risk_assessment(self, currency: str, db: Session) -> Dict[str, Any]:
        """Get risk assessment for a currency."""
        # Initialize risk analytics if needed
        if not self.risk_analytics:
            from ..analytics import AnalyticsConfig
            config = AnalyticsConfig(name="risk_scoring")
            self.risk_analytics = RiskScoringAnalytics(config, db)

        # Run risk analysis
        results = await self.risk_analytics.run_analysis()

        # Extract relevant risk data for the currency
        # This is a simplified implementation
        return {
            "currency": currency,
            "timestamp": datetime.now(),
            "overall_risk_score": 0.4,
            "risk_level": "medium",
            "risk_components": [
                {
                    "name": "sentiment_volatility",
                    "score": 0.3,
                    "description": "Market sentiment volatility",
                    "impact": "medium"
                }
            ],
            "trend_direction": "stable",
            "recommendations": [
                "Monitor risk indicators regularly",
                "Consider hedging strategies"
            ]
        }

    async def get_trend_analysis(self, timeframe: str, db: Session) -> Dict[str, Any]:
        """Get comprehensive trend analysis."""
        # This would implement trend analysis across all markets
        return {
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "trends": {
                "sentiment": "neutral",
                "market": "sideways",
                "volume": "normal"
            },
            "anomalies": [],
            "insights": [
                "No significant anomalies detected",
                "Market conditions are relatively stable"
            ]
        }


class DataProcessingService:
    """Service for background data processing."""

    def __init__(self):
        self.news_extractor = None
        self.twitter_extractor = None
        self.fx_extractor = None

    async def process_data_cycle(self):
        """Run one cycle of data processing."""
        logger.info("Starting data processing cycle")

        try:
            # Initialize extractors if needed
            if not self.news_extractor:
                from ..extractors import ExtractorConfig
                from ..models import Base
                from ..database import SessionLocal

                db = SessionLocal()
                config = ExtractorConfig(name="news_extractor")
                self.news_extractor = NewsExtractor(config, db)

            if not self.twitter_extractor:
                from ..extractors import ExtractorConfig
                from ..database import SessionLocal

                db = SessionLocal()
                config = ExtractorConfig(name="twitter_extractor")
                self.twitter_extractor = TwitterExtractor(config, db)

            if not self.fx_extractor:
                from ..extractors import ExtractorConfig
                from ..database import SessionLocal

                db = SessionLocal()
                config = ExtractorConfig(name="fx_extractor")
                self.fx_extractor = FXExtractor(config, db)

            # Run extractions
            await self.news_extractor.run_extraction()
            await self.twitter_extractor.run_extraction()
            await self.fx_extractor.run_extraction()

            logger.info("Data processing cycle completed successfully")

        except Exception as e:
            logger.error(f"Error in data processing cycle: {e}")
            raise
