"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    language: Optional[str] = Field("ro", description="Language code (ro, en)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        supported_languages = ['ro', 'en']
        if v not in supported_languages:
            raise ValueError(f'Language must be one of: {supported_languages}')
        return v


class SentimentData(BaseModel):
    """Sentiment analysis result data."""

    label: str = Field(..., description="Sentiment label (positive, negative, neutral)")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted entities")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    currency: str
    timeframe: str
    timestamp: datetime
    overall_sentiment: SentimentData
    trend_direction: str
    trend_strength: float
    data_points: int
    insights: List[str] = Field(default_factory=list)


class MarketDataPoint(BaseModel):
    """Market data point."""

    currency_pair: str
    rate: float
    source: str
    timestamp: datetime
    volume: Optional[float] = None


class MarketInsightsResponse(BaseModel):
    """Response model for market insights."""

    currency: str
    risk_level: str
    timestamp: datetime
    market_sentiment: str
    price_direction: str
    confidence: float
    recommendations: List[str]
    supporting_data: Dict[str, Any]


class RiskComponent(BaseModel):
    """Individual risk component."""

    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    description: str
    impact: str


class RiskScoreResponse(BaseModel):
    """Response model for risk scoring."""

    currency: str
    timestamp: datetime
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    risk_components: List[RiskComponent]
    trend_direction: str
    recommendations: List[str]


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    type: str  # sentiment_update, market_update, alert, etc.
    data: Dict[str, Any]
    timestamp: datetime


class AlertData(BaseModel):
    """Alert information."""

    alert_id: str
    type: str
    severity: str  # low, medium, high, critical
    title: str
    message: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """System status information."""

    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: int
    active_connections: int
    last_data_update: Optional[datetime]
    services_status: Dict[str, str]
