"""
API routes for AffectRON.
Defines all REST endpoints for the application.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from .websocket import manager


logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
sentiment_service = SentimentService()
market_service = MarketService()
analytics_service = AnalyticsService()


@router.get("/sentiment", response_model=SentimentResponse)
async def get_sentiment(
    currency: str = Query(..., description="Currency code (RON, EUR, USD, BTC, ETH)"),
    timeframe: str = Query("1h", description="Time frame (15m, 1h, 6h, 24h, 7d)"),
    limit: int = Query(100, description="Maximum number of results"),
    current_user: dict = Depends(require_scopes(READ_SENTIMENT)),
    db: Session = Depends(get_db)
):
    """
    Get sentiment analysis for a specific currency.

    Returns sentiment scores and analysis for the specified currency and time frame.
    """
    try:
        result = await sentiment_service.get_sentiment_analysis(
            currency=currency,
            timeframe=timeframe,
            limit=limit,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting sentiment for {currency}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sentiment data: {str(e)}"
        )


@router.get("/sentiment/history", response_model=List[SentimentResponse])
async def get_sentiment_history(
    currency: str = Query(..., description="Currency code"),
    start_date: datetime = Query(..., description="Start date for historical data"),
    end_date: datetime = Query(..., description="End date for historical data"),
    current_user: dict = Depends(require_scopes(READ_SENTIMENT)),
    db: Session = Depends(get_db)
):
    """
    Get historical sentiment data for a currency.

    Returns sentiment analysis over a specific time period.
    """
    try:
        result = await sentiment_service.get_sentiment_history(
            currency=currency,
            start_date=start_date,
            end_date=end_date,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting sentiment history for {currency}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sentiment history: {str(e)}"
        )


@router.post("/sentiment/analyze", response_model=SentimentResponse)
async def analyze_text_sentiment(
    request: SentimentRequest,
    current_user: dict = Depends(require_scopes(READ_SENTIMENT)),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment of custom text.

    Accepts custom text and returns sentiment analysis results.
    """
    try:
        result = await sentiment_service.analyze_custom_text(
            text=request.text,
            language=request.language or "ro",
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing custom text sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text sentiment: {str(e)}"
        )


@router.get("/market/rates", response_model=Dict[str, Any])
async def get_exchange_rates(
    currencies: List[str] = Query(..., description="List of currency codes"),
    current_user: dict = Depends(require_scopes(READ_MARKET)),
    db: Session = Depends(get_db)
):
    """
    Get current exchange rates for specified currencies.

    Returns latest exchange rates from various sources.
    """
    try:
        result = await market_service.get_exchange_rates(
            currencies=currencies,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting exchange rates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving exchange rates: {str(e)}"
        )


@router.get("/market/insights", response_model=MarketInsightsResponse)
async def get_market_insights(
    currency: str = Query(..., description="Currency code"),
    risk_level: str = Query("medium", description="Risk tolerance level"),
    current_user: dict = Depends(require_scopes(READ_ANALYTICS)),
    db: Session = Depends(get_db)
):
    """
    Get market insights and recommendations.

    Returns AI-powered market insights based on sentiment and market data.
    """
    try:
        result = await analytics_service.get_market_insights(
            currency=currency,
            risk_level=risk_level,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting market insights for {currency}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market insights: {str(e)}"
        )


@router.get("/analytics/risk", response_model=RiskScoreResponse)
async def get_risk_score(
    currency: str = Query(..., description="Currency code"),
    current_user: dict = Depends(require_scopes(READ_ANALYTICS)),
    db: Session = Depends(get_db)
):
    """
    Get risk assessment for a currency.

    Returns comprehensive risk scoring based on multiple factors.
    """
    try:
        result = await analytics_service.get_risk_assessment(
            currency=currency,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting risk assessment for {currency}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving risk assessment: {str(e)}"
        )


@router.get("/analytics/trends", response_model=Dict[str, Any])
async def get_trends(
    timeframe: str = Query("24h", description="Analysis time frame"),
    current_user: dict = Depends(require_scopes(READ_ANALYTICS)),
    db: Session = Depends(get_db)
):
    """
    Get trend analysis across all markets.

    Returns trend analysis and anomaly detection results.
    """
    try:
        result = await analytics_service.get_trend_analysis(
            timeframe=timeframe,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"Error getting trend analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving trend analysis: {str(e)}"
        )


@router.get("/data/sources")
async def get_data_sources(
    current_user: dict = Depends(require_scopes(READ_ANALYTICS)),
    db: Session = Depends(get_db)
):
    """
    Get information about data sources.

    Returns list of configured data sources and their status.
    """
    try:
        # This would return information about news sources, APIs, etc.
        sources = [
            {
                "name": "BNR Official",
                "type": "exchange_rates",
                "status": "active",
                "last_update": datetime.utcnow().isoformat(),
                "description": "Romanian National Bank exchange rates"
            },
            {
                "name": "Romanian Financial News",
                "type": "news",
                "status": "active",
                "last_update": datetime.utcnow().isoformat(),
                "description": "Financial news from Romanian media"
            },
            {
                "name": "Twitter Financial Sentiment",
                "type": "social_media",
                "status": "active",
                "last_update": datetime.utcnow().isoformat(),
                "description": "Social media sentiment analysis"
            }
        ]
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving data sources: {str(e)}"
        )


@router.get("/alerts")
async def get_alerts(
    limit: int = Query(50, description="Maximum number of alerts"),
    current_user: dict = Depends(require_scopes(READ_ANALYTICS)),
    db: Session = Depends(get_db)
):
    """
    Get recent system alerts.

    Returns alerts about risk levels, anomalies, and system status.
    """
    try:
        # This would query the alerts table
        alerts = [
            {
                "id": 1,
                "type": "risk",
                "severity": "medium",
                "title": "EUR/RON Volatility Alert",
                "message": "Increased volatility detected in EUR/RON pair",
                "timestamp": datetime.utcnow().isoformat(),
                "is_read": False
            }
        ]
        return {"alerts": alerts[:limit]}
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
@router.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """
    WebSocket endpoint for real-time updates.

    Available channels:
    - sentiment: Sentiment analysis updates
    - market: Market data updates
    - alerts: System alerts
    - general: General updates
    """
    await manager.connect(websocket, channel)

    try:
        while True:
            # Receive messages from client (if needed)
            data = await websocket.receive_text()

            # Echo back for now - could implement client commands
            await manager.send_personal_message(
                {"type": "echo", "data": data, "timestamp": datetime.now().isoformat()},
                websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, channel)
