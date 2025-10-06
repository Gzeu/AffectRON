"""
API package for AffectRON.
Contains FastAPI application, routes, models, and services.
"""

from .main import app, create_application
from .config import get_settings, Settings
from .database import get_db, engine
from .auth import get_current_user, require_scopes, verify_token
from .models import (
    SentimentRequest, SentimentResponse, MarketInsightsResponse,
    RiskScoreResponse, WebSocketMessage, AlertData, SystemStatus
)
from .services import SentimentService, MarketService, AnalyticsService, DataProcessingService
from .websocket import ConnectionManager
from .enhanced_websocket import (
    EnhancedConnectionManager, MessageType, SubscriptionType,
    WebSocketMessage as EnhancedWebSocketMessage, UserSubscription
)

__all__ = [
    'app',
    'create_application',
    'get_settings',
    'Settings',
    'get_db',
    'engine',
    'get_current_user',
    'require_scopes',
    'verify_token',
    'SentimentRequest',
    'SentimentResponse',
    'MarketInsightsResponse',
    'RiskScoreResponse',
    'WebSocketMessage',
    'AlertData',
    'SystemStatus',
    'SentimentService',
    'MarketService',
    'AnalyticsService',
    'DataProcessingService',
    'ConnectionManager',
    'EnhancedConnectionManager',
    'MessageType',
    'SubscriptionType',
    'EnhancedWebSocketMessage',
    'UserSubscription'
]