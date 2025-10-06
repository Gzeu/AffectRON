"""
Analytics package for AffectRON.
Contains modules for market insights and risk scoring.
"""

from .base import BaseAnalytics, AnalyticsConfig, AnalyticsResult
from .market_insights import MarketInsightsAnalytics
from .risk_scoring import RiskScoringAnalytics

__all__ = [
    'BaseAnalytics',
    'AnalyticsConfig',
    'AnalyticsResult',
    'MarketInsightsAnalytics',
    'RiskScoringAnalytics'
]