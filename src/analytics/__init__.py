"""
Analytics package for AffectRON.
Contains modules for market insights, risk scoring, and correlation analysis.
"""

from .base import BaseAnalytics, AnalyticsConfig, AnalyticsResult
from .market_insights import MarketInsightsAnalytics
from .risk_scoring import RiskScoringAnalytics
from .correlation_analysis import CorrelationAnalytics, CorrelationResult

__all__ = [
    'BaseAnalytics',
    'AnalyticsConfig',
    'AnalyticsResult',
    'MarketInsightsAnalytics',
    'RiskScoringAnalytics',
    'CorrelationAnalytics',
    'CorrelationResult'
]