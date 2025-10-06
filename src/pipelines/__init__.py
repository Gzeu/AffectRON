"""
AI processing pipelines package for AffectRON.
Contains sentiment analysis, NER, and trend analysis pipelines.
"""

from .base import BasePipeline, PipelineConfig, PipelineResult
from .sentiment_pipeline import SentimentPipeline
from .ner_pipeline import NERPipeline
from .trend_analyzer import TrendAnalyzer
from .enhanced_sentiment import EnhancedSentimentPipeline, SentimentResult
from .aspect_sentiment import AspectBasedSentimentPipeline, AspectSentimentResult

__all__ = [
    'BasePipeline',
    'PipelineConfig',
    'PipelineResult',
    'SentimentPipeline',
    'NERPipeline',
    'TrendAnalyzer',
    'EnhancedSentimentPipeline',
    'SentimentResult',
    'AspectBasedSentimentPipeline',
    'AspectSentimentResult'
]