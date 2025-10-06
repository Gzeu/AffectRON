"""
AI processing pipelines package for AffectRON.
Contains sentiment analysis, NER, and trend analysis pipelines.
"""

from .base import BasePipeline, PipelineConfig, PipelineResult
from .sentiment_pipeline import SentimentPipeline
from .ner_pipeline import NERPipeline
from .trend_analyzer import TrendAnalyzer

__all__ = [
    'BasePipeline',
    'PipelineConfig',
    'PipelineResult',
    'SentimentPipeline',
    'NERPipeline',
    'TrendAnalyzer'
]