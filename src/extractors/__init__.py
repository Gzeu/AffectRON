"""
Data extractors package for AffectRON.
Contains modules for extracting financial data from various sources.
"""

from .base import BaseExtractor, ExtractorConfig, ExtractedContent
from .news_extractor import NewsExtractor
from .twitter_extractor import TwitterExtractor
from .fx_extractor import FXExtractor

__all__ = [
    'BaseExtractor',
    'ExtractorConfig',
    'ExtractedContent',
    'NewsExtractor',
    'TwitterExtractor',
    'FXExtractor'
]