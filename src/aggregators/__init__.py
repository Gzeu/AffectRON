"""
Data aggregators package for AffectRON.
Contains modules for data aggregation, deduplication, and temporal clustering.
"""

from .base import BaseAggregator, AggregatorConfig, AggregatedResult
from .merge_dedup import MergeDedupAggregator
from .temporal_cluster import TemporalClusteringAggregator

__all__ = [
    'BaseAggregator',
    'AggregatorConfig',
    'AggregatedResult',
    'MergeDedupAggregator',
    'TemporalClusteringAggregator'
]