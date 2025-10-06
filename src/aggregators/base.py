"""
Base aggregator module for AffectRON data aggregation system.
Provides common functionality for all data aggregators.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import hashlib

from sqlalchemy.orm import Session

from ..models import ExtractedData, AggregatedData


logger = logging.getLogger(__name__)


class AggregatorConfig:
    """Configuration for data aggregators."""

    def __init__(self,
                 name: str,
                 batch_size: int = 100,
                 time_window: timedelta = timedelta(hours=1),
                 similarity_threshold: float = 0.8,
                 enabled: bool = True):
        self.name = name
        self.batch_size = batch_size
        self.time_window = time_window
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled


class AggregatedResult:
    """Standardized result format for aggregation operations."""

    def __init__(self,
                 aggregator_name: str,
                 aggregation_type: str,
                 data_points: List[int],
                 result_data: Dict[str, Any],
                 created_at: datetime = None):
        self.aggregator_name = aggregator_name
        self.aggregation_type = aggregation_type
        self.data_points = data_points
        self.result_data = result_data
        self.created_at = created_at or datetime.now()


class BaseAggregator(ABC):
    """Abstract base class for all data aggregators."""

    def __init__(self, config: AggregatorConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self.last_aggregation: Optional[datetime] = None

    @abstractmethod
    async def aggregate(self, data_batch: List[ExtractedData]) -> List[AggregatedResult]:
        """Aggregate a batch of data. Must be implemented by subclasses."""
        pass

    def should_aggregate(self) -> bool:
        """Check if aggregation should run based on time window."""
        if not self.config.enabled:
            return False

        if self.last_aggregation is None:
            return True

        next_aggregation = self.last_aggregation + self.config.time_window
        return datetime.now() >= next_aggregation

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple hashing."""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def generate_content_hash(self, content: str, title: str = None) -> str:
        """Generate a hash for content deduplication."""
        # Create a normalized version of the content for hashing
        normalized = content.lower().strip()
        if title:
            normalized = f"{title.lower().strip()} {normalized}"

        # Remove extra whitespace and punctuation for better matching
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)

        return hashlib.md5(normalized.encode()).hexdigest()

    def get_unprocessed_batch(self) -> List[ExtractedData]:
        """Get a batch of unprocessed data for aggregation."""
        return self.db_session.query(ExtractedData).filter_by(
            is_processed=True
        ).limit(self.config.batch_size).all()

    def save_aggregated_results(self, results: List[AggregatedResult]):
        """Save aggregated results to database."""
        for result in results:
            # Check if aggregation already exists
            existing = self.db_session.query(AggregatedData).filter_by(
                aggregator_name=result.aggregator_name,
                aggregation_type=result.aggregation_type,
                created_at=result.created_at
            ).first()

            if existing:
                # Update existing record
                existing.result_data = json.dumps(result.result_data)
                existing.updated_at = datetime.now()
            else:
                # Create new record
                agg_record = AggregatedData(
                    aggregator_name=result.aggregator_name,
                    aggregation_type=result.aggregation_type,
                    data_point_count=len(result.data_points),
                    result_data=json.dumps(result.result_data),
                    created_at=result.created_at
                )
                self.db_session.add(agg_record)

        self.db_session.commit()
        self.last_aggregation = datetime.now()
        logger.info(f"Saved {len(results)} aggregated results for {self.config.name}")

    async def run_aggregation(self) -> List[AggregatedResult]:
        """Run the aggregation process."""
        if not self.should_aggregate():
            logger.info(f"Skipping aggregation for {self.config.name} - too soon")
            return []

        try:
            logger.info(f"Starting aggregation for {self.config.name}")
            data_batch = self.get_unprocessed_batch()

            if not data_batch:
                logger.info(f"No data to aggregate for {self.config.name}")
                return []

            results = await self.aggregate(data_batch)
            self.save_aggregated_results(results)
            logger.info(f"Successfully aggregated {len(data_batch)} items into {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during aggregation for {self.config.name}: {str(e)}")
            raise
