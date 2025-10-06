"""
Base pipeline module for AffectRON AI processing pipelines.
Provides common functionality for all AI processing pipelines.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import torch
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models import ExtractedData, SentimentAnalysis


logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """Configuration for AI pipelines."""

    name: str
    model_path: str
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"  # auto, cpu, cuda
    cache_dir: Optional[str] = None
    enabled: bool = True


class PipelineResult(BaseModel):
    """Standardized result format for pipeline processing."""

    data_id: int
    pipeline_name: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = {}


class BasePipeline(ABC):
    """Abstract base class for all AI processing pipelines."""

    def __init__(self, config: PipelineConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None

    def _get_device(self) -> str:
        """Determine the appropriate device for model inference."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    @abstractmethod
    async def load_model(self):
        """Load the AI model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts. Must be implemented by subclasses."""
        pass

    def should_process(self, data: ExtractedData) -> bool:
        """Check if data should be processed by this pipeline."""
        if not self.config.enabled:
            return False

        # Check if already processed
        existing = self.db_session.query(SentimentAnalysis).filter_by(
            data_id=data.id,
            model_name=self.config.name
        ).first()

        return existing is None

    def save_results(self, results: List[PipelineResult]):
        """Save pipeline results to database."""
        for result in results:
            # Create sentiment analysis record if applicable
            if 'sentiment' in result.results:
                sentiment_record = SentimentAnalysis(
                    data_id=result.data_id,
                    model_name=result.pipeline_name,
                    sentiment_label=result.results['sentiment']['label'],
                    sentiment_score=result.results['sentiment']['score'],
                    confidence_score=result.confidence,
                    entities=result.results.get('entities', {}),
                    created_at=datetime.now()
                )
                self.db_session.add(sentiment_record)

            # Update the original data record
            data_record = self.db_session.query(ExtractedData).filter_by(id=result.data_id).first()
            if data_record:
                if 'sentiment' in result.results:
                    data_record.sentiment_score = result.results['sentiment']['score']
                    data_record.confidence_score = result.confidence
                data_record.is_processed = True
                data_record.updated_at = datetime.now()

        self.db_session.commit()
        logger.info(f"Saved {len(results)} pipeline results for {self.config.name}")

    async def run_pipeline(self) -> List[PipelineResult]:
        """Run the pipeline on unprocessed data."""
        if not self.model:
            await self.load_model()

        # Get unprocessed data
        unprocessed_data = self.db_session.query(ExtractedData).filter_by(
            is_processed=False
        ).limit(self.config.batch_size).all()

        if not unprocessed_data:
            logger.info(f"No unprocessed data for {self.config.name}")
            return []

        # Extract texts for processing
        texts = [data.content for data in unprocessed_data]
        data_ids = [data.id for data in unprocessed_data]

        try:
            logger.info(f"Processing {len(texts)} texts with {self.config.name}")

            # Process batch
            start_time = datetime.now()
            batch_results = await self.process_batch(texts)
            processing_time = (datetime.now() - start_time).total_seconds()

            # Convert to PipelineResult objects
            results = []
            for i, result in enumerate(batch_results):
                pipeline_result = PipelineResult(
                    data_id=data_ids[i],
                    pipeline_name=self.config.name,
                    results=result,
                    confidence=result.get('confidence', 0.0),
                    processing_time=processing_time / len(texts),
                    metadata={
                        'model_path': self.config.model_path,
                        'device': self.device,
                        'batch_size': len(texts)
                    }
                )
                results.append(pipeline_result)

            # Save results
            self.save_results(results)

            logger.info(f"Successfully processed {len(results)} texts with {self.config.name}")
            return results

        except Exception as e:
            logger.error(f"Error during pipeline processing for {self.config.name}: {str(e)}")
            raise
