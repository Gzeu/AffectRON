"""
Base extractor module for AffectRON data extraction system.
Provides common functionality for all data extractors.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import aiohttp
import requests
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models import DataSource, ExtractedData


logger = logging.getLogger(__name__)


class ExtractorConfig(BaseModel):
    """Configuration for data extractors."""

    name: str
    update_interval: int = 3600  # seconds
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100
    enabled: bool = True


class ExtractedContent(BaseModel):
    """Standardized format for extracted content."""

    source_id: str
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class BaseExtractor(ABC):
    """Abstract base class for all data extractors."""

    def __init__(self, config: ExtractorConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self.last_extraction: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get HTTP session."""
        if self._session is None:
            raise RuntimeError("Extractor must be used as async context manager")
        return self._session

    @abstractmethod
    async def extract(self) -> List[ExtractedContent]:
        """Extract data from the source. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source."""
        pass

    def should_extract(self) -> bool:
        """Check if extraction should run based on update interval."""
        if not self.config.enabled:
            return False

        if self.last_extraction is None:
            return True

        next_extraction = self.last_extraction + timedelta(seconds=self.config.update_interval)
        return datetime.now() >= next_extraction

    def save_extracted_data(self, contents: List[ExtractedContent]):
        """Save extracted data to database."""
        for content in contents:
            # Check if data already exists
            existing = self.db_session.query(ExtractedData).filter_by(
                source_id=content.source_id,
                url=content.url
            ).first()

            if existing:
                # Update existing record
                existing.content = content.content
                existing.title = content.title
                existing.published_at = content.published_at
                existing.metadata = json.dumps(content.metadata)
                existing.updated_at = datetime.now()
            else:
                # Create new record
                data_record = ExtractedData(
                    source_id=content.source_id,
                    content=content.content,
                    title=content.title,
                    url=content.url,
                    published_at=content.published_at,
                    metadata=json.dumps(content.metadata),
                    created_at=datetime.now()
                )
                self.db_session.add(data_record)

        self.db_session.commit()
        self.last_extraction = datetime.now()
        logger.info(f"Saved {len(contents)} records for {self.config.name}")

    async def run_extraction(self) -> List[ExtractedContent]:
        """Run the extraction process."""
        if not self.should_extract():
            logger.info(f"Skipping extraction for {self.config.name} - too soon")
            return []

        try:
            logger.info(f"Starting extraction for {self.config.name}")
            contents = await self.extract()
            self.save_extracted_data(contents)
            logger.info(f"Successfully extracted {len(contents)} items for {self.config.name}")
            return contents
        except Exception as e:
            logger.error(f"Error during extraction for {self.config.name}: {str(e)}")
            raise
