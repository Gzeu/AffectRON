"""
Database models for AffectRON data extraction system.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DataSource(Base):
    """Model for data sources (news sites, APIs, etc.)."""

    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    source_type = Column(String(50), nullable=False)  # news, twitter, fx, rss
    url = Column(String(500), nullable=False)
    api_key = Column(String(200))  # Optional API key
    config = Column(JSON)  # Source-specific configuration
    is_active = Column(Boolean, default=True)
    last_extraction = Column(DateTime)
    extraction_interval = Column(Integer, default=3600)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    extracted_data = relationship("ExtractedData", back_populates="source")


class ExtractedData(Base):
    """Model for extracted content data."""

    __tablename__ = "extracted_data"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    content = Column(Text, nullable=False)
    title = Column(String(500))
    url = Column(String(1000), index=True)
    published_at = Column(DateTime)
    metadata = Column(JSON)  # Additional metadata (author, tags, etc.)
    sentiment_score = Column(Float)  # Will be populated by pipeline
    confidence_score = Column(Float)
    language = Column(String(10), default="ro")
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    source = relationship("DataSource", back_populates="extracted_data")


class SentimentAnalysis(Base):
    """Model for sentiment analysis results."""

    __tablename__ = "sentiment_analysis"

    id = Column(Integer, primary_key=True, index=True)
    data_id = Column(Integer, ForeignKey("extracted_data.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    sentiment_label = Column(String(20), nullable=False)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    confidence_score = Column(Float, nullable=False)  # 0 to 1
    entities = Column(JSON)  # Named entities found in text
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    data = relationship("ExtractedData", backref="sentiment_analyses")


class MarketData(Base):
    """Model for market and exchange rate data."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    currency_pair = Column(String(10), nullable=False, index=True)  # EUR-RON, USD-RON, etc.
    rate = Column(Float, nullable=False)
    source = Column(String(50), nullable=False)  # BNR, ECB, etc.
    timestamp = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSON)  # Additional market data
    created_at = Column(DateTime, default=datetime.utcnow)


class AggregatedData(Base):
    """Model for aggregated data results."""

    __tablename__ = "aggregated_data"

    id = Column(Integer, primary_key=True, index=True)
    aggregator_name = Column(String(100), nullable=False, index=True)
    aggregation_type = Column(String(50), nullable=False)  # merge, cluster, etc.
    data_point_count = Column(Integer, nullable=False)
    result_data = Column(JSON, nullable=False)  # Aggregated results
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
