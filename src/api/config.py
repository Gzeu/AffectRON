"""
Configuration management for AffectRON API.
Loads settings from environment variables and provides validation.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=True)
    api_version: str = Field(default="v1")

    # Security
    secret_key: str = Field(...)
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    # CORS
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"])

    # Database
    database_url: str = Field(...)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # External APIs
    twitter_bearer_token: Optional[str] = Field(default=None)
    news_api_key: Optional[str] = Field(default=None)
    cryptocompare_api_key: Optional[str] = Field(default=None)

    # AI Models
    finbert_model_path: str = Field(default="ProsusAI/finbert")
    sentiment_threshold: float = Field(default=0.7)
    confidence_threshold: float = Field(default=0.8)

    # Data Processing
    batch_size: int = Field(default=100)
    processing_interval_seconds: int = Field(default=300)
    data_retention_days: int = Field(default=90)

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None)
    prometheus_port: int = Field(default=8000)

    # Feature Flags
    enable_twitter_analysis: bool = Field(default=True)
    enable_news_analysis: bool = Field(default=True)
    enable_crypto_analysis: bool = Field(default=True)
    enable_real_time_alerts: bool = Field(default=True)

    @validator('cors_origins', pre=True)
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from string to list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @validator('allowed_hosts', pre=True)
    def assemble_allowed_hosts(cls, v):
        """Parse allowed hosts from string to list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @validator('database_url')
    def validate_database_url(cls, v):
        """Ensure database URL is provided."""
        if not v:
            raise ValueError('DATABASE_URL must be provided')
        return v

    @validator('secret_key')
    def validate_secret_key(cls, v):
        """Ensure secret key is provided and secure."""
        if not v:
            raise ValueError('SECRET_KEY must be provided')
        if len(v) < 32:
            raise ValueError('SECRET_KEY must be at least 32 characters long')
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
