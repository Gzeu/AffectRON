"""
Main FastAPI application for AffectRON API.
Provides REST and GraphQL endpoints for financial sentiment analysis.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .config import get_settings
from .database import engine, get_db
from .models import Base
from .auth import verify_token
from .routes import router as api_router
from .websocket import ConnectionManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global connection manager for WebSocket connections
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AffectRON API server...")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Initialize background tasks
    asyncio.create_task(start_background_tasks())

    yield

    # Shutdown
    logger.info("Shutting down AffectRON API server...")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Initialize settings
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title="AffectRON API",
        description="Professional Financial Sentiment Analysis API for Romanian Markets",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add trusted host middleware for security
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "AffectRON API",
            "description": "Professional Financial Sentiment Analysis for Romanian Markets",
            "version": "1.0.0",
            "docs": "/docs" if settings.debug else None,
            "health": "/health"
        }

    return app


async def start_background_tasks():
    """Start background tasks for data processing."""
    logger.info("Starting background data processing tasks...")

    # Import here to avoid circular imports
    from .services import DataProcessingService

    service = DataProcessingService()

    while True:
        try:
            await service.process_data_cycle()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error in background processing: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


# Create the application instance
app = create_application()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
