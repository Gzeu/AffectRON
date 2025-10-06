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
from .market_data_streamer import market_streamer, start_market_data_streaming, stop_market_data_streaming
from .multi_region_deployment import multi_region_manager, initialize_multi_region_deployment, get_multi_region_status


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global connection manager for WebSocket connections
manager = ConnectionManager()
enhanced_manager = connection_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AffectRON API server...")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Initialize multi-region deployment
    await initialize_multi_region_deployment()

    # Initialize marketplace
    await initialize_marketplace()

    # Initialize plugin system
    await initialize_plugin_system()

    # Initialize trading platforms
    await initialize_trading_platforms()

    # Start alert engine
    await start_alert_engine()

    # Start market data streaming
    await start_market_data_streaming()

    # Initialize background tasks
    asyncio.create_task(start_background_tasks())

    # Start WebSocket cleanup task
    asyncio.create_task(enhanced_manager.start_cleanup_task())

    yield

    # Shutdown
    logger.info("Shutting down AffectRON API server...")

    # Stop market data streaming
    await stop_market_data_streaming()

    # Close all WebSocket connections
    for connection_id in list(enhanced_manager.active_connections.keys()):
        await enhanced_manager.disconnect(connection_id)


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

    # Market data streaming status
    @app.get("/api/v1/streaming/status", tags=["Streaming"])
    async def get_streaming_status():
        """Get market data streaming status."""
        from .market_data_streamer import get_market_data_status

        return get_market_data_status()

    # Alert engine status
    @app.get("/api/v1/alerts/engine/status", tags=["Alerts"])
    async def get_alert_engine_status():
        """Get alert engine status."""
        return get_alert_engine_status()

    # Get active alerts
    @app.get("/api/v1/alerts/active", tags=["Alerts"])
    async def get_active_alerts():
        """Get all active alerts."""
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "type": alert.alert_type.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in alert_engine.get_active_alerts()
            ]
        }

    # Trading platform status
    @app.get("/api/v1/trading/status", tags=["Trading"])
    async def get_trading_status():
        """Get trading platform status."""
        return get_trading_status()

    # Plugin system status
    @app.get("/api/v1/plugins/status", tags=["Plugins"])
    async def get_plugin_status():
        """Get plugin system status."""
        return get_plugin_system_status()

    # Marketplace status
    @app.get("/api/v1/marketplace/status", tags=["Marketplace"])
    async def get_marketplace_status():
        """Get plugin marketplace status."""
        return get_marketplace_status()

    # Search marketplace plugins
    @app.get("/api/v1/marketplace/plugins", tags=["Marketplace"])
    async def search_marketplace_plugins(query: str = "", category: str = "", limit: int = 20):
        """Search for plugins in marketplace."""
        plugins = await plugin_marketplace.search_plugins(query, category, limit)
        return {
            "plugins": [
                {
                    "id": plugin.id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author,
                    "rating": plugin.rating,
                    "download_count": plugin.download_count,
                    "categories": plugin.categories,
                    "featured": plugin.featured,
                    "verified": plugin.verified
                }
                for plugin in plugins
            ]
        }

    # Multi-region deployment status
    @app.get("/api/v1/deployment/status", tags=["Deployment"])
    async def get_deployment_status():
        """Get multi-region deployment status."""
        return get_multi_region_status()

    # Get deployment regions
    @app.get("/api/v1/deployment/regions", tags=["Deployment"])
    async def get_deployment_regions():
        """Get information about deployment regions."""
        return {
            "regions": [
                {
                    "code": region.code,
                    "name": region.name,
                    "provider": region.provider,
                    "timezone": region.timezone,
                    "is_active": region.is_active,
                    "current_instances": region.current_instances,
                    "current_load": region.current_load,
                    "primary_currencies": region.primary_currencies,
                    "target_latency_ms": region.target_latency_ms
                }
                for region in multi_region_manager.regions.values()
            ]
        }

    # Get deployment costs
    @app.get("/api/v1/deployment/costs", tags=["Deployment"])
    async def get_deployment_costs():
        """Get deployment costs across regions."""
        return multi_region_manager.get_deployment_costs()

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
