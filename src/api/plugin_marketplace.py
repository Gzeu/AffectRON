"""
Plugin marketplace for AffectRON.
Community-driven marketplace for plugins and integrations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import hashlib

from .plugin_system import PluginInfo, plugin_manager


@dataclass
class MarketplacePlugin:
    """Plugin available in marketplace."""
    id: str
    name: str
    version: str
    description: str
    author: str
    author_email: str
    license: str = "MIT"

    # Marketplace metadata
    download_url: str
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None

    # Statistics
    download_count: int = 0
    rating: float = 0.0
    review_count: int = 0

    # Compatibility
    min_affectron_version: str = "1.0.0"
    supported_platforms: List[str] = None

    # Categories and tags
    categories: List[str] = None
    tags: List[str] = None

    # Status
    verified: bool = False
    featured: bool = False
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.supported_platforms is None:
            self.supported_platforms = ['linux', 'macos', 'windows']
        if self.categories is None:
            self.categories = ['utility']
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class PluginMarketplace:
    """Plugin marketplace management."""

    def __init__(self, marketplace_url: str = "https://marketplace.affectron.com"):
        self.marketplace_url = marketplace_url
        self.local_plugins: Dict[str, MarketplacePlugin] = {}
        self.installed_plugins: Dict[str, MarketplacePlugin] = {}

        # Cache for marketplace data
        self._cache = {}
        self._cache_expiry = {}

        self.logger = logging.getLogger(__name__)

    async def search_plugins(self, query: str = "", category: str = "", limit: int = 20) -> List[MarketplacePlugin]:
        """Search for plugins in marketplace."""
        cache_key = f"search_{hashlib.md5(f'{query}_{category}_{limit}'.encode()).hexdigest()}"

        # Check cache first
        if cache_key in self._cache and datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
            return self._cache[cache_key]

        try:
            # In production, this would query the marketplace API
            # For now, return mock data
            plugins = await self._search_mock_plugins(query, category, limit)

            # Cache results for 1 hour
            self._cache[cache_key] = plugins
            self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)

            return plugins

        except Exception as e:
            self.logger.error(f"Error searching marketplace: {e}")
            return []

    async def _search_mock_plugins(self, query: str, category: str, limit: int) -> List[MarketplacePlugin]:
        """Mock marketplace search (replace with real API call)."""
        # Mock plugin data
        mock_plugins = [
            MarketplacePlugin(
                id="twitter-sentiment-enhancer",
                name="Twitter Sentiment Enhancer",
                version="1.2.0",
                description="Enhanced sentiment analysis for Twitter data with hashtag and mention extraction",
                author="AffectRON Community",
                author_email="community@affectron.com",
                download_url="https://github.com/affectron/plugins/raw/main/twitter-sentiment-enhancer.zip",
                repository_url="https://github.com/affectron/plugins/tree/main/twitter-sentiment-enhancer",
                documentation_url="https://docs.affectron.com/plugins/twitter-sentiment-enhancer",
                download_count=1250,
                rating=4.8,
                review_count=45,
                categories=["sentiment", "social-media"],
                tags=["twitter", "sentiment", "social"],
                verified=True,
                featured=True
            ),
            MarketplacePlugin(
                id="forex-trading-bot",
                name="Forex Trading Bot",
                version="2.1.0",
                description="Automated trading bot that executes trades based on AffectRON sentiment signals",
                author="TradingExperts",
                author_email="info@tradingexperts.dev",
                download_url="https://github.com/tradingexperts/affectron-forex-bot/raw/main/bot.zip",
                repository_url="https://github.com/tradingexperts/affectron-forex-bot",
                documentation_url="https://tradingexperts.dev/affectron-bot-docs",
                download_count=890,
                rating=4.6,
                review_count=32,
                categories=["trading", "automation"],
                tags=["forex", "trading", "bot", "automation"],
                verified=False,
                featured=True
            ),
            MarketplacePlugin(
                id="email-alerts-extended",
                name="Extended Email Alerts",
                version="1.0.5",
                description="Extended email notification system with HTML templates and custom formatting",
                author="NotificationGuru",
                author_email="guru@notifications.dev",
                download_url="https://github.com/notificationguru/affectron-email-alerts/raw/main/plugin.zip",
                repository_url="https://github.com/notificationguru/affectron-email-alerts",
                download_count=567,
                rating=4.2,
                review_count=18,
                categories=["notifications", "email"],
                tags=["email", "notifications", "templates"],
                verified=True
            ),
            MarketplacePlugin(
                id="crypto-sentiment-tracker",
                name="Crypto Sentiment Tracker",
                version="1.3.0",
                description="Specialized sentiment analysis for cryptocurrency markets",
                author="CryptoAnalytics",
                author_email="analytics@crypto.dev",
                download_url="https://github.com/cryptoanalytics/affectron-crypto/raw/main/plugin.zip",
                repository_url="https://github.com/cryptoanalytics/affectron-crypto",
                documentation_url="https://cryptoanalytics.dev/affectron-crypto",
                download_count=445,
                rating=4.7,
                review_count=23,
                categories=["cryptocurrency", "sentiment"],
                tags=["crypto", "bitcoin", "ethereum", "sentiment"],
                verified=True
            ),
            MarketplacePlugin(
                id="advanced-risk-calculator",
                name="Advanced Risk Calculator",
                version="2.0.0",
                description="Advanced risk assessment with Monte Carlo simulation and stress testing",
                author="RiskMasters",
                author_email="info@riskmasters.dev",
                download_url="https://github.com/riskmasters/affectron-risk-calc/raw/main/plugin.zip",
                repository_url="https://github.com/riskmasters/affectron-risk-calc",
                download_count=678,
                rating=4.9,
                review_count=41,
                categories=["risk", "analytics"],
                tags=["risk", "monte-carlo", "simulation", "stress-test"],
                verified=True,
                featured=True
            )
        ]

        # Filter by query and category
        filtered_plugins = []

        for plugin in mock_plugins:
            # Filter by query
            if query and query.lower() not in plugin.name.lower() and query.lower() not in plugin.description.lower():
                continue

            # Filter by category
            if category and category not in plugin.categories:
                continue

            filtered_plugins.append(plugin)

            if len(filtered_plugins) >= limit:
                break

        return filtered_plugins

    async def get_featured_plugins(self, limit: int = 6) -> List[MarketplacePlugin]:
        """Get featured plugins from marketplace."""
        cache_key = f"featured_{limit}"

        if cache_key in self._cache and datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
            return self._cache[cache_key]

        try:
            # In production, this would query featured plugins
            plugins = await self.search_plugins("", "", limit * 2)  # Get more to filter featured

            featured = [p for p in plugins if p.featured][:limit]

            # Cache results for 30 minutes
            self._cache[cache_key] = featured
            self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

            return featured

        except Exception as e:
            self.logger.error(f"Error getting featured plugins: {e}")
            return []

    async def get_plugin_details(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get detailed information about a specific plugin."""
        cache_key = f"plugin_{plugin_id}"

        if cache_key in self._cache and datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
            return self._cache[cache_key]

        try:
            # Search for the specific plugin
            plugins = await self.search_plugins(plugin_id, "", 1)

            if plugins and plugins[0].id == plugin_id:
                plugin = plugins[0]

                # Cache for 1 hour
                self._cache[cache_key] = plugin
                self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)

                return plugin

        except Exception as e:
            self.logger.error(f"Error getting plugin details for {plugin_id}: {e}")

        return None

    async def download_plugin(self, plugin_id: str, install: bool = True) -> Optional[str]:
        """Download plugin from marketplace."""
        try:
            plugin = await self.get_plugin_details(plugin_id)

            if not plugin:
                self.logger.error(f"Plugin {plugin_id} not found in marketplace")
                return None

            # Download plugin file
            async with aiohttp.ClientSession() as session:
                async with session.get(plugin.download_url) as response:
                    if response.status == 200:
                        plugin_data = await response.read()

                        if install:
                            # Save to plugins directory
                            plugin_file = f"plugins/{plugin_id}.zip"
                            with open(plugin_file, 'wb') as f:
                                f.write(plugin_data)

                            # Extract and install plugin
                            await self._install_plugin_from_zip(plugin_file, plugin)

                            return plugin_file

                        return plugin_data.decode('utf-8', errors='ignore')

        except Exception as e:
            self.logger.error(f"Error downloading plugin {plugin_id}: {e}")

        return None

    async def _install_plugin_from_zip(self, zip_path: str, plugin: MarketplacePlugin):
        """Install plugin from ZIP file."""
        try:
            import zipfile
            import tempfile
            import shutil

            # Extract ZIP
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Look for plugin files
                plugin_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.py') and file != '__init__.py':
                            plugin_files.append(os.path.join(root, file))

                if plugin_files:
                    # Copy to plugins directory
                    for plugin_file in plugin_files:
                        relative_path = os.path.relpath(plugin_file, temp_dir)
                        dest_path = os.path.join("plugins", relative_path)

                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(plugin_file, dest_path)

                    # Load the plugin
                    plugin_manager.load_plugin(plugin_files[0])

                    # Track installation
                    self.installed_plugins[plugin.id] = plugin

                    self.logger.info(f"Installed plugin: {plugin.name}")

                # Clean up ZIP file
                os.remove(zip_path)

        except Exception as e:
            self.logger.error(f"Error installing plugin from ZIP: {e}")

    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall plugin."""
        try:
            # Remove from installed plugins
            if plugin_id in self.installed_plugins:
                del self.installed_plugins[plugin_id]

            # Remove from plugin manager
            plugin_manager.uninstall_plugin(plugin_id)

            # Remove plugin files
            plugin_dir = f"plugins/{plugin_id}"
            if os.path.exists(plugin_dir):
                import shutil
                shutil.rmtree(plugin_dir)

            self.logger.info(f"Uninstalled plugin: {plugin_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error uninstalling plugin {plugin_id}: {e}")
            return False

    def get_installed_plugins(self) -> Dict[str, MarketplacePlugin]:
        """Get list of installed plugins."""
        return self.installed_plugins.copy()

    async def rate_plugin(self, plugin_id: str, rating: int, review: str = "") -> bool:
        """Rate a plugin in the marketplace."""
        try:
            # In production, this would send rating to marketplace API
            plugin = await self.get_plugin_details(plugin_id)

            if plugin:
                self.logger.info(f"Rating plugin {plugin_id}: {rating}/5 stars")

                # Update local cache if needed
                return True

        except Exception as e:
            self.logger.error(f"Error rating plugin {plugin_id}: {e}")

        return False

    def get_marketplace_categories(self) -> List[str]:
        """Get available plugin categories."""
        return [
            "sentiment",
            "trading",
            "analytics",
            "notifications",
            "data-sources",
            "visualization",
            "automation",
            "integration",
            "utility"
        ]

    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        return {
            'total_plugins': 156,  # Mock data
            'categories': self.get_marketplace_categories(),
            'featured_plugins': 12,
            'verified_plugins': 89,
            'total_downloads': 45678,
            'average_rating': 4.3,
            'last_updated': datetime.now().isoformat()
        }

    async def check_for_updates(self) -> List[Dict[str, Any]]:
        """Check for plugin updates."""
        updates = []

        for plugin_id, plugin in self.installed_plugins.items():
            try:
                # Get latest version from marketplace
                latest_plugin = await self.get_plugin_details(plugin_id)

                if latest_plugin and latest_plugin.version != plugin.version:
                    updates.append({
                        'plugin_id': plugin_id,
                        'current_version': plugin.version,
                        'latest_version': latest_plugin.version,
                        'plugin_name': plugin.name,
                        'update_url': latest_plugin.download_url
                    })

            except Exception as e:
                self.logger.error(f"Error checking updates for {plugin_id}: {e}")

        return updates


# Global marketplace instance
plugin_marketplace = PluginMarketplace()


async def initialize_marketplace():
    """Initialize plugin marketplace."""
    # Check for plugin updates
    updates = await plugin_marketplace.check_for_updates()

    if updates:
        logging.getLogger(__name__).info(f"Found {len(updates)} plugin updates available")

    logging.getLogger(__name__).info("Plugin marketplace initialized")


def get_marketplace_status():
    """Get marketplace status."""
    return {
        'initialized': True,
        'installed_plugins': len(plugin_marketplace.get_installed_plugins()),
        'marketplace_url': plugin_marketplace.marketplace_url,
        'statistics': plugin_marketplace.get_marketplace_statistics()
    }
