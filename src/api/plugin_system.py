"""
Plugin system for AffectRON.
Allows third-party extensions and custom functionality integration.
"""

import asyncio
import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Type
from pathlib import Path
from dataclasses import dataclass
import sys

from ..api.main import app
from ..api.enhanced_websocket import connection_manager, MessageType, WebSocketMessage


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    author_email: str
    license: str = "MIT"

    # Plugin capabilities
    hooks: List[str] = None
    dependencies: List[str] = None
    permissions: List[str] = None

    # Plugin status
    enabled: bool = True
    priority: int = 100  # Lower number = higher priority

    def __post_init__(self):
        if self.hooks is None:
            self.hooks = []
        if self.dependencies is None:
            self.dependencies = []
        if self.permissions is None:
            self.permissions = []


class PluginHook(ABC):
    """Base class for plugin hooks."""

    @abstractmethod
    def get_name(self) -> str:
        """Return hook name."""
        pass

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute hook logic."""
        pass


class SentimentAnalysisHook(PluginHook):
    """Hook for custom sentiment analysis."""

    def get_name(self) -> str:
        return "sentiment_analysis"

    async def execute(self, text: str, language: str = "ro") -> Dict[str, Any]:
        """Execute custom sentiment analysis."""
        # Default implementation - plugins should override
        return {
            'sentiment': {'label': 'neutral', 'score': 0.5, 'confidence': 0.0},
            'entities': {},
            'custom_data': {}
        }


class MarketDataHook(PluginHook):
    """Hook for custom market data processing."""

    def get_name(self) -> str:
        return "market_data_processing"

    async def execute(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data."""
        # Default implementation - plugins should override
        return market_data


class AlertHook(PluginHook):
    """Hook for custom alert generation."""

    def get_name(self) -> str:
        return "alert_generation"

    async def execute(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate custom alert if conditions met."""
        # Default implementation - plugins should override
        return None


class PluginManager:
    """Manages plugin loading, registration, and execution."""

    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_hooks: Dict[str, List[PluginHook]] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}

        # Ensure plugins directory exists
        self.plugins_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def load_plugin(self, plugin_path: str) -> bool:
        """Load a plugin from file path."""
        try:
            plugin_name = Path(plugin_path).stem

            # Add to Python path
            plugin_dir = str(Path(plugin_path).parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            # Import plugin module
            module = importlib.import_module(plugin_name)

            # Check if plugin has required attributes
            if not hasattr(module, 'PLUGIN_INFO'):
                self.logger.error(f"Plugin {plugin_name} missing PLUGIN_INFO")
                return False

            if not hasattr(module, 'Plugin'):
                self.logger.error(f"Plugin {plugin_name} missing Plugin class")
                return False

            plugin_info = module.PLUGIN_INFO
            plugin_class = module.Plugin

            # Validate plugin info
            if not isinstance(plugin_info, PluginInfo):
                self.logger.error(f"Plugin {plugin_name} PLUGIN_INFO not a PluginInfo instance")
                return False

            # Instantiate plugin
            plugin_instance = plugin_class()

            # Register plugin
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_info[plugin_name] = plugin_info

            # Register hooks
            self._register_plugin_hooks(plugin_instance, plugin_name)

            self.logger.info(f"Loaded plugin: {plugin_name} v{plugin_info.version}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_path}: {e}")
            return False

    def _register_plugin_hooks(self, plugin_instance: Any, plugin_name: str):
        """Register plugin hooks."""
        # Check for hook methods
        hook_methods = {}

        for attr_name in dir(plugin_instance):
            attr = getattr(plugin_instance, attr_name)

            if (isinstance(attr, type) and
                issubclass(attr, PluginHook) and
                attr != PluginHook):

                hook_methods[attr_name] = attr

        # Register hooks by name
        for hook_name, hook_class in hook_methods.items():
            if hook_name not in self.plugin_hooks:
                self.plugin_hooks[hook_name] = []

            # Create hook instance
            hook_instance = hook_class()
            self.plugin_hooks[hook_name].append(hook_instance)

    async def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all plugins for a specific hook."""
        if hook_name not in self.plugin_hooks:
            return []

        results = []

        # Sort by priority
        hooks = sorted(self.plugin_hooks[hook_name],
                      key=lambda h: self.plugin_info.get(h.__class__.__module__, PluginInfo("", "1.0.0", "")).priority)

        for hook in hooks:
            try:
                result = await hook.execute(*args, **kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing hook {hook_name}: {e}")

        return results

    def get_loaded_plugins(self) -> Dict[str, PluginInfo]:
        """Get information about loaded plugins."""
        return self.plugin_info.copy()

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = True
            self.logger.info(f"Enabled plugin: {plugin_name}")
            return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = False
            self.logger.info(f"Disabled plugin: {plugin_name}")
            return True
        return False

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]

        if plugin_name in self.plugin_info:
            del self.plugin_info[plugin_name]

        # Remove from hooks
        for hook_list in self.plugin_hooks.values():
            hook_list[:] = [h for h in hook_list if h.__class__.__module__ != plugin_name]

        self.logger.info(f"Uninstalled plugin: {plugin_name}")
        return True

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        enabled_plugins = sum(1 for info in self.plugin_info.values() if info.enabled)

        return {
            'total_plugins': len(self.loaded_plugins),
            'enabled_plugins': enabled_plugins,
            'disabled_plugins': len(self.plugin_info) - enabled_plugins,
            'available_hooks': list(self.plugin_hooks.keys()),
            'hooks_count': {name: len(hooks) for name, hooks in self.plugin_hooks.items()}
        }


class BasePlugin:
    """Base class for AffectRON plugins."""

    PLUGIN_INFO = PluginInfo(
        name="Base Plugin",
        version="1.0.0",
        description="Base plugin class",
        author="AffectRON Team",
        author_email="plugins@affectron.com"
    )

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")


# Example plugins
class TwitterSentimentPlugin(BasePlugin):
    """Plugin for enhanced Twitter sentiment analysis."""

    PLUGIN_INFO = PluginInfo(
        name="Twitter Sentiment Enhancer",
        version="1.0.0",
        description="Enhanced sentiment analysis for Twitter data",
        author="AffectRON Team",
        author_email="plugins@affectron.com",
        hooks=["sentiment_analysis"],
        permissions=["read_twitter_data", "write_sentiment_results"]
    )

    class TwitterSentimentHook(SentimentAnalysisHook):
        """Hook for Twitter-specific sentiment analysis."""

        def get_name(self) -> str:
            return "twitter_sentiment"

        async def execute(self, text: str, language: str = "ro") -> Dict[str, Any]:
            """Enhanced sentiment analysis for Twitter content."""
            # Twitter-specific preprocessing
            clean_text = self._preprocess_twitter_text(text)

            # Enhanced sentiment analysis
            base_result = await super().execute(clean_text, language)

            # Add Twitter-specific insights
            base_result['twitter_insights'] = {
                'hashtags': self._extract_hashtags(text),
                'mentions': self._extract_mentions(text),
                'is_retweet': self._is_retweet(text),
                'engagement_potential': self._calculate_engagement_potential(clean_text)
            }

            return base_result

        def _preprocess_twitter_text(self, text: str) -> str:
            """Preprocess Twitter-specific content."""
            import re

            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

            # Remove mentions (keep for entity extraction)
            # text = re.sub(r'@\w+', '', text)

            # Remove extra whitespace
            text = ' '.join(text.split())

            return text

        def _extract_hashtags(self, text: str) -> List[str]:
            """Extract hashtags from Twitter text."""
            import re
            return re.findall(r'#\w+', text)

        def _extract_mentions(self, text: str) -> List[str]:
            """Extract mentions from Twitter text."""
            import re
            return re.findall(r'@\w+', text)

        def _is_retweet(self, text: str) -> bool:
            """Check if text is a retweet."""
            return 'rt @' in text.lower() or text.lower().startswith('rt')

        def _calculate_engagement_potential(self, text: str) -> float:
            """Calculate engagement potential score."""
            score = 0.0

            # Hashtags increase engagement
            hashtags = self._extract_hashtags(text)
            score += len(hashtags) * 0.1

            # Mentions increase engagement
            mentions = self._extract_mentions(text)
            score += len(mentions) * 0.15

            # Question marks increase engagement
            score += text.count('?') * 0.2

            # Exclamation marks show enthusiasm
            score += text.count('!') * 0.05

            return min(score, 1.0)

    def __init__(self):
        super().__init__()
        self.twitter_hook = self.TwitterSentimentHook()


class TradingSignalPlugin(BasePlugin):
    """Plugin for custom trading signal generation."""

    PLUGIN_INFO = PluginInfo(
        name="Trading Signal Generator",
        version="1.0.0",
        description="Custom trading signal generation based on sentiment",
        author="AffectRON Team",
        author_email="plugins@affectron.com",
        hooks=["trading_signal"],
        permissions=["read_sentiment_data", "write_trading_signals"]
    )

    class TradingSignalHook(PluginHook):
        """Hook for trading signal generation."""

        def get_name(self) -> str:
            return "trading_signal"

        async def execute(self, sentiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Generate trading signal based on sentiment."""
            currency = sentiment_data.get('currency', '')
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            confidence = sentiment_data.get('confidence', 0)

            # Custom signal generation logic
            if confidence > 0.7:
                if sentiment_score > 0.6:
                    signal = {
                        'type': 'BUY',
                        'symbol': currency,
                        'confidence': confidence,
                        'reason': 'Strong positive sentiment',
                        'stop_loss': sentiment_data.get('current_price', 0) * 0.98,
                        'take_profit': sentiment_data.get('current_price', 0) * 1.05
                    }
                    return signal

                elif sentiment_score < -0.6:
                    signal = {
                        'type': 'SELL',
                        'symbol': currency,
                        'confidence': confidence,
                        'reason': 'Strong negative sentiment',
                        'stop_loss': sentiment_data.get('current_price', 0) * 1.02,
                        'take_profit': sentiment_data.get('current_price', 0) * 0.95
                    }
                    return signal

            return None

    def __init__(self):
        super().__init__()
        self.trading_hook = self.TradingSignalHook()


# Global plugin manager
plugin_manager = PluginManager()


async def initialize_plugin_system():
    """Initialize plugin system and load available plugins."""
    # Load plugins from plugins directory
    plugins_dir = Path("plugins")

    if plugins_dir.exists():
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name != "__init__.py":
                plugin_manager.load_plugin(str(plugin_file))

    # Register default plugins
    twitter_plugin = TwitterSentimentPlugin()
    plugin_manager.loaded_plugins['twitter_sentiment'] = twitter_plugin
    plugin_manager.plugin_info['twitter_sentiment'] = twitter_plugin.PLUGIN_INFO

    trading_plugin = TradingSignalPlugin()
    plugin_manager.loaded_plugins['trading_signals'] = trading_plugin
    plugin_manager.plugin_info['trading_signals'] = trading_plugin.PLUGIN_INFO

    logging.getLogger(__name__).info(f"Plugin system initialized with {len(plugin_manager.loaded_plugins)} plugins")


def get_plugin_system_status():
    """Get plugin system status."""
    return {
        'initialized': True,
        'loaded_plugins': len(plugin_manager.loaded_plugins),
        'enabled_plugins': len([p for p in plugin_manager.plugin_info.values() if p.enabled]),
        'available_hooks': list(plugin_manager.plugin_hooks.keys()),
        'statistics': plugin_manager.get_plugin_statistics()
    }
