"""
AffectRON Python SDK.
Provides easy integration with AffectRON API for Python applications.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import requests
from urllib.parse import urljoin


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    currency: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    entities: Dict[str, List[str]]
    timestamp: datetime


@dataclass
class MarketData:
    """Market data result."""
    currency_pair: str
    rate: float
    change: float
    change_percent: float
    volume: Optional[float]
    timestamp: datetime


@dataclass
class AnalyticsInsight:
    """Analytics insight result."""
    insight_type: str
    title: str
    description: str
    confidence: float
    recommendations: List[str]
    data: Dict[str, Any]


class AffectRONSDK:
    """AffectRON SDK for Python applications."""

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize AffectRON SDK.

        Args:
            api_key: Your AffectRON API key
            base_url: Base URL for AffectRON API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # Session for HTTP requests
        self.session = None

        # Headers for authenticated requests
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _get_sync_session(self):
        """Get requests session for sync operations."""
        return requests.Session()

    # Synchronous methods (for compatibility)
    def analyze_sentiment_sync(self, text: str, language: str = "ro") -> Dict[str, Any]:
        """Analyze sentiment of text (synchronous)."""
        endpoint = f"{self.base_url}/api/v1/sentiment/analyze"

        data = {
            "text": text,
            "language": language
        }

        response = requests.post(endpoint, json=data, headers=self.headers, timeout=self.timeout)

        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Sentiment analysis failed: {response.status_code} - {response.text}")
            raise Exception(f"API request failed: {response.status_code}")

    def get_market_rates_sync(self, currencies: List[str]) -> Dict[str, Any]:
        """Get market rates (synchronous)."""
        endpoint = f"{self.base_url}/api/v1/market/rates"

        params = {"currencies": ",".join(currencies)}

        response = requests.get(endpoint, params=params, headers=self.headers, timeout=self.timeout)

        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Market rates request failed: {response.status_code} - {response.text}")
            raise Exception(f"API request failed: {response.status_code}")

    # Asynchronous methods (recommended for production)
    async def analyze_sentiment(self, text: str, language: str = "ro") -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/sentiment/analyze"

        data = {
            "text": text,
            "language": language
        }

        async with self.session.post(endpoint, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Sentiment analysis failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_sentiment_trends(self, currency: str, timeframe: str = "24h") -> Dict[str, Any]:
        """Get sentiment trends for currency."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/sentiment"
        params = {"currency": currency, "timeframe": timeframe}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Sentiment trends request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_market_rates(self, currencies: List[str]) -> Dict[str, Any]:
        """Get current market rates."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/market/rates"
        params = {"currencies": ",".join(currencies)}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Market rates request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_market_insights(self, currency: str, risk_level: str = "medium") -> Dict[str, Any]:
        """Get market insights for currency."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/analytics/insights/market"
        params = {"currency": currency, "risk_level": risk_level}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Market insights request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_risk_assessment(self, currency: str) -> Dict[str, Any]:
        """Get risk assessment for currency."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/analytics/risk"
        params = {"currency": currency}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Risk assessment request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_trend_analysis(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Get trend analysis."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/analytics/trends"
        params = {"timeframe": timeframe}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Trend analysis request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_alerts(self, limit: int = 50) -> Dict[str, Any]:
        """Get active alerts."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/alerts/active"
        params = {"limit": limit}

        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Alerts request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge"

        async with self.session.post(endpoint) as response:
            return response.status == 200

    async def get_data_sources(self) -> Dict[str, Any]:
        """Get data sources status."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/api/v1/data/sources"

        async with self.session.get(endpoint) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"Data sources request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout))

        endpoint = f"{self.base_url}/health"

        async with self.session.get(endpoint) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                self.logger.error(f"System status request failed: {response.status} - {error_text}")
                raise Exception(f"API request failed: {response.status}")

    # WebSocket support for real-time data
    def get_websocket_url(self, subscriptions: List[str] = None) -> str:
        """Get WebSocket URL for real-time data."""
        base_ws_url = self.base_url.replace('http', 'ws')

        if subscriptions:
            params = "&".join([f"subscribe={sub}" for sub in subscriptions])
            return f"{base_ws_url}/ws?{params}"
        else:
            return f"{base_ws_url}/ws"

    async def stream_real_time_data(self, subscriptions: List[str], callback: callable):
        """Stream real-time data via WebSocket."""
        try:
            import websockets

            ws_url = self.get_websocket_url(subscriptions)

            async with websockets.connect(ws_url, extra_headers=self.headers) as websocket:
                self.logger.info(f"Connected to WebSocket: {ws_url}")

                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)

                        # Call user callback with data
                        await callback(data)

                    except websockets.exceptions.ConnectionClosed:
                        self.logger.info("WebSocket connection closed")
                        break

        except ImportError:
            self.logger.error("websockets package required for real-time streaming")
            raise
        except Exception as e:
            self.logger.error(f"WebSocket streaming error: {e}")
            raise

    # Utility methods
    def format_sentiment_result(self, api_response: Dict[str, Any]) -> SentimentResult:
        """Format API response as SentimentResult."""
        return SentimentResult(
            currency=api_response.get('currency', ''),
            sentiment_label=api_response.get('sentiment', {}).get('label', 'neutral'),
            sentiment_score=api_response.get('sentiment', {}).get('score', 0.5),
            confidence=api_response.get('sentiment', {}).get('confidence', 0.0),
            entities=api_response.get('entities', {}),
            timestamp=datetime.fromisoformat(api_response.get('timestamp', datetime.now().isoformat()))
        )

    def format_market_data(self, api_response: Dict[str, Any]) -> List[MarketData]:
        """Format API response as MarketData list."""
        rates = api_response.get('rates', {})
        timestamp = api_response.get('timestamp', datetime.now().isoformat())

        market_data_list = []
        for pair, data in rates.items():
            market_data = MarketData(
                currency_pair=pair,
                rate=data.get('rate', 0),
                change=data.get('change', 0),
                change_percent=data.get('change_percent', 0),
                volume=data.get('volume'),
                timestamp=datetime.fromisoformat(timestamp)
            )
            market_data_list.append(market_data)

        return market_data_list

    def format_analytics_insight(self, api_response: Dict[str, Any]) -> AnalyticsInsight:
        """Format API response as AnalyticsInsight."""
        return AnalyticsInsight(
            insight_type=api_response.get('insight_type', 'general'),
            title=api_response.get('title', ''),
            description=api_response.get('description', ''),
            confidence=api_response.get('confidence', 0.0),
            recommendations=api_response.get('recommendations', []),
            data=api_response.get('data', {})
        )


# Convenience functions for quick usage
async def quick_sentiment_analysis(api_key: str, text: str, language: str = "ro") -> SentimentResult:
    """Quick sentiment analysis without SDK initialization."""
    sdk = AffectRONSDK(api_key)
    async with sdk:
        result = await sdk.analyze_sentiment(text, language)
        return sdk.format_sentiment_result(result)


def quick_sentiment_analysis_sync(api_key: str, text: str, language: str = "ro") -> SentimentResult:
    """Quick sentiment analysis (synchronous)."""
    sdk = AffectRONSDK(api_key)
    result = sdk.analyze_sentiment_sync(text, language)
    return sdk.format_sentiment_result(result)


async def quick_market_rates(api_key: str, currencies: List[str]) -> List[MarketData]:
    """Quick market rates lookup."""
    sdk = AffectRONSDK(api_key)
    async with sdk:
        result = await sdk.get_market_rates(currencies)
        return sdk.format_market_data(result)


def quick_market_rates_sync(api_key: str, currencies: List[str]) -> List[MarketData]:
    """Quick market rates lookup (synchronous)."""
    sdk = AffectRONSDK(api_key)
    result = sdk.get_market_rates_sync(currencies)
    return sdk.format_market_data(result)
