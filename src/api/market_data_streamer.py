"""
Real-time market data streaming service.
Integrates with external APIs and provides live market data feeds.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import asyncio_mqtt as mqtt

from .enhanced_websocket import connection_manager, MessageType, WebSocketMessage


@dataclass
class MarketDataPoint:
    """Real-time market data point."""
    currency_pair: str
    rate: float
    timestamp: datetime
    source: str
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None


@dataclass
class DataSourceConfig:
    """Configuration for external data source."""
    name: str
    url: str
    api_key: Optional[str] = None
    update_interval: int = 60  # seconds
    enabled: bool = True
    source_type: str = "rest"  # rest, websocket, mqtt


class MarketDataStreamer:
    """Real-time market data streaming service."""

    def __init__(self):
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # External data source configurations
        self._setup_default_sources()

        self.logger = logging.getLogger(__name__)

    def _setup_default_sources(self):
        """Set up default external data sources."""
        self.data_sources = {
            'bnr': DataSourceConfig(
                name='BNR Official Rates',
                url='https://www.bnr.ro/nbrfxrates.xml',
                update_interval=3600,  # 1 hour
                enabled=True,
                source_type='rest'
            ),
            'ecb': DataSourceConfig(
                name='ECB Reference Rates',
                url='https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml',
                update_interval=86400,  # 24 hours
                enabled=True,
                source_type='rest'
            ),
            'cryptocompare': DataSourceConfig(
                name='CryptoCompare',
                url='https://min-api.cryptocompare.com/data/pricemultifull',
                api_key=None,  # Free tier doesn't require key for basic data
                update_interval=300,  # 5 minutes
                enabled=True,
                source_type='rest'
            ),
            'forex_mqtt': DataSourceConfig(
                name='Forex MQTT Feed',
                url='mqtt://test.mosquitto.org:1883',
                update_interval=30,
                enabled=False,  # Disabled by default
                source_type='mqtt'
            )
        }

    async def start(self):
        """Start the market data streaming service."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting market data streamer...")

        # Start all enabled data sources
        for source_name, config in self.data_sources.items():
            if config.enabled:
                task = asyncio.create_task(
                    self._run_data_source(source_name, config)
                )
                self.tasks.append(task)

        # Start MQTT client if needed
        await self._start_mqtt_client()

        self.logger.info("Market data streamer started")

    async def stop(self):
        """Stop the market data streaming service."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping market data streamer...")

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks.clear()
        self.logger.info("Market data streamer stopped")

    async def _run_data_source(self, source_name: str, config: DataSourceConfig):
        """Run a specific data source."""
        self.logger.info(f"Starting data source: {source_name}")

        while self.running:
            try:
                if config.source_type == 'rest':
                    await self._fetch_rest_data(source_name, config)
                elif config.source_type == 'mqtt':
                    await self._fetch_mqtt_data(source_name, config)

                await asyncio.sleep(config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in data source {source_name}: {e}")
                await asyncio.sleep(config.update_interval)

    async def _fetch_rest_data(self, source_name: str, config: DataSourceConfig):
        """Fetch data from REST API."""
        try:
            headers = {}
            if config.api_key:
                headers['Authorization'] = f'Bearer {config.api_key}'

            async with aiohttp.ClientSession() as session:
                async with session.get(config.url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.text()

                        # Parse data based on source
                        if source_name == 'bnr':
                            market_data = self._parse_bnr_data(data)
                        elif source_name == 'ecb':
                            market_data = self._parse_ecb_data(data)
                        elif source_name == 'cryptocompare':
                            market_data = self._parse_cryptocompare_data(data)
                        else:
                            return

                        # Process and broadcast data
                        await self._process_market_data(market_data, source_name)

                    else:
                        self.logger.error(f"HTTP {response.status} from {source_name}")

        except Exception as e:
            self.logger.error(f"Error fetching REST data from {source_name}: {e}")

    async def _fetch_mqtt_data(self, source_name: str, config: DataSourceConfig):
        """Fetch data from MQTT broker."""
        try:
            async with mqtt.Client(config.url) as client:
                await client.subscribe("forex/#")

                async for message in client.messages:
                    topic = message.topic
                    payload = message.payload.decode()

                    # Parse MQTT message
                    market_data = self._parse_mqtt_data(topic, payload)

                    if market_data:
                        await self._process_market_data([market_data], source_name)

        except Exception as e:
            self.logger.error(f"Error fetching MQTT data from {source_name}: {e}")

    def _parse_bnr_data(self, xml_data: str) -> List[MarketDataPoint]:
        """Parse BNR XML data."""
        import xml.etree.ElementTree as ET

        market_data = []

        try:
            root = ET.fromstring(xml_data)

            # Find the date
            date_elem = root.find('.//Date')
            if date_elem is not None and date_elem.text:
                date_str = date_elem.text
                # Parse date (format: "01/01/2024")
                try:
                    timestamp = datetime.strptime(date_str, "%d/%m/%Y")
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            # Find all Rate elements
            for rate_elem in root.findall('.//Rate'):
                currency = rate_elem.get('currency')
                if currency and currency in ['EUR', 'USD']:
                    try:
                        rate = float(rate_elem.text)

                        market_data.append(MarketDataPoint(
                            currency_pair=f"RON/{currency}",
                            rate=rate,
                            timestamp=timestamp,
                            source='BNR',
                            volume=None,
                            bid=rate * 0.999,  # Small spread
                            ask=rate * 1.001
                        ))
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error parsing BNR rate for {currency}: {e}")

        except ET.ParseError as e:
            self.logger.error(f"Error parsing BNR XML: {e}")

        return market_data

    def _parse_ecb_data(self, xml_data: str) -> List[MarketDataPoint]:
        """Parse ECB XML data."""
        import xml.etree.ElementTree as ET

        market_data = []

        try:
            root = ET.fromstring(xml_data)

            # Find the date
            date_elem = root.find('.//time')
            if date_elem is not None:
                timestamp = datetime.now()  # ECB doesn't provide date in this format
            else:
                timestamp = datetime.now()

            # Find all Cube elements with currency and rate
            for cube_elem in root.findall('.//Cube[@currency][@rate]'):
                currency = cube_elem.get('currency')
                if currency:
                    try:
                        rate = float(cube_elem.get('rate'))

                        market_data.append(MarketDataPoint(
                            currency_pair=f"EUR/{currency}",
                            rate=rate,
                            timestamp=timestamp,
                            source='ECB',
                            volume=None,
                            bid=rate * 0.999,
                            ask=rate * 1.001
                        ))
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error parsing ECB rate for {currency}: {e}")

        except ET.ParseError as e:
            self.logger.error(f"Error parsing ECB XML: {e}")

        return market_data

    def _parse_cryptocompare_data(self, json_data: str) -> List[MarketDataPoint]:
        """Parse CryptoCompare API data."""
        market_data = []

        try:
            data = json.loads(json_data)

            if 'RAW' in data:
                raw_data = data['RAW']

                for base_currency, quotes in raw_data.items():
                    if base_currency in ['BTC', 'ETH']:
                        for quote_currency, quote_data in quotes.items():
                            if quote_currency == 'USD':
                                try:
                                    price = quote_data.get('PRICE', 0)
                                    volume = quote_data.get('VOLUME24HOUR', 0)
                                    change_pct = quote_data.get('CHANGEPCT24HOUR', 0)

                                    market_data.append(MarketDataPoint(
                                        currency_pair=f"{base_currency}/USD",
                                        rate=price,
                                        timestamp=datetime.now(),
                                        source='CryptoCompare',
                                        volume=volume,
                                        change=quote_data.get('CHANGE24HOUR', 0),
                                        change_percent=change_pct
                                    ))
                                except (ValueError, TypeError, KeyError) as e:
                                    self.logger.warning(f"Error parsing CryptoCompare data for {base_currency}: {e}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing CryptoCompare JSON: {e}")

        return market_data

    def _parse_mqtt_data(self, topic: str, payload: str) -> Optional[MarketDataPoint]:
        """Parse MQTT message data."""
        try:
            data = json.loads(payload)

            currency_pair = data.get('pair')
            rate = data.get('rate')
            timestamp = data.get('timestamp')

            if currency_pair and rate:
                return MarketDataPoint(
                    currency_pair=currency_pair,
                    rate=float(rate),
                    timestamp=datetime.fromisoformat(timestamp) if timestamp else datetime.now(),
                    source='MQTT',
                    volume=data.get('volume'),
                    bid=data.get('bid'),
                    ask=data.get('ask')
                )

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing MQTT data: {e}")

        return None

    async def _process_market_data(self, market_data: List[MarketDataPoint], source_name: str):
        """Process and broadcast market data."""
        for data_point in market_data:
            # Create message for WebSocket
            message_data = {
                'currency_pair': data_point.currency_pair,
                'rate': data_point.rate,
                'timestamp': data_point.timestamp.isoformat(),
                'source': data_point.source,
                'volume': data_point.volume,
                'bid': data_point.bid,
                'ask': data_point.ask,
                'change': data_point.change,
                'change_percent': data_point.change_percent
            }

            # Send via WebSocket
            await connection_manager.send_market_data(message_data)

            # Notify subscribers
            await self._notify_subscribers(source_name, data_point)

            # Store in database (if needed)
            await self._store_market_data(data_point)

            self.logger.debug(f"Broadcast market data: {data_point.currency_pair} = {data_point.rate}")

    async def _notify_subscribers(self, source_name: str, data_point: MarketDataPoint):
        """Notify registered subscribers."""
        if source_name in self.subscribers:
            for callback in self.subscribers[source_name]:
                try:
                    await callback(data_point)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {e}")

    async def _store_market_data(self, data_point: MarketDataPoint):
        """Store market data in database."""
        # This would integrate with the database models
        # For now, just log the data
        self.logger.debug(f"Would store market data: {data_point}")

    def subscribe_to_source(self, source_name: str, callback: Callable):
        """Subscribe to updates from a specific data source."""
        if source_name not in self.subscribers:
            self.subscribers[source_name] = set()

        self.subscribers[source_name].add(callback)

    def unsubscribe_from_source(self, source_name: str, callback: Callable):
        """Unsubscribe from updates from a specific data source."""
        if source_name in self.subscribers:
            self.subscribers[source_name].discard(callback)

    async def _start_mqtt_client(self):
        """Start MQTT client for real-time data."""
        # This would be implemented for production use
        # For now, it's a placeholder
        pass

    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}

        for name, config in self.data_sources.items():
            status[name] = {
                'enabled': config.enabled,
                'type': config.source_type,
                'last_update': None,  # Would track actual last update time
                'update_interval': config.update_interval,
                'subscribers': len(self.subscribers.get(name, []))
            }

        return status

    def add_data_source(self, name: str, config: DataSourceConfig):
        """Add a new data source."""
        self.data_sources[name] = config
        self.logger.info(f"Added data source: {name}")

    def remove_data_source(self, name: str):
        """Remove a data source."""
        if name in self.data_sources:
            del self.data_sources[name]

            if name in self.subscribers:
                del self.subscribers[name]

            self.logger.info(f"Removed data source: {name}")

    def enable_source(self, name: str):
        """Enable a data source."""
        if name in self.data_sources:
            self.data_sources[name].enabled = True
            self.logger.info(f"Enabled data source: {name}")

    def disable_source(self, name: str):
        """Disable a data source."""
        if name in self.data_sources:
            self.data_sources[name].enabled = False
            self.logger.info(f"Disabled data source: {name}")


# Global market data streamer instance
market_streamer = MarketDataStreamer()


async def start_market_data_streaming():
    """Start market data streaming service."""
    await market_streamer.start()


async def stop_market_data_streaming():
    """Stop market data streaming service."""
    await market_streamer.stop()


def get_market_data_status():
    """Get current market data streaming status."""
    return {
        'running': market_streamer.running,
        'sources': market_streamer.get_source_status(),
        'active_tasks': len(market_streamer.tasks)
    }
