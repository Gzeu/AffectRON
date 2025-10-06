"""
Tests for market data streaming functionality.
Tests real-time data fetching, parsing, and broadcasting.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.api.market_data_streamer import (
    MarketDataStreamer, MarketDataPoint, DataSourceConfig
)


class TestMarketDataStreamer:
    """Test MarketDataStreamer functionality."""

    @pytest.fixture
    def streamer(self):
        """Create MarketDataStreamer instance for testing."""
        return MarketDataStreamer()

    def test_initialization(self, streamer):
        """Test streamer initialization."""
        assert streamer.running == False
        assert len(streamer.tasks) == 0
        assert len(streamer.data_sources) > 0
        assert 'bnr' in streamer.data_sources
        assert 'ecb' in streamer.data_sources

    def test_data_source_configuration(self, streamer):
        """Test data source configuration."""
        bnr_config = streamer.data_sources['bnr']

        assert bnr_config.name == 'BNR Official Rates'
        assert bnr_config.enabled == True
        assert bnr_config.source_type == 'rest'
        assert bnr_config.update_interval == 3600

    @pytest.mark.asyncio
    async def test_start_stop_service(self, streamer):
        """Test starting and stopping the streaming service."""
        # Start service
        await streamer.start()

        assert streamer.running == True
        assert len(streamer.tasks) > 0

        # Stop service
        await streamer.stop()

        assert streamer.running == False
        assert len(streamer.tasks) == 0

    def test_parse_bnr_data_valid_xml(self, streamer):
        """Test parsing valid BNR XML data."""
        xml_data = """<?xml version="1.0" encoding="utf-8"?>
<DataSet>
    <Body>
        <Cube>
            <Rate currency="EUR" multiplier="1">4.9750</Rate>
            <Rate currency="USD" multiplier="1">4.5800</Rate>
        </Cube>
    </Body>
</DataSet>"""

        market_data = streamer._parse_bnr_data(xml_data)

        assert len(market_data) == 2

        # Check EUR data
        eur_data = next((d for d in market_data if d.currency_pair == 'RON/EUR'), None)
        assert eur_data is not None
        assert eur_data.rate == 4.9750
        assert eur_data.source == 'BNR'

        # Check USD data
        usd_data = next((d for d in market_data if d.currency_pair == 'RON/USD'), None)
        assert usd_data is not None
        assert usd_data.rate == 4.5800

    def test_parse_bnr_data_invalid_xml(self, streamer):
        """Test parsing invalid BNR XML data."""
        xml_data = "<invalid>xml<data>"

        market_data = streamer._parse_bnr_data(xml_data)

        assert len(market_data) == 0

    def test_parse_ecb_data_valid_xml(self, streamer):
        """Test parsing valid ECB XML data."""
        xml_data = """<?xml version="1.0" encoding="utf-8"?>
<gesmes:Envelope>
    <Cube>
        <Cube time="2024-01-01">
            <Cube currency="USD" rate="1.0850"/>
            <Cube currency="GBP" rate="0.8750"/>
        </Cube>
    </Cube>
</gesmes:Envelope>"""

        market_data = streamer._parse_ecb_data(xml_data)

        assert len(market_data) == 2

        # Check USD data
        usd_data = next((d for d in market_data if d.currency_pair == 'EUR/USD'), None)
        assert usd_data is not None
        assert usd_data.rate == 1.0850
        assert usd_data.source == 'ECB'

    def test_parse_cryptocompare_data_valid_json(self, streamer):
        """Test parsing valid CryptoCompare JSON data."""
        json_data = """{
            "RAW": {
                "BTC": {
                    "USD": {
                        "PRICE": 45000.0,
                        "VOLUME24HOUR": 1500000000.0,
                        "CHANGEPCT24HOUR": 2.5
                    }
                },
                "ETH": {
                    "USD": {
                        "PRICE": 2800.0,
                        "VOLUME24HOUR": 800000000.0,
                        "CHANGEPCT24HOUR": -1.2
                    }
                }
            }
        }"""

        market_data = streamer._parse_cryptocompare_data(json_data)

        assert len(market_data) == 2

        # Check BTC data
        btc_data = next((d for d in market_data if d.currency_pair == 'BTC/USD'), None)
        assert btc_data is not None
        assert btc_data.rate == 45000.0
        assert btc_data.volume == 1500000000.0
        assert btc_data.change_percent == 2.5

    def test_parse_mqtt_data_valid_message(self, streamer):
        """Test parsing valid MQTT message."""
        topic = "forex/EUR/USD"
        payload = json.dumps({
            "pair": "EUR/USD",
            "rate": 1.0850,
            "timestamp": "2024-01-01T12:00:00",
            "volume": 1000000.0
        })

        market_data = streamer._parse_mqtt_data(topic, payload)

        assert market_data is not None
        assert market_data.currency_pair == 'EUR/USD'
        assert market_data.rate == 1.0850
        assert market_data.volume == 1000000.0

    def test_parse_mqtt_data_invalid_message(self, streamer):
        """Test parsing invalid MQTT message."""
        topic = "forex/EUR/USD"
        payload = "invalid json"

        market_data = streamer._parse_mqtt_data(topic, payload)

        assert market_data is None

    def test_add_remove_data_source(self, streamer):
        """Test adding and removing data sources."""
        # Add new source
        new_config = DataSourceConfig(
            name="Test Source",
            url="https://test.com/api",
            update_interval=300,
            enabled=True
        )

        streamer.add_data_source("test_source", new_config)

        assert "test_source" in streamer.data_sources
        assert streamer.data_sources["test_source"].name == "Test Source"

        # Remove source
        streamer.remove_data_source("test_source")

        assert "test_source" not in streamer.data_sources

    def test_enable_disable_source(self, streamer):
        """Test enabling and disabling data sources."""
        # Disable BNR source
        streamer.disable_source("bnr")

        assert streamer.data_sources["bnr"].enabled == False

        # Enable BNR source
        streamer.enable_source("bnr")

        assert streamer.data_sources["bnr"].enabled == True

    def test_subscribe_unsubscribe_to_source(self, streamer):
        """Test subscribing and unsubscribing to data sources."""
        async def test_callback(data_point):
            pass

        # Subscribe to BNR source
        streamer.subscribe_to_source("bnr", test_callback)

        assert "bnr" in streamer.subscribers
        assert test_callback in streamer.subscribers["bnr"]

        # Unsubscribe from BNR source
        streamer.unsubscribe_from_source("bnr", test_callback)

        assert test_callback not in streamer.subscribers["bnr"]

    def test_get_source_status(self, streamer):
        """Test getting source status."""
        status = streamer.get_source_status()

        assert "bnr" in status
        assert "ecb" in status
        assert status["bnr"]["enabled"] == True
        assert status["ecb"]["enabled"] == True
        assert status["bnr"]["type"] == "rest"
        assert status["ecb"]["type"] == "rest"

    @pytest.mark.asyncio
    async def test_process_market_data_with_subscribers(self, streamer):
        """Test processing market data with subscribers."""
        callback_results = []

        async def test_callback(data_point):
            callback_results.append(data_point)

        # Subscribe to BNR source
        streamer.subscribe_to_source("bnr", test_callback)

        # Create test market data
        market_data = [
            MarketDataPoint(
                currency_pair="RON/EUR",
                rate=4.9750,
                timestamp=datetime.now(),
                source="BNR"
            )
        ]

        # Process data (without WebSocket broadcasting for test)
        with patch.object(streamer, '_notify_subscribers') as mock_notify, \
             patch.object(streamer, '_store_market_data') as mock_store:

            await streamer._process_market_data(market_data, "bnr")

            # Check that subscribers were notified
            mock_notify.assert_called_once()
            assert len(callback_results) == 1
            assert callback_results[0].rate == 4.9750

            # Check that data would be stored
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_data_source_rest(self, streamer):
        """Test running REST data source."""
        # Mock aiohttp session
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<xml>test</xml>")
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            # Mock XML parsing
            with patch.object(streamer, '_parse_bnr_data', return_value=[]) as mock_parse:
                # Run source briefly
                task = asyncio.create_task(streamer._run_data_source("bnr", streamer.data_sources["bnr"]))

                # Let it run for a short time
                await asyncio.sleep(0.1)

                # Stop the task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Check that HTTP request was made
                mock_session.assert_called()

    def test_market_data_point_creation(self):
        """Test MarketDataPoint creation."""
        point = MarketDataPoint(
            currency_pair="EUR/RON",
            rate=4.9750,
            timestamp=datetime.now(),
            source="BNR",
            volume=1000000.0,
            bid=4.9740,
            ask=4.9760,
            change=0.002,
            change_percent=0.04
        )

        assert point.currency_pair == "EUR/RON"
        assert point.rate == 4.9750
        assert point.source == "BNR"
        assert point.volume == 1000000.0
        assert point.bid == 4.9740
        assert point.ask == 4.9760


class TestIntegration:
    """Integration tests for market data streaming."""

    @pytest.mark.asyncio
    async def test_full_streaming_lifecycle(self, streamer):
        """Test complete streaming service lifecycle."""
        # Test callback
        received_data = []

        async def test_callback(data_point):
            received_data.append(data_point)

        # Subscribe to BNR
        streamer.subscribe_to_source("bnr", test_callback)

        # Start service
        await streamer.start()

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop service
        await streamer.stop()

        # Check that service started and stopped correctly
        assert streamer.running == False

    def test_multiple_data_sources(self, streamer):
        """Test configuration of multiple data sources."""
        # Check that all default sources are configured
        expected_sources = ['bnr', 'ecb', 'cryptocompare', 'forex_mqtt']

        for source in expected_sources:
            assert source in streamer.data_sources

        # Check source types
        assert streamer.data_sources['bnr'].source_type == 'rest'
        assert streamer.data_sources['ecb'].source_type == 'rest'
        assert streamer.data_sources['cryptocompare'].source_type == 'rest'
        assert streamer.data_sources['forex_mqtt'].source_type == 'mqtt'

    @pytest.mark.asyncio
    async def test_error_handling_in_data_processing(self, streamer):
        """Test error handling during data processing."""
        # Create data source that will fail
        failing_config = DataSourceConfig(
            name="Failing Source",
            url="https://invalid-url-that-will-fail.com",
            update_interval=1,
            enabled=True
        )

        streamer.add_data_source("failing", failing_config)

        # Subscribe to failing source
        error_count = 0

        async def error_callback(data_point):
            nonlocal error_count
            error_count += 1

        streamer.subscribe_to_source("failing", error_callback)

        # Run source briefly - should handle errors gracefully
        task = asyncio.create_task(streamer._run_data_source("failing", failing_config))

        # Let it try to run and fail
        await asyncio.sleep(0.5)

        # Stop the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have logged errors but not crashed
        # (In a real test, we'd check logs)

    def test_data_source_config_creation(self):
        """Test DataSourceConfig creation."""
        config = DataSourceConfig(
            name="Test API",
            url="https://api.test.com/data",
            api_key="test_key_123",
            update_interval=300,
            enabled=True,
            source_type="rest"
        )

        assert config.name == "Test API"
        assert config.url == "https://api.test.com/data"
        assert config.api_key == "test_key_123"
        assert config.update_interval == 300
        assert config.enabled == True
        assert config.source_type == "rest"
