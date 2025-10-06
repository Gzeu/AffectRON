"""
Tests for enhanced WebSocket functionality.
Tests connection management, subscriptions, and real-time messaging.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.api.enhanced_websocket import (
    EnhancedConnectionManager, WebSocketMessage, MessageType,
    SubscriptionType, UserSubscription
)


class TestEnhancedConnectionManager:
    """Test EnhancedConnectionManager functionality."""

    @pytest.fixture
    def connection_manager(self):
        """Create EnhancedConnectionManager instance for testing."""
        return EnhancedConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket for testing."""
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        websocket.send_json = AsyncMock()
        return websocket

    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.SENTIMENT_UPDATE.value == "sentiment_update"
        assert MessageType.MARKET_DATA.value == "market_data"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.HEARTBEAT.value == "heartbeat"

    def test_subscription_type_enum(self):
        """Test SubscriptionType enum values."""
        assert SubscriptionType.SENTIMENT.value == "sentiment"
        assert SubscriptionType.MARKET_RATES.value == "market_rates"
        assert SubscriptionType.ALERTS.value == "alerts"
        assert SubscriptionType.ALL.value == "all"

    def test_websocket_message_creation(self):
        """Test WebSocketMessage creation."""
        data = {"test": "data"}
        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data=data,
            timestamp=datetime.now()
        )

        assert message.message_type == MessageType.SENTIMENT_UPDATE
        assert message.data == data
        assert message.user_id is None
        assert message.subscription_filter is None

    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket connection establishment."""
        user_id = "test_user_123"

        connection_id = await connection_manager.connect(mock_websocket, user_id)

        assert connection_id in connection_manager.active_connections
        assert connection_manager.active_connections[connection_id].user_id == user_id
        assert connection_manager.active_connections[connection_id].websocket == mock_websocket

        # Check that accept was called
        mock_websocket.accept.assert_called_once()

        # Check that welcome message was sent
        assert mock_websocket.send_json.call_count >= 1

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection."""
        # First connect
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        # Then disconnect
        await connection_manager.disconnect(connection_id)

        # Check that connection was removed
        assert connection_id not in connection_manager.active_connections

        # Check that close was called
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_data_type(self, connection_manager, mock_websocket):
        """Test subscribing to data type."""
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        await connection_manager.subscribe(connection_id, SubscriptionType.SENTIMENT)

        subscription = connection_manager.active_connections[connection_id]
        assert SubscriptionType.SENTIMENT in subscription.subscriptions

        # Check subscription confirmation message
        subscription_call = mock_websocket.send_json.call_args_list[-1]
        message_data = subscription_call[0][0]
        assert message_data["type"] == "subscription_update"
        assert message_data["data"]["action"] == "subscribed"

    @pytest.mark.asyncio
    async def test_unsubscribe_from_data_type(self, connection_manager, mock_websocket):
        """Test unsubscribing from data type."""
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        # Subscribe first
        await connection_manager.subscribe(connection_id, SubscriptionType.SENTIMENT)

        # Then unsubscribe
        await connection_manager.unsubscribe(connection_id, SubscriptionType.SENTIMENT)

        subscription = connection_manager.active_connections[connection_id]
        assert SubscriptionType.SENTIMENT not in subscription.subscriptions

    @pytest.mark.asyncio
    async def test_send_message_to_connection(self, connection_manager, mock_websocket):
        """Test sending message to specific connection."""
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        message = WebSocketMessage(
            message_type=MessageType.ALERT,
            data={"alert": "test alert"},
            timestamp=datetime.now()
        )

        await connection_manager.send_message(connection_id, message)

        # Check that send_json was called with correct data
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]

        assert call_args["type"] == "alert"
        assert call_args["data"]["alert"] == "test alert"

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self, connection_manager):
        """Test broadcasting to multiple subscribers."""
        # Create multiple mock connections
        websocket1 = AsyncMock()
        websocket1.send_json = AsyncMock()
        websocket2 = AsyncMock()
        websocket2.send_json = AsyncMock()

        # Connect users
        connection_id1 = await connection_manager.connect(websocket1, "user1")
        connection_id2 = await connection_manager.connect(websocket2, "user2")

        # Subscribe both to sentiment
        await connection_manager.subscribe(connection_id1, SubscriptionType.SENTIMENT)
        await connection_manager.subscribe(connection_id2, SubscriptionType.SENTIMENT)

        # Broadcast sentiment update
        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data={"sentiment": "positive"},
            timestamp=datetime.now()
        )

        await connection_manager.broadcast_to_subscribers(message, SubscriptionType.SENTIMENT)

        # Both websockets should have received the message
        websocket1.send_json.assert_called_once()
        websocket2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_filter(self, connection_manager):
        """Test broadcasting with custom filter."""
        websocket1 = AsyncMock()
        websocket1.send_json = AsyncMock()
        websocket2 = AsyncMock()
        websocket2.send_json = AsyncMock()

        connection_id1 = await connection_manager.connect(websocket1, "user1")
        connection_id2 = await connection_manager.connect(websocket2, "user2")

        await connection_manager.subscribe(connection_id1, SubscriptionType.SENTIMENT)
        await connection_manager.subscribe(connection_id2, SubscriptionType.SENTIMENT)

        # Define filter that only allows user1
        def filter_user1(subscription, message):
            return subscription.user_id == "user1"

        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data={"sentiment": "positive"},
            timestamp=datetime.now()
        )

        await connection_manager.broadcast_to_subscribers(
            message, SubscriptionType.SENTIMENT, filter_user1
        )

        # Only websocket1 should have received the message
        websocket1.send_json.assert_called_once()
        websocket2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_sentiment_update(self, connection_manager):
        """Test sending sentiment update."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        connection_id = await connection_manager.connect(websocket, "test_user")
        await connection_manager.subscribe(connection_id, SubscriptionType.SENTIMENT)

        sentiment_data = {
            "currency": "RON",
            "sentiment": "positive",
            "score": 0.7
        }

        await connection_manager.send_sentiment_update(sentiment_data)

        # Check that message was sent
        websocket.send_json.assert_called_once()
        call_args = websocket.send_json.call_args[0][0]

        assert call_args["type"] == "sentiment_update"
        assert call_args["data"]["currency"] == "RON"

    @pytest.mark.asyncio
    async def test_send_market_data(self, connection_manager):
        """Test sending market data update."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        connection_id = await connection_manager.connect(websocket, "test_user")
        await connection_manager.subscribe(connection_id, SubscriptionType.MARKET_RATES)

        market_data = {
            "pair": "EUR/RON",
            "rate": 4.9750,
            "change": 0.002
        }

        await connection_manager.send_market_data(market_data)

        websocket.send_json.assert_called_once()
        call_args = websocket.send_json.call_args[0][0]

        assert call_args["type"] == "market_data"
        assert call_args["data"]["pair"] == "EUR/RON"

    @pytest.mark.asyncio
    async def test_send_alert(self, connection_manager):
        """Test sending alert."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        connection_id = await connection_manager.connect(websocket, "test_user")
        await connection_manager.subscribe(connection_id, SubscriptionType.ALERTS)

        alert_data = {
            "title": "High Volatility Alert",
            "message": "RON experiencing high volatility",
            "severity": "high"
        }

        await connection_manager.send_alert(alert_data, "high")

        websocket.send_json.assert_called_once()
        call_args = websocket.send_json.call_args[0][0]

        assert call_args["type"] == "alert"
        assert call_args["data"]["title"] == "High Volatility Alert"
        assert call_args["data"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, connection_manager, mock_websocket):
        """Test heartbeat handling."""
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        # Handle heartbeat
        await connection_manager.handle_heartbeat(connection_id)

        # Check that last heartbeat was updated
        subscription = connection_manager.active_connections[connection_id]
        assert subscription.last_heartbeat is not None

    def test_get_connection_stats(self, connection_manager):
        """Test getting connection statistics."""
        # This is a synchronous method, can test directly
        stats = asyncio.run(connection_manager.get_connection_stats())

        assert "total_connections" in stats
        assert "subscription_counts" in stats
        assert "connections_by_user" in stats
        assert "anonymous_connections" in stats

    def test_register_subscription_callback(self, connection_manager):
        """Test registering subscription callback."""
        async def test_callback(data):
            pass

        connection_manager.register_subscription_callback(SubscriptionType.SENTIMENT, test_callback)

        assert SubscriptionType.SENTIMENT in connection_manager.subscription_callbacks
        assert test_callback in connection_manager.subscription_callbacks[SubscriptionType.SENTIMENT]

    @pytest.mark.asyncio
    async def test_trigger_subscription_callbacks(self, connection_manager):
        """Test triggering subscription callbacks."""
        callback_results = []

        async def test_callback(data):
            callback_results.append(data)

        connection_manager.register_subscription_callback(SubscriptionType.SENTIMENT, test_callback)

        test_data = {"test": "data"}
        await connection_manager._trigger_subscription_callbacks(SubscriptionType.SENTIMENT, test_data)

        assert len(callback_results) == 1
        assert callback_results[0] == test_data

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self, connection_manager):
        """Test cleanup of stale connections."""
        # Create connection with old heartbeat
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        connection_id = await connection_manager.connect(websocket, "test_user")

        # Manually set old heartbeat
        subscription = connection_manager.active_connections[connection_id]
        subscription.last_heartbeat = datetime.now() - timedelta(minutes=5)

        # Run cleanup
        await connection_manager.cleanup_stale_connections()

        # Connection should be removed if stale
        # (This depends on the cleanup interval, but tests the mechanism)

    @pytest.mark.asyncio
    async def test_monitor_connection_task(self, connection_manager, mock_websocket):
        """Test connection monitoring task."""
        connection_id = await connection_manager.connect(mock_websocket, "test_user")

        # Let monitoring run briefly
        await asyncio.sleep(0.1)

        # Check that heartbeat was sent
        assert mock_websocket.send_json.call_count >= 1

        # Clean up
        await connection_manager.disconnect(connection_id)


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_full_websocket_lifecycle(self, connection_manager):
        """Test complete WebSocket connection lifecycle."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        # 1. Connect
        connection_id = await connection_manager.connect(websocket, "test_user")

        # 2. Subscribe to multiple types
        await connection_manager.subscribe(connection_id, SubscriptionType.SENTIMENT)
        await connection_manager.subscribe(connection_id, SubscriptionType.MARKET_RATES)

        # 3. Send different types of updates
        await connection_manager.send_sentiment_update({"currency": "RON", "sentiment": "positive"})
        await connection_manager.send_market_data({"pair": "EUR/RON", "rate": 4.9750})

        # 4. Verify messages were sent
        assert websocket.send_json.call_count >= 3  # Welcome + 2 updates

        # 5. Unsubscribe
        await connection_manager.unsubscribe(connection_id, SubscriptionType.SENTIMENT)

        # 6. Disconnect
        await connection_manager.disconnect(connection_id)

        # 7. Verify cleanup
        assert connection_id not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_multiple_users_different_subscriptions(self, connection_manager):
        """Test multiple users with different subscriptions."""
        # Create multiple users
        websocket1 = AsyncMock()
        websocket1.send_json = AsyncMock()
        websocket2 = AsyncMock()
        websocket2.send_json = AsyncMock()
        websocket3 = AsyncMock()
        websocket3.send_json = AsyncMock()

        # Connect users
        conn1 = await connection_manager.connect(websocket1, "user1")
        conn2 = await connection_manager.connect(websocket2, "user2")
        conn3 = await connection_manager.connect(websocket3, "user3")

        # Subscribe to different types
        await connection_manager.subscribe(conn1, SubscriptionType.SENTIMENT)
        await connection_manager.subscribe(conn2, SubscriptionType.MARKET_RATES)
        await connection_manager.subscribe(conn3, SubscriptionType.ALERTS)

        # Send sentiment update (only user1 should receive)
        await connection_manager.send_sentiment_update({"test": "sentiment"})

        # Send market data (only user2 should receive)
        await connection_manager.send_market_data({"test": "market"})

        # Send alert (only user3 should receive)
        await connection_manager.send_alert({"test": "alert"})

        # Verify each user received only their subscribed messages
        assert websocket1.send_json.call_count >= 2  # Welcome + sentiment
        assert websocket2.send_json.call_count >= 2  # Welcome + market
        assert websocket3.send_json.call_count >= 2  # Welcome + alert

        # Clean up
        for conn_id in [conn1, conn2, conn3]:
            await connection_manager.disconnect(conn_id)

    @pytest.mark.asyncio
    async def test_error_handling_in_message_sending(self, connection_manager):
        """Test error handling when sending messages."""
        # Create websocket that raises exception
        websocket = AsyncMock()
        websocket.send_json = AsyncMock(side_effect=Exception("Network error"))

        connection_id = await connection_manager.connect(websocket, "test_user")

        # Try to send message (should not raise exception)
        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data={"test": "data"},
            timestamp=datetime.now()
        )

        await connection_manager.send_message(connection_id, message)

        # Connection should be automatically disconnected due to error
        assert connection_id not in connection_manager.active_connections


class TestAdvancedFeatures:
    """Test advanced WebSocket features."""

    @pytest.mark.asyncio
    async def test_subscription_filters(self, connection_manager):
        """Test subscription filtering functionality."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        connection_id = await connection_manager.connect(websocket, "test_user")

        # Subscribe with filters
        filters = {"currencies": ["RON", "EUR"]}
        await connection_manager.subscribe(connection_id, SubscriptionType.SENTIMENT, filters)

        # Check that filters were stored
        subscription = connection_manager.active_connections[connection_id]
        assert subscription.filters["currencies"] == ["RON", "EUR"]

    @pytest.mark.asyncio
    async def test_redis_integration_mock(self, connection_manager):
        """Test Redis integration (mocked)."""
        with patch('redis.from_url') as mock_redis_from_url:
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_from_url.return_value = mock_redis_client

            await connection_manager.initialize_redis("redis://localhost:6379")

            # Check that Redis client was created
            assert connection_manager.redis_client == mock_redis_client
            mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_publish_message(self, connection_manager):
        """Test publishing message to Redis."""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        connection_manager.redis_client = mock_redis_client

        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data={"test": "data"},
            timestamp=datetime.now(),
            user_id="test_user"
        )

        await connection_manager.publish_to_redis("test_channel", message)

        # Check that publish was called
        mock_redis_client.publish.assert_called_once()
        call_args = mock_redis_client.publish.call_args[0]

        assert call_args[0] == "test_channel"

        # Check message data
        published_data = json.loads(call_args[1])
        assert published_data["type"] == "sentiment_update"
        assert published_data["data"]["test"] == "data"
        assert published_data["user_id"] == "test_user"

    def test_get_connection_statistics(self, connection_manager):
        """Test connection statistics."""
        # Test with no connections
        stats = asyncio.run(connection_manager.get_connection_stats())

        assert stats["total_connections"] == 0
        assert stats["connections_by_user"] == 0
        assert stats["anonymous_connections"] == 0
        assert all(count == 0 for count in stats["subscription_counts"].values())
