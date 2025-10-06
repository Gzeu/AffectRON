"""
Enhanced WebSocket manager for real-time updates.
Supports streaming data, user subscriptions, and advanced real-time features.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
import asyncio_mqtt as mqtt

from .main import app


class MessageType(Enum):
    """Types of WebSocket messages."""
    SENTIMENT_UPDATE = "sentiment_update"
    MARKET_DATA = "market_data"
    ANALYTICS_UPDATE = "analytics_update"
    ALERT = "alert"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION_UPDATE = "subscription_update"
    ERROR = "error"


class SubscriptionType(Enum):
    """Types of data subscriptions."""
    SENTIMENT = "sentiment"
    MARKET_RATES = "market_rates"
    ANALYTICS = "analytics"
    ALERTS = "alerts"
    ALL = "all"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    subscription_filter: Optional[str] = None


@dataclass
class UserSubscription:
    """User subscription information."""
    websocket: WebSocket
    subscriptions: Set[SubscriptionType]
    last_heartbeat: datetime
    user_id: Optional[str] = None
    filters: Dict[str, Any] = None


class EnhancedConnectionManager:
    """Enhanced WebSocket connection manager with advanced features."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.active_connections: Dict[str, UserSubscription] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 60   # seconds
        self.subscription_callbacks: Dict[SubscriptionType, List[Callable]] = {}

        # MQTT client for external data sources (optional)
        self.mqtt_client = None

        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Accept WebSocket connection and register user."""
        await websocket.accept()

        connection_id = f"{user_id}_{id(websocket)}" if user_id else f"anon_{id(websocket)}"

        subscription = UserSubscription(
            websocket=websocket,
            subscriptions=set(),
            last_heartbeat=datetime.now(),
            user_id=user_id,
            filters={}
        )

        self.active_connections[connection_id] = subscription

        # Start heartbeat monitoring
        asyncio.create_task(self._monitor_connection(connection_id))

        self.logger.info(f"WebSocket connected: {connection_id}")

        # Send welcome message
        await self.send_message(
            connection_id,
            WebSocketMessage(
                message_type=MessageType.SYSTEM_STATUS,
                data={
                    "status": "connected",
                    "connection_id": connection_id,
                    "supported_subscriptions": [sub.value for sub in SubscriptionType]
                }
            )
        )

        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect WebSocket and cleanup."""
        if connection_id in self.active_connections:
            subscription = self.active_connections[connection_id]

            # Close WebSocket
            try:
                await subscription.websocket.close()
            except:
                pass  # Already closed

            # Remove from active connections
            del self.active_connections[connection_id]

            self.logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe(self, connection_id: str, subscription_type: SubscriptionType,
                       filters: Optional[Dict[str, Any]] = None):
        """Subscribe user to specific data type."""
        if connection_id not in self.active_connections:
            raise ValueError(f"Connection {connection_id} not found")

        subscription = self.active_connections[connection_id]
        subscription.subscriptions.add(subscription_type)

        if filters:
            subscription.filters.update(filters)

        # Send subscription confirmation
        await self.send_message(
            connection_id,
            WebSocketMessage(
                message_type=MessageType.SUBSCRIPTION_UPDATE,
                data={
                    "action": "subscribed",
                    "subscription_type": subscription_type.value,
                    "filters": filters
                }
            )
        )

        self.logger.info(f"User {connection_id} subscribed to {subscription_type.value}")

    async def unsubscribe(self, connection_id: str, subscription_type: SubscriptionType):
        """Unsubscribe user from specific data type."""
        if connection_id not in self.active_connections:
            return

        subscription = self.active_connections[connection_id]
        subscription.subscriptions.discard(subscription_type)

        # Send unsubscription confirmation
        await self.send_message(
            connection_id,
            WebSocketMessage(
                message_type=MessageType.SUBSCRIPTION_UPDATE,
                data={
                    "action": "unsubscribed",
                    "subscription_type": subscription_type.value
                }
            )
        )

    async def send_message(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return

        subscription = self.active_connections[connection_id]
        websocket = subscription.websocket

        try:
            message_data = {
                "type": message.message_type.value,
                "data": message.data,
                "timestamp": message.timestamp.isoformat(),
                "user_id": message.user_id
            }

            await websocket.send_json(message_data)

        except WebSocketDisconnect:
            # Connection closed, cleanup
            await self.disconnect(connection_id)
        except Exception as e:
            self.logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def broadcast_to_subscribers(self, message: WebSocketMessage,
                                     subscription_type: SubscriptionType,
                                     filter_func: Optional[Callable] = None):
        """Broadcast message to all subscribers of a specific type."""
        disconnected_connections = []

        for connection_id, subscription in self.active_connections.items():
            # Check if user is subscribed to this type
            if subscription_type not in subscription.subscriptions and subscription_type != SubscriptionType.ALL:
                continue

            # Apply custom filter if provided
            if filter_func and not filter_func(subscription, message):
                continue

            # Send message
            try:
                await self.send_message(connection_id, message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected_connections.append(connection_id)

        # Cleanup disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)

    async def send_sentiment_update(self, sentiment_data: Dict[str, Any]):
        """Send sentiment analysis update."""
        message = WebSocketMessage(
            message_type=MessageType.SENTIMENT_UPDATE,
            data=sentiment_data,
            timestamp=datetime.now()
        )

        await self.broadcast_to_subscribers(message, SubscriptionType.SENTIMENT)

        # Also trigger callbacks
        await self._trigger_subscription_callbacks(SubscriptionType.SENTIMENT, sentiment_data)

    async def send_market_data(self, market_data: Dict[str, Any]):
        """Send market data update."""
        message = WebSocketMessage(
            message_type=MessageType.MARKET_DATA,
            data=market_data,
            timestamp=datetime.now()
        )

        await self.broadcast_to_subscribers(message, SubscriptionType.MARKET_RATES)

    async def send_analytics_update(self, analytics_data: Dict[str, Any]):
        """Send analytics update."""
        message = WebSocketMessage(
            message_type=MessageType.ANALYTICS_UPDATE,
            data=analytics_data,
            timestamp=datetime.now()
        )

        await self.broadcast_to_subscribers(message, SubscriptionType.ANALYTICS)

    async def send_alert(self, alert_data: Dict[str, Any], priority: str = "medium"):
        """Send alert to subscribers."""
        message = WebSocketMessage(
            message_type=MessageType.ALERT,
            data={
                **alert_data,
                "priority": priority,
                "alert_id": f"alert_{datetime.now().timestamp()}"
            },
            timestamp=datetime.now()
        )

        # Send to all subscribers for high priority alerts
        if priority == "high":
            await self.broadcast_to_subscribers(message, SubscriptionType.ALL)
        else:
            await self.broadcast_to_subscribers(message, SubscriptionType.ALERTS)

    async def _monitor_connection(self, connection_id: str):
        """Monitor connection health and handle heartbeat."""
        subscription = self.active_connections[connection_id]

        while connection_id in self.active_connections:
            try:
                # Check if heartbeat is overdue
                time_since_heartbeat = datetime.now() - subscription.last_heartbeat

                if time_since_heartbeat.total_seconds() > self.heartbeat_interval * 2:
                    self.logger.warning(f"Connection {connection_id} heartbeat overdue, disconnecting")
                    await self.disconnect(connection_id)
                    break

                # Send heartbeat
                await self.send_message(
                    connection_id,
                    WebSocketMessage(
                        message_type=MessageType.HEARTBEAT,
                        data={"status": "alive"},
                        timestamp=datetime.now()
                    )
                )

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring connection {connection_id}: {e}")
                await self.disconnect(connection_id)
                break

    async def handle_heartbeat(self, connection_id: str):
        """Handle incoming heartbeat from client."""
        if connection_id in self.active_connections:
            self.active_connections[connection_id].last_heartbeat = datetime.now()

    async def _trigger_subscription_callbacks(self, subscription_type: SubscriptionType, data: Dict[str, Any]):
        """Trigger registered callbacks for subscription type."""
        if subscription_type in self.subscription_callbacks:
            for callback in self.subscription_callbacks[subscription_type]:
                try:
                    await callback(data)
                except Exception as e:
                    self.logger.error(f"Error in subscription callback: {e}")

    def register_subscription_callback(self, subscription_type: SubscriptionType, callback: Callable):
        """Register callback for subscription type."""
        if subscription_type not in self.subscription_callbacks:
            self.subscription_callbacks[subscription_type] = []

        self.subscription_callbacks[subscription_type].append(callback)

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total_connections = len(self.active_connections)

        # Count subscriptions
        subscription_counts = {sub_type.value: 0 for sub_type in SubscriptionType}
        for subscription in self.active_connections.values():
            for sub_type in subscription.subscriptions:
                subscription_counts[sub_type.value] += 1

        return {
            "total_connections": total_connections,
            "subscription_counts": subscription_counts,
            "connections_by_user": len(set(sub.user_id for sub in self.active_connections.values() if sub.user_id)),
            "anonymous_connections": len([sub for sub in self.active_connections.values() if not sub.user_id])
        }

    async def cleanup_stale_connections(self):
        """Clean up stale connections."""
        current_time = datetime.now()
        stale_connections = []

        for connection_id, subscription in self.active_connections.items():
            time_since_heartbeat = current_time - subscription.last_heartbeat

            if time_since_heartbeat.total_seconds() > self.heartbeat_interval * 3:
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            await self.disconnect(connection_id)

        if stale_connections:
            self.logger.info(f"Cleaned up {len(stale_connections)} stale connections")

    async def initialize_redis(self, redis_url: str):
        """Initialize Redis client for distributed messaging."""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.logger.info("Redis client initialized for WebSocket messaging")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    async def publish_to_redis(self, channel: str, message: WebSocketMessage):
        """Publish message to Redis channel for distributed systems."""
        if not self.redis_client:
            return

        try:
            message_data = {
                "type": message.message_type.value,
                "data": message.data,
                "timestamp": message.timestamp.isoformat(),
                "user_id": message.user_id
            }

            await self.redis_client.publish(channel, json.dumps(message_data))

        except Exception as e:
            self.logger.error(f"Error publishing to Redis: {e}")

    async def start_cleanup_task(self):
        """Start periodic cleanup task."""
        while True:
            try:
                await self.cleanup_stale_connections()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(self.cleanup_interval)


# Global connection manager instance
connection_manager = EnhancedConnectionManager()


@app.on_event("startup")
async def startup_websocket_manager():
    """Initialize WebSocket manager on startup."""
    # Initialize Redis if configured
    redis_url = getattr(app.state, 'settings', None)
    if redis_url and hasattr(redis_url, 'redis_url'):
        await connection_manager.initialize_redis(redis_url.redis_url)

    # Start cleanup task
    asyncio.create_task(connection_manager.start_cleanup_task())


@app.on_event("shutdown")
async def shutdown_websocket_manager():
    """Cleanup WebSocket manager on shutdown."""
    # Close all connections
    for connection_id in list(connection_manager.active_connections.keys()):
        await connection_manager.disconnect(connection_id)
