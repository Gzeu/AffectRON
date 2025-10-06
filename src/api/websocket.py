"""
WebSocket connection manager for real-time updates.
Handles WebSocket connections and message broadcasting.
"""

import json
import logging
from typing import Dict, List

from fastapi import WebSocket
from datetime import datetime


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "sentiment": [],
            "market": [],
            "alerts": [],
            "general": []
        }

    async def connect(self, websocket: WebSocket, channel: str = "general"):
        """
        Accept a WebSocket connection.

        Args:
            websocket: WebSocket connection
            channel: Channel to subscribe to (sentiment, market, alerts, general)
        """
        await websocket.accept()

        if channel not in self.active_connections:
            channel = "general"

        self.active_connections[channel].append(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")

    def disconnect(self, websocket: WebSocket, channel: str = "general"):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
            channel: Channel the connection was subscribed to
        """
        if channel in self.active_connections:
            try:
                self.active_connections[channel].remove(websocket)
                logger.info(f"WebSocket disconnected from channel: {channel}")
            except ValueError:
                logger.warning(f"WebSocket not found in channel: {channel}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            # Remove broken connection
            for channel, connections in self.active_connections.items():
                if websocket in connections:
                    connections.remove(websocket)

    async def broadcast_to_channel(self, message: dict, channel: str):
        """
        Broadcast a message to all connections in a channel.

        Args:
            message: Message to broadcast
            channel: Target channel
        """
        if channel not in self.active_connections:
            logger.warning(f"Unknown channel: {channel}")
            return

        disconnected_connections = []
        message["timestamp"] = datetime.now().isoformat()

        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected_connections.append(connection)

        # Remove broken connections
        for connection in disconnected_connections:
            self.active_connections[channel].remove(connection)

    async def broadcast_sentiment_update(self, currency: str, sentiment_data: dict):
        """
        Broadcast sentiment update to relevant channels.

        Args:
            currency: Currency that was updated
            sentiment_data: Sentiment analysis data
        """
        message = {
            "type": "sentiment_update",
            "currency": currency,
            "data": sentiment_data
        }

        # Broadcast to sentiment channel and general channel
        await self.broadcast_to_channel(message, "sentiment")
        await self.broadcast_to_channel(message, "general")

    async def broadcast_market_update(self, market_data: dict):
        """
        Broadcast market data update.

        Args:
            market_data: Market data update
        """
        message = {
            "type": "market_update",
            "data": market_data
        }

        # Broadcast to market channel and general channel
        await self.broadcast_to_channel(message, "market")
        await self.broadcast_to_channel(message, "general")

    async def broadcast_alert(self, alert_data: dict):
        """
        Broadcast alert to all channels.

        Args:
            alert_data: Alert information
        """
        message = {
            "type": "alert",
            "data": alert_data
        }

        # Broadcast alert to all channels
        for channel in self.active_connections.keys():
            await self.broadcast_to_channel(message, channel)

    async def get_connection_count(self) -> Dict[str, int]:
        """
        Get count of active connections per channel.

        Returns:
            Dictionary with connection counts per channel
        """
        return {
            channel: len(connections)
            for channel, connections in self.active_connections.items()
        }
