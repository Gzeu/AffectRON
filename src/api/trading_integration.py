"""
Trading platform integration module.
Provides connectors for MT4/MT5, FIX protocol, and custom broker APIs.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import socket
import struct

from .enhanced_websocket import connection_manager, MessageType, WebSocketMessage


@dataclass
class TradingSignal:
    """Trading signal data."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'CLOSE'
    confidence: float
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class Position:
    """Trading position information."""
    ticket: str
    symbol: str
    position_type: str  # 'BUY', 'SELL'
    volume: float
    open_price: float
    current_price: float
    profit_loss: float
    open_time: datetime
    broker: str


class TradingPlatformConnector(ABC):
    """Abstract base class for trading platform connectors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to trading platform."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from trading platform."""
        pass

    @abstractmethod
    async def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal to platform."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def close_position(self, ticket: str) -> bool:
        """Close a position."""
        pass


class MT4Connector(TradingPlatformConnector):
    """MetaTrader 4 platform connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8222)  # Default MT4 Manager API port
        self.username = config.get('username')
        self.password = config.get('password')
        self.socket = None

    async def connect(self) -> bool:
        """Connect to MT4 Manager API."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))

            # Authenticate (simplified - real implementation would use proper MT4 protocol)
            auth_data = {
                'username': self.username,
                'password': self.password,
                'action': 'login'
            }

            await self._send_data(auth_data)

            response = await self._receive_data()
            if response.get('status') == 'success':
                self.connected = True
                self.logger.info(f"Connected to MT4 at {self.host}:{self.port}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MT4: {e}")

        return False

    async def disconnect(self):
        """Disconnect from MT4."""
        if self.socket:
            await self._send_data({'action': 'logout'})
            self.socket.close()
            self.socket = None

        self.connected = False
        self.logger.info("Disconnected from MT4")

    async def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal to MT4."""
        if not self.connected:
            return False

        try:
            # Convert signal to MT4 format
            mt4_signal = {
                'action': 'signal',
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'volume': signal.metadata.get('volume', 0.1) if signal.metadata else 0.1,
                'price': signal.price_target or 0,
                'sl': signal.stop_loss or 0,
                'tp': signal.take_profit or 0,
                'comment': f"AffectRON Signal - Confidence: {signal.confidence:.2f}"
            }

            await self._send_data(mt4_signal)
            response = await self._receive_data()

            if response.get('status') == 'success':
                self.logger.info(f"Signal sent to MT4: {signal.symbol} {signal.signal_type}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to send signal to MT4: {e}")

        return False

    async def get_positions(self) -> List[Position]:
        """Get current MT4 positions."""
        if not self.connected:
            return []

        try:
            await self._send_data({'action': 'get_positions'})
            response = await self._receive_data()

            positions = []
            for pos_data in response.get('positions', []):
                position = Position(
                    ticket=pos_data.get('ticket'),
                    symbol=pos_data.get('symbol'),
                    position_type=pos_data.get('type'),
                    volume=pos_data.get('volume', 0),
                    open_price=pos_data.get('open_price', 0),
                    current_price=pos_data.get('current_price', 0),
                    profit_loss=pos_data.get('profit', 0),
                    open_time=datetime.fromisoformat(pos_data.get('open_time')),
                    broker='MT4'
                )
                positions.append(position)

            return positions

        except Exception as e:
            self.logger.error(f"Failed to get MT4 positions: {e}")
            return []

    async def close_position(self, ticket: str) -> bool:
        """Close MT4 position."""
        if not self.connected:
            return False

        try:
            await self._send_data({
                'action': 'close_position',
                'ticket': ticket
            })

            response = await self._receive_data()
            return response.get('status') == 'success'

        except Exception as e:
            self.logger.error(f"Failed to close MT4 position {ticket}: {e}")
            return False

    async def _send_data(self, data: Dict):
        """Send data to MT4 socket."""
        if self.socket:
            json_data = json.dumps(data).encode('utf-8')
            length = struct.pack('<I', len(json_data))
            self.socket.send(length + json_data)

    async def _receive_data(self) -> Dict:
        """Receive data from MT4 socket."""
        if not self.socket:
            return {}

        try:
            # Read length prefix
            length_data = self.socket.recv(4)
            if not length_data:
                return {}

            length = struct.unpack('<I', length_data)[0]
            data = self.socket.recv(length)

            return json.loads(data.decode('utf-8'))

        except Exception as e:
            self.logger.error(f"Error receiving MT4 data: {e}")
            return {}


class FIXConnector(TradingPlatformConnector):
    """FIX protocol connector for professional trading platforms."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sender_comp_id = config.get('sender_comp_id', 'AFFECTRON')
        self.target_comp_id = config.get('target_comp_id', 'BROKER')
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 9876)
        self.socket = None
        self.sequence_number = 1

    async def connect(self) -> bool:
        """Connect to FIX server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))

            # Send logon message
            logon_msg = self._build_fix_message('A', {
                '35': 'A',  # Logon
                '98': '0',  # No encryption
                '108': '30'  # Heartbeat interval
            })

            self.socket.send(logon_msg.encode('ascii'))

            # Wait for logon response
            response = self.socket.recv(1024).decode('ascii')

            if '35=A' in response and '39=0' in response:  # Logon accepted
                self.connected = True
                self.logger.info(f"Connected to FIX server at {self.host}:{self.port}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to connect to FIX server: {e}")

        return False

    async def disconnect(self):
        """Disconnect from FIX server."""
        if self.socket:
            # Send logout message
            logout_msg = self._build_fix_message('5', {'35': '5'})
            self.socket.send(logout_msg.encode('ascii'))
            self.socket.close()
            self.socket = None

        self.connected = False
        self.logger.info("Disconnected from FIX server")

    async def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal via FIX protocol."""
        if not self.connected:
            return False

        try:
            # Build New Order Single message
            fix_fields = {
                '35': 'D',  # New Order Single
                '54': '1' if signal.signal_type == 'BUY' else '2',  # Side
                '55': signal.symbol,  # Symbol
                '38': str(signal.metadata.get('volume', 0.1)) if signal.metadata else '0.1',  # OrderQty
                '44': str(signal.price_target) if signal.price_target else '',  # Price
                '41': 'AFFECTRON',  # Originating source
                '58': f"AffectRON Signal - Confidence: {signal.confidence:.2f}"  # Text
            }

            # Add stop loss and take profit if specified
            if signal.stop_loss:
                fix_fields['123'] = 'Y'  # StopPx present
                fix_fields['38'] = str(signal.stop_loss)  # StopPx

            if signal.take_profit:
                fix_fields['124'] = 'Y'  # Take profit present
                fix_fields['39'] = str(signal.take_profit)  # Take profit price

            fix_message = self._build_fix_message('D', fix_fields)
            self.socket.send(fix_message.encode('ascii'))

            # Wait for execution report
            response = self.socket.recv(1024).decode('ascii')

            if '35=8' in response and '39=0' in response:  # Execution accepted
                self.logger.info(f"FIX signal sent: {signal.symbol} {signal.signal_type}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to send FIX signal: {e}")

        return False

    async def get_positions(self) -> List[Position]:
        """Get positions via FIX (simplified implementation)."""
        # FIX doesn't have a direct "get positions" message
        # This would typically be handled via Request for Positions (AN)
        return []

    async def close_position(self, ticket: str) -> bool:
        """Close position via FIX."""
        if not self.connected:
            return False

        try:
            # Build Order Cancel Request
            fix_fields = {
                '35': 'F',  # Order Cancel Request
                '41': 'AFFECTRON',
                '37': ticket  # Original order reference
            }

            fix_message = self._build_fix_message('F', fix_fields)
            self.socket.send(fix_message.encode('ascii'))

            return True

        except Exception as e:
            self.logger.error(f"Failed to close FIX position {ticket}: {e}")
            return False

    def _build_fix_message(self, msg_type: str, fields: Dict[str, str]) -> str:
        """Build FIX message."""
        message = []

        # Standard fields
        message.append(f"35={msg_type}")  # MsgType
        message.append(f"49={self.sender_comp_id}")  # SenderCompID
        message.append(f"56={self.target_comp_id}")  # TargetCompID
        message.append(f"34={self.sequence_number}")  # MsgSeqNum
        message.append(f"52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}")  # SendingTime

        # Add custom fields
        for tag, value in fields.items():
            if value:  # Only add non-empty values
                message.append(f"{tag}={value}")

        # Calculate body length and checksum
        body = "|".join(message)
        body_length = len(body)

        # Add BeginString, BodyLength, and CheckSum
        fix_message = f"8=FIX.4.4|35={msg_type}|34={self.sequence_number}|49={self.sender_comp_id}|56={self.target_comp_id}|52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}"
        for tag, value in fields.items():
            if value:
                fix_message += f"|{tag}={value}"

        fix_message += f"|10={self._calculate_checksum(fix_message)}"

        self.sequence_number += 1
        return fix_message

    def _calculate_checksum(self, message: str) -> str:
        """Calculate FIX message checksum."""
        checksum = sum(ord(c) for c in message) % 256
        return f"{checksum:03d}"


class TradingPlatformManager:
    """Manages multiple trading platform connectors."""

    def __init__(self):
        self.connectors: Dict[str, TradingPlatformConnector] = {}
        self.signal_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)

    def register_connector(self, name: str, connector: TradingPlatformConnector):
        """Register a trading platform connector."""
        self.connectors[name] = connector
        self.logger.info(f"Registered trading connector: {name}")

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered platforms."""
        results = {}

        for name, connector in self.connectors.items():
            try:
                success = await connector.connect()
                results[name] = success
            except Exception as e:
                self.logger.error(f"Error connecting to {name}: {e}")
                results[name] = False

        return results

    async def disconnect_all(self):
        """Disconnect from all platforms."""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from {name}: {e}")

    async def send_signal_to_all(self, signal: TradingSignal) -> Dict[str, bool]:
        """Send signal to all connected platforms."""
        results = {}

        for name, connector in self.connectors.items():
            if connector.connected:
                try:
                    success = await connector.send_signal(signal)
                    results[name] = success
                except Exception as e:
                    self.logger.error(f"Error sending signal to {name}: {e}")
                    results[name] = False
            else:
                results[name] = False

        return results

    async def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all platforms."""
        all_positions = {}

        for name, connector in self.connectors.items():
            if connector.connected:
                try:
                    positions = await connector.get_positions()
                    all_positions[name] = positions
                except Exception as e:
                    self.logger.error(f"Error getting positions from {name}: {e}")
                    all_positions[name] = []

        return all_positions

    def register_signal_callback(self, callback: Callable):
        """Register callback for trading signals."""
        self.signal_callbacks.append(callback)

    async def _notify_signal_callbacks(self, signal: TradingSignal, results: Dict[str, bool]):
        """Notify registered callbacks about signal execution."""
        for callback in self.signal_callbacks:
            try:
                await callback(signal, results)
            except Exception as e:
                self.logger.error(f"Error in signal callback: {e}")

    def create_mt4_connector(self, config: Dict[str, Any]) -> MT4Connector:
        """Create MT4 connector."""
        return MT4Connector(config)

    def create_fix_connector(self, config: Dict[str, Any]) -> FIXConnector:
        """Create FIX connector."""
        return FIXConnector(config)

    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors."""
        status = {}

        for name, connector in self.connectors.items():
            status[name] = {
                'connected': connector.connected,
                'type': connector.__class__.__name__,
                'config': {k: v for k, v in connector.config.items() if k != 'password'}
            }

        return status


# Global trading platform manager
trading_manager = TradingPlatformManager()


async def initialize_trading_platforms():
    """Initialize trading platform connectors."""
    # This would load configurations from settings
    # For now, just log initialization
    logging.getLogger(__name__).info("Trading platform manager initialized")


def create_trading_signal(sentiment_data: Dict[str, Any], confidence_threshold: float = 0.7) -> Optional[TradingSignal]:
    """Create trading signal from sentiment data."""
    currency = sentiment_data.get('currency', '')
    sentiment_score = sentiment_data.get('sentiment_score', 0)
    confidence = sentiment_data.get('confidence', 0)

    if confidence < confidence_threshold:
        return None

    # Determine signal type based on sentiment
    if sentiment_score > 0.6:
        signal_type = 'BUY'
    elif sentiment_score < -0.6:
        signal_type = 'SELL'
    else:
        return None  # Neutral sentiment, no signal

    # Create signal with metadata
    metadata = {
        'sentiment_score': sentiment_score,
        'confidence': confidence,
        'source': 'affectron_sentiment',
        'timestamp': datetime.now().isoformat()
    }

    return TradingSignal(
        symbol=currency,
        signal_type=signal_type,
        confidence=confidence,
        timestamp=datetime.now(),
        metadata=metadata
    )


async def process_sentiment_for_trading(sentiment_data: Dict[str, Any]):
    """Process sentiment data for trading signals."""
    signal = create_trading_signal(sentiment_data)

    if signal:
        # Send signal to all connected platforms
        results = await trading_manager.send_signal_to_all(signal)

        # Notify WebSocket clients
        await connection_manager.send_message(
            "trading_system",
            WebSocketMessage(
                message_type=MessageType.ALERT,
                data={
                    'type': 'trading_signal',
                    'signal': {
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence
                    },
                    'results': results
                },
                timestamp=datetime.now()
            )
        )

        # Notify callbacks
        await trading_manager._notify_signal_callbacks(signal, results)


def get_trading_status():
    """Get trading platform status."""
    return {
        'connectors': trading_manager.get_connector_status(),
        'manager_initialized': True
    }
