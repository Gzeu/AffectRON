"""
Real-time alert and notification system.
Provides intelligent alerting based on sentiment, market conditions, and custom rules.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Callable, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .enhanced_websocket import connection_manager, MessageType, WebSocketMessage


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    SENTIMENT_SPIKE = "sentiment_spike"
    MARKET_VOLATILITY = "market_volatility"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    DATA_SOURCE_ERROR = "data_source_error"
    SYSTEM_ERROR = "system_error"
    CUSTOM_RULE = "custom_rule"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    conditions: Dict[str, Any]
    cooldown_minutes: int = 60
    enabled: bool = True
    description: str = ""

    # Notification settings
    notify_email: bool = False
    notify_websocket: bool = True
    notify_webhook: bool = False
    webhook_url: Optional[str] = None

    # Affected entities (currencies, sources, etc.)
    affected_entities: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class AlertEngine:
    """Intelligent alert and notification engine."""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: Dict[str, Callable] = {}

        # Email configuration (would be loaded from settings)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': None,
            'password': None,
            'from_email': 'alerts@affectron.com'
        }

        # Rate limiting for alerts
        self.alert_rate_limits: Dict[str, datetime] = {}

        self.logger = logging.getLogger(__name__)

        # Set up default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                name="high_sentiment_volatility",
                alert_type=AlertType.SENTIMENT_SPIKE,
                severity=AlertSeverity.HIGH,
                conditions={
                    'sentiment_change_threshold': 0.3,
                    'time_window_minutes': 15,
                    'min_data_points': 5
                },
                description="Alert when sentiment volatility exceeds threshold"
            ),
            AlertRule(
                name="market_volatility_alert",
                alert_type=AlertType.MARKET_VOLATILITY,
                severity=AlertSeverity.MEDIUM,
                conditions={
                    'price_change_threshold': 0.02,
                    'time_window_minutes': 30,
                    'affected_currencies': ['RON', 'EUR', 'USD']
                },
                description="Alert when market shows unusual volatility"
            ),
            AlertRule(
                name="correlation_breakdown",
                alert_type=AlertType.CORRELATION_BREAKDOWN,
                severity=AlertSeverity.HIGH,
                conditions={
                    'correlation_threshold': 0.3,
                    'time_window_days': 7,
                    'min_correlation_samples': 20
                },
                description="Alert when sentiment-price correlation breaks down"
            ),
            AlertRule(
                name="data_source_error",
                alert_type=AlertType.DATA_SOURCE_ERROR,
                severity=AlertSeverity.MEDIUM,
                conditions={
                    'error_count_threshold': 3,
                    'time_window_minutes': 60
                },
                description="Alert when data source experiences repeated errors"
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    async def check_sentiment_volatility(self, currency: str, sentiment_data: List[Dict]) -> Optional[Alert]:
        """Check for sentiment volatility alerts."""
        rule = self.alert_rules.get("high_sentiment_volatility")
        if not rule or not rule.enabled:
            return None

        if len(sentiment_data) < rule.conditions['min_data_points']:
            return None

        # Calculate sentiment volatility
        sentiment_scores = [d['sentiment_score'] for d in sentiment_data]
        volatility = max(sentiment_scores) - min(sentiment_scores)

        if volatility > rule.conditions['sentiment_change_threshold']:
            return await self._create_alert(
                rule=rule,
                title=f"High Sentiment Volatility Detected - {currency}",
                message=f"Sentiment volatility of {volatility".2f"} exceeds threshold of {rule.conditions['sentiment_change_threshold']} for {currency}",
                data={
                    'currency': currency,
                    'volatility': volatility,
                    'sentiment_data': sentiment_data[-5:]  # Last 5 data points
                }
            )

        return None

    async def check_market_volatility(self, market_data: Dict) -> Optional[Alert]:
        """Check for market volatility alerts."""
        rule = self.alert_rules.get("market_volatility_alert")
        if not rule or not rule.enabled:
            return None

        currency = market_data.get('currency_pair', '').split('/')[0] if '/' in market_data.get('currency_pair', '') else market_data.get('currency')

        if currency not in rule.conditions.get('affected_currencies', []):
            return None

        price_change = abs(market_data.get('change_percent', 0)) / 100  # Convert percentage to decimal

        if price_change > rule.conditions['price_change_threshold']:
            return await self._create_alert(
                rule=rule,
                title=f"High Market Volatility - {currency}",
                message=f"Price change of {price_change".2%"} exceeds threshold for {currency}",
                data={
                    'currency': currency,
                    'price_change': price_change,
                    'market_data': market_data
                }
            )

        return None

    async def check_correlation_breakdown(self, correlation_data: Dict) -> Optional[Alert]:
        """Check for correlation breakdown alerts."""
        rule = self.alert_rules.get("correlation_breakdown")
        if not rule or not rule.enabled:
            return None

        correlation_coef = abs(correlation_data.get('correlation_coefficient', 0))

        if correlation_coef < rule.conditions['correlation_threshold']:
            currency_pair = correlation_data.get('currency_pair', 'Unknown')

            return await self._create_alert(
                rule=rule,
                title=f"Correlation Breakdown - {currency_pair}",
                message=f"Sentiment-price correlation ({correlation_coef".3f"}) below threshold for {currency_pair}",
                data={
                    'currency_pair': currency_pair,
                    'correlation_coefficient': correlation_coef,
                    'correlation_data': correlation_data
                }
            )

        return None

    async def check_data_source_errors(self, source_name: str, error_count: int) -> Optional[Alert]:
        """Check for data source error alerts."""
        rule = self.alert_rules.get("data_source_error")
        if not rule or not rule.enabled:
            return None

        if error_count >= rule.conditions['error_count_threshold']:
            return await self._create_alert(
                rule=rule,
                title=f"Data Source Error - {source_name}",
                message=f"Data source '{source_name}' has {error_count} consecutive errors",
                data={
                    'source_name': source_name,
                    'error_count': error_count
                }
            )

        return None

    async def _create_alert(self, rule: AlertRule, title: str, message: str, data: Dict) -> Alert:
        """Create and register a new alert."""
        alert_id = f"alert_{datetime.now().timestamp()}_{rule.name}"

        # Check rate limiting
        if not self._can_trigger_alert(rule.name):
            return None

        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=title,
            message=message,
            data=data,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)  # Alerts expire after 24 hours
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update rule last triggered time
        rule.last_triggered = datetime.now()
        self.alert_rate_limits[rule.name] = datetime.now()

        # Send notifications
        await self._send_notifications(alert, rule)

        self.logger.info(f"Created alert: {alert_id} - {title}")

        return alert

    def _can_trigger_alert(self, rule_name: str) -> bool:
        """Check if alert can be triggered (rate limiting)."""
        rule = self.alert_rules.get(rule_name)
        if not rule:
            return False

        last_triggered = self.alert_rate_limits.get(rule_name)

        if last_triggered:
            time_since_last = datetime.now() - last_triggered
            if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                return False

        return True

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        # WebSocket notification (always enabled for active alerts)
        if rule.notify_websocket:
            await self._send_websocket_notification(alert)

        # Email notification
        if rule.notify_email:
            await self._send_email_notification(alert, rule)

        # Webhook notification
        if rule.notify_webhook and rule.webhook_url:
            await self._send_webhook_notification(alert, rule)

    async def _send_websocket_notification(self, alert: Alert):
        """Send alert via WebSocket."""
        message_data = {
            'alert_id': alert.id,
            'title': alert.title,
            'message': alert.message,
            'severity': alert.severity.value,
            'alert_type': alert.alert_type.value,
            'data': alert.data,
            'timestamp': alert.timestamp.isoformat()
        }

        await connection_manager.send_alert(message_data, alert.severity.value)

    async def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Send alert via email."""
        if not all([self.email_config['smtp_server'], self.email_config['username'],
                   self.email_config['password']]):
            self.logger.warning("Email configuration incomplete, skipping email notification")
            return

        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = "admin@affectron.com"  # Would be configurable
            msg['Subject'] = f"AffectRON Alert: {alert.title}"

            # Email body
            body = f"""
AffectRON Alert Notification

Alert ID: {alert.id}
Severity: {alert.severity.value.upper()}
Type: {alert.alert_type.value}

Title: {alert.title}

Message: {alert.message}

Data: {json.dumps(alert.data, indent=2)}

Timestamp: {alert.timestamp.isoformat()}

---
This is an automated alert from AffectRON.
"""

            msg.attach(MimeText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            self.logger.info(f"Email alert sent: {alert.id}")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    async def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Send alert via webhook."""
        try:
            import aiohttp

            webhook_data = {
                'alert_id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'alert_type': alert.alert_type.value,
                'data': alert.data,
                'timestamp': alert.timestamp.isoformat()
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(rule.webhook_url, json=webhook_data) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook alert sent: {alert.id}")
                    else:
                        self.logger.error(f"Webhook alert failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")

    def update_alert_rule(self, rule_name: str, updates: Dict[str, Any]):
        """Update an existing alert rule."""
        if rule_name in self.alert_rules:
            rule = self.alert_rules[rule_name]

            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            self.logger.info(f"Updated alert rule: {rule_name}")

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.now()

            self.logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
            return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.active_alerts.values() if alert.severity == severity]

    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type."""
        return [alert for alert in self.active_alerts.values() if alert.alert_type == alert_type]

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []

    def cleanup_expired_alerts(self):
        """Clean up expired alerts."""
        current_time = datetime.now()
        expired_alerts = []

        for alert_id, alert in self.active_alerts.items():
            if alert.expires_at and current_time > alert.expires_at:
                expired_alerts.append(alert_id)

        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]

        if expired_alerts:
            self.logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {'total_alerts': 0, 'alerts_by_severity': {}, 'alerts_by_type': {}}

        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1

        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': severity_counts,
            'alerts_by_type': type_counts,
            'acknowledged_alerts': len([a for a in self.active_alerts.values() if a.acknowledged]),
            'unacknowledged_alerts': len([a for a in self.active_alerts.values() if not a.acknowledged])
        }

    async def process_market_data_for_alerts(self, market_data: Dict):
        """Process market data for potential alerts."""
        # Check market volatility
        await self.check_market_volatility(market_data)

    async def process_sentiment_data_for_alerts(self, sentiment_data: Dict):
        """Process sentiment data for potential alerts."""
        currency = sentiment_data.get('currency', '')

        if currency:
            # This would need historical sentiment data
            # For now, just check if sentiment is extreme
            sentiment_score = sentiment_data.get('sentiment_score', 0)

            if abs(sentiment_score) > 0.7:  # Very strong sentiment
                rule = AlertRule(
                    name="extreme_sentiment",
                    alert_type=AlertType.SENTIMENT_SPIKE,
                    severity=AlertSeverity.HIGH if abs(sentiment_score) > 0.8 else AlertSeverity.MEDIUM,
                    conditions={'threshold': 0.7},
                    description="Extreme sentiment detected"
                )

                await self._create_alert(
                    rule=rule,
                    title=f"Extreme Sentiment - {currency}",
                    message=f"Very strong sentiment detected: {sentiment_score".2f"}",
                    data=sentiment_data
                )

    async def process_correlation_data_for_alerts(self, correlation_data: Dict):
        """Process correlation data for potential alerts."""
        await self.check_correlation_breakdown(correlation_data)


# Global alert engine instance
alert_engine = AlertEngine()


async def start_alert_engine():
    """Start the alert engine."""
    # Start periodic cleanup
    asyncio.create_task(_alert_cleanup_task())


async def _alert_cleanup_task():
    """Periodic task to clean up expired alerts."""
    while True:
        try:
            alert_engine.cleanup_expired_alerts()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in alert cleanup task: {e}")
            await asyncio.sleep(300)


def get_alert_engine_status():
    """Get alert engine status."""
    return {
        'active_alerts': len(alert_engine.active_alerts),
        'total_rules': len(alert_engine.alert_rules),
        'alert_statistics': alert_engine.get_alert_statistics(),
        'engine_running': True
    }
