"""
Global currency and market support for AffectRON.
Extends support beyond Romanian markets to global financial markets.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .enhanced_websocket import connection_manager, MessageType, WebSocketMessage
from .alert_engine import alert_engine


@dataclass
class Currency:
    """Currency information."""
    code: str
    name: str
    symbol: str
    region: str
    category: str  # 'major', 'emerging', 'crypto', 'commodity'
    is_active: bool = True
    base_currency: Optional[str] = None  # For currency pairs

    # Economic indicators
    interest_rate: Optional[float] = None
    inflation_rate: Optional[float] = None
    gdp_growth: Optional[float] = None

    # Market information
    average_daily_volume: Optional[float] = None
    market_cap: Optional[float] = None


@dataclass
class CurrencyPair:
    """Currency pair trading information."""
    base_currency: str
    quote_currency: str
    pair_name: str  # e.g., "EUR/USD"
    exchange_rate: float
    bid: float
    ask: float
    spread: float
    volume: int
    timestamp: datetime

    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    moving_average_50: Optional[float] = None
    moving_average_200: Optional[float] = None


@dataclass
class MarketRegion:
    """Market region information."""
    code: str
    name: str
    timezone: str
    trading_hours: Dict[str, Tuple[str, str]]  # Day -> (open, close) times
    major_currencies: List[str]
    economic_calendar_url: Optional[str] = None


class GlobalCurrencyManager:
    """Manages global currency support and market data."""

    def __init__(self):
        # Comprehensive currency database
        self.currencies = self._initialize_currencies()

        # Market regions
        self.market_regions = self._initialize_market_regions()

        # Currency pairs
        self.currency_pairs = self._initialize_currency_pairs()

        # Real-time exchange rates (would be updated by market data streamer)
        self.exchange_rates: Dict[str, float] = {}

        # Economic indicators
        self.economic_indicators: Dict[str, Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

    def _initialize_currencies(self) -> Dict[str, Currency]:
        """Initialize comprehensive currency database."""
        return {
            # Major currencies
            'USD': Currency('USD', 'US Dollar', '$', 'North America', 'major',
                          interest_rate=5.25, average_daily_volume=6.6e12),
            'EUR': Currency('EUR', 'Euro', '€', 'Europe', 'major',
                          interest_rate=4.25, average_daily_volume=1.2e12),
            'GBP': Currency('GBP', 'British Pound', '£', 'Europe', 'major',
                          interest_rate=5.25, average_daily_volume=2.7e12),
            'JPY': Currency('JPY', 'Japanese Yen', '¥', 'Asia', 'major',
                          interest_rate=-0.1, average_daily_volume=1.2e12),
            'CHF': Currency('CHF', 'Swiss Franc', 'CHF', 'Europe', 'major',
                          interest_rate=1.75, average_daily_volume=3.2e11),
            'CAD': Currency('CAD', 'Canadian Dollar', 'C$', 'North America', 'major',
                          interest_rate=5.0, average_daily_volume=1.5e11),
            'AUD': Currency('AUD', 'Australian Dollar', 'A$', 'Oceania', 'major',
                          interest_rate=4.35, average_daily_volume=2.2e11),
            'NZD': Currency('NZD', 'New Zealand Dollar', 'NZ$', 'Oceania', 'major',
                          interest_rate=5.5, average_daily_volume=8.0e10),

            # Romanian and regional currencies
            'RON': Currency('RON', 'Romanian Leu', 'lei', 'Eastern Europe', 'emerging',
                          interest_rate=7.0, average_daily_volume=5.0e9),
            'PLN': Currency('PLN', 'Polish Złoty', 'zł', 'Eastern Europe', 'emerging',
                          interest_rate=5.75, average_daily_volume=8.0e9),
            'HUF': Currency('HUF', 'Hungarian Forint', 'Ft', 'Eastern Europe', 'emerging',
                          interest_rate=13.0, average_daily_volume=6.0e9),
            'CZK': Currency('CZK', 'Czech Koruna', 'Kč', 'Eastern Europe', 'emerging',
                          interest_rate=6.75, average_daily_volume=7.0e9),
            'BGN': Currency('BGN', 'Bulgarian Lev', 'лв', 'Eastern Europe', 'emerging',
                          interest_rate=3.79, average_daily_volume=2.0e9),

            # Other emerging market currencies
            'TRY': Currency('TRY', 'Turkish Lira', '₺', 'Middle East', 'emerging',
                          interest_rate=45.0, average_daily_volume=1.5e10),
            'RUB': Currency('RUB', 'Russian Ruble', '₽', 'Eastern Europe', 'emerging',
                          interest_rate=16.0, average_daily_volume=2.5e10),
            'INR': Currency('INR', 'Indian Rupee', '₹', 'Asia', 'emerging',
                          interest_rate=6.5, average_daily_volume=8.0e10),
            'BRL': Currency('BRL', 'Brazilian Real', 'R$', 'South America', 'emerging',
                          interest_rate=10.75, average_daily_volume=2.0e10),
            'MXN': Currency('MXN', 'Mexican Peso', 'Mex$', 'North America', 'emerging',
                          interest_rate=11.25, average_daily_volume=1.2e11),
            'ZAR': Currency('ZAR', 'South African Rand', 'R', 'Africa', 'emerging',
                          interest_rate=8.25, average_daily_volume=3.0e10),

            # Cryptocurrencies (as digital currencies)
            'BTC': Currency('BTC', 'Bitcoin', '₿', 'Global', 'crypto',
                          market_cap=1.4e12, average_daily_volume=3.0e10),
            'ETH': Currency('ETH', 'Ethereum', 'Ξ', 'Global', 'crypto',
                          market_cap=4.5e11, average_daily_volume=1.5e10),
            'BNB': Currency('BNB', 'Binance Coin', 'BNB', 'Global', 'crypto',
                          market_cap=8.0e10, average_daily_volume=2.0e9),
            'ADA': Currency('ADA', 'Cardano', 'ADA', 'Global', 'crypto',
                          market_cap=2.5e10, average_daily_volume=1.2e9),
            'SOL': Currency('SOL', 'Solana', 'SOL', 'Global', 'crypto',
                          market_cap=6.0e10, average_daily_volume=3.0e9),

            # Commodity currencies
            'XAU': Currency('XAU', 'Gold', 'oz', 'Global', 'commodity',
                          average_daily_volume=1.0e11),
            'XAG': Currency('XAG', 'Silver', 'oz', 'Global', 'commodity',
                          average_daily_volume=2.0e10),
            'WTI': Currency('WTI', 'WTI Crude Oil', 'bbl', 'Global', 'commodity',
                          average_daily_volume=5.0e10),
        }

    def _initialize_market_regions(self) -> Dict[str, MarketRegion]:
        """Initialize market regions with trading hours."""
        return {
            'europe': MarketRegion(
                code='europe',
                name='Europe',
                timezone='Europe/London',
                trading_hours={
                    'monday': ('08:00', '16:30'),
                    'tuesday': ('08:00', '16:30'),
                    'wednesday': ('08:00', '16:30'),
                    'thursday': ('08:00', '16:30'),
                    'friday': ('08:00', '16:30'),
                    'saturday': ('00:00', '00:00'),  # Closed
                    'sunday': ('00:00', '00:00')     # Closed
                },
                major_currencies=['EUR', 'GBP', 'CHF', 'RON', 'PLN', 'HUF', 'CZK', 'BGN'],
                economic_calendar_url='https://www.forexfactory.com/calendar.php'
            ),
            'north_america': MarketRegion(
                code='north_america',
                name='North America',
                timezone='America/New_York',
                trading_hours={
                    'monday': ('13:30', '20:00'),  # Overlaps with Europe
                    'tuesday': ('13:30', '20:00'),
                    'wednesday': ('13:30', '20:00'),
                    'thursday': ('13:30', '20:00'),
                    'friday': ('13:30', '20:00'),
                    'saturday': ('00:00', '00:00'),
                    'sunday': ('00:00', '00:00')
                },
                major_currencies=['USD', 'CAD', 'MXN'],
                economic_calendar_url='https://www.forexfactory.com/calendar.php'
            ),
            'asia': MarketRegion(
                code='asia',
                name='Asia-Pacific',
                timezone='Asia/Tokyo',
                trading_hours={
                    'monday': ('00:00', '09:00'),  # Overlaps with Europe close
                    'tuesday': ('00:00', '09:00'),
                    'wednesday': ('00:00', '09:00'),
                    'thursday': ('00:00', '09:00'),
                    'friday': ('00:00', '09:00'),
                    'saturday': ('00:00', '00:00'),
                    'sunday': ('00:00', '00:00')
                },
                major_currencies=['JPY', 'AUD', 'NZD', 'CNY', 'KRW', 'SGD'],
                economic_calendar_url='https://www.forexfactory.com/calendar.php'
            ),
            'crypto': MarketRegion(
                code='crypto',
                name='Cryptocurrency',
                timezone='UTC',
                trading_hours={
                    'monday': ('00:00', '24:00'),
                    'tuesday': ('00:00', '24:00'),
                    'wednesday': ('00:00', '24:00'),
                    'thursday': ('00:00', '24:00'),
                    'friday': ('00:00', '24:00'),
                    'saturday': ('00:00', '24:00'),
                    'sunday': ('00:00', '24:00')
                },
                major_currencies=['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
                economic_calendar_url='https://coingecko.com/en/crypto-events'
            )
        }

    def _initialize_currency_pairs(self) -> List[str]:
        """Initialize supported currency pairs."""
        return [
            # Major pairs
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD',

            # Cross pairs
            'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'GBP/JPY', 'GBP/CHF', 'CHF/JPY',

            # Emerging market pairs
            'EUR/RON', 'USD/RON', 'GBP/RON', 'CHF/RON',
            'EUR/PLN', 'USD/PLN', 'EUR/HUF', 'USD/HUF',
            'EUR/CZK', 'USD/CZK', 'EUR/TRY', 'USD/TRY',

            # Cryptocurrency pairs
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD',
            'BTC/EUR', 'ETH/EUR',

            # Commodity pairs
            'XAU/USD', 'XAG/USD', 'WTI/USD'
        ]

    def get_currency_info(self, currency_code: str) -> Optional[Currency]:
        """Get currency information."""
        return self.currencies.get(currency_code.upper())

    def get_currencies_by_region(self, region: str) -> List[Currency]:
        """Get currencies by region."""
        return [currency for currency in self.currencies.values()
                if currency.region.lower() == region.lower()]

    def get_currencies_by_category(self, category: str) -> List[Currency]:
        """Get currencies by category."""
        return [currency for currency in self.currencies.values()
                if currency.category.lower() == category.lower()]

    def get_supported_currency_pairs(self) -> List[str]:
        """Get all supported currency pairs."""
        return self.currency_pairs.copy()

    def is_currency_supported(self, currency_code: str) -> bool:
        """Check if currency is supported."""
        return currency_code.upper() in self.currencies

    def get_market_region(self, currency_code: str) -> Optional[MarketRegion]:
        """Get market region for currency."""
        currency = self.get_currency_info(currency_code)
        if not currency:
            return None

        # Find region that contains this currency
        for region in self.market_regions.values():
            if currency_code in region.major_currencies:
                return region

        return None

    def is_market_open(self, currency_code: str) -> bool:
        """Check if market is currently open for currency."""
        region = self.get_market_region(currency_code)
        if not region:
            return False  # Unknown market

        # Check current time in market timezone
        import pytz
        market_tz = pytz.timezone(region.timezone)
        current_time = datetime.now(market_tz)

        # Get current day and time
        current_day = current_time.strftime('%A').lower()
        current_hour_minute = current_time.strftime('%H:%M')

        # Check if market is open
        if current_day in region.trading_hours:
            open_time, close_time = region.trading_hours[current_day]

            if open_time == '00:00' and close_time == '00:00':
                return False  # Closed

            # Handle 24-hour markets (crypto)
            if open_time == '00:00' and close_time == '24:00':
                return True

            # Check time range
            return open_time <= current_hour_minute <= close_time

        return False

    async def update_exchange_rates(self, rates: Dict[str, float]):
        """Update exchange rates."""
        self.exchange_rates.update(rates)

        # Broadcast updates via WebSocket
        for pair, rate in rates.items():
            await connection_manager.send_market_data({
                'currency_pair': pair,
                'rate': rate,
                'timestamp': datetime.now().isoformat(),
                'source': 'global_currency_manager'
            })

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate between two currencies."""
        pair = f"{from_currency.upper()}/{to_currency.upper()}"

        if pair in self.exchange_rates:
            return self.exchange_rates[pair]

        # Try reverse pair
        reverse_pair = f"{to_currency.upper()}/{from_currency.upper()}"
        if reverse_pair in self.exchange_rates:
            return 1.0 / self.exchange_rates[reverse_pair]

        return None

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Optional[float]:
        """Convert amount between currencies."""
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate:
            return amount * rate
        return None

    async def get_economic_calendar(self, region: str = "europe", days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Get economic calendar for region."""
        region_info = self.market_regions.get(region.lower())
        if not region_info or not region_info.economic_calendar_url:
            return []

        # In production, this would scrape or API call to get calendar
        # For now, return mock data
        mock_events = [
            {
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '10:00',
                'event': 'ECB Interest Rate Decision',
                'currency': 'EUR',
                'impact': 'high',
                'forecast': '4.25%',
                'previous': '4.25%'
            },
            {
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '09:00',
                'event': 'German CPI',
                'currency': 'EUR',
                'impact': 'medium',
                'forecast': '2.3%',
                'previous': '2.4%'
            },
            {
                'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'time': '14:30',
                'event': 'US Non-Farm Payrolls',
                'currency': 'USD',
                'impact': 'high',
                'forecast': '180K',
                'previous': '199K'
            }
        ]

        return mock_events

    def get_currency_correlations(self, base_currency: str, days: int = 30) -> Dict[str, float]:
        """Get correlation coefficients with other currencies."""
        # In production, this would calculate actual correlations
        # For now, return mock correlations based on region and category

        correlations = {}
        base_curr = self.get_currency_info(base_currency)

        if not base_curr:
            return correlations

        for curr_code, currency in self.currencies.items():
            if curr_code == base_currency:
                continue

            # Calculate mock correlation based on region and category similarity
            correlation = 0.0

            # Same region = higher correlation
            if currency.region == base_curr.region:
                correlation += 0.4

            # Same category = higher correlation
            if currency.category == base_curr.category:
                correlation += 0.3

            # Geographic proximity bonus
            if base_curr.region in ['Eastern Europe', 'Europe'] and currency.region in ['Eastern Europe', 'Europe']:
                correlation += 0.2

            # Random factor for realism
            import random
            correlation += random.uniform(-0.1, 0.1)

            correlations[curr_code] = min(max(correlation, -0.9), 0.9)

        return correlations

    def get_global_market_summary(self) -> Dict[str, Any]:
        """Get global market summary."""
        # Count currencies by region and category
        region_counts = {}
        category_counts = {}

        for currency in self.currencies.values():
            region_counts[currency.region] = region_counts.get(currency.region, 0) + 1
            category_counts[currency.category] = category_counts.get(currency.category, 0) + 1

        # Market status
        markets_open = sum(1 for curr in self.currencies.values() if self.is_market_open(curr.code))

        return {
            'total_currencies': len(self.currencies),
            'currencies_by_region': region_counts,
            'currencies_by_category': category_counts,
            'markets_currently_open': markets_open,
            'supported_pairs': len(self.currency_pairs),
            'last_updated': datetime.now().isoformat()
        }

    async def process_cross_market_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment data for cross-market analysis."""
        currency = sentiment_data.get('currency', '')

        if not currency or not self.is_currency_supported(currency):
            return {}

        # Get currency information
        currency_info = self.get_currency_info(currency)

        # Get correlated currencies
        correlations = self.get_currency_correlations(currency)

        # Find highly correlated currencies (> 0.7 correlation)
        highly_correlated = {
            curr: corr for curr, corr in correlations.items()
            if corr > 0.7 and curr != currency
        }

        # Analyze cross-market impact
        cross_market_impact = {}

        for correlated_currency in highly_correlated.keys():
            # Calculate potential impact
            impact_score = highly_correlated[correlated_currency] * sentiment_data.get('sentiment_score', 0)

            cross_market_impact[correlated_currency] = {
                'correlation': highly_correlated[correlated_currency],
                'expected_impact': impact_score,
                'region': self.get_currency_info(correlated_currency).region,
                'category': self.get_currency_info(correlated_currency).category
            }

        return {
            'base_currency': currency,
            'base_currency_info': {
                'name': currency_info.name,
                'region': currency_info.region,
                'category': currency_info.category
            },
            'correlated_currencies': highly_correlated,
            'cross_market_impact': cross_market_impact,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_optimal_trading_hours(self, currency_code: str) -> Dict[str, Any]:
        """Get optimal trading hours for currency."""
        region = self.get_market_region(currency_code)

        if not region:
            return {'error': 'Currency not supported'}

        # Get current time in market timezone
        import pytz
        market_tz = pytz.timezone(region.timezone)
        current_time = datetime.now(market_tz)

        # Find next trading session
        optimal_hours = []

        for i in range(7):  # Check next 7 days
            check_date = current_time + timedelta(days=i)
            day_name = check_date.strftime('%A').lower()

            if day_name in region.trading_hours:
                open_time, close_time = region.trading_hours[day_name]

                if open_time != '00:00' or close_time != '00:00':  # Not closed
                    optimal_hours.append({
                        'date': check_date.strftime('%Y-%m-%d'),
                        'day': day_name,
                        'open_time': open_time,
                        'close_time': close_time,
                        'timezone': region.timezone
                    })

        return {
            'currency': currency_code,
            'region': region.name,
            'next_trading_sessions': optimal_hours[:3],  # Next 3 sessions
            'current_market_status': 'open' if self.is_market_open(currency_code) else 'closed'
        }


# Global currency manager instance
global_currency_manager = GlobalCurrencyManager()


async def initialize_global_currency_support():
    """Initialize global currency support."""
    logging.getLogger(__name__).info(f"Global currency support initialized with {len(global_currency_manager.currencies)} currencies")


def get_global_currency_status():
    """Get global currency system status."""
    return {
        'initialized': True,
        'total_currencies': len(global_currency_manager.currencies),
        'supported_pairs': len(global_currency_manager.currency_pairs),
        'market_regions': list(global_currency_manager.market_regions.keys()),
        'summary': global_currency_manager.get_global_market_summary()
    }
