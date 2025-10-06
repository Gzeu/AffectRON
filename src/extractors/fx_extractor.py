"""
Foreign Exchange (FX) extractor for currency rates.
Extracts exchange rates from BNR, ECB, and other financial data sources.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from decimal import Decimal

import aiohttp
import requests
from bs4 import BeautifulSoup

from .base import BaseExtractor, ExtractorConfig, ExtractedContent


logger = logging.getLogger(__name__)


class FXExtractor(BaseExtractor):
    """Extractor for foreign exchange rates and market data."""

    def __init__(self, config: ExtractorConfig, db_session):
        super().__init__(config, db_session)

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about FX data sources."""
        return {
            "name": "Foreign Exchange Rate Extractor",
            "type": "fx",
            "sources": [
                "BNR (Banca Națională a României)",
                "ECB (European Central Bank)",
                "Federal Reserve",
                "CryptoCompare API"
            ],
            "currencies": ["RON", "EUR", "USD", "GBP", "CHF", "BTC", "ETH"]
        }

    async def extract_bnr_rates(self) -> List[ExtractedContent]:
        """Extract exchange rates from BNR (Romanian National Bank)."""
        contents = []

        try:
            # BNR provides XML data
            bnr_url = os.getenv('BNR_BASE_URL', 'https://www.bnr.ro/nbrfxrates.xml')

            async with self.session.get(bnr_url) as response:
                response.raise_for_status()
                xml_data = await response.text()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Extract publishing date
            publishing_date = None
            date_elem = root.find('.//PublishingDate')
            if date_elem is not None:
                publishing_date = datetime.strptime(date_elem.text, '%Y-%m-%d')

            # Extract exchange rates
            for rate_elem in root.findall('.//Rate'):
                currency = rate_elem.get('currency')
                multiplier = rate_elem.get('multiplier', '1')
                rate_value = rate_elem.text

                if currency and rate_value:
                    content = ExtractedContent(
                        source_id=f"bnr_{currency}",
                        content=f"BNR Exchange Rate for {currency}: {rate_value} RON",
                        title=f"{currency}/RON Exchange Rate",
                        url=bnr_url,
                        published_at=publishing_date,
                        metadata={
                            'source': 'BNR',
                            'currency': currency,
                            'rate': float(rate_value),
                            'multiplier': int(multiplier),
                            'base_currency': 'RON',
                            'publishing_date': publishing_date.isoformat() if publishing_date else None
                        }
                    )
                    contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting BNR rates: {str(e)}")

        return contents

    async def extract_ecb_rates(self) -> List[ExtractedContent]:
        """Extract exchange rates from ECB (European Central Bank)."""
        contents = []

        try:
            ecb_url = os.getenv('ECB_API_URL',
                'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml')

            async with self.session.get(ecb_url) as response:
                response.raise_for_status()
                xml_data = await response.text()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Extract date
            date_elem = root.find('.//{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}Cube/[@time]')
            if date_elem is not None:
                ecb_date = datetime.strptime(date_elem.get('time'), '%Y-%m-%d')

            # Extract rates
            for rate_elem in root.findall('.//{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}Cube[@rate]'):
                currency = rate_elem.get('currency')
                rate_value = rate_elem.get('rate')

                if currency and rate_value:
                    # Convert to RON using current EUR/RON rate from BNR
                    # For now, use approximate conversion (1 EUR ≈ 4.9 RON)
                    eur_ron_rate = 4.9  # This should come from BNR data
                    ron_rate = float(rate_value) * eur_ron_rate

                    content = ExtractedContent(
                        source_id=f"ecb_{currency}",
                        content=f"ECB Exchange Rate for {currency}: {rate_value} EUR ({ron_rate:.4f} RON)",
                        title=f"{currency}/EUR Exchange Rate (ECB)",
                        url=ecb_url,
                        published_at=ecb_date,
                        metadata={
                            'source': 'ECB',
                            'currency': currency,
                            'rate_eur': float(rate_value),
                            'rate_ron': ron_rate,
                            'base_currency': 'EUR',
                            'date': ecb_date.isoformat() if ecb_date else None
                        }
                    )
                    contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting ECB rates: {str(e)}")

        return contents

    async def extract_crypto_rates(self) -> List[ExtractedContent]:
        """Extract cryptocurrency rates."""
        contents = []

        try:
            api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
            if not api_key:
                logger.warning("CryptoCompare API key not found")
                return contents

            # Get RON rates for major cryptocurrencies
            crypto_url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC,ETH&tsyms=RON&api_key={api_key}"

            async with self.session.get(crypto_url) as response:
                response.raise_for_status()
                data = await response.json()

            for crypto in ['BTC', 'ETH']:
                if crypto in data['RAW'] and 'RON' in data['RAW'][crypto]:
                    crypto_data = data['RAW'][crypto]['RON']

                    content = ExtractedContent(
                        source_id=f"crypto_{crypto}",
                        content=f"Cryptocurrency Rate for {crypto}: {crypto_data['PRICE']} RON",
                        title=f"{crypto}/RON Exchange Rate",
                        url=f"https://www.cryptocompare.com/coins/{crypto.lower()}/overview",
                        published_at=datetime.now(),
                        metadata={
                            'source': 'CryptoCompare',
                            'cryptocurrency': crypto,
                            'rate_ron': crypto_data['PRICE'],
                            'change_24h': crypto_data['CHANGEPCT24HOUR'],
                            'volume_24h': crypto_data['VOLUME24HOUR'],
                            'market_cap': crypto_data['MKTCAP'],
                            'last_update': crypto_data['LASTUPDATE']
                        }
                    )
                    contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting crypto rates: {str(e)}")

        return contents

    async def extract_federal_reserve_data(self) -> List[ExtractedContent]:
        """Extract USD-related economic indicators from Federal Reserve."""
        contents = []

        try:
            # Federal Reserve Economic Data (FRED) API would require API key
            # For now, we'll extract from their public website
            fred_url = "https://www.federalreserve.gov/monetarypolicy.htm"

            async with self.session.get(fred_url) as response:
                response.raise_for_status()
                html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Extract federal funds rate or other key indicators
            # This is a simplified example - real implementation would need proper parsing
            rate_text = soup.get_text()
            fed_rate_match = re.search(r'(\d+\.?\d*)%', rate_text)

            if fed_rate_match:
                fed_rate = fed_rate_match.group(1)

                content = ExtractedContent(
                    source_id="fed_funds_rate",
                    content=f"Federal Funds Rate: {fed_rate}%",
                    title="US Federal Funds Rate",
                    url=fred_url,
                    published_at=datetime.now(),
                    metadata={
                        'source': 'Federal Reserve',
                        'indicator': 'Federal Funds Rate',
                        'rate': float(fed_rate),
                        'currency': 'USD'
                    }
                )
                contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting Federal Reserve data: {str(e)}")

        return contents

    async def extract(self) -> List[ExtractedContent]:
        """Main extraction method."""
        all_contents = []

        # Extract BNR rates
        bnr_rates = await self.extract_bnr_rates()
        all_contents.extend(bnr_rates)

        # Extract ECB rates
        ecb_rates = await self.extract_ecb_rates()
        all_contents.extend(ecb_rates)

        # Extract cryptocurrency rates
        crypto_rates = await self.extract_crypto_rates()
        all_contents.extend(crypto_rates)

        # Extract Federal Reserve data
        fed_data = await self.extract_federal_reserve_data()
        all_contents.extend(fed_data)

        logger.info(f"Extracted {len(all_contents)} FX data points")
        return all_contents

    def _parse_xml_rate(self, rate_text: str) -> Optional[float]:
        """Parse exchange rate from XML text."""
        try:
            return float(rate_text.replace(',', '.'))
        except (ValueError, AttributeError):
            return None
