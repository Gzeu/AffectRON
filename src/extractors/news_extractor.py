"""
News extractor for financial news sources.
Extracts news articles from Romanian financial media and international sources.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
import feedparser
from fake_useragent import UserAgent

from .base import BaseExtractor, ExtractorConfig, ExtractedContent


logger = logging.getLogger(__name__)


class NewsExtractor(BaseExtractor):
    """Extractor for financial news from various sources."""

    def __init__(self, config: ExtractorConfig, db_session):
        super().__init__(config, db_session)
        self.ua = UserAgent()

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about news sources."""
        return {
            "name": "Financial News Extractor",
            "type": "news",
            "sources": [
                "https://www.bursa.ro/rss",
                "https://www.profit.ro/rss",
                "https://www.zf.ro/rss",
                "https://www.economica.net/rss",
                "https://www.cursbnr.ro/rss"
            ],
            "update_interval": self.config.update_interval
        }

    async def extract_from_rss(self, rss_url: str) -> List[ExtractedContent]:
        """Extract news from RSS feeds."""
        contents = []

        try:
            # Use requests for RSS parsing (more reliable than aiohttp for XML)
            import requests
            response = requests.get(rss_url, timeout=self.config.timeout)
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            for entry in feed.entries[:self.config.batch_size]:
                content = ExtractedContent(
                    source_id=f"rss_{urlparse(rss_url).netloc}",
                    content=entry.get('summary', entry.get('description', '')),
                    title=entry.get('title', ''),
                    url=entry.get('link', ''),
                    published_at=self._parse_datetime(entry.get('published', '')),
                    metadata={
                        'feed_title': feed.feed.get('title', ''),
                        'author': entry.get('author', ''),
                        'tags': [tag.term for tag in entry.get('tags', [])]
                    }
                )
                contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting from RSS {rss_url}: {str(e)}")

        return contents

    async def extract_from_website(self, url: str, selectors: Dict[str, str]) -> List[ExtractedContent]:
        """Extract news from website HTML."""
        contents = []

        try:
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ro,en-US;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Extract articles based on CSS selectors
            articles = soup.select(selectors.get('article', 'article'))

            for article in articles[:self.config.batch_size]:
                title_elem = article.select_one(selectors.get('title', 'h1, h2, h3'))
                content_elem = article.select_one(selectors.get('content', 'p'))
                url_elem = article.select_one(selectors.get('url', 'a'))

                if title_elem and content_elem:
                    content = ExtractedContent(
                        source_id=f"web_{urlparse(url).netloc}",
                        content=content_elem.get_text(strip=True),
                        title=title_elem.get_text(strip=True),
                        url=url_elem.get('href') if url_elem else url,
                        published_at=datetime.now(),  # Websites often don't have dates
                        metadata={
                            'extraction_method': 'website_scraping',
                            'article_tags': [tag.get('content') for tag in article.select('meta[property="article:tag"]')]
                        }
                    )
                    contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting from website {url}: {str(e)}")

        return contents

    async def extract(self) -> List[ExtractedContent]:
        """Main extraction method."""
        all_contents = []

        # RSS sources
        rss_sources = [
            "https://www.bursa.ro/rss",
            "https://www.profit.ro/rss",
            "https://www.zf.ro/rss",
            "https://www.economica.net/rss",
            "https://www.cursbnr.ro/rss"
        ]

        # Website sources with selectors
        website_sources = {
            "https://www.hotnews.ro/economie": {
                'article': '.article',
                'title': 'h1',
                'content': '.article-content p',
                'url': 'a'
            },
            "https://www.mediafax.ro/economic": {
                'article': '.article-item',
                'title': 'h2',
                'content': 'p',
                'url': 'a'
            }
        }

        # Extract from RSS feeds
        for rss_url in rss_sources:
            contents = await self.extract_from_rss(rss_url)
            all_contents.extend(contents)

        # Extract from websites
        for url, selectors in website_sources.items():
            contents = await self.extract_from_website(url, selectors)
            all_contents.extend(contents)

        # Filter for financial content
        filtered_contents = self._filter_financial_content(all_contents)

        logger.info(f"Extracted {len(filtered_contents)} financial news articles")
        return filtered_contents

    def _filter_financial_content(self, contents: List[ExtractedContent]) -> List[ExtractedContent]:
        """Filter content to keep only financial/economic news."""
        financial_keywords = [
            'ron', 'eur', 'usd', 'leu', 'euro', 'dolar',
            'bursa', 'actiuni', 'obligatiuni', 'investitii',
            'banca', 'banc', 'moneda', 'curs valutar',
            'piata', 'tranzactie', 'profit', 'pierdere',
            'dobanda', 'rata', 'credit', 'depozit',
            'BNR', 'Banca Nationala', 'guvern', 'minister',
            'economie', 'economic', 'financiar', 'fiscal',
            'buget', 'deficit', 'inflatie', 'crestere',
            'scadere', 'evolutie', 'previziune'
        ]

        filtered = []
        for content in contents:
            text_to_check = f"{content.title} {content.content}".lower()
            if any(keyword in text_to_check for keyword in financial_keywords):
                filtered.append(content)

        return filtered

    def _parse_datetime(self, date_string: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        if not date_string:
            return None

        # Common RSS date formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_string}")
        return None
