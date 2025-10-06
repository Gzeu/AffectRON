"""
Romanian Financial Corpus Builder.
Collects, processes, and manages Romanian financial text data for model training.
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import csv
import zipfile

import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class CorpusDocument:
    """Document in the financial corpus."""
    id: str
    title: str
    content: str
    source: str
    url: Optional[str] = None
    published_date: Optional[datetime] = None
    language: str = "ro"
    category: str = "financial"
    sentiment_label: Optional[str] = None
    entities: Dict[str, List[str]] = None

    # Metadata
    word_count: int = 0
    quality_score: float = 0.0
    processed: bool = False


class RomanianFinancialCorpus:
    """Romanian financial corpus management system."""

    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = corpus_dir
        self.documents: List[CorpusDocument] = []
        self.sources_config = {}

        # Ensure corpus directory exists
        os.makedirs(corpus_dir, exist_ok=True)

        # Romanian financial sources
        self._setup_romanian_sources()

        self.logger = logging.getLogger(__name__)

    def _setup_romanian_sources(self):
        """Set up Romanian financial data sources."""
        self.sources_config = {
            'bnr': {
                'name': 'Banca Națională a României',
                'url': 'https://www.bnr.ro/Comunicate-de-presa-627.aspx',
                'type': 'press_releases',
                'language': 'ro',
                'frequency': 'weekly'
            },
            'bvb': {
                'name': 'Bursa de Valori București',
                'url': 'https://www.bvb.ro/FinancialInstruments/Details/',
                'type': 'market_data',
                'language': 'ro',
                'frequency': 'daily'
            },
            'mediafax': {
                'name': 'Mediafax Economic',
                'url': 'https://www.mediafax.ro/economic/',
                'type': 'news',
                'language': 'ro',
                'frequency': 'hourly'
            },
            'zf': {
                'name': 'Ziarul Financiar',
                'url': 'https://www.zf.ro/rss/zf-24/',
                'type': 'news',
                'language': 'ro',
                'frequency': 'hourly'
            },
            'economica': {
                'name': 'Economica.net',
                'url': 'https://www.economica.net/rss/',
                'type': 'news',
                'language': 'ro',
                'frequency': 'hourly'
            },
            'cursbnr': {
                'name': 'Curs BNR',
                'url': 'https://www.cursbnr.ro/',
                'type': 'exchange_rates',
                'language': 'ro',
                'frequency': 'daily'
            }
        }

    async def collect_from_rss_feeds(self) -> int:
        """Collect documents from RSS feeds."""
        collected_count = 0

        for source_id, config in self.sources_config.items():
            try:
                if config['type'] == 'news' and 'rss' in config['url']:
                    documents = await self._collect_from_rss(config)
                    self.documents.extend(documents)
                    collected_count += len(documents)

                    self.logger.info(f"Collected {len(documents)} documents from {source_id}")

            except Exception as e:
                self.logger.error(f"Error collecting from {source_id}: {e}")

        return collected_count

    async def _collect_from_rss(self, config: Dict[str, Any]) -> List[CorpusDocument]:
        """Collect documents from a single RSS feed."""
        documents = []

        try:
            import feedparser

            feed = feedparser.parse(config['url'])

            for entry in feed.entries[:50]:  # Limit to recent 50 entries
                try:
                    # Extract content
                    title = entry.get('title', '')
                    content = entry.get('summary', '') or entry.get('description', '')

                    # Skip if too short
                    if len(title) < 10 or len(content) < 50:
                        continue

                    # Create document
                    doc_id = f"{config['name'].lower().replace(' ', '_')}_{hash(title + content) % 1000000}"

                    document = CorpusDocument(
                        id=doc_id,
                        title=title,
                        content=content,
                        source=config['name'],
                        url=entry.get('link'),
                        published_date=self._parse_date(entry.get('published', '')),
                        language=config['language'],
                        category='financial'
                    )

                    # Calculate quality score
                    document.quality_score = self._calculate_document_quality(document)
                    document.word_count = len(content.split())

                    documents.append(document)

                except Exception as e:
                    self.logger.warning(f"Error processing RSS entry: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error parsing RSS feed {config['url']}: {e}")

        return documents

    async def collect_from_websites(self) -> int:
        """Collect documents by scraping financial websites."""
        collected_count = 0

        for source_id, config in self.sources_config.items():
            try:
                if config['type'] in ['press_releases', 'market_data']:
                    documents = await self._scrape_website(config)
                    self.documents.extend(documents)
                    collected_count += len(documents)

                    self.logger.info(f"Scraped {len(documents)} documents from {source_id}")

            except Exception as e:
                self.logger.error(f"Error scraping {source_id}: {e}")

        return collected_count

    async def _scrape_website(self, config: Dict[str, Any]) -> List[CorpusDocument]:
        """Scrape documents from a website."""
        documents = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config['url']) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Extract articles based on source
                        if 'bnr' in config['url']:
                            articles = self._extract_bnr_articles(soup)
                        elif 'bvb' in config['url']:
                            articles = self._extract_bvb_articles(soup)
                        else:
                            articles = self._extract_generic_articles(soup)

                        for article in articles[:20]:  # Limit articles
                            if len(article.get('content', '')) > 100:
                                document = CorpusDocument(
                                    id=f"{config['name'].lower().replace(' ', '_')}_{len(documents)}",
                                    title=article.get('title', ''),
                                    content=article.get('content', ''),
                                    source=config['name'],
                                    url=article.get('url'),
                                    published_date=article.get('date'),
                                    language=config['language']
                                )

                                document.quality_score = self._calculate_document_quality(document)
                                document.word_count = len(document.content.split())

                                documents.append(document)

        except Exception as e:
            self.logger.error(f"Error scraping website {config['url']}: {e}")

        return documents

    def _extract_bnr_articles(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract articles from BNR website."""
        articles = []

        # Look for press releases or news items
        for item in soup.find_all(['div', 'article'], class_=lambda x: x and ('comunicat' in x.lower() or 'press' in x.lower() or 'news' in x.lower())):
            title_elem = item.find(['h1', 'h2', 'h3', 'h4'])
            content_elem = item.find(['div', 'p'], class_=lambda x: x and ('content' in x.lower() or 'text' in x.lower()))

            if title_elem and content_elem:
                articles.append({
                    'title': title_elem.get_text(strip=True),
                    'content': content_elem.get_text(strip=True),
                    'url': item.find('a', href=True)['href'] if item.find('a', href=True) else None,
                    'date': None  # Would need date extraction logic
                })

        return articles

    def _extract_bvb_articles(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract articles from BVB website."""
        articles = []

        # Look for market reports or announcements
        for item in soup.find_all(['div', 'article'], class_=lambda x: x and ('report' in x.lower() or 'announcement' in x.lower())):
            title_elem = item.find(['h1', 'h2', 'h3'])
            content_elem = item.find(['div', 'p'])

            if title_elem and content_elem:
                articles.append({
                    'title': title_elem.get_text(strip=True),
                    'content': content_elem.get_text(strip=True),
                    'url': None,
                    'date': None
                })

        return articles

    def _extract_generic_articles(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract articles from generic financial websites."""
        articles = []

        # Look for article elements
        for article in soup.find_all('article'):
            title_elem = article.find(['h1', 'h2', 'h3'])
            content_elem = article.find(['div', 'p'], class_=lambda x: x and ('content' in x.lower() or 'entry' in x.lower()))

            if title_elem and content_elem:
                articles.append({
                    'title': title_elem.get_text(strip=True),
                    'content': content_elem.get_text(strip=True),
                    'url': article.find('a', href=True)['href'] if article.find('a', href=True) else None,
                    'date': None
                })

        return articles

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_string:
            return None

        # Common Romanian date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%d.%m.%Y %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%d.%m.%Y'
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        return None

    def _calculate_document_quality(self, document: CorpusDocument) -> float:
        """Calculate quality score for document."""
        score = 0.0

        # Length score (prefer medium-length documents)
        word_count = len(document.content.split())
        if 100 <= word_count <= 1000:
            score += 0.3
        elif word_count > 50:
            score += 0.1

        # Financial relevance score
        financial_terms = [
            'ron', 'eur', 'usd', 'leu', 'banca', 'bursă', 'acțiuni', 'obligațiuni',
            'dobândă', 'inflație', 'pib', 'economie', 'financiar', 'monetar'
        ]

        term_count = sum(1 for term in financial_terms if term.lower() in document.content.lower())
        financial_score = min(term_count / 10, 1.0)  # Normalize to 0-1
        score += financial_score * 0.4

        # Source reliability score
        reliable_sources = ['bnr', 'bvb', 'mediafax', 'zf', 'economica']
        if any(source in document.source.lower() for source in reliable_sources):
            score += 0.3

        return min(score, 1.0)

    def filter_high_quality_documents(self, min_quality_score: float = 0.5) -> List[CorpusDocument]:
        """Filter documents by quality score."""
        return [doc for doc in self.documents if doc.quality_score >= min_quality_score]

    def deduplicate_documents(self, similarity_threshold: float = 0.8) -> List[CorpusDocument]:
        """Remove duplicate or very similar documents."""
        if not self.documents:
            return []

        # Simple deduplication based on content hash
        seen_hashes = set()
        unique_documents = []

        for doc in self.documents:
            # Create content hash
            content_hash = hash(doc.content) % 1000000

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)

        return unique_documents

    def extract_financial_entities(self) -> Dict[str, Set[str]]:
        """Extract financial entities from corpus."""
        entities = {
            'currencies': set(),
            'institutions': set(),
            'companies': set(),
            'economic_terms': set()
        }

        currency_patterns = [
            r'\bRON\b', r'\bEUR\b', r'\bUSD\b', r'\bGBP\b', r'\bCHF\b',
            r'\bleu\b', r'\beuro\b', r'\bdolar\b'
        ]

        institution_patterns = [
            r'\bBNR\b', r'\bBanca Națională\b', r'\bBVB\b', r'\bBursa de Valori\b',
            r'\bMinisterul Finanțelor\b', r'\bECB\b', r'\bFed\b', r'\bFederal Reserve\b'
        ]

        for doc in self.documents:
            content_lower = doc.content.lower()

            # Extract currencies
            for pattern in currency_patterns:
                if re.search(pattern, content_lower):
                    entities['currencies'].add(pattern.replace('\\b', '').replace('\\B', ''))

            # Extract institutions
            for pattern in institution_patterns:
                if re.search(pattern.lower(), content_lower):
                    entities['institutions'].add(pattern.replace('\\b', '').replace('\\B', ''))

        return entities

    def create_training_dataset(self, output_dir: str = "data/training") -> Dict[str, Any]:
        """Create training dataset from corpus."""
        os.makedirs(output_dir, exist_ok=True)

        # Filter high-quality documents
        high_quality_docs = self.filter_high_quality_documents(0.6)
        unique_docs = self.deduplicate_documents()

        # Combine filters
        training_docs = [doc for doc in high_quality_docs if doc in unique_docs]

        # Split into train/validation/test
        if len(training_docs) < 10:
            return {'error': 'Insufficient high-quality documents for training'}

        # 80% train, 10% validation, 10% test
        train_docs, temp_docs = train_test_split(training_docs, test_size=0.2, random_state=42)
        val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

        # Save datasets
        datasets = {
            'train': self._save_dataset_split(train_docs, f"{output_dir}/train.jsonl"),
            'validation': self._save_dataset_split(val_docs, f"{output_dir}/validation.jsonl"),
            'test': self._save_dataset_split(test_docs, f"{output_dir}/test.jsonl")
        }

        # Create metadata
        metadata = {
            'total_documents': len(training_docs),
            'train_size': len(train_docs),
            'validation_size': len(val_docs),
            'test_size': len(test_docs),
            'created_at': datetime.now().isoformat(),
            'sources': list(set(doc.source for doc in training_docs)),
            'quality_threshold': 0.6
        }

        with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            'status': 'success',
            'datasets': datasets,
            'metadata': metadata
        }

    def _save_dataset_split(self, documents: List[CorpusDocument], filepath: str) -> Dict[str, Any]:
        """Save dataset split to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                # Create training example
                example = {
                    'id': doc.id,
                    'text': doc.title + " " + doc.content,
                    'source': doc.source,
                    'quality_score': doc.quality_score,
                    'word_count': doc.word_count,
                    'category': doc.category
                }

                # Add sentiment label if available
                if doc.sentiment_label:
                    example['sentiment'] = doc.sentiment_label

                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        return {
            'filepath': filepath,
            'document_count': len(documents),
            'total_words': sum(doc.word_count for doc in documents)
        }

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        if not self.documents:
            return {'total_documents': 0}

        # Basic statistics
        total_docs = len(self.documents)
        total_words = sum(doc.word_count for doc in self.documents)
        avg_quality = np.mean([doc.quality_score for doc in self.documents])

        # Source distribution
        sources = {}
        for doc in self.documents:
            sources[doc.source] = sources.get(doc.source, 0) + 1

        # Quality distribution
        quality_ranges = {
            'high': len([doc for doc in self.documents if doc.quality_score >= 0.8]),
            'medium': len([doc for doc in self.documents if 0.5 <= doc.quality_score < 0.8]),
            'low': len([doc for doc in self.documents if doc.quality_score < 0.5])
        }

        # Language distribution
        languages = {}
        for doc in self.documents:
            languages[doc.language] = languages.get(doc.language, 0) + 1

        return {
            'total_documents': total_docs,
            'total_words': total_words,
            'average_quality_score': avg_quality,
            'sources_distribution': sources,
            'quality_distribution': quality_ranges,
            'language_distribution': languages,
            'collection_date': datetime.now().isoformat()
        }

    async def build_corpus(self, days_back: int = 30) -> Dict[str, Any]:
        """Build complete financial corpus."""
        self.logger.info("Building Romanian financial corpus...")

        # Clear existing documents
        self.documents = []

        # Collect from RSS feeds
        rss_count = await self.collect_from_rss_feeds()

        # Collect from websites
        web_count = await self.collect_from_websites()

        # Process and enhance documents
        await self._enhance_documents()

        # Save corpus
        corpus_file = f"{self.corpus_dir}/corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content,
                    'source': doc.source,
                    'url': doc.url,
                    'published_date': doc.published_date.isoformat() if doc.published_date else None,
                    'language': doc.language,
                    'category': doc.category,
                    'quality_score': doc.quality_score,
                    'word_count': doc.word_count
                }
                for doc in self.documents
            ], f, ensure_ascii=False, indent=2)

        # Get statistics
        stats = self.get_corpus_statistics()

        self.logger.info(f"Corpus built with {stats['total_documents']} documents")

        return {
            'status': 'success',
            'corpus_file': corpus_file,
            'statistics': stats,
            'rss_sources_collected': rss_count,
            'web_sources_collected': web_count
        }

    async def _enhance_documents(self):
        """Enhance documents with additional processing."""
        for doc in self.documents:
            # Extract entities if not already done
            if not doc.entities:
                doc.entities = self._extract_basic_entities(doc.content)

            # Set sentiment label if not already done (simplified)
            if not doc.sentiment_label:
                doc.sentiment_label = self._classify_sentiment_simple(doc.content)

    def _extract_basic_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic financial entities from text."""
        entities = {
            'currencies': [],
            'institutions': [],
            'companies': []
        }

        text_lower = text.lower()

        # Currency extraction
        currencies = ['ron', 'eur', 'usd', 'gbp', 'chf', 'jpy', 'btc', 'eth']
        for currency in currencies:
            if currency in text_lower:
                entities['currencies'].append(currency.upper())

        # Institution extraction
        institutions = ['bnr', 'bvb', 'ecb', 'fed', 'ministerul finanțelor']
        for institution in institutions:
            if institution in text_lower:
                entities['institutions'].append(institution.upper())

        return entities

    def _classify_sentiment_simple(self, text: str) -> str:
        """Simple sentiment classification."""
        text_lower = text.lower()

        positive_words = ['crește', 'creștere', 'profit', 'câștig', 'bullish', 'pozitiv', 'stabil']
        negative_words = ['scade', 'scădere', 'pierdere', 'pierderi', 'bearish', 'negativ', 'volatil']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def export_corpus_formats(self, output_dir: str = "data/exports") -> Dict[str, str]:
        """Export corpus in various formats."""
        os.makedirs(output_dir, exist_ok=True)

        exports = {}

        # JSON export
        json_file = f"{output_dir}/corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content,
                    'source': doc.source,
                    'metadata': {
                        'quality_score': doc.quality_score,
                        'word_count': doc.word_count,
                        'entities': doc.entities
                    }
                }
                for doc in self.documents
            ], f, ensure_ascii=False, indent=2)
        exports['json'] = json_file

        # CSV export
        csv_file = f"{output_dir}/corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'title', 'content', 'source', 'quality_score', 'word_count'])

            for doc in self.documents:
                writer.writerow([
                    doc.id,
                    doc.title,
                    doc.content[:500] + '...' if len(doc.content) > 500 else doc.content,
                    doc.source,
                    doc.quality_score,
                    doc.word_count
                ])
        exports['csv'] = csv_file

        # Compressed archive
        archive_file = f"{output_dir}/corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_file, os.path.basename(json_file))
            zipf.write(csv_file, os.path.basename(csv_file))
        exports['archive'] = archive_file

        return exports


# Global corpus instance
romanian_corpus = RomanianFinancialCorpus()


async def build_romanian_financial_corpus() -> Dict[str, Any]:
    """Build Romanian financial corpus."""
    return await romanian_corpus.build_corpus(days_back=30)


def get_corpus_status() -> Dict[str, Any]:
    """Get corpus building status."""
    return {
        'documents_collected': len(romanian_corpus.documents),
        'sources_configured': len(romanian_corpus.sources_config),
        'statistics': romanian_corpus.get_corpus_statistics(),
        'last_updated': datetime.now().isoformat()
    }
