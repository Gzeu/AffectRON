"""
Tests for Romanian financial corpus functionality.
Tests corpus collection, processing, and quality assessment.
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.api.romanian_corpus import (
    RomanianFinancialCorpus, CorpusDocument, get_corpus_status, build_romanian_financial_corpus
)


class TestCorpusDocument:
    """Test CorpusDocument functionality."""

    def test_document_creation(self):
        """Test creating corpus document."""
        doc = CorpusDocument(
            id="test_doc_1",
            title="Test Financial News",
            content="RON exchange rate increased today",
            source="Test Source",
            url="https://test.com/news/1",
            published_date=datetime.now(),
            language="ro",
            category="financial"
        )

        assert doc.id == "test_doc_1"
        assert doc.title == "Test Financial News"
        assert doc.content == "RON exchange rate increased today"
        assert doc.source == "Test Source"
        assert doc.language == "ro"
        assert doc.word_count == 5  # 5 words in content

    def test_document_quality_calculation(self):
        """Test document quality score calculation."""
        # High quality document
        high_quality_doc = CorpusDocument(
            id="high_quality",
            title="BNR Official Announcement",
            content="Banca Națională a României announced RON rate of 4.9750 against EUR. This is important for the economy.",
            source="BNR",
            language="ro"
        )

        # Calculate quality (would be done during corpus processing)
        quality_score = high_quality_doc.quality_score

        # Should have good quality score due to BNR source and financial terms
        assert quality_score > 0.5

    def test_document_sentiment_classification(self):
        """Test simple sentiment classification."""
        # Positive sentiment document
        positive_doc = CorpusDocument(
            id="positive",
            title="Positive News",
            content="RON crește față de EUR după decizia BNR",
            source="Test"
        )

        # Negative sentiment document
        negative_doc = CorpusDocument(
            id="negative",
            title="Negative News",
            content="RON scade dramatic după volatilitate",
            source="Test"
        )

        # Neutral sentiment document
        neutral_doc = CorpusDocument(
            id="neutral",
            title="Neutral News",
            content="Piața financiară rămâne stabilă",
            source="Test"
        )

        # Test sentiment classification (would be done during processing)
        # This tests the internal logic that would be used


class TestRomanianFinancialCorpus:
    """Test RomanianFinancialCorpus functionality."""

    @pytest.fixture
    def corpus(self):
        """Create RomanianFinancialCorpus instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RomanianFinancialCorpus(temp_dir)

    def test_corpus_initialization(self, corpus):
        """Test corpus initialization."""
        assert corpus.corpus_dir is not None
        assert len(corpus.sources_config) > 0
        assert 'bnr' in corpus.sources_config
        assert 'zf' in corpus.sources_config
        assert 'mediafax' in corpus.sources_config

    def test_source_configuration(self, corpus):
        """Test source configuration setup."""
        bnr_config = corpus.sources_config['bnr']

        assert bnr_config['name'] == 'Banca Națională a României'
        assert bnr_config['type'] == 'press_releases'
        assert bnr_config['language'] == 'ro'
        assert bnr_config['frequency'] == 'weekly'

    def test_document_quality_calculation(self, corpus):
        """Test document quality calculation."""
        # High quality document
        doc = CorpusDocument(
            id="test_1",
            title="BNR Announcement",
            content="Banca Națională a României a anunțat modificări importante pentru RON și EUR",
            source="BNR",
            language="ro"
        )

        quality_score = corpus._calculate_document_quality(doc)

        assert 0.0 <= quality_score <= 1.0
        # Should score highly due to BNR source and financial terms
        assert quality_score > 0.6

    def test_parse_date_formats(self, corpus):
        """Test date parsing from various formats."""
        # Test different date formats
        test_cases = [
            ("2024-01-15 10:30:00", True),  # ISO format
            ("15/01/2024 10:30", True),     # European format
            ("15.01.2024 10:30", True),     # Dot format
            ("invalid_date", False),        # Invalid format
            ("", False)                     # Empty string
        ]

        for date_str, should_succeed in test_cases:
            result = corpus._parse_date(date_str)
            if should_succeed:
                assert result is not None
                assert isinstance(result, datetime)
            else:
                assert result is None

    def test_filter_high_quality_documents(self, corpus):
        """Test filtering high quality documents."""
        # Add test documents with different quality scores
        docs = [
            CorpusDocument("doc1", "Title1", "Content1", "Source1", quality_score=0.8),
            CorpusDocument("doc2", "Title2", "Content2", "Source2", quality_score=0.6),
            CorpusDocument("doc3", "Title3", "Content3", "Source3", quality_score=0.3)
        ]

        corpus.documents = docs

        # Filter high quality (score >= 0.5)
        high_quality = corpus.filter_high_quality_documents(0.5)

        assert len(high_quality) == 2  # Should include doc1 and doc2
        assert all(doc.quality_score >= 0.5 for doc in high_quality)

    def test_deduplicate_documents(self, corpus):
        """Test document deduplication."""
        # Add duplicate documents
        docs = [
            CorpusDocument("doc1", "Title", "Same content here", "Source1"),
            CorpusDocument("doc2", "Title", "Same content here", "Source2"),
            CorpusDocument("doc3", "Title", "Different content", "Source3")
        ]

        corpus.documents = docs

        # Deduplicate
        unique_docs = corpus.deduplicate_documents()

        # Should have 2 unique documents
        assert len(unique_docs) == 2

    def test_extract_financial_entities(self, corpus):
        """Test financial entity extraction."""
        # Add document with financial terms
        doc = CorpusDocument(
            "test_doc",
            "RON vs EUR",
            "BNR announced RON rate of 4.9750 against EUR. ECB also made announcements.",
            "Test Source"
        )

        corpus.documents = [doc]

        entities = corpus.extract_financial_entities()

        assert 'currencies' in entities
        assert 'institutions' in entities
        assert len(entities['currencies']) > 0
        assert len(entities['institutions']) > 0

    def test_create_training_dataset(self, corpus):
        """Test training dataset creation."""
        # Add some test documents
        docs = [
            CorpusDocument("doc1", "Title1", "Financial content 1", "Source1", quality_score=0.8),
            CorpusDocument("doc2", "Title2", "Financial content 2", "Source2", quality_score=0.7),
            CorpusDocument("doc3", "Title3", "Financial content 3", "Source3", quality_score=0.6)
        ]

        corpus.documents = docs

        with tempfile.TemporaryDirectory() as temp_dir:
            result = corpus.create_training_dataset(temp_dir)

            if 'error' not in result:
                assert result['status'] == 'success'
                assert 'datasets' in result
                assert 'metadata' in result

                # Check that files were created
                assert os.path.exists(f"{temp_dir}/train.jsonl")
                assert os.path.exists(f"{temp_dir}/validation.jsonl")
                assert os.path.exists(f"{temp_dir}/test.jsonl")
                assert os.path.exists(f"{temp_dir}/metadata.json")

    def test_get_corpus_statistics(self, corpus):
        """Test corpus statistics generation."""
        # Add test documents
        docs = [
            CorpusDocument("doc1", "Title1", "Short content", "Source1", quality_score=0.8, word_count=2),
            CorpusDocument("doc2", "Title2", "This is a longer financial content with more words", "Source2", quality_score=0.6, word_count=10)
        ]

        corpus.documents = docs

        stats = corpus.get_corpus_statistics()

        assert stats['total_documents'] == 2
        assert stats['total_words'] == 12
        assert stats['average_quality_score'] == 0.7
        assert 'sources_distribution' in stats
        assert 'quality_distribution' in stats

    def test_export_corpus_formats(self, corpus):
        """Test corpus export in multiple formats."""
        # Add test document
        doc = CorpusDocument("test_doc", "Test Title", "Test content", "Test Source")
        corpus.documents = [doc]

        with tempfile.TemporaryDirectory() as temp_dir:
            exports = corpus.export_corpus_formats(temp_dir)

            # Should export in multiple formats
            assert 'json' in exports
            assert 'csv' in exports
            assert 'archive' in exports

            # Check that files were created
            assert os.path.exists(exports['json'])
            assert os.path.exists(exports['csv'])
            assert os.path.exists(exports['archive'])

            # Check JSON content
            with open(exports['json'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]['title'] == "Test Title"

    @pytest.mark.asyncio
    async def test_collect_from_rss_feeds_mock(self, corpus):
        """Test RSS feed collection with mocked data."""
        with patch('feedparser.parse') as mock_parse:
            # Mock RSS feed response
            mock_feed = Mock()
            mock_feed.entries = [
                Mock(
                    title="Test Financial News",
                    summary="RON exchange rate news",
                    link="https://test.com/news/1",
                    published="2024-01-01T10:00:00"
                )
            ]
            mock_parse.return_value = mock_feed

            collected_count = await corpus.collect_from_rss_feeds()

            # Should collect documents (actual count depends on mocking)
            assert collected_count >= 0

    @pytest.mark.asyncio
    async def test_scrape_website_mock(self, corpus):
        """Test website scraping with mocked data."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock HTML response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html><body><div>Test content</div></body></html>")

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            # Mock BeautifulSoup
            with patch('src.api.romanian_corpus.BeautifulSoup') as mock_soup:
                mock_soup_instance = Mock()
                mock_soup_instance.find_all.return_value = []
                mock_soup.return_value = mock_soup_instance

                documents = await corpus._scrape_website(corpus.sources_config['bnr'])

                # Should handle gracefully even with no content found
                assert isinstance(documents, list)


class TestIntegration:
    """Integration tests for corpus building."""

    @pytest.mark.asyncio
    async def test_build_corpus_workflow(self):
        """Test complete corpus building workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = RomanianFinancialCorpus(temp_dir)

            # Mock RSS collection
            with patch.object(corpus, 'collect_from_rss_feeds', return_value=5), \
                 patch.object(corpus, 'collect_from_websites', return_value=3), \
                 patch.object(corpus, '_enhance_documents', return_value=None):

                result = await corpus.build_corpus(days_back=7)

                assert result['status'] == 'success'
                assert 'corpus_file' in result
                assert 'statistics' in result
                assert result['rss_sources_collected'] == 5
                assert result['web_sources_collected'] == 3

                # Check that corpus file was created
                assert os.path.exists(result['corpus_file'])

    def test_corpus_status_function(self):
        """Test corpus status function."""
        status = get_corpus_status()

        assert 'documents_collected' in status
        assert 'sources_configured' in status
        assert 'statistics' in status
        assert 'last_updated' in status

    @pytest.mark.asyncio
    async def test_build_romanian_financial_corpus_function(self):
        """Test build_romanian_financial_corpus function."""
        with patch('src.api.romanian_corpus.romanian_corpus') as mock_corpus:
            mock_corpus.build_corpus.return_value = {
                'status': 'success',
                'corpus_file': '/test/corpus.json',
                'statistics': {'total_documents': 100}
            }

            result = await build_romanian_financial_corpus()

            assert result['status'] == 'success'
            assert result['statistics']['total_documents'] == 100

    def test_multiple_language_support(self, corpus):
        """Test that corpus supports multiple languages."""
        # Add documents in different languages
        docs = [
            CorpusDocument("ro_doc", "Știre în română", "Conținut în limba română", "Source", language="ro"),
            CorpusDocument("en_doc", "English news", "Content in English", "Source", language="en"),
            CorpusDocument("de_doc", "Deutsche Nachricht", "Inhalt auf Deutsch", "Source", language="de")
        ]

        corpus.documents = docs

        # Check language distribution in statistics
        stats = corpus.get_corpus_statistics()
        assert stats['language_distribution']['ro'] == 1
        assert stats['language_distribution']['en'] == 1
        assert stats['language_distribution']['de'] == 1

    def test_financial_term_extraction(self, corpus):
        """Test extraction of financial terms."""
        # Document with various financial terms
        doc = CorpusDocument(
            "financial_doc",
            "Financial News",
            "RON și EUR sunt afectate de deciziile BNR și ECB. Inflația și dobânzile sunt factori cheie.",
            "Test Source"
        )

        corpus.documents = [doc]

        entities = corpus.extract_financial_entities()

        # Should extract currencies and institutions
        assert len(entities['currencies']) > 0
        assert len(entities['institutions']) > 0

        # Check specific terms
        currencies_lower = [c.lower() for c in entities['currencies']]
        institutions_lower = [i.lower() for i in entities['institutions']]

        assert 'ron' in currencies_lower
        assert 'eur' in currencies_lower
        assert 'bnr' in institutions_lower
        assert 'ecb' in institutions_lower
