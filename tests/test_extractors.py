"""
Tests for data extractors.
Tests all extractor components and their functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta

from src.extractors.base import BaseExtractor, ExtractorConfig, ExtractedContent
from src.extractors.news_extractor import NewsExtractor
from src.extractors.twitter_extractor import TwitterExtractor
from src.extractors.fx_extractor import FXExtractor
from src.models import DataSource


class TestExtractorConfig:
    """Test ExtractorConfig functionality."""

    def test_config_creation(self):
        """Test creating extractor configuration."""
        config = ExtractorConfig(
            name="test_extractor",
            update_interval=300,
            timeout=10,
            batch_size=50
        )

        assert config.name == "test_extractor"
        assert config.update_interval == 300
        assert config.timeout == 10
        assert config.batch_size == 50
        assert config.enabled == True

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ExtractorConfig(name="test")

        assert config.update_interval == 3600
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.batch_size == 100
        assert config.enabled == True


class TestExtractedContent:
    """Test ExtractedContent model."""

    def test_content_creation(self):
        """Test creating extracted content."""
        content = ExtractedContent(
            source_id="test_source_123",
            content="Sample financial news content",
            title="Test News Article",
            url="https://example.com/news/1",
            published_at=datetime.now(),
            metadata={"author": "Test Author"}
        )

        assert content.source_id == "test_source_123"
        assert content.content == "Sample financial news content"
        assert content.title == "Test News Article"
        assert content.url == "https://example.com/news/1"
        assert "author" in content.metadata


class TestBaseExtractor:
    """Test BaseExtractor abstract functionality."""

    def test_should_extract_initial(self, db_session):
        """Test initial extraction should run."""
        config = ExtractorConfig(name="test_extractor", update_interval=3600)
        extractor = BaseExtractor(config, db_session)

        assert extractor.should_extract() == True

    def test_should_extract_after_interval(self, db_session):
        """Test extraction after update interval."""
        config = ExtractorConfig(name="test_extractor", update_interval=1)  # 1 second
        extractor = BaseExtractor(config, db_session)
        extractor.last_extraction = datetime.now() - timedelta(seconds=2)

        assert extractor.should_extract() == True

    def test_should_not_extract_too_soon(self, db_session):
        """Test extraction blocked when too soon."""
        config = ExtractorConfig(name="test_extractor", update_interval=3600)
        extractor = BaseExtractor(config, db_session)
        extractor.last_extraction = datetime.now()

        assert extractor.should_extract() == False

    def test_calculate_text_similarity(self, db_session):
        """Test text similarity calculation."""
        config = ExtractorConfig(name="test_extractor")
        extractor = BaseExtractor(config, db_session)

        # Identical texts
        similarity = extractor.calculate_text_similarity("test content", "test content")
        assert similarity == 1.0

        # Completely different texts
        similarity = extractor.calculate_text_similarity("test content", "different content")
        assert similarity < 1.0

        # Empty texts
        similarity = extractor.calculate_text_similarity("", "content")
        assert similarity == 0.0


class TestNewsExtractor:
    """Test NewsExtractor functionality."""

    @pytest.fixture
    def news_extractor(self, db_session):
        """Create NewsExtractor instance for testing."""
        config = ExtractorConfig(name="news_extractor", update_interval=3600)
        return NewsExtractor(config, db_session)

    def test_get_source_info(self, news_extractor):
        """Test getting source information."""
        info = news_extractor.get_source_info()

        assert info["name"] == "Financial News Extractor"
        assert info["type"] == "news"
        assert "sources" in info
        assert len(info["sources"]) > 0

    @pytest.mark.asyncio
    async def test_extract_from_rss_success(self, news_extractor):
        """Test successful RSS extraction."""
        rss_url = "https://www.bursa.ro/rss"

        with patch('requests.get') as mock_get:
            # Mock RSS response
            mock_response = Mock()
            mock_response.content = """
            <?xml version="1.0"?>
            <rss>
                <channel>
                    <title>Test Feed</title>
                    <item>
                        <title>Test Financial News</title>
                        <description>Test content about RON rates</description>
                        <link>https://test.com/news/1</link>
                        <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                    </item>
                </channel>
            </rss>
            """
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with patch('feedparser.parse') as mock_parse:
                mock_parse.return_value = {
                    'feed': {'title': 'Test Feed'},
                    'entries': [{
                        'title': 'Test Financial News',
                        'description': 'Test content about RON rates',
                        'link': 'https://test.com/news/1',
                        'published': 'Mon, 01 Jan 2024 12:00:00 GMT'
                    }]
                }

                contents = await news_extractor.extract_from_rss(rss_url)

                assert len(contents) == 1
                assert contents[0].title == "Test Financial News"
                assert "RON" in contents[0].content

    def test_filter_financial_content(self, news_extractor):
        """Test financial content filtering."""
        # Financial content
        financial_content = ExtractedContent(
            source_id="test",
            content="RON exchange rate increased by 0.5% today",
            title="RON Rate Update"
        )

        # Non-financial content
        non_financial_content = ExtractedContent(
            source_id="test",
            content="Weather forecast for tomorrow",
            title="Weather News"
        )

        contents = [financial_content, non_financial_content]
        filtered = news_extractor._filter_financial_content(contents)

        assert len(filtered) == 1
        assert filtered[0].title == "RON Rate Update"


class TestTwitterExtractor:
    """Test TwitterExtractor functionality."""

    @pytest.fixture
    def twitter_extractor(self, db_session):
        """Create TwitterExtractor instance for testing."""
        config = ExtractorConfig(name="twitter_extractor", update_interval=900)
        return TwitterExtractor(config, db_session)

    def test_get_source_info(self, twitter_extractor):
        """Test getting Twitter source information."""
        info = twitter_extractor.get_source_info()

        assert info["name"] == "Twitter Financial Sentiment Extractor"
        assert info["type"] == "twitter"
        assert "keywords" in info
        assert "accounts" in info

    def test_is_financial_content(self, twitter_extractor):
        """Test financial content detection in tweets."""
        # Financial tweet
        financial_tweet = "RON exchange rate is stable today #BNR #RON"
        assert twitter_extractor._is_financial_content(financial_tweet) == True

        # Non-financial tweet
        non_financial_tweet = "Beautiful weather today in Bucharest!"
        assert twitter_extractor._is_financial_content(non_financial_tweet) == False


class TestFXExtractor:
    """Test FXExtractor functionality."""

    @pytest.fixture
    def fx_extractor(self, db_session):
        """Create FXExtractor instance for testing."""
        config = ExtractorConfig(name="fx_extractor", update_interval=3600)
        return FXExtractor(config, db_session)

    def test_get_source_info(self, fx_extractor):
        """Test getting FX source information."""
        info = fx_extractor.get_source_info()

        assert info["name"] == "Foreign Exchange Rate Extractor"
        assert info["type"] == "fx"
        assert "sources" in info
        assert "currencies" in info
        assert "RON" in info["currencies"]

    @pytest.mark.asyncio
    async def test_extract_bnr_rates_success(self, fx_extractor):
        """Test successful BNR rates extraction."""
        with patch.object(fx_extractor, 'session') as mock_session:
            mock_response = AsyncMock()
            mock_response.text = """
            <?xml version="1.0" encoding="utf-8"?>
            <DataSet>
                <Body>
                    <Cube>
                        <Rate currency="EUR" multiplier="1">4.9750</Rate>
                        <Rate currency="USD" multiplier="1">4.5800</Rate>
                    </Cube>
                </Body>
            </DataSet>
            """
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch('xml.etree.ElementTree.fromstring') as mock_xml:
                # Mock XML parsing
                mock_rate1 = Mock()
                mock_rate1.get.return_value = "EUR"
                mock_rate1.text = "4.9750"
                mock_rate2 = Mock()
                mock_rate2.get.return_value = "USD"
                mock_rate2.text = "4.5800"

                mock_cube = Mock()
                mock_cube.findall.return_value = [mock_rate1, mock_rate2]

                mock_root = Mock()
                mock_root.find.return_value.text = "2024-01-01"
                mock_root.findall.return_value = [mock_cube]

                mock_xml.return_value = mock_root

                contents = await fx_extractor.extract_bnr_rates()

                assert len(contents) == 2
                assert any("EUR" in content.content for content in contents)
                assert any("USD" in content.content for content in contents)
