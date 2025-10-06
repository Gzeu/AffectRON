"""
Tests for database models and aggregators.
Tests data models and aggregation functionality.
"""

import pytest
from datetime import datetime, timedelta

from src.models import Base, DataSource, ExtractedData, SentimentAnalysis, MarketData, AggregatedData
from src.aggregators.base import BaseAggregator, AggregatorConfig, AggregatedResult
from src.aggregators.merge_dedup import MergeDedupAggregator
from src.aggregators.temporal_cluster import TemporalClusteringAggregator


class TestDataSource:
    """Test DataSource model."""

    def test_create_data_source(self, db_session):
        """Test creating a data source."""
        data_source = DataSource(
            name="Test News Source",
            source_type="news",
            url="https://test.com/rss",
            is_active=True,
            extraction_interval=3600
        )

        db_session.add(data_source)
        db_session.commit()

        # Query back
        saved_source = db_session.query(DataSource).filter_by(name="Test News Source").first()

        assert saved_source is not None
        assert saved_source.source_type == "news"
        assert saved_source.url == "https://test.com/rss"
        assert saved_source.is_active == True
        assert saved_source.extraction_interval == 3600

    def test_data_source_relationships(self, db_session):
        """Test data source relationships with extracted data."""
        # Create data source
        data_source = DataSource(
            name="Test Source",
            source_type="test",
            url="https://test.com",
            is_active=True
        )
        db_session.add(data_source)
        db_session.commit()

        # Create extracted data
        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="Test content",
            title="Test title"
        )
        db_session.add(extracted_data)
        db_session.commit()

        # Check relationship
        assert len(data_source.extracted_data) == 1
        assert data_source.extracted_data[0].content == "Test content"


class TestExtractedData:
    """Test ExtractedData model."""

    def test_create_extracted_data(self, db_session):
        """Test creating extracted data."""
        # Create data source first
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="Financial news content about RON rates",
            title="RON Rate Update",
            url="https://example.com/news/1",
            published_at=datetime.now(),
            metadata='{"author": "Test", "category": "finance"}',
            sentiment_score=0.7,
            confidence_score=0.85,
            is_processed=True
        )

        db_session.add(extracted_data)
        db_session.commit()

        # Query back
        saved_data = db_session.query(ExtractedData).filter_by(title="RON Rate Update").first()

        assert saved_data is not None
        assert saved_data.content == "Financial news content about RON rates"
        assert saved_data.sentiment_score == 0.7
        assert saved_data.is_processed == True
        assert saved_data.metadata == '{"author": "Test", "category": "finance"}'


class TestSentimentAnalysis:
    """Test SentimentAnalysis model."""

    def test_create_sentiment_analysis(self, db_session):
        """Test creating sentiment analysis."""
        # Create data source and extracted data first
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="test",
            title="test"
        )
        db_session.add(extracted_data)
        db_session.commit()

        sentiment = SentimentAnalysis(
            data_id=extracted_data.id,
            model_name="finbert_test",
            sentiment_label="positive",
            sentiment_score=0.8,
            confidence_score=0.9,
            entities='{"currencies": ["RON"], "organizations": ["BNR"]}'
        )

        db_session.add(sentiment)
        db_session.commit()

        # Query back
        saved_sentiment = db_session.query(SentimentAnalysis).filter_by(model_name="finbert_test").first()

        assert saved_sentiment is not None
        assert saved_sentiment.sentiment_label == "positive"
        assert saved_sentiment.sentiment_score == 0.8
        assert saved_sentiment.confidence_score == 0.9


class TestMarketData:
    """Test MarketData model."""

    def test_create_market_data(self, db_session):
        """Test creating market data."""
        market_data = MarketData(
            currency_pair="EUR/RON",
            rate=4.9750,
            source="BNR",
            timestamp=datetime.now(),
            metadata='{"volume": 1000000}'
        )

        db_session.add(market_data)
        db_session.commit()

        # Query back
        saved_data = db_session.query(MarketData).filter_by(currency_pair="EUR/RON").first()

        assert saved_data is not None
        assert saved_data.rate == 4.9750
        assert saved_data.source == "BNR"


class TestAggregatorConfig:
    """Test AggregatorConfig functionality."""

    def test_config_creation(self):
        """Test creating aggregator configuration."""
        config = AggregatorConfig(
            name="test_aggregator",
            batch_size=50,
            time_window=timedelta(minutes=30),
            similarity_threshold=0.9
        )

        assert config.name == "test_aggregator"
        assert config.batch_size == 50
        assert config.time_window == timedelta(minutes=30)
        assert config.similarity_threshold == 0.9
        assert config.enabled == True


class TestAggregatedResult:
    """Test AggregatedResult model."""

    def test_result_creation(self):
        """Test creating aggregated result."""
        result_data = {
            "merged_content": "Combined financial news",
            "source_count": 3,
            "sentiment_average": 0.6
        }

        result = AggregatedResult(
            aggregator_name="test_aggregator",
            aggregation_type="merge_dedup",
            data_points=[1, 2, 3],
            result_data=result_data
        )

        assert result.aggregator_name == "test_aggregator"
        assert result.aggregation_type == "merge_dedup"
        assert result.data_points == [1, 2, 3]
        assert result.result_data == result_data


class TestMergeDedupAggregator:
    """Test MergeDedupAggregator functionality."""

    @pytest.fixture
    def merge_aggregator(self, db_session):
        """Create MergeDedupAggregator instance for testing."""
        config = AggregatorConfig(
            name="test_merge_dedup",
            batch_size=10,
            time_window=timedelta(minutes=15)
        )
        return MergeDedupAggregator(config, db_session)

    def test_generate_content_signature(self, merge_aggregator):
        """Test content signature generation."""
        from src.models import ExtractedData

        # Create test data
        data = ExtractedData(
            content="RON rate increased by 0.5%",
            title="RON Update"
        )

        signature = merge_aggregator._generate_content_signature(data)

        assert isinstance(signature, str)
        assert len(signature) == 32  # MD5 hash length

        # Same content should generate same signature
        data2 = ExtractedData(
            content="RON rate increased by 0.5%",
            title="RON Update"
        )
        signature2 = merge_aggregator._generate_content_signature(data2)

        assert signature == signature2

    def test_group_similar_content(self, merge_aggregator, db_session):
        """Test content grouping by similarity."""
        from src.models import ExtractedData, DataSource

        # Create data source
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        # Create similar content
        data1 = ExtractedData(
            source_id=data_source.id,
            content="RON exchange rate increased today",
            title="RON Rate News"
        )
        data2 = ExtractedData(
            source_id=data_source.id,
            content="RON exchange rate rose today",
            title="RON Rate Update"
        )
        data3 = ExtractedData(
            source_id=data_source.id,
            content="Weather is nice today",
            title="Weather News"
        )

        db_session.add_all([data1, data2, data3])
        db_session.commit()

        groups = merge_aggregator._group_similar_content([data1, data2, data3])

        # Should have groups for similar content
        assert len(groups) >= 1  # At least one group for similar RON content

        # Financial content should be grouped together
        ron_groups = [k for k, v in groups.items() if any("RON" in str(data.content) for data in v)]
        assert len(ron_groups) >= 1


class TestTemporalClusteringAggregator:
    """Test TemporalClusteringAggregator functionality."""

    @pytest.fixture
    def temporal_aggregator(self, db_session):
        """Create TemporalClusteringAggregator instance for testing."""
        config = AggregatorConfig(
            name="test_temporal",
            batch_size=20,
            time_window=timedelta(hours=1)
        )
        return TemporalClusteringAggregator(config, db_session)

    def test_create_temporal_clusters(self, temporal_aggregator):
        """Test temporal cluster creation."""
        from src.models import ExtractedData, DataSource

        # Create data source
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        data_source.save()

        # Create data at different times
        base_time = datetime.now()
        data_points = []

        for i in range(5):
            data = ExtractedData(
                source_id=data_source.id,
                content=f"test content {i}",
                title=f"test {i}",
                published_at=base_time + timedelta(minutes=i*10)
            )
            data_points.append(data)

        clusters = temporal_aggregator._create_temporal_clusters(data_points, timedelta(minutes=30))

        # Should create clusters based on time windows
        assert isinstance(clusters, dict)
        assert len(clusters) >= 1  # At least one cluster

    def test_analyze_temporal_cluster(self, temporal_aggregator):
        """Test temporal cluster analysis."""
        from src.models import ExtractedData, DataSource

        # Create test data
        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        data_source.save()

        base_time = datetime.now()
        cluster_items = []

        for i in range(3):
            data = ExtractedData(
                source_id=data_source.id,
                content=f"financial news {i}",
                title=f"news {i}",
                published_at=base_time + timedelta(minutes=i*5)
            )
            cluster_items.append(data)

        analysis = temporal_aggregator._analyze_temporal_cluster(cluster_items, "15_minutes")

        assert "window_name" in analysis
        assert "temporal_statistics" in analysis
        assert "content_analysis" in analysis
        assert analysis["window_name"] == "15_minutes"
        assert analysis["temporal_statistics"]["cluster_size"] == 3

    def test_is_financial_content(self, temporal_aggregator):
        """Test financial content detection."""
        from src.models import ExtractedData

        # Financial content
        financial_data = ExtractedData(
            content="RON exchange rate analysis with EUR comparison",
            title="RON/EUR Analysis"
        )

        # Non-financial content
        non_financial_data = ExtractedData(
            content="Weather forecast for tomorrow",
            title="Weather News"
        )

        assert temporal_aggregator._is_financial_content(financial_data) == True
        assert temporal_aggregator._is_financial_content(non_financial_data) == False


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""

    def test_unique_data_source_name(self, db_session):
        """Test that data source names must be unique."""
        # Create first data source
        ds1 = DataSource(name="test_source", source_type="test", url="test1", is_active=True)
        db_session.add(ds1)
        db_session.commit()

        # Try to create duplicate
        ds2 = DataSource(name="test_source", source_type="test", url="test2", is_active=True)
        db_session.add(ds2)

        with pytest.raises(Exception):  # Should raise integrity error
            db_session.commit()

    def test_cascade_deletion(self, db_session):
        """Test cascade deletion behavior."""
        # Create data source
        data_source = DataSource(name="test_cascade", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        # Create extracted data
        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="test",
            title="test"
        )
        db_session.add(extracted_data)
        db_session.commit()

        # Create sentiment analysis
        sentiment = SentimentAnalysis(
            data_id=extracted_data.id,
            model_name="test",
            sentiment_label="positive",
            sentiment_score=0.5,
            confidence_score=0.8
        )
        db_session.add(sentiment)
        db_session.commit()

        # Delete data source (should cascade)
        db_session.delete(data_source)
        db_session.commit()

        # Check that related data is also deleted
        assert db_session.query(ExtractedData).filter_by(source_id=data_source.id).first() is None
