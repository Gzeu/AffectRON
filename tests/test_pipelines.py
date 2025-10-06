"""
Tests for AI processing pipelines.
Tests sentiment analysis, NER, and trend analysis pipelines.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json

from src.pipelines.base import BasePipeline, PipelineConfig, PipelineResult
from src.pipelines.sentiment_pipeline import SentimentPipeline
from src.pipelines.ner_pipeline import NERPipeline
from src.pipelines.trend_analyzer import TrendAnalyzer


class TestPipelineConfig:
    """Test PipelineConfig functionality."""

    def test_config_creation(self):
        """Test creating pipeline configuration."""
        config = PipelineConfig(
            name="test_pipeline",
            model_path="test/model",
            batch_size=16,
            max_length=256
        )

        assert config.name == "test_pipeline"
        assert config.model_path == "test/model"
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.enabled == True

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PipelineConfig(name="test", model_path="test/model")

        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.enabled == True


class TestPipelineResult:
    """Test PipelineResult model."""

    def test_result_creation(self):
        """Test creating pipeline result."""
        result = PipelineResult(
            data_id=123,
            pipeline_name="test_pipeline",
            results={"sentiment": {"label": "positive", "score": 0.7}},
            confidence=0.85
        )

        assert result.data_id == 123
        assert result.pipeline_name == "test_pipeline"
        assert result.results["sentiment"]["label"] == "positive"
        assert result.confidence == 0.85


class TestSentimentPipeline:
    """Test SentimentPipeline functionality."""

    @pytest.fixture
    def sentiment_pipeline(self, db_session):
        """Create SentimentPipeline instance for testing."""
        config = PipelineConfig(
            name="test_sentiment",
            model_path="ProsusAI/finbert",
            batch_size=8
        )
        return SentimentPipeline(config, db_session)

    def test_preprocess_text(self, sentiment_pipeline):
        """Test text preprocessing."""
        # Test URL removal
        text_with_url = "Check this link: https://example.com/news for more info"
        processed = sentiment_pipeline.preprocess_text(text_with_url)
        assert "https://" not in processed

        # Test mention removal
        text_with_mention = "Thanks @username for the tip"
        processed = sentiment_pipeline.preprocess_text(text_with_mention)
        assert "@username" not in processed

        # Test length truncation
        long_text = "A" * 1000
        processed = sentiment_pipeline.preprocess_text(long_text)
        assert len(processed) < 1000

    def test_extract_financial_context(self, sentiment_pipeline):
        """Test financial context extraction."""
        text = "RON exchange rate increased today EUR/USD also rose"
        context = sentiment_pipeline.extract_financial_context(text)

        assert context["has_financial_terms"] == True
        assert "RON" in context["mentioned_currencies"]
        assert "EUR" in context["mentioned_currencies"]

    def test_adjust_sentiment_for_context(self, sentiment_pipeline):
        """Test sentiment adjustment based on context."""
        # Test with financial terms
        context = {
            "has_financial_terms": True,
            "mentioned_currencies": ["RON"],
            "sentiment_indicators": {"positive": ["growth"], "negative": []}
        }

        adjusted_label, adjusted_score = sentiment_pipeline._adjust_sentiment_for_context(
            "neutral", 0.6, context
        )

        # Should boost confidence for financial content
        assert adjusted_score >= 0.6

    @pytest.mark.asyncio
    async def test_process_batch_mock(self, sentiment_pipeline):
        """Test batch processing with mocked model."""
        texts = [
            "RON exchange rate is stable",
            "EUR/USD volatility increased significantly"
        ]

        # Mock the sentiment pipeline
        with patch.object(sentiment_pipeline, 'sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = [
                [{"label": "LABEL_1", "score": 0.8}],  # neutral
                [{"label": "LABEL_0", "score": 0.9}]   # negative
            ]

            results = await sentiment_pipeline.process_batch(texts)

            assert len(results) == 2
            assert "sentiment" in results[0]
            assert "entities" in results[0]
            assert results[0]["sentiment"]["label"] in ["positive", "negative", "neutral"]


class TestNERPipeline:
    """Test NERPipeline functionality."""

    @pytest.fixture
    def ner_pipeline(self, db_session):
        """Create NERPipeline instance for testing."""
        config = PipelineConfig(
            name="test_ner",
            model_path="test/ner/model"
        )
        return NERPipeline(config, db_session)

    def test_extract_entities_with_patterns(self, ner_pipeline):
        """Test pattern-based entity extraction."""
        text = "BNR announced RON rate of 4.9750 against EUR"

        entities = ner_pipeline.extract_entities_with_patterns(text)

        assert "CURRENCY" in entities
        assert len(entities["CURRENCY"]) > 0
        assert "ORGANIZATION" in entities
        assert len(entities["ORGANIZATION"]) > 0

    def test_extract_entities_with_spacy_mock(self, ner_pipeline):
        """Test spaCy-based entity extraction."""
        text = "Ministerul Finanțelor announced new RON policies"

        # Mock spaCy
        with patch.object(ner_pipeline, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "Ministerul Finanțelor"
            mock_ent.label_ = "ORG"
            mock_ent.start_char = 0
            mock_ent.end_char = 22
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc

            entities = ner_pipeline.extract_entities_with_spacy(text)

            assert "ORGANIZATION" in entities
            assert len(entities["ORGANIZATION"]) > 0


class TestTrendAnalyzer:
    """Test TrendAnalyzer functionality."""

    @pytest.fixture
    def trend_analyzer(self, db_session):
        """Create TrendAnalyzer instance for testing."""
        config = PipelineConfig(
            name="test_trends",
            model_path="test/trend/model"
        )
        return TrendAnalyzer(config, db_session)

    def test_calculate_sentiment_trends(self, trend_analyzer, db_session):
        """Test sentiment trend calculation."""
        # Add test data
        from src.models import SentimentAnalysis, ExtractedData, DataSource

        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        extracted_data = ExtractedData(
            source_id=data_source.id,
            content="test",
            title="test",
            is_processed=True
        )
        db_session.add(extracted_data)
        db_session.commit()

        sentiment = SentimentAnalysis(
            data_id=extracted_data.id,
            model_name="test",
            sentiment_label="positive",
            sentiment_score=0.7,
            confidence_score=0.8,
            created_at=datetime.now()
        )
        db_session.add(sentiment)
        db_session.commit()

        trends = trend_analyzer.calculate_sentiment_trends(timedelta(hours=1))

        # Should return trend data (may be empty if no recent data)
        assert isinstance(trends, dict)

    def test_analyze_volume_trends(self, trend_analyzer, db_session):
        """Test volume trend analysis."""
        # Add test data
        from src.models import ExtractedData, DataSource

        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        for i in range(5):
            extracted_data = ExtractedData(
                source_id=data_source.id,
                content=f"test {i}",
                title=f"test {i}",
                is_processed=True,
                created_at=datetime.now() - timedelta(minutes=i*10)
            )
            db_session.add(extracted_data)

        db_session.commit()

        volume_trends = trend_analyzer.analyze_volume_trends(timedelta(hours=1))

        assert "total_data_points" in volume_trends
        assert "avg_hourly_volume" in volume_trends
        assert volume_trends["total_data_points"] >= 5

    def test_detect_anomalies(self, trend_analyzer, db_session):
        """Test anomaly detection."""
        # Add test data with potential anomalies
        from src.models import ExtractedData, DataSource

        data_source = DataSource(name="test", source_type="test", url="test", is_active=True)
        db_session.add(data_source)
        db_session.commit()

        # Add normal volume data
        base_time = datetime.now()
        for i in range(10):
            extracted_data = ExtractedData(
                source_id=data_source.id,
                content=f"test {i}",
                title=f"test {i}",
                is_processed=True,
                created_at=base_time - timedelta(minutes=i*5)
            )
            db_session.add(extracted_data)

        db_session.commit()

        anomalies = trend_analyzer.detect_anomalies()

        assert "volume_anomalies" in anomalies
        assert "sentiment_anomalies" in anomalies
        assert "recent_volume_avg" in anomalies
