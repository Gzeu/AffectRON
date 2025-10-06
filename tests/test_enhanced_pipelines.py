"""
Tests for enhanced sentiment analysis pipeline.
Tests multi-language support and aspect-based sentiment analysis.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.pipelines.enhanced_sentiment import EnhancedSentimentPipeline, SentimentResult
from src.pipelines.aspect_sentiment import AspectBasedSentimentPipeline, AspectSentimentResult


class TestEnhancedSentimentPipeline:
    """Test EnhancedSentimentPipeline functionality."""

    @pytest.fixture
    def enhanced_pipeline(self, db_session):
        """Create EnhancedSentimentPipeline instance for testing."""
        config = PipelineConfig(
            name="test_enhanced_sentiment",
            model_path="test/model",
            batch_size=8
        )
        return EnhancedSentimentPipeline(config, db_session)

    def test_detect_language_romanian(self, enhanced_pipeline):
        """Test Romanian language detection."""
        romanian_text = "BNR a anunțat creșterea ratei RON cu 0.5% față de EUR"

        language = enhanced_pipeline.detect_language(romanian_text)
        assert language == 'ro'

    def test_detect_language_english(self, enhanced_pipeline):
        """Test English language detection."""
        english_text = "Federal Reserve announced interest rate changes"

        language = enhanced_pipeline.detect_language(english_text)
        assert language == 'en'

    def test_extract_financial_context_romanian(self, enhanced_pipeline):
        """Test financial context extraction for Romanian."""
        text = "BNR a crescut rata de dobândă pentru RON, afectând cursul EUR/RON"

        context = enhanced_pipeline.extract_financial_context(text, 'ro')

        assert 'RON' in context['currencies_mentioned']
        assert 'EUR' in context['currencies_mentioned']
        assert 'BNR' in context['institutions_mentioned']
        assert context['financial_relevance_score'] > 0.5

    def test_extract_financial_context_english(self, enhanced_pipeline):
        """Test financial context extraction for English."""
        text = "Federal Reserve raised interest rates, impacting EUR/USD exchange rate"

        context = enhanced_pipeline.extract_financial_context(text, 'en')

        assert 'EUR' in context['currencies_mentioned']
        assert 'USD' in context['currencies_mentioned']
        assert len(context['institutions_mentioned']) > 0
        assert context['financial_relevance_score'] > 0.3

    def test_preprocess_text_romanian(self, enhanced_pipeline):
        """Test text preprocessing for Romanian."""
        text = "BNR @utilizator a crescut https://example.com rata RON! Multe detalii aici."

        processed = enhanced_pipeline.preprocess_text(text, 'ro')

        # Should remove URLs, mentions, and normalize text
        assert 'https://' not in processed
        assert '@utilizator' not in processed
        assert 'BNR' in processed
        assert 'RON' in processed

    def test_analyze_sentiment_enhanced_romanian(self, enhanced_pipeline):
        """Test enhanced sentiment analysis for Romanian."""
        text = "BNR a crescut rata de dobândă pentru RON, veste excelentă pentru economie"

        # Mock model prediction
        with patch.object(enhanced_pipeline, 'analyze_sentiment_enhanced') as mock_analyze:
            mock_result = SentimentResult(
                label='positive',
                score=0.8,
                confidence=0.9,
                language='ro',
                entities={'currencies': ['RON'], 'institutions': ['BNR']},
                sentiment_intensity='strong',
                market_relevance=0.9
            )
            mock_analyze.return_value = mock_result

            result = enhanced_pipeline.analyze_sentiment_enhanced(text, 'ro')

            assert result.language == 'ro'
            assert result.label == 'positive'
            assert result.market_relevance > 0.5

    def test_adjust_sentiment_with_context_positive(self, enhanced_pipeline):
        """Test sentiment adjustment with positive financial context."""
        context = {
            'sentiment_indicators': {
                'positive': ['crește', 'profit'],
                'negative': []
            },
            'financial_relevance_score': 0.8
        }

        adjusted_label, adjusted_score, intensity = enhanced_pipeline._adjust_sentiment_with_context(
            'neutral', 0.6, context, 'ro'
        )

        assert adjusted_score > 0.6  # Should be boosted
        assert adjusted_label == 'positive'  # Should be upgraded

    def test_adjust_sentiment_with_context_negative(self, enhanced_pipeline):
        """Test sentiment adjustment with negative financial context."""
        context = {
            'sentiment_indicators': {
                'positive': [],
                'negative': ['scădere', 'pierdere', 'volatil']
            },
            'financial_relevance_score': 0.7
        }

        adjusted_label, adjusted_score, intensity = enhanced_pipeline._adjust_sentiment_with_context(
            'neutral', 0.6, context, 'ro'
        )

        assert adjusted_score < 0.6  # Should be penalized
        assert adjusted_label == 'negative'  # Should be downgraded


class TestAspectBasedSentimentPipeline:
    """Test AspectBasedSentimentPipeline functionality."""

    @pytest.fixture
    def aspect_pipeline(self, db_session):
        """Create AspectBasedSentimentPipeline instance for testing."""
        config = PipelineConfig(
            name="test_aspect_sentiment",
            model_path="test/model"
        )
        return AspectBasedSentimentPipeline(config, db_session)

    def test_extract_aspects_from_text_currencies(self, aspect_pipeline):
        """Test aspect extraction for currencies."""
        text = "RON a crescut față de EUR și USD după decizia BNR"

        aspects = aspect_pipeline.extract_aspects_from_text(text)

        assert len(aspects['currencies']) > 0
        assert any(aspect[0] == 'RON' for aspect in aspects['currencies'])
        assert any(aspect[0] == 'EUR' for aspect in aspects['currencies'])
        assert any(aspect[0] == 'USD' for aspect in aspects['currencies'])

    def test_extract_aspects_from_text_institutions(self, aspect_pipeline):
        """Test aspect extraction for institutions."""
        text = "BNR și ECB au anunțat noi măsuri de politică monetară"

        aspects = aspect_pipeline.extract_aspects_from_text(text)

        assert len(aspects['institutions']) > 0
        assert any(aspect[0] == 'BNR' for aspect in aspects['institutions'])
        assert any(aspect[0] == 'ECB' for aspect in aspects['institutions'])

    def test_find_aspect_mentions(self, aspect_pipeline):
        """Test finding specific aspect mentions."""
        text = "BNR a crescut rata de dobândă. Decizia BNR va afecta RON. ECB urmărește evoluția EUR."

        mentions = aspect_pipeline.find_aspect_mentions(text, ['BNR'])

        assert len(mentions) >= 2
        assert any('BNR a crescut rata' in mention for mention in mentions)

    def test_analyze_aspect_sentiment_positive(self, aspect_pipeline):
        """Test aspect sentiment analysis for positive sentiment."""
        text = "BNR a luat decizii excelente pentru stabilitatea RON"

        # Mock model prediction
        with patch.object(aspect_pipeline, 'analyze_aspect_sentiment') as mock_analyze:
            mock_result = AspectSentimentResult(
                aspect='BNR',
                aspect_type='institution',
                sentiment_label='positive',
                sentiment_score=0.8,
                confidence=0.9,
                mentions=['BNR a luat decizii excelente'],
                frequency=1
            )
            mock_analyze.return_value = mock_result

            result = aspect_pipeline.analyze_aspect_sentiment(text, 'BNR', 'institution')

            assert result.aspect == 'BNR'
            assert result.sentiment_label == 'positive'
            assert result.sentiment_score > 0.5

    def test_analyze_aspect_sentiment_no_mentions(self, aspect_pipeline):
        """Test aspect sentiment analysis when aspect not mentioned."""
        text = "Piața de capital arată evoluții interesante astăzi"

        result = aspect_pipeline.analyze_aspect_sentiment(text, 'FED', 'institution')

        assert result.frequency == 0
        assert result.sentiment_label == 'neutral'
        assert result.confidence == 0.0

    def test_calculate_overall_sentiment_balanced(self, aspect_pipeline):
        """Test overall sentiment calculation with mixed aspects."""
        aspect_results = [
            {
                'aspect_type': 'currency',
                'sentiment_label': 'positive',
                'sentiment_score': 0.8,
                'confidence': 0.9
            },
            {
                'aspect_type': 'institution',
                'sentiment_label': 'negative',
                'sentiment_score': 0.3,
                'confidence': 0.8
            }
        ]

        overall = aspect_pipeline._calculate_overall_sentiment(aspect_results)

        assert 0.3 < overall['score'] < 0.7  # Should be in neutral range
        assert overall['label'] == 'neutral'

    def test_calculate_overall_sentiment_positive(self, aspect_pipeline):
        """Test overall sentiment calculation with positive aspects."""
        aspect_results = [
            {
                'aspect_type': 'currency',
                'sentiment_label': 'positive',
                'sentiment_score': 0.8,
                'confidence': 0.9
            },
            {
                'aspect_type': 'institution',
                'sentiment_label': 'positive',
                'sentiment_score': 0.7,
                'confidence': 0.8
            }
        ]

        overall = aspect_pipeline._calculate_overall_sentiment(aspect_results)

        assert overall['score'] > 0.6
        assert overall['label'] == 'positive'

    def test_get_aspect_summary(self, aspect_pipeline):
        """Test aspect analysis summary generation."""
        aspect_results = [
            {
                'aspect_type': 'currency',
                'sentiment_label': 'positive',
                'confidence': 0.8
            },
            {
                'aspect_type': 'institution',
                'sentiment_label': 'positive',
                'confidence': 0.9
            },
            {
                'aspect_type': 'currency',
                'sentiment_label': 'negative',
                'confidence': 0.7
            }
        ]

        summary = aspect_pipeline.get_aspect_summary(aspect_results)

        assert summary['total_aspects'] == 3
        assert summary['aspect_types']['currency'] == 2
        assert summary['aspect_types']['institution'] == 1
        assert summary['sentiment_distribution']['positive'] == 2
        assert summary['sentiment_distribution']['negative'] == 1

    def test_cluster_related_aspects(self, aspect_pipeline):
        """Test aspect clustering functionality."""
        aspects = ['RON', 'EUR', 'BNR', 'ECB']
        texts = [
            'RON și EUR sunt stabile după deciziile BNR',
            'ECB și BNR colaborează pentru stabilitatea EUR și RON'
        ]

        clusters = aspect_pipeline.cluster_related_aspects(aspects, texts)

        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        # Should cluster related currencies and institutions
        assert any(len(cluster) > 1 for cluster in clusters.values())

    def test_get_supported_aspects(self, aspect_pipeline):
        """Test getting supported aspects."""
        aspects = aspect_pipeline.get_supported_aspects()

        assert 'currencies' in aspects
        assert 'institutions' in aspects
        assert 'economic_indicators' in aspects
        assert 'market_sentiment' in aspects

        assert 'RON' in aspects['currencies']
        assert 'BNR' in aspects['institutions']


class TestIntegration:
    """Integration tests for enhanced pipelines."""

    @pytest.mark.asyncio
    async def test_enhanced_pipeline_batch_processing(self, db_session):
        """Test batch processing with enhanced sentiment pipeline."""
        config = PipelineConfig(name="integration_test")
        pipeline = EnhancedSentimentPipeline(config, db_session)

        texts = [
            "BNR a crescut rata de dobândă pentru RON - veste bună pentru leu",
            "EUR scade față de USD după decizia Federal Reserve",
            "Piața cripto este foarte volatilă astăzi"
        ]

        results = await pipeline.process_batch(texts)

        assert len(results) == 3
        assert all('sentiment' in result for result in results)
        assert all('language' in result for result in results)
        assert all('market_relevance' in result for result in results)

    @pytest.mark.asyncio
    async def test_aspect_pipeline_batch_processing(self, db_session):
        """Test batch processing with aspect-based sentiment pipeline."""
        config = PipelineConfig(name="integration_test")
        pipeline = AspectBasedSentimentPipeline(config, db_session)

        texts = [
            "RON și EUR sunt afectate de deciziile BNR și ECB",
            "Federal Reserve menține dobânzile stabile pentru USD",
            "BNR anunță noi măsuri pentru stabilitatea leului românesc"
        ]

        results = await pipeline.process_batch(texts)

        assert len(results) == 3
        assert all('aspect_sentiments' in result for result in results)
        assert all('overall_sentiment' in result for result in results)

        # At least some results should have found aspects
        aspects_found = sum(result['aspects_found'] for result in results)
        assert aspects_found > 0
