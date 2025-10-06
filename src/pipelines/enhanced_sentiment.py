"""
Enhanced sentiment analysis pipeline with multi-language support.
Supports Romanian, English, German, and French financial sentiment analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline as hf_pipeline
)
import numpy as np
from langdetect import detect, LangDetectError
import re

from .base import BasePipeline, PipelineConfig, PipelineResult


@dataclass
class SentimentResult:
    """Enhanced sentiment analysis result."""
    label: str
    score: float
    confidence: float
    language: str
    entities: Dict[str, List[str]]
    sentiment_intensity: str  # 'weak', 'moderate', 'strong'
    market_relevance: float  # 0-1 score for financial relevance


class EnhancedSentimentPipeline(BasePipeline):
    """Enhanced sentiment analysis with multi-language support."""

    def __init__(self, config: PipelineConfig, db_session):
        super().__init__(config, db_session)

        # Multi-language model configurations
        self.models = {
            'ro': {
                'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'model': None,
                'tokenizer': None,
                'thresholds': {'positive': 0.6, 'negative': 0.4}
            },
            'en': {
                'model_name': 'ProsusAI/finbert',
                'model': None,
                'tokenizer': None,
                'thresholds': {'positive': 0.7, 'negative': 0.3}
            },
            'de': {
                'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'model': None,
                'tokenizer': None,
                'thresholds': {'positive': 0.6, 'negative': 0.4}
            },
            'fr': {
                'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'model': None,
                'tokenizer': None,
                'thresholds': {'positive': 0.6, 'negative': 0.4}
            }
        }

        # Financial terminology for each language
        self.financial_terms = {
            'ro': {
                'currencies': ['ron', 'leu', 'eur', 'euro', 'usd', 'dolar', 'btc', 'bitcoin', 'eth', 'ethereum'],
                'institutions': ['bnr', 'banca națională', 'ministerul finanțelor', 'bursă', 'bvb'],
                'positive': ['crește', 'creștere', 'profit', 'câștig', 'bullish', 'optimist', 'stabil'],
                'negative': ['scade', 'scădere', 'pierdere', 'pierderi', 'bearish', 'pesimist', 'volatil'],
                'verbs': ['crește', 'scade', 'urcă', 'coboară', 'apreciază', 'depreciază']
            },
            'en': {
                'currencies': ['ron', 'eur', 'usd', 'btc', 'eth', 'gbp', 'jpy', 'chf'],
                'institutions': ['federal reserve', 'ecb', 'bank of england', 'fed', 'central bank'],
                'positive': ['rises', 'gains', 'profit', 'bullish', 'optimistic', 'stable', 'growth'],
                'negative': ['falls', 'losses', 'bearish', 'pessimistic', 'volatile', 'decline', 'drop'],
                'verbs': ['rises', 'falls', 'gains', 'loses', 'increases', 'decreases']
            },
            'de': {
                'currencies': ['eur', 'usd', 'gbp', 'chf', 'btc', 'eth'],
                'institutions': ['ezb', 'bundesbank', 'europäische zentralbank'],
                'positive': ['steigt', 'gewinne', 'profit', 'bullish', 'optimistisch', 'stabil'],
                'negative': ['fällt', 'verluste', 'bearish', 'pessimistisch', 'volatil'],
                'verbs': ['steigt', 'fällt', 'gewinnt', 'verliert']
            },
            'fr': {
                'currencies': ['eur', 'usd', 'gbp', 'chf', 'btc', 'eth'],
                'institutions': ['bce', 'banque de france', 'banque centrale européenne'],
                'positive': ['augmente', 'gains', 'profit', 'hausse', 'optimiste', 'stable'],
                'negative': ['baisse', 'pertes', 'bearish', 'pessimiste', 'volatile'],
                'verbs': ['augmente', 'baisse', 'gagne', 'perd']
            }
        }

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize all language models."""
        self.logger.info("Initializing enhanced sentiment models...")

        for lang, config in self.models.items():
            try:
                self.logger.info(f"Loading {lang} model: {config['model_name']}")

                config['tokenizer'] = AutoTokenizer.from_pretrained(config['model_name'])
                config['model'] = AutoModelForSequenceClassification.from_pretrained(config['model_name'])

                # Move to GPU if available
                if torch.cuda.is_available():
                    config['model'] = config['model'].to('cuda')

                self.logger.info(f"✅ {lang} model loaded successfully")

            except Exception as e:
                self.logger.error(f"❌ Failed to load {lang} model: {e}")
                # Fallback to multilingual model
                if lang != 'ro':
                    config['model_name'] = 'nlptown/bert-base-multilingual-uncased-sentiment'
                    config['tokenizer'] = AutoTokenizer.from_pretrained(config['model_name'])
                    config['model'] = AutoModelForSequenceClassification.from_pretrained(config['model_name'])

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            # First try to detect with langdetect
            detected = detect(text)

            # Map to our supported languages
            lang_map = {
                'romanian': 'ro',
                'english': 'en',
                'german': 'de',
                'french': 'fr'
            }

            return lang_map.get(detected, 'en')  # Default to English

        except LangDetectError:
            # Fallback to simple heuristics
            return self._detect_language_heuristics(text)

    def _detect_language_heuristics(self, text: str) -> str:
        """Simple language detection based on common words."""
        text_lower = text.lower()

        # Romanian indicators
        if any(word in text_lower for word in ['românia', 'românească', 'leu', 'bnr', 'bucurești']):
            return 'ro'

        # German indicators
        if any(word in text_lower for word in ['deutschland', 'euro', 'frankfurt', 'bundesbank']):
            return 'de'

        # French indicators
        if any(word in text_lower for word in ['france', 'paris', 'banque de france']):
            return 'fr'

        return 'en'  # Default to English

    def extract_financial_context(self, text: str, language: str) -> Dict[str, Any]:
        """Extract financial context from text."""
        terms = self.financial_terms.get(language, self.financial_terms['en'])

        context = {
            'currencies_mentioned': [],
            'institutions_mentioned': [],
            'sentiment_indicators': {'positive': [], 'negative': [], 'neutral': []},
            'financial_relevance_score': 0.0,
            'key_phrases': []
        }

        text_lower = text.lower()

        # Extract currencies
        for currency in terms['currencies']:
            if currency in text_lower:
                context['currencies_mentioned'].append(currency.upper())

        # Extract institutions
        for institution in terms['institutions']:
            if institution.lower() in text_lower:
                context['institutions_mentioned'].append(institution)

        # Extract sentiment indicators
        for sentiment_type in ['positive', 'negative']:
            for indicator in terms[sentiment_type]:
                if indicator in text_lower:
                    context['sentiment_indicators'][sentiment_type].append(indicator)

        # Calculate financial relevance
        financial_words = (terms['currencies'] + terms['institutions'] +
                          terms['positive'] + terms['negative'] + terms['verbs'])
        financial_count = sum(1 for word in financial_words if word in text_lower)
        context['financial_relevance_score'] = min(financial_count / len(text.split()) * 2, 1.0)

        return context

    def preprocess_text(self, text: str, language: str) -> str:
        """Enhanced text preprocessing."""
        # Basic cleaning
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags

        # Language-specific preprocessing
        if language == 'ro':
            # Romanian specific cleaning
            text = re.sub(r'[ăâîșțĂÂÎȘȚ]', lambda m: {
                'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
                'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
            }[m.group()], text)

        # Truncate if too long (keep financial context)
        words = text.split()
        if len(words) > 200:
            # Try to keep financial terms
            terms = self.financial_terms.get(language, self.financial_terms['en'])
            financial_words = set(terms['currencies'] + terms['institutions'] +
                                terms['positive'] + terms['negative'])

            important_words = [w for w in words if w.lower() in financial_words]
            remaining_words = 200 - len(important_words)

            if remaining_words > 0:
                other_words = [w for w in words if w.lower() not in financial_words][:remaining_words]
                text = ' '.join(important_words + other_words)
            else:
                text = ' '.join(important_words[:200])

        return text.strip()

    def analyze_sentiment_enhanced(self, text: str, language: str) -> SentimentResult:
        """Enhanced sentiment analysis with financial context."""
        # Get model for language
        model_config = self.models.get(language, self.models['en'])

        if model_config['model'] is None:
            raise ValueError(f"Model not loaded for language: {language}")

        # Preprocess text
        clean_text = self.preprocess_text(text, language)

        # Extract financial context
        context = self.extract_financial_context(clean_text, language)

        # Get model prediction
        inputs = model_config['tokenizer'](
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_config['model'](**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to sentiment label and score
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

        # Map to sentiment labels (depends on model)
        if 'finbert' in model_config['model_name']:
            # FinBERT: 0=negative, 1=neutral, 2=positive
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            base_label = label_map.get(predicted_class, 'neutral')
        else:
            # Multilingual BERT: 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive
            if predicted_class <= 1:
                base_label = 'negative'
            elif predicted_class == 2:
                base_label = 'neutral'
            else:
                base_label = 'positive'

        # Adjust sentiment based on financial context
        adjusted_label, adjusted_score, intensity = self._adjust_sentiment_with_context(
            base_label, confidence, context, language
        )

        return SentimentResult(
            label=adjusted_label,
            score=adjusted_score,
            confidence=confidence,
            language=language,
            entities={
                'currencies': context['currencies_mentioned'],
                'institutions': context['institutions_mentioned']
            },
            sentiment_intensity=intensity,
            market_relevance=context['financial_relevance_score']
        )

    def _adjust_sentiment_with_context(self, base_label: str, confidence: float,
                                     context: Dict, language: str) -> Tuple[str, float, str]:
        """Adjust sentiment based on financial context."""

        # Calculate context influence
        positive_indicators = len(context['sentiment_indicators']['positive'])
        negative_indicators = len(context['sentiment_indicators']['negative'])
        financial_relevance = context['financial_relevance_score']

        # Determine intensity
        total_indicators = positive_indicators + negative_indicators
        if total_indicators == 0:
            intensity = 'weak'
        elif total_indicators <= 2:
            intensity = 'moderate'
        else:
            intensity = 'strong'

        # Adjust sentiment based on indicators
        if positive_indicators > negative_indicators:
            sentiment_boost = 0.1 * min(positive_indicators, 3)  # Cap at 0.3
            adjusted_score = min(confidence + sentiment_boost, 1.0)

            # If we have strong positive indicators, upgrade to positive
            if positive_indicators >= 2 and base_label == 'neutral':
                adjusted_label = 'positive'
            else:
                adjusted_label = base_label

        elif negative_indicators > positive_indicators:
            sentiment_penalty = 0.1 * min(negative_indicators, 3)  # Cap at 0.3
            adjusted_score = max(confidence - sentiment_penalty, 0.0)

            # If we have strong negative indicators, downgrade to negative
            if negative_indicators >= 2 and base_label == 'neutral':
                adjusted_label = 'negative'
            else:
                adjusted_label = base_label

        else:
            # Neutral indicators or balanced
            adjusted_score = confidence
            adjusted_label = base_label

        # Apply financial relevance multiplier
        if financial_relevance > 0.7:
            # High financial relevance - boost confidence
            adjusted_score = min(adjusted_score * 1.1, 1.0)
        elif financial_relevance < 0.3:
            # Low financial relevance - reduce confidence
            adjusted_score = adjusted_score * 0.9

        return adjusted_label, adjusted_score, intensity

    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts with enhanced sentiment analysis."""
        results = []

        for text in texts:
            try:
                # Detect language
                language = self.detect_language(text)

                # Analyze sentiment
                sentiment_result = self.analyze_sentiment_enhanced(text, language)

                # Convert to expected format
                result = {
                    'text': text,
                    'sentiment': {
                        'label': sentiment_result.label,
                        'score': sentiment_result.score,
                        'confidence': sentiment_result.confidence,
                        'intensity': sentiment_result.sentiment_intensity
                    },
                    'entities': sentiment_result.entities,
                    'language': sentiment_result.language,
                    'market_relevance': sentiment_result.market_relevance,
                    'processed_at': datetime.now().isoformat()
                }

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing text: {e}")
                # Return basic error result
                results.append({
                    'text': text,
                    'sentiment': {'label': 'neutral', 'score': 0.5, 'confidence': 0.0},
                    'entities': {},
                    'language': 'unknown',
                    'market_relevance': 0.0,
                    'error': str(e)
                })

        return results

    async def run_analysis(self) -> List[PipelineResult]:
        """Run sentiment analysis on pending data."""
        # This would integrate with the main data processing pipeline
        # For now, return empty results
        return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'supported_languages': list(self.models.keys()),
            'models_loaded': {
                lang: config['model'] is not None
                for lang, config in self.models.items()
            },
            'financial_terms_count': {
                lang: len(terms)
                for lang, terms in self.financial_terms.items()
            }
        }
