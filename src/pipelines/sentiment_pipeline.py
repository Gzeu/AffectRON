"""
Sentiment analysis pipeline using FinBERT for Romanian financial text.
Analyzes sentiment of financial news, tweets, and other text data.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

from .base import BasePipeline, PipelineConfig, PipelineResult


logger = logging.getLogger(__name__)


class SentimentPipeline(BasePipeline):
    """Financial sentiment analysis pipeline using FinBERT."""

    def __init__(self, config: PipelineConfig, db_session):
        super().__init__(config, db_session)

        # Romanian-specific sentiment enhancements
        self.positive_indicators = [
            'crestere', 'profit', 'castig', 'investitie', 'buna',
            'pozitiv', 'optimist', 'succes', 'revenire', 'stabilitate',
            'dezvoltare', 'oportunitate', 'avantaj', 'performanta'
        ]

        self.negative_indicators = [
            'scadere', 'pierdere', 'pierdut', 'problema', 'riscuri',
            'negativ', 'pesimist', 'esec', 'faliment', 'instabilitate',
            'recesiune', 'criza', 'deficit', 'inflatie', 'somaj'
        ]

        self.financial_terms = [
            'ron', 'eur', 'usd', 'leu', 'euro', 'dolar',
            'banca', 'burs', 'actiuni', 'obligatiuni',
            'dobanda', 'rata', 'credit', 'depozit',
            'investitie', 'profit', 'pierdere', 'dividend'
        ]

    async def load_model(self):
        """Load the FinBERT sentiment analysis model."""
        try:
            logger.info(f"Loading sentiment model from {self.config.model_path}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                cache_dir=self.config.cache_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path,
                cache_dir=self.config.cache_dir,
                num_labels=3  # negative, neutral, positive
            )

            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )

            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Clean text
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Truncate if too long
        if len(text) > self.config.max_length - 50:  # Leave room for special tokens
            text = text[:self.config.max_length - 50]

        return text.strip()

    def extract_financial_context(self, text: str) -> Dict[str, Any]:
        """Extract financial context from text."""
        text_lower = text.lower()

        context = {
            'has_financial_terms': False,
            'mentioned_currencies': [],
            'sentiment_indicators': {
                'positive': [],
                'negative': []
            }
        }

        # Check for financial terms
        for term in self.financial_terms:
            if term in text_lower:
                context['has_financial_terms'] = True
                break

        # Extract currencies
        currency_patterns = {
            'RON': ['ron', 'leu', 'lei'],
            'EUR': ['eur', 'euro'],
            'USD': ['usd', 'dolar', 'dolari'],
            'BTC': ['btc', 'bitcoin'],
            'ETH': ['eth', 'ethereum']
        }

        for currency, patterns in currency_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    context['mentioned_currencies'].append(currency)
                    break

        # Extract sentiment indicators
        for indicator in self.positive_indicators:
            if indicator in text_lower:
                context['sentiment_indicators']['positive'].append(indicator)

        for indicator in self.negative_indicators:
            if indicator in text_lower:
                context['sentiment_indicators']['negative'].append(indicator)

        return context

    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts for sentiment analysis."""
        if not self.sentiment_pipeline:
            raise RuntimeError("Model not loaded")

        results = []

        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]

            # Run sentiment analysis
            raw_results = self.sentiment_pipeline(processed_texts)

            for i, raw_result in enumerate(raw_results):
                original_text = texts[i]
                processed_text = processed_texts[i]

                # Get the highest confidence prediction
                sentiment_scores = raw_result[0]  # pipeline returns nested structure

                # Find the label with highest score
                best_prediction = max(sentiment_scores, key=lambda x: x['score'])

                # Map labels to standard format
                label_mapping = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral',
                    'LABEL_2': 'positive'
                }

                predicted_label = label_mapping.get(best_prediction['label'], 'neutral')

                # Apply Romanian financial context adjustments
                financial_context = self.extract_financial_context(original_text)
                adjusted_label, adjusted_score = self._adjust_sentiment_for_context(
                    predicted_label, best_prediction['score'], financial_context
                )

                # Extract entities if present
                entities = self._extract_entities(original_text)

                result = {
                    'sentiment': {
                        'label': adjusted_label,
                        'score': adjusted_score,
                        'confidence': best_prediction['score']
                    },
                    'entities': entities,
                    'financial_context': financial_context,
                    'text_length': len(original_text),
                    'processed_length': len(processed_text)
                }

                results.append(result)

        except Exception as e:
            logger.error(f"Error in sentiment analysis batch processing: {str(e)}")
            # Return neutral sentiment for failed analyses
            for text in texts:
                results.append({
                    'sentiment': {
                        'label': 'neutral',
                        'score': 0.0,
                        'confidence': 0.0
                    },
                    'entities': {},
                    'financial_context': {},
                    'text_length': len(text),
                    'processed_length': len(text)
                })

        return results

    def _adjust_sentiment_for_context(self, predicted_label: str, score: float,
                                    context: Dict[str, Any]) -> tuple[str, float]:
        """Adjust sentiment based on Romanian financial context."""

        # If no financial terms, reduce confidence
        if not context['has_financial_terms']:
            score *= 0.8

        # Adjust based on sentiment indicators
        positive_indicators = len(context['sentiment_indicators']['positive'])
        negative_indicators = len(context['sentiment_indicators']['negative'])

        # Simple rule-based adjustment
        if positive_indicators > negative_indicators and predicted_label == 'neutral':
            if score > 0.4:  # Only adjust if reasonably confident
                predicted_label = 'positive'
                score = min(score + 0.1, 1.0)

        elif negative_indicators > positive_indicators and predicted_label == 'neutral':
            if score > 0.4:
                predicted_label = 'negative'
                score = min(score + 0.1, 1.0)

        # Currency mentions can affect sentiment intensity
        if context['mentioned_currencies']:
            score = min(score * 1.1, 1.0)  # Boost confidence for currency mentions

        return predicted_label, score

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text."""
        entities = {
            'currencies': [],
            'organizations': [],
            'financial_terms': []
        }

        # Simple pattern-based entity extraction
        currency_patterns = [
            r'\bRON\b', r'\bEUR\b', r'\bUSD\b', r'\bBTC\b', r'\bETH\b',
            r'\bleu\b', r'\blei\b', r'\beuro\b', r'\bdolar\b', r'\bdolari\b'
        ]

        org_patterns = [
            r'\bBNR\b', r'\bBanca\s+Naţională\b', r'\bMinisterul\s+Finanţelor\b',
            r'\bBursa\s+de\s+Valori\b', r'\bCEC\s+Bank\b'
        ]

        financial_patterns = [
            r'\bdobândă?\b', r'\brată?\b', r'\binflaţie\b', r'\brecesiune\b',
            r'\bprofit\b', r'\bpierdere\b', r'\binvestiţii?\b'
        ]

        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['currencies'].extend(matches)

        for pattern in org_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['organizations'].extend(matches)

        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['financial_terms'].extend(matches)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities
