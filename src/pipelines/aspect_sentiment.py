"""
Aspect-based sentiment analysis for financial entities.
Analyzes sentiment towards specific currencies, institutions, and market aspects.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import re

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from .base import BasePipeline, PipelineConfig, PipelineResult


@dataclass
class AspectSentimentResult:
    """Aspect-based sentiment analysis result."""
    aspect: str
    aspect_type: str  # 'currency', 'institution', 'market', 'economic_indicator'
    sentiment_label: str
    sentiment_score: float
    confidence: float
    mentions: List[str]  # Text snippets mentioning this aspect
    frequency: int  # How often this aspect is mentioned


class AspectBasedSentimentPipeline(BasePipeline):
    """Aspect-based sentiment analysis for financial entities."""

    def __init__(self, config: PipelineConfig, db_session):
        super().__init__(config, db_session)

        # Financial aspects to track
        self.financial_aspects = {
            'currencies': {
                'RON': ['ron', 'leu', 'romanian leu', 'leul românesc'],
                'EUR': ['eur', 'euro', 'european euro'],
                'USD': ['usd', 'dollar', 'us dollar', 'american dollar'],
                'GBP': ['gbp', 'pound', 'british pound', 'sterling'],
                'CHF': ['chf', 'franc', 'swiss franc'],
                'JPY': ['jpy', 'yen', 'japanese yen'],
                'BTC': ['btc', 'bitcoin'],
                'ETH': ['eth', 'ethereum']
            },
            'institutions': {
                'BNR': ['bnr', 'banca națională a româniei', 'banca nationala'],
                'ECB': ['ecb', 'european central bank', 'banca centrală europeană'],
                'FED': ['fed', 'federal reserve', 'federal reserve system'],
                'BOE': ['boe', 'bank of england'],
                'SNB': ['snb', 'swiss national bank']
            },
            'economic_indicators': {
                'inflation': ['inflation', 'inflație', 'ipc', 'consumer price index'],
                'interest_rates': ['interest rates', 'dobânzi', 'rate dobândă'],
                'gdp': ['gdp', 'pib', 'gross domestic product'],
                'unemployment': ['unemployment', 'șomaj', 'rata șomajului'],
                'trade_balance': ['trade balance', 'balanța comercială', 'export', 'import']
            },
            'market_sentiment': {
                'bullish': ['bullish', 'optimist', 'pozitiv', 'creștere', 'rally'],
                'bearish': ['bearish', 'pesimist', 'negativ', 'scădere', 'corecție'],
                'volatile': ['volatile', 'volatilitate', 'instabil'],
                'stable': ['stable', 'stabil', 'constant']
            }
        }

        # Sentiment model for aspect classification
        self.model_name = "ProsusAI/finbert"
        self.model = None
        self.tokenizer = None

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the aspect-based sentiment model."""
        self.logger.info(f"Loading aspect-based sentiment model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self.model = self.model.to('cuda')

            self.logger.info("✅ Aspect-based sentiment model loaded")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def extract_aspects_from_text(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract financial aspects mentioned in text."""
        text_lower = text.lower()
        found_aspects = {category: [] for category in self.financial_aspects.keys()}

        for category, aspects in self.financial_aspects.items():
            for aspect_name, keywords in aspects.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        found_aspects[category].append((aspect_name, keyword))
                        break  # Only count each aspect once per category

        return found_aspects

    def find_aspect_mentions(self, text: str, aspect_keywords: List[str]) -> List[str]:
        """Find specific mentions of an aspect in text."""
        mentions = []

        for keyword in aspect_keywords:
            # Find sentences containing the keyword
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    mentions.append(sentence.strip())

        return mentions

    def analyze_aspect_sentiment(self, text: str, aspect: str, aspect_type: str) -> AspectSentimentResult:
        """Analyze sentiment towards a specific financial aspect."""

        # Get aspect keywords
        aspect_keywords = []
        for category, aspects in self.financial_aspects.items():
            if aspect in aspects:
                aspect_keywords = aspects[aspect]
                break

        if not aspect_keywords:
            return AspectSentimentResult(
                aspect=aspect,
                aspect_type=aspect_type,
                sentiment_label='neutral',
                sentiment_score=0.5,
                confidence=0.0,
                mentions=[],
                frequency=0
            )

        # Find mentions of this aspect
        mentions = self.find_aspect_mentions(text, aspect_keywords)

        if not mentions:
            return AspectSentimentResult(
                aspect=aspect,
                aspect_type=aspect_type,
                sentiment_label='neutral',
                sentiment_score=0.5,
                confidence=0.0,
                mentions=[],
                frequency=0
            )

        # Analyze sentiment of mentions
        sentiments = []
        confidences = []

        for mention in mentions:
            # Get model prediction for this mention
            inputs = self.tokenizer(
                mention,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

            # Map to sentiment labels (FinBERT: 0=negative, 1=neutral, 2=positive)
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_label = label_map.get(predicted_class, 'neutral')

            sentiments.append(sentiment_label)
            confidences.append(confidence)

        # Aggregate results
        if not sentiments:
            return AspectSentimentResult(
                aspect=aspect,
                aspect_type=aspect_type,
                sentiment_label='neutral',
                sentiment_score=0.5,
                confidence=0.0,
                mentions=mentions,
                frequency=len(mentions)
            )

        # Calculate weighted sentiment score
        sentiment_scores = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}
        weighted_score = sum(
            sentiment_scores[sent] * conf
            for sent, conf in zip(sentiments, confidences)
        ) / sum(confidences) if confidences else 0.5

        # Determine dominant sentiment
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        for sent in sentiments:
            sentiment_counts[sent] += 1

        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        return AspectSentimentResult(
            aspect=aspect,
            aspect_type=aspect_type,
            sentiment_label=dominant_sentiment,
            sentiment_score=weighted_score,
            confidence=np.mean(confidences),
            mentions=mentions,
            frequency=len(mentions)
        )

    def cluster_related_aspects(self, aspects: List[str], texts: List[str]) -> Dict[str, List[str]]:
        """Cluster related aspects using text similarity."""
        if len(aspects) < 2:
            return {aspect: [aspect] for aspect in aspects}

        # Create TF-IDF vectors for aspect-related text
        aspect_texts = []
        for aspect in aspects:
            # Combine texts that mention this aspect
            related_texts = []
            for text in texts:
                if any(keyword.lower() in text.lower()
                      for category in self.financial_aspects.values()
                      for keywords in category.values()
                      if aspect in keywords
                      for keyword in keywords):
                    related_texts.append(text)

            aspect_texts.append(' '.join(related_texts) if related_texts else aspect)

        # Create clusters
        if len(aspect_texts) > 1:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(aspect_texts)

                # Use K-means clustering
                n_clusters = min(len(aspects) // 2 + 1, 5)  # Adaptive cluster count
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix.toarray())

                # Group aspects by cluster
                clustered_aspects = {}
                for i, aspect in enumerate(aspects):
                    cluster_id = clusters[i]
                    if cluster_id not in clustered_aspects:
                        clustered_aspects[cluster_id] = []
                    clustered_aspects[cluster_id].append(aspect)

                return clustered_aspects

            except Exception as e:
                self.logger.warning(f"Clustering failed: {e}")

        # Fallback: return each aspect as its own cluster
        return {aspect: [aspect] for aspect in aspects}

    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process batch of texts for aspect-based sentiment analysis."""
        all_results = []

        for text in texts:
            try:
                # Extract all aspects mentioned in text
                found_aspects = self.extract_aspects_from_text(text)

                aspect_results = []

                # Analyze sentiment for each found aspect
                for category, aspects in found_aspects.items():
                    for aspect_name, _ in aspects:
                        aspect_result = self.analyze_aspect_sentiment(text, aspect_name, category)

                        if aspect_result.frequency > 0:  # Only include aspects that were actually mentioned
                            aspect_results.append({
                                'aspect': aspect_result.aspect,
                                'aspect_type': aspect_result.aspect_type,
                                'sentiment_label': aspect_result.sentiment_label,
                                'sentiment_score': aspect_result.sentiment_score,
                                'confidence': aspect_result.confidence,
                                'mentions': aspect_result.mentions,
                                'frequency': aspect_result.frequency
                            })

                # Add overall text sentiment
                overall_sentiment = self._calculate_overall_sentiment(aspect_results)

                result = {
                    'text': text,
                    'aspect_sentiments': aspect_results,
                    'overall_sentiment': overall_sentiment,
                    'aspects_found': len(aspect_results),
                    'processed_at': datetime.now().isoformat()
                }

                all_results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing text for aspect analysis: {e}")
                all_results.append({
                    'text': text,
                    'aspect_sentiments': [],
                    'overall_sentiment': {'label': 'neutral', 'score': 0.5},
                    'aspects_found': 0,
                    'error': str(e)
                })

        return all_results

    def _calculate_overall_sentiment(self, aspect_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall sentiment from aspect results."""
        if not aspect_results:
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.0}

        # Weight by confidence and frequency
        total_weighted_score = 0
        total_weight = 0

        for result in aspect_results:
            weight = result['confidence'] * result['frequency']
            total_weighted_score += result['sentiment_score'] * weight
            total_weight += weight

        if total_weight == 0:
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.0}

        overall_score = total_weighted_score / total_weight

        # Determine label
        if overall_score > 0.6:
            label = 'positive'
        elif overall_score < 0.4:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'label': label,
            'score': overall_score,
            'confidence': np.mean([r['confidence'] for r in aspect_results])
        }

    def get_aspect_summary(self, aspect_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of aspect-based analysis."""
        if not aspect_results:
            return {
                'total_aspects': 0,
                'aspect_types': {},
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }

        # Count by type
        type_counts = {}
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        for result in aspect_results:
            aspect_type = result['aspect_type']
            type_counts[aspect_type] = type_counts.get(aspect_type, 0) + 1

            sentiment_label = result['sentiment_label']
            sentiment_counts[sentiment_label] = sentiment_counts.get(sentiment_label, 0) + 1

        return {
            'total_aspects': len(aspect_results),
            'aspect_types': type_counts,
            'sentiment_distribution': sentiment_counts,
            'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
        }

    async def run_analysis(self) -> List[PipelineResult]:
        """Run aspect-based sentiment analysis on pending data."""
        # This would integrate with the main data processing pipeline
        return []

    def get_supported_aspects(self) -> Dict[str, List[str]]:
        """Get all supported financial aspects."""
        return self.financial_aspects
