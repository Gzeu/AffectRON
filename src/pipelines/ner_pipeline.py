"""
Named Entity Recognition (NER) pipeline for financial entities.
Extracts currencies, organizations, and financial terms from text.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional

import torch
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from .base import BasePipeline, PipelineConfig, PipelineResult


logger = logging.getLogger(__name__)


class NERPipeline(BasePipeline):
    """Financial Named Entity Recognition pipeline."""

    def __init__(self, config: PipelineConfig, db_session):
        super().__init__(config, db_session)

        # Financial entity patterns for Romanian text
        self.currency_patterns = [
            r'\b(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(RON|EUR|USD|BTC|ETH|lei|euro|dolari)\b',
            r'\b(RON|EUR|USD|BTC|ETH)\s*(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\b',
            r'\b(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(lei|euro|dolari|bitcoin|ethereum)\b'
        ]

        self.organization_patterns = [
            r'\b(BNR|Banca\s+Naţională\s+a\s+României)\b',
            r'\b(Ministerul\s+Finanţelor)\b',
            r'\b(Bursa\s+de\s+Valori\s+Bucureşti|BVB)\b',
            r'\b(CEC\s+Bank|Banca\s+Transilvania|BRD|BCR)\b'
        ]

        self.financial_term_patterns = [
            r'\b(dobândă|dobânzi|rată|rate)\b',
            r'\b(inflaţie|deflaţie)\b',
            r'\b(PIB|produs\s+intern\s+brut)\b',
            r'\b(obligaţiuni|acţiuni)\b',
            r'\b(investiţii|investiţie)\b'
        ]

    async def load_model(self):
        """Load the NER model."""
        try:
            logger.info(f"Loading NER model from {self.config.model_path}")

            # Try to load a Romanian NER model if available
            try:
                # Check if Romanian spaCy model is available
                self.nlp = spacy.load("ro_core_news_sm")
                self.use_spacy = True
                logger.info("Loaded Romanian spaCy NER model")
            except OSError:
                # Fallback to basic pattern matching
                self.nlp = None
                self.use_spacy = False
                logger.info("Romanian spaCy model not available, using pattern matching")

            # Also try to load a transformer-based NER model if specified
            if hasattr(self.config, 'transformer_model') and self.config.transformer_model:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.transformer_model,
                    cache_dir=self.config.cache_dir
                )

                self.ner_model = AutoModelForTokenClassification.from_pretrained(
                    self.config.transformer_model,
                    cache_dir=self.config.cache_dir
                )

                self.ner_model.to(self.device)
                self.ner_model.eval()

                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.ner_model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )

                logger.info("Loaded transformer NER model")

        except Exception as e:
            logger.error(f"Error loading NER model: {str(e)}")
            raise

    def extract_entities_with_patterns(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using regex patterns."""
        entities = {
            'CURRENCY': [],
            'ORGANIZATION': [],
            'FINANCIAL_TERM': [],
            'MONEY': []
        }

        # Extract currency amounts
        for pattern in self.currency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = match.group(1) or match.group(2)
                currency = match.group(2) or match.group(1)

                if amount and currency:
                    entities['CURRENCY'].append({
                        'text': match.group(0),
                        'amount': amount.replace('.', '').replace(',', '.'),
                        'currency': currency.upper(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9
                    })

        # Extract organizations
        for pattern in self.organization_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['ORGANIZATION'].append({
                    'text': match.group(0),
                    'label': 'ORGANIZATION',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })

        # Extract financial terms
        for pattern in self.financial_term_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['FINANCIAL_TERM'].append({
                    'text': match.group(0),
                    'label': 'FINANCIAL_TERM',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })

        # Extract money amounts without currency
        money_pattern = r'\b(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(milioane|miliarde|lei|euro|dolari)?\b'
        matches = re.finditer(money_pattern, text, re.IGNORECASE)
        for match in matches:
            amount = match.group(1)
            unit = match.group(2) or ''

            entities['MONEY'].append({
                'text': match.group(0),
                'amount': amount.replace('.', '').replace(',', '.'),
                'unit': unit,
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })

        return entities

    def extract_entities_with_spacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy."""
        entities = {
            'CURRENCY': [],
            'ORGANIZATION': [],
            'FINANCIAL_TERM': [],
            'MONEY': []
        }

        if not self.nlp:
            return self.extract_entities_with_patterns(text)

        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'ORG', 'PERSON']:
                # Map spaCy labels to our categories
                if ent.label_ == 'MONEY':
                    entities['CURRENCY'].append({
                        'text': ent.text,
                        'label': 'CURRENCY',
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8
                    })
                elif ent.label_ == 'ORG':
                    entities['ORGANIZATION'].append({
                        'text': ent.text,
                        'label': 'ORGANIZATION',
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8
                    })

        # Add pattern-based extraction for financial terms
        financial_entities = self.extract_entities_with_patterns(text)
        for category in ['FINANCIAL_TERM', 'CURRENCY', 'MONEY']:
            entities[category].extend(financial_entities.get(category, []))

        return entities

    def extract_entities_with_transformers(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using transformer models."""
        entities = {
            'CURRENCY': [],
            'ORGANIZATION': [],
            'FINANCIAL_TERM': [],
            'MONEY': []
        }

        if not hasattr(self, 'ner_pipeline'):
            return self.extract_entities_with_spacy(text)

        try:
            ner_results = self.ner_pipeline(text)

            for entity in ner_results:
                label = entity['entity_group']

                # Map transformer labels to our categories
                if label in ['MISC', 'LOC']:  # Financial terms often labeled as MISC
                    if any(term in entity['word'].lower() for term in
                          ['ron', 'eur', 'usd', 'leu', 'euro', 'dolar', 'dobanda', 'rata']):
                        entities['FINANCIAL_TERM'].append({
                            'text': entity['word'],
                            'label': 'FINANCIAL_TERM',
                            'start': entity['start'],
                            'end': entity['end'],
                            'confidence': entity['score']
                        })
                elif label == 'ORG':
                    entities['ORGANIZATION'].append({
                        'text': entity['word'],
                        'label': 'ORGANIZATION',
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })

        except Exception as e:
            logger.error(f"Error in transformer NER: {str(e)}")
            return self.extract_entities_with_spacy(text)

        return entities

    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts for NER."""
        results = []

        for text in texts:
            try:
                # Choose extraction method based on available models
                if hasattr(self, 'ner_pipeline') and self.ner_pipeline:
                    entities = self.extract_entities_with_transformers(text)
                elif self.use_spacy:
                    entities = self.extract_entities_with_spacy(text)
                else:
                    entities = self.extract_entities_with_patterns(text)

                # Calculate statistics
                stats = {
                    'total_entities': sum(len(ents) for ents in entities.values()),
                    'currency_count': len(entities['CURRENCY']),
                    'organization_count': len(entities['ORGANIZATION']),
                    'financial_term_count': len(entities['FINANCIAL_TERM']),
                    'money_count': len(entities['MONEY'])
                }

                # Extract unique entity texts for easier processing
                unique_entities = {}
                for category, entity_list in entities.items():
                    unique_entities[category] = list(set(
                        f"{ent['text']} ({ent.get('currency', ent.get('amount', ''))})"
                        for ent in entity_list
                    ))

                result = {
                    'entities': entities,
                    'unique_entities': unique_entities,
                    'statistics': stats,
                    'extraction_method': 'transformer' if hasattr(self, 'ner_pipeline') and self.ner_pipeline
                                       else 'spacy' if self.use_spacy else 'pattern'
                }

            except Exception as e:
                logger.error(f"Error processing text for NER: {str(e)}")
                result = {
                    'entities': {'CURRENCY': [], 'ORGANIZATION': [], 'FINANCIAL_TERM': [], 'MONEY': []},
                    'unique_entities': {'CURRENCY': [], 'ORGANIZATION': [], 'FINANCIAL_TERM': [], 'MONEY': []},
                    'statistics': {'total_entities': 0, 'currency_count': 0, 'organization_count': 0,
                                 'financial_term_count': 0, 'money_count': 0},
                    'extraction_method': 'error'
                }

            results.append(result)

        return results
