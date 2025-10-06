"""
Computer vision module for AffectRON.
Provides chart pattern recognition, document processing, and image analysis for financial data.
"""

import asyncio
import logging
import io
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import base64

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


@dataclass
class ChartPattern:
    """Detected chart pattern."""
    pattern_type: str  # 'head_and_shoulders', 'triangle', 'flag', 'wedge', etc.
    confidence: float
    start_point: Tuple[int, int]  # (x, y) coordinates
    end_point: Tuple[int, int]
    pattern_data: Dict[str, Any]
    detected_at: datetime


@dataclass
class DocumentAnalysis:
    """Document analysis result."""
    document_type: str  # 'financial_report', 'news_article', 'chart_image'
    extracted_text: str
    tables: List[pd.DataFrame]
    charts: List[ChartPattern]
    entities: Dict[str, List[str]]
    confidence: float


class FinancialComputerVision:
    """Computer vision system for financial data analysis."""

    def __init__(self):
        self.ocr_reader = None
        if OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['ro', 'en'])
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not initialize OCR: {e}")

        self.pattern_templates = self._load_pattern_templates()
        self.logger = logging.getLogger(__name__)

    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load chart pattern templates."""
        # Simplified pattern templates - in production would use trained models
        return {
            'head_and_shoulders': {
                'description': 'Head and shoulders pattern',
                'min_points': 5,
                'pattern_shape': 'M_shape'
            },
            'triangle': {
                'description': 'Triangle pattern',
                'min_points': 4,
                'pattern_shape': 'triangle'
            },
            'flag': {
                'description': 'Flag pattern',
                'min_points': 3,
                'pattern_shape': 'rectangle'
            }
        }

    async def analyze_chart_image(self, image_data: bytes) -> List[ChartPattern]:
        """Analyze chart image for patterns."""
        patterns = []

        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            # Preprocess image
            processed_image = self._preprocess_chart_image(image_array)

            # Detect patterns
            detected_patterns = await self._detect_chart_patterns(processed_image)

            for pattern in detected_patterns:
                chart_pattern = ChartPattern(
                    pattern_type=pattern['type'],
                    confidence=pattern['confidence'],
                    start_point=tuple(pattern['start_point']),
                    end_point=tuple(pattern['end_point']),
                    pattern_data=pattern['data'],
                    detected_at=datetime.now()
                )
                patterns.append(chart_pattern)

        except Exception as e:
            self.logger.error(f"Error analyzing chart image: {e}")

        return patterns

    def _preprocess_chart_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess chart image for analysis."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    async def _detect_chart_patterns(self, processed_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect chart patterns in processed image."""
        patterns = []

        # Find contours
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Analyze contour shape
            pattern_info = self._analyze_contour_shape(contour)

            if pattern_info:
                patterns.append(pattern_info)

        return patterns

    def _analyze_contour_shape(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze contour shape for pattern recognition."""
        # Get contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter < 50:  # Too small to be a pattern
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0

        # Simple pattern detection based on shape characteristics
        if 0.8 < aspect_ratio < 1.2 and area > 500:
            # Could be a triangle or similar pattern
            return {
                'type': 'triangle',
                'confidence': 0.6,
                'start_point': (x, y),
                'end_point': (x + w, y + h),
                'data': {
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'width': w,
                    'height': h
                }
            }

        elif aspect_ratio > 2.0 and area > 300:
            # Could be a flag pattern
            return {
                'type': 'flag',
                'confidence': 0.5,
                'start_point': (x, y),
                'end_point': (x + w, y + h),
                'data': {
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'width': w,
                    'height': h
                }
            }

        return None

    async def analyze_financial_document(self, file_data: bytes, filename: str) -> DocumentAnalysis:
        """Analyze financial document (PDF, image, etc.)."""
        try:
            # Determine file type
            if filename.lower().endswith('.pdf') and PDF_AVAILABLE:
                analysis = await self._analyze_pdf_document(file_data)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                analysis = await self._analyze_image_document(file_data)
            else:
                # Try to detect content type
                analysis = await self._analyze_generic_document(file_data)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing document {filename}: {e}")
            return DocumentAnalysis(
                document_type='unknown',
                extracted_text='',
                tables=[],
                charts=[],
                entities={},
                confidence=0.0
            )

    async def _analyze_pdf_document(self, file_data: bytes) -> DocumentAnalysis:
        """Analyze PDF financial document."""
        extracted_text = ""
        tables = []
        charts = []

        try:
            with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # More than just header
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)

                    # Look for chart-like elements (simplified)
                    # In production, would use more sophisticated chart detection

            # Extract entities from text
            entities = self._extract_entities_from_text(extracted_text)

            # Determine document type based on content
            document_type = self._classify_document_type(extracted_text)

            return DocumentAnalysis(
                document_type=document_type,
                extracted_text=extracted_text,
                tables=tables,
                charts=charts,
                entities=entities,
                confidence=0.8 if extracted_text else 0.3
            )

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return DocumentAnalysis(
                document_type='pdf_error',
                extracted_text='',
                tables=[],
                charts=[],
                entities={},
                confidence=0.0
            )

    async def _analyze_image_document(self, file_data: bytes) -> DocumentAnalysis:
        """Analyze image document."""
        extracted_text = ""
        tables = []
        charts = []

        try:
            # Try OCR if available
            if self.ocr_reader:
                image = Image.open(io.BytesIO(file_data))

                # Perform OCR
                results = self.ocr_reader.readtext(np.array(image))

                # Extract text from OCR results
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # Only high-confidence text
                        extracted_text += text + " "

                # Analyze for chart patterns
                chart_patterns = await self.analyze_chart_image(file_data)
                charts.extend(chart_patterns)

            # Extract entities
            entities = self._extract_entities_from_text(extracted_text)

            # Classify document type
            document_type = 'chart_image' if charts else 'text_image'

            return DocumentAnalysis(
                document_type=document_type,
                extracted_text=extracted_text,
                tables=tables,
                charts=charts,
                entities=entities,
                confidence=0.7 if extracted_text else 0.4
            )

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return DocumentAnalysis(
                document_type='image_error',
                extracted_text='',
                tables=[],
                charts=[],
                entities={},
                confidence=0.0
            )

    async def _analyze_generic_document(self, file_data: bytes) -> DocumentAnalysis:
        """Analyze generic document type."""
        # Try to detect if it's text-based
        try:
            text_content = file_data.decode('utf-8', errors='ignore')

            if len(text_content) > 100:
                entities = self._extract_entities_from_text(text_content)
                document_type = self._classify_document_type(text_content)

                return DocumentAnalysis(
                    document_type=document_type,
                    extracted_text=text_content,
                    tables=[],
                    charts=[],
                    entities=entities,
                    confidence=0.6
                )

        except Exception as e:
            self.logger.error(f"Error processing generic document: {e}")

        return DocumentAnalysis(
            document_type='unknown',
            extracted_text='',
            tables=[],
            charts=[],
            entities={},
            confidence=0.0
        )

    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text."""
        entities = {
            'currencies': [],
            'institutions': [],
            'companies': [],
            'numbers': []
        }

        if not text:
            return entities

        text_lower = text.lower()

        # Currency patterns
        currency_patterns = [
            r'\b\d+\.?\d*\s*(ron|eur|usd|gbp|chf)\b',
            r'\b(ron|eur|usd|gbp|chf)\s*\d+\.?\d*\b',
            r'\b(leu|dolar|euro|dollar|pound|franc)\b'
        ]

        for pattern in currency_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities['currencies'].extend(matches)

        # Institution patterns
        institution_patterns = [
            r'\b(bnr|banca naţională|ecb|fed|bank of england)\b',
            r'\b(ministerul finanţelor|boursa de valori|central bank)\b'
        ]

        for pattern in institution_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities['institutions'].extend(matches)

        # Number extraction (financial figures)
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, text)
        entities['numbers'] = numbers[:20]  # Limit to first 20 numbers

        return entities

    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()

        # Financial report indicators
        financial_indicators = [
            'raport financiar', 'financial report', 'balance sheet', 'income statement',
            'cash flow', 'profit and loss', 'bilanţ', 'cont de profit şi pierdere'
        ]

        # News indicators
        news_indicators = [
            'ştire', 'news', 'anunţ', 'announcement', 'press release',
            'comunicat de presă', 'article', 'articol'
        ]

        # Market data indicators
        market_indicators = [
            'cotaţii', 'quotations', 'exchange rate', 'curs valutar',
            'market data', 'price', 'preţ'
        ]

        financial_count = sum(1 for indicator in financial_indicators if indicator in text_lower)
        news_count = sum(1 for indicator in news_indicators if indicator in text_lower)
        market_count = sum(1 for indicator in market_indicators if indicator in text_lower)

        if financial_count >= news_count and financial_count >= market_count:
            return 'financial_report'
        elif news_count >= market_count:
            return 'news_article'
        elif market_count > 0:
            return 'market_data'
        else:
            return 'generic_document'

    async def extract_chart_data_from_image(self, image_data: bytes) -> Optional[pd.DataFrame]:
        """Extract numerical data from chart image."""
        try:
            # Convert to image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            # This is a simplified implementation
            # In production, would use more sophisticated methods

            # Try to detect axes and extract data points
            chart_data = self._extract_data_points(image_array)

            if chart_data:
                return pd.DataFrame(chart_data)

        except Exception as e:
            self.logger.error(f"Error extracting chart data: {e}")

        return None

    def _extract_data_points(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract data points from chart image."""
        # Simplified data point extraction
        # In production, would use computer vision techniques

        # Find non-white pixels as potential data points
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Threshold to find data points (assuming dark points on light background)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        data_points = []
        for contour in contours:
            if 5 < cv2.contourArea(contour) < 100:  # Reasonable size for data points
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                data_points.append({
                    'x': center_x,
                    'y': center_y,
                    'size': cv2.contourArea(contour)
                })

        return data_points

    def generate_chart_analysis_report(self, patterns: List[ChartPattern]) -> Dict[str, Any]:
        """Generate analysis report for detected chart patterns."""
        if not patterns:
            return {'patterns_detected': 0, 'analysis': 'No patterns detected'}

        # Count patterns by type
        pattern_counts = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

        # Find highest confidence pattern
        highest_confidence = max(patterns, key=lambda p: p.confidence)

        # Generate insights
        insights = []
        for pattern in patterns:
            if pattern.confidence > 0.7:
                insights.append(f"Strong {pattern.pattern_type} pattern detected with {pattern.confidence".2f"} confidence")

        return {
            'patterns_detected': len(patterns),
            'pattern_types': pattern_counts,
            'highest_confidence_pattern': {
                'type': highest_confidence.pattern_type,
                'confidence': highest_confidence.confidence,
                'location': {
                    'start': highest_confidence.start_point,
                    'end': highest_confidence.end_point
                }
            },
            'insights': insights,
            'analysis_timestamp': datetime.now().isoformat()
        }

    async def process_financial_chart(self, chart_image: bytes, chart_type: str = 'auto') -> Dict[str, Any]:
        """Process financial chart for analysis."""
        try:
            # Detect patterns
            patterns = await self.analyze_chart_image(chart_image)

            # Extract data points
            chart_data = await self.extract_chart_data_from_image(chart_image)

            # Generate report
            analysis_report = self.generate_chart_analysis_report(patterns)

            return {
                'status': 'success',
                'chart_type': chart_type,
                'patterns': [
                    {
                        'type': p.pattern_type,
                        'confidence': p.confidence,
                        'start_point': p.start_point,
                        'end_point': p.end_point,
                        'data': p.pattern_data
                    }
                    for p in patterns
                ],
                'data_points': chart_data.to_dict('records') if chart_data is not None else [],
                'analysis': analysis_report,
                'processed_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing financial chart: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }

    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types."""
        return ['line_chart', 'bar_chart', 'candlestick', 'area_chart', 'scatter_plot']

    def get_vision_capabilities(self) -> Dict[str, Any]:
        """Get computer vision capabilities status."""
        return {
            'ocr_available': OCR_AVAILABLE,
            'pdf_processing_available': PDF_AVAILABLE,
            'pattern_recognition_available': True,
            'supported_image_formats': ['PNG', 'JPEG', 'PDF'],
            'supported_languages': ['ro', 'en'] if OCR_AVAILABLE else []
        }


# Global computer vision instance
financial_vision = FinancialComputerVision()


async def analyze_financial_image(image_data: bytes, filename: str) -> DocumentAnalysis:
    """Analyze financial image or document."""
    return await financial_vision.analyze_financial_document(image_data, filename)


async def detect_chart_patterns(image_data: bytes) -> List[ChartPattern]:
    """Detect chart patterns in image."""
    return await financial_vision.analyze_chart_image(image_data)


def get_vision_status() -> Dict[str, Any]:
    """Get computer vision system status."""
    return {
        'vision_system_initialized': True,
        'capabilities': financial_vision.get_vision_capabilities(),
        'supported_chart_types': financial_vision.get_supported_chart_types()
    }
