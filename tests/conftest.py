"""
Test configuration for AffectRON.
Comprehensive testing setup for all components.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.api.main import create_application
from src.models import Base


# Test database configuration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///./test_affectron.db")

# Create test engine and session
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in TEST_DATABASE_URL else {}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Create test database tables."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Create a fresh database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()

    # Configure session to use the connection with transaction
    session = TestSessionLocal(bind=connection)

    yield session

    # Rollback transaction and close connection
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    app = create_application()
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    # Database
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")  # Use different DB for tests

    # Security
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only-32-chars-min")

    # Environment
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DEBUG", "true")

    # Disable external API calls in tests
    monkeypatch.setenv("MOCK_EXTERNAL_APIS", "true")


@pytest.fixture
def sample_extracted_data(db_session):
    """Create sample extracted data for testing."""
    from src.models import DataSource, ExtractedData

    # Create test data source
    data_source = DataSource(
        name="Test News Source",
        source_type="news",
        url="https://test.com/rss",
        is_active=True
    )
    db_session.add(data_source)
    db_session.commit()

    # Create sample extracted data
    extracted_data = ExtractedData(
        source_id=data_source.id,
        content="Test financial news content about RON currency trends.",
        title="RON Exchange Rate Update",
        url="https://test.com/news/1",
        metadata='{"author": "Test Author", "category": "finance"}'
    )
    db_session.add(extracted_data)
    db_session.commit()

    return extracted_data


@pytest.fixture
def sample_sentiment_data(db_session, sample_extracted_data):
    """Create sample sentiment analysis data."""
    from src.models import SentimentAnalysis

    sentiment = SentimentAnalysis(
        data_id=sample_extracted_data.id,
        model_name="test_model",
        sentiment_label="positive",
        sentiment_score=0.7,
        confidence_score=0.85,
        entities='{"currencies": ["RON"], "organizations": ["BNR"]}'
    )
    db_session.add(sentiment)
    db_session.commit()

    return sentiment


# Test utilities
def assert_sentiment_in_range(score, min_val=-1.0, max_val=1.0):
    """Assert sentiment score is within valid range."""
    assert min_val <= score <= max_val, f"Sentiment score {score} not in range [{min_val}, {max_val}]"


def assert_confidence_in_range(confidence, min_val=0.0, max_val=1.0):
    """Assert confidence score is within valid range."""
    assert min_val <= confidence <= max_val, f"Confidence score {confidence} not in range [{min_val}, {max_val}]"
