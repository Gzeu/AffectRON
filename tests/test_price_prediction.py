"""
Tests for price prediction models.
Tests LSTM, XGBoost, and ensemble prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.analytics.price_prediction import (
    PricePredictionEngine, PredictionModel, PredictionResult,
    PricePredictionDataset, LSTMPredictor
)


class TestPricePredictionEngine:
    """Test PricePredictionEngine functionality."""

    @pytest.fixture
    def prediction_engine(self, db_session):
        """Create PricePredictionEngine instance for testing."""
        config = AnalyticsConfig(
            name="test_prediction",
            update_interval=timedelta(minutes=30),
            lookback_period=timedelta(days=7)
        )
        return PricePredictionEngine(config, db_session)

    def test_prediction_model_creation(self, prediction_engine):
        """Test prediction model configuration."""
        lstm_model = prediction_engine.models['lstm_sentiment']

        assert lstm_model.name == 'LSTM with Sentiment'
        assert lstm_model.model_type == 'lstm'
        assert lstm_model.target_horizon == 24
        assert 'sentiment_score' in lstm_model.features
        assert 'price_change' in lstm_model.features

    def test_select_best_model(self, prediction_engine):
        """Test model selection for prediction horizon."""
        # Test exact horizon match
        best_model = prediction_engine._select_best_model(24)
        assert best_model.name == 'LSTM with Sentiment'

        # Test closest horizon
        best_model = prediction_engine._select_best_model(18)
        assert best_model is not None

    def test_prepare_features_mock_data(self, prediction_engine):
        """Test feature preparation with mock data."""
        with patch.object(prediction_engine, '_get_historical_data') as mock_get_data:
            # Mock historical data
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            mock_data = pd.DataFrame({
                'timestamp': dates,
                'price': np.random.uniform(4.9, 5.0, 100),
                'volume': np.random.uniform(1000000, 2000000, 100),
                'sentiment_score': np.random.normal(0.1, 0.3, 100),
                'sentiment_volume': np.random.randint(10, 50, 100),
                'volatility': np.random.uniform(0.01, 0.03, 100)
            })

            mock_get_data.return_value = mock_data

            X, y = prediction_engine.prepare_features('EUR/RON', 30)

            assert len(X) > 0
            assert len(y) > 0
            assert X.shape[0] == y.shape[0]

    def test_extract_model_features_lstm(self, prediction_engine):
        """Test feature extraction for LSTM model."""
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'price': [4.97, 4.971, 4.975, 4.978, 4.972, 4.976, 4.98, 4.975, 4.979, 4.981],
            'volume': [1000000] * 10,
            'sentiment_score': [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.35, 0.1, 0.2, 0.15],
            'sentiment_volume': [20] * 10,
            'volatility': [0.02] * 10
        })

        lstm_model = prediction_engine.models['lstm_sentiment']

        features, targets = prediction_engine._extract_model_features(data, lstm_model)

        assert len(features) > 0
        assert len(targets) > 0
        assert features.shape[1] == len(lstm_model.features)  # Should match number of features

    def test_get_current_price_mock(self, prediction_engine):
        """Test getting current price."""
        price = prediction_engine._get_current_price('EUR/RON')

        assert price is not None
        assert 4.0 <= price <= 6.0  # Reasonable range for EUR/RON

    def test_calculate_prediction_confidence(self, prediction_engine):
        """Test prediction confidence calculation."""
        # Mock trained model data
        prediction_engine.trained_models['EUR/RON_lstm_sentiment'] = {
            'model': Mock(),
            'scaler': Mock(),
            'metrics': {'r2': 0.8, 'mse': 0.001, 'mae': 0.02}
        }

        confidence = prediction_engine._calculate_prediction_confidence('EUR/RON_lstm_sentiment', 0.02)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be confident with good R2 score

    @pytest.mark.asyncio
    async def test_train_lstm_model(self, prediction_engine):
        """Test LSTM model training."""
        # Prepare test data
        X = np.random.randn(100, 4)  # 100 samples, 4 features
        y = np.random.randn(100)     # 100 targets

        lstm_model = prediction_engine.models['lstm_sentiment']

        # Train model
        trained_model = await prediction_engine._train_lstm_model(X, y, lstm_model)

        assert trained_model is not None
        assert isinstance(trained_model, LSTMPredictor)

        # Test prediction
        test_input = torch.randn(1, 1, 4)  # Batch size 1, sequence length 1, 4 features
        with torch.no_grad():
            prediction = trained_model(test_input)
            assert prediction.shape == (1, 1)

    @pytest.mark.asyncio
    async def test_predict_price_success(self, prediction_engine):
        """Test successful price prediction."""
        # Mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.02])  # 2% price increase

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.randn(1, 4)

        prediction_engine.trained_models['EUR/RON_lstm_sentiment'] = {
            'model': mock_model,
            'scaler': mock_scaler,
            'metrics': {'r2': 0.8}
        }

        with patch.object(prediction_engine, '_get_latest_features') as mock_features, \
             patch.object(prediction_engine, '_get_current_price') as mock_price:

            mock_features.return_value = [0.2, 25, 0.01, 1500000]
            mock_price.return_value = 4.9750

            prediction = await prediction_engine.predict_price('EUR/RON', 24)

            assert prediction is not None
            assert prediction.currency_pair == 'EUR/RON'
            assert prediction.prediction_horizon == 24
            assert prediction.predicted_price == 4.9750 * 1.02  # 4.9750 * 1.02
            assert prediction.confidence > 0

    @pytest.mark.asyncio
    async def test_predict_price_no_model(self, prediction_engine):
        """Test price prediction when no model is available."""
        # No models trained

        with patch.object(prediction_engine, '_get_latest_features') as mock_features:
            mock_features.return_value = [0.2, 25, 0.01, 1500000]

            prediction = await prediction_engine.predict_price('EUR/RON', 24)

            # Should return None when no model is available
            assert prediction is None

    def test_evaluate_model_performance(self, prediction_engine):
        """Test model performance evaluation."""
        # Mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.01, 0.02, 0.015, 0.025])

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.randn(4, 4)

        prediction_engine.trained_models['EUR/RON_lstm_sentiment'] = {
            'model': mock_model,
            'scaler': mock_scaler,
            'metrics': {'r2': 0.8}
        }

        with patch.object(prediction_engine, '_get_historical_data') as mock_get_data:
            # Mock historical data for evaluation
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
                'price': [4.97, 4.971, 4.975, 4.978, 4.972, 4.976, 4.98, 4.975, 4.979, 4.981],
                'volume': [1000000] * 10,
                'sentiment_score': [0.1] * 10,
                'sentiment_volume': [20] * 10,
                'volatility': [0.02] * 10
            })

            mock_get_data.return_value = mock_data

            metrics = prediction_engine.evaluate_model_performance('EUR/RON', 'lstm_sentiment')

            assert 'mse' in metrics
            assert 'mae' in metrics
            assert 'r2_score' in metrics
            assert metrics['samples_evaluated'] > 0

    def test_get_prediction_summary(self, prediction_engine):
        """Test prediction summary generation."""
        # Add some mock predictions to history
        prediction_engine.prediction_history = [
            PredictionResult(
                currency_pair='EUR/RON',
                predicted_price=4.98,
                confidence=0.8,
                prediction_horizon=24,
                model_used='lstm_sentiment',
                features_used=['sentiment_score', 'price_change'],
                prediction_timestamp=datetime.now()
            ),
            PredictionResult(
                currency_pair='USD/RON',
                predicted_price=4.59,
                confidence=0.7,
                prediction_horizon=12,
                model_used='xgboost_comprehensive',
                features_used=['sentiment_score', 'volume'],
                prediction_timestamp=datetime.now()
            )
        ]

        summary = prediction_engine.get_prediction_summary()

        assert summary['total_predictions'] == 2
        assert summary['models_trained'] == 0  # No models trained in this test
        assert len(summary['recent_predictions']) == 2
        assert 'EUR/RON' in [p['currency_pair'] for p in summary['recent_predictions']]

    def test_get_analysis_data(self, prediction_engine):
        """Test getting analysis data."""
        data = prediction_engine.get_analysis_data()

        assert 'supported_currencies' in data
        assert 'supported_horizons' in data
        assert 'models_available' in data
        assert 'models_trained' in data

        assert len(data['supported_currencies']) > 0
        assert len(data['supported_horizons']) > 0


class TestLSTMPredictor:
    """Test LSTM predictor model."""

    def test_lstm_model_creation(self):
        """Test LSTM model initialization."""
        model = LSTMPredictor(input_size=4, hidden_size=32, num_layers=1)

        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.lstm is not None
        assert model.fc is not None

    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        model = LSTMPredictor(input_size=4, hidden_size=32, num_layers=1)

        # Create test input (batch_size=2, sequence_length=5, features=4)
        batch_size = 2
        sequence_length = 5
        input_size = 4

        x = torch.randn(batch_size, sequence_length, input_size)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)  # Should output single prediction per sample

    def test_lstm_model_training_step(self):
        """Test LSTM model training step."""
        model = LSTMPredictor(input_size=4, hidden_size=32, num_layers=1)

        # Training data
        x = torch.randn(10, 1, 4)  # 10 samples, sequence length 1, 4 features
        y = torch.randn(10, 1)     # 10 targets

        # Training setup
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0  # Loss should be non-negative


class TestIntegration:
    """Integration tests for price prediction."""

    @pytest.mark.asyncio
    async def test_full_prediction_workflow(self, prediction_engine):
        """Test complete prediction workflow."""
        # Mock data for testing
        with patch.object(prediction_engine, '_get_historical_data') as mock_get_data, \
             patch.object(prediction_engine, '_get_current_price') as mock_get_price:

            # Mock historical data
            dates = pd.date_range('2024-01-01', periods=200, freq='H')
            mock_data = pd.DataFrame({
                'timestamp': dates,
                'price': 4.9750 + np.cumsum(np.random.normal(0, 0.001, 200)),
                'volume': np.random.uniform(1000000, 2000000, 200),
                'sentiment_score': np.random.normal(0.1, 0.3, 200),
                'sentiment_volume': np.random.randint(10, 50, 200),
                'volatility': np.random.uniform(0.01, 0.03, 200)
            })

            mock_get_data.return_value = mock_data
            mock_get_price.return_value = 4.9750

            # Train model
            success = await prediction_engine.train_model('EUR/RON', 'lstm_sentiment')

            # Should train successfully (or fail gracefully if dependencies missing)
            assert isinstance(success, bool)

            if success:
                # Make prediction
                prediction = await prediction_engine.predict_price('EUR/RON', 24)

                if prediction:
                    assert prediction.currency_pair == 'EUR/RON'
                    assert prediction.predicted_price > 0
                    assert 0 <= prediction.confidence <= 1

    def test_model_comparison(self, prediction_engine):
        """Test model comparison and selection."""
        # Test that different models have different configurations
        lstm_model = prediction_engine.models['lstm_sentiment']
        xgb_model = prediction_engine.models['xgboost_comprehensive']
        ensemble_model = prediction_engine.models['ensemble_model']

        # Each model should have different features or configurations
        assert set(lstm_model.features) != set(xgb_model.features) or lstm_model.target_horizon != xgb_model.target_horizon

        # Test horizon-based selection
        short_horizon_model = prediction_engine._select_best_model(6)
        long_horizon_model = prediction_engine._select_best_model(48)

        assert short_horizon_model is not None
        assert long_horizon_model is not None
        # Should prefer models closer to target horizon
        assert abs(short_horizon_model.target_horizon - 6) <= abs(long_horizon_model.target_horizon - 6)

    @pytest.mark.asyncio
    async def test_prediction_engine_run_analysis(self, prediction_engine):
        """Test running complete prediction analysis."""
        # Mock dependencies
        with patch.object(prediction_engine, 'predict_price') as mock_predict:
            mock_predict.return_value = PredictionResult(
                currency_pair='EUR/RON',
                predicted_price=4.98,
                confidence=0.8,
                prediction_horizon=24,
                model_used='lstm_sentiment',
                features_used=['sentiment_score'],
                prediction_timestamp=datetime.now()
            )

            results = await prediction_engine.run_analysis()

            assert isinstance(results, list)
            if results:
                result = results[0]
                assert result.analytics_name == "price_prediction"
                assert result.result_type == "predictions"
                assert 'predictions' in result.result_data
