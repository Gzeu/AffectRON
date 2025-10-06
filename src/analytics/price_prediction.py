"""
Advanced price prediction models for AffectRON.
Uses machine learning to predict currency price movements based on sentiment and market data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from ..analytics.base import BaseAnalytics, AnalyticsConfig, AnalyticsResult


@dataclass
class PredictionModel:
    """Price prediction model configuration."""
    name: str
    model_type: str  # 'lstm', 'transformer', 'xgboost', 'ensemble'
    features: List[str]
    target_horizon: int  # Hours ahead to predict
    training_data_days: int = 90
    validation_split: float = 0.2
    hyperparameters: Dict[str, Any] = None


@dataclass
class PredictionResult:
    """Price prediction result."""
    currency_pair: str
    predicted_price: float
    confidence: float
    prediction_horizon: int  # Hours
    model_used: str
    features_used: List[str]
    prediction_timestamp: datetime
    actual_price: Optional[float] = None
    error: Optional[float] = None


class PricePredictionDataset(Dataset):
    """Dataset for price prediction models."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    """LSTM-based price prediction model."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and linear layer
        output = self.dropout(last_output)
        prediction = self.fc(output)

        return prediction


class PricePredictionEngine(BaseAnalytics):
    """Advanced price prediction using ML models."""

    def __init__(self, config: AnalyticsConfig, db_session):
        super().__init__(config, db_session)

        # Available prediction models
        self.models = {
            'lstm_sentiment': PredictionModel(
                name='LSTM with Sentiment',
                model_type='lstm',
                features=['sentiment_score', 'sentiment_volume', 'price_change', 'volume'],
                target_horizon=24,
                hyperparameters={
                    'hidden_size': 64,
                    'num_layers': 2,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            ),
            'xgboost_comprehensive': PredictionModel(
                name='XGBoost Comprehensive',
                model_type='xgboost',
                features=['sentiment_score', 'price_change', 'volume', 'volatility', 'correlation'],
                target_horizon=12,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            ),
            'ensemble_model': PredictionModel(
                name='Ensemble Model',
                model_type='ensemble',
                features=['sentiment_score', 'price_change', 'volume', 'technical_indicators'],
                target_horizon=6,
                hyperparameters={
                    'lstm_weight': 0.4,
                    'xgboost_weight': 0.4,
                    'linear_weight': 0.2
                }
            )
        }

        self.trained_models = {}
        self.scalers = {}

        # Performance tracking
        self.prediction_history: List[PredictionResult] = []

        self.logger = logging.getLogger(__name__)

    def prepare_features(self, currency_pair: str, days: int = 90) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        # Get historical data
        data = self._get_historical_data(currency_pair, days)

        if data.empty:
            return np.array([]), np.array([])

        # Extract features based on model requirements
        features = []
        targets = []

        for model_name, model in self.models.items():
            model_features, model_targets = self._extract_model_features(data, model)
            if len(model_features) > 0:
                features.append(model_features)
                targets.append(model_targets)

        if not features:
            return np.array([]), np.array([])

        # Combine features from all models
        X = np.concatenate(features, axis=1) if len(features) > 1 else features[0]
        y = targets[0]  # Use first model's targets for now

        return X, y

    def _get_historical_data(self, currency_pair: str, days: int) -> pd.DataFrame:
        """Get historical market and sentiment data."""
        # This would query the database for historical data
        # For now, return mock data structure

        dates = pd.date_range(
            datetime.now() - timedelta(days=days),
            datetime.now(),
            freq='H'
        )

        # Generate realistic market data
        base_price = 4.9750 if 'EUR' in currency_pair else 4.5800 if 'USD' in currency_pair else 4.8000

        # Simulate price movements
        price_changes = np.random.normal(0, 0.001, len(dates)).cumsum()
        prices = base_price + price_changes

        # Generate sentiment data
        sentiment_scores = np.random.normal(0.1, 0.3, len(dates))

        data = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.uniform(1000000, 5000000, len(dates)),
            'sentiment_score': sentiment_scores,
            'sentiment_volume': np.random.randint(10, 100, len(dates)),
            'volatility': np.random.uniform(0.01, 0.05, len(dates))
        })

        return data

    def _extract_model_features(self, data: pd.DataFrame, model: PredictionModel) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for specific model."""
        if data.empty:
            return np.array([]), np.array([])

        features = []
        targets = []

        # Feature engineering based on model requirements
        for i in range(len(data) - model.target_horizon):
            feature_window = []

            for feature in model.features:
                if feature == 'sentiment_score':
                    feature_window.append(data.iloc[i]['sentiment_score'])
                elif feature == 'sentiment_volume':
                    feature_window.append(data.iloc[i]['sentiment_volume'])
                elif feature == 'price_change':
                    # Calculate price change from previous periods
                    if i > 0:
                        price_change = (data.iloc[i]['price'] - data.iloc[i-1]['price']) / data.iloc[i-1]['price']
                    else:
                        price_change = 0
                    feature_window.append(price_change)
                elif feature == 'volume':
                    feature_window.append(data.iloc[i]['volume'])
                elif feature == 'volatility':
                    feature_window.append(data.iloc[i]['volatility'])

            if len(feature_window) == len(model.features):
                features.append(feature_window)

                # Target: price change after target horizon
                target_price = data.iloc[i + model.target_horizon]['price']
                current_price = data.iloc[i]['price']
                price_change = (target_price - current_price) / current_price
                targets.append(price_change)

        return np.array(features), np.array(targets)

    async def train_model(self, currency_pair: str, model_name: str) -> bool:
        """Train prediction model for currency pair."""
        if model_name not in self.models:
            self.logger.error(f"Unknown model: {model_name}")
            return False

        model = self.models[model_name]

        try:
            self.logger.info(f"Training {model_name} for {currency_pair}")

            # Prepare data
            X, y = self.prepare_features(currency_pair, model.training_data_days)

            if len(X) == 0 or len(y) == 0:
                self.logger.error(f"Insufficient data for training {model_name}")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=model.validation_split, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model based on type
            if model.model_type == 'lstm':
                trained_model = await self._train_lstm_model(X_train_scaled, y_train, model)
            elif model.model_type == 'xgboost':
                trained_model = await self._train_xgboost_model(X_train_scaled, y_train, model)
            elif model.model_type == 'ensemble':
                trained_model = await self._train_ensemble_model(X_train_scaled, y_train, model)
            else:
                self.logger.error(f"Unsupported model type: {model.model_type}")
                return False

            # Evaluate model
            test_predictions = trained_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, test_predictions)
            mae = mean_absolute_error(y_test, test_predictions)
            r2 = r2_score(y_test, test_predictions)

            self.logger.info(f"Model {model_name} trained - MSE: {mse".6f"}, MAE: {mae".6f"}, R2: {r2".4f"}")

            # Store trained model
            self.trained_models[f"{currency_pair}_{model_name}"] = {
                'model': trained_model,
                'scaler': scaler,
                'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
                'training_date': datetime.now()
            }

            # Save model to disk
            await self._save_model(currency_pair, model_name, trained_model, scaler)

            return True

        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
            return False

    async def _train_lstm_model(self, X: np.ndarray, y: np.ndarray, model: PredictionModel) -> LSTMPredictor:
        """Train LSTM model."""
        # Reshape for LSTM (batch_size, sequence_length, features)
        # For simplicity, we'll use sequence length of 1
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Create model
        input_size = X.shape[1]
        lstm_model = LSTMPredictor(
            input_size=input_size,
            hidden_size=model.hyperparameters.get('hidden_size', 64),
            num_layers=model.hyperparameters.get('num_layers', 2)
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            lstm_model.parameters(),
            lr=model.hyperparameters.get('learning_rate', 0.001)
        )

        # Train for specified epochs
        epochs = model.hyperparameters.get('epochs', 100)

        for epoch in range(epochs):
            lstm_model.train()

            # Forward pass
            outputs = lstm_model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                self.logger.debug(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item()".6f"}")

        return lstm_model

    async def _train_xgboost_model(self, X: np.ndarray, y: np.ndarray, model: PredictionModel) -> Any:
        """Train XGBoost model."""
        try:
            from xgboost import XGBRegressor

            xgb_model = XGBRegressor(
                n_estimators=model.hyperparameters.get('n_estimators', 100),
                max_depth=model.hyperparameters.get('max_depth', 6),
                learning_rate=model.hyperparameters.get('learning_rate', 0.1),
                random_state=42
            )

            xgb_model.fit(X, y)

            return xgb_model

        except ImportError:
            self.logger.error("XGBoost not available")
            return None

    async def _train_ensemble_model(self, X: np.ndarray, y: np.ndarray, model: PredictionModel) -> Dict:
        """Train ensemble model combining multiple approaches."""
        # This would train multiple models and combine them
        # For now, return a simple linear model as placeholder

        from sklearn.linear_model import LinearRegression

        linear_model = LinearRegression()
        linear_model.fit(X, y)

        return {
            'linear_model': linear_model,
            'weights': model.hyperparameters
        }

    async def predict_price(self, currency_pair: str, horizon_hours: int = 24) -> Optional[PredictionResult]:
        """Predict price for currency pair."""
        # Find best model for this horizon
        best_model = self._select_best_model(horizon_hours)

        if not best_model:
            return None

        model_key = f"{currency_pair}_{best_model.name}"

        if model_key not in self.trained_models:
            # Try to train model if not available
            await self.train_model(currency_pair, best_model.name)

            if model_key not in self.trained_models:
                return None

        trained_model_data = self.trained_models[model_key]
        model = trained_model_data['model']
        scaler = trained_model_data['scaler']

        # Get latest features
        latest_features = self._get_latest_features(currency_pair, best_model)

        if latest_features is None:
            return None

        # Scale features
        latest_features_scaled = scaler.transform([latest_features])

        # Make prediction
        if best_model.model_type == 'lstm':
            # LSTM prediction
            features_tensor = torch.FloatTensor(latest_features_scaled).unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                prediction = model(features_tensor).item()
        elif best_model.model_type == 'xgboost':
            prediction = model.predict(latest_features_scaled)[0]
        elif best_model.model_type == 'ensemble':
            # Ensemble prediction
            linear_pred = model['linear_model'].predict(latest_features_scaled)[0]
            prediction = linear_pred  # Simplified
        else:
            return None

        # Get current price for reference
        current_price = self._get_current_price(currency_pair)

        if current_price is None:
            return None

        # Calculate predicted price
        predicted_price = current_price * (1 + prediction)

        # Calculate confidence based on model performance
        confidence = self._calculate_prediction_confidence(model_key, prediction)

        return PredictionResult(
            currency_pair=currency_pair,
            predicted_price=predicted_price,
            confidence=confidence,
            prediction_horizon=horizon_hours,
            model_used=best_model.name,
            features_used=best_model.features,
            prediction_timestamp=datetime.now()
        )

    def _select_best_model(self, horizon_hours: int) -> Optional[PredictionModel]:
        """Select best model for given prediction horizon."""
        best_model = None
        best_score = 0

        for model in self.models.values():
            # Score based on how close target horizon matches model horizon
            horizon_match = 1 - abs(model.target_horizon - horizon_hours) / max(model.target_horizon, horizon_hours)

            if horizon_match > best_score:
                best_score = horizon_match
                best_model = model

        return best_model

    def _get_latest_features(self, currency_pair: str, model: PredictionModel) -> Optional[List[float]]:
        """Get latest features for prediction."""
        # Get recent data for feature extraction
        recent_data = self._get_historical_data(currency_pair, days=7)  # Last week

        if recent_data.empty or len(recent_data) < 2:
            return None

        # Extract features similar to training
        features = []

        for feature in model.features:
            if feature == 'sentiment_score':
                features.append(recent_data.iloc[-1]['sentiment_score'])
            elif feature == 'sentiment_volume':
                features.append(recent_data.iloc[-1]['sentiment_volume'])
            elif feature == 'price_change':
                # Latest price change
                current_price = recent_data.iloc[-1]['price']
                previous_price = recent_data.iloc[-2]['price']
                features.append((current_price - previous_price) / previous_price)
            elif feature == 'volume':
                features.append(recent_data.iloc[-1]['volume'])
            elif feature == 'volatility':
                features.append(recent_data.iloc[-1]['volatility'])

        return features if len(features) == len(model.features) else None

    def _get_current_price(self, currency_pair: str) -> Optional[float]:
        """Get current market price."""
        # This would query real-time market data
        # For now, return mock price based on currency pair
        price_map = {
            'EUR/RON': 4.9750,
            'USD/RON': 4.5800,
            'GBP/RON': 5.8500,
            'CHF/RON': 5.2000
        }

        return price_map.get(currency_pair)

    def _calculate_prediction_confidence(self, model_key: str, prediction: float) -> float:
        """Calculate confidence for prediction."""
        if model_key not in self.trained_models:
            return 0.0

        model_data = self.trained_models[model_key]
        metrics = model_data['metrics']

        # Base confidence on R2 score
        r2_score = metrics.get('r2', 0)

        # Adjust based on prediction magnitude (extreme predictions are less confident)
        magnitude_penalty = min(abs(prediction) * 2, 0.3)

        confidence = max(0.1, r2_score - magnitude_penalty)

        return min(confidence, 0.95)

    async def _save_model(self, currency_pair: str, model_name: str, model: Any, scaler: Any):
        """Save trained model to disk."""
        model_dir = f"models/prediction/{currency_pair}"
        import os

        os.makedirs(model_dir, exist_ok=True)

        # Save model
        if hasattr(model, 'save'):
            model.save(f"{model_dir}/{model_name}.json")
        else:
            joblib.dump(model, f"{model_dir}/{model_name}.pkl")

        # Save scaler
        joblib.dump(scaler, f"{model_dir}/{model_name}_scaler.pkl")

    async def load_model(self, currency_pair: str, model_name: str) -> bool:
        """Load trained model from disk."""
        try:
            model_path = f"models/prediction/{currency_pair}/{model_name}.pkl"
            scaler_path = f"models/prediction/{currency_pair}/{model_name}_scaler.pkl"

            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                return False

            # Load model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            self.trained_models[f"{currency_pair}_{model_name}"] = {
                'model': model,
                'scaler': scaler,
                'loaded_from_disk': True
            }

            return True

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return False

    def evaluate_model_performance(self, currency_pair: str, model_name: str) -> Dict[str, float]:
        """Evaluate model performance on recent data."""
        model_key = f"{currency_pair}_{model_name}"

        if model_key not in self.trained_models:
            return {'error': 'Model not trained'}

        # Get recent data for evaluation
        test_data = self._get_historical_data(currency_pair, days=30)

        if test_data.empty:
            return {'error': 'Insufficient data'}

        model = self.models[model_name]
        X, y = self._extract_model_features(test_data, model)

        if len(X) == 0:
            return {'error': 'Cannot extract features'}

        trained_model_data = self.trained_models[model_key]
        model_obj = trained_model_data['model']
        scaler = trained_model_data['scaler']

        # Scale features and make predictions
        X_scaled = scaler.transform(X)
        predictions = model_obj.predict(X_scaled)

        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(mse),
            'mean_absolute_percentage_error': np.mean(np.abs((y - predictions) / y)) * 100,
            'samples_evaluated': len(y)
        }

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions and model performance."""
        summary = {
            'total_predictions': len(self.prediction_history),
            'models_trained': len(self.trained_models),
            'available_models': list(self.models.keys()),
            'recent_predictions': []
        }

        # Add recent predictions
        for prediction in self.prediction_history[-10:]:  # Last 10 predictions
            summary['recent_predictions'].append({
                'currency_pair': prediction.currency_pair,
                'predicted_price': prediction.predicted_price,
                'confidence': prediction.confidence,
                'model_used': prediction.model_used,
                'timestamp': prediction.prediction_timestamp.isoformat()
            })

        # Add model performance summary
        model_performance = {}
        for model_key, model_data in self.trained_models.items():
            currency_pair = model_key.split('_')[0]
            model_name = '_'.join(model_key.split('_')[1:])

            if model_name not in model_performance:
                model_performance[model_name] = []

            model_performance[model_name].append({
                'currency_pair': currency_pair,
                'metrics': model_data.get('metrics', {})
            })

        summary['model_performance'] = model_performance

        return summary

    async def run_analysis(self) -> List[AnalyticsResult]:
        """Run price prediction analysis."""
        results = []

        try:
            # Generate predictions for major currency pairs
            currency_pairs = ['EUR/RON', 'USD/RON', 'GBP/RON']

            predictions = []

            for pair in currency_pairs:
                for horizon in [6, 12, 24]:  # Different time horizons
                    prediction = await self.predict_price(pair, horizon)
                    if prediction:
                        predictions.append(prediction)

                        # Store in history
                        self.prediction_history.append(prediction)

            # Create analytics result
            if predictions:
                result = AnalyticsResult(
                    analytics_name="price_prediction",
                    result_type="predictions",
                    insights={
                        'predictions': [
                            {
                                'currency_pair': p.currency_pair,
                                'predicted_price': p.predicted_price,
                                'confidence': p.confidence,
                                'horizon_hours': p.prediction_horizon,
                                'model': p.model_used
                            }
                            for p in predictions
                        ],
                        'summary': self.get_prediction_summary()
                    },
                    confidence=np.mean([p.confidence for p in predictions]) if predictions else 0.0
                )

                results.append(result)

        except Exception as e:
            self.logger.error(f"Error in price prediction analysis: {e}")

        return results

    def get_analysis_data(self) -> Dict[str, Any]:
        """Get data for price prediction analysis."""
        return {
            'supported_currencies': ['EUR', 'USD', 'GBP', 'CHF'],
            'supported_horizons': [6, 12, 24, 48],
            'models_available': list(self.models.keys()),
            'models_trained': len(self.trained_models)
        }
