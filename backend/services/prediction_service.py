"""
Prediction Service
Handles ML model predictions with ensemble support
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from backend.core.database import get_database
from backend.services.prediction_pipeline import AQIPredictor


class PredictionService:
    """Service for generating AQI predictions"""

    def __init__(self):
        """Initialize prediction service"""
        self.predictor = AQIPredictor()
        self.db = get_database()

    def predict(
        self, location: str, hours_ahead: int = 72, model_name: str = "ensemble"
    ) -> List[Dict]:
        """
        Generate predictions for specified location

        Args:
            location: City name
            hours_ahead: Number of hours to predict
            model_name: Model to use (ensemble, xgboost, random_forest, lstm)

        Returns:
            List of prediction dictionaries
        """
        try:
            # Get recent data for the location
            df = self.db.get_recent_data(hours=168, table="live_data", location=location)

            if df.empty or len(df) < 50:
                raise ValueError(f"Insufficient data for {location}. Need at least 50 records.")

            # Prepare features
            latest = df.iloc[-1]

            # Generate predictions based on model
            predictions = []
            prediction_timestamp = datetime.now()

            for hour in range(1, hours_ahead + 1):
                target_timestamp = prediction_timestamp + timedelta(hours=hour)

                if model_name == "ensemble":
                    # Use ensemble prediction
                    predicted_aqi = self._predict_ensemble(df, hour)
                elif model_name == "xgboost":
                    predicted_aqi = self._predict_single(df, hour, "xgboost")
                elif model_name == "random_forest":
                    predicted_aqi = self._predict_single(df, hour, "random_forest")
                elif model_name == "lstm":
                    predicted_aqi = self._predict_lstm(df, hour)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # Determine category
                category = self._get_aqi_category(predicted_aqi)

                # Create prediction object
                prediction = {
                    "prediction_timestamp": prediction_timestamp,
                    "target_timestamp": target_timestamp,
                    "location": location,
                    "predicted_aqi": float(predicted_aqi),
                    "predicted_category": category,
                    "model_name": model_name,
                    "confidence_score": 0.85,  # Placeholder
                }

                predictions.append(prediction)

                # Store in database
                self.db.insert_prediction(
                    prediction_timestamp=prediction_timestamp,
                    target_timestamp=target_timestamp,
                    predicted_aqi=float(predicted_aqi),
                    model_name=model_name,
                    predicted_category=category,
                    confidence_score=0.85,
                )

            return predictions

        except Exception as e:
            print(f"Prediction error: {e}")
            return []

    def _predict_ensemble(self, df: pd.DataFrame, hours_ahead: int) -> float:
        """Generate ensemble prediction"""
        try:
            # Use existing predictor
            features = self.predictor.prepare_features(df)

            if len(features) == 0:
                return df["AQI"].iloc[-1]  # Fallback to last known value

            # Get predictions from both models
            xgb_pred = self.predictor.xgboost_model.predict(features.iloc[-1:].values)[0]
            rf_pred = self.predictor.random_forest_model.predict(features.iloc[-1:].values)[0]

            # Ensemble average
            return (xgb_pred + rf_pred) / 2

        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return df["AQI"].iloc[-1]

    def _predict_single(self, df: pd.DataFrame, hours_ahead: int, model_name: str) -> float:
        """Generate prediction from single model"""
        try:
            features = self.predictor.prepare_features(df)

            if len(features) == 0:
                return df["AQI"].iloc[-1]

            if model_name == "xgboost":
                return self.predictor.xgboost_model.predict(features.iloc[-1:].values)[0]
            else:
                return self.predictor.random_forest_model.predict(features.iloc[-1:].values)[0]

        except Exception as e:
            print(f"Single model prediction error: {e}")
            return df["AQI"].iloc[-1]

    def _predict_lstm(self, df: pd.DataFrame, hours_ahead: int) -> float:
        """Generate LSTM prediction (placeholder)"""
        # TODO: Implement LSTM model prediction
        # For now, return simple moving average
        return df["AQI"].tail(24).mean()

    def _get_aqi_category(self, aqi: float) -> str:
        """Get AQI category from value"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
