"""
Prediction Pipeline for Air Quality Forecasting
Implements recursive 3-day ahead AQI forecasting using trained models
"""

import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AQIPredictor:
    """Real-time AQI prediction with recursive forecasting"""

    def __init__(self, models_dir="models"):
        """Initialize predictor with trained models"""
        # Fix path to work from backend directory
        import os
        if not os.path.isabs(models_dir):
            # Get project root (parent of backend)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            models_dir = os.path.join(project_root, models_dir)
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        self.lstm_lookback = None
        self._load_models()

    def _load_models(self):
        """Load all trained models"""
        print("Loading models...")

        try:
            # Load Random Forest
            with open(f"{self.models_dir}/random_forest_model.pkl", "rb") as f:
                self.models["random_forest"] = pickle.load(f)
            print("Random Forest loaded")
        except Exception as e:
            print(f"WARNING: Random Forest not loaded: {e}")

        try:
            # Load XGBoost
            # Try pickle first (preferred for consistent environment)
            try:
                with open(f"{self.models_dir}/xgboost_model.pkl", "rb") as f:
                    self.models["xgboost"] = pickle.load(f)
                print("XGBoost loaded (pickle)")
            except FileNotFoundError:
                # Fallback to legacy JSON
                xgb_model = xgb.XGBRegressor()
                xgb_model.load_model(f"{self.models_dir}/xgboost_model.json")
                self.models["xgboost"] = xgb_model
                print("XGBoost loaded (JSON)")
        except Exception as e:
            print(f"WARNING: XGBoost not loaded: {e}")

        if TENSORFLOW_AVAILABLE:
            try:
                # Try loading Keras 3 format first, then fall back to H5
                lstm_path_keras = f"{self.models_dir}/lstm_model.keras"
                lstm_path_h5 = f"{self.models_dir}/lstm_model.h5"
                
                if os.path.exists(lstm_path_keras):
                    self.models["lstm"] = keras.models.load_model(lstm_path_keras)
                    print("LSTM loaded (Keras 3 format)")
                elif os.path.exists(lstm_path_h5):
                    self.models["lstm"] = keras.models.load_model(lstm_path_h5, compile=False)
                    # Recompile with proper metrics
                    self.models["lstm"].compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss="mse",
                        metrics=["mae"]
                    )
                    print("LSTM loaded (H5 format, recompiled)")
                else:
                    raise FileNotFoundError("LSTM model not found")

                # Load LSTM config
                with open(f"{self.models_dir}/lstm_config.json", "r") as f:
                    config = json.load(f)
                    self.lstm_lookback = config["lookback"]
            except Exception as e:
                print(f"⚠ LSTM not loaded: {e}")
        else:
            print("WARNING: LSTM not available - TensorFlow not installed")

        # Load scaler
        with open(f"{self.models_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded")

        # Load feature columns
        with open(f"{self.models_dir}/feature_columns.pkl", "rb") as f:
            self.feature_cols = pickle.load(f)
        
        # Identify AQI column index for exclusion during prediction (Scaler needs it, Model doesn't)
        self.aqi_index = None
        if "AQI" in self.feature_cols:
            self.aqi_index = self.feature_cols.index("AQI")
            print(f"Target 'AQI' found at index {self.aqi_index} - will be handled during prediction")
            
        print(f"Feature columns loaded ({len(self.feature_cols)} features)")

    def prepare_features_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from raw data (temporal, lag, rolling)
        Similar to preprocessing notebook
        """
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        df = df.sort_index()

        # Temporal features
        df["Hour"] = df.index.hour
        df["DayOfWeek"] = df.index.dayofweek
        df["Month"] = df.index.month
        df["DayOfMonth"] = df.index.day
        df["DayOfYear"] = df.index.dayofyear
        df["WeekOfYear"] = df.index.isocalendar().week
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

        # Cyclical features
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # Check if we have required columns
        required_cols = ["CO", "NOx", "temperature"]
        if not all(col in df.columns for col in required_cols):
            print("Warning: Missing required columns for feature generation")
            return df

        # Lag features
        lag_hours = [1, 2, 3, 6, 12, 24]
        for lag in lag_hours:
            if "AQI" in df.columns:
                df[f"AQI_lag_{lag}h"] = df["AQI"].shift(lag)
            df[f"CO_lag_{lag}h"] = df["CO"].shift(lag)
            df[f"NOx_lag_{lag}h"] = df["NOx"].shift(lag)
            df[f"Temp_lag_{lag}h"] = df["temperature"].shift(lag)

        # Rolling statistics
        windows = [6, 12, 24]
        for window in windows:
            if "AQI" in df.columns:
                df[f"AQI_rolling_mean_{window}h"] = df["AQI"].rolling(window=window).mean()
                df[f"AQI_rolling_std_{window}h"] = df["AQI"].rolling(window=window).std()
            df[f"CO_rolling_mean_{window}h"] = df["CO"].rolling(window=window).mean()
            df[f"NOx_rolling_mean_{window}h"] = df["NOx"].rolling(window=window).mean()

        # Rate of change
        if "AQI" in df.columns:
            df["AQI_change_1h"] = df["AQI"].diff(1)
            df["AQI_change_3h"] = df["AQI"].diff(3)
        df["Temp_change_1h"] = df["temperature"].diff(1)
        if "humidity" in df.columns:
            df["RH_change_1h"] = df["humidity"].diff(1)

        return df

    def predict_single_step(self, features: np.ndarray, model_name: str = "xgboost") -> float:
        """Predict AQI for single time step"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        # Predict
        if model_name == "lstm" and self.lstm_lookback:
            # LSTM needs sequence [lookback, n_features]
            # Ensure features is 2D: (lookback, n_features)
            if features.ndim == 1:
                # If simplified 1D passed (not recommended for LSTM), reshape/repeat or fail
                # Assuming context generates correct 2D array
                raise ValueError("LSTM requires sequence input (2D array)")

            # Scale the sequence
            # features shape: (lookback, n_features)
            features_scaled = self.scaler.transform(features)
            
            # Remove AQI column from sequence if needed (Scaler used 67, Model needs 66)
            if self.aqi_index is not None:
                features_scaled = np.delete(features_scaled, self.aqi_index, axis=1)
            
            # Reshape for model input: (1, lookback, n_features_reduced)
            input_seq = features_scaled.reshape(1, features.shape[0], features.shape[1] - 1 if self.aqi_index is not None else features.shape[1])
            
            prediction = self.models[model_name].predict(input_seq, verbose=0)[0][0]
        else:
            # Traditional ML models (Random Forest, XGBoost) take 1D features
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Remove AQI column if needed
            if self.aqi_index is not None:
                features_scaled = np.delete(features_scaled, self.aqi_index, axis=1)
            
            prediction = self.models[model_name].predict(features_scaled)[0]

        return max(0, prediction)  # AQI can't be negative

    def recursive_forecast(
        self, initial_data: pd.DataFrame, hours_ahead: int = 72, model_name: str = "xgboost"
    ) -> pd.DataFrame:
        """
        Recursive forecasting for multiple hours ahead
        Uses predicted AQI as input for next prediction
        """
        print(f"\nStarting recursive forecast for {hours_ahead} hours ({hours_ahead//24} days)...")
        print(f"Model: {model_name}")

        # Prepare initial features
        df_features = self.prepare_features_from_data(initial_data)
        df_features = df_features.dropna()

        if len(df_features) == 0:
            raise ValueError("No valid data after feature generation")

        # Initialize results
        forecast_times = []
        forecast_values = []
        forecast_categories = []

        # Get last known state
        last_timestamp = df_features.index[-1]
        current_df = df_features.copy()

        print(f"Starting from: {last_timestamp}")
        print(f"Forecasting until: {last_timestamp + timedelta(hours=hours_ahead)}")

        # Recursive prediction loop
        for h in range(1, hours_ahead + 1):
            target_time = last_timestamp + timedelta(hours=h)

            # Retrieve feature vector(s)
            
            if model_name == "lstm" and self.lstm_lookback:
                # LSTM requires a sequence of the last 'lookback' rows
                if len(current_df) < self.lstm_lookback:
                    raise ValueError(f"Insufficient history for LSTM. Need {self.lstm_lookback} rows.")
                
                # Get last N rows
                seq_df = current_df.iloc[-self.lstm_lookback:]
                
                # Build 2D feature array
                seq_vectors = []
                for _, row in seq_df.iterrows():
                    vec = []
                    for col in self.feature_cols:
                        if col in row:
                            vec.append(row[col])
                        else:
                            vec.append(0)
                    seq_vectors.append(vec)
                
                feature_vector = np.array(seq_vectors) # (24, n_features)
                
            else:
                # RF/XGB use only the latest row
                last_row = current_df.iloc[-1]
                feature_vector = []
                for col in self.feature_cols:
                    if col in last_row:
                        feature_vector.append(last_row[col])
                    else:
                        feature_vector.append(0)  # Default value for missing features
                feature_vector = np.array(feature_vector)

            # Predict
            predicted_aqi = self.predict_single_step(feature_vector, model_name)

            # Store prediction
            forecast_times.append(target_time)
            forecast_values.append(predicted_aqi)
            forecast_categories.append(self._categorize_aqi(predicted_aqi))

            # Update dataframe with prediction for next iteration
            # Create new row with predicted AQI and updated temporal features
            # Base it on the PREVIOUS timestamp (last_row of current_df) + 1 hour
            # Note: target_time IS last_timestamp + h, which is correct
            
            previous_row = current_df.iloc[-1]
            new_row = previous_row.copy()
            new_row.name = target_time

            # Update temporal features for next timestep
            new_row["Hour"] = target_time.hour
            new_row["DayOfWeek"] = target_time.dayofweek
            new_row["Month"] = target_time.month
            new_row["DayOfMonth"] = target_time.day
            new_row["DayOfYear"] = target_time.dayofyear
            new_row["IsWeekend"] = int(target_time.dayofweek >= 5)

            # Update cyclical features
            new_row["Hour_sin"] = np.sin(2 * np.pi * new_row["Hour"] / 24)
            new_row["Hour_cos"] = np.cos(2 * np.pi * new_row["Hour"] / 24)
            new_row["Month_sin"] = np.sin(2 * np.pi * new_row["Month"] / 12)
            new_row["Month_cos"] = np.cos(2 * np.pi * new_row["Month"] / 12)
            new_row["DayOfWeek_sin"] = np.sin(2 * np.pi * new_row["DayOfWeek"] / 7)
            new_row["DayOfWeek_cos"] = np.cos(2 * np.pi * new_row["DayOfWeek"] / 7)

            # Update AQI value
            if "AQI" in new_row:
                new_row["AQI"] = predicted_aqi

            # Add new row to dataframe
            current_df = pd.concat([current_df, new_row.to_frame().T])

            # Regenerate lag and rolling features for the entire updated dataframe
            current_df = self.prepare_features_from_data(current_df)
            current_df = current_df.dropna(how="all")

            # Progress indicator
            if h % 24 == 0:
                print(
                    f"  Day {h//24}: AQI {predicted_aqi:.1f} ({self._categorize_aqi(predicted_aqi)})"
                )

        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            {
                "timestamp": forecast_times,
                "predicted_AQI": forecast_values,
                "predicted_category": forecast_categories,
                "model": model_name,
            }
        )

        print(f"\nForecast completed: {len(forecast_df)} predictions")
        return forecast_df

    def ensemble_forecast(self, initial_data: pd.DataFrame, hours_ahead: int = 72) -> pd.DataFrame:
        """
        Create ensemble forecast using multiple models
        Averages predictions from available models
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE FORECASTING")
        print("=" * 60)

        forecasts = {}

        # Get forecasts from each model
        for model_name in ["random_forest", "xgboost"]:
            if model_name in self.models:
                try:
                    forecast = self.recursive_forecast(initial_data, hours_ahead, model_name)
                    forecasts[model_name] = forecast["predicted_AQI"].values
                    print(f"{model_name} forecast completed")
                except Exception as e:
                    print(f"WARNING: {model_name} forecast failed: {e}")

        if not forecasts:
            raise ValueError("No successful forecasts from any model")

        # Average predictions
        forecast_array = np.array(list(forecasts.values()))
        ensemble_predictions = np.mean(forecast_array, axis=0)

        # Create ensemble dataframe
        last_timestamp = (
            initial_data.index[-1]
            if isinstance(initial_data.index, pd.DatetimeIndex)
            else pd.to_datetime(initial_data["timestamp"].iloc[-1])
        )
        forecast_times = [last_timestamp + timedelta(hours=h) for h in range(1, hours_ahead + 1)]

        ensemble_df = pd.DataFrame(
            {
                "timestamp": forecast_times,
                "predicted_AQI": ensemble_predictions,
                "predicted_category": [self._categorize_aqi(aqi) for aqi in ensemble_predictions],
                "model": "ensemble",
            }
        )

        # Add individual model predictions for comparison
        for model_name, predictions in forecasts.items():
            ensemble_df[f"{model_name}_AQI"] = predictions

        print(f"\nEnsemble forecast completed")
        return ensemble_df

    def _categorize_aqi(self, aqi: float) -> str:
        """Categorize AQI value"""
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


def test_predictor():
    """Test prediction pipeline"""
    print("=" * 60)
    print("Testing Prediction Pipeline")
    print("=" * 60)

    # Load some sample data for testing
    try:
        df = pd.read_csv("processed_air_quality.csv", index_col=0, parse_dates=True)
        print(f"Loaded data: {len(df)} records")

        # Use last 100 hours as initial data
        initial_data = df.tail(100)

        # Initialize predictor
        predictor = AQIPredictor()

        # Test single model forecast
        forecast = predictor.recursive_forecast(initial_data, hours_ahead=72, model_name="xgboost")

        print("\nForecast Sample:")
        print(forecast.head(10))

        print("\nDaily Averages:")
        forecast["day"] = pd.to_datetime(forecast["timestamp"]).dt.date
        daily_avg = forecast.groupby("day")["predicted_AQI"].mean()
        for day, aqi in daily_avg.items():
            print(f"  {day}: {aqi:.1f}")

        # Test ensemble forecast
        ensemble_forecast = predictor.ensemble_forecast(initial_data, hours_ahead=72)

        print("\nEnsemble Forecast Sample:")
        print(ensemble_forecast.head(10))

        print("\n" + "=" * 60)
        print("Prediction pipeline test completed!")

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_predictor()
