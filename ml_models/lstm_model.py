"""
LSTM Model for Air Quality Forecasting
Time Series Deep Learning Model using TensorFlow/Keras
"""

import os
import pickle
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.core.database_main import AirQualityDatabase


class LSTMAQIModel:
    """LSTM model for AQI time series forecasting"""

    def __init__(self, sequence_length: int = 24, features: int = 10):
        """
        Initialize LSTM model

        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
        """
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = []
        self.model_path = "models/lstm_model.h5"
        self.scaler_path = "models/lstm_scalers.pkl"

    def build_model(self):
        """Build improved LSTM neural network architecture"""
        model = keras.Sequential(
            [
                # First LSTM layer - reduced complexity
                layers.LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.features),
                    recurrent_dropout=0.2,
                ),
                layers.BatchNormalization(),
                # Second LSTM layer
                layers.LSTM(32, return_sequences=False, recurrent_dropout=0.2),
                layers.BatchNormalization(),
                # Dense layers - simplified
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(16, activation="relu"),
                layers.Dropout(0.2),
                # Output layer
                layers.Dense(1, activation="linear"),
            ]
        )

        # Compile model with better optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss="mse",  # Mean squared error for regression
            metrics=[
                "mae",
                keras.metrics.MeanSquaredError(name="mse"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

        self.model = model
        return model

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for LSTM

        Args:
            data: DataFrame with time series data

        Returns:
            X (sequences), y (targets)
        """
        # Select relevant features
        feature_cols = [
            "AQI",
            "PM25",
            "PM10",
            "CO",
            "NO2",
            "O3",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
        ]

        # Filter available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        self.feature_columns = available_cols

        # Extract features and target
        X_data = data[available_cols].fillna(method="ffill").fillna(method="bfill").fillna(0).values
        y_data = data["AQI"].fillna(method="ffill").fillna(method="bfill").values.reshape(-1, 1)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data)

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(len(X_scaled) - self.sequence_length):
            X_sequences.append(X_scaled[i : i + self.sequence_length])
            y_sequences.append(y_scaled[i + self.sequence_length])

        return np.array(X_sequences), np.array(y_sequences)

    def train(
        self,
        location: str = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        """
        Train LSTM model

        Args:
            location: Optional location filter
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data split

        Returns:
            Training history
        """
        try:
            # Get training data from database
            db = AirQualityDatabase()

            # Get historical data (last 180 days)
            df = db.get_recent_data(hours=180 * 24, table="historical_data", location=location)

            if df.empty or len(df) < 100:
                print(f"⚠️ Insufficient data for training. Need at least 100 records.")
                return None

            print(f"📊 Training data: {len(df)} records")

            # Prepare sequences
            X, y = self.prepare_sequences(df)
            self.features = X.shape[2]

            print(f"📐 Sequences shape: X={X.shape}, y={y.shape}")

            # Build model
            if self.model is None:
                self.build_model()

            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            )

            checkpoint = callbacks.ModelCheckpoint(
                self.model_path, monitor="val_loss", save_best_only=True, verbose=1
            )

            # Train model
            print("🚀 Starting LSTM training...")
            history = self.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1,
            )

            # Save scalers
            with open(self.scaler_path, "wb") as f:
                pickle.dump(
                    {
                        "scaler_X": self.scaler_X,
                        "scaler_y": self.scaler_y,
                        "feature_columns": self.feature_columns,
                    },
                    f,
                )

            print(f"✅ LSTM model trained and saved to {self.model_path}")

            return history

        except Exception as e:
            print(f"❌ Training error: {e}")
            return None

    def train_with_data(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        """
        Train LSTM model with provided DataFrame

        Args:
            df: DataFrame with training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data split

        Returns:
            Training history
        """
        try:
            if df.empty or len(df) < 100:
                print(f"⚠️ Insufficient data for training. Need at least 100 records.")
                return None

            print(f"📊 Training data: {len(df)} records")

            # Prepare sequences
            X, y = self.prepare_sequences(df)
            self.features = X.shape[2]

            print(f"📐 Sequences shape: X={X.shape}, y={y.shape}")

            # Build model
            if self.model is None:
                self.build_model()

            # Callbacks - adjusted for better training
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, min_delta=0.001
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
            )

            # Ensure models directory exists
            import os

            os.makedirs("models", exist_ok=True)

            checkpoint = callbacks.ModelCheckpoint(
                "models/lstm_model.h5", monitor="val_loss", save_best_only=True, verbose=1
            )

            # Train model
            print("🚀 Starting LSTM training...")
            history = self.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1,
            )

            # Save final model in Keras 3 format
            self.model.save("models/lstm_model.keras")
            print("✅ Model saved in Keras 3 format")

            # Save config
            import json

            config = {
                "lookback": self.sequence_length,
                "features": self.features,
                "feature_columns": self.feature_columns,
            }
            with open("models/lstm_config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Save scalers
            with open("models/lstm_scalers.pkl", "wb") as f:
                pickle.dump(
                    {
                        "scaler_X": self.scaler_X,
                        "scaler_y": self.scaler_y,
                        "feature_columns": self.feature_columns,
                    },
                    f,
                )

            print(f"✅ LSTM model trained and saved to models/lstm_model.h5")

            return history

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def predict(self, data: pd.DataFrame, steps_ahead: int = 24) -> np.ndarray:
        """
        Generate predictions

        Args:
            data: Recent data for sequence
            steps_ahead: Number of steps to predict

        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                self.load_model()

            # Prepare last sequence
            X_data = (
                data[self.feature_columns]
                .fillna(method="ffill")
                .fillna(0)
                .values[-self.sequence_length :]
            )
            X_scaled = self.scaler_X.transform(X_data)

            predictions = []
            current_sequence = X_scaled.copy()

            for _ in range(steps_ahead):
                # Reshape for prediction
                X_input = current_sequence.reshape(1, self.sequence_length, self.features)

                # Predict
                y_pred_scaled = self.model.predict(X_input, verbose=0)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0][0]

                predictions.append(y_pred)

                # Update sequence (roll forward)
                # Append predicted AQI and repeat last feature values
                new_features = current_sequence[-1].copy()
                new_features[0] = y_pred_scaled[0][0]  # Update AQI

                current_sequence = np.vstack([current_sequence[1:], new_features])

            return np.array(predictions)

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return np.array([])

    def load_model(self):
        """Load saved model and scalers"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ LSTM model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                return False

            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    scaler_data = pickle.load(f)
                    self.scaler_X = scaler_data["scaler_X"]
                    self.scaler_y = scaler_data["scaler_y"]
                    self.feature_columns = scaler_data["feature_columns"]
                print("✅ Scalers loaded")
            else:
                print(f"⚠️ Scaler file not found: {self.scaler_path}")
                return False

            return True

        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False

    def evaluate(self, data: pd.DataFrame) -> dict:
        """
        Evaluate model performance

        Args:
            data: Test data

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            X, y = self.prepare_sequences(data)

            if self.model is None:
                self.load_model()

            # Evaluate
            results = self.model.evaluate(X, y, verbose=0)

            metrics = {
                "loss": float(results[0]),
                "mae": float(results[1]),
                "mse": float(results[2]),
                "rmse": float(np.sqrt(results[2])),
            }

            print(f"📊 Evaluation metrics: {metrics}")

            return metrics

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {}


if __name__ == "__main__":
    # Train LSTM model
    print("🚀 Training LSTM Model for AQI Forecasting")

    lstm = LSTMAQIModel(sequence_length=24, features=10)
    history = lstm.train(epochs=50, batch_size=32)

    if history:
        print("✅ LSTM training completed successfully!")
