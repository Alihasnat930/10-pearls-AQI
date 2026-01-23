"""
Air Quality Forecasting - Model Training Script
Trains multiple ML models (Random Forest, XGBoost, LSTM) for AQI prediction
"""

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class AQIModelTrainer:
    """Train and evaluate multiple models for AQI prediction"""

    def __init__(self, data_path="processed_air_quality.csv"):
        """Initialize trainer with processed data"""
        print("Loading processed data...")
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.models = {}
        self.scalers = {}
        self.metrics = {}

    def prepare_data(self, target_col="AQI", test_size=0.2):
        """Prepare features and target for training"""
        print("\nPreparing data for training...")

        # Select feature columns (exclude target and categorical)
        # Note: We include AQI in features so the LSTM can use past AQI values for autoregression
        exclude_cols = ["AQI_Category", "DayName"]
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        print(f"Number of features: {len(self.feature_cols)}")

        # Features and target
        X = self.df[self.feature_cols].values
        y = self.df[target_col].values

        # Time-based split (preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        print("\n" + "=" * 60)
        print("Training Random Forest Regressor...")
        print("=" * 60)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        metrics = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test, "Random Forest")

        self.models["random_forest"] = model
        self.metrics["random_forest"] = metrics

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": self.feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nTop 15 Important Features:")
        print(feature_importance.head(15))

        return model, metrics

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost Regressor...")
        print("=" * 60)

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=1,
        )

        model.fit(
            X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=50
        )

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        metrics = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test, "XGBoost")

        self.models["xgboost"] = model
        self.metrics["xgboost"] = metrics

        return model, metrics

    def train_lstm(self, X_train, X_test, y_train, y_test, lookback=24):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("\n⚠️  Skipping LSTM training - TensorFlow not available")
            return None, {}

        print("\n" + "=" * 60)
        print("Training LSTM Network...")
        print("=" * 60)

        # Reshape for LSTM [samples, timesteps, features]
        def create_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(len(X) - lookback):
                Xs.append(X[i : (i + lookback)])
                ys.append(y[i + lookback])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

        print(f"LSTM Training shape: {X_train_seq.shape}")
        print(f"LSTM Test shape: {X_test_seq.shape}")

        # Build LSTM model - Enhanced for >90% Accuracy
        from tensorflow.keras.layers import Bidirectional, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        model = Sequential(
            [
                # First Bidirectional LSTM layer
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback, X_train.shape[1])),
                BatchNormalization(),
                Dropout(0.3),
                
                # Second Bidirectional LSTM layer
                Bidirectional(LSTM(128, return_sequences=True)),
                BatchNormalization(),
                Dropout(0.3),

                # Third LSTM layer
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),

                # Dense layers
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1), # Linear activation for regression
            ]
        )

        # Use lower learning rate for stability
        optimizer = Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        print("\nModel Architecture:")
        model.summary()

        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            "models/best_lstm_model.h5", save_best_only=True, monitor="val_loss"
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1
        )

        # Train
        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=200,                # Increased epochs
            batch_size=32,             # Smaller batch size for better generalization
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1,
        )

        # Predictions
        y_pred_train = model.predict(X_train_seq).flatten()
        y_pred_test = model.predict(X_test_seq).flatten()

        # Evaluate
        metrics = self._evaluate_model(y_train_seq, y_pred_train, y_test_seq, y_pred_test, "LSTM")

        self.models["lstm"] = model
        self.metrics["lstm"] = metrics
        self.lstm_lookback = lookback

        # Plot training history
        self._plot_training_history(history)

        return model, metrics

    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test, model_name):
        """Calculate and display evaluation metrics"""
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\n{model_name} Performance:")
        print("-" * 50)
        print(f"Training   - RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
        print(f"Test       - RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}")

        metrics = {
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        }

        return metrics

    def _plot_training_history(self, history):
        """Plot LSTM training history"""
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mae"], label="Training MAE")
        plt.plot(history.history["val_mae"], label="Validation MAE")
        plt.title("Model MAE", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("models/lstm_training_history.png", dpi=300, bbox_inches="tight")
        print("\nTraining history plot saved to 'models/lstm_training_history.png'")
        plt.close()

    def save_models(self):
        """Save trained models and scalers"""
        import os

        os.makedirs("models", exist_ok=True)

        print("\n" + "=" * 60)
        print("Saving models...")
        print("=" * 60)

        # Save Random Forest
        if "random_forest" in self.models:
            with open("models/random_forest_model.pkl", "wb") as f:
                pickle.dump(self.models["random_forest"], f)
                print("[OK] Random Forest saved")

        # Save XGBoost
        if "xgboost" in self.models:
            # Use pickle for consistent serialization across sklearn wrappers
            with open("models/xgboost_model.pkl", "wb") as f:
                pickle.dump(self.models["xgboost"], f)
            print("[OK] XGBoost saved")

        # Save LSTM (already saved during training with checkpoint)
        if "lstm" in self.models:
            self.models["lstm"].save("models/lstm_model.h5")
            print("[OK] LSTM saved")

        # Save scaler
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
            print("[OK] Scaler saved")

        # Save feature columns
        with open("models/feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
            print("[OK] Feature columns saved")

        # Save metrics
        with open("models/model_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
            print("[OK] Metrics saved")

        # Save LSTM lookback
        if hasattr(self, "lstm_lookback"):
            with open("models/lstm_config.json", "w") as f:
                json.dump({"lookback": self.lstm_lookback}, f)
                print("[OK] LSTM config saved")

        print("\nAll models saved successfully in 'models/' directory")

    def compare_models(self):
        """Compare performance of all models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df[["test_rmse", "test_mae", "test_r2"]]
        comparison_df.columns = ["RMSE", "MAE", "R²"]

        print(comparison_df.to_string())

        # Find best model
        best_model = comparison_df["RMSE"].idxmin()
        print(f"\n🏆 Best Model (lowest RMSE): {best_model.upper()}")

        return comparison_df


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AIR QUALITY FORECASTING - MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize trainer
    trainer = AQIModelTrainer(data_path="data/processed_air_quality.csv")

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()

    # Train models
    # try:
    #     trainer.train_random_forest(X_train, X_test, y_train, y_test)
    # except Exception as e:
    #     print(f"Error training Random Forest: {e}")

    # try:
    #     trainer.train_xgboost(X_train, X_test, y_train, y_test)
    # except Exception as e:
    #     print(f"Error training XGBoost: {e}")

    if TENSORFLOW_AVAILABLE:
        try:
            trainer.train_lstm(X_train, X_test, y_train, y_test, lookback=24)
        except Exception as e:
            print(f"Error training LSTM: {e}")
    else:
        print("\n⚠️  Skipping LSTM - TensorFlow not available")

    # Compare models
    trainer.compare_models()

    # Save everything
    trainer.save_models()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
