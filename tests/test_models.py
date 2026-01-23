"""
Test ML models
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from backend.services.prediction_pipeline import AQIPredictor


def test_model_loading():
    """Test model loading"""
    print("Testing model loading...")
    predictor = AQIPredictor()

    assert predictor.xgboost_model is not None, "XGBoost model not loaded"
    assert predictor.random_forest_model is not None, "Random Forest model not loaded"

    print("✅ Models loaded successfully")


def test_model_inference():
    """Test model inference"""
    print("Testing model inference...")
    predictor = AQIPredictor()

    # Create dummy data
    features = predictor.feature_columns
    dummy_data = pd.DataFrame(np.random.rand(100, len(features)), columns=features)

    # Test XGBoost prediction
    xgb_pred = predictor.xgboost_model.predict(dummy_data.values[:1])
    assert len(xgb_pred) > 0, "XGBoost prediction failed"
    assert xgb_pred[0] >= 0, "Invalid XGBoost prediction"

    # Test Random Forest prediction
    rf_pred = predictor.random_forest_model.predict(dummy_data.values[:1])
    assert len(rf_pred) > 0, "Random Forest prediction failed"
    assert rf_pred[0] >= 0, "Invalid Random Forest prediction"

    print(f"✅ XGBoost prediction: {xgb_pred[0]:.2f}")
    print(f"✅ Random Forest prediction: {rf_pred[0]:.2f}")


if __name__ == "__main__":
    test_model_loading()
    test_model_inference()
    print("\n✅ All model tests passed!")
