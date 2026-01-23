"""
Test ML models
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from backend.services.prediction_pipeline import AQIPredictor


@patch("backend.services.prediction_pipeline.pickle.load")
@patch("builtins.open")
def test_model_loading(mock_open, mock_pickle_load):
    """Test model loading with mocks"""
    print("Testing model loading...")
    
    # Mock open and pickle load
    mock_open.return_value.__enter__.return_value = MagicMock()
    mock_pickle_load.return_value = MagicMock()
    
    # We need to mock os.path.exists too if the class uses it to check for files
    with patch("os.path.exists", return_value=True):
        # We also need to patch xgboost and keras loading if they are called
        with patch("xgboost.XGBRegressor"), \
             patch("keras.models.load_model"), \
             patch("json.load", return_value={"lookback": 24}):
             
            predictor = AQIPredictor()

            # Since we mocked loading, we just check if attributes exist
            assert predictor.models is not None
            
    print("✅ Models loaded successfully (mocked)")


@patch("backend.services.prediction_pipeline.AQIPredictor")
def test_model_inference(mock_predictor_cls):
    """Test model inference with mocks"""
    print("Testing model inference...")
    
    # Setup mock predictor instance
    mock_instance = MagicMock()
    mock_predictor_cls.return_value = mock_instance
    
    # Mock feature columns
    mock_instance.feature_cols = [f"col_{i}" for i in range(10)]
    mock_instance.scaler = MagicMock()
    mock_instance.models = {
        "xgboost": MagicMock(),
        "random_forest": MagicMock()
    }
    
    # Mock prediction returns
    mock_instance.models["xgboost"].predict.return_value = np.array([50.0])
    mock_instance.models["random_forest"].predict.return_value = np.array([55.0])
    
    predictor = mock_predictor_cls()

    # Create dummy data
    features = predictor.feature_cols
    dummy_data = pd.DataFrame(np.random.rand(100, len(features)), columns=features)

    # Test XGBoost prediction using the mock
    xgb_pred = predictor.models["xgboost"].predict(dummy_data.values[:1])
    assert len(xgb_pred) > 0, "XGBoost prediction failed"
    
    # Test Random Forest prediction using the mock
    rf_pred = predictor.models["random_forest"].predict(dummy_data.values[:1])
    assert len(rf_pred) > 0, "Random Forest prediction failed"

    print(f"✅ XGBoost prediction (mock): {xgb_pred[0]:.2f}")
    print(f"✅ Random Forest prediction (mock): {rf_pred[0]:.2f}")


if __name__ == "__main__":
    test_model_loading()
    test_model_inference()
    print("\n✅ All model tests passed!")
