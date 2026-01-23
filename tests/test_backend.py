"""
Unit tests for backend API services
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime

from backend.core.database import get_database
from backend.services.prediction_service import PredictionService


class TestPredictionService:
    """Test prediction service"""

    @patch("backend.services.prediction_pipeline.AQIPredictor")
    def test_prediction_service_init(self, mock_predictor):
        """Test service initialization"""
        mock_predictor.return_value = MagicMock()
        service = PredictionService()
        assert service is not None
        assert service.predictor is not None

    @patch("backend.services.prediction_pipeline.AQIPredictor")
    def test_get_aqi_category(self, mock_predictor):
        """Test AQI category classification"""
        mock_predictor.return_value = MagicMock()
        service = PredictionService()

        assert service._get_aqi_category(25) == "Good"
        assert service._get_aqi_category(75) == "Moderate"
        assert service._get_aqi_category(125) == "Unhealthy for Sensitive Groups"
        assert service._get_aqi_category(175) == "Unhealthy"
        assert service._get_aqi_category(250) == "Very Unhealthy"
        assert service._get_aqi_category(350) == "Hazardous"


class TestDatabase:
    """Test database operations"""

    def test_database_connection(self):
        """Test MongoDB connection or graceful offline mode"""
        db = get_database()
        assert db is not None
        # In CI without secrets, client might be None (offline mode)
        # We verify that it initializes without crashing
        if os.getenv("MONGODB_URI"):
            assert db.client is not None
        else:
            print("Skipping client check in offline mode")

    def test_insert_live_data(self):
        """Test inserting live data"""
        db = get_database()

        test_data = {
            "timestamp": datetime.now(),
            "location": "Test City",
            "AQI": 75.0,
            "PM2.5": 25.5,
            "PM10": 45.2,
            "temperature": 25.0,
            "humidity": 60.0,
            "AQI_category": "Moderate",
        }

        # Should not raise exception
        db.insert_live_data(test_data)


class TestModels:
    """Test ML model loading and inference"""

    def test_xgboost_model_exists(self):
        """Test XGBoost model file exists or skip if missing (CI)"""
        if not os.path.exists("models/xgboost_model.json"):
            pytest.skip("Model file not found - skipping test")
        assert os.path.exists("models/xgboost_model.json")

    def test_model_metrics_exists(self):
        """Test model metrics file exists or skip if missing (CI)"""
        if not os.path.exists("models/model_metrics.json"):
            pytest.skip("Metrics file not found - skipping test")
        assert os.path.exists("models/model_metrics.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
