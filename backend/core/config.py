"""
Backend Configuration Settings
Environment-based configuration for FastAPI backend
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Pearl AQI Intelligence API"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "REST API for Air Quality Forecasting with ML & SHAP Explanations"

    # CORS
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:8502",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "pearl_aqi_db")

    # External APIs
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    WAQI_API_KEY: str = os.getenv("WAQI_API_KEY", "")

    # Model Paths
    MODEL_DIR: str = "ml_models"
    XGBOOST_MODEL: str = "models/xgboost_model.json"
    RANDOM_FOREST_MODEL: str = "models/random_forest_model.pkl"
    LSTM_MODEL: str = "ml_models/lstm_model.h5"

    # Feature Store
    FEATURE_COLUMNS_FILE: str = "feature_columns.txt"

    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
