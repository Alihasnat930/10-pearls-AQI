"""
Pydantic Models for API Request/Response Validation
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class AQICategory(str, Enum):
    """AQI category enumeration"""

    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"


class LocationRequest(BaseModel):
    """Location request model"""

    city: str = Field(..., example="Karachi")
    country: Optional[str] = Field(None, example="Pakistan")
    latitude: Optional[float] = Field(None, ge=-90, le=90, example=24.8607)
    longitude: Optional[float] = Field(None, ge=-180, le=180, example=67.0011)


class AirQualityData(BaseModel):
    """Air quality data model"""

    timestamp: datetime
    location: str
    AQI: float = Field(..., ge=0, le=500)
    AQI_category: str
    PM25: Optional[float] = Field(None, ge=0)
    PM10: Optional[float] = Field(None, ge=0)
    CO: Optional[float] = Field(None, ge=0)
    NO2: Optional[float] = Field(None, ge=0)
    O3: Optional[float] = Field(None, ge=0)
    temperature: Optional[float] = None
    humidity: Optional[float] = Field(None, ge=0, le=100)
    pressure: Optional[float] = Field(None, ge=0)
    wind_speed: Optional[float] = Field(None, ge=0)
    wind_direction: Optional[float] = Field(None, ge=0, le=360)

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-01-16T17:30:00",
                "location": "Karachi",
                "AQI": 150,
                "AQI_category": "Unhealthy for Sensitive Groups",
                "PM25": 55.5,
                "PM10": 85.2,
                "CO": 0.8,
                "NO2": 35.0,
                "O3": 40.0,
                "temperature": 25.1,
                "humidity": 47,
                "pressure": 1013,
                "wind_speed": 3.5,
                "wind_direction": 180,
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request model"""

    location: str = Field(..., example="Karachi")
    hours_ahead: int = Field(72, ge=1, le=168, description="Hours to predict (1-168)")
    model: Optional[str] = Field("ensemble", example="ensemble")

    @validator("model")
    def validate_model(cls, v):
        allowed = ["ensemble", "xgboost", "random_forest", "lstm"]
        if v not in allowed:
            raise ValueError(f"Model must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    """Prediction response model"""

    prediction_timestamp: datetime
    target_timestamp: datetime
    location: str
    predicted_aqi: float
    predicted_category: str
    model_name: str
    confidence_score: Optional[float] = None


class SHAPExplanation(BaseModel):
    """SHAP explanation model"""

    feature_name: str
    shap_value: float
    feature_value: float
    impact: str  # "positive" or "negative"


class ExplainabilityResponse(BaseModel):
    """Model explainability response"""

    prediction: float
    base_value: float
    shap_values: List[SHAPExplanation]
    top_features: List[str]
    explanation_plot: Optional[str] = None  # Base64 encoded plot


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = "healthy"
    version: str
    timestamp: datetime
    database_connected: bool
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    detail: Optional[str] = None
    timestamp: datetime
