"""
API Routes for Pearl AQI Platform
Endpoints for predictions, data fetching, and SHAP explanations
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.api.models import (
    AirQualityData,
    ExplainabilityResponse,
    HealthResponse,
    LocationRequest,
    PredictionRequest,
    PredictionResponse,
)
from backend.core.database import get_database
from backend.services.api_fetcher import AirQualityAPIFetcher
from backend.services.prediction_service import PredictionService
from backend.services.shap_service import SHAPService

router = APIRouter()

# Initialize services
prediction_service = PredictionService()
shap_service = SHAPService()
api_fetcher = AirQualityAPIFetcher()


@router.get("/data/current", response_model=List[AirQualityData])
async def get_current_data(
    location: Optional[str] = Query(None, description="Filter by location"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
):
    """Get current air quality data"""
    try:
        db = get_database()
        df = db.get_recent_data(hours=hours, table="live_data", location=location)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for location: {location}")

        # Convert DataFrame to list of dicts
        data = df.to_dict("records")
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/fetch")
async def fetch_live_data(location: LocationRequest):
    """Fetch live air quality data from external APIs"""
    try:
        # Set location
        api_fetcher.set_location(
            city=location.city,
            latitude=location.latitude,
            longitude=location.longitude,
            country=location.country,
        )

        # Fetch data
        data = api_fetcher.fetch_combined_data()

        if not data:
            # Try mock data as fallback
            data = api_fetcher.generate_mock_data()

        if not data:
            raise HTTPException(status_code=404, detail="Could not fetch data from any source")

        # Store in database
        db = get_database()
        db.insert_live_data(data)

        return {
            "status": "success",
            "message": f"Data fetched for {location.city}",
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=List[PredictionResponse])
async def predict_aqi(request: PredictionRequest):
    """Predict future AQI values"""
    try:
        predictions = prediction_service.predict(
            location=request.location, hours_ahead=request.hours_ahead, model_name=request.model
        )

        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate predictions for {request.location}. Ensure sufficient historical data exists.",
            )

        return predictions

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{location}", response_model=List[PredictionResponse])
async def get_predictions(
    location: str, days: int = Query(3, ge=1, le=7, description="Days of predictions to retrieve")
):
    """Get stored predictions for a location"""
    try:
        db = get_database()
        df = db.get_predictions(days=days)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for {location}")

        # Filter by location
        df_filtered = df[df["location"] == location] if "location" in df.columns else df

        if df_filtered.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for {location}")

        data = df_filtered.to_dict("records")
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """Get SHAP explanations for AQI prediction"""
    try:
        explanation = shap_service.explain(
            location=request.location,
            hours_ahead=24,  # Explain next 24h prediction
            model_name=request.model if request.model != "ensemble" else "xgboost",
        )

        if not explanation:
            raise HTTPException(
                status_code=404,
                detail="Could not generate explanation. Ensure model is trained and data exists.",
            )

        return explanation

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/locations")
async def get_available_locations():
    """Get list of locations with available data"""
    try:
        db = get_database()

        # Get distinct locations from live_data collection
        locations = db.db.live_data.distinct("location")

        return {
            "locations": locations,
            "count": len(locations),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(location: Optional[str] = None):
    """Get database statistics"""
    try:
        db = get_database()
        stats = db.get_data_statistics()

        if location:
            # Filter stats by location
            live_count = db.db.live_data.count_documents({"location": location})
            hist_count = db.db.historical_data.count_documents({"location": location})
            pred_count = db.db.predictions.count_documents({"location": location})

            stats = {
                "location": location,
                "live_records": live_count,
                "historical_records": hist_count,
                "predictions": pred_count,
            }

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/clear")
async def clear_old_data(days: int = Query(30, ge=1, description="Keep data from last N days")):
    """Clear old data from database"""
    try:
        db = get_database()
        cutoff_date = datetime.now() - timedelta(days=days)

        # Delete old records
        live_result = db.db.live_data.delete_many({"timestamp": {"$lt": cutoff_date}})
        hist_result = db.db.historical_data.delete_many({"timestamp": {"$lt": cutoff_date}})

        return {
            "status": "success",
            "deleted": {
                "live_data": live_result.deleted_count,
                "historical_data": hist_result.deleted_count,
            },
            "cutoff_date": cutoff_date.isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
