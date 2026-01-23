"""
FastAPI Main Application
Pearl AQI Intelligence Platform Backend
"""

import os
import sys
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.api import routes
from backend.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pearl AQI Intelligence Platform API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_v1": settings.API_V1_STR,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from backend.core.database import get_database

    try:
        db = get_database()
        db_connected = db.client is not None
    except:
        db_connected = False

    # Check if models exist
    models_status = {
        "xgboost": os.path.exists(settings.XGBOOST_MODEL),
        "random_forest": os.path.exists(settings.RANDOM_FOREST_MODEL),
        "lstm": os.path.exists(settings.LSTM_MODEL),
    }

    return {
        "status": "healthy" if db_connected else "degraded",
        "version": settings.VERSION,
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_connected,
        "models_loaded": models_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
