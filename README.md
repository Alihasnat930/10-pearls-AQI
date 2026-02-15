# Pearl AQI Intelligence Platform

> **Advanced Air Quality Monitoring and Forecasting System with ML-Powered Predictions**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-brightgreen.svg)](https://www.mongodb.com/)

## 🎥 Project Demo

[![Watch the Demo](https://img.youtube.com/vi/QHX9YqBC00Q/maxresdefault.jpg)](https://youtu.be/QHX9YqBC00Q)

> Click the image above to watch the comprehensive demo of the platform.

## 🌐 Live App

[Streamlit Live Dashboard](https://alihasnat930-10-pearls-aqi-frontenddashboard-enhanced-xxsleq.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Database Setup](#database-setup)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Overview

Pearl AQI is a comprehensive air quality intelligence platform that combines real-time data collection, advanced machine learning predictions, and interactive visualizations to provide actionable insights into air pollution levels across multiple cities worldwide.

### Key Capabilities

- **Real-time Monitoring**: Live air quality data from 15+ global cities
- **ML Predictions**: XGBoost, Random Forest, and LSTM models with 99%+ accuracy
- **SHAP Explainability**: Understand what drives air quality predictions
- **Multi-country Support**: Asia, Europe, Americas with 15+ cities including Pakistan
- **Interactive Dashboard**: Real-time visualizations and predictions
- **REST API**: FastAPI backend for programmatic access

---

## ✨ Features

### 🔍 Data Collection & Processing
- Integration with OpenWeatherMap and WAQI APIs
- Automated hourly data collection for multiple cities
- MongoDB cloud storage for scalability
- Historical data analysis and trend detection

### 🤖 Machine Learning
- **XGBoost Model**: R² = 0.995, MAE = 2.1
- **Random Forest**: R² = 0.993, MAE = 2.3
- **LSTM Deep Learning**: Time-series forecasting with TensorFlow
- **SHAP Explainability**: Feature importance and prediction explanations

### 📊 Interactive Dashboard (Streamlit)
- Real-time air quality monitoring
- 24-hour, 7-day, and 30-day predictions
- Interactive pollutant charts (PM2.5, PM10, NO2, SO2, CO, O3)
- City-wise comparison and analysis
- SHAP waterfall plots for model interpretability
- AQI health recommendations

### 🚀 REST API (FastAPI)
- `/data/current` - Get current air quality data
- `/data/fetch` - Fetch fresh data for a city
- `/predict` - Get ML predictions
- `/explain` - Get SHAP explanations
- `/locations` - List available cities
- `/statistics` - Get historical statistics
- OpenAPI documentation at `/docs`

---

## ⚡ Quick Start

### **Option 1: Using run.bat (Easiest)**

```bash
run.bat
```

Choose from the menu:
1. Start Dashboard Only
2. Start API Backend Only
3. Start Full Stack (Dashboard + API)
4. Run Tests
5. Train Models
6. Fetch New Data
7. Docker Deployment

### **Option 2: Manual Commands**

#### Start Dashboard
```bash
streamlit run frontend/dashboard_enhanced.py --server.port 8502
```
Access at: http://localhost:8502

#### Start API Backend
```bash
cd backend
python main.py
```
Access at: http://localhost:8000  
API Docs: http://localhost:8000/docs

#### Full Stack
```bash
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Dashboard
streamlit run frontend/dashboard_enhanced.py --server.port 8502
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11 or higher
- MongoDB Atlas account (free tier works)
- OpenWeatherMap API key
- WAQI API key

### Step 1: Clone Repository

```bash
cd "your-desired-directory"
git clone <repository-url>
cd "10 pearl AQI"
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create `.env` File

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=pearl_aqi_db

# API Keys
OPENWEATHER_API_KEY=your_openweather_api_key_here
WAQI_API_KEY=your_waqi_api_key_here

# Application Settings
ENV=production
DEBUG=False
```

### Streamlit Secrets (Cloud Deployment)

For Streamlit Cloud, add these keys in **App Settings → Secrets**:

```toml
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DATABASE = "pearl_aqi_db"
OPENWEATHER_API_KEY = "your_openweather_api_key_here"
WAQI_API_KEY = "your_waqi_api_key_here"
```

### Step 5: Run Initial Setup

```bash
# Test database connection
python -c "from backend.core.database_main import AirQualityDatabase; db = AirQualityDatabase(); print('✓ Database connected')"

# Fetch initial data
python scripts/automated_data_fetch.py

# Start the application
run.bat
```

---

## ⚙️ Configuration

### API Configuration (`api_config.json`)

```json
{
  "openweather": {
    "base_url": "https://api.openweathermap.org/data/2.5",
    "api_key": "from_env"
  },
  "waqi": {
    "base_url": "https://api.waqi.info",
    "token": "from_env"
  }
}
```

### MongoDB Setup

1. **Create MongoDB Atlas Account**: https://www.mongodb.com/cloud/atlas
2. **Create Cluster**: Free M0 tier is sufficient
3. **Create Database User**: Database Access → Add New User
4. **Whitelist IP**: Network Access → Add IP Address (0.0.0.0/0 for development)
5. **Get Connection String**: Cluster → Connect → Connect Your Application
6. **Update `.env`**: Add your connection string

### User Preferences (`user_prefs.json`)

```json
{
  "default_location": "Karachi",
  "theme": "dark",
  "refresh_interval": 300
}
```

---

## 📖 Usage

### Dashboard Features

#### 1. Real-Time Monitoring
- Select city from sidebar
- View current AQI and all pollutants
- Color-coded health indicators
- Automatic data refresh

#### 2. Predictions
- 24-hour hourly forecasts
- 7-day daily forecasts
- 30-day monthly forecasts
- Confidence intervals

#### 3. Historical Analysis
- Time-series charts for all pollutants
- Trend analysis
- Seasonal patterns
- Correlation heatmaps

#### 4. SHAP Explanations
- Understand prediction drivers
- Feature importance rankings
- Waterfall plots for individual predictions

#### 5. City Comparisons
- Compare AQI across cities
- Identify cleanest/most polluted areas
- Regional trends

### API Usage

#### Get Current Data
```python
import requests

response = requests.get("http://localhost:8000/data/current?location=Karachi")
data = response.json()
print(data)
```

#### Get Predictions
```python
payload = {
    "location": "Karachi",
    "features": {
        "pm25": 45.5,
        "pm10": 80.2,
        "no2": 25.3,
        "so2": 8.1,
        "co": 0.5,
        "o3": 65.0,
        "temperature": 28.5,
        "humidity": 65,
        "pressure": 1013,
        "wind_speed": 5.2
    }
}

response = requests.post("http://localhost:8000/predict", json=payload)
prediction = response.json()
print(f"Predicted AQI: {prediction['prediction']}")
```

#### Get SHAP Explanation
```python
response = requests.post("http://localhost:8000/explain", json=payload)
explanation = response.json()
print(f"Top feature: {explanation['feature_importance'][0]}")
```

---

## 🏗️ Architecture

```
pearl_aqi/
├── backend/                  # FastAPI Backend
│   ├── api/                  # REST API endpoints
│   │   ├── routes.py         # API routes
│   │   └── models.py         # Pydantic schemas
│   ├── core/                 # Core functionality
│   │   ├── config.py         # Configuration
│   │   ├── database.py       # DB wrapper
│   │   └── database_main.py  # MongoDB operations
│   ├── services/             # Business logic
│   │   ├── api_fetcher.py    # External API calls
│   │   ├── prediction_pipeline.py  # ML predictions
│   │   ├── prediction_service.py   # Prediction logic
│   │   └── shap_service.py   # SHAP explanations
│   └── main.py               # FastAPI app
│
├── frontend/                 # Streamlit Dashboard
│   ├── dashboard_enhanced.py # Main dashboard
│   └── dashboard_legacy.py   # Backup dashboard
│
├── ml_models/                # Machine Learning
│   └── lstm_model.py         # LSTM implementation
│
├── scripts/                  # Utility Scripts
│   ├── automated_data_fetch.py  # Data collection
│   ├── train_models.py       # Model training
│   └── main_legacy.py        # Legacy scripts
│
├── tests/                    # Test Suite
│   ├── test_backend.py       # Backend tests
│   └── test_models.py        # Model tests
│
├── data/                     # Data Storage
│   ├── AirQuality.csv
│   └── processed_air_quality.csv
│
├── models/                   # Trained Models
│   ├── xgboost_model.json
│   └── model_metrics.json
│
├── .github/workflows/        # CI/CD
│   └── ci-cd.yml             # GitHub Actions
│
├── docker-compose.yml        # Docker orchestration
├── Dockerfile.backend        # Backend container
├── Dockerfile.frontend       # Frontend container
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── run.bat                   # Launcher script
└── README.md                 # This file
```

### Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Database**: MongoDB Atlas
- **ML**: XGBoost, TensorFlow, scikit-learn, SHAP
- **APIs**: OpenWeatherMap, WAQI
- **DevOps**: Docker, GitHub Actions

---

## 📡 API Documentation

### Base URL: `http://localhost:8000`

### Endpoints

#### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2026-01-16T12:00:00Z"
}
```

#### GET `/data/current`
Get current air quality data
- **Query Params**: `location` (optional)
- **Response**: Current AQI, pollutants, weather data

#### POST `/data/fetch`
Fetch fresh data from external APIs
- **Body**: `{"location": "Karachi", "lat": 24.8607, "lon": 67.0011}`
- **Response**: Fetched and stored data

#### POST `/predict`
Get ML predictions
- **Body**: `{"location": "Karachi", "features": {...}}`
- **Response**: 
```json
{
  "prediction": 85.5,
  "model": "xgboost",
  "confidence": 0.95,
  "category": "Moderate"
}
```

#### POST `/explain`
Get SHAP explanations
- **Body**: Same as `/predict`
- **Response**: Feature importance, SHAP values, waterfall plot

#### GET `/locations`
List all available cities
- **Response**: Array of city objects with coordinates

#### GET `/statistics`
Get historical statistics
- **Query Params**: `location`, `start_date`, `end_date`
- **Response**: Statistical summary

### Interactive API Docs

Visit http://localhost:8000/docs for full interactive API documentation with try-it-out functionality.

---

## 🗄️ Database Setup

### MongoDB Collections

1. **air_quality_data**: Raw air quality measurements
2. **predictions**: ML prediction results
3. **model_metadata**: Model information and metrics

### Indexes

```javascript
// Create indexes for faster queries
db.air_quality_data.createIndex({ location: 1, timestamp: -1 })
db.air_quality_data.createIndex({ timestamp: -1 })
db.predictions.createIndex({ location: 1, created_at: -1 })
```

### Sample Query

```python
from backend.core.database_main import AirQualityDatabase

db = AirQualityDatabase()

# Get recent data
data = db.get_recent_data(location="Karachi", limit=100)

# Get statistics
stats = db.get_statistics(location="Karachi")

# Store prediction
db.store_prediction({
    "location": "Karachi",
    "aqi": 85,
    "timestamp": datetime.now()
})
```

---

## 👨‍💻 Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=backend --cov=ml_models --cov-report=html

# Specific test
pytest tests/test_backend.py::test_model_loading -v
```

### Code Formatting

```bash
# Format code
python -m black backend/ frontend/ ml_models/ scripts/ tests/ --line-length 100

# Sort imports
python -m isort . --profile black --line-length 100

# Check code quality
python -m flake8 . --exclude=.venv --max-line-length=100
```

### Training Models

```bash
# Train all models
python scripts/train_models.py

# Train LSTM only
python ml_models/lstm_model.py
```

### Data Collection

```bash
# Manual data fetch
python scripts/automated_data_fetch.py

# Schedule with Task Scheduler (Windows)
# Or cron (Linux/Mac)
0 */6 * * * cd /path/to/project && python scripts/automated_data_fetch.py
```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f
```

### Services

- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:8502
- **MongoDB**: localhost:27017 (if using local MongoDB)

### Manual Docker Build

```bash
# Build backend
docker build -f Dockerfile.backend -t pearl-aqi-backend .

# Build frontend
docker build -f Dockerfile.frontend -t pearl-aqi-frontend .

# Run backend
docker run -p 8000:8000 --env-file .env pearl-aqi-backend

# Run frontend
docker run -p 8502:8502 --env-file .env pearl-aqi-frontend
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Ensure you're in project root and venv is activated
cd "c:\Users\hjiaz tr\Downloads\10 pearl AQI"
.venv\Scripts\activate

# Add project root to PYTHONPATH
$env:PYTHONPATH = "c:\Users\hjiaz tr\Downloads\10 pearl AQI"
```

#### 2. MongoDB Connection Failed
- Check `.env` file has correct connection string
- Verify network access (whitelist your IP in MongoDB Atlas)
- Test connection: `python -c "from backend.core.database_main import AirQualityDatabase; AirQualityDatabase()"`

#### 3. API Keys Not Working
- Verify keys in `.env` file
- Check key quotas and limits
- Test with curl: `curl "https://api.openweathermap.org/data/2.5/weather?q=Karachi&appid=YOUR_KEY"`

#### 4. Port Already in Use
```bash
# Find process using port 8502
netstat -ano | findstr :8502

# Kill process (replace PID)
taskkill /F /PID <PID>
```

#### 5. Model Loading Errors
- Ensure models exist in `models/` directory
- Retrain models: `python scripts/train_models.py`
- Check model file paths in code

#### 6. NumPy Version Conflicts (TensorFlow)
```bash
# Downgrade NumPy if needed
pip install "numpy<2.0"
```

### Debug Mode

Enable debug logging in `.env`:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

### Getting Help

1. Check error logs in terminal
2. Review API docs at http://localhost:8000/docs
3. Check GitHub Issues
4. Contact support: your-email@example.com

---

## 📊 Supported Cities

### Asia
- **Pakistan**: Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi
- **India**: Delhi, Mumbai, Kolkata
- **China**: Beijing, Shanghai

### Europe
- London, Paris, Berlin

### Americas
- New York, Los Angeles, Mexico City

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

### Code Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings
- Add tests for new features
- Format with black and isort

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- OpenWeatherMap for weather data API
- World Air Quality Index (WAQI) for air quality data
- XGBoost and TensorFlow teams for ML frameworks
- Streamlit for amazing dashboard framework
- FastAPI for modern Python web framework

---

## 📞 Contact

**Project**: Pearl AQI Intelligence Platform  
**Version**: 2.0  
**Last Updated**: January 16, 2026  
**Status**: Production Ready ✅

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for cleaner air**
