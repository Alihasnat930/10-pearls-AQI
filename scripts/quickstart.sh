#!/bin/bash
# Quick Start Script for Pearl AQI Platform

echo "🚀 Pearl AQI Intelligence Platform - Quick Start"
echo "================================================"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=pearl_aqi_db

# External APIs (optional)
OPENWEATHER_API_KEY=your_key_here
WAQI_API_KEY=your_key_here
EOF
    echo "✓ .env template created. Please update with your credentials."
else
    echo "✓ .env file found"
fi

# Test MongoDB connection
echo "🔌 Testing MongoDB connection..."
python -c "from database import AirQualityDatabase; db = AirQualityDatabase(); print('✓ MongoDB connected')"

# Options menu
echo ""
echo "What would you like to run?"
echo "1) FastAPI Backend (port 8000)"
echo "2) Streamlit Dashboard (port 8502)"
echo "3) Both (Backend + Frontend)"
echo "4) Docker Compose (all services)"
echo "5) Train LSTM Model"
echo "6) Run Tests"
echo "7) Fetch Data for All Cities"

read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo "🚀 Starting FastAPI Backend..."
        cd backend
        python main.py
        ;;
    2)
        echo "🚀 Starting Streamlit Dashboard..."
        streamlit run frontend/dashboard_enhanced.py --server.port 8502
        ;;
    3)
        echo "🚀 Starting Backend and Frontend..."
        cd backend
        python main.py &
        cd ..
        streamlit run frontend/dashboard_enhanced.py --server.port 8502
        ;;
    4)
        echo "🐳 Starting Docker Compose..."
        docker-compose up --build
        ;;
    5)
        echo "🧠 Training LSTM Model..."
        python ml_models/lstm_model.py
        ;;
    6)
        echo "🧪 Running Tests..."
        pytest tests/ --cov=backend --cov=ml_models -v
        ;;
    7)
        echo "📡 Fetching Data..."
        python scripts/automated_data_fetch.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
