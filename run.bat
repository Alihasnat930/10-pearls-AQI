@echo off
REM ===============================================
REM Pearl AQI Intelligence Platform - Launcher
REM Version 2.0 - Complete Project Runner
REM ===============================================

title Pearl AQI - Main Launcher

:MENU
cls
echo.
echo ================================================
echo     Pearl AQI Intelligence Platform v2.0
echo     Complete Air Quality Monitoring System
echo ================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found - creating now...
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Please make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created!
    echo.
    echo [INFO] Installing required packages...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo [SUCCESS] All packages installed!
    echo.
    pause
)

REM Activate virtual environment
call .venv\Scripts\activate.bat


echo [MENU] Select an option:
echo.
echo [1] Start Dashboard Only (Streamlit)
echo [2] Start API Backend Only (FastAPI)
echo [3] Start Full Stack (Dashboard + API)
echo [4] Run Tests
echo [5] Train ML Models
echo [6] Fetch New Data
echo [7] Docker Deployment
echo [8] Check System Status
echo [0] Exit
echo.
set /p choice="Enter your choice [0-8]: "

if "%choice%"=="1" goto DASHBOARD
if "%choice%"=="2" goto BACKEND
if "%choice%"=="3" goto FULLSTACK
if "%choice%"=="4" goto TESTS
if "%choice%"=="5" goto TRAIN
if "%choice%"=="6" goto FETCH
if "%choice%"=="7" goto DOCKER
if "%choice%"=="8" goto STATUS
if "%choice%"=="0" goto EXIT
if "%choice%"=="9" goto EXIT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 /nobreak >nul
goto MENU

REM ===============================================
REM Option 1: Dashboard Only
REM ===============================================
:DASHBOARD
cls
echo.
echo ================================================
echo Starting Streamlit Dashboard...
echo ================================================
echo.
echo Dashboard will open at: http://localhost:8502
echo Press Ctrl+C to stop
echo.
timeout /t 2 /nobreak >nul
streamlit run frontend\dashboard_enhanced.py --server.port 8502
goto MENU

REM ===============================================
REM Option 2: Backend API Only
REM ===============================================
:BACKEND
cls
echo.
echo ================================================
echo Starting FastAPI Backend...
echo ================================================
echo.
echo API will be available at: http://localhost:8000
echo API Docs at: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo.
timeout /t 2 /nobreak >nul
cd backend
python main.py
cd ..
goto MENU

REM ===============================================
REM Option 3: Full Stack (Dashboard + API)
REM ===============================================
:FULLSTACK
cls
echo.
echo ================================================
echo Starting Full Stack Application...
echo ================================================
echo.
echo This will open 2 new windows:
echo   1. Backend API (FastAPI)
echo   2. Frontend Dashboard (Streamlit)
echo.
echo Starting Backend API...
start "Pearl AQI - Backend API" cmd /k "cd /d "%CD%" && call .venv\Scripts\activate.bat && cd backend && python main.py"
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend Dashboard...
start "Pearl AQI - Frontend Dashboard" cmd /k "cd /d "%CD%" && call .venv\Scripts\activate.bat && streamlit run frontend\dashboard_enhanced.py --server.port 8502"

echo.
echo ================================================
echo Full Stack Started Successfully!
echo ================================================
echo.
echo Backend API:   http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo Dashboard:     http://localhost:8502
echo.
echo Both services are running in separate windows.
echo Close those windows to stop the services.
echo.
pause
goto MENU

REM ===============================================
REM Option 4: Run Tests
REM ===============================================
:TESTS
cls
echo.
echo ================================================
echo Running Test Suite...
echo ================================================
echo.

REM Check if pytest is installed
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo [WARNING] pytest not found. Installing...
    pip install pytest pytest-cov pytest-asyncio
)

echo Running all tests...
echo.
pytest tests\ -v --cov=backend --cov=ml_models --cov-report=term-missing

echo.
echo ================================================
echo Tests Complete!
echo ================================================
echo.
pause
goto MENU

REM ===============================================
REM Option 5: Train ML Models
REM ===============================================
:TRAIN
cls
echo.
echo ================================================
echo Training ML Models...
echo ================================================
echo.
echo [1] Train All Models (XGBoost, Random Forest, LSTM)
echo [2] Train XGBoost and Random Forest Only
echo [3] Train LSTM Only
echo [4] Back to Main Menu
echo.
set /p train_choice="Select training option [1-4]: "

if "%train_choice%"=="1" (
    echo.
    echo Training all models...
    python models\train_models.py
    echo.
    echo Training LSTM model...
    python ml_models\lstm_model.py
) else if "%train_choice%"=="2" (
    echo.
    echo Training XGBoost and Random Forest...
    python models\train_models.py
) else if "%train_choice%"=="3" (
    echo.
    echo Training LSTM model...
    python ml_models\lstm_model.py
) else if "%train_choice%"=="4" (
    goto MENU
) else (
    echo Invalid choice.
    timeout /t 2 /nobreak >nul
    goto TRAIN
)

echo.
echo ================================================
echo Model Training Complete!
echo ================================================
echo.
echo Models saved in: models/
echo.
pause
goto MENU

REM ===============================================
REM Option 6: Fetch New Data
REM ===============================================
:FETCH
cls
echo.
echo ================================================
echo Fetching Fresh Air Quality Data...
echo ================================================
echo.
echo Fetching data for all configured cities...
echo This may take a few minutes...
echo.

python scripts\automated_data_fetch.py

echo.
echo ================================================
echo Data Fetch Complete!
echo ================================================
echo.
echo Data stored in MongoDB database.
echo.
pause
goto MENU

REM ===============================================
REM Option 7: Docker Deployment
REM ===============================================
:DOCKER
cls
echo.
echo ================================================
echo Docker Deployment
echo ================================================
echo.
echo [1] Build and Start All Services
echo [2] Start Services (already built)
echo [3] Stop All Services
echo [4] View Logs
echo [5] Rebuild Images
echo [6] Back to Main Menu
echo.
set /p docker_choice="Select option [1-6]: "

if "%docker_choice%"=="1" (
    echo.
    echo Building and starting Docker services...
    docker-compose up --build -d
    echo.
    echo Services started!
    echo Backend API:   http://localhost:8000
    echo Dashboard:     http://localhost:8502
) else if "%docker_choice%"=="2" (
    echo.
    echo Starting Docker services...
    docker-compose up -d
    echo Services started!
) else if "%docker_choice%"=="3" (
    echo.
    echo Stopping Docker services...
    docker-compose down
    echo Services stopped!
) else if "%docker_choice%"=="4" (
    echo.
    echo Viewing logs (Press Ctrl+C to exit)...
    docker-compose logs -f
) else if "%docker_choice%"=="5" (
    echo.
    echo Rebuilding Docker images...
    docker-compose build --no-cache
    echo Rebuild complete!
) else if "%docker_choice%"=="6" (
    goto MENU
) else (
    echo Invalid choice.
    timeout /t 2 /nobreak >nul
    goto DOCKER
)

echo.
pause
goto MENU

REM ===============================================
REM Option 8: System Status Check
REM ===============================================
:STATUS
cls
echo.
echo ================================================
echo System Status Check
echo ================================================
echo.

echo [1] Checking Python version...
python --version
echo.

echo [2] Checking virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found
) else (
    echo [ERROR] Virtual environment not found!
)
echo.

echo [3] Checking required packages...
python -c "import fastapi, streamlit, pymongo, xgboost, tensorflow; print('[OK] All core packages installed')" 2>nul || echo [WARNING] Some packages may be missing
echo.

echo [4] Checking MongoDB connection...
python -c "from backend.core.database_main import AirQualityDatabase; db = AirQualityDatabase(); print('[OK] MongoDB connected successfully')" 2>nul || echo [ERROR] MongoDB connection failed
echo.

echo [5] Checking model files...
if exist "models\xgboost_model.json" (
    echo [OK] XGBoost model found
) else (
    echo [WARNING] XGBoost model not found. Run option 5 to train.
)
echo.

echo [6] Checking port availability...
netstat -an | findstr ":8000.*LISTENING" >nul
if errorlevel 1 (
    echo [OK] Port 8000 available for backend
) else (
    echo [WARNING] Port 8000 is in use
)

netstat -an | findstr ":8502.*LISTENING" >nul
if errorlevel 1 (
    echo [OK] Port 8502 available for dashboard
) else (
    echo [WARNING] Port 8502 is in use
)
echo.

echo [7] File structure...
echo Root files: 
dir /b *.md *.txt *.json *.bat 2>nul | find /c /v ""
echo Backend files: 
dir /b /s backend\*.py 2>nul | find /c /v ""
echo Frontend files: 
dir /b /s frontend\*.py 2>nul | find /c /v ""
echo.

echo ================================================
echo Status Check Complete!
echo ================================================
echo.
pause
goto MENU

REM ===============================================
REM Exit
REM ===============================================
:EXIT
cls
echo.
echo ================================================
echo Thank you for using Pearl AQI!
echo ================================================
echo.
echo Cleaning up...
timeout /t 1 /nobreak >nul
exit /b 0
