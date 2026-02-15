@echo off
REM ===============================================
REM Pearl AQI - Quick Start Script
REM ===============================================

title Pearl AQI - Quick Start

echo ================================================
echo    Pearl AQI Intelligence Platform
echo    Quick Start - Complete Project
echo ================================================
echo.

REM Check if virtual environment exists
set "ROOT=%CD%"
set "VENV_DIR=%ROOT%\.venv"
if exist "%ROOT%\.venv_tf\Scripts\activate.bat" set "VENV_DIR=%ROOT%\.venv_tf"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

if not exist "%VENV_ACTIVATE%" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Please ensure Python 3.11+ is installed.
        pause
        exit /b 1
    )
    
    call "%VENV_ACTIVATE%"
    echo.
    echo [SETUP] Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Setup complete!
    echo.
)

REM Activate virtual environment
call "%VENV_ACTIVATE%"

echo [INFO] Starting Full Stack Application...
echo.
echo This will start:
echo   - Backend API at http://localhost:8000
echo   - Frontend Dashboard at http://localhost:8502
echo.

REM Start backend in a new window
start "Pearl AQI - Backend API" cmd /k "cd /d "%ROOT%\backend" && call "%VENV_ACTIVATE%" && python main.py"

echo [INFO] Waiting for backend to start...
timeout /t 5 /nobreak >nul

REM Start frontend in a new window
start "Pearl AQI - Dashboard" cmd /k "cd /d "%ROOT%" && call "%VENV_ACTIVATE%" && python -m streamlit run frontend\dashboard_enhanced.py --server.port 8502 --server.headless false"

echo.
echo ================================================
echo     Application Started Successfully!
echo ================================================
echo.
echo Backend API:   http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo Dashboard:     http://localhost:8502
echo.
echo Two windows have been opened for the services.
echo Close those windows or press Ctrl+C to stop.
echo.
echo You can close this window now.
echo.
pause
