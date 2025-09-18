@echo off
echo Starting Face Recognition API Server
echo ====================================

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import fastapi, uvicorn, cv2, deepface" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Initialize database
echo Initializing database...
python -c "from database.db import ensure_table_exists; ensure_table_exists(); print('Database initialized')"

echo.
echo Starting API server on http://localhost:8000
echo API Documentation: http://localhost:8000/api/docs
echo.

REM Start the server
python main.py

pause