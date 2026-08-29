@echo off
title DRDO AI ANC - Demo Launcher
cd /d "%~dp0"

echo ========================================
echo   DRDO AI ANC - Presentation Launcher
echo ========================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [1/3] Running smoke test...
python scripts\smoke_test.py
if errorlevel 1 (
    echo Smoke test FAILED - check errors above.
    pause
    exit /b 1
)

echo.
echo [2/3] Creating jury package...
python scripts\create_jury_package.py

echo.
echo [3/3] Launching dashboard...
echo Open browser at: http://localhost:8501
echo.
streamlit run app\dashboard.py
