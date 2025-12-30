@echo off
REM Mode Miner - Setup and Run Script for Windows
REM Usage: Double-click this file or run from Command Prompt

echo === Mode Miner Setup ===

cd /d "%~dp0"

REM Clear Qt environment variables that may conflict with PySide6
REM (Fixes issues when Anaconda is installed)
set QT_PLUGIN_PATH=
set QML2_IMPORT_PATH=
set QT_QPA_PLATFORM_PLUGIN_PATH=

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Python not found.
        echo Please install Python from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    set PYTHON=py
) else (
    set PYTHON=python
)

echo Using Python: %PYTHON%
%PYTHON% --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

echo.
echo === Setup Complete ===
echo.

REM Run the application
echo Starting Mode Miner...
python run_test.py

pause

