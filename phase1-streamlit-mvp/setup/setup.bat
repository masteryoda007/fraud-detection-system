@echo off
:: 🔒 AI Fraud Detection System - Quick Setup Script (Windows)
:: Run this script to automatically set up the conda environment

echo 🔒 AI Fraud Detection System - Phase 1 Setup
echo ==============================================

:: Check if conda is installed
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ conda found

:: Create conda environment
echo 📦 Creating conda environment 'fraud-detection'...
conda create -n fraud-detection python=3.11 -y

if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment
    pause
    exit /b 1
)

echo ✅ Conda environment created

:: Activate environment and install packages
echo 📥 Installing core packages via conda...
call conda activate fraud-detection

:: Install core packages via conda
conda install -c conda-forge pandas numpy scikit-learn plotly streamlit xgboost lightgbm imbalanced-learn requests -y

if %errorlevel% neq 0 (
    echo ❌ Failed to install core packages
    pause
    exit /b 1
)

echo ✅ Core packages installed

:: Install additional packages via pip
echo 📥 Installing additional packages via pip...
pip install shap optuna faiss-cpu openai joblib

if %errorlevel% neq 0 (
    echo ⚠️  Some additional packages failed to install
    echo The app will still work, but some features may be limited
)

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo To start the application:
echo 1. conda activate fraud-detection
echo 2. streamlit run app.py
echo.
echo Optional: Set OpenAI API key for AI chat features:
echo set OPENAI_API_KEY=your-api-key-here
echo.
echo 📖 See README.md for detailed usage instructions
pause