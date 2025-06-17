#!/bin/bash
# Environment Fix Script for Fraud Detection App
# Run this in your non-working environment

echo "🔧 Fixing environment for fraud detection app..."

# Activate your environment first (uncomment the line that matches your setup)
# source /Users/yoda/sree_ai_ml_projects/fraud-detection-system/phase1-streamlit-mvp/fraud_detection_env_minimal/bin/activate

echo "📦 Installing missing lightgbm..."
pip install lightgbm==4.5.0

echo "🔄 Updating critical packages to compatible versions..."
pip install --upgrade streamlit==1.35.0
pip install --upgrade pandas==2.2.0  
pip install --upgrade numpy==1.26.0
pip install --upgrade plotly==5.22.0
pip install --upgrade scikit-learn==1.4.0
pip install --upgrade xgboost==2.1.0
pip install --upgrade openai==1.50.0

echo "🧪 Installing additional missing dependencies..."
pip install imbalanced-learn==0.11.0
pip install joblib==1.3.2
pip install requests==2.31.0
pip install python-dateutil==2.8.2

# Optional: Try to install optuna if not present
pip install optuna==3.4.0

echo "✅ Environment fix complete!"
echo ""
echo "🧪 Test your installation:"
echo "python -c \"import lightgbm, streamlit, pandas, numpy, sklearn, plotly, xgboost, shap, faiss, openai; print('✅ All packages imported successfully!')\""
echo ""
echo "🚀 Run the app:"
echo "streamlit run app.py"