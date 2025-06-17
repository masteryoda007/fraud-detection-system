#!/usr/bin/env python3
"""
Quick test script for local fraud detection setup
Run this to verify everything works before launching the full app
"""

def test_imports():
    """Test all required imports"""
    print("🔍 Testing core imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas + NumPy")
    except ImportError as e:
        print(f"❌ Pandas/NumPy: {e}")
        return False
    
    try:
        import plotly.express as px
        import matplotlib.pyplot as plt
        print("✅ Plotting libraries")
    except ImportError as e:
        print(f"❌ Plotting: {e}")
        return False
    
    try:
        import sklearn
        import xgboost as xgb
        print("✅ Core ML libraries")
    except ImportError as e:
        print(f"❌ ML libraries: {e}")
        return False
    
    # Test optional packages
    try:
        import lightgbm as lgb
        print("✅ LightGBM (optional)")
    except ImportError:
        print("⚠️ LightGBM not available (will use 3 models instead of 4)")
    
    try:
        import shap
        print("✅ SHAP (optional)")
    except ImportError:
        print("⚠️ SHAP not available (basic explanations will be used)")
    
    try:
        import faiss
        print("✅ FAISS (optional)")
    except ImportError:
        print("⚠️ FAISS not available (vector search disabled)")
    
    return True

def test_data_generation():
    """Test basic data generation"""
    print("\n🔍 Testing data generation...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate small test dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Amount': np.random.lognormal(3, 1, n_samples),
            'V1': np.random.normal(0, 1, n_samples),
            'V2': np.random.normal(0, 1, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        }
        
        df = pd.DataFrame(data)
        print(f"✅ Generated test dataset: {len(df)} samples")
        
        # Test model training
        X = df[['Amount', 'V1', 'V2']]
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model training successful: {accuracy:.2%} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔒 Fraud Detection System - Local Environment Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ CRITICAL: Core imports failed!")
        print("Fix the missing packages before running the app.")
        return
    
    # Test data generation
    data_ok = test_data_generation()
    
    if not data_ok:
        print("\n❌ Data generation failed!")
        return
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Your local environment is ready!")
    print("\n🚀 You can now run:")
    print("   streamlit run app.py")
    print("\n💡 Note: The app will work with 3 models if LightGBM isn't available")
    print("💡 SHAP and FAISS are optional - core functionality works without them")

if __name__ == "__main__":
    main()
