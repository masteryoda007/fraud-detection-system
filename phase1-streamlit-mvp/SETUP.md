# 🚀 Setup Guide - AI Fraud Detection System

**Author**: Sreekumar Prathap | **LinkedIn**: [sreekumar-prathap-22b36a13](https://www.linkedin.com/in/sreekumar-prathap-22b36a13/)

This guide covers multiple setup approaches for different development preferences and deployment scenarios.

---

## 📋 Quick Start Options

| **Method** | **Best For** | **Time** | **Complexity** |
|-----------|--------------|----------|----------------|
| [🐍 Conda Setup](#-conda-setup-recommended) | Data scientists, ML engineers | 5 min | Easy |
| [📦 Pip Setup](#-pip-setup-alternative) | Python developers, CI/CD | 10 min | Medium |
| [☁️ Cloud Deploy](#️-streamlit-cloud-deployment) | Sharing, demos | 3 min | Easy |

---

## 🐍 Conda Setup (Recommended)

**Ideal for**: Data science workflows, M1/M2 Macs, complex ML dependencies

### Prerequisites
```bash
# Install Miniconda (if not already installed)
# macOS:
brew install miniconda
# Or download from: https://docs.conda.io/en/latest/miniconda.html
```

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/masteryoda007/fraud-detection-system.git
cd fraud-detection-system

# 2. Create conda environment
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection

# 3. Install core ML packages (conda-forge has latest optimized versions)
conda install -c conda-forge -y \
    lightgbm=4.6.0 \
    shap=0.48.0 \
    faiss-cpu=1.9.0 \
    xgboost=2.0.3 \
    scikit-learn=1.3.2 \
    pandas=2.1.4 \
    numpy=1.25.2 \
    plotly=5.17.0 \
    matplotlib=3.8.2 \
    streamlit=1.40.0

# 4. Install remaining packages via pip
pip install imbalanced-learn==0.11.0 openai==1.6.1

# 5. Verify installation
python -c "import lightgbm, shap, faiss; print('✅ All ML packages installed!')"
```

### Run Application
```bash
# Navigate to app directory
cd phase1-streamlit-mvp

# Launch Streamlit app
streamlit run app.py

# 🎉 App opens at: http://localhost:8501
```

### Why Conda?
- ✅ **Optimized binaries**: Faster ML computations
- ✅ **Dependency resolution**: Handles complex ML package conflicts
- ✅ **M1/M2 Mac support**: No compilation issues
- ✅ **Reproducible**: Exact environment recreation

---

## 📦 Pip Setup (Alternative)

**Ideal for**: Docker, CI/CD, lightweight deployments

### Prerequisites
```bash
# Python 3.11+ required
python --version  # Should be 3.11+

# Virtual environment (recommended)
python -m venv fraud-detection-env
source fraud-detection-env/bin/activate  # macOS/Linux
# fraud-detection-env\Scripts\activate    # Windows
```

### Installation
```bash
# 1. Clone repository
git clone https://github.com/masteryoda007/fraud-detection-system.git
cd fraud-detection-system

# 2. Install from minimal requirements (local development)
pip install -r phase1-streamlit-mvp/requirements.txt

# 3. Add optional ML packages (if no compilation issues)
pip install lightgbm==4.5.0 shap==0.46.0 faiss-cpu==1.8.0

# 4. Run application
cd phase1-streamlit-mvp
streamlit run app.py
```

### Troubleshooting Pip Issues
```bash
# If lightgbm fails to compile (especially on older systems):
# The app will still work with 3/4 ML models

# Alternative: Use conda for problematic packages
conda install lightgbm shap faiss-cpu
pip install streamlit pandas plotly  # Rest via pip
```

---

## ☁️ Streamlit Cloud Deployment

**For sharing and portfolio demonstration**

### Prerequisites
- GitHub repository (✅ you have this)
- Streamlit Cloud account ([share.streamlit.io](https://share.streamlit.io))

### Deployment Setup
```bash
# 1. Ensure requirements.txt in repository ROOT
# (Already done for this project!)
ls requirements.txt  # Should exist in fraud-detection-system/

# 2. Deploy via Streamlit Cloud:
# - Connect GitHub repo
# - Set main file: phase1-streamlit-mvp/app.py
# - Auto-deploys on push to main branch

# 3. App URL: https://ai-fraud-detection-app.streamlit.app
```

### Cloud vs Local Differences
| **Aspect** | **Local (Conda)** | **Cloud (Pip)** |
|------------|-------------------|-----------------|
| Package Manager | conda-forge | PyPI |
| LightGBM Version | 4.6.0 | 4.5.0 |
| SHAP Version | 0.48.0 | 0.46.0 |
| Setup Time | 5 minutes | Auto (3 min) |
| Customization | Full control | Limited |

---

## 🧪 Feature Availability

### Core Features (Always Available)
- ✅ **Advanced Dashboard**: Transaction analysis, visualizations
- ✅ **Live Analysis**: Real-time fraud scoring
- ✅ **Model Performance**: 4-model ensemble comparison
- ✅ **Rule-based Chat**: Natural language fraud analysis

### Optional Features (Require Additional Packages)
- 🔬 **SHAP Explanations**: Install `shap` package
- 🔍 **Vector Search**: Install `faiss-cpu` package  
- 🤖 **AI Chat**: Set `OPENAI_API_KEY` environment variable
- ⚡ **LightGBM Model**: Install `lightgbm` package

### Check Feature Status
```bash
# Run this to see what's available:
python -c "
try:
    import lightgbm; print('✅ LightGBM available')
except: print('❌ LightGBM missing')
    
try:
    import shap; print('✅ SHAP available')
except: print('❌ SHAP missing')
    
try:
    import faiss; print('✅ FAISS available')
except: print('❌ FAISS missing')
"
```

---

## 🛠️ Development Workflow

### Recommended Development Process
```bash
# 1. Setup (one time)
conda create -n fraud-detection python=3.11
conda activate fraud-detection
# ... install packages as above

# 2. Daily development
conda activate fraud-detection
cd fraud-detection-system/phase1-streamlit-mvp
streamlit run app.py

# 3. Code changes
# Edit app.py, save, auto-reloads in browser

# 4. Deploy updates
git add .
git commit -m "Feature: Your changes"
git push origin main
# Auto-deploys to cloud!
```

### Performance Optimization
```bash
# For faster startup (optional):
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501

# For development mode:
streamlit run app.py --server.runOnSave=true
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### "ModuleNotFoundError: No module named 'lightgbm'"
```bash
# Local fix:
conda install -c conda-forge lightgbm

# Cloud fix:
# Ensure requirements.txt in repo root includes lightgbm
```

#### "SHAP explanations not available"
```bash
conda install -c conda-forge shap
# Or: pip install shap
```

#### "OpenAI API not connected"
```bash
# Set environment variable:
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file (don't commit!):
echo "OPENAI_API_KEY=your-key" > .env
```

#### App running slowly
```bash
# Check memory usage:
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")

# Reduce data size for testing:
# Edit app.py, reduce n_samples in load_real_fraud_dataset()
```

#### Port already in use
```bash
# Kill existing process:
lsof -ti:8501 | xargs kill -9

# Or use different port:
streamlit run app.py --server.port=8502
```

---

## 📊 Development Environment Details

### Author's Setup (Recommended)
- **OS**: macOS (M1/M2 optimized)
- **Python**: 3.11.x
- **Package Manager**: Conda (conda-forge channel)
- **IDE**: VSCode with Python extension
- **Browser**: Chrome (best Streamlit support)

### Tested Environments
| **OS** | **Python** | **Package Manager** | **Status** |
|--------|------------|-------------------|------------|
| macOS M1/M2 | 3.11 | Conda | ✅ Excellent |
| macOS Intel | 3.11 | Conda | ✅ Excellent |
| Ubuntu 20.04+ | 3.11 | Pip | ✅ Good |
| Windows 11 | 3.11 | Conda | ✅ Good |
| Streamlit Cloud | 3.11 | Pip | ✅ Excellent |

---

## 🎯 Next Steps

### After Setup
1. **🔍 Explore Features**: Try all tabs in the application
2. **📊 Analyze Data**: Use the Live Analysis tab with different scenarios  
3. **🤖 Chat with AI**: Set OpenAI key and try the AI chat feature
4. **🔎 Vector Search**: Find similar fraud patterns
5. **📈 Model Performance**: Compare the 4-model ensemble

### For Developers
1. **📖 Code Review**: Study the ML pipeline in `app.py`
2. **🧪 Experiments**: Modify parameters and see results
3. **🚀 Contribute**: Submit issues/PRs to improve the system
4. **📝 Documentation**: Add your own insights to this guide

### For Portfolio Use
1. **🔗 Share**: Send the Streamlit Cloud URL to employers
2. **📊 Demo**: Walk through the features in interviews
3. **💼 Context**: Explain the technical decisions and trade-offs
4. **🎯 Results**: Highlight the 99%+ AUC performance

---

## 💬 Support & Contact

- **💼 LinkedIn**: [sreekumar-prathap-22b36a13](https://www.linkedin.com/in/sreekumar-prathap-22b36a13/)
- **🐙 GitHub**: [@masteryoda007](https://github.com/masteryoda007)
- **📧 Issues**: Use GitHub Issues for bug reports
- **💡 Features**: Use GitHub Discussions for feature requests

---

## 📄 License & Attribution

This is a portfolio project demonstrating advanced ML engineering capabilities. Feel free to:
- ✅ Fork and modify for learning
- ✅ Reference in academic work (with attribution)
- ✅ Use as inspiration for your own projects
- ❌ Use commercially without permission

**Built with expertise in Python, ML algorithms, production systems, and fintech domain knowledge.**

© 2025 Sreekumar Prathap - Portfolio Project Showcasing Advanced AI/ML Capabilities