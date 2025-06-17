# 🗂️ Recommended Project Structure

fraud-detection-system/
├── README.md                    # Main project overview
├── .gitignore                  # Git ignore rules
├── TODO.md                     # Project roadmap
├── docs/                       # Project documentation
├── phase1-streamlit-mvp/       # 🔒 Phase 1: Advanced ML MVP
│   ├── app.py                  # Main Streamlit application
│   ├── requirements.txt        # Python dependencies
│   ├── README.md              # Phase 1 specific documentation
│   ├── quick_test.py          # Environment testing
│   ├── setup/                 # Setup scripts
│   │   ├── setup_script.sh    # Linux/Mac setup
│   │   ├── setup.bat          # Windows setup
│   │   └── env_fix_script.sh  # Environment fix
│   └── data/                  # Generated data (ignored by Git)
├── phase2-production-backend/  # 🚀 Phase 2: FastAPI Backend
├── phase3-advanced-ai/        # 🧠 Phase 3: Advanced AI Features  
└── phase4-enterprise/         # 🏢 Phase 4: Enterprise Deployment

# Commands to organize:
cd /Users/yoda/sree_ai_ml_projects/fraud-detection-system/phase1-streamlit-mvp

# Create setup directory and move scripts
mkdir -p setup
mv setup_script.sh setup/
mv env_fix_script.sh setup/

# Convert setup_bat.txt to proper .bat file
if [ -f setup_bat.txt ]; then
    mv setup_bat.txt setup/setup.bat
fi

# Remove duplicate files
rm -f gitignore.txt readme_md.md requirements_txt.txt

# Keep app_backup.py as backup or remove it
# mv app_backup.py app_backup_$(date +%Y%m%d).py  # Timestamp backup
# or
# rm app_backup.py  # Remove if app.py is the latest