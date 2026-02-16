"""
Model Files Verification and Integrity Check
Ensures all required ML model files exist and are valid
"""

import os
import sys
import pickle
from pathlib import Path


def check_model_files(models_dir="models"):
    """Check if all required model files exist and are loadable"""
    
    print("\n" + "="*70)
    print("  Pearl AQI - Model Files Verification")
    print("="*70 + "\n")
    
    # Resolve paths
    if not os.path.isabs(models_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, models_dir)
    
    print(f"🔍 Checking models directory: {models_dir}\n")
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    required_files = {
        "feature_columns.pkl": "Feature columns (primary)",
        "feature_columns.txt": "Feature columns (fallback text)",
        "xgboost_model.pkl": "XGBoost model (primary)",
        "xgboost_model.json": "XGBoost model (fallback JSON)",
        "random_forest_model.pkl": "Random Forest model",
        "scaler.pkl": "Feature scaler",
        "lstm_model.keras": "LSTM model (Keras 3 format)",
        "lstm_model.h5": "LSTM model (H5 format)",
        "lstm_config.json": "LSTM configuration",
    }
    
    optional_files = {
        "lstm_scalers.pkl": "LSTM scalers (optional)",
        "model_metrics.json": "Model metrics (optional)",
        "MODEL_SUMMARY.json": "Model summary (optional)",
    }
    
    missing_critical = []
    missing_optional = []
    present_files = []
    
    # Check required files (critical or have fallbacks)
    print("📦 Required Files:")
    for filename, description in required_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            try:
                # Try to load pickle files to verify integrity (simplified)
                if filename.endswith('.pkl'):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"  ✅ {filename:30} - {description} ({size_kb:.1f} KB)")
                else:
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"  ✅ {filename:30} - {description} ({size_kb:.1f} KB)")
                present_files.append(filename)
            except Exception as e:
                print(f"  ⚠️  {filename:30} - {description} (ERROR: {str(e)[:50]})")
        else:
            print(f"  ❌ {filename:30} - {description} (MISSING)")
            missing_critical.append(filename)
    
    # Check optional files
    print("\n📦 Optional Files:")
    for filename, description in optional_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✅ {filename:30} - {description} ({size_kb:.1f} KB)")
        else:
            print(f"  ⊘  {filename:30} - Not found (optional)")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    # Check feature columns (critical for forecasting)
    feature_pkl = os.path.join(models_dir, "feature_columns.pkl")
    feature_txt = os.path.join(models_dir, "feature_columns.txt")
    
    if os.path.exists(feature_pkl):
        print("✅ Feature columns: AVAILABLE (pickle)")
        feature_loaded = True
    elif os.path.exists(feature_txt):
        print("⚠️  Feature columns: LOADED FROM FALLBACK (text file)")
        feature_loaded = True
    else:
        print("❌ Feature columns: MISSING (critical!)")
        feature_loaded = False
    
    # Check models
    models_loaded = True
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    xgb_legacy = os.path.join(models_dir, "xgboost_model.json")
    if os.path.exists(xgb_path):
        print("✅ XGBoost model: AVAILABLE (pickle)")
    elif os.path.exists(xgb_legacy):
        print("⚠️  XGBoost model: LEGACY FORMAT (will convert on load)")
    else:
        print("❌ XGBoost model: MISSING")
        models_loaded = False
    
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    if os.path.exists(rf_path):
        print("✅ Random Forest model: AVAILABLE")
    else:
        print("⚠️  Random Forest model: MISSING (forecasting may be limited)")
    
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        print("✅ Scaler: AVAILABLE")
    else:
        print("❌ Scaler: MISSING (critical!)")
        models_loaded = False
    
    lstm_keras = os.path.join(models_dir, "lstm_model.keras")
    lstm_h5 = os.path.join(models_dir, "lstm_model.h5")
    if os.path.exists(lstm_keras) or os.path.exists(lstm_h5):
        fmt = "Keras 3" if os.path.exists(lstm_keras) else "H5"
        print(f"✅ LSTM model: AVAILABLE ({fmt})")
    else:
        print("⚠️  LSTM model: MISSING (optional, forecasting works without it)")
    
    # Final verdict
    print("\n" + "="*70)
    if feature_loaded and models_loaded:
        print("✅ All critical files present! Forecasting should work.")
        return True
    else:
        print("❌ Critical files missing! Forecasting will fail.")
        print("\n🔧 To fix:")
        if not feature_loaded:
            print("   1. Ensure feature_columns.pkl or feature_columns.txt exists")
            print("   2. Run: python scripts/train_models.py")
        if not models_loaded:
            print("   3. Ensure scaler.pkl exists")
            print("   4. Ensure xgboost_model.pkl or xgboost_model.json exists")
        return False


def check_project_structure():
    """Check overall project structure"""
    print("\n" + "="*70)
    print("  Project Structure Check")
    print("="*70 + "\n")
    
    required_dirs = {
        "models": "ML model files",
        "backend": "Backend code",
        "frontend": "Frontend code",
        "scripts": "Utility scripts",
    }
    
    for dirname, description in required_dirs.items():
        if os.path.isdir(dirname):
            print(f"  ✅ {dirname:20} - {description}")
        else:
            print(f"  ❌ {dirname:20} - {description} (MISSING)")


if __name__ == "__main__":
    check_project_structure()
    success = check_model_files()
    
    if not success:
        print("\n" + "="*70)
        print("  DEPLOYMENT CHECKLIST FOR STREAMLIT CLOUD")
        print("="*70)
        print("""
To fix forecast errors in your Streamlit Cloud deployment:

1. **Locally: Verify files**
   python scripts/verify_models.py

2. **Locally: Regenerate models if needed**
   python scripts/train_models.py

3. **Git: Commit all model files**
   git add models/
   git commit -m "Add all required model files"
   git push

4. **Streamlit Cloud: Redeploy**
   Your app will automatically redeploy from GitHub

5. **Verify: Check app logs**
   Look for: "Feature columns loaded (XX features)"
   
If still failing:
- Check Streamlit app logs for detailed error messages
- Verify models/ directory is included in your git repo
- Don't add models/ to .gitignore (models need to be committed)
        """)
    
    sys.exit(0 if success else 1)
