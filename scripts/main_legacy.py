"""
Main execution script for Air Quality Forecasting System
Run this to execute the complete pipeline
"""

import os
import sys
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def main():
    """Main execution pipeline"""
    print_header("AIR QUALITY FORECASTING SYSTEM")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check if processed data exists
    if not os.path.exists("processed_air_quality.csv"):
        print("[ERROR] Processed data not found!")
        print("Please run the Jupyter notebook first:")
        print("   jupyter notebook eda_preprocessing.ipynb")
        print("\nOr run preprocessing directly:")
        response = input("Would you like to run preprocessing now? (y/n): ")
        if response.lower() == "y":
            print("\nRunning preprocessing...")
            # Run preprocessing
            import pandas as pd

            print("Loading and processing data...")
            # This would need the preprocessing logic
            print("[WARN] Please run the Jupyter notebook manually for best results.")
            return
        else:
            print("Exiting. Please preprocess data first.")
            return

    print("[OK] Processed data found\n")

    # Check if models exist
    if not os.path.exists("models/xgboost_model.json"):
        print("[ERROR] Trained models not found!")
        print("Training models now...\n")

        try:
            from ml_models.train_models import main as train_main

            train_main()
        except Exception as e:
            print(f"\n[ERROR] Error during training: {e}")
            print("Please run: python train_models.py")
            return
            return
    else:
        print("[OK] Trained models found\n")

    # Initialize database
    print_header("📊 INITIALIZING DATABASE")
    try:
        from database import AirQualityDatabase

        db = AirQualityDatabase()

        # Check if we need to populate with historical data
        stats = db.get_data_statistics()
        if stats.get("historical", {}).get("count", 0) == 0:
            print("Loading historical data into database...")
            import pandas as pd

            df = pd.read_csv("processed_air_quality.csv", index_col=0, parse_dates=True)
            db.insert_historical_data(df.head(1000))  # Load subset for demo
            print("[OK] Historical data loaded")
        else:
            print("[OK] Database already populated")

        db.close()
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")

    # Fetch live data
    print_header("🌐 FETCHING LIVE DATA")
    try:
        from api_fetcher import AirQualityAPIFetcher
        from database import AirQualityDatabase

        db = AirQualityDatabase()
        fetcher = AirQualityAPIFetcher()

        print("Attempting to fetch from APIs...")
        data = fetcher.fetch_combined_data()

        if not data:
            print("⚠️  APIs not available, using mock data")
            data = fetcher.generate_mock_data()

        db.insert_live_data(data)
        print(f"[OK] Current AQI: {data['AQI']:.1f} ({data['AQI_category']})")
        db.close()

    except Exception as e:
        print(f"⚠️  Error fetching data: {e}")

    # Launch dashboard
    print_header("LAUNCHING DASHBOARD")
    print("Starting Streamlit dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")

    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", "dashboard.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
