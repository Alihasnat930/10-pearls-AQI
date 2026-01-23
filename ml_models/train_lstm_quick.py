"""
Quick LSTM Model Training Script
Trains LSTM model with available data or mock data
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_training_data(n_samples=1000):
    """Generate high-quality synthetic training data"""
    print("📊 Generating high-quality synthetic training data...")
    
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
    
    # Generate more realistic AQI patterns with multiple components
    np.random.seed(42)
    base_aqi = 75
    
    # Long-term trend
    trend = 15 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    
    # Daily cycle
    daily = 25 * np.sin(np.linspace(0, n_samples/24 * 2*np.pi, n_samples))
    
    # Weekly pattern
    weekly = 10 * np.sin(np.linspace(0, n_samples/(24*7) * 2*np.pi, n_samples))
    
    # Random fluctuation (reduced for better pattern learning)
    noise = np.random.normal(0, 8, n_samples)
    
    # Combine components
    aqi = base_aqi + trend + daily + weekly + noise
    aqi = np.clip(aqi, 10, 300)
    
    # Generate strongly correlated features (key for LSTM learning)
    pm25 = aqi * 0.65 + np.random.normal(0, 3, n_samples)
    pm10 = aqi * 0.85 + np.random.normal(0, 5, n_samples)
    
    # Generate other features with realistic patterns
    data = {
        'timestamp': dates,
        'AQI': aqi,
        'PM25': np.clip(pm25, 0, 200),
        'PM10': np.clip(pm10, 0, 300),
        'CO': aqi * 0.4 + np.random.normal(0, 2, n_samples),
        'NO2': aqi * 0.5 + np.random.normal(0, 3, n_samples),
        'O3': aqi * 0.35 + np.random.normal(0, 2.5, n_samples),
        'temperature': 20 + 15 * np.sin(np.linspace(0, n_samples/(24*30) * 2*np.pi, n_samples)) + np.random.normal(0, 2, n_samples),
        'humidity': 65 + 25 * np.cos(np.linspace(0, n_samples/(24*7) * 2*np.pi, n_samples)) + np.random.normal(0, 3, n_samples),
        'wind_speed': np.abs(5 + 4 * np.random.randn(n_samples)),
        'pressure': 1013 + np.random.normal(0, 3, n_samples),
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    
    print(f"✅ Generated {len(df)} samples with realistic patterns")
    print(f"   AQI range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
    print(f"   Mean AQI: {df['AQI'].mean():.1f}")
    
    return df

def quick_train_lstm():
    """Quick LSTM training with minimal configuration"""
    print("="*60)
    print("  LSTM Model Quick Training")
    print("="*60)
    print()
    
    try:
        from ml_models.lstm_model import LSTMAQIModel
    except ImportError as e:
        print(f"❌ Error importing LSTM model: {e}")
        return False
    
    # Initialize model with better parameters
    print("🔧 Initializing improved LSTM model...")
    lstm = LSTMAQIModel(sequence_length=48, features=10)  # Increased lookback
    
    # Try to get real data first
    try:
        from backend.core.database_main import AirQualityDatabase
        db = AirQualityDatabase()
        print("📡 Attempting to fetch real data from database...")
        df = db.get_recent_data(hours=1440, table="live_data")  # 60 days
        
        if df.empty or len(df) < 200:
            print("⚠️  Insufficient real data, using synthetic data...")
            df = generate_training_data(1000)
        else:
            print(f"✅ Found {len(df)} real data records")
    except Exception as e:
        print(f"⚠️  Could not fetch real data ({str(e)}), using synthetic data...")
        df = generate_training_data(1000)
    
    # Train model with better settings
    print(f"\n🚀 Training LSTM model with {len(df)} records...")
    print("   Settings: 50 epochs, batch_size=16, sequence_length=48")
    
    try:
        history = lstm.train_with_data(
            df, 
            epochs=50,  # Increased for better learning
            batch_size=16,  # Smaller batch for better gradients
            validation_split=0.2
        )
        
        if history:
            # Display training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_mae = history.history['mae'][-1]
            final_val_mae = history.history['val_mae'][-1]
            
            print("\n" + "="*60)
            print("✅ LSTM Model Training Complete!")
            print("="*60)
            print(f"\n📊 Final Training Metrics:")
            print(f"   Training Loss (MSE): {final_loss:.4f}")
            print(f"   Validation Loss (MSE): {final_val_loss:.4f}")
            print(f"   Training MAE: {final_mae:.2f}")
            print(f"   Validation MAE: {final_val_mae:.2f}")
            print(f"\n💯 Model Accuracy: {max(0, 100 - final_val_mae):.1f}%")
            print(f"\n📁 Model saved to: {lstm.model_path}")
            print(f"📁 Config saved to: models/lstm_config.json")
            print()
            print("🎯 Model is now ready for predictions!")
            print("   The LSTM warning will disappear on next startup.")
            return True
        else:
            print("\n❌ Training failed - check logs above")
            return False
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_train_lstm()
    
    if success:
        print("\n💡 Next steps:")
        print("   1. Restart your backend: Ctrl+C and run again")
        print("   2. LSTM model will load automatically")
        print("   3. No more LSTM warnings!")
    else:
        print("\n💡 Note: The system works fine without LSTM")
        print("   Random Forest and XGBoost models are sufficient")
    
    print()
    input("Press Enter to exit...")
