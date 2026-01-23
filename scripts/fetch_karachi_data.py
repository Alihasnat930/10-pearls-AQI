"""
Quick script to fetch fresh data for a specific city
Run this to get data for Karachi if dashboard shows wrong city
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.services.api_fetcher import AirQualityAPIFetcher
from backend.core.database_main import AirQualityDatabase

def fetch_city_data(city_name, lat, lon, country):
    """Fetch and store data for a specific city"""
    print(f"\n{'='*60}")
    print(f"🔄 Fetching data for {city_name}, {country}")
    print(f"{'='*60}\n")
    
    # Initialize components
    api_fetcher = AirQualityAPIFetcher()
    db = AirQualityDatabase()
    
    # Set location
    api_fetcher.set_location(city=city_name, latitude=lat, longitude=lon, country=country)
    
    print(f"📍 Location set to: {city_name}")
    print(f"🌐 Coordinates: {lat}, {lon}")
    print(f"🗺️  Country: {country}\n")
    
    # Fetch data
    print("📡 Fetching live data from APIs...")
    data = api_fetcher.fetch_combined_data()
    
    if not data:
        print("⚠️  No live data available, generating sample data...")
        data = api_fetcher.generate_mock_data()
    
    # Store in database
    if data:
        print(f"\n✅ Data received!")
        print(f"   AQI: {data['AQI']:.0f}")
        print(f"   Temperature: {data.get('Temperature', 'N/A')}°C")
        print(f"   PM2.5: {data.get('PM2.5', 'N/A')} µg/m³")
        print(f"   PM10: {data.get('PM10', 'N/A')} µg/m³")
        
        print(f"\n💾 Storing in database...")
        db.insert_live_data(data)
        print(f"✅ Data stored successfully!")
        
        # Verify
        print(f"\n🔍 Verifying database...")
        stats = db.get_data_statistics()
        print(f"   Live records: {stats.get('live', {}).get('count', 0)}")
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {city_name} data is ready!")
        print(f"{'='*60}\n")
        print(f"🌐 Now open your dashboard and select {city_name}")
        return True
    else:
        print(f"\n❌ Failed to get data for {city_name}")
        return False

if __name__ == "__main__":
    # Karachi coordinates
    KARACHI = {
        "city": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011,
        "country": "Pakistan"
    }
    
    # You can easily change this to any other city
    fetch_city_data(
        city_name=KARACHI["city"],
        lat=KARACHI["lat"],
        lon=KARACHI["lon"],
        country=KARACHI["country"]
    )
    
    print("\n💡 TIP: Run this script anytime you want fresh data for a city!")
    print("     You can edit the KARACHI dictionary to fetch data for other cities.\n")
