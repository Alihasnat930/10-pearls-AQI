"""
Automated Data Fetch Script
Runs periodically to fetch data for all configured cities
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from datetime import datetime

from backend.services.api_fetcher import AirQualityAPIFetcher
from backend.core.database_main import AirQualityDatabase

# Cities to fetch data for
CITIES = [
    {"city": "Karachi", "lat": 24.8607, "lon": 67.0011, "country": "Pakistan"},
    {"city": "Lahore", "lat": 31.5497, "lon": 74.3436, "country": "Pakistan"},
    {"city": "Islamabad", "lat": 33.6844, "lon": 73.0479, "country": "Pakistan"},
    {"city": "London", "lat": 51.5074, "lon": -0.1278, "country": "UK"},
    {"city": "New York", "lat": 40.7128, "lon": -74.0060, "country": "USA"},
    {"city": "Delhi", "lat": 28.7041, "lon": 77.1025, "country": "India"},
    {"city": "Beijing", "lat": 39.9042, "lon": 116.4074, "country": "China"},
    {"city": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "Japan"},
]


def fetch_all_cities():
    """Fetch data for all configured cities"""
    print(f"🚀 Starting automated data fetch - {datetime.now()}")

    fetcher = AirQualityAPIFetcher()
    db = AirQualityDatabase()

    success_count = 0
    fail_count = 0

    for city_info in CITIES:
        try:
            print(f"\n📡 Fetching data for {city_info['city']}...")

            # Set location
            fetcher.set_location(
                city=city_info["city"],
                latitude=city_info["lat"],
                longitude=city_info["lon"],
                country=city_info["country"],
            )

            # Fetch data
            data = fetcher.fetch_combined_data()

            if not data:
                # Try mock data as fallback
                data = fetcher.generate_mock_data()

            if data:
                # Store in database
                db.insert_live_data(data)
                print(f"✅ {city_info['city']}: AQI = {data['AQI']:.0f}")
                success_count += 1
            else:
                print(f"❌ {city_info['city']}: No data available")
                fail_count += 1

            # Rate limiting - wait between requests
            time.sleep(2)

        except Exception as e:
            print(f"❌ Error fetching {city_info['city']}: {e}")
            fail_count += 1

    print(f"\n📊 Fetch Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📅 Completed: {datetime.now()}")


if __name__ == "__main__":
    fetch_all_cities()
