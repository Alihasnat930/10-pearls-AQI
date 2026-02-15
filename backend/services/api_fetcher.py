"""
API Data Fetcher Module
Fetches live weather and pollutant data from external APIs
Supports: OpenWeatherMap, AQI CN, WAQI (World Air Quality Index)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Optional

import requests


class AirQualityAPIFetcher:
    """Fetch real-time air quality and weather data from APIs"""

    def __init__(self, config_file: str = "api_config.json"):
        """Initialize with API configuration"""
        self.config = self._load_config(config_file)
        self.api_keys = self.config.get("api_keys", {})
        self.location = self.config.get("default_location", {})

        owm_key = self._get_secret("OPENWEATHER_API_KEY")
        if owm_key:
            self.api_keys["openweathermap"] = owm_key

        waqi_key = self._get_secret("WAQI_API_KEY")
        if waqi_key:
            self.api_keys["waqi"] = waqi_key

    def _get_secret(self, key: str, default: str = "") -> str:
        """Read from env, then Streamlit secrets when available."""
        value = os.getenv(key)
        if value:
            return value
        try:
            import streamlit as st
        except Exception:
            return default
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
        return default

    def set_location(
        self, city: str = None, latitude: float = None, longitude: float = None, country: str = None
    ):
        """Set the default location for subsequent API calls"""
        if city:
            self.location["city"] = city
        if latitude is not None and longitude is not None:
            self.location["latitude"] = latitude
            self.location["longitude"] = longitude
        if country:
            self.location["country"] = country

    def _load_config(self, config_file: str) -> Dict:
        """Load API configuration from JSON file"""
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                "api_keys": {
                    "openweathermap": "YOUR_OPENWEATHERMAP_API_KEY",
                    "waqi": "YOUR_WAQI_API_KEY",
                },
                "default_location": {
                    "city": "London",
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                    "country": "UK",
                },
                "update_interval_minutes": 60,
            }

            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=4)

            print(f"Created default config file: {config_file}")
            print("Please update with your API keys!")
            return default_config

    def fetch_openweathermap_data(self, lat: float = None, lon: float = None) -> Optional[Dict]:
        """
        Fetch data from OpenWeatherMap API
        Provides: Temperature, humidity, pressure, wind, and basic air pollution
        API: https://openweathermap.org/api
        """
        try:
            api_key = self.api_keys.get("openweathermap")
            if not api_key or api_key == "YOUR_OPENWEATHERMAP_API_KEY":
                print("OpenWeatherMap API key not configured")
                return None

            lat = lat or self.location.get("latitude")
            lon = lon or self.location.get("longitude")

            # Weather data
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout=10)
            weather_data = weather_response.json()

            # Air pollution data
            pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
            pollution_response = requests.get(pollution_url, timeout=10)
            pollution_data = pollution_response.json()

            if weather_response.status_code == 200 and pollution_response.status_code == 200:
                # Parse data
                result = {
                    "timestamp": datetime.fromtimestamp(weather_data["dt"]),
                    "location": weather_data["name"],
                    "temperature": weather_data["main"]["temp"],
                    "humidity": weather_data["main"]["humidity"],
                    "pressure": weather_data["main"]["pressure"],
                    "wind_speed": weather_data["wind"]["speed"],
                    "wind_direction": weather_data["wind"].get("deg", 0),
                    "api_source": "OpenWeatherMap",
                }

                # Air quality components
                if "list" in pollution_data and len(pollution_data["list"]) > 0:
                    components = pollution_data["list"][0]["components"]
                    result.update(
                        {
                            "CO": components.get("co", 0) / 1000,  # Convert to mg/m³
                            "NO2": components.get("no2", 0),
                            "NOx": components.get("no", 0)
                            + components.get("no2", 0),  # Approximate
                            "O3": components.get("o3", 0),
                            "PM25": components.get("pm2_5", 0),
                            "PM10": components.get("pm10", 0),
                            "AQI": pollution_data["list"][0]["main"]["aqi"]
                            * 50,  # Convert to US AQI scale (rough)
                        }
                    )

                    # Categorize AQI
                    result["AQI_category"] = self._categorize_aqi(result["AQI"])

                print(f"Fetched OpenWeatherMap data for {result['location']}")
                return result
            else:
                print(f"Error: OpenWeatherMap API returned status {weather_response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching OpenWeatherMap data: {e}")
            return None

    def geocode_city(self, city: str, country: str = None) -> Optional[tuple]:
        """Geocode a city name to (lat, lon) using OpenWeatherMap Geocoding API"""
        try:
            api_key = self.api_keys.get("openweathermap")
            if not api_key or api_key == "YOUR_OPENWEATHERMAP_API_KEY":
                print("OpenWeatherMap API key not configured for geocoding")
                return None

            q = city
            if country:
                q = f"{city},{country}"

            url = f"http://api.openweathermap.org/geo/1.0/direct?q={q}&limit=1&appid={api_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if resp.status_code == 200 and isinstance(data, list) and len(data) > 0:
                return (data[0]["lat"], data[0]["lon"])
            return None
        except Exception as e:
            print(f"Error geocoding city: {e}")
            return None

    def fetch_waqi_data(self, city: str = None) -> Optional[Dict]:
        """
        Fetch data from World Air Quality Index (WAQI) API
        Provides: Comprehensive air quality data
        API: https://aqicn.org/api/
        """
        try:
            api_key = self.api_keys.get("waqi")
            if not api_key or api_key == "YOUR_WAQI_API_KEY":
                print("WAQI API key not configured")
                return None

            city = city or self.location.get("city", "london")

            url = f"https://api.waqi.info/feed/{city}/?token={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code == 200 and data["status"] == "ok":
                feed_data = data["data"]

                result = {
                    "timestamp": datetime.fromtimestamp(feed_data["time"]["v"]),
                    "location": feed_data["city"]["name"],
                    "AQI": float(feed_data["aqi"]),
                    "AQI_category": self._categorize_aqi(float(feed_data["aqi"])),
                    "api_source": "WAQI",
                }

                # Extract pollutants if available
                iaqi = feed_data.get("iaqi", {})

                pollutant_mapping = {
                    "co": "CO",
                    "no2": "NO2",
                    "o3": "O3",
                    "pm25": "PM25",
                    "pm10": "PM10",
                    "t": "temperature",
                    "h": "humidity",
                    "p": "pressure",
                    "w": "wind_speed",
                }

                for api_key, result_key in pollutant_mapping.items():
                    if api_key in iaqi:
                        result[result_key] = float(iaqi[api_key]["v"])

                # Estimate NOx if we have NO2
                if "NO2" in result:
                    result["NOx"] = result["NO2"] * 1.5  # Rough estimate

                print(f"Fetched WAQI data for {result['location']}")
                return result
            else:
                print(f"Error: WAQI API returned status {data.get('status')}")
                return None

        except Exception as e:
            print(f"Error fetching WAQI data: {e}")
            return None

    def fetch_combined_data(
        self, city: str = None, lat: float = None, lon: float = None, country: str = None
    ) -> Optional[Dict]:
        """
        Fetch and combine data from multiple APIs for better coverage
        Prioritizes: WAQI for AQI, OpenWeatherMap for weather
        """
        combined = {}

        # If city provided and no lat/lon, attempt geocoding
        if city and (lat is None or lon is None):
            geo = self.geocode_city(city, country)
            if geo:
                lat, lon = geo

        # Try WAQI first (best for AQI)
        waqi_data = self.fetch_waqi_data(city=city)
        if waqi_data:
            combined.update(waqi_data)

        # Get weather from OpenWeatherMap if needed
        owm_data = self.fetch_openweathermap_data(lat=lat, lon=lon)
        if owm_data:
            # Fill in missing values
            for key, value in owm_data.items():
                if key not in combined or combined[key] is None:
                    combined[key] = value

        if combined:
            combined["timestamp"] = datetime.now()
            combined["api_source"] = "Combined"
            return combined

        print("Warning: Could not fetch data from any API")
        return None

    def _categorize_aqi(self, aqi: float) -> str:
        """Categorize AQI value"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def generate_mock_data(self) -> Dict:
        """
        Generate mock data for testing when APIs are not available
        Based on typical patterns from historical data
        """
        import random

        # Simulate realistic variations
        base_values = {
            "CO": 2.0 + random.uniform(-0.5, 1.0),
            "NOx": 120 + random.uniform(-30, 50),
            "NO2": 85 + random.uniform(-20, 30),
            "O3": 45 + random.uniform(-10, 20),
            "PM25": 35 + random.uniform(-10, 25),
            "PM10": 50 + random.uniform(-15, 30),
            "temperature": 20 + random.uniform(-5, 10),
            "humidity": 60 + random.uniform(-15, 20),
            "pressure": 1013 + random.uniform(-10, 10),
            "wind_speed": 3 + random.uniform(0, 5),
            "wind_direction": random.uniform(0, 360),
        }

        # Calculate mock AQI
        aqi = max(
            base_values["CO"] / 40.0 * 100,
            base_values["NOx"] / 400.0 * 100,
            base_values["NO2"] / 200.0 * 100,
            base_values["PM25"] / 35.0 * 100,
            base_values["PM10"] / 50.0 * 100,
        )

        result = {
            "timestamp": datetime.now(),
            "location": self.location.get("city", "Mock City"),
            "AQI": min(aqi, 500),
            "AQI_category": self._categorize_aqi(aqi),
            "api_source": "Mock Data",
            **base_values,
        }

        print("Generated mock data (API not available)")
        return result


class ContinuousDataCollector:
    """Continuously collect data from APIs at regular intervals"""

    def __init__(self, fetcher: AirQualityAPIFetcher, database, interval_minutes: int = 60):
        """Initialize continuous collector"""
        self.fetcher = fetcher
        self.database = database
        self.interval_minutes = interval_minutes
        self.running = False

    def start_collection(self, use_mock: bool = False):
        """Start continuous data collection"""
        self.running = True
        print(f"\n{'='*60}")
        print("Starting Continuous Data Collection")
        print(f"Interval: {self.interval_minutes} minutes")
        print(f"{'='*60}\n")

        iteration = 0

        try:
            while self.running:
                iteration += 1
                print(f"\n[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Fetch data
                if use_mock:
                    data = self.fetcher.generate_mock_data()
                else:
                    data = self.fetcher.fetch_combined_data()

                    # Fallback to mock if API fails
                    if not data:
                        print("Falling back to mock data...")
                        data = self.fetcher.generate_mock_data()

                # Store in database
                if data:
                    self.database.insert_live_data(data)
                    print(f"✓ Stored data - AQI: {data['AQI']:.1f} ({data['AQI_category']})")

                # Wait for next interval
                print(f"Waiting {self.interval_minutes} minutes until next collection...")
                time.sleep(self.interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nStopping data collection...")
            self.running = False
        except Exception as e:
            print(f"\nError in collection loop: {e}")
            self.running = False

    def stop_collection(self):
        """Stop data collection"""
        self.running = False


def test_api_fetcher():
    """Test API fetcher functionality"""
    print("=" * 60)
    print("Testing API Data Fetcher")
    print("=" * 60)

    fetcher = AirQualityAPIFetcher()

    print("\n1. Testing OpenWeatherMap API...")
    owm_data = fetcher.fetch_openweathermap_data()
    if owm_data:
        print(json.dumps({k: str(v) for k, v in owm_data.items()}, indent=2))

    print("\n2. Testing WAQI API...")
    waqi_data = fetcher.fetch_waqi_data()
    if waqi_data:
        print(json.dumps({k: str(v) for k, v in waqi_data.items()}, indent=2))

    print("\n3. Testing Combined Data...")
    combined_data = fetcher.fetch_combined_data()
    if combined_data:
        print(json.dumps({k: str(v) for k, v in combined_data.items()}, indent=2))

    print("\n4. Testing Mock Data...")
    mock_data = fetcher.generate_mock_data()
    print(json.dumps({k: str(v) for k, v in mock_data.items()}, indent=2))

    print("\n" + "=" * 60)
    print("API Fetcher test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_api_fetcher()
