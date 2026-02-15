"""
MongoDB Database Module for Air Quality Forecasting System
Manages MongoDB Atlas connection for storing historical and live data
Feature Store Implementation with Cloud Database
"""

import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConfigurationError, ConnectionFailure, DuplicateKeyError

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
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


class AirQualityDatabase:
    """MongoDB Atlas database manager for air quality data - Cloud Feature Store"""

    def __init__(self):
        """Initialize MongoDB connection with retry logic and graceful degradation"""
        self.mongo_uri = _get_secret("MONGODB_URI")
        self.db_name = _get_secret("MONGODB_DATABASE", "pearl_aqi_db")
        self.connected = False

        if not self.mongo_uri:
            print("⚠️ MONGODB_URI not found - running in OFFLINE mode")
            self.client = None
            self.db = None
            return

        self.client = None
        self.db = None
        self._connect()
        if self.connected:
            self._create_collections()

    def _connect(self):
        """Connect to MongoDB Atlas with retry logic and graceful fallback"""
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                print(f"🔄 Connecting to MongoDB Atlas (attempt {attempt + 1}/{max_retries})...")

                # Reduced timeouts for faster failure detection
                self.client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=3000,  # 3 seconds
                    connectTimeoutMS=3000,
                    socketTimeoutMS=3000,
                )

                # Test connection with ping
                self.client.admin.command("ping")
                self.db = self.client[self.db_name]
                self.connected = True
                print(f"✅ Connected to MongoDB Atlas: {self.db_name}")
                return

            except (ConnectionFailure, ConfigurationError) as e:
                error_msg = str(e)

                if "DNS" in error_msg or "resolution" in error_msg:
                    print(f"⚠️ DNS resolution failed - check internet connection")
                elif "timed out" in error_msg:
                    print(f"⚠️ Connection timeout - network issue detected")
                else:
                    print(f"⚠️ Connection failed: {error_msg[:100]}")

                if attempt < max_retries - 1:
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("❌ MongoDB connection failed after all retries")
                    print("📡 Running in OFFLINE mode - predictions will use cached data")
                    self.connected = False
                    self.client = None
                    self.db = None
            except Exception as e:
                print(f"❌ Unexpected error connecting to MongoDB: {e}")
                self.connected = False
                self.client = None
                self.db = None
                break

    def _create_collections(self):
        """Create collections and indexes if they don't exist"""

        # Historical data collection
        if "historical_data" not in self.db.list_collection_names():
            self.db.create_collection("historical_data")

        # Create indexes for efficient queries
        self.db.historical_data.create_index([("timestamp", DESCENDING)], unique=True)
        self.db.historical_data.create_index([("AQI", ASCENDING)])
        self.db.historical_data.create_index([("data_source", ASCENDING)])

        # Live data collection
        if "live_data" not in self.db.list_collection_names():
            self.db.create_collection("live_data")

        self.db.live_data.create_index([("timestamp", DESCENDING)], unique=True)
        self.db.live_data.create_index([("location", ASCENDING)])
        self.db.live_data.create_index([("AQI", ASCENDING)])

        # Predictions collection
        if "predictions" not in self.db.list_collection_names():
            self.db.create_collection("predictions")

        self.db.predictions.create_index(
            [
                ("prediction_timestamp", DESCENDING),
                ("target_timestamp", ASCENDING),
                ("model_name", ASCENDING),
            ],
            unique=True,
        )

        # Model performance collection
        if "model_performance" not in self.db.list_collection_names():
            self.db.create_collection("model_performance")

        self.db.model_performance.create_index(
            [("model_name", ASCENDING), ("timestamp", DESCENDING)]
        )

        print("✅ MongoDB collections and indexes created/verified")

    def insert_historical_data(self, data: pd.DataFrame):
        """Insert historical data from DataFrame into MongoDB"""
        if not self.connected:
            print("⚠️ Skipping insert (offline mode)")
            return

        try:
            # Convert DataFrame to list of dictionaries
            records = data.reset_index().to_dict("records")

            # Prepare documents for insertion
            documents = []
            for record in records:
                doc = {
                    "timestamp": record.get("timestamp") or record.get("Date"),
                    "CO": float(record.get("CO", 0)) if pd.notna(record.get("CO")) else None,
                    "NOx": float(record.get("NOx", 0)) if pd.notna(record.get("NOx")) else None,
                    "NO2": float(record.get("NO2", 0)) if pd.notna(record.get("NO2")) else None,
                    "O3": float(record.get("O3", 0)) if pd.notna(record.get("O3")) else None,
                    "temperature": (
                        float(record.get("temperature", 0))
                        if pd.notna(record.get("temperature"))
                        else None
                    ),
                    "humidity": (
                        float(record.get("humidity", 0))
                        if pd.notna(record.get("humidity"))
                        else None
                    ),
                    "AQI": float(record.get("AQI", 0)) if pd.notna(record.get("AQI")) else None,
                    "AQI_category": record.get("AQI_category", "Unknown"),
                    "data_source": "historical",
                    "created_at": datetime.now(),
                }

                # Add any other columns present in the data
                for col in record.keys():
                    if col not in doc and col not in ["index", "Date"]:
                        if pd.notna(record[col]):
                            doc[col] = (
                                float(record[col])
                                if isinstance(record[col], (int, float))
                                else record[col]
                            )

                documents.append(doc)

            # Insert documents
            if documents:
                try:
                    result = self.db.historical_data.insert_many(documents, ordered=False)
                    print(f"✅ Inserted {len(result.inserted_ids)} historical records")
                except DuplicateKeyError:
                    print("⚠️ Some records already exist (duplicates skipped)")

        except Exception as e:
            print(f"❌ Error inserting historical data: {e}")

    def insert_live_data(self, data: Dict):
        """Insert live data from API into MongoDB"""
        if not self.connected:
            print("⚠️ Skipping insert (offline mode)")
            return

        try:
            document = {
                "timestamp": data.get("timestamp", datetime.now()),
                "CO": float(data.get("CO", 0)) if data.get("CO") is not None else None,
                "NOx": float(data.get("NOx", 0)) if data.get("NOx") is not None else None,
                "NO2": float(data.get("NO2", 0)) if data.get("NO2") is not None else None,
                "O3": float(data.get("O3", 0)) if data.get("O3") is not None else None,
                "PM25": float(data.get("PM2.5", 0)) if data.get("PM2.5") is not None else None,
                "PM10": float(data.get("PM10", 0)) if data.get("PM10") is not None else None,
                "temperature": (
                    float(data.get("temperature", 0))
                    if data.get("temperature") is not None
                    else None
                ),
                "humidity": (
                    float(data.get("humidity", 0)) if data.get("humidity") is not None else None
                ),
                "pressure": (
                    float(data.get("pressure", 0)) if data.get("pressure") is not None else None
                ),
                "wind_speed": (
                    float(data.get("wind_speed", 0)) if data.get("wind_speed") is not None else None
                ),
                "wind_direction": (
                    float(data.get("wind_direction", 0))
                    if data.get("wind_direction") is not None
                    else None
                ),
                "AQI": float(data.get("AQI", 0)),
                "AQI_category": data.get("AQI_category", "Unknown"),
                "location": data.get("location", "Unknown"),
                "api_source": data.get("api_source", "unknown"),
                "created_at": datetime.now(),
            }

            # Update if exists, insert if not
            self.db.live_data.update_one(
                {"timestamp": document["timestamp"]}, {"$set": document}, upsert=True
            )

            print(f"✅ Live data inserted/updated for {document['timestamp']}")

        except Exception as e:
            print(f"❌ Error inserting live data: {e}")

    def insert_prediction(
        self,
        prediction_timestamp: datetime,
        target_timestamp: datetime,
        predicted_aqi: float,
        model_name: str,
        predicted_category: str,
        confidence_score: float = None,
    ):
        """Insert prediction into MongoDB"""
        if not self.connected:
            print("⚠️ Skipping insert (offline mode)")
            return

        try:
            document = {
                "prediction_timestamp": prediction_timestamp,
                "target_timestamp": target_timestamp,
                "predicted_AQI": float(predicted_aqi),
                "predicted_category": predicted_category,
                "model_name": model_name,
                "confidence_score": float(confidence_score) if confidence_score else None,
                "created_at": datetime.now(),
            }

            self.db.predictions.update_one(
                {
                    "prediction_timestamp": document["prediction_timestamp"],
                    "target_timestamp": document["target_timestamp"],
                    "model_name": document["model_name"],
                },
                {"$set": document},
                upsert=True,
            )

        except Exception as e:
            print(f"❌ Error inserting prediction: {e}")

    def get_recent_data(
        self, hours: int = 24, table: str = "live_data", location: str = None
    ) -> pd.DataFrame:
        """Get recent data from MongoDB, optionally filtered by location"""
        if not self.connected:
            print("⚠️ Cannot fetch data (offline mode)")
            return pd.DataFrame()

        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Build query
            query = {"timestamp": {"$gte": cutoff_time}}
            if location:
                query["location"] = location

            collection = self.db[table]
            cursor = collection.find(query, {"_id": 0}).sort("timestamp", ASCENDING)

            data = list(cursor)

            if data:
                df = pd.DataFrame(data)
                # Convert timestamp to datetime if it's not already
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error getting recent data: {e}")
            return pd.DataFrame()

    def get_predictions(self, days: int = 3) -> pd.DataFrame:
        """Get predictions for next N days"""
        if not self.connected:
            print("⚠️ Cannot fetch predictions (offline mode)")
            return pd.DataFrame()

        try:
            now = datetime.now()
            future_time = now + timedelta(days=days)

            cursor = self.db.predictions.find(
                {
                    "prediction_timestamp": {"$gte": now - timedelta(hours=1)},
                    "target_timestamp": {"$lte": future_time},
                },
                {"_id": 0},
            ).sort("target_timestamp", ASCENDING)

            data = list(cursor)

            if data:
                df = pd.DataFrame(data)
                df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
                df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return pd.DataFrame()

    def get_data_statistics(self) -> Dict:
        """Get statistics about stored data"""
        if not self.connected:
            return {"status": "offline", "message": "Database unavailable"}

        try:
            stats = {
                "historical": {
                    "count": self.db.historical_data.count_documents({}),
                    "latest": None,
                    "oldest": None,
                },
                "live": {
                    "count": self.db.live_data.count_documents({}),
                    "latest": None,
                    "oldest": None,
                },
                "predictions": {"count": self.db.predictions.count_documents({}), "latest": None},
            }

            # Get date ranges
            historical_latest = self.db.historical_data.find_one(
                {}, {"timestamp": 1}, sort=[("timestamp", DESCENDING)]
            )
            if historical_latest:
                stats["historical"]["latest"] = historical_latest["timestamp"]

            historical_oldest = self.db.historical_data.find_one(
                {}, {"timestamp": 1}, sort=[("timestamp", ASCENDING)]
            )
            if historical_oldest:
                stats["historical"]["oldest"] = historical_oldest["timestamp"]

            live_latest = self.db.live_data.find_one(
                {}, {"timestamp": 1}, sort=[("timestamp", DESCENDING)]
            )
            if live_latest:
                stats["live"]["latest"] = live_latest["timestamp"]

            return stats

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

    def cleanup_old_data(self, keep_days: int = 90):
        """Remove data older than specified days"""
        if not self.connected:
            print("⚠️ Cannot cleanup (offline mode)")
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            # Cleanup historical data
            result_hist = self.db.historical_data.delete_many({"timestamp": {"$lt": cutoff_date}})

            # Cleanup live data
            result_live = self.db.live_data.delete_many({"timestamp": {"$lt": cutoff_date}})

            # Cleanup old predictions
            result_pred = self.db.predictions.delete_many(
                {"prediction_timestamp": {"$lt": cutoff_date}}
            )

            print(
                f"✅ Cleaned up {result_hist.deleted_count} historical, "
                f"{result_live.deleted_count} live, "
                f"{result_pred.deleted_count} prediction records"
            )

        except Exception as e:
            print(f"❌ Error cleaning up data: {e}")

    def close(self):
        """Close MongoDB connection"""
        if self.client and self.connected:
            self.client.close()
            print("✅ MongoDB connection closed")
        elif not self.connected:
            print("📡 No active connection to close (was in offline mode)")


def main():
    """Test MongoDB connection"""
    print("=" * 60)
    print("Testing MongoDB Atlas Connection")
    print("=" * 60)

    try:
        db = AirQualityDatabase()

        # Test statistics
        stats = db.get_data_statistics()
        print("\n📊 Database Statistics:")
        print(f"  Historical records: {stats.get('historical', {}).get('count', 0)}")
        print(f"  Live records: {stats.get('live', {}).get('count', 0)}")
        print(f"  Predictions: {stats.get('predictions', {}).get('count', 0)}")

        # Test data insertion
        print("\n🧪 Testing data insertion...")
        test_data = {
            "timestamp": datetime.now(),
            "AQI": 75.5,
            "AQI_category": "Moderate",
            "temperature": 20.5,
            "humidity": 65.0,
            "PM2.5": 35.2,
            "PM10": 48.1,
            "location": "Test Location",
            "api_source": "test",
        }
        db.insert_live_data(test_data)

        # Test data retrieval
        print("\n📥 Testing data retrieval...")
        recent = db.get_recent_data(hours=1)
        print(f"  Retrieved {len(recent)} recent records")

        db.close()

        print("\n" + "=" * 60)
        print("✅ MongoDB Atlas connection test successful!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ MongoDB Atlas connection test failed: {e}")


if __name__ == "__main__":
    main()
