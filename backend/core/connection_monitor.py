"""
Database Connection Monitor for Dashboard
Provides real-time feedback about database connectivity
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st

class DatabaseConnectionMonitor:
    """Monitor and report database connection status"""
    
    def __init__(self):
        self.last_check = None
        self.last_status = None
        self.connection_error = None
    
    @staticmethod
    def test_connection(db) -> Dict:
        """
        Test database connection and return detailed status
        
        Returns:
            Dict with status, error details, and stats
        """
        try:
            if not db.connected:
                # Try to get connection status details
                stats = db.get_data_statistics()
                return {
                    "connected": False,
                    "error": "Database offline - check MongoDB Atlas",
                    "reason": "Connection initialization failed",
                    "stats": stats,
                }
            
            # Test actual query
            stats = db.get_data_statistics()
            
            if "error" in str(stats).lower() or "offline" in str(stats).lower():
                return {
                    "connected": False,
                    "error": "Database query failed",
                    "reason": "Cannot retrieve data from MongoDB",
                    "stats": stats,
                }
            
            return {
                "connected": True,
                "error": None,
                "stats": stats,
                "timestamp": datetime.now(),
            }
        
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "reason": "Exception during connection test",
            }
    
    @staticmethod
    def display_connection_status(db, location: str = ""):
        """
        Display connection status with actionable messages
        Should be called early in the dashboard before trying to fetch data
        """
        status = DatabaseConnectionMonitor.test_connection(db)
        
        if status["connected"]:
            st.success("✅ Database Connection: ACTIVE", icon="✓")
            return True
        else:
            # Show error with troubleshooting steps
            with st.container(border=True):
                st.error(f"⚠️ Database Connection: OFFLINE", icon="✗")
                
                error_box = st.empty()
                with error_box.container():
                    st.markdown("""
### Troubleshooting Database Connection
                    
**Why is this happening?**
- MongoDB Atlas cluster is unreachable
- Connection credentials are invalid
- Network firewall/IP whitelist blocking access
- MongoDB Atlas cluster status is down

**Quick fixes to try:**
1. **Check MongoDB Atlas Status**
   - Go to https://cloud.mongodb.com/
   - Verify your cluster is running (green status)
   - Check Network Access → IP Whitelist
   
2. **For Streamlit Cloud deployment:**
   - Go to your Streamlit app settings
   - Click "Secrets" in the sidebar
   - Verify MONGODB_URI is correct (no typos)
   - Make sure IP whitelist allows 0.0.0.0/0 or Streamlit's IP range
   
3. **For local development:**
   - Run: `python scripts/diagnose_mongodb.py`
   - Check your internet connection
   - Verify .env file has correct credentials

4. **Connection String Format:**
   ```
   mongodb+srv://username:password@cluster.mongodb.net/
   ```
   - Special characters in password? URL encode them (@→%40, #→%23)
   - Don't include `<database>` placeholder
                    """)
                
                # Show what we tried to connect to
                with st.expander("📋 Connection Details"):
                    uri = os.getenv("MONGODB_URI", "NOT SET")
                    db_name = os.getenv("MONGODB_DATABASE", "NOT SET")
                    
                    # Hide sensitive parts
                    if uri and len(uri) > 0:
                        # Show masked URI
                        if "@" in uri:
                            before_at = uri.split("@")[0].replace("mongodb+srv://", "")
                            username = before_at.split(":")[0] if ":" in before_at else "N/A"
                            masked = f"mongodb+srv://{username}:***@..." 
                        else:
                            masked = uri[:30] + "..." if len(uri) > 30 else uri
                        
                        st.text(f"Connection URI: {masked}")
                    st.text(f"Database: {db_name}")
                    
                    if hasattr(status, 'get') and status.get("error"):
                        st.error(f"Error: {status.get('error')}")
            
            # Try to show cached data info
            if location and location in st.session_state.get("city_data_cache", {}):
                cached = st.session_state.city_data_cache[location]
                cache_age = (datetime.now() - cached['timestamp']).total_seconds() / 60
                
                with st.info(icon="📦"):
                    st.markdown(f"""
### Using Cached Data
**Last updated:** {cached['timestamp'].strftime("%H:%M:%S")} ({cache_age:.0f} minutes ago)
**Status:** Database is offline, showing most recent cached values
                    """)
            
            return False

    @staticmethod
    def get_connection_health(db) -> str:
        """Get simple health indicator: 'healthy', 'degraded', or 'offline'"""
        try:
            if not db.connected:
                return "offline"
            
            stats = db.get_data_statistics()
            
            if "error" in str(stats).lower() or "offline" in str(stats).lower():
                return "offline"
            
            # Check if we have recent data
            if stats.get("live", {}).get("latest"):
                from datetime import timedelta
                latest = stats["live"]["latest"]
                age_hours = (datetime.now() - latest).total_seconds() / 3600
                
                if age_hours < 24:
                    return "healthy"
                else:
                    return "degraded"
            
            return "degraded"
        
        except Exception:
            return "offline"


def auto_fetch_missing_data(api_fetcher, db, location: str, max_retries: int = 3):
    """
    Attempt to fetch missing data from API if database is not available
    
    Args:
        api_fetcher: AirQualityAPIFetcher instance
        db: Database instance
        location: City name
        max_retries: Number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                st.info(f"🔄 Retrying data fetch (attempt {attempt + 1}/{max_retries})...")
                time.sleep(2)  # Wait before retry
            
            # Try each API in sequence
            data = api_fetcher.fetch_combined_data()
            
            if data:
                st.success(f"✅ Data fetched from API: AQI {data.get('AQI', 0):.0f}")
                
                # Try to save to database for next time
                try:
                    db.insert_live_data(data)
                except Exception:
                    pass  # Silent fail - database is offline anyway
                
                return data
        
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ Could not fetch live data: {str(e)[:100]}")
            continue
    
    return None
