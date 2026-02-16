"""
MongoDB Connection Diagnostic Tool
Helps identify and fix database connectivity issues
"""

import os
import sys
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment
load_dotenv()

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_env_file():
    """Check .env file configuration"""
    print_section("1. Environment Configuration Check")
    
    uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DATABASE", "")
    
    print(f"✓ MONGODB_URI present: {bool(uri)}")
    print(f"✓ MONGODB_DATABASE present: {bool(db_name)}")
    
    if uri:
        # Parse URI
        if "mongodb+srv://" in uri:
            print("✓ Using MongoDB Atlas (SRV connection)")
        elif "mongodb://" in uri:
            print("✓ Using standard MongoDB connection")
        else:
            print("✗ Invalid URI format")
            return False
            
        # Check for credentials
        if "@" in uri:
            cred_part = uri.split("@")[0].replace("mongodb+srv://", "").replace("mongodb://", "")
            if ":" in cred_part:
                username = cred_part.split(":")[0]
                print(f"✓ Username configured: {username}")
            else:
                print("✗ No password in connection string")
        else:
            print("⚠ No authentication credentials found")
    
    return bool(uri and db_name)

def check_hostname_resolution():
    """Check if MongoDB host is reachable"""
    print_section("2. Hostname Resolution Check")
    
    uri = os.getenv("MONGODB_URI", "")
    
    if "mongodb+srv://" in uri:
        # Extract hostname from SRV
        parts = uri.replace("mongodb+srv://", "").split("@")[1].split("/")[0]
        hostname = parts.split("?")[0]
        print(f"🔍 MongoDB host: {hostname}")
        
        try:
            import socket
            ip_address = socket.gethostbyname(hostname)
            print(f"✓ DNS resolution successful: {hostname} → {ip_address}")
            return True
        except socket.gaierror as e:
            print(f"✗ DNS resolution failed: {e}")
            print("  → Check internet connection or MongoDB URI")
            return False
    
    return True

def check_mongodb_connection():
    """Test actual MongoDB connection"""
    print_section("3. MongoDB Connection Test")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ConfigurationError
        
        uri = os.getenv("MONGODB_URI", "")
        if not uri:
            print("✗ No MONGODB_URI configured")
            return False
        
        print("🔄 Attempting connection (3 second timeout)...")
        
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
        )
        
        # Test ping
        client.admin.command("ping")
        print("✓ Successfully connected to MongoDB Atlas")
        
        # Check database
        db_name = os.getenv("MONGODB_DATABASE", "pearl_aqi_db")
        db = client[db_name]
        print(f"✓ Database accessible: {db_name}")
        
        # List collections
        collections = db.list_collection_names()
        print(f"✓ Collections found: {len(collections)}")
        if collections:
            for col in collections:
                count = db[col].count_documents({})
                print(f"   - {col}: {count} documents")
        
        client.close()
        return True
        
    except ConnectionFailure as e:
        print(f"✗ Connection timeout: {e}")
        print("\n  🔧 Possible solutions:")
        print("    1. Check MongoDB Atlas cluster status")
        print("    2. Verify IP whitelist in MongoDB Atlas")
        print("    3. Check internet connectivity")
        print("    4. Verify credentials are correct")
        return False
        
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        print("\n  🔧 Possible solutions:")
        print("    1. Fix MONGODB_URI format")
        print("    2. Check for special characters in password")
        print("    3. Ensure username/password are correct")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_api_keys():
    """Check API key configuration"""
    print_section("4. API Keys Check")
    
    openweather_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    waqi_key = os.getenv("WAQI_API_KEY", "").strip()
    
    print(f"✓ OPENWEATHER_API_KEY: {'Configured' if openweather_key else 'NOT SET'}")
    if openweather_key:
        print(f"   Length: {len(openweather_key)} chars")
    
    print(f"✓ WAQI_API_KEY: {'Configured' if waqi_key else 'NOT SET'}")
    if waqi_key:
        print(f"   Length: {len(waqi_key)} chars")
    
    return bool(openweather_key)

def print_recommendations():
    """Print recommendations for Streamlit deployment"""
    print_section("5. Streamlit Deployment Setup")
    
    print("""
For Streamlit Cloud deployment, create a .streamlit/secrets.toml file:

[secrets]
MONGODB_URI = "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/"
MONGODB_DATABASE = "pearl_aqi_db"
OPENWEATHER_API_KEY = "your_api_key"
WAQI_API_KEY = "your_waqi_key"

Steps to fix deployed app:
1. Go to your Streamlit Cloud app settings
2. Click "Secrets" in the left sidebar
3. Paste the secrets content in the text area
4. Deploy will use these environment variables

Common MongoDB Atlas Issues:
- IP Whitelist: Add your Streamlit IP to MongoDB Atlas security settings
  → Use 0.0.0.0/0 (allow all IPs) for testing, restrict later
- Network Access: Ensure app environment can reach MongoDB
- Credentials: Double-check username and password encode special chars as %XX
""")

def print_mongodb_setup():
    """Print MongoDB Atlas setup guide"""
    print_section("6. MongoDB Atlas Setup Verification")
    
    print("""
Verify these in MongoDB Atlas:
1. Cluster Status: Ensure cluster is running (green status)
2. IP Whitelist: Add your application server IP
   - For Streamlit Cloud: Use 0.0.0.0/0 or ask Streamlit for static IP
   - For local development: Add your local IP
3. Database User: Verify credentials exist
   - Go to Database → Users
   - Confirm username and password are correct
4. Connection String: Copy from MongoDB Atlas
   - Click "Connect" button on cluster
   - Choose "Connect your application"
   - Copy the connection string carefully

Connection string troubleshooting:
- Ensure password is URL-encoded (@ → %40, # → %23, etc.)
- Don't include the <database> placeholder
- For SRV: mongodb+srv://user:pass@cluster.mongodb.net/
""")

def main():
    """Run all diagnostics"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Pearl AQI - MongoDB Connection Diagnostic Tool".center(68) + "║")
    print("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Run checks
    env_ok = check_env_file()
    dns_ok = check_hostname_resolution()
    conn_ok = check_mongodb_connection()
    api_ok = check_api_keys()
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    checks = {
        "Environment Configuration": env_ok,
        "DNS Resolution": dns_ok,
        "MongoDB Connection": conn_ok,
        "API Keys": api_ok,
    }
    
    for check, status in checks.items():
        icon = "✓" if status else "✗"
        print(f"{icon} {check}")
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("\n✅ All tests passed! Your app should work.")
    else:
        print("\n⚠️  Some checks failed. See details above.")
    
    # Print guides
    print_recommendations()
    print_mongodb_setup()
    
    # Exit code
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
