# Fixing "Using Cached Data" Issue - Production Deployment Guide

## Problem Summary

Your deployed app shows: **📦 Using cached data from 07:56 (database offline)**

This means the **MongoDB Atlas connection is failing**, not an API key issue.

---

## Root Cause Analysis

### Why This Happens
1. **App tries to fetch fresh data from APIs** → Works fine with API keys ✅
2. **App tries to save data to MongoDB** → Connection fails ❌
3. **App falls back to cached data** from session state → Old data shown 📦
4. **API keys exist but are unused** → Fresh data never reaches database

### Common Causes
| Issue | Solution |
|-------|----------|
| MongoDB cluster is **down** | Restart cluster in MongoDB Atlas |
| **IP whitelist** blocks your app | Add app IP to Network Access |
| **Wrong credentials** in URI | Verify username/password |
| **Missing database name** in URI | URI should end with `/` not `/database_name` |
| **Special chars not encoded** | Use URL encoding: `@` → `%40`, `#` → `%23` |

---

## Quick Diagnostics

### Step 1: Run Diagnostic Script
```bash
python scripts/diagnose_mongodb.py
```

This will check:
- ✓ Environment variables loaded
- ✓ DNS resolution (can reach MongoDB)
- ✓ Actual connection to MongoDB Atlas
- ✓ API keys configured

### Step 2: Check MongoDB Atlas Console
1. Go to https://cloud.mongodb.com/
2. Select your project and cluster
3. Check status indicator (should be green)
4. Click "Connect" → "Network Access"
5. Verify your IP is whitelisted (0.0.0.0/0 for testing)

---

## For Streamlit Cloud Deployment

### The Issue with Current Setup
Your `.env` file is **not deployed** to Streamlit Cloud. You need to use **Secrets**.

### How to Fix

#### Step 1: Add Secrets in Streamlit Cloud
1. Go to your Streamlit app: https://share.streamlit.io/
2. Click your app under "My Apps"
3. Click menu (⋮) → "Settings" → "Secrets"
4. Paste this content:

```toml
MONGODB_URI = "mongodb+srv://syed59750_db_user:zn3AOO1ZO9pSHBzj@cluster0.fyofuxr.mongodb.net/"
MONGODB_DATABASE = "pearl_aqi_db"
OPENWEATHER_API_KEY = "2139715b6a745bde80c2658c7904799e"
WAQI_API_KEY = ""
ENVIRONMENT = "production"
DEBUG = false
```

#### Step 2: Fix MongoDB Atlas IP Whitelist
1. Go to MongoDB Atlas: https://cloud.mongodb.com/
2. Select your cluster
3. Go to "Security" → "Network Access"
4. Click "Add IP Address"
5. For testing: Use `0.0.0.0/0` (allows all IPs)
6. For production: Get Streamlit's IP range and add it specifically

#### Step 3: Verify Connection String Format
```
✓ Correct: mongodb+srv://user:pass@cluster.mongodb.net/
✗ Wrong:   mongodb+srv://user:pass@cluster.mongodb.net/database_name
✗ Wrong:   mongodb+srv://user:pass@cluster.mongodb.net (missing trailing /)
```

#### Step 4: Redeploy
1. Push code to your GitHub repo
2. Streamlit will auto-redeploy
3. Check app logs for connection confirmation

---

## For Docker/Self-Hosted Deployment

### Option 1: Using .env file
```bash
# Make sure .env exists in root directory with:
MONGODB_URI=mongodb+srv://syed59750_db_user:zn3AOO1ZO9pSHBzj@cluster0.fyofuxr.mongodb.net/
MONGODB_DATABASE=pearl_aqi_db
OPENWEATHER_API_KEY=2139715b6a745bde80c2658c7904799e
```

### Option 2: Using Environment Variables
```bash
export MONGODB_URI="mongodb+srv://syed59750_db_user:zn3AOO1ZO9pSHBzj@cluster0.fyofuxr.mongodb.net/"
export MONGODB_DATABASE="pearl_aqi_db"
export OPENWEATHER_API_KEY="2139715b6a745bde80c2658c7904799e"

streamlit run frontend/dashboard_enhanced.py
```

### Option 3: Docker with Environment Variables
```bash
docker run -e MONGODB_URI="mongodb+srv://..." \
           -e MONGODB_DATABASE="pearl_aqi_db" \
           -e OPENWEATHER_API_KEY="..." \
           your-app-image
```

---

## Troubleshooting Connection String

### If Password Contains Special Characters

MongoDB will reject passwords with special chars. **URL encode them:**

| Character | Encoded |
|-----------|---------|
| @ | %40 |
| # | %23 |
| : | %3A |
| / | %2F |
| ? | %3F |
| & | %26 |

Example:
```
Password: my@pass#123
Encoded: my%40pass%23123

Connection:
mongodb+srv://user:my%40pass%23123@cluster.mongodb.net/
```

### If Getting "DNS Resolution Failed"
1. Check internet connection
2. Verify MongoDB URI domain spelling (should be `*.mongodb.net`)
3. Check firewall settings

### If Getting "Connection Timeout"
1. **IP Whitelist**: MongoDB Atlas might be blocking your IP
   - Go to Network Access
   - Add your IP or use `0.0.0.0/0` for testing
2. **Firewall**: Your network might block outbound port 27017
   - Contact your network admin

---

## How to Find Your Real Credentials

Go to MongoDB Atlas and copy the connection string:

1. https://cloud.mongodb.com/
2. Click your cluster → "Connect"
3. Choose "Connect your application"
4. Copy the connection string

It will look like:
```
mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/<dbname>?retryWrites=true&w=majority
```

- Replace `<username>` and `<password>` with your DB user credentials
- Remove `<dbname>` from the end (not needed for Atlas)
- Final format: `mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/`

---

## Verification Checklist

After applying fixes, verify each step:

- [ ] Run `python scripts/diagnose_mongodb.py` and all tests pass
- [ ] Can DNS resolve MongoDB hostname
- [ ] MongoDB connection test succeeds
- [ ] No "database offline" message in dashboard
- [ ] Fresh data appears (not showing old cache time)
- [ ] AQI value updates when you click "Refresh"

---

## Advanced: Testing Connection Manually

```python
from pymongo import MongoClient
import os

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri, serverSelectionTimeoutMS=3000)
client.admin.command("ping")  # This should succeed
print("✅ Connected successfully!")

# Try fetching data
db = client["pearl_aqi_db"]
count = db.live_data.count_documents({})
print(f"✅ Found {count} live data records")
```

---

## When to Call for Help

If after following this guide you still see "Using cached data", provide these details:

1. Output from: `python scripts/diagnose_mongodb.py`
2. Your MongoDB Atlas cluster status (green/red)
3. Your deployment platform (Streamlit Cloud/Docker/VPS)
4. Screenshot of Network Access IP whitelist in MongoDB Atlas
5. Recent app logs/errors

---

## Summary: What Changed?

| Before | After |
|--------|-------|
| Using old cached data | Fetching fresh data |
| Database offline ❌ | Database connected ✅ |
| API keys unused | API keys working |
| Error: 07:56 cache | Success: Real-time data |

The fix is **ensuring MongoDB connection works**, not changing API keys.
