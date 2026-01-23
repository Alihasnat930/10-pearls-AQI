"""
Database connection and operations
Wrapper around MongoDB for FastAPI
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.core.database_main import AirQualityDatabase

# Singleton database instance
_db_instance = None


def get_database() -> AirQualityDatabase:
    """Get database instance (singleton pattern)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = AirQualityDatabase()
    return _db_instance
