import sqlite3
import pandas as pd
from pathlib import Path

# 1. Setup absolute paths correctly
# This gets the directory where online_store.py lives
CURRENT_DIR = Path(__file__).resolve().parent 

# This creates 'data/online_features.db' inside 'src/feature_store/'
DB_PATH = CURRENT_DIR / "data" / "online_features.db"

import os

def init__db():
    # Force absolute path and ensure folder is created
    db_dir = DB_PATH.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Path to a raw string for Windows
    db_string = os.path.abspath(str(DB_PATH))
    
    print(f"Checking directory permissions for: {db_dir}")
    if not os.access(db_dir, os.W_OK):
        print(f"❌ CRITICAL: Directory {db_dir} is NOT writable. Try running as Admin.")
        return

    try:
        # Using 'uri=True' can sometimes bypass Windows path locks
        conn = sqlite3.connect(db_string)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_features(
                PUlocationID INTEGER PRIMARY KEY,
                rolling_avg_duration REAL,
                zone_trip_count INTEGER
            )
        """)
        conn.commit()
        conn.close()
        print(f"✅ SUCCESS: Database created at {db_string}")
    except sqlite3.OperationalError as e:
        print(f"❌ SQLITE ERROR: {e}")
        print("Tip: Check if 'online_features.db' is open in a DB Browser or another terminal.")



        

def write_online_features(df):
    # Ensure directory exists before writing
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    df[["PUlocationID", "rolling_avg_duration", "zone_trip_count"]].drop_duplicates("PUlocationID").to_sql(
        "zone_features", conn, if_exists="replace", index=False
    )
    conn.close()
    print("✅ Online features written to disk")

def read_online_features(PUlocationID: int) -> dict:
    if not DB_PATH.exists():
        return {"zone_trip_count": 0, "rolling_avg_duration": 15.0}

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT zone_trip_count, rolling_avg_duration FROM zone_features WHERE PUlocationID = ?",
        (int(PUlocationID),)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"zone_trip_count": 0, "rolling_avg_duration": 15.0}

    return {
        "zone_trip_count": row[0],
        "rolling_avg_duration": row[1]
    }

if __name__ == "__main__":
    init__db()