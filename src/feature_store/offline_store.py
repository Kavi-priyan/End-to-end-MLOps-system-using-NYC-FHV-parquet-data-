import pandas as pd
from pathlib import Path

OFFLINE_PATH = Path("src/feature_store/offline_store/features.parquet")

def write_offline_features(df):
    OFFLINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OFFLINE_PATH)
    print("✅ Offline features written")

def read_offline_features():
    return pd.read_parquet(OFFLINE_PATH)
