import pandas as pd
from pathlib import Path

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2024-12.parquet"

RAW_PATH = Path("data/raw/fhv_2024_12.parquet")

def ingest():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(URL)
    df.to_parquet(RAW_PATH)
    print("✅ Data ingested:", df.shape)

if __name__ == "__main__":
    ingest()
