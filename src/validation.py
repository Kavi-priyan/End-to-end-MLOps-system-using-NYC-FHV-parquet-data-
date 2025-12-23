import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/fhv_2024_12.parquet")

def validate_data():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {RAW_PATH}. Please run the ingestion process first.")
    
    df = pd.read_parquet(RAW_PATH)
    
    # Example validation checks
    if df.empty:
        raise ValueError("The dataset is empty.")
    
    expected_columns = {"dispatching_base_num", "pickup_datetime", "dropOff_datetime", "PUlocationID", "DOlocationID"}


    if not expected_columns.issubset(set(df.columns)):
        missing_cols = expected_columns - set(df.columns)
        raise ValueError(f"The dataset is missing expected columns: {missing_cols}")
    
    if df['pickup_datetime'].isnull().any() or df['dropOff_datetime'].isnull().any():
        raise ValueError("There are null values in datetime columns.")
    
    print("✅ Data validation passed:", df.shape)



if __name__ == "__main__":
    validate_data()