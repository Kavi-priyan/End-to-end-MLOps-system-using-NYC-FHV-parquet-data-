import pandas as pd
from pathlib import Path

MAX_DURATION=4* 3600 # 4 hours

ROLLING_WINDOW=100

def feature_engineering(input_path: Path, output_path: Path):

    if not input_path.exists():
        raise FileNotFoundError(f"Input data file not found at {input_path}. Please run the ingestion process first.")
    
    df = pd.read_parquet(input_path)

    df.dropna(subset=["pickup_datetime","dropOff_datetime"],inplace=True)

    df=df.sort_values("pickup_datetime")
    

    # Example feature engineering: create trip duration in minutes
    df["pickup_datetime"] = pd.to_datetime(df['pickup_datetime'])
    df["dropOff_datetime"] = pd.to_datetime(df['dropOff_datetime'])
    df["trip_duration_minutes"] = (df["dropOff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0

    df["pickup_week_of_year"]=df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["pickup_hour"]=df["pickup_datetime"].dt.hour.astype(int)

    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_day"] = df["pickup_datetime"].dt.day

    df["hour_zone"] = (
        df["pickup_hour"].astype(str) + "_" + df["PUlocationID"].astype(str)
    )

    
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    df["zone_trip_count"] = (
        df.groupby("PUlocationID").cumcount()
    )
    
    # Handle negative or zero durations
    df = df[(df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] <= (MAX_DURATION / 60.0))]


    df['rolling_avg_duration']=(df.groupby('PUlocationID')['trip_duration_minutes'].transform(
        lambda x:x.rolling(window=ROLLING_WINDOW,min_periods=1).mean(
    ) ))

    leakage_columns=[
        'dropOff_datetime',
        'feedback',
        'payment_type',
        'fare_amount',
    ]


    leaks=[c for c in leakage_columns if c in df.columns
           
           ]
    
    df=df.drop(columns=leaks)


    df.dropna(subset=['rolling_avg_duration'],inplace=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print("✅ Feature engineering completed:", df.shape)


if __name__=="__main__":
    INPUT_PATH = Path("data/raw/fhv_2024_12.parquet")
    OUTPUT_PATH = Path("data/processed/fhv_2024_12_features.parquet")
    feature_engineering(INPUT_PATH, OUTPUT_PATH)