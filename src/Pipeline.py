import pandas as pd
from pathlib import Path

from ingest import ingest
from validation import validate_data
from features import feature_engineering
from trainer import train

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def run_pipeline():
  

    ingest()
    validate_data()
    feature_engineering(
        input_path=Path("data/raw/fhv_2024_12.parquet"),
        output_path=Path("data/processed/fhv_2024_12_features.parquet")
    )
    train()


if __name__=="__main__":
    run_pipeline()