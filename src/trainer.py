import pandas as pd
from pathlib import Path
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


import mlflow
import mlflow.sklearn

DATA_PATH= Path("data/processed/fhv_2024_12_features.parquet")
MODEL_PATH=Path("models/model.pkl")


DATE="2024-12-24"
TARGET="trip_duration_minutes"

def train():
    df=pd.read_parquet(DATA_PATH)

   

    y=df[TARGET]
    X=df.drop(columns=[TARGET])

    X_test=X[df["pickup_datetime"]>=DATE]
    Y_test=y[df["pickup_datetime"]>=DATE]

    X_train=X[df["pickup_datetime"]<DATE]
    Y_train=y[df["pickup_datetime"]<DATE]

    categorical_features=["pickup_week_of_year","pickup_hour","pickup_weekday","pickup_day","is_peak_hour","is_weekend","PUlocationID","DOlocationID","hour_zone"]

    numerical_features=["rolling_avg_duration","zone_trip_count"]


    preprocessor=ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),numerical_features),
            ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)

        ]
        )
    
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    pipeline=Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("model",model)
        ]
    )

    pipeline.fit(X_train,Y_train)

    preds=pipeline.predict(X_test)


    mlflow.set_experiment("fhv_trip_duration")
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("learning_rate", 0.05)

        # Train
        pipeline.fit(X_train, Y_train)

        # Evaluate
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(Y_test, preds)
        rmse = root_mean_squared_error(Y_test, preds)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("✅ Training complete")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Save locally too
        joblib.dump(pipeline, MODEL_PATH)
        print("✅ Model saved locally:", MODEL_PATH)

if __name__=="__main__":
    train()