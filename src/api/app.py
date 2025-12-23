from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ---------------------------
# App
# ---------------------------
app = FastAPI(
    title="NYC FHV Trip Duration Prediction API",
    version="1.0.0"
)

# ---------------------------
# Load trained pipeline
# ---------------------------
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# ---------------------------
# Input Schema (STRICT)
# ---------------------------
class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0)

    PUlocationID: int
    DOlocationID: int

    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_weekday: int = Field(..., ge=0, le=6)
    pickup_day: int = Field(..., ge=1, le=31)
    pickup_week_of_year: int = Field(..., ge=1, le=53)

    is_weekend: int = Field(..., ge=0, le=1)
    is_peak_hour: int = Field(..., ge=0, le=1)

    hour_zone: str

    # Rolling historical features (from feature engineering)
    rolling_avg_duration: float = Field(..., gt=0)
    zone_trip_count: int = Field(..., ge=0)

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict_trip_duration(payload: TripFeatures):
    """
    Predict trip duration (seconds) for an FHV ride.
    """

    # Convert request → DataFrame
    df = pd.DataFrame([payload.dict()])

    # Predict
    prediction = model.predict(df)[0]

    return {
        "predicted_trip_duration_sec": round(float(prediction), 2),
        "predicted_trip_duration_min": round(float(prediction) / 60, 2)
    }
