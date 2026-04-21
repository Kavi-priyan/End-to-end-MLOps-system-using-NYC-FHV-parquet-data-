from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

from src.feature_store.online_store import read_online_features


app = FastAPI(title="NYC FHV Trip Duration Predictor")

templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/templates/static"), name="static")

model = joblib.load("models/model.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ---------------------------
# Prediction Form Page
# ---------------------------
@app.get("/predict-ui")
def predict_ui(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {"request": request}
    )

# ---------------------------
# Handle Prediction
# ---------------------------
@app.post("/predict-ui")
def predict_result(
    request: Request,
    PUlocationID: int = Form(...),
    DOlocationID: int = Form(...),
    pickup_hour: int = Form(...),
    pickup_weekday: int = Form(...),
    pickup_day: int = Form(...),
    pickup_week_of_year: int = Form(...),
    is_weekend: int = Form(...),
    is_peak_hour: int = Form(...)
):
    hour_zone = f"{pickup_hour}_{PUlocationID}"

    zone_features = read_online_features(PUlocationID)

   

    data = {
        
        "PUlocationID": PUlocationID,
        "DOlocationID": DOlocationID,
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "pickup_day": pickup_day,
        "pickup_week_of_year": pickup_week_of_year,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "hour_zone": hour_zone,
        **zone_features
      
    }

    

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "seconds": round(prediction * 60, 2),
            "minutes": round(prediction, 2)
        }
    )
