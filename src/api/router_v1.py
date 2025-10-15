from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import joblib

router = APIRouter()

artifact = joblib.load("models/energy_model.joblib")
model = artifact["model"]
features = artifact["features"]

class PredictionRequest(BaseModel):
    features: dict

@router.get("/")
def root():
    return {"message": "API v1 is running successfully"}

@router.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame([request.features], columns=features)
    pred = model.predict(data)[0]
    return {"predicted_kW": round(float(pred), 3)}
