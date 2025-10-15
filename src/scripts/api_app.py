from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

app = FastAPI(title="Cairo Smart Energy Optimizer API")

DATA_PATH = "data/processed/energy_features.csv"
MODEL_PATH = "models/energy_model.pkl"
METRICS_LOG = "data/metrics/training_log.csv"


os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_LOG), exist_ok=True)


class EnergyInput(BaseModel):
    global_active_power: float
    global_reactive_power: float
    voltage: float
    global_intensity: float
    sub_metering_1: float
    sub_metering_2: float
    sub_metering_3: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    global_active_power_lag_1: float
    global_active_power_lag_24: float
    global_active_power_roll_6h: float



def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found. Please train it first.")
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Cairo Smart Energy Optimizer API is running"}


@app.post("/predict")
def predict(data: EnergyInput):
    model = load_model()
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_energy_kW": round(float(prediction), 4)}



@app.post("/train")
def train_model():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

   
    df = df.select_dtypes(include=[np.number])

  
    possible_targets = ["energy_consumed_kW", "global_active_power", "predicted_kW"]
    target_col = next((col for col in possible_targets if col in df.columns), None)

    if not target_col:
        raise HTTPException(status_code=400, detail="No valid target column found for training.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, MODEL_PATH)

    log_entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MAE": mae,
        "R2": r2
    }
    if os.path.exists(METRICS_LOG):
        existing = pd.read_csv(METRICS_LOG)
        updated = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
        updated.to_csv(METRICS_LOG, index=False)
    else:
        pd.DataFrame([log_entry]).to_csv(METRICS_LOG, index=False)

    return {
        "message": "Model trained successfully",
        "MAE": mae,
        "R2": r2,
        "metrics_logged_to": METRICS_LOG
    }


@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_LOG):
        raise HTTPException(status_code=404, detail="No metrics found. Train the model first.")
    df = pd.read_csv(METRICS_LOG)
    return df.to_dict(orient="records")
