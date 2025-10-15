import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

df= pd.read_csv("data/processed/energy_features.csv")
target= "global_active_power"
X=df.drop(columns=["datetime",target])
y=df[target]
split_index = int(0.8*len(df))
X_train,X_test= X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print("training model")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(" model trained MAE = {mae:.4f}")
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": list(X.columns)}, "models/energy_model.joblib")
print("model saved")

