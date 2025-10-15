import pandas as pd 
from pathlib import Path

df =pd.read_csv("data/processed/cleaned_data.csv", parse_dates=["datetime"])
df=df.sort_values("datetime")
df_hourly = df.set_index("datetime").resample("H").mean().reset_index()
df_hourly["hour"]=df_hourly["datetime"].dt.hour 
df_hourly["day_of_week"] = df_hourly["datetime"].dt.dayofweek
df_hourly["month"]=df_hourly["datetime"].dt.month 
df_hourly["is_weekend"]=(df_hourly["day_of_week"] >= 5).astype(int)

target = "global_active_power"
df_hourly[f"{target}_lag_1"] = df_hourly[target].shift(1)
df_hourly[f"{target}_lag_24"] = df_hourly[target].shift(24)
df_hourly[f"{target}_roll_6h"] = df_hourly[target].rolling(window=6).mean()
df_hourly = df_hourly.dropna()

Path("data/processed").mkdir(parents=True,exist_ok=True)
df_hourly.to_csv("data/processed/energy_features.csv", index=False)
print("saved data successfully:", len(df_hourly))