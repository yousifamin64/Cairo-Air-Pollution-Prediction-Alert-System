import pandas as pd
from pathlib import Path

data = Path("data/raw/household_power_consumption.txt")

print(" Loading dataset")
df = pd.read_csv(
    data,
    sep=";", 
    parse_dates={"datetime": ["Date", "Time"]},  
    na_values=["?"], 
    low_memory=False
)

print("Load:", df.shape, "rows")

df.columns = [c.lower() for c in df.columns]

df = df.dropna(subset=["global_active_power"])

for col in df.columns:
    if col != "datetime":
        df[col] = pd.to_numeric(df[col], errors="coerce")

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/cleaned_data.csv", index=False)
print(" Cleaned data saved")
