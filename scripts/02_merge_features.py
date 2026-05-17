"""
02_merge_features.py
Merge filtered prices with NASA weather and engineer all features.
Output: data/processed/master_dataset.csv (~15,200 rows, 24 columns)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ── Load data ─────────────────────────────────────────────────────────
print("Loading data...")
prices = pd.read_csv(PROCESSED / "prices_filtered.csv", parse_dates=["date"])
weather = pd.read_csv(RAW / "nasa_weather_1994_2026.csv")

print(f"  Prices:  {len(prices):,} rows")
print(f"  Weather: {len(weather):,} rows")

# ── Step 3.1: Merge on (market, year, month) ─────────────────────────
prices["year"] = prices["date"].dt.year
prices["month"] = prices["date"].dt.month

df = prices.merge(
    weather[["market", "year", "month", "temperature_mean", "rainfall_monthly", "humidity_mean"]],
    on=["market", "year", "month"],
    how="left",
)

# Forward-fill missing weather up to 3 months per series
for col in ["temperature_mean", "rainfall_monthly", "humidity_mean"]:
    df[col] = df.groupby("series_id")[col].transform(
        lambda x: x.ffill(limit=3)
    )

missing_weather = df["temperature_mean"].isna().sum()
print(f"  Missing weather after ffill: {missing_weather} rows")
df = df.dropna(subset=["temperature_mean"]).copy()

# ── Step 3.2: Log-transform price ────────────────────────────────────
df["log_price"] = np.log1p(df["price"])

# ── Step 3.3: Cyclical month encoding ────────────────────────────────
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ── Step 3.4: Season label (Indian agricultural calendar) ────────────
def get_season(month):
    if month in [7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2]:
        return "Rabi"
    else:
        return "Zaid"

df["season"] = df["month"].apply(get_season)

# ── Step 3.5: Weather shock indicators ───────────────────────────────
df["rain_deficit"] = (df["rainfall_monthly"] < 50).astype(int)
df["rain_excess"] = (df["rainfall_monthly"] > 400).astype(int)
df["heat_stress"] = (df["temperature_mean"] > 38).astype(int)
df["cold_stress"] = (df["temperature_mean"] < 10).astype(int)

# ── Step 3.6: Price lag features ─────────────────────────────────────
df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

df["price_lag_1m"] = df.groupby("series_id")["log_price"].shift(1)
df["price_lag_12m"] = df.groupby("series_id")["log_price"].shift(12)
df["rolling_3m"] = df.groupby("series_id")["log_price"].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
df["rolling_6m"] = df.groupby("series_id")["log_price"].transform(
    lambda x: x.shift(1).rolling(6).mean()
)

# ── Step 3.7: YoY change ─────────────────────────────────────────────
df["yoy_change"] = df.groupby("series_id")["log_price"].pct_change(12)

# ── Step 3.8: Drop lag warmup rows ───────────────────────────────────
before = len(df)
df = df.dropna(subset=["price_lag_12m", "rolling_6m"]).copy()
print(f"  Dropped {before - len(df)} lag warmup rows")

# ── Step 3.9: COVID lockdown flag ────────────────────────────────────
df["covid_lockdown"] = (
    (df["date"] >= "2020-03-15") & (df["date"] <= "2020-09-15")
).astype(int)

# ── Final column selection and output ─────────────────────────────────
output_cols = [
    "time_idx", "series_id", "commodity", "market", "admin1",
    "date", "year", "month",
    "month_sin", "month_cos", "season", "covid_lockdown",
    "log_price",
    "temperature_mean", "rainfall_monthly", "humidity_mean",
    "price_lag_1m", "price_lag_12m", "rolling_3m", "rolling_6m",
    "yoy_change",
    "rain_deficit", "rain_excess", "heat_stress", "cold_stress",
    "price",
]
output_cols = [c for c in output_cols if c in df.columns]

df = df[output_cols].sort_values(["series_id", "time_idx"]).reset_index(drop=True)
df.to_csv(PROCESSED / "master_dataset.csv", index=False)

print(f"\n{'='*60}")
print(f"OUTPUT: {PROCESSED / 'master_dataset.csv'}")
print(f"  Rows:      {len(df):,}")
print(f"  Columns:   {len(df.columns)}")
print(f"  Series:    {df['series_id'].nunique()}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
for comm in df["commodity"].unique():
    n = df[df["commodity"] == comm]["series_id"].nunique()
    print(f"  {comm:10s}: {n} series")
print(f"{'='*60}")
