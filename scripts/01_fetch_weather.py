"""
01_fetch_weather.py
Fetch monthly weather data from NASA POWER API for all market locations.
Output: data/raw/nasa_weather_1994_2026.csv
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

NASA_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
PARAMS_LIST = "T2M,PRECTOTCORR,RH2M"
START_DATE = "1994"
END_DATE = "2025"
SLEEP_BETWEEN = 2  # seconds between API calls (rate limit: 30/min)

# ── Load filtered prices to get unique market locations ───────────────
prices = pd.read_csv(PROCESSED / "prices_filtered.csv")
locations = prices[["market", "latitude", "longitude"]].drop_duplicates(subset=["market"])
# Deduplicate on (lat, lon) to avoid redundant API calls
locations["lat_round"] = locations["latitude"].round(4)
locations["lon_round"] = locations["longitude"].round(4)
unique_coords = locations.drop_duplicates(subset=["lat_round", "lon_round"])

print(f"Total markets: {len(locations)}")
print(f"Unique coordinates to fetch: {len(unique_coords)}")


def fetch_nasa_weather(lat, lon):
    """Fetch monthly weather from NASA POWER for a single location."""
    params = {
        "parameters": PARAMS_LIST,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    resp = requests.get(NASA_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = []
    props = data.get("properties", {}).get("parameter", {})
    t2m = props.get("T2M", {})
    prec = props.get("PRECTOTCORR", {})
    rh2m = props.get("RH2M", {})

    for key in t2m:
        # Keys are formatted as YYYYMM (e.g., "200001") — skip annual entries (13th month)
        if len(key) == 6:
            year = int(key[:4])
            month = int(key[4:])
            if 1 <= month <= 12:
                records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "year": year,
                    "month": month,
                    "temperature_mean": t2m.get(key, np.nan),
                    "rainfall_monthly": prec.get(key, np.nan),
                    "humidity_mean": rh2m.get(key, np.nan),
                })

    return pd.DataFrame(records)


# ── Fetch weather for all unique coordinates ──────────────────────────
all_weather = []
for i, (_, row) in enumerate(unique_coords.iterrows()):
    lat, lon = row["latitude"], row["longitude"]
    print(f"  [{i+1}/{len(unique_coords)}] Fetching ({lat:.4f}, {lon:.4f})...", end="")
    try:
        df_w = fetch_nasa_weather(lat, lon)
        # Replace NASA fill values (-999) with NaN
        for col in ["temperature_mean", "rainfall_monthly", "humidity_mean"]:
            df_w.loc[df_w[col] < -900, col] = np.nan
        all_weather.append(df_w)
        print(f" OK ({len(df_w)} rows)")
    except Exception as e:
        print(f" FAILED: {e}")

    if i < len(unique_coords) - 1:
        time.sleep(SLEEP_BETWEEN)

weather = pd.concat(all_weather, ignore_index=True)

# ── Map back to all markets (some share coordinates) ──────────────────
# Join locations with their weather data via rounded coordinates
weather["lat_round"] = weather["latitude"].round(4)
weather["lon_round"] = weather["longitude"].round(4)

market_weather = locations[["market", "latitude", "longitude", "lat_round", "lon_round"]].merge(
    weather.drop(columns=["latitude", "longitude"]),
    on=["lat_round", "lon_round"],
    how="left",
)

market_weather = market_weather.drop(columns=["lat_round", "lon_round"])
market_weather = market_weather.sort_values(["market", "year", "month"]).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────
market_weather.to_csv(RAW / "nasa_weather_1994_2026.csv", index=False)

print(f"\n{'='*60}")
print(f"OUTPUT: {RAW / 'nasa_weather_1994_2026.csv'}")
print(f"  Rows:      {len(market_weather):,}")
print(f"  Markets:   {market_weather['market'].nunique()}")
print(f"  Date range: {market_weather['year'].min()}-{market_weather['month'].min():02d} "
      f"to {market_weather['year'].max()}-{market_weather['month'].max():02d}")
print(f"{'='*60}")
