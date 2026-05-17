"""
00_filter_prices.py
Clean and filter WFP India food prices CSV.
Output: data/processed/prices_filtered.csv (~16,701 rows, 142 series)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SELECTED_COMMODITIES = ["Onions", "Tomatoes", "Rice"]
MIN_SERIES_MONTHS = 60

# ── Step 1.1: Load both CSVs ────────────────────────────────────────
print("Loading raw data...")
prices = pd.read_csv(RAW / "wfp_food_prices_ind.csv")
markets = pd.read_csv(RAW / "wfp_markets_ind.csv")

prices["date"] = pd.to_datetime(prices["date"])

print(f"  Raw prices: {len(prices):,} rows")
print(f"  Markets:    {len(markets):,} rows")

# ── Step 1.2: Drop "National Average" aggregate rows ─────────────────
before = len(prices)
prices = prices[prices["market"] != "National Average"].copy()
print(f"  Dropped {before - len(prices)} 'National Average' rows")

# ── Step 1.3: Keep only Retail ───────────────────────────────────────
prices = prices[prices["pricetype"] == "Retail"].copy()
print(f"  After Retail filter: {len(prices):,} rows")

# ── Step 1.4: Keep only selected commodities ─────────────────────────
prices = prices[prices["commodity"].isin(SELECTED_COMMODITIES)].copy()
print(f"  After commodity filter ({', '.join(SELECTED_COMMODITIES)}): {len(prices):,} rows")

# ── Step 1.5: Keep only KG unit ──────────────────────────────────────
prices = prices[prices["unit"] == "KG"].copy()
print(f"  After KG filter: {len(prices):,} rows")

# ── Step 1.6: Merge lat/lon from markets CSV ─────────────────────────
market_cols = [c for c in markets.columns if c in ["market_id", "latitude", "longitude"]]
if "latitude" not in prices.columns and "market_id" in prices.columns and "market_id" in markets.columns:
    prices = prices.merge(
        markets[market_cols],
        on="market_id",
        how="left",
        suffixes=("", "_mkt"),
    )
    print(f"  Merged lat/lon from markets CSV")
else:
    print(f"  Lat/lon already present or merge key missing — skipping merge")

# ── Step 1.7: Normalize duplicate market names sharing coordinates ────
# e.g., Tiruvanantapuram / Trivandrum / T.Puram → same (lat, lon)
if "latitude" in prices.columns and "longitude" in prices.columns:
    coord_to_market = (
        prices[["market", "latitude", "longitude"]]
        .drop_duplicates(subset=["market"])
        .round({"latitude": 3, "longitude": 3})
    )
    # Group markets that share the same rounded coordinates
    dup_coords = coord_to_market.groupby(["latitude", "longitude"])["market"].apply(list)
    dup_coords = dup_coords[dup_coords.apply(len) > 1]
    rename_map = {}
    for coord, market_list in dup_coords.items():
        canonical = sorted(market_list, key=len)[0]  # keep shortest name
        for m in market_list:
            if m != canonical:
                rename_map[m] = canonical
    if rename_map:
        prices["market"] = prices["market"].replace(rename_map)
        print(f"  Normalized {len(rename_map)} duplicate market names: {rename_map}")
    else:
        print(f"  No duplicate market names found")

# ── Step 1.8: Remove duplicate (market, commodity, date) rows ────────
before = len(prices)
prices = prices.sort_values("price", ascending=False)
prices = prices.drop_duplicates(subset=["market", "commodity", "date"], keep="first")
prices = prices.sort_values(["commodity", "market", "date"])
print(f"  Removed {before - len(prices)} duplicate rows")

# ── Step 1.7 & 1.8: Series length filter ─────────────────────────────
series_lengths = prices.groupby(["market", "commodity"])["date"].nunique()
valid_pairs = series_lengths[series_lengths >= MIN_SERIES_MONTHS].index
prices = prices.set_index(["market", "commodity"])
prices = prices.loc[prices.index.isin(valid_pairs)].reset_index()
print(f"  After >= {MIN_SERIES_MONTHS} months filter: {len(prices):,} rows")

# ── Step 1.9: Create series_id ───────────────────────────────────────
prices["series_id"] = prices["commodity"] + "_" + prices["market"]

# ── Step 1.10: Create time_idx ────────────────────────────────────────
prices["time_idx"] = (prices["date"].dt.year - 1994) * 12 + (prices["date"].dt.month - 1)

# ── Step 1.14: Forward-fill gaps <= 3 consecutive months ──────────────
filled_frames = []
for sid, grp in prices.groupby("series_id"):
    grp = grp.set_index("time_idx").sort_index()
    full_idx = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
    grp = grp.reindex(full_idx)
    # Forward-fill short gaps (<=3 months)
    grp["price"] = grp["price"].ffill(limit=3)
    # Propagate metadata
    for col in ["series_id", "commodity", "market", "admin1", "admin2",
                "latitude", "longitude", "category", "unit", "priceflag",
                "pricetype", "currency"]:
        if col in grp.columns:
            grp[col] = grp[col].ffill().bfill()
    # Reconstruct date from time_idx
    grp.index.name = "time_idx"
    grp = grp.reset_index()
    grp["date"] = pd.to_datetime(
        ((grp["time_idx"] // 12) + 1994).astype(int).astype(str)
        + "-"
        + ((grp["time_idx"] % 12) + 1).astype(int).astype(str).str.zfill(2)
        + "-15"
    )
    filled_frames.append(grp)

prices = pd.concat(filled_frames, ignore_index=True)

# ── Step 1.15: Drop series with gaps > 6 consecutive months ──────────
drop_ids = []
for sid, grp in prices.groupby("series_id"):
    grp_sorted = grp.set_index("time_idx").sort_index()
    if grp_sorted["price"].isna().any():
        na_mask = grp_sorted["price"].isna()
        if na_mask.any():
            gaps = na_mask.astype(int).groupby((~na_mask).cumsum()).sum()
            if gaps.max() > 6:
                drop_ids.append(sid)

prices = prices[~prices["series_id"].isin(drop_ids)].copy()
prices = prices.dropna(subset=["price"])
print(f"  Dropped {len(drop_ids)} series with gaps > 6 months")

# ── Final output ──────────────────────────────────────────────────────
output_cols = ["date", "admin1", "admin2", "market", "commodity",
               "series_id", "time_idx", "latitude", "longitude", "price"]
output_cols = [c for c in output_cols if c in prices.columns]
prices = prices[output_cols].sort_values(["series_id", "time_idx"]).reset_index(drop=True)

prices.to_csv(PROCESSED / "prices_filtered.csv", index=False)

n_series = prices["series_id"].nunique()
print(f"\n{'='*60}")
print(f"OUTPUT: {PROCESSED / 'prices_filtered.csv'}")
print(f"  Rows:      {len(prices):,}")
print(f"  Series:    {n_series}")
for comm in SELECTED_COMMODITIES:
    n = prices[prices["commodity"] == comm]["series_id"].nunique()
    print(f"  {comm:10s}: {n} series")
print(f"{'='*60}")
