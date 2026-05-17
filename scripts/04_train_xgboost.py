"""
04_train_xgboost.py
Train XGBoost baseline for comparison against TFT.
Output: models/xgb_baseline.pkl
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gpu_utils import xgb_training_settings

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request (0=CPU)")
args = parser.parse_args()

# ── Load master dataset ───────────────────────────────────────────────
print("Loading master dataset...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])

# ── Encode categoricals ──────────────────────────────────────────────
label_encoders = {}
for col in ["commodity", "market", "admin1", "season"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ── Feature columns ──────────────────────────────────────────────────
feature_cols = [
    "time_idx", "year", "month", "month_sin", "month_cos",
    "commodity_enc", "market_enc", "admin1_enc", "season_enc",
    "covid_lockdown",
    "temperature_mean", "rainfall_monthly", "humidity_mean",
    "price_lag_1m", "price_lag_12m", "rolling_3m", "rolling_6m",
    "yoy_change",
    "rain_deficit", "rain_excess", "heat_stress", "cold_stress",
]

target_col = "log_price"

# ── Train / Test split ───────────────────────────────────────────────
df_train = df[df["date"] <= "2022-12-31"].copy()
df_test = df[df["date"] >= "2023-01-01"].copy()

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"  Train: {len(X_train):,} rows")
print(f"  Test:  {len(X_test):,} rows")

# ── Train XGBoost baseline ────────────────────────────────────────────
print("\nTraining XGBoost baseline...")
xgb_settings, xgb_message = xgb_training_settings(args.gpus)
print(f"  {xgb_message}")
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    n_jobs=-1,
    random_state=42,
    verbosity=1,
    **xgb_settings,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert back to price space
train_mae = np.mean(np.abs(np.expm1(y_train) - np.expm1(y_pred_train)))
test_mae = np.mean(np.abs(np.expm1(y_test) - np.expm1(y_pred_test)))
test_mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred_test)) / np.expm1(y_test))) * 100

print(f"\n  Train MAE: {train_mae:.2f} Rs/KG")
print(f"  Test MAE:  {test_mae:.2f} Rs/KG")
print(f"  Test MAPE: {test_mape:.1f}%")

# ── Feature importance ────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nTop 10 features:")
for feat, imp in importances.head(10).items():
    print(f"  {feat:25s} {imp:.4f}")

# ── Save ──────────────────────────────────────────────────────────────
joblib.dump({
    "model": model,
    "label_encoders": label_encoders,
    "feature_cols": feature_cols,
}, MODELS / "xgb_baseline.pkl")

print(f"\n{'='*60}")
print(f"OUTPUT: {MODELS / 'xgb_baseline.pkl'}")
print(f"{'='*60}")
