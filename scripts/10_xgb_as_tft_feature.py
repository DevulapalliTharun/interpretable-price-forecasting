"""
10_xgb_as_tft_feature.py - Step 5 of TFT_improve_steps.md

Pattern 5a: XGBoost-as-TFT-input-feature fusion

Procedure:
 1. Train a "clean" XGBoost model on data <= 2019-12-31 only
     (buffer before the TFT train
      cutoff of 2021-12-31) to minimise label leakage into the TFT training rows
      that matter most (2020-2021).
 2. Use that model to predict xgb_log_pred for every row in the master dataset.
     - Historical rows <= 2019: mild leakage (model saw them); signal is
       essentially "XGBoost says the price from lagged features", weak effect.
      - Rows 2020+: leak-free, genuine predictions.
 3. Save master_dataset_xgbfused.csv with the extra column.
 4. Retrain the TFT using xgb_log_pred as a time_varying_known_real feature.
    TFT then learns to correct the baseline residual — VSN will rank how much
    it trusts the fused signal vs weather / lags.

Outputs:
    data/processed/master_dataset_xgbfused.csv
    models/xgb_clean_2019.pkl
    models/tft_best_xgbfused-v*.ckpt   (lightning auto-versions)
    models/tft_config_xgbfused.json

Run:
    python scripts/10_xgb_as_tft_feature.py
    python scripts/10_xgb_as_tft_feature.py --epochs 30 --batch_size 64
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gpu_utils import tft_trainer_settings, xgb_training_settings

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
args = parser.parse_args()

# ---------------------------------------------------------------
# 1. Clean XGBoost baseline on pre-2020 data
# ---------------------------------------------------------------
print("[1/4] Loading master dataset...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])
for col in ["commodity", "market", "admin1", "season"]:
    df[col] = df[col].astype(str)

label_encoders = {}
for col in ["commodity", "market", "admin1", "season"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

feature_cols = [
    "time_idx", "year", "month", "month_sin", "month_cos",
    "commodity_enc", "market_enc", "admin1_enc", "season_enc",
    "covid_lockdown",
    "temperature_mean", "rainfall_monthly", "humidity_mean",
    "price_lag_1m", "price_lag_12m", "rolling_3m", "rolling_6m",
    "yoy_change",
    "rain_deficit", "rain_excess", "heat_stress", "cold_stress",
]

xgb_cutoff = "2019-12-31"
df_xgb_train = df[df["date"] <= xgb_cutoff].copy()
print(f"  Clean XGBoost train: {len(df_xgb_train):,} rows (<= {xgb_cutoff})")

print("[1/4] Training clean XGBoost baseline on pre-2020 data...")
xgb_settings, xgb_message = xgb_training_settings(args.gpus)
print(f"  {xgb_message}")
xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    **xgb_settings,
)
xgb.fit(df_xgb_train[feature_cols].values, df_xgb_train["log_price"].values)

joblib.dump({
    "model": xgb,
    "label_encoders": label_encoders,
    "feature_cols": feature_cols,
    "train_cutoff": xgb_cutoff,
}, MODELS / "xgb_clean_2019.pkl")
print(f"  Saved: {MODELS / 'xgb_clean_2019.pkl'}")

# ---------------------------------------------------------------
# 2. Predict xgb_log_pred for ALL master rows
# ---------------------------------------------------------------
print("[2/4] Generating baseline predictions for entire dataset...")
df["xgb_log_pred"] = xgb.predict(df[feature_cols].values)

# Sanity: report baseline performance on pre-2020 (train) and 2020+ (unseen)
for name, mask in [
    ("Train <=2019 (seen)", df["date"] <= xgb_cutoff),
    ("Val 2020-2022 (unseen)", (df["date"] >= "2020-01-01") & (df["date"] <= "2022-12-31")),
    ("Test 2023+ (unseen)", df["date"] >= "2023-01-01"),
]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    y = np.expm1(sub["log_price"])
    yhat = np.expm1(sub["xgb_log_pred"])
    mae = np.mean(np.abs(y - yhat))
    mape = np.mean(np.abs((y - yhat) / y.replace(0, np.nan))) * 100
    print(f"  XGBoost on {name:26s} n={len(sub):>5}  MAE={mae:5.2f}  MAPE={mape:4.1f}%")

# Drop the *_enc helpers before saving (TFT doesn't need them)
out_cols = [c for c in df.columns if not c.endswith("_enc")]
fused_path = PROCESSED / "master_dataset_xgbfused.csv"
df[out_cols].to_csv(fused_path, index=False)
print(f"  Saved fused dataset: {fused_path}")

# ---------------------------------------------------------------
# 3. Build TFT training dataset with xgb_log_pred as known_real
# ---------------------------------------------------------------
print("[3/4] Building TFT dataset with xgb_log_pred feature...")
df_tft = df[out_cols].copy()  # xgb_log_pred included, *_enc stripped

df_train = df_tft[df_tft["date"] <= "2021-12-31"].copy()
df_val = df_tft[(df_tft["date"] >= "2022-01-01") & (df_tft["date"] <= "2022-12-31")].copy()
df_test = df_tft[df_tft["date"] >= "2023-01-01"].copy()
print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

max_encoder_length = 24
max_prediction_length = 6

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="log_price",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    min_encoder_length=12,
    max_prediction_length=max_prediction_length,
    min_prediction_length=1,
    static_categoricals=["commodity", "market", "admin1"],
    time_varying_known_categoricals=["season"],
    time_varying_known_reals=[
        "time_idx", "year", "month", "month_sin", "month_cos", "covid_lockdown",
        "xgb_log_pred",   # <-- fused feature
    ],
    time_varying_unknown_reals=[
        "log_price",
        "temperature_mean", "rainfall_monthly", "humidity_mean",
        "price_lag_1m", "price_lag_12m", "rolling_3m", "rolling_6m",
        "yoy_change",
        "rain_deficit", "rain_excess", "heat_stress", "cold_stress",
    ],
    target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
    categorical_encoders={
        "commodity": NaNLabelEncoder(add_nan=True),
        "market": NaNLabelEncoder(add_nan=True),
        "admin1": NaNLabelEncoder(add_nan=True),
        "season": NaNLabelEncoder(add_nan=True),
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

df_val_ctx = df_tft[(df_tft["date"] >= "2020-01-01") & (df_tft["date"] <= "2022-12-31")].copy()
validation = TimeSeriesDataSet.from_dataset(
    training, df_val_ctx, predict=True, stop_randomization=True
)

train_dl = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0, shuffle=True)
val_dl = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0, shuffle=False)

# ---------------------------------------------------------------
# 4. Train TFT
# ---------------------------------------------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=32,
    attention_head_size=2,
    lstm_layers=1,
    hidden_continuous_size=16,
    dropout=0.2,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    learning_rate=args.lr,
    optimizer="ranger",
    reduce_on_plateau_patience=3,
    log_interval=10,
    mask_bias=-float("inf"),
)
print(f"  Model parameters: {sum(p.numel() for p in tft.parameters()) / 1e3:.1f}k")

trainer_settings, trainer_message = tft_trainer_settings(args.gpus)
print(trainer_message)
print(f"[4/4] Training TFT-XGBoost-fused on {trainer_settings['accelerator']}...")

trainer = pl.Trainer(
    max_epochs=args.epochs,
    **trainer_settings,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=args.patience, min_delta=1e-4, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=str(MODELS),
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="tft_best_xgbfused",
        ),
    ],
    enable_progress_bar=True,
    log_every_n_steps=25,
)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

with open(MODELS / "tft_config_xgbfused.json", "w") as f:
    json.dump({
        "hidden_size": 32, "attention_head_size": 2, "lstm_layers": 1,
        "hidden_continuous_size": 16, "dropout": 0.2,
        "quantiles": [0.1, 0.5, 0.9],
        "learning_rate": args.lr,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "train_cutoff": "2021-12-31",
        "val_window": "2022",
        "xgb_train_cutoff": xgb_cutoff,
        "fused_feature": "xgb_log_pred",
        "train_rows": len(df_train), "val_rows": len(df_val), "test_rows": len(df_test),
    }, f, indent=2)

print("=" * 60)
print(f"Best checkpoint:  {trainer.checkpoint_callback.best_model_path}")
print(f"Best val_loss:    {trainer.checkpoint_callback.best_model_score:.6f}")
print(f"Config:           {MODELS / 'tft_config_xgbfused.json'}")
print(f"Fused dataset:    {fused_path}")
print("=" * 60)
