"""
09_retrain_tft_2022.py — Step 1 of TFT_improve_steps.md

Retrains the TFT on an extended train window (<= Dec 2021) with 2022 as the
validation split, so the test set (2023+) is compared against a model that
saw as much data as the XGBoost baseline did. The original TFT
stopped at Dec 2020 while the baseline trained through Dec 2022 — that was
the unfair part of the comparison.

Output checkpoint: models/tft_best_2022-v*.ckpt (lightning auto-versions)
Output config:     models/tft_config_2022.json

Uses CUDA by default when a CUDA-enabled PyTorch build is available.
Pass --gpus 0 to force CPU mode.

Run:
    python scripts/09_retrain_tft_2022.py
    python scripts/09_retrain_tft_2022.py --epochs 30 --batch_size 64
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gpu_utils import tft_trainer_settings

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
args = parser.parse_args()

print("Loading master dataset...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])
for col in ["commodity", "market", "admin1", "season"]:
    df[col] = df[col].astype(str)

# Extended split: train through 2021, val on 2022, test on 2023+.
df_train = df[df["date"] <= "2021-12-31"].copy()
df_val = df[(df["date"] >= "2022-01-01") & (df["date"] <= "2022-12-31")].copy()
df_test = df[df["date"] >= "2023-01-01"].copy()

print(f"  Train: {len(df_train):,} rows ({df_train['date'].min().date()} - {df_train['date'].max().date()})")
print(f"  Val:   {len(df_val):,} rows ({df_val['date'].min().date()} - {df_val['date'].max().date()})")
print(f"  Test:  {len(df_test):,} rows ({df_test['date'].min().date()} - {df_test['date'].max().date()})")

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
        "time_idx", "year", "month", "month_sin", "month_cos", "covid_lockdown"
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

# Build val with the FULL historical context (need prior 2+ years to form
# encoder windows that land in 2022). We filter the predictions post-hoc.
df_val_ctx = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2022-12-31")].copy()
validation = TimeSeriesDataSet.from_dataset(
    training, df_val_ctx, predict=True, stop_randomization=True
)

train_dl = training.to_dataloader(
    train=True, batch_size=args.batch_size, num_workers=0, shuffle=True
)
val_dl = validation.to_dataloader(
    train=False, batch_size=args.batch_size, num_workers=0, shuffle=False
)

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
print(f"Model parameters: {sum(p.numel() for p in tft.parameters()) / 1e3:.1f}k")

trainer_settings, trainer_message = tft_trainer_settings(args.gpus)
print(trainer_message)

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
            filename="tft_best_2022",
        ),
    ],
    enable_progress_bar=True,
    log_every_n_steps=25,
)

print(f"Training: lr={args.lr} epochs={args.epochs} batch_size={args.batch_size}")
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

with open(MODELS / "tft_config_2022.json", "w") as f:
    json.dump({
        "hidden_size": 32, "attention_head_size": 2, "lstm_layers": 1,
        "hidden_continuous_size": 16, "dropout": 0.2,
        "quantiles": [0.1, 0.5, 0.9],
        "learning_rate": args.lr,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "train_cutoff": "2021-12-31",
        "val_window": "2022",
        "train_rows": len(df_train), "val_rows": len(df_val), "test_rows": len(df_test),
    }, f, indent=2)

print("=" * 60)
print(f"Best checkpoint:  {trainer.checkpoint_callback.best_model_path}")
print(f"Best val_loss:    {trainer.checkpoint_callback.best_model_score:.6f}")
print(f"Config:           {MODELS / 'tft_config_2022.json'}")
print("=" * 60)
