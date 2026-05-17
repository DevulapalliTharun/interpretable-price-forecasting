"""
03_train_tft.py
Train Temporal Fusion Transformer on the master dataset.
Output: models/tft_best.pt, models/tft_config.json
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner

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

# ── Parse arguments ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lr_finder", "--lr_finder_only", action="store_true",
                    help="Run LR finder only, then exit")
parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request (0=CPU)")
args = parser.parse_args()

# ── Load master dataset ───────────────────────────────────────────────
print("Loading master dataset...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])

# Encode categoricals as strings for pytorch-forecasting
for col in ["commodity", "market", "admin1", "season"]:
    df[col] = df[col].astype(str)

# ── Train / Validation / Test split (time-based) ─────────────────────
df_train = df[df["date"] <= "2020-12-31"].copy()
df_val = df[(df["date"] >= "2021-01-01") & (df["date"] <= "2022-12-31")].copy()
df_test = df[df["date"] >= "2023-01-01"].copy()

print(f"  Train: {len(df_train):,} rows ({df_train['date'].min()} to {df_train['date'].max()})")
print(f"  Val:   {len(df_val):,} rows ({df_val['date'].min()} to {df_val['date'].max()})")
print(f"  Test:  {len(df_test):,} rows ({df_test['date'].min()} to {df_test['date'].max()})")

# ── TimeSeriesDataSet ─────────────────────────────────────────────────
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
    static_reals=[],
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
    target_normalizer=GroupNormalizer(
        groups=["series_id"],
        transformation="softplus",
    ),
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

validation = TimeSeriesDataSet.from_dataset(
    training, df_val, predict=True, stop_randomization=True
)

train_dataloader = training.to_dataloader(
    train=True, batch_size=args.batch_size, num_workers=0, shuffle=True
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=args.batch_size, num_workers=0, shuffle=False
)

# ── TFT Model ────────────────────────────────────────────────────────
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

param_count = sum(p.numel() for p in tft.parameters())
print(f"\nModel parameters: {param_count / 1e3:.1f}k")

# ── Trainer ───────────────────────────────────────────────────────────
trainer_settings, trainer_message = tft_trainer_settings(args.gpus)
print(trainer_message)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    **trainer_settings,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            mode="min",
        ),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=str(MODELS),
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="tft_best",
        ),
    ],
)

# ── LR Finder ────────────────────────────────────────────────────────
if args.lr_finder:
    print("\nRunning Learning Rate Finder...")
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=0.1,
        min_lr=1e-5,
    )
    suggested_lr = res.suggestion()
    print(f"\nSuggested LR: {suggested_lr:.6f}")
    print("Use: python scripts/03_train_tft.py --lr {:.6f}".format(suggested_lr))
    exit(0)

# ── Train ─────────────────────────────────────────────────────────────
print(
    f"\nTraining TFT (lr={args.lr}, epochs={args.epochs}, "
    f"accelerator={trainer_settings['accelerator']}, devices={trainer_settings['devices']})..."
)
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# ── Save config ───────────────────────────────────────────────────────
config = {
    "hidden_size": 32,
    "attention_head_size": 2,
    "lstm_layers": 1,
    "hidden_continuous_size": 16,
    "dropout": 0.2,
    "quantiles": [0.1, 0.5, 0.9],
    "learning_rate": args.lr,
    "max_encoder_length": max_encoder_length,
    "max_prediction_length": max_prediction_length,
    "parameters": param_count,
    "train_rows": len(df_train),
    "val_rows": len(df_val),
    "test_rows": len(df_test),
}
with open(MODELS / "tft_config.json", "w") as f:
    json.dump(config, f, indent=2)

best_model_path = trainer.checkpoint_callback.best_model_path
print(f"\n{'='*60}")
print(f"Best model: {best_model_path}")
print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")
print(f"Config: {MODELS / 'tft_config.json'}")
print(f"{'='*60}")
