"""
07_ensemble_predict.py — Step 2 of TFT_improve_steps.md

Averages the quantile predictions (q10, q50, q90) of every checkpoint in a
"family" and writes the ensembled predictions for the val + test splits.

Checkpoint families (in tft_utils.CHECKPOINT_PATTERNS):
    original  -- tft_best.ckpt + tft_best-v*.ckpt (train through 2020)
    step1     -- tft_best_2022*.ckpt              (train through 2021)
    step5     -- tft_best_xgbfused*.ckpt          (XGBoost-as-feature variant)

Output:
    data/processed/tft_predictions_ensemble_<family>.csv

Run:
    python scripts/07_ensemble_predict.py --family original
    python scripts/07_ensemble_predict.py --family step1
    python scripts/07_ensemble_predict.py --family step5
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gpu_utils import tft_predict_trainer_kwargs
from tft_utils import list_checkpoints, load_tft_from_checkpoint, normalize_prediction_output

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["original", "step1", "step5"], default="original")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request (0=CPU)")
args = parser.parse_args()
FAMILY = args.family

predict_kwargs, predict_message = tft_predict_trainer_kwargs(args.gpus)
print(predict_message)

# Family-specific config
FAMILY_CONFIG = {
    "original": {
        "dataset_csv": "master_dataset.csv",
        "train_cutoff": "2020-12-31",
        "extra_known_reals": [],
    },
    "step1": {
        "dataset_csv": "master_dataset.csv",
        "train_cutoff": "2021-12-31",
        "extra_known_reals": [],
    },
    "step5": {
        "dataset_csv": "master_dataset_xgbfused.csv",
        "train_cutoff": "2021-12-31",
        "extra_known_reals": ["xgb_log_pred"],
    },
}[FAMILY]

print(f"Family: {FAMILY}")
print(f"Loading {FAMILY_CONFIG['dataset_csv']}...")
dataset_path = PROCESSED / FAMILY_CONFIG["dataset_csv"]
if not dataset_path.exists():
    raise SystemExit(f"ERROR: {dataset_path} not found. Generate it first.")

df = pd.read_csv(dataset_path, parse_dates=["date"])
for col in ["commodity", "market", "admin1", "season"]:
    df[col] = df[col].astype(str)

df_train = df[df["date"] <= FAMILY_CONFIG["train_cutoff"]].copy()

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
    time_varying_known_reals=(
        ["time_idx", "year", "month", "month_sin", "month_cos", "covid_lockdown"]
        + FAMILY_CONFIG["extra_known_reals"]
    ),
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

checkpoints = list_checkpoints(MODELS, family=FAMILY)
if not checkpoints:
    raise SystemExit(f"ERROR: no checkpoints found for family '{FAMILY}' in {MODELS}")

print(f"Found {len(checkpoints)} checkpoints in family '{FAMILY}':")
for ck in checkpoints:
    print(f"  - {ck.name}")


def predict_split(model, quantiles, split_df, split_name):
    q10_idx = quantiles.index(0.1)
    q50_idx = quantiles.index(0.5)
    q90_idx = quantiles.index(0.9)

    dataset = TimeSeriesDataSet.from_dataset(
        training, split_df, predict=False, stop_randomization=True
    )
    dataloader = dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0, shuffle=False
    )
    print(f"    {split_name}: {len(dataset)} windows")
    quantile_preds = normalize_prediction_output(
        model.predict(dataloader, mode="quantiles", trainer_kwargs=predict_kwargs)
    )
    decoded = dataset.decoded_index.reset_index(drop=True)

    records = []
    for i in range(min(len(decoded), quantile_preds.shape[0])):
        row = decoded.iloc[i]
        sid = row["series_id"]
        first_pred_time = row["time_idx_first_prediction"]
        series_df = df[df["series_id"] == sid].sort_values("time_idx")
        for step in range(max_prediction_length):
            target_time = first_pred_time + step
            match = series_df[series_df["time_idx"] == target_time]
            if len(match) == 0:
                continue
            actual = match.iloc[0]
            records.append({
                "series_id": sid,
                "date": actual["date"],
                "commodity": actual["commodity"],
                "market": actual["market"],
                "price": actual["price"],
                "horizon": step + 1,
                "split": split_name,
                "log_q10": quantile_preds[i, step, q10_idx].item(),
                "log_q50": quantile_preds[i, step, q50_idx].item(),
                "log_q90": quantile_preds[i, step, q90_idx].item(),
            })
    return pd.DataFrame(records)


# VAL window depends on family (2021-22 for original, 2022 for step1/step5)
if FAMILY == "original":
    val_range = ("2021-01-01", "2022-12-31")
else:
    val_range = ("2022-01-01", "2022-12-31")
df_val_ctx = df[(df["date"] >= "2020-01-01") & (df["date"] <= val_range[1])].copy()
df_test_ctx = df[df["date"] >= "2020-01-01"].copy()

all_member_dfs = []
for idx, ck in enumerate(checkpoints, start=1):
    print(f"\n[{idx}/{len(checkpoints)}] Loading {ck.name}...")
    model, quantiles = load_tft_from_checkpoint(training, ck)
    val_df = predict_split(model, quantiles, df_val_ctx, "val")
    val_df = val_df[(val_df["date"] >= val_range[0]) & (val_df["date"] <= val_range[1])]
    test_df = predict_split(model, quantiles, df_test_ctx, "test")
    test_df = test_df[test_df["date"] >= "2023-01-01"]
    member = pd.concat([val_df, test_df], ignore_index=True)
    member["checkpoint"] = ck.name
    all_member_dfs.append(member)
    del model

print("\nAveraging quantile outputs across checkpoints (in log space)...")
keys = ["series_id", "date", "commodity", "market", "price", "horizon", "split"]
stacked = pd.concat(all_member_dfs, ignore_index=True)
ensemble = (
    stacked.groupby(keys, as_index=False)[["log_q10", "log_q50", "log_q90"]].mean()
)
ensemble["tft_q10"] = np.expm1(ensemble["log_q10"])
ensemble["tft_q50"] = np.expm1(ensemble["log_q50"])
ensemble["tft_q90"] = np.expm1(ensemble["log_q90"])
ensemble["band_width"] = ensemble["tft_q90"] - ensemble["tft_q10"]

agg = (
    ensemble.groupby(["series_id", "date", "commodity", "market", "price", "split"],
                     as_index=False)
    .agg(
        tft_q10=("tft_q10", "mean"),
        tft_q50=("tft_q50", "mean"),
        tft_q90=("tft_q90", "mean"),
        log_q10=("log_q10", "mean"),
        log_q50=("log_q50", "mean"),
        log_q90=("log_q90", "mean"),
    )
)
agg["band_width"] = agg["tft_q90"] - agg["tft_q10"]
agg["date"] = pd.to_datetime(agg["date"])
agg = agg.sort_values(["series_id", "date"]).reset_index(drop=True)

out_path = PROCESSED / f"tft_predictions_ensemble_{FAMILY}.csv"
agg.to_csv(out_path, index=False)

print(f"\nWrote {out_path} ({len(agg):,} rows)\n")
print("=" * 60)
print(f"ENSEMBLE METRICS -- family = {FAMILY}")
print("=" * 60)
for name, rng in [(f"Val ({val_range[0][:4]}-{val_range[1][:4]})", val_range),
                  ("Test (2023+)", ("2023-01-01", "2030-01-01"))]:
    mask = (agg["date"] >= rng[0]) & (agg["date"] <= rng[1])
    sub = agg[mask]
    if len(sub) == 0:
        continue
    mae = np.mean(np.abs(sub["price"] - sub["tft_q50"]))
    mape = np.mean(np.abs((sub["price"] - sub["tft_q50"]) / sub["price"])) * 100
    cov = np.mean((sub["price"] >= sub["tft_q10"]) & (sub["price"] <= sub["tft_q90"])) * 100
    bw = np.mean(sub["tft_q90"] - sub["tft_q10"])
    print(f"  {name}: MAE={mae:.2f}  MAPE={mape:.1f}%  Coverage={cov:.1f}%  BandWidth={bw:.2f}")
    for commodity in sorted(sub["commodity"].unique()):
        csub = sub[sub["commodity"] == commodity]
        cmae = np.mean(np.abs(csub["price"] - csub["tft_q50"]))
        cmape = np.mean(np.abs((csub["price"] - csub["tft_q50"]) / csub["price"])) * 100
        ccov = np.mean((csub["price"] >= csub["tft_q10"]) & (csub["price"] <= csub["tft_q90"])) * 100
        print(f"    {commodity:10s} MAE={cmae:.2f}  MAPE={cmape:.1f}%  Cov={ccov:.0f}%")
print("=" * 60)
