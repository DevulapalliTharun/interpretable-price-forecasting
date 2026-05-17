"""
08_conformal_calibrate.py — Steps 3 + 4 of TFT_improve_steps.md

Per-commodity Conformalized Quantile Regression (CQR, Romano et al. 2019).

Reads tft_predictions_ensemble_<family>.csv produced by 07_ensemble_predict.py.
Uses the VAL split as the calibration set to compute, for each commodity, a
conformity offset Q_hat that makes the 90% band attain >= 90% coverage.

Outputs:
  data/processed/tft_predictions_calibrated_<family>.csv
  models/conformal_offsets_<family>.json

Run:
    python scripts/08_conformal_calibrate.py --family original
    python scripts/08_conformal_calibrate.py --family step1
    python scripts/08_conformal_calibrate.py --family step5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["original", "step1", "step5"], default="original")
args = parser.parse_args()
FAMILY = args.family

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ALPHA = 0.10  # target 90% coverage

src = PROCESSED / f"tft_predictions_ensemble_{FAMILY}.csv"
if not src.exists():
    raise SystemExit(
        f"ERROR: {src} not found. Run scripts/07_ensemble_predict.py --family {FAMILY} first."
    )

print(f"Family: {FAMILY}")
print(f"Loading {src.name}...")
df = pd.read_csv(src, parse_dates=["date"])

# Work in log space: this matches how the TFT produces quantiles.
df["log_y"] = np.log1p(df["price"])

val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()
print(f"Calibration set (val): {len(val)} rows")
print(f"Test set:              {len(test)} rows")

offsets = {}
print("\nPer-commodity conformity offset (log space):")
print("-" * 60)
for commodity, sub in val.groupby("commodity"):
    # CQR conformity score:
    #   E_i = max( log_q10_i - log_y_i,  log_y_i - log_q90_i )
    E = np.maximum(sub["log_q10"] - sub["log_y"], sub["log_y"] - sub["log_q90"]).values
    n = len(E)
    # finite-sample adjusted quantile level
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    k = min(k, n)
    q_hat = float(np.sort(E)[k - 1])
    offsets[commodity] = q_hat
    print(f"  {commodity:10s} n={n:4d}  Q_hat(log)={q_hat:+.4f}")

print("-" * 60)

offsets_path = MODELS / f"conformal_offsets_{FAMILY}.json"
offsets_path.write_text(
    json.dumps({"family": FAMILY, "alpha": ALPHA, "offsets_log_space": offsets}, indent=2)
)
print(f"Saved offsets -> {offsets_path}")


def apply_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    q_hat_vec = out["commodity"].map(offsets).astype(float)
    out["log_q10_cal"] = out["log_q10"] - q_hat_vec
    out["log_q90_cal"] = out["log_q90"] + q_hat_vec
    out["tft_q10_cal"] = np.expm1(out["log_q10_cal"])
    out["tft_q50_cal"] = out["tft_q50"]  # median unchanged
    out["tft_q90_cal"] = np.expm1(out["log_q90_cal"])
    out["band_width_cal"] = out["tft_q90_cal"] - out["tft_q10_cal"]
    return out


calibrated = apply_calibration(df)
out_path = PROCESSED / f"tft_predictions_calibrated_{FAMILY}.csv"
calibrated.to_csv(out_path, index=False)
print(f"Wrote calibrated predictions -> {out_path} ({len(calibrated):,} rows)\n")


def report(name: str, sub: pd.DataFrame):
    if len(sub) == 0:
        return
    mae = np.mean(np.abs(sub["price"] - sub["tft_q50"]))
    mape = np.mean(np.abs((sub["price"] - sub["tft_q50"]) / sub["price"])) * 100
    cov_raw = np.mean(
        (sub["price"] >= sub["tft_q10"]) & (sub["price"] <= sub["tft_q90"])
    ) * 100
    bw_raw = np.mean(sub["tft_q90"] - sub["tft_q10"])
    cov_cal = np.mean(
        (sub["price"] >= sub["tft_q10_cal"]) & (sub["price"] <= sub["tft_q90_cal"])
    ) * 100
    bw_cal = np.mean(sub["tft_q90_cal"] - sub["tft_q10_cal"])
    print(f"  {name}")
    print(f"    MAE={mae:.2f} Rs/KG  MAPE={mape:.1f}%  (median unchanged)")
    print(f"    Raw   band:  Coverage={cov_raw:5.1f}%  BandWidth={bw_raw:6.2f}")
    print(f"    CQR   band:  Coverage={cov_cal:5.1f}%  BandWidth={bw_cal:6.2f}")


print("=" * 60)
print("CALIBRATION REPORT (target coverage = 90%)")
print("=" * 60)
print("\n[VAL  — calibration split]")
report("Overall", calibrated[calibrated["split"] == "val"])
for c in sorted(calibrated["commodity"].unique()):
    sub = calibrated[(calibrated["split"] == "val") & (calibrated["commodity"] == c)]
    report(c, sub)

print("\n[TEST — held-out, true generalization]")
report("Overall", calibrated[calibrated["split"] == "test"])
for c in sorted(calibrated["commodity"].unique()):
    sub = calibrated[(calibrated["split"] == "test") & (calibrated["commodity"] == c)]
    report(c, sub)
print("=" * 60)
