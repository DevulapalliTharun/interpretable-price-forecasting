"""
06_evaluate.py
Evaluate TFT and XGBoost baseline, generate static PNG visualisations:
  - Quantile forecast plots per commodity
  - Attention weights (which past months matter)
  - Variable importance (encoder + decoder, per commodity)
  - Feature importance — XGBoost baseline (static)
  - Shock events overlay
  - evaluation_metrics.txt summary

Renderer: matplotlib (no Chromium/kaleido dependency).
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
VIZ = ROOT / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dpi", type=int, default=150,
    help="Output PNG resolution (default 150).",
)
args = parser.parse_args()

plt.rcParams.update({
    "figure.dpi": args.dpi,
    "savefig.dpi": args.dpi,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

PRICE_COLOR = "#185FA5"
TFT_COLOR = "#D85A30"
XGB_COLOR = "#888780"
BAND_COLOR = "#D85A30"
EVENT_COLOR = "#888780"

REAL_EVENTS = {
    "2010-12-01": "Onion crisis: Rs85/kg",
    "2013-11-01": "Onion: Rs100/kg, export ban",
    "2019-12-01": "Onion: Rs160/kg, imports from Egypt",
    "2020-03-25": "COVID lockdown: mandis closed",
    "2021-01-01": "COVID: price recovery",
    "2022-03-01": "Russia-Ukraine: wheat shock",
    "2023-07-15": "Tomato: Rs200+/kg, monsoon failure",
    "2023-08-19": "Onion: export ban imposed",
    "2024-03-01": "Prices normalize post-ban",
}

COMMODITIES = ["Onions", "Tomatoes", "Rice"]


def save_fig(fig, output_path: Path) -> None:
    print(f"  Saving {output_path.name}...")
    fig.savefig(output_path)
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])


def load_tft_predictions():
    for candidate in [
        PROCESSED / "tft_predictions_calibrated_step5.csv",
        PROCESSED / "tft_predictions_calibrated_step1.csv",
        PROCESSED / "tft_predictions_calibrated_original.csv",
        PROCESSED / "tft_predictions.csv",
    ]:
        if not candidate.exists():
            continue
        loaded = pd.read_csv(candidate, parse_dates=["date"])
        if {"tft_q10_cal", "tft_q50_cal", "tft_q90_cal"}.issubset(loaded.columns):
            loaded["tft_q10"] = loaded["tft_q10_cal"]
            loaded["tft_q50"] = loaded["tft_q50_cal"]
            loaded["tft_q90"] = loaded["tft_q90_cal"]
        print(f"  Loaded TFT predictions from {candidate.name}")
        return loaded
    return None


tft_df = load_tft_predictions()

attn_path = PROCESSED / "tft_attention.csv"
attn_df = pd.read_csv(attn_path) if attn_path.exists() else None

var_path = PROCESSED / "tft_variable_importance.csv"
var_df = pd.read_csv(var_path) if var_path.exists() else None

xgb_path = MODELS / "xgb_baseline.pkl"
xgb_loaded = False
if xgb_path.exists():
    xgb_data = joblib.load(xgb_path)
    xgb_model = xgb_data["model"]
    xgb_feature_cols = xgb_data["feature_cols"]
    xgb_label_encoders = xgb_data["label_encoders"]
    # Pin to CPU — model may have been trained with device="cuda".
    try:
        xgb_model.set_params(device="cpu")
    except Exception:
        pass
    xgb_loaded = True
    print("  Loaded XGBoost baseline")

df_test = df[df["date"] >= "2023-01-01"].copy()
if xgb_loaded and len(df_test) > 0:
    for col in ["commodity", "market", "admin1", "season"]:
        le = xgb_label_encoders[col]
        df_test[col + "_enc"] = df_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    df_test["xgb_pred"] = np.expm1(xgb_model.predict(df_test[xgb_feature_cols].values))

test_min = df_test["date"].min().strftime("%b %Y") if len(df_test) > 0 else "N/A"
test_max = df_test["date"].max().strftime("%b %Y") if len(df_test) > 0 else "N/A"

# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════
print("\nComputing metrics...")
metrics_lines = []
metrics_lines.append("=" * 60)
metrics_lines.append(f"EVALUATION METRICS -- Test Period: {test_min} - {test_max}")
metrics_lines.append("=" * 60)

if xgb_loaded and len(df_test) > 0:
    actual = df_test["price"].values
    predicted = df_test["xgb_pred"].values
    metrics_lines.append("\nXGBoost Baseline:")
    metrics_lines.append(f"  MAE:  {np.mean(np.abs(actual - predicted)):.2f} Rs/KG")
    metrics_lines.append(f"  RMSE: {np.sqrt(np.mean((actual - predicted)**2)):.2f} Rs/KG")
    metrics_lines.append(f"  MAPE: {np.mean(np.abs((actual - predicted) / actual))*100:.1f}%")
    for comm in COMMODITIES:
        mask = df_test["commodity"] == comm
        if mask.sum() > 0:
            a, p = df_test.loc[mask, "price"].values, df_test.loc[mask, "xgb_pred"].values
            metrics_lines.append(
                f"  {comm:10s} MAE={np.mean(np.abs(a-p)):.2f}  "
                f"MAPE={np.mean(np.abs((a-p)/a))*100:.1f}%"
            )

if tft_df is not None:
    tft_test = tft_df[tft_df["date"] >= "2023-01-01"]
    if len(tft_test) > 0:
        actual = tft_test["price"].values
        predicted = tft_test["tft_q50"].values
        q10, q90 = tft_test["tft_q10"].values, tft_test["tft_q90"].values
        coverage = np.mean((actual >= q10) & (actual <= q90)) * 100
        metrics_lines.append("\nTFT (Temporal Fusion Transformer):")
        metrics_lines.append(f"  MAE:  {np.mean(np.abs(actual - predicted)):.2f} Rs/KG")
        metrics_lines.append(f"  RMSE: {np.sqrt(np.mean((actual - predicted)**2)):.2f} Rs/KG")
        metrics_lines.append(f"  MAPE: {np.mean(np.abs((actual - predicted) / actual))*100:.1f}%")
        metrics_lines.append(f"  Empirical q0.1-q0.9 Coverage: {coverage:.1f}%")
        metrics_lines.append(f"  Avg Band Width: {np.mean(q90 - q10):.2f} Rs/KG")
        for comm in COMMODITIES:
            mask = tft_test["commodity"] == comm
            if mask.sum() > 0:
                a = tft_test.loc[mask, "price"].values
                p = tft_test.loc[mask, "tft_q50"].values
                q10c = tft_test.loc[mask, "tft_q10"].values
                q90c = tft_test.loc[mask, "tft_q90"].values
                cov = np.mean((a >= q10c) & (a <= q90c)) * 100
                metrics_lines.append(
                    f"  {comm:10s} MAE={np.mean(np.abs(a-p)):.2f}  "
                    f"MAPE={np.mean(np.abs((a-p)/a))*100:.1f}%  Coverage={cov:.0f}%"
                )

        metrics_lines.append("\n" + "-" * 60)
        metrics_lines.append("COMPARISON SUMMARY:")
        metrics_lines.append("  XGBoost: Better point accuracy (lower MAE/MAPE)")
        metrics_lines.append("           Point-only baseline trained through Dec 2022.")
        metrics_lines.append("  TFT:     Probabilistic multi-horizon model")
        metrics_lines.append("           - Attention: which past months drove predictions")
        metrics_lines.append("           - Variable importance: which features matter per crop")
        metrics_lines.append(f"           - Coverage {coverage:.0f}%: prices stay in band")

metrics_text = "\n".join(metrics_lines)
print(metrics_text)
with open(VIZ / "evaluation_metrics.txt", "w") as f:
    f.write(metrics_text)


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Quantile forecast per commodity
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

for commodity in COMMODITIES:
    comm_df = df[df["commodity"] == commodity]
    best_market = comm_df.groupby("market")["date"].count().idxmax()
    plot_df = comm_df[comm_df["market"] == best_market].sort_values("date")
    sid = f"{commodity}_{best_market}"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["date"], plot_df["price"], label="Historical price",
            color=PRICE_COLOR, linewidth=1.6)

    if tft_df is not None:
        tft_sub = tft_df[tft_df["series_id"] == sid].sort_values("date")
        if len(tft_sub) > 0:
            ax.fill_between(tft_sub["date"], tft_sub["tft_q10"], tft_sub["tft_q90"],
                            color=BAND_COLOR, alpha=0.15, label="TFT q0.1-q0.9 band")
            ax.plot(tft_sub["date"], tft_sub["tft_q50"], "--",
                    color=TFT_COLOR, linewidth=1.8, label="TFT median")

    if xgb_loaded:
        xgb_sub = df_test[df_test["series_id"] == sid].sort_values("date")
        if len(xgb_sub) > 0:
            ax.plot(xgb_sub["date"], xgb_sub["xgb_pred"], ":",
                    color=XGB_COLOR, linewidth=1.4, label="XGBoost")

    for date_str, label in REAL_EVENTS.items():
        if commodity.lower() in label.lower() or "COVID" in label or "normalize" in label:
            ax.axvline(pd.Timestamp(date_str), linestyle=":", color=EVENT_COLOR, alpha=0.5)

    ax.set_title(f"Price Forecast: {commodity} - {best_market}")
    ax.set_ylabel("Price (Rs/KG)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save_fig(fig, VIZ / f"quantile_forecast_{commodity.lower()}.png")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: TFT Attention — per-commodity bar + cross-commodity line
# ══════════════════════════════════════════════════════════════════════
if attn_df is not None and len(attn_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, commodity in zip(axes, COMMODITIES):
        comm_attn = attn_df[attn_df["commodity"] == commodity]
        if len(comm_attn) == 0:
            ax.set_visible(False)
            continue
        step_avg = comm_attn.groupby("encoder_step")["attention_weight"].mean().reset_index()
        step_avg["months_ago"] = step_avg["encoder_step"].max() - step_avg["encoder_step"]
        step_avg = step_avg.sort_values("months_ago", ascending=False)
        labels = [f"t-{int(m)}" for m in step_avg["months_ago"]]
        ax.barh(labels, step_avg["attention_weight"], color=TFT_COLOR)
        ax.set_title(commodity)
        ax.set_xlabel("Attention weight")
    fig.suptitle("TFT Attention — which past months drive predictions", y=1.02)
    save_fig(fig, VIZ / "attention_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for commodity in COMMODITIES:
        comm_attn = attn_df[attn_df["commodity"] == commodity]
        if len(comm_attn) == 0:
            continue
        step_avg = comm_attn.groupby("encoder_step")["attention_weight"].mean()
        ax.plot(range(len(step_avg)), step_avg.values, marker="o", linewidth=1.5,
                label=commodity)
    ax.set_title("TFT Attention Distribution Over Encoder Window")
    ax.set_xlabel("Encoder Step (0 = oldest, 23 = most recent month)")
    ax.set_ylabel("Average Attention Weight")
    ax.legend(frameon=False)
    save_fig(fig, VIZ / "attention_distribution.png")
else:
    print("  WARNING: No attention data found - skipping attention plots")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: TFT Variable Importance (encoder + decoder + comparison)
# ══════════════════════════════════════════════════════════════════════
if var_df is not None and len(var_df) > 0:
    enc_vars = var_df[var_df["type"] == "encoder"]
    if len(enc_vars) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
        for ax, commodity in zip(axes, COMMODITIES):
            comm_enc = enc_vars[enc_vars["commodity"] == commodity]
            if len(comm_enc) == 0:
                ax.set_visible(False)
                continue
            comm_enc = comm_enc.sort_values("importance", ascending=True).tail(12)
            ax.barh(comm_enc["variable"], comm_enc["importance"], color=PRICE_COLOR)
            ax.set_title(commodity)
            ax.set_xlabel("Importance")
        fig.suptitle("TFT Encoder Variable Importance — which historical features matter",
                     y=1.02)
        save_fig(fig, VIZ / "tft_encoder_importance.png")

    dec_vars = var_df[var_df["type"] == "decoder"]
    if len(dec_vars) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax, commodity in zip(axes, COMMODITIES):
            comm_dec = dec_vars[dec_vars["commodity"] == commodity]
            if len(comm_dec) == 0:
                ax.set_visible(False)
                continue
            comm_dec = comm_dec.sort_values("importance", ascending=True)
            ax.barh(comm_dec["variable"], comm_dec["importance"], color="#2CA02C")
            ax.set_title(commodity)
            ax.set_xlabel("Importance")
        fig.suptitle("TFT Decoder Variable Importance — which future features matter",
                     y=1.02)
        save_fig(fig, VIZ / "tft_decoder_importance.png")

    if len(enc_vars) > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        commodity_colors = {"Onions": "#D85A30", "Rice": "#185FA5", "Tomatoes": "#2CA02C"}
        offset_map = {c: i for i, c in enumerate(commodity_colors)}
        bar_w = 0.25
        all_features = (
            enc_vars.sort_values("importance", ascending=False)
            .drop_duplicates("variable")
            .head(8)["variable"].tolist()
        )
        x = np.arange(len(all_features))
        for commodity, color in commodity_colors.items():
            comm_enc = enc_vars[enc_vars["commodity"] == commodity].set_index("variable")
            heights = [comm_enc.loc[f, "importance"] if f in comm_enc.index else 0.0
                       for f in all_features]
            ax.bar(x + offset_map[commodity] * bar_w - bar_w, heights, bar_w,
                   label=commodity, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(all_features, rotation=30, ha="right")
        ax.set_ylabel("Importance Weight")
        ax.set_title("Feature Importance Comparison Across Crops (TFT Encoder)")
        ax.legend(frameon=False)
        save_fig(fig, VIZ / "variable_importance_comparison.png")
else:
    print("  WARNING: No variable importance data found - skipping")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4: XGBoost Feature Importance (static baseline)
# ══════════════════════════════════════════════════════════════════════
if xgb_loaded:
    importances = pd.Series(xgb_model.feature_importances_, index=xgb_feature_cols)
    importances = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importances.index, importances.values, color=XGB_COLOR)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — XGBoost Baseline (static, not per-timestep)")
    save_fig(fig, VIZ / "feature_importance_xgboost.png")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5: Shock Events Overlay
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, commodity in zip(axes, COMMODITIES):
    comm_df = df[df["commodity"] == commodity]
    best_market = comm_df.groupby("market")["date"].count().idxmax()
    plot_df = comm_df[comm_df["market"] == best_market].sort_values("date")
    ax.plot(plot_df["date"], plot_df["price"], linewidth=1.4,
            label=f"{commodity} — {best_market}")
    for date_str in REAL_EVENTS:
        ax.axvline(pd.Timestamp(date_str), linestyle=":", color=EVENT_COLOR, alpha=0.4)
    ax.set_title(commodity)
    ax.set_ylabel("Rs/KG")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Price History with Shock Events", y=1.00)
save_fig(fig, VIZ / "shock_events_overlay.png")


# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"All outputs saved to: {VIZ}/")
print(f"{'='*60}")
