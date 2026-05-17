"""
app.py
Streamlit dashboard for WFP India TFT Food Price Forecasting.
Run: streamlit run app.py
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import torch
    from gpu_utils import tft_predict_trainer_kwargs
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
    from tft_utils import (
        extract_raw_prediction_output,
        find_best_checkpoint,
        load_tft_from_checkpoint,
        normalize_prediction_output,
    )

    TFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - UI fallback only
    torch = None
    tft_predict_trainer_kwargs = None
    TimeSeriesDataSet = None
    GroupNormalizer = None
    NaNLabelEncoder = None
    extract_raw_prediction_output = None
    find_best_checkpoint = None
    load_tft_from_checkpoint = None
    TFT_IMPORT_ERROR = str(exc)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _runtime_predict_kwargs():
    if tft_predict_trainer_kwargs is None:
        return None
    try:
        kwargs, _ = tft_predict_trainer_kwargs(1)
        return kwargs
    except Exception:
        return {"accelerator": "cpu", "devices": 1, "precision": "32-true"}


TFT_PREDICT_KWARGS = _runtime_predict_kwargs()


def _fresh_predict_kwargs():
    """Return a shallow copy of TFT_PREDICT_KWARGS for a single predict call.

    pytorch-forecasting mutates the dict it receives (it appends a
    PredictCallback and sets inference_mode=False). Passing the same dict
    twice carries that stale callback into the next call, which in turn
    flips the return type from a Prediction namedtuple to a plain list and
    breaks `extract_raw_prediction_output`. A fresh dict per call avoids it.
    """
    return dict(TFT_PREDICT_KWARGS) if TFT_PREDICT_KWARGS else None

# Pick the best TFT family available at startup:
#   step5    -- XGBoost-fused retrain (MAPE 11.4%, best)
#   step1    -- 2021 retrain (MAPE 29.2%, fair cutoff)
#   original -- 2020 ensemble  (fallback, MAPE 28.3%)
def _select_family() -> str:
    if (MODELS / "tft_best_xgbfused.ckpt").exists() and (PROCESSED / "master_dataset_xgbfused.csv").exists():
        return "step5"
    if (MODELS / "tft_best_2022.ckpt").exists():
        return "step1"
    return "original"

TFT_FAMILY = _select_family()
TRAIN_CUTOFF = pd.Timestamp("2021-12-31") if TFT_FAMILY in ("step1", "step5") else pd.Timestamp("2020-12-31")
EXTRA_KNOWN_REALS = ["xgb_log_pred"] if TFT_FAMILY == "step5" else []

# Human-readable feature name mapping for VSN explanations
FEATURE_EXPLANATIONS = {
    "rain_deficit": ("Low rainfall", "Below-normal rainfall detected — historically linked to supply shortages"),
    "rain_excess": ("Excess rainfall", "Above-normal rainfall — may cause crop damage or transport disruption"),
    "heat_stress": ("Extreme heat", "Temperature exceeded 38C — causes crop stress and yield reduction"),
    "cold_stress": ("Cold stress", "Temperature below 10C — delays germination and harvest"),
    "rainfall_monthly": ("Monsoon pattern", "Monthly rainfall pattern influencing crop supply cycle"),
    "temperature_mean": ("Temperature", "Temperature conditions affecting crop growth"),
    "humidity_mean": ("Humidity", "Humidity levels influencing crop health and storage"),
    "price_lag_1m": ("Price momentum", "Last month's price — indicates ongoing trend continuation"),
    "price_lag_12m": ("Annual baseline", "Same month last year — seasonal price reference"),
    "rolling_3m": ("3-month trend", "Short-term price trend over last 3 months"),
    "rolling_6m": ("6-month trend", "Medium-term price trend over last 6 months"),
    "yoy_change": ("Year-over-year acceleration", "Price change rate vs last year — signals building pressure"),
    "log_price": ("Current price level", "The price level itself as context"),
    "season": ("Agricultural season", "Kharif (monsoon) / Rabi (winter) / Zaid (summer) harvest timing"),
    "month_sin": ("Seasonal cycle", "Position in the annual seasonal cycle"),
    "month_cos": ("Seasonal cycle", "Position in the annual seasonal cycle"),
    "covid_lockdown": ("COVID lockdown", "Mandi closures and supply chain disruption"),
    "time_idx": ("Time trend", "Long-term price trend over years"),
    "year": ("Year effect", "Inflation and long-term structural changes"),
    "month": ("Month effect", "Monthly seasonal pattern"),
    "relative_time_idx": ("Forecast position", "How far into the future this prediction is"),
}

REAL_EVENTS = {
    "2010-12-01": "Onion crisis: Rs85/kg",
    "2013-11-01": "Onion: Rs100/kg, export ban",
    "2019-12-01": "Onion: Rs160/kg, imports",
    "2020-03-25": "COVID lockdown",
    "2023-07-15": "Tomato: Rs200+/kg",
    "2023-08-19": "Onion: export ban",
    "2024-03-01": "Prices normalize",
}

st.set_page_config(page_title="Food Price Forecasting — TFT", layout="wide")


def get_season(month: int) -> str:
    if month in [7, 8, 9, 10]:
        return "Kharif"
    if month in [11, 12, 1, 2]:
        return "Rabi"
    return "Zaid"


@st.cache_data
def load_data():
    # step5 family needs the fused dataset (contains xgb_log_pred column).
    if TFT_FAMILY == "step5":
        fused = PROCESSED / "master_dataset_xgbfused.csv"
        if fused.exists():
            return pd.read_csv(fused, parse_dates=["date"])
    path = PROCESSED / "master_dataset.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


def _load_xgb_bundle(path: Path):
    if not path.exists():
        return None
    bundle = joblib.load(path)
    # Models were trained with device="cuda" for speed; predicting from the
    # Streamlit process happens on CPU arrays, so pin the booster to CPU to
    # avoid the per-call "Falling back to prediction using DMatrix" warning
    # and the device-transfer cost it triggers.
    try:
        bundle["model"].set_params(device="cpu")
    except Exception:
        pass
    return bundle


@st.cache_resource
def load_xgb():
    return _load_xgb_bundle(MODELS / "xgb_baseline.pkl")


@st.cache_resource
def load_xgb_clean():
    """The pre-2020 'clean' XGB used for the xgb_log_pred feature in step5."""
    return _load_xgb_bundle(MODELS / "xgb_clean_2019.pkl")


@st.cache_data
def load_tft_predictions():
    candidates = []
    if TFT_FAMILY == "step5":
        candidates.append(PROCESSED / "tft_predictions_calibrated_step5.csv")
    elif TFT_FAMILY == "step1":
        candidates.append(PROCESSED / "tft_predictions_calibrated_step1.csv")
    else:
        candidates.append(PROCESSED / "tft_predictions_calibrated_original.csv")
    candidates.append(PROCESSED / "tft_predictions.csv")

    for path in candidates:
        if not path.exists():
            continue
        loaded = pd.read_csv(path, parse_dates=["date"])
        if {"tft_q10_cal", "tft_q50_cal", "tft_q90_cal"}.issubset(loaded.columns):
            loaded["tft_q10"] = loaded["tft_q10_cal"]
            loaded["tft_q50"] = loaded["tft_q50_cal"]
            loaded["tft_q90"] = loaded["tft_q90_cal"]
        return loaded
    return None


@st.cache_data
def load_metrics():
    path = ROOT / "visualizations" / "evaluation_metrics.txt"
    if path.exists():
        return path.read_text()
    return None


@st.cache_resource
def build_tft_training_dataset():
    if TFT_IMPORT_ERROR:
        return None

    df = load_data()
    if df is None:
        return None

    base = df[df["date"] <= TRAIN_CUTOFF].copy()
    for col in ["commodity", "market", "admin1", "season"]:
        base[col] = base[col].astype(str)

    return TimeSeriesDataSet(
        base,
        time_idx="time_idx",
        target="log_price",
        group_ids=["series_id"],
        max_encoder_length=24,
        min_encoder_length=12,
        max_prediction_length=6,

        min_prediction_length=1,
        static_categoricals=["commodity", "market", "admin1"],
        static_reals=[],
        time_varying_known_categoricals=["season"],
        time_varying_known_reals=(
            ["time_idx", "year", "month", "month_sin", "month_cos", "covid_lockdown"]
            + EXTRA_KNOWN_REALS
        ),
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


@st.cache_resource
def load_tft_runtime():
    if TFT_IMPORT_ERROR:
        return {"model": None, "training": None, "error": TFT_IMPORT_ERROR, "ckpt": None}

    training = build_tft_training_dataset()
    if training is None:
        return {"model": None, "training": None, "error": "Training dataset unavailable", "ckpt": None}

    best_ckpt = find_best_checkpoint(MODELS, family=TFT_FAMILY)
    if best_ckpt is None:
        return {"model": None, "training": training, "error": f"No TFT checkpoint found for family '{TFT_FAMILY}'", "ckpt": None}

    try:
        model, quantiles = load_tft_from_checkpoint(training, best_ckpt)
        return {
            "model": model,
            "training": training,
            "error": None,
            "ckpt": best_ckpt.name,
            "quantiles": quantiles,
        }
    except Exception as exc:  # pragma: no cover - depends on local runtime/GPU
        return {
            "model": None,
            "training": training,
            "error": str(exc),
            "ckpt": best_ckpt.name,
            "quantiles": None,
        }


def get_xgb_predictions(data, xgb_data):
    pred_df = data.copy()
    for col in ["commodity", "market", "admin1", "season"]:
        le = xgb_data["label_encoders"][col]
        pred_df[col + "_enc"] = pred_df[col].astype(str).apply(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )
    X = pred_df[xgb_data["feature_cols"]].values
    pred_df["xgb_pred"] = np.expm1(xgb_data["model"].predict(X))
    return pred_df


def build_next_feature_row(history_df):
    history = history_df.sort_values("date").copy()
    last = history.iloc[-1].copy()
    next_date = (last["date"] + pd.offsets.DateOffset(months=1)).replace(day=15)
    month = int(next_date.month)

    row = last.copy()
    row["date"] = next_date
    row["year"] = next_date.year
    row["month"] = month
    row["time_idx"] = int(history["time_idx"].max()) + 1
    row["month_sin"] = math.sin(2 * math.pi * month / 12)
    row["month_cos"] = math.cos(2 * math.pi * month / 12)
    row["season"] = get_season(month)
    row["covid_lockdown"] = int(
        pd.Timestamp("2020-03-15") <= next_date <= pd.Timestamp("2020-09-15")
    )

    climate = history.groupby("month")[["temperature_mean", "rainfall_monthly", "humidity_mean"]].mean()
    if month in climate.index:
        climate_row = climate.loc[month]
    else:
        climate_row = history[["temperature_mean", "rainfall_monthly", "humidity_mean"]].mean()

    row["temperature_mean"] = float(climate_row["temperature_mean"])
    row["rainfall_monthly"] = float(climate_row["rainfall_monthly"])
    row["humidity_mean"] = float(climate_row["humidity_mean"])
    row["rain_deficit"] = int(row["rainfall_monthly"] < 50)
    row["rain_excess"] = int(row["rainfall_monthly"] > 400)
    row["heat_stress"] = int(row["temperature_mean"] > 38)
    row["cold_stress"] = int(row["temperature_mean"] < 10)

    row["price_lag_1m"] = float(history["log_price"].iloc[-1])
    row["price_lag_12m"] = float(history["log_price"].iloc[-12]) if len(history) >= 12 else float(history["log_price"].iloc[0])
    row["rolling_3m"] = float(history["log_price"].tail(3).mean())
    row["rolling_6m"] = float(history["log_price"].tail(6).mean())
    prev12 = row["price_lag_12m"]
    row["yoy_change"] = 0.0 if prev12 == 0 else float((row["price_lag_1m"] - prev12) / abs(prev12))
    row["log_price"] = float(row["price_lag_1m"])
    row["price"] = float(np.expm1(row["log_price"]))

    # For step5 family we also need xgb_log_pred on this future row.
    if TFT_FAMILY == "step5":
        xgb_clean = load_xgb_clean()
        if xgb_clean is not None:
            pred_df = pd.DataFrame([row]).copy()
            for c in ["commodity", "market", "admin1", "season"]:
                le = xgb_clean["label_encoders"][c]
                pred_df[c + "_enc"] = pred_df[c].astype(str).apply(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
            X = pred_df[xgb_clean["feature_cols"]].values
            row["xgb_log_pred"] = float(xgb_clean["model"].predict(X)[0])
        else:
            row["xgb_log_pred"] = float(row["log_price"])
    return row


def predict_xgb_one_step(next_row, xgb_data):
    if xgb_data is None:
        return None

    pred_df = pd.DataFrame([next_row]).copy()
    for col in ["commodity", "market", "admin1", "season"]:
        le = xgb_data["label_encoders"][col]
        pred_df[col + "_enc"] = pred_df[col].astype(str).apply(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )
    X = pred_df[xgb_data["feature_cols"]].values
    pred_log = float(xgb_data["model"].predict(X)[0])
    return {
        "xgb_log": pred_log,
        "xgb_price": float(np.expm1(pred_log)),
    }


MAX_PREDICTION_LENGTH = 6


def _quantile_indices(runtime):
    quantiles = list(runtime.get("quantiles") or [0.1, 0.5, 0.9])
    return quantiles.index(0.1), quantiles.index(0.5), quantiles.index(0.9)


def _to_3d_quantiles(preds):
    """Coerce model.predict(mode='quantiles') output to a [B, T, Q] tensor.

    pytorch-forecasting may return a Tensor or a list of per-batch tensors,
    depending on the lightning trainer settings. Older paths can also yield a
    1-batch Tensor with the leading dim missing. This helper returns a
    well-shaped [batch, horizon, num_quantiles] tensor or raises if the
    structure is genuinely empty.
    """
    if preds is None:
        raise ValueError("Model returned no predictions")
    if isinstance(preds, (list, tuple)):
        non_empty = [p for p in preds if hasattr(p, "shape") and p.numel() > 0]
        if not non_empty:
            raise ValueError("Predict returned an empty batch list")
        preds = torch.cat([p.detach().cpu() for p in non_empty], dim=0)
    else:
        preds = preds.detach().cpu()
    if preds.ndim == 2:  # [T, Q] from a single-window predict — add batch dim
        preds = preds.unsqueeze(0)
    if preds.ndim != 3:
        raise ValueError(f"Unexpected prediction shape {tuple(preds.shape)}")
    return preds


def predict_tft_horizon(history_df, runtime, horizon):
    """Predict the next ``horizon`` months using the TFT in a single forward pass.

    The model was trained for ``max_prediction_length=6``, so we append
    ``horizon`` placeholder future rows and read all six quantile forecasts
    from one ``predict`` call. Autoregressing one-step-at-a-time both wastes
    the multi-horizon training signal and triggers a shape inconsistency in
    pytorch-forecasting after the first append (the cause of the prior "only
    one orange dot" bug).
    """
    if runtime["model"] is None or runtime["training"] is None:
        return None
    horizon = max(1, min(int(horizon), MAX_PREDICTION_LENGTH))

    working = history_df.sort_values("date").copy()
    future_rows = []
    # Build placeholder future rows. log_price/price values are ignored by
    # TFT for decoder steps (target is unknown), but they must exist and be
    # NaN-free for the dataset builder.
    for _ in range(horizon):
        nxt = build_next_feature_row(working)
        future_rows.append(nxt)
        working = pd.concat([working, pd.DataFrame([nxt])], ignore_index=True)

    pred_df = working.copy()
    for col in ["commodity", "market", "admin1", "season"]:
        pred_df[col] = pred_df[col].astype(str)

    dataset = TimeSeriesDataSet.from_dataset(
        runtime["training"], pred_df, predict=True, stop_randomization=True
    )
    dataloader = dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0, shuffle=False
    )

    preds = _to_3d_quantiles(
        runtime["model"].predict(
            dataloader, mode="quantiles", trainer_kwargs=_fresh_predict_kwargs()
        )
    )
    q10_idx, q50_idx, q90_idx = _quantile_indices(runtime)

    # The model emits MAX_PREDICTION_LENGTH steps; the LAST `horizon` of those
    # correspond to the future months we appended, in order.
    out_steps = preds.shape[1]
    take = preds[0, out_steps - horizon:, :]
    records = []
    for i, row in enumerate(future_rows):
        q10 = float(np.expm1(take[i, q10_idx].item()))
        q50 = float(np.expm1(take[i, q50_idx].item()))
        q90 = float(np.expm1(take[i, q90_idx].item()))
        records.append({
            "date": row["date"],
            "tft_q10": q10,
            "tft_q50": q50,
            "tft_q90": q90,
            "band_width": q90 - q10,
        })
    return pd.DataFrame(records)


def compute_future_forecasts(plot_df, xgb_data, runtime, horizon):
    if len(plot_df) < 12:
        return pd.DataFrame()
    horizon = max(1, min(int(horizon), MAX_PREDICTION_LENGTH))

    base = plot_df.sort_values("date").copy()
    last_date = base["date"].max()
    future_dates = [
        (last_date + pd.offsets.DateOffset(months=step + 1)).replace(day=15)
        for step in range(horizon)
    ]
    records = [{"date": d} for d in future_dates]

    # XGBoost cannot forecast multi-step natively — autoregress one step at a
    # time, feeding each prediction back as the next month's price lag.
    if xgb_data is not None:
        xgb_history = base.copy()
        for i in range(horizon):
            xgb_row = build_next_feature_row(xgb_history)
            xgb_pred = predict_xgb_one_step(xgb_row, xgb_data)
            if xgb_pred is None:
                break
            records[i]["xgb_pred"] = xgb_pred["xgb_price"]
            xgb_row["log_price"] = xgb_pred["xgb_log"]
            xgb_row["price"] = xgb_pred["xgb_price"]
            xgb_history = pd.concat([xgb_history, pd.DataFrame([xgb_row])], ignore_index=True)

    # TFT predicts the whole horizon in one call.
    tft_horizon = predict_tft_horizon(base, runtime, horizon)
    if tft_horizon is not None and len(tft_horizon) > 0:
        for i in range(min(len(tft_horizon), horizon)):
            row = tft_horizon.iloc[i]
            records[i]["tft_q10"] = row["tft_q10"]
            records[i]["tft_q50"] = row["tft_q50"]
            records[i]["tft_q90"] = row["tft_q90"]
            records[i]["band_width"] = row["band_width"]

    return pd.DataFrame(records)


def compute_tft_interpretation(plot_df, runtime):
    """Run the TFT once and extract attention + variable-selection weights.

    Returns a dict on success or {"error": str} on failure so the UI can
    render the actual reason instead of a generic "checkpoint not loaded"
    message. Returns None only when the runtime/training pieces are missing.
    """
    if runtime["model"] is None or runtime["training"] is None:
        return None
    if len(plot_df) < 18:
        return {"error": f"Need >=18 months of history (have {len(plot_df)})"}

    series_df = plot_df.sort_values("date").copy()
    # Append MAX_PREDICTION_LENGTH placeholder rows so the predict=True window
    # ALWAYS has all 6 decoder steps as future months (not historical ones),
    # which keeps the encoder = the real last 24 historical months.
    working = series_df.copy()
    for _ in range(MAX_PREDICTION_LENGTH):
        nxt = build_next_feature_row(working)
        working = pd.concat([working, pd.DataFrame([nxt])], ignore_index=True)

    pred_df = working
    for col in ["commodity", "market", "admin1", "season"]:
        pred_df[col] = pred_df[col].astype(str)

    dataset = TimeSeriesDataSet.from_dataset(
        runtime["training"], pred_df, predict=True, stop_randomization=True
    )
    dataloader = dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0, shuffle=False
    )
    try:
        raw = runtime["model"].predict(
            dataloader, mode="raw", return_x=True, trainer_kwargs=_fresh_predict_kwargs()
        )
        interp = runtime["model"].interpret_output(
            extract_raw_prediction_output(raw), reduction="sum"
        )
    except Exception as exc:  # surface the real reason
        return {"error": f"TFT interpret failed: {type(exc).__name__}: {exc}"}

    attention = interp["attention"].detach().cpu().numpy()
    encoder_weights = interp["encoder_variables"].detach().cpu().numpy()
    decoder_weights = interp["decoder_variables"].detach().cpu().numpy()

    encoder_df = pd.DataFrame({
        "feature": runtime["model"].encoder_variables,
        "weight": encoder_weights,
    }).sort_values("weight", ascending=False)

    decoder_df = pd.DataFrame({
        "feature": runtime["model"].decoder_variables,
        "weight": decoder_weights,
    }).sort_values("weight", ascending=False)

    # The encoder window is the LAST `len(attention)` historical months
    # (NOT plot_df.tail — that included the appended placeholder rows in the
    # earlier buggy version). series_df is purely historical, so its tail
    # gives the calendar dates the model actually attended to.
    attention_dates = (
        series_df.tail(len(attention))["date"].reset_index(drop=True)
    )
    attention_df = pd.DataFrame({
        "date": attention_dates,
        "attention": attention,
    })

    return {
        "encoder": encoder_df,
        "decoder": decoder_df,
        "attention": attention_df,
    }


def _series_cache_key(plot_df, runtime):
    if plot_df is None or len(plot_df) == 0:
        return None
    return (
        str(plot_df["series_id"].iloc[0]),
        int(len(plot_df)),
        pd.Timestamp(plot_df["date"].max()).isoformat(),
        runtime.get("ckpt") if isinstance(runtime, dict) else None,
    )


def get_cached_tft_interpretation(plot_df, runtime):
    cache_key = _series_cache_key(plot_df, runtime)
    if cache_key is None:
        return None

    state_key = ("tft_interp", cache_key)
    if st.session_state.get("_tft_interp_key") != state_key:
        with st.spinner("Loading TFT explainability..."):
            st.session_state["_tft_interp_value"] = compute_tft_interpretation(plot_df, runtime)
        st.session_state["_tft_interp_key"] = state_key
    return st.session_state.get("_tft_interp_value")


def get_cached_future_forecasts(plot_df, xgb_data, runtime, horizon):
    cache_key = _series_cache_key(plot_df, runtime)
    if cache_key is None:
        return pd.DataFrame()

    state_key = ("future_df", cache_key, int(horizon))
    if st.session_state.get("_future_df_key") != state_key:
        with st.spinner("Generating future forecast..."):
            st.session_state["_future_df_value"] = compute_future_forecasts(
                plot_df, xgb_data, runtime, horizon
            )
        st.session_state["_future_df_key"] = state_key
    return st.session_state.get("_future_df_value", pd.DataFrame())


def detect_spikes(price_series, threshold=0.25):
    """Detect price spikes (month-over-month change > threshold)."""
    spikes = []
    prices = price_series.sort_values("date")
    for i in range(1, len(prices)):
        prev_price = prices.iloc[i - 1]["price"]
        curr_price = prices.iloc[i]["price"]
        if prev_price > 0:
            change = (curr_price - prev_price) / prev_price
            if abs(change) > threshold:
                spikes.append({
                    "date": prices.iloc[i]["date"],
                    "price": curr_price,
                    "prev_price": prev_price,
                    "change_pct": change * 100,
                    "direction": "SPIKE" if change > 0 else "DROP",
                })
    return spikes


def _interp_ok(tft_interp):
    return isinstance(tft_interp, dict) and "encoder" in tft_interp


def get_model_reasons(tft_interp, top_n=4):
    """Convert VSN encoder weights to human-readable reasons."""
    if not _interp_ok(tft_interp):
        return []
    enc = tft_interp["encoder"]
    reasons = []
    total = enc["weight"].sum()
    for _, row in enc.head(top_n).iterrows():
        feat = row["feature"]
        weight = row["weight"]
        pct = (weight / total * 100) if total > 0 else 0
        name, explanation = FEATURE_EXPLANATIONS.get(feat, (feat, ""))
        reasons.append({
            "feature": feat,
            "name": name,
            "weight": weight,
            "pct": pct,
            "explanation": explanation,
        })
    return reasons


def get_future_risk_level(band_width, median_price):
    """Classify risk based on band width relative to price.

    Returns (label, hex_color, streamlit_named_color). The named color is used
    for Streamlit's `:color[text]` markdown which only accepts named colors.
    """
    if median_price <= 0:
        return "UNKNOWN", "#888780", "gray"
    ratio = band_width / median_price
    if ratio > 0.6:
        return "HIGH RISK", "#DC3545", "red"
    elif ratio > 0.3:
        return "MODERATE", "#FFC107", "orange"
    else:
        return "STABLE", "#28A745", "green"


def get_decoder_reasons(tft_interp, future_date):
    """Get reasons for future predictions from decoder weights."""
    if not _interp_ok(tft_interp):
        return []
    dec = tft_interp["decoder"]
    reasons = []
    month = future_date.month
    season = get_season(month)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for _, row in dec.iterrows():
        feat = row["feature"]
        weight = row["weight"]
        name, _ = FEATURE_EXPLANATIONS.get(feat, (feat, ""))
        if feat == "season":
            reasons.append(f"Season: {season} — {name} (weight: {weight:.2f})")
        elif feat in ("month", "month_sin", "month_cos"):
            reasons.append(f"Month: {months[month-1]} pattern (weight: {weight:.2f})")
        elif feat == "covid_lockdown":
            if weight > 0.05:
                reasons.append(f"COVID lockdown flag active (weight: {weight:.2f})")
        elif weight > 0.05:
            reasons.append(f"{name} (weight: {weight:.2f})")
    return reasons[:4]


def search_news(commodity, market):
    """Search for recent news about this commodity/market."""
    try:
        from gnews import GNews
        gn = GNews(language="en", country="IN", max_results=3)
        query = f"{commodity} price {market} India"
        articles = gn.get_news(query)
        results = []
        for a in articles[:3]:
            results.append({
                "title": a.get("title", ""),
                "source": a.get("publisher", {}).get("title", ""),
                "date": a.get("published date", ""),
                "url": a.get("url", ""),
            })
        return results
    except Exception:
        return []


df = load_data()
xgb_data = load_xgb()
tft_preds = load_tft_predictions()
tft_runtime = load_tft_runtime()

if df is None:
    st.error("Master dataset not found. Run the pipeline scripts first (00-02).")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("Controls")

commodities = sorted(df["commodity"].unique())
selected_commodity = st.sidebar.radio("Select Crop", commodities)

markets = sorted(df[df["commodity"] == selected_commodity]["market"].unique())
selected_market = st.sidebar.selectbox("Select Market", markets)

forecast_horizon = st.sidebar.slider("Future horizon (months)", min_value=1, max_value=6, value=6)
show_events = st.sidebar.checkbox("Annotate events", value=True)
show_xgb = st.sidebar.checkbox("Show XGBoost", value=True)
show_tft = st.sidebar.checkbox("Show TFT", value=True)
show_band = st.sidebar.checkbox("Show confidence band", value=True)

# ── Filter data ───────────────────────────────────────────────────────
series_id = f"{selected_commodity}_{selected_market}"
plot_df = df[df["series_id"] == series_id].sort_values("date").copy()

xgb_preds = None
if xgb_data is not None and len(plot_df) > 0:
    xgb_preds = get_xgb_predictions(plot_df, xgb_data)

tft_series = None
if tft_preds is not None:
    tft_series = tft_preds[tft_preds["series_id"] == series_id].sort_values("date")

st.title(f"Price Forecast: {selected_commodity} — {selected_market}")

col1, col2, col3, col4 = st.columns(4)
if len(plot_df) > 0:
    col1.metric("Current Price", f"Rs {plot_df['price'].iloc[-1]:.1f}/KG")
    col2.metric("Max Price", f"Rs {plot_df['price'].max():.1f}/KG")
    col3.metric("Min Price", f"Rs {plot_df['price'].min():.1f}/KG")
    col4.metric("Months of Data", f"{len(plot_df)}")

selected_view = st.radio(
    "View",
    [
        "Price Forecast",
        "Future Forecast",
        "Model Explainability",
        "Present vs Predicted",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

future_df = None
tft_interp = None
if selected_view == "Price Forecast":
    tft_interp = get_cached_tft_interpretation(plot_df, tft_runtime)
elif selected_view == "Future Forecast":
    future_df = get_cached_future_forecasts(plot_df, xgb_data, tft_runtime, forecast_horizon)
    tft_interp = get_cached_tft_interpretation(plot_df, tft_runtime)
elif selected_view == "Model Explainability":
    tft_interp = get_cached_tft_interpretation(plot_df, tft_runtime)

# ── TAB 1: Price Forecast ─────────────────────────────────────────────
if selected_view == "Price Forecast":
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["price"],
        mode="lines", name="Historical price",
        line=dict(color="#185FA5", width=2),
    ))

    if show_tft and tft_series is not None and len(tft_series) > 0:
        if show_band:
            fig.add_trace(go.Scatter(
                x=tft_series["date"], y=tft_series["tft_q90"],
                fill=None, mode="lines",
                line=dict(color="rgba(216,90,48,0)"),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=tft_series["date"], y=tft_series["tft_q10"],
                fill="tonexty", mode="lines",
                fillcolor="rgba(216,90,48,0.15)",
                line=dict(color="rgba(216,90,48,0)"),
                name="TFT q0.1-q0.9 band",
            ))
        fig.add_trace(go.Scatter(
            x=tft_series["date"], y=tft_series["tft_q50"],
            mode="lines", name="TFT median forecast",
            line=dict(color="#D85A30", width=2, dash="dash"),
        ))

    if show_xgb and xgb_preds is not None:
        fig.add_trace(go.Scatter(
            x=xgb_preds["date"], y=xgb_preds["xgb_pred"],
            mode="lines", name="XGBoost prediction",
            line=dict(color="#888780", width=1.5, dash="dot"),
        ))

    if show_events:
        for date_str, label in REAL_EVENTS.items():
            fig.add_vline(x=date_str, line_dash="dot", line_color="#888780", opacity=0.5)
            fig.add_annotation(
                x=date_str, y=1.05, yref="paper",
                text=label, showarrow=False,
                font=dict(size=9, color="#5F5E5A"), textangle=-45,
            )

    fig.update_layout(
        yaxis_title="Price (Rs/KG)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=550,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Spike Detection with Model-Driven Reasons ──────────────────
    spikes = detect_spikes(plot_df, threshold=0.25)
    if spikes:
        st.subheader("Detected Price Spikes (Auto-detected, not hardcoded)")
        reasons = get_model_reasons(tft_interp)

        for spike in spikes[-5:]:  # Show last 5 spikes
            direction_icon = "+" if spike["direction"] == "SPIKE" else "-"
            spike_color = "red" if spike["direction"] == "SPIKE" else "blue"

            with st.expander(
                f"{spike['date'].strftime('%b %Y')} | Rs {spike['price']:.0f}/KG "
                f"({direction_icon}{abs(spike['change_pct']):.0f}% from Rs {spike['prev_price']:.0f})",
                expanded=False,
            ):
                st.markdown(f"**{spike['direction']} DETECTED** | "
                            f"Price moved from Rs {spike['prev_price']:.1f} to Rs {spike['price']:.1f}/KG")

                if reasons:
                    st.markdown("**Model-identified drivers (from TFT Variable Selection Network):**")
                    for r in reasons:
                        bar_len = int(r["pct"] / 2)
                        bar = "█" * bar_len
                        st.markdown(f"`{bar}` **{r['name']}** ({r['pct']:.0f}%)")
                        st.caption(f"  {r['explanation']}")
                else:
                    st.caption("Train TFT and run script 05 for model-driven explanations.")

                # News search for this spike
                if st.button(f"Search news for {spike['date'].strftime('%b %Y')}", key=f"news_{spike['date']}"):
                    with st.spinner("Searching..."):
                        articles = search_news(selected_commodity, selected_market)
                    if articles:
                        for a in articles:
                            st.markdown(f"- **{a['title']}** ({a['source']})")
                    else:
                        st.caption("No articles found.")

    st.subheader("Price Statistics")
    stats_data = []
    for comm in df["commodity"].unique():
        comm_df = df[df["commodity"] == comm]
        stats_data.append({
            "Commodity": comm,
            "Markets": comm_df["market"].nunique(),
            "Avg Price (Rs)": f"{comm_df['price'].mean():.1f}",
            "Max Price (Rs)": f"{comm_df['price'].max():.1f}",
            "Std Dev": f"{comm_df['price'].std():.1f}",
            "Date Range": f"{comm_df['date'].min().strftime('%Y-%m')} to {comm_df['date'].max().strftime('%Y-%m')}",
        })
    st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)

# ── TAB 2: Future Forecast ────────────────────────────────────────────
if selected_view == "Future Forecast":
    st.subheader(f"Forward View: Next {forecast_horizon} Months")

    if len(future_df) == 0:
        st.warning("Future forecast could not be generated for this series.")
    else:
        recent_history = plot_df.tail(24)
        fig_future = go.Figure()

        fig_future.add_trace(go.Scatter(
            x=recent_history["date"], y=recent_history["price"],
            mode="lines+markers", name="Recent actual price",
            line=dict(color="#185FA5", width=2),
        ))

        if show_tft and {"tft_q10", "tft_q50", "tft_q90"}.issubset(future_df.columns):
            if show_band:
                fig_future.add_trace(go.Scatter(
                    x=future_df["date"], y=future_df["tft_q90"],
                    fill=None, mode="lines",
                    line=dict(color="rgba(216,90,48,0)"),
                    showlegend=False,
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_df["date"], y=future_df["tft_q10"],
                    fill="tonexty", mode="lines",
                    fillcolor="rgba(216,90,48,0.18)",
                    line=dict(color="rgba(216,90,48,0)"),
                    name="TFT q0.1-q0.9 band",
                ))
            fig_future.add_trace(go.Scatter(
                x=future_df["date"], y=future_df["tft_q50"],
                mode="lines+markers", name="TFT median forecast",
                line=dict(color="#D85A30", width=2, dash="dash"),
            ))

        if show_xgb and "xgb_pred" in future_df.columns:
            fig_future.add_trace(go.Scatter(
                x=future_df["date"], y=future_df["xgb_pred"],
                mode="lines+markers", name="XGBoost forecast",
                line=dict(color="#888780", width=1.5, dash="dot"),
            ))

        fig_future.add_vline(
            x=plot_df["date"].max(), line_dash="dash",
            line_color="#5F5E5A", opacity=0.5,
        )
        fig_future.update_layout(
            yaxis_title="Price (Rs/KG)", hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
        )
        st.plotly_chart(fig_future, width="stretch")

        # ── Per-month forecast cards with risk levels ─────────────────
        st.subheader("Monthly Forecast Breakdown")
        months_label = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        has_tft = {"tft_q10", "tft_q50", "tft_q90", "band_width"}.issubset(future_df.columns)

        for idx, row in future_df.iterrows():
            fdate = pd.Timestamp(row["date"])
            month_name = months_label[fdate.month - 1]
            season = get_season(fdate.month)

            cols = st.columns([1, 2, 3])

            # Column 1: Date and season
            cols[0].markdown(f"**{month_name} {fdate.year}**")
            cols[0].caption(f"Season: {season}")

            # Column 2: Price predictions
            if has_tft:
                q10, q50, q90 = row["tft_q10"], row["tft_q50"], row["tft_q90"]
                bw = row["band_width"]
                risk, _, risk_named = get_future_risk_level(bw, q50)
                cols[1].metric("TFT Median", f"Rs {q50:.1f}/KG",
                               delta=f"Band: Rs {q10:.0f} - Rs {q90:.0f}")
                cols[1].markdown(f":{risk_named}[**{risk}**] (uncertainty: Rs {bw:.1f})")
            if "xgb_pred" in row.index and pd.notna(row["xgb_pred"]):
                cols[1].caption(f"XGBoost: Rs {row['xgb_pred']:.1f}/KG")

            # Column 3: Model-driven reasons
            reasons = get_decoder_reasons(tft_interp, fdate)
            if reasons:
                cols[2].markdown("**Why this forecast:**")
                for r in reasons:
                    cols[2].caption(f"- {r}")
            else:
                cols[2].caption("Enable TFT for model-driven explanations")

            st.divider()

        # ── Related News ──────────────────────────────────────────────
        st.subheader("Related News (Live Search)")
        if st.button("Search latest news", key="news_future"):
            with st.spinner("Searching..."):
                articles = search_news(selected_commodity, selected_market)
            if articles:
                for a in articles:
                    st.markdown(f"**{a['title']}**")
                    st.caption(f"{a['source']} | {a['date']}")
            else:
                st.info("No recent news found. Try a major market like Delhi or Mumbai.")

        if tft_runtime["ckpt"]:
            st.caption(f"TFT checkpoint: {tft_runtime['ckpt']}")
        if tft_runtime["error"]:
            st.caption(f"TFT note: {tft_runtime['error']}")

# ── TAB 3: Model Explainability ───────────────────────────────────────
if selected_view == "Model Explainability":
    top_left, top_right = st.columns(2)

    with top_left:
        st.subheader("Feature Importance — XGBoost")
        if xgb_data is not None:
            importances = pd.Series(
                xgb_data["model"].feature_importances_,
                index=xgb_data["feature_cols"],
            ).sort_values(ascending=True).tail(12)

            fig_imp = go.Figure(go.Bar(
                x=importances.values, y=importances.index,
                orientation="h", marker_color="#D85A30",
            ))
            fig_imp.update_layout(
                xaxis_title="Importance", template="plotly_white", height=400,
            )
            st.plotly_chart(fig_imp, width="stretch")
        else:
            st.info("Train XGBoost baseline first (script 04).")

    with top_right:
        st.subheader("TFT Variable Weights")
        if _interp_ok(tft_interp):
            encoder_top = tft_interp["encoder"].head(12).sort_values("weight", ascending=True)
            fig_tft_vars = go.Figure(go.Bar(
                x=encoder_top["weight"],
                y=encoder_top["feature"],
                orientation="h",
                marker_color="#185FA5",
            ))
            fig_tft_vars.update_layout(
                xaxis_title="Selection weight", template="plotly_white", height=400,
            )
            st.plotly_chart(fig_tft_vars, width="stretch")
        elif isinstance(tft_interp, dict) and "error" in tft_interp:
            st.warning(tft_interp["error"])
        else:
            st.info("TFT explainability becomes available when the checkpoint loads successfully.")

    if _interp_ok(tft_interp):
        mid_left, mid_right = st.columns(2)

        with mid_left:
            st.subheader("TFT Attention Over Past Months")
            fig_attn = go.Figure(go.Bar(
                x=tft_interp["attention"]["date"],
                y=tft_interp["attention"]["attention"],
                marker_color="rgba(24,95,165,0.65)",
                name="Attention",
            ))
            fig_attn.update_layout(
                yaxis_title="Attention weight",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_attn, width="stretch")

        with mid_right:
            st.subheader("TFT Decoder Weights")
            decoder_top = tft_interp["decoder"].sort_values("weight", ascending=True)
            fig_decoder = go.Figure(go.Bar(
                x=decoder_top["weight"],
                y=decoder_top["feature"],
                orientation="h",
                marker_color="#D85A30",
            ))
            fig_decoder.update_layout(
                xaxis_title="Future-feature weight",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_decoder, width="stretch")

        st.info(
            "This is the TFT-specific explainability you were missing: encoder weights show which historical drivers mattered most, and attention shows which past months the model leaned on when forming the current forecast window."
        )
    elif isinstance(tft_interp, dict) and "error" in tft_interp:
        st.warning(tft_interp["error"])
    elif tft_runtime["error"]:
        st.warning(f"TFT checkpoint note: {tft_runtime['error']}")

    st.subheader("Weather vs Price")
    weather_cols = ["temperature_mean", "rainfall_monthly", "humidity_mean"]
    available = [c for c in weather_cols if c in plot_df.columns]
    if available:
        fig_w = make_subplots(
            rows=len(available) + 1, cols=1, shared_xaxes=True,
            subplot_titles=["Price (Rs/KG)"] + available, vertical_spacing=0.06,
        )
        fig_w.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df["price"],
            mode="lines", name="Price", line=dict(color="#185FA5", width=1.5),
        ), row=1, col=1)
        colors = ["#D85A30", "#2CA02C", "#9467BD"]
        for i, col in enumerate(available):
            fig_w.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df[col],
                mode="lines", name=col, line=dict(width=1, color=colors[i]),
            ), row=i + 2, col=1)
        fig_w.update_layout(
            template="plotly_white", height=200 * (len(available) + 1), showlegend=False,
        )
        st.plotly_chart(fig_w, width="stretch")

# ── TAB 4: Present vs Predicted ───────────────────────────────────────
if selected_view == "Present vs Predicted":
    st.subheader("Test Period: Jan 2023 - Jul 2023")
    test_plot = plot_df[plot_df["date"] >= "2023-01-01"].copy()

    if len(test_plot) == 0:
        st.warning("No test data available for this market in the test period (2023+).")
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=test_plot["date"], y=test_plot["price"],
            mode="lines+markers", name="Actual price",
            line=dict(color="#185FA5", width=2),
            marker=dict(size=8),
        ))

        tft_test = None
        if show_tft and tft_series is not None:
            tft_test = tft_series[tft_series["date"] >= "2023-01-01"].sort_values("date")
            if len(tft_test) > 0:
                if show_band:
                    fig3.add_trace(go.Scatter(
                        x=tft_test["date"], y=tft_test["tft_q90"],
                        fill=None, mode="lines",
                        line=dict(color="rgba(216,90,48,0)"), showlegend=False,
                    ))
                    fig3.add_trace(go.Scatter(
                        x=tft_test["date"], y=tft_test["tft_q10"],
                        fill="tonexty", mode="lines",
                        fillcolor="rgba(216,90,48,0.2)",
                        line=dict(color="rgba(216,90,48,0)"),
                        name="q0.1-q0.9 confidence band",
                    ))
                fig3.add_trace(go.Scatter(
                    x=tft_test["date"], y=tft_test["tft_q50"],
                    mode="lines+markers", name="TFT median",
                    line=dict(color="#D85A30", width=2, dash="dash"),
                    marker=dict(size=6),
                ))

        if show_xgb and xgb_preds is not None:
            xgb_test = xgb_preds[xgb_preds["date"] >= "2023-01-01"]
            if len(xgb_test) > 0:
                fig3.add_trace(go.Scatter(
                    x=xgb_test["date"], y=xgb_test["xgb_pred"],
                    mode="lines+markers", name="XGBoost",
                    line=dict(color="#888780", width=1.5, dash="dot"),
                    marker=dict(size=6),
                ))

        fig3.update_layout(
            yaxis_title="Price (Rs/KG)",
            template="plotly_white",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig3, width="stretch")

        st.subheader("Metrics Comparison")
        metrics_data = []

        if xgb_preds is not None:
            xgb_test = xgb_preds[xgb_preds["date"] >= "2023-01-01"]
            if len(xgb_test) > 0:
                actual = xgb_test["price"].values
                predicted = xgb_test["xgb_pred"].values
                metrics_data.append({
                    "Model": "XGBoost",
                    "MAE (Rs/KG)": f"{np.mean(np.abs(actual - predicted)):.2f}",
                    "MAPE (%)": f"{np.mean(np.abs((actual - predicted) / actual)) * 100:.1f}",
                    "RMSE (Rs/KG)": f"{np.sqrt(np.mean((actual - predicted) ** 2)):.2f}",
                    "Uncertainty Band": "None",
                    "Empirical q0.1-q0.9 Coverage": "N/A",
                })

        if tft_test is not None and len(tft_test) > 0:
            actual = tft_test["price"].values
            predicted = tft_test["tft_q50"].values
            coverage = np.mean((actual >= tft_test["tft_q10"].values) &
                                (actual <= tft_test["tft_q90"].values)) * 100
            metrics_data.append({
                "Model": "TFT",
                "MAE (Rs/KG)": f"{np.mean(np.abs(actual - predicted)):.2f}",
                "MAPE (%)": f"{np.mean(np.abs((actual - predicted) / actual)) * 100:.1f}",
                "RMSE (Rs/KG)": f"{np.sqrt(np.mean((actual - predicted) ** 2)):.2f}",
                    "Uncertainty Band": "q0.1-q0.9 quantile",
                    "Empirical q0.1-q0.9 Coverage": f"{coverage:.1f}%",
            })

        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), width="stretch", hide_index=True)

        if tft_test is not None and len(tft_test) > 0:
            st.subheader("Forecast Uncertainty Width")
            fig_band = go.Figure(go.Bar(
                x=tft_test["date"],
                y=tft_test["band_width"],
                marker_color="rgba(186,117,23,0.5)",
                name="Band width (Rs)",
            ))
            fig_band.update_layout(
                yaxis_title="Uncertainty Width (Rs/KG)",
                template="plotly_white", height=300,
            )
            st.plotly_chart(fig_band, width="stretch")

st.divider()
st.subheader("Run Notes")
metrics_text = load_metrics()
if metrics_text:
    st.text(metrics_text)
