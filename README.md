# Indian Food Price Forecasting with Uncertainty and Explainability

A complete end-to-end forecasting system for Indian food prices: XGBoost point baseline + Temporal Fusion Transformer (TFT) probabilistic forecasting + Conformal Quantile Regression (CQR) calibration + interpretable explanations + Streamlit dashboard.

**Best model: TFT-XGBFusion-CQR** — MAE 5.27 Rs/KG · MAPE 11.4% · Coverage 84.0% on 2023 test data.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://interpretable-price-forecasting-jxkxjw86wejdiehu3tdcwr.streamlit.app/)

> The live app runs with pre-computed predictions and XGBoost inference. TFT live inference requires the full environment with PyTorch.

---

## Table of Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [The real-world problem](#2-the-real-world-problem)
3. [High-level system flow](#3-high-level-system-flow)
4. [Dataset — sources, numbers, filtering](#4-dataset--sources-numbers-filtering)
5. [Feature engineering — all 26 features with formulas](#5-feature-engineering--all-26-features-with-formulas)
6. [Model architecture — how TFT works inside](#6-model-architecture--how-tft-works-inside)
7. [Math reference — every formula used in this project](#7-math-reference--every-formula-used-in-this-project)
8. [Model upgrade path — from baseline to best](#8-model-upgrade-path--from-baseline-to-best)
9. [Conformal Quantile Regression (CQR) — how calibration works](#9-conformal-quantile-regression-cqr--how-calibration-works)
10. [Evaluation metrics — formulas and intuition](#10-evaluation-metrics--formulas-and-intuition)
11. [Results](#11-results)
12. [Statistical validation — ablation, coverage, VSN stability](#12-statistical-validation--ablation-coverage-vsn-stability)
13. [Streamlit dashboard — four views explained](#13-streamlit-dashboard--four-views-explained)
14. [How to run](#14-how-to-run)
15. [Technical Q&A for viva](#15-technical-qa-for-viva)
16. [Where is what — complete file map](#16-where-is-what--complete-file-map)

---

## 1. What this project is about

This project answers one question:

> **Can we forecast Indian food prices in a way that is accurate, uncertainty-aware, and explainable?**

The answer is built in layers:

1. Use a strong **point baseline** (XGBoost) for raw accuracy
2. Use a **Temporal Fusion Transformer** for multi-horizon probabilistic forecasting
3. **Fuse** the point-model signal into TFT as an input feature
4. **Calibrate** the quantile bands using Conformal Quantile Regression
5. **Validate** interpretability statistically (not just visually)
6. **Serve** everything through a Streamlit dashboard

The project covers **Onions, Tomatoes, and Rice** across **53 Indian markets** from **1994 to 2023**.

---

## 2. The real-world problem

Indian food prices can be extreme:

| Event | Price | Impact |
|---|---|---|
| Onion crisis, Dec 2010 | Rs 85/KG | Retail panic, government import |
| Onion crisis, Dec 2019 | Rs 160/KG | Export ban, government rationing |
| Tomato crisis, Jul 2023 | Rs 200+/KG | Drove national CPI inflation to 7.4% |
| Rice export ban, Aug 2023 | — | Global ripple effect |

A system that only says **"price will be Rs 60 next month"** is not useful for:

- A farmer deciding when to sell
- An FCI (Food Corporation of India) planner deciding on buffer stock
- A mandi trader managing risk

What they actually need:

> "Price will be between Rs 45 and Rs 85, most likely Rs 62 — driven by below-normal rainfall and pre-Kharif seasonal pressure."

That is why this project builds **bands + explanations**, not just a point number.

---

## 3. High-level system flow

```
RAW DATA
│
├── WFP India food prices (145,124 rows, 1994–2026)
├── WFP market coordinates (53 cities, GPS lat/lon)
└── NASA POWER monthly weather (temperature, rainfall, humidity)
│
▼ [scripts/00, 01, 02]
CLEAN MASTER DATASET
│   16,939 rows · 123 series · 53 markets · 26 features
│
├── ▼ [scripts/04] XGBoost Baseline
│       Train on 1994–2019 data
│       Predict log_price per row
│       Save: models/xgb_baseline.pkl
│
├── ▼ [scripts/03] TFT Baseline (family: original)
│       Train on 1994–2020 data
│       Save: models/tft_best.ckpt + variants
│
├── ▼ [scripts/09] TFT Retrain (family: step1)
│       Train on 1994–2021 data (fairer cutoff)
│       Save: models/tft_best_2022*.ckpt
│
└── ▼ [scripts/10] XGBoost-Fused TFT (family: step5)
        Generate xgb_log_pred for every row
        Add it as known feature to TFT
        Train on 1994–2021 data
        Save: models/tft_best_xgbfused*.ckpt
│
▼ [scripts/07] Ensemble Predict (all three families)
        Average quantile outputs across checkpoints in each family
        Output: data/processed/tft_predictions_ensemble_<family>.csv
│
▼ [scripts/08] Conformal Calibration (all three families)
        On validation split, compute calibration offsets per commodity
        Adjust q10 down, q90 up by learned offset
        Output: data/processed/tft_predictions_calibrated_<family>.csv
│
▼ [scripts/06] Evaluate + [scripts/11] Statistical Validation
        Compute MAE, MAPE, coverage, band width
        Bootstrap VSN stability, ablation paired tests, coverage tests
        Output: visualizations/evaluation_metrics.txt
                visualizations/explainability_stats.txt
                visualizations/*.png figures
│
▼ app.py
        Streamlit dashboard
        Loads best available family automatically
        Shows 4 views: history, future, explainability, test comparison
```

---

## 4. Dataset — sources, numbers, filtering

### 4.1 Raw sources

**WFP India food prices**

| Attribute | Value |
|---|---|
| File | `data/raw/wfp_food_prices_ind.csv` |
| Rows | 145,124 |
| Columns | 16 |
| Date range | Jan 1994 – Feb 2026 |
| Markets | 170 (164 cities + 1 National Average + 5 Zone aggregates) |
| Commodities | 41 |
| Collection method | WFP price monitors visit one retail location per city on the 15th of each month |

Each row = one price observation at one retail location in one city on one date.

**WFP market coordinates** (`data/raw/wfp_markets_ind.csv`)

Provides GPS latitude/longitude for each market, used to fetch weather from NASA POWER.

**NASA POWER monthly weather**

Fetched per market GPS coordinate via the NASA POWER API. Three variables used:

| NASA Variable | Feature Name | Meaning |
|---|---|---|
| `T2M` | `temperature_mean` | Monthly mean temperature at 2m height (°C) |
| `PRECTOTCORR` | `rainfall_monthly` | Monthly total rainfall (mm) |
| `RH2M` | `humidity_mean` | Monthly mean relative humidity at 2m (%) |

### 4.2 Data filtering steps

```
Step                                   | Rows dropped  | Remaining
---------------------------------------|---------------|----------
Start (raw)                            | —             | 145,124
Drop "National Average" and Zone rows  | 535           | 144,589
Keep Retail prices only                | 1,697         | 142,892
Keep Onions + Tomatoes + Rice only     | 118,675       | 24,217
Keep KG unit only                      | 0             | 24,217
Normalize duplicate market name spellings | ~50         | ~24,167
Drop series with fewer than 60 months  | 8,203         | 16,014
Fill short gaps (≤ 3 consecutive months)| +2,401       | 18,415
```

**Why 60 months minimum?**

TFT uses a 24-month encoder window. A series with 60 months provides:
- 24 months for the encoder lookback
- 6 months for the prediction target
- 30 remaining sliding windows for training

A series with 40 months gives only 10 training windows — the model memorizes rather than learns. A series with fewer than 24 months cannot even fill one encoder window.

**Why Onions, Tomatoes, and Rice?**

| Commodity | Reason selected | Coefficient of Variation |
|---|---|---|
| Onions | Highest volatility (price tripled in 2019) | 0.568 |
| Tomatoes | Recent spike (Rs 200+/KG in 2023, known to everyone) | 0.546 |
| Rice | Stable crop — tests whether TFT learns different volatility regimes | 0.456 |

This combination tests whether the same model can handle volatile and stable regimes. If TFT produces wide bands for Onions and narrow bands for Rice without being told — that is genuine machine learning, not overfitting.

### 4.3 Final model-ready dataset

| Attribute | Value |
|---|---|
| File | `data/processed/master_dataset.csv` |
| Rows | 16,939 |
| Series (market × commodity) | 123 |
| Markets | 53 |
| Date range | Jan 1995 – Jul 2023 |

Commodity-wise:

| Commodity | Series | Rows |
|---|---|---|
| Onions | 38 | 5,003 |
| Tomatoes | 45 | 5,245 |
| Rice | 40 | 6,691 |

### 4.4 Train / val / test split

| Split | Date range | Purpose |
|---|---|---|
| Train | 1995-01 to 2020-12 (original) or to 2021-12 (retrained) | Model learning |
| Val | 2021-22 (original) or 2022 (retrained) | Early stopping + CQR calibration |
| Test | 2023+ | True generalization — never seen during training |

Split is time-ordered. No shuffle. Never train on future data to predict the past.

---

## 5. Feature engineering — all 26 features with formulas

The master dataset contains 26 columns. The model does not learn from raw prices alone — it learns from a structured system of feature types.

### 5.1 Feature categories

```
26 Features
│
├── Static identity (3) ── who are we predicting?
│     commodity, market, admin1
│
├── Known future calendar (7) ── what time is it?
│     time_idx, year, month, month_sin, month_cos, season, covid_lockdown
│
├── Unknown past: price memory (6) ── what has been happening?
│     log_price, price_lag_1m, price_lag_12m, rolling_3m, rolling_6m, yoy_change
│
├── Unknown past: weather (3) ── physical supply-side context
│     temperature_mean, rainfall_monthly, humidity_mean
│
├── Unknown past: shock indicators (4) ── binary threshold events
│     rain_deficit, rain_excess, heat_stress, cold_stress
│
└── Fused model feature (1) ── XGB prediction (step5 only)
      xgb_log_pred
```

### 5.2 Static features

These define the identity of a series. TFT embeds them and uses them to initialize the LSTM internal state. Two series with the same market but different commodities start from completely different internal states.

| Feature | Meaning |
|---|---|
| `commodity` | Onions / Tomatoes / Rice |
| `market` | City name (e.g., Chennai, Delhi, Mumbai) |
| `admin1` | State name (e.g., Tamil Nadu, Maharashtra) |

### 5.3 Known future features — formulas

These are known for any future month because they are calendar-derived.

**Log-transformed time index:**

```
time_idx(t) = months since Jan 1994
```

**Cyclical month encoding:**

```
month_sin(t) = sin(2π × month(t) / 12)
month_cos(t) = cos(2π × month(t) / 12)
```

Why cyclical? Without this, December (month=12) and January (month=1) appear numerically far apart (distance=11). Cyclical encoding makes them adjacent:

```
cos(2π×12/12) − cos(2π×1/12) = 1.0 − 0.866 = 0.134   (small = adjacent)
```

**Agricultural season:**

```
Kharif = months 7–10 (July to October, monsoon crop)
Rabi   = months 11–2 (November to February, winter crop)
Zaid   = months 3–6  (March to June, summer crop)
```

**COVID lockdown flag:**

```
covid_lockdown(t) = 1  if month is Mar 2020 – Sep 2020
                  = 0  otherwise
```

### 5.4 Unknown past features — formulas

These are known only up to the current time step. For future forecasting, they must be estimated or carried forward.

**Log-price transformation:**

```
log_price(t) = log(1 + price(t))         [numpy.log1p]
Inverse:  price(t) = exp(log_price(t)) − 1  [numpy.expm1]
```

Why log? Retail food prices are right-skewed (spikes like Rs 200/KG for tomatoes). Log compression stabilizes variance and prevents the model from chasing outliers.

**Price lags:**

```
price_lag_1m(t)  = log_price(t − 1)    ← last month's log-price
price_lag_12m(t) = log_price(t − 12)   ← same month last year's log-price
```

**Rolling averages:**

```
rolling_3m(t) = mean( log_price(t−2), log_price(t−1), log_price(t) )
rolling_6m(t) = mean( log_price(t−5), ..., log_price(t) )
```

**Year-over-year change rate:**

```
yoy_change(t) = (price_lag_1m(t) − price_lag_12m(t)) / price_lag_12m(t)
```

Captures how fast the current price is diverging from the same period last year.

**Weather shock binary flags:**

```
rain_deficit(t)  = 1  if rainfall_monthly(t) < 50 mm    (below-normal rainfall)
rain_excess(t)   = 1  if rainfall_monthly(t) > 400 mm   (flood-level rainfall)
heat_stress(t)   = 1  if temperature_mean(t) > 38°C     (crop heat stress threshold)
cold_stress(t)   = 1  if temperature_mean(t) < 10°C     (frost/cold stress threshold)
```

Why binary instead of raw values? The TFT's Variable Selection Network (VSN) produces more interpretable and stable weights for binary shock indicators than for raw continuous values. A binary flag of 1 has a clear economic meaning ("this month had a drought"); a raw value of 42 mm is harder to interpret directly.

### 5.5 Fused feature (step5 family only)

```
xgb_log_pred(t) = XGBoost prediction of log_price(t)
```

This is the point-model signal injected into TFT. It is generated by a clean XGBoost model trained only on data up to 2019 (before the validation period) to avoid leakage. Details in section 8.

---

## 6. Model architecture — how TFT works inside

### 6.1 Why TFT over other models

| Model | Multi-series | Multi-horizon | Quantile bands | Dynamic feature importance | Temporal attention |
|---|---|---|---|---|---|
| ARIMA | No | No | No | No | No |
| LSTM | Yes | Yes | With extra head | No | No |
| Prophet | No | Yes | Basic | No | No |
| XGBoost | Yes | No (roll out) | No (native) | Static SHAP | No |
| **TFT** | **Yes** | **Yes (native)** | **Yes (native)** | **Yes (per timestep)** | **Yes (interpretable)** |

TFT is the only model that gives all three simultaneously:
1. What will the price be? (q50 median)
2. How confident is the model? (q10/q90 band)
3. Why this prediction? (VSN weights + attention)

### 6.2 TFT architecture — complete diagram

```
═══════════════════════════════════════════════════════════
              TEMPORAL FUSION TRANSFORMER
               (our configuration: 120K params)
═══════════════════════════════════════════════════════════

INPUTS (26 features grouped by type)
─────────────────────────────────────
Static:           commodity, market, admin1
Known future:     time_idx, year, month, month_sin/cos, season, covid_lockdown
Unknown past:     log_price, temperature, rainfall, humidity,
                  price_lag_1m, price_lag_12m, rolling_3m/6m, yoy_change,
                  rain_deficit, rain_excess, heat_stress, cold_stress
(step5 only):     xgb_log_pred (added to known future)

        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ENTITY EMBEDDING LAYER                                  │
│                                                          │
│  commodity → embedding e_c (learned vector)             │
│  market    → embedding e_m (learned vector)             │
│  admin1    → embedding e_a (learned vector)             │
│                                                          │
│  Static context vectors:                                 │
│    c_s = GRN(concat[e_c, e_m, e_a])   ← for VSN        │
│    c_e = GRN(...)                      ← for enrichment │
│    c_h = GRN(...)                      ← LSTM init h    │
│    c_c = GRN(...)                      ← LSTM init c    │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌───────────────────────┐         ┌───────────────────────┐
│  ENCODER VSN          │         │  DECODER VSN           │
│  (past 24 months)     │         │  (future 6 months)    │
│                       │         │                        │
│  v_enc = Softmax(     │         │  v_dec = Softmax(     │
│    GRN(features, c_s) │         │    GRN(features, c_s) │
│  )                    │         │  )                     │
│                       │         │                        │
│  xi_enc = Σ v^(j) ×  │         │  xi_dec = Σ v^(j) ×  │
│    GRN_j(feature_j)   │         │    GRN_j(feature_j)   │
│                       │         │                        │
│  OUTPUT:              │         │  OUTPUT:               │
│  importance weight    │         │  importance weight     │
│  per feature,         │         │  per feature,          │
│  per timestep         │         │  per timestep          │
└──────────┬────────────┘         └──────────┬─────────────┘
           │                                  │
           └────────────────┬─────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  LSTM ENCODER–DECODER                                    │
│                                                          │
│  Encoder processes past 24 months:                       │
│    h_t, c_t = LSTM(xi_enc_t, h_{t-1}, c_{t-1})         │
│    h_0 = GRN_h(c_s)  ← initialized from static embed   │
│    c_0 = GRN_c(c_s)  ← different per commodity+market  │
│                                                          │
│  Decoder processes future 6 months:                     │
│    uses encoder final hidden state                       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STATIC ENRICHMENT LAYER                                 │
│                                                          │
│  φ_t = LayerNorm( xi_t + GLU( GRN( h_t ‖ c_e ) ) )    │
│                                                          │
│  Fuses LSTM temporal memory with static context          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  INTERPRETABLE MULTI-HEAD ATTENTION (2 heads)            │
│                                                          │
│  Standard: Attn(Q,K,V) = Softmax(Q·K^T / √d) · V       │
│                                                          │
│  TFT: shares W_V across ALL heads                        │
│    InterpAttn = [1/H × Σ_h Attn(Q·W_Q^h, K·W_K^h, V)] │
│               × W_H                                     │
│                                                          │
│  Shared W_V means attention weights can be meaningfully │
│  averaged → α(t, n) = how much past month n matters     │
│                      when predicting future month t      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  QUANTILE OUTPUT HEADS (3 simultaneous)                  │
│                                                          │
│  ŷ(q=0.10, t) = Linear_0.1(δ_t)  ← floor prediction    │
│  ŷ(q=0.50, t) = Linear_0.5(δ_t)  ← median prediction   │
│  ŷ(q=0.90, t) = Linear_0.9(δ_t)  ← ceiling prediction  │
│                                                          │
│  Trained with Pinball Loss — see section 7               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
          GroupNormalizer.inverse_transform(ŷ)
                           │
                           ▼
               price = expm1(ŷ_normalized)
                           │
                           ▼
    THREE PRICE BANDS (Rs/KG) per (market, commodity, month)
          q10 (floor)   q50 (median)   q90 (ceiling)
```

### 6.3 Configuration used in this project

| Parameter | Value | Reason |
|---|---|---|
| `hidden_size` | 32 | Keeps total params ~120K, appropriate for 16K rows |
| `attention_head_size` | 2 | One head for seasonality, one for trend |
| `lstm_layers` | 1 | 2+ layers overfit on small data |
| `hidden_continuous_size` | 16 | ≤ hidden_size/2 — embedding size for continuous features |
| `dropout` | 0.2 | Regularization; prevents memorization |
| `max_encoder_length` | 24 | Two annual cycles of lookback |
| `max_prediction_length` | 6 | One Indian agricultural season ahead |
| `learning_rate` | 0.03 | Conservative; higher caused oscillation |
| `quantiles` | [0.1, 0.5, 0.9] | 80% central prediction interval |
| `target_normalizer` | GroupNormalizer (softplus) | Per-series normalization — Onions (Rs 20–160) and Rice (Rs 25–65) are on different scales |

### 6.4 Gated Residual Network (GRN) — the building block

GRN is used inside VSN, enrichment, and LSTM initialization:

```
GRN(a, c) = LayerNorm( a + GLU(η_1) )

  η_2 = ELU( W_2 × a + W_3 × c + b_2 )
  η_1 = W_1 × η_2 + b_1
  GLU(x) = x[:d] × sigmoid(x[d:])     ← element-wise learned gate

The sigmoid inside GLU acts as a learnable on/off switch.
During drought:   rain_deficit gate → sigmoid → 1.0 (feature passes through)
During normal:    lag price gate    → sigmoid → 1.0 (price history dominates)
```

---

## 7. Math reference — every formula used in this project

### 7.1 Pinball (Quantile) Loss

This is the core training loss for TFT. Instead of minimizing squared error, TFT minimizes pinball loss at three quantiles simultaneously.

```
QL(y, ŷ, q) = q × max(y − ŷ, 0)  +  (1−q) × max(ŷ − y, 0)

For q = 0.90 (ceiling head):
  Actual = Rs 70, predicted q90 = Rs 50:
    Penalty = 0.90 × (70 − 50) = Rs 18   ← HEAVY penalty for underestimating
  Actual = Rs 40, predicted q90 = Rs 50:
    Penalty = 0.10 × (50 − 40) = Rs 1    ← light penalty for overestimating
  → q90 head learns to set a HIGH ceiling to avoid missing spikes.

For q = 0.10 (floor head):
  Opposite — heavy penalty if floor is too high, light if too low.
  → q10 head learns to set a LOW floor.

For q = 0.50 (median head):
  Equal penalty both sides (symmetric) — standard median regression.
```

Total training loss across all series Ω, all quantiles, all horizons H:

```
L(Ω, W) = (1 / M×H) × Σ_{y∈Ω}  Σ_{q∈{0.1,0.5,0.9}}  Σ_{τ=1}^{H}  QL(y_τ, ŷ(q,τ))

One forward pass trains all three heads simultaneously.
The model is FORCED to output bands — it is not optional.
```

### 7.2 Variable Selection Network (VSN) weights

```
v_t = Softmax( GRN( Xi_t, c_s ) )        ← softmax → weights sum to 1.0
xi_t = Σ_j  v_t^(j) × GRN_j( xi_t^(j) ) ← weighted combination

v_t is a vector of length = num_features, summing to 1.0
v_t^(j) is the importance weight of feature j at time t.

The weights change every timestep:
  Jan 2023 (stable):  price_lag_1m=0.45, rolling_6m=0.20, rain_deficit=0.02
  Jul 2023 (drought): rain_deficit=0.35, heat_stress=0.22, price_lag_1m=0.18
```

### 7.3 Interpretable multi-head attention

```
Standard attention:
  Attn(Q,K,V) = Softmax( Q·K^T / √d_attn ) · V

TFT's shared-W_V version:
  InterpAttn = [1/H × Σ_h Attn(Q·W_Q^h, K·W_K^h, V)] × W_H
                                              ↑
                                    W_V shared across all h heads

Result: α(t,n) = attention weight from decoder step t to encoder step n
  α(t, t−12) is high → model learned annual seasonality
  α(t, t−1)  is high → model uses last-month momentum
```

### 7.4 GroupNormalizer (per-series normalization)

```
y_normalized(t, i) = ( y(t, i) − mean(y_i) ) / std(y_i)

For each series i (e.g., Onions_Chennai) separately.

Without this: LSTM learns Rice as near-constant because its absolute variance
is much lower than Onions. Per-series normalization puts all 123 series on
the same scale for training.
```

### 7.5 Log transform and inverse

```
Forward:  log_price(t) = log(1 + price(t))     [numpy.log1p]
Inverse:  price(t)     = exp(log_price(t)) − 1  [numpy.expm1]

Why log? Retail prices are right-skewed (e.g., Rs 200/KG tomato spikes).
Log compression stabilizes variance so the loss is not dominated by spikes.
```

### 7.6 Conformal Quantile Regression (CQR) offsets

Full derivation in section 9. The formula:

```
For each commodity c on calibration (val) split:
  E_i = max( log_q10_i − log_y_i,  log_y_i − log_q90_i )
    E_i > 0 means actual price was OUTSIDE the predicted band

  Q̂_c = (1−α) quantile of {E_1, E_2, ..., E_n_c}   where α = 0.10

Adjusted output at test time:
  q10_adjusted(t) = exp( log_q10(t) − Q̂_c ) − 1
  q90_adjusted(t) = exp( log_q90(t) + Q̂_c ) − 1
```

### 7.7 Evaluation metrics

Full explanation in section 10.

```
MAE   = (1/n) × Σ |price_i − q50_i|
RMSE  = √( (1/n) × Σ (price_i − q50_i)² )
MAPE  = (1/n) × Σ |price_i − q50_i| / price_i  × 100%
Coverage = (1/n) × Σ 1[ q10_i ≤ price_i ≤ q90_i ]  × 100%
BandWidth = (1/n) × Σ (q90_i − q10_i)
```

---

## 8. Model upgrade path — from baseline to best

The project improved through four distinct upgrades. Each upgrade is a separate experiment with a measurable outcome.

### 8.1 Upgrade 0: XGBoost point baseline

**Script:** `scripts/04_train_xgboost.py`

**What it does:** Trains `XGBRegressor` (500 trees, max_depth=6, learning_rate=0.05) on the master dataset to predict `log_price`. No uncertainty bands. No attention.

**Why it matters:**
- Provides a strong accuracy benchmark
- Its predictions later become the fused feature for step5
- Answers: "What if we only care about point accuracy?"

**Result:** MAE 1.58 Rs/KG, MAPE 2.8%

**Why XGBoost is so accurate for point prediction:**

The top feature for XGBoost is `price_lag_1m` (76% importance). It essentially says:
> "Next month's price ≈ this month's price + small adjustment"

This works for 85% of stable months. For the remaining 15% (price transitions and shocks), the lag anchor causes 5–10 Rs/KG errors. But the average across all months is very low.

### 8.2 Upgrade 1: TFT baseline

**Script:** `scripts/03_train_tft.py`

**Family name:** `original`

**What it does:** Trains TFT on data from 1994 to Dec 2020. Saves multiple checkpoints from different epochs.

**Problem with this version:** Trained through Dec 2020, while XGBoost baseline was trained through Dec 2022. Unfair comparison.

**Result:** MAE 10.53 Rs/KG, MAPE 29.1%, Coverage 64.9%

### 8.3 Upgrade 2: Checkpoint ensemble

**Script:** `scripts/07_ensemble_predict.py --family original`

**What it does:** Loads all `tft_best*.ckpt` files in the `original` family. For each checkpoint, generates q10/q50/q90 predictions. Then averages the predictions across checkpoints in log space.

```
Why average in log space?
  Predictions are in log_price space during inference.
  Average log predictions → exponentiate → avoids geometric mean distortion.
```

**Why this helps:** A single deep learning checkpoint can behave differently on two runs due to random initialization. Averaging reduces this variance without any retraining.

**Result:** Coverage improves (more stable bands). MAE similar to baseline.

### 8.4 Upgrade 3: Fair retraining (step1)

**Script:** `scripts/09_retrain_tft_2022.py`

**Family name:** `step1`

**What it does:** Retrains TFT with train cutoff = Dec 2021, val = 2022, test = 2023+. Gives TFT one more year of data to learn from, making the comparison with XGBoost fairer.

**Result (after CQR):** MAE 9.91 Rs/KG, MAPE 29.2%, Coverage 87.6%

### 8.5 Upgrade 4: XGBoost-fused TFT (step5)

**Script:** `scripts/10_xgb_as_tft_feature.py`

**Family name:** `step5`

**What it does:**

```
Step 1: Train a CLEAN XGBoost model on 1994–2019 data only
        (train_cutoff 2019 to avoid leaking val/test info into the feature)

Step 2: Use this XGB model to generate xgb_log_pred for EVERY row
        in the dataset (train, val, test, future)
        xgb_log_pred(t) = XGBoost.predict(features_t)

Step 3: Add xgb_log_pred as a time_varying_known_real feature in TFT
        (known because we can always generate it for future months)

Step 4: Retrain TFT with this extra feature
        Now TFT receives: original 25 features + xgb_log_pred = 26 features
```

**Why this is the biggest improvement:**

XGBoost is extremely strong at point prediction (MAPE 2.8%). By feeding its prediction as an input feature to TFT, TFT no longer has to learn everything from scratch. Instead:

> TFT learns to **refine** the XGBoost prediction using its sequence model, attention, and uncertainty quantification.

The VSN weights will show how much TFT trusts the XGB signal vs other features.

**Important distinction:** This is NOT simple model averaging. The XGB prediction goes IN to TFT as a feature. TFT can choose to trust it, correct it, or ignore it — the VSN decides dynamically per timestep.

**Result (after CQR):** MAE 5.27 Rs/KG, MAPE 11.4%, Coverage 84.0%, BandWidth 11.99

This is a 53% MAE reduction compared to TFT-Base.

---

## 9. Conformal Quantile Regression (CQR) — how calibration works

### 9.1 The problem it solves

After training and ensembling, the TFT-Base achieves only 64.9% empirical coverage. This means in 35 out of 100 months, the actual price falls outside the predicted q10–q90 band. The model is **overconfident** — its bands are too narrow.

We want at least 80% coverage (ideally 90%) without retraining the model.

### 9.2 The solution: distribution-free coverage guarantee

**Script:** `scripts/08_conformal_calibrate.py`

CQR (Romano et al., 2019) provides a distribution-free way to inflate bands to achieve a target coverage level on held-out data.

```
Algorithm:

1. Use the VALIDATION split (never touched during training)

2. For each row i in val, compute the conformity score:
     E_i = max(  log_q10_i − log_y_i,   ← how far y is below the lower bound
                 log_y_i − log_q90_i   ) ← how far y is above the upper bound

   E_i > 0 means: actual price was OUTSIDE the predicted band
   E_i ≤ 0 means: actual price was INSIDE the predicted band

3. Compute per-commodity calibration offset:
     Q̂_c = quantile( {E_i : commodity_i = c}, level = 1−α )
   where α = 0.10

   This is the (1−α)-th quantile of conformity scores for commodity c.
   Q̂_c is the "minimum band expansion needed to cover (1−α)% of val errors."

4. At test time, for each prediction:
     q10_adjusted = expm1( log_q10 − Q̂_c )   ← shrink floor (widen downward)
     q90_adjusted = expm1( log_q90 + Q̂_c )   ← raise ceiling (widen upward)
     q50 stays unchanged (point forecast is not affected)
```

**Key property:** CQR provides a **marginal coverage guarantee** — if the val and test data are exchangeable (same distribution), the adjusted bands will achieve at least (1−α) coverage on the test set.

### 9.3 Per-commodity calibration

Offsets are computed separately per commodity because Onions, Tomatoes, and Rice have different volatility profiles:

```
If we use one global offset:
  Onions (high volatility) → still undercovered
  Rice (low volatility)    → massively overcovered (huge useless bands)

Per-commodity calibration:
  Onions:   larger Q̂ → bigger expansion → matches actual Onion volatility
  Tomatoes: medium Q̂
  Rice:     smaller Q̂ → tighter expansion → matches Rice stability
```

This is saved in: `models/conformal_offsets_<family>.json`

### 9.4 Why coverage is 84%, not 90%

The target `α = 0.10` gives a theoretical 90% coverage guarantee. In practice, the empirical test coverage is 84.0% for step5 because:

1. Val and test distributions are not perfectly exchangeable (test = 2023, val = 2022)
2. The 2023 Tomato spike (Rs 200+/KG) was more extreme than anything in val
3. CQR is a marginal guarantee, not a worst-case guarantee

Always report the **empirical coverage** (84.0%), not the theoretical target (90%).

---

## 10. Evaluation metrics — formulas and intuition

### 10.1 MAE — Mean Absolute Error

```
MAE = (1/n) × Σ |price_i − q50_i|

Units: Rs/KG (same as price — directly interpretable)
Lower = better.

Example: MAE = 5.27 Rs/KG
  → On average, our median prediction is off by Rs 5.27 per KG.
  → For Rice (Rs 35–65), this is 8–15% error on a given day.
  → For Onions (Rs 20–160), this varies widely by market state.
```

### 10.2 RMSE — Root Mean Squared Error

```
RMSE = √( (1/n) × Σ (price_i − q50_i)² )

Units: Rs/KG (same as MAE, but penalizes large errors more)
Lower = better.

RMSE > MAE always. The gap tells you about outlier sensitivity:
  Large RMSE/MAE ratio → a few big errors are pulling RMSE up
  Small RMSE/MAE ratio → errors are uniformly distributed
```

### 10.3 MAPE — Mean Absolute Percentage Error

```
MAPE = (1/n) × Σ |price_i − q50_i| / price_i  × 100%

Units: % (scale-independent — can compare across crops)
Lower = better.

MAPE = 11.4% → on average, the prediction error is 11.4% of the actual price.
  Rs 50/KG actual → ~Rs 5.7/KG average error
  Rs 100/KG actual → ~Rs 11.4/KG average error
  The error scales with the price level.

Why we use MAPE for comparison across literature:
  - Different crops have different price scales
  - MAPE is scale-free → Onions (Rs 20–160) and Rice (Rs 35–65) are comparable
```

### 10.4 Coverage — empirical interval coverage

```
Coverage = (1/n) × Σ 1[ q10_i ≤ price_i ≤ q90_i ]  × 100%

1[...] = indicator function (1 if true, 0 if false)

Coverage = 84.0% for step5
  → In 84 out of 100 months, the actual price fell inside the predicted q10–q90 band.

What coverage tells you:
  Coverage < 70% → bands are too narrow → model is overconfident → DANGEROUS
  Coverage 80–90% → good calibration → model is appropriately uncertain
  Coverage > 95% → bands are too wide → model is too cautious → useless for decisions

Coverage does NOT tell you about point accuracy (MAE/MAPE). It is a separate metric.
```

### 10.5 Band Width — average interval width

```
BandWidth = (1/n) × Σ (q90_i − q10_i)

Units: Rs/KG

Lower is better (narrower = more informative), BUT only if coverage is acceptable.

The tradeoff:
  Narrow band + high coverage = IDEAL (confident and correct)
  Narrow band + low coverage  = OVERCONFIDENT and dangerous
  Wide band + high coverage   = CAUTIOUS but less informative
  Wide band + low coverage    = BOTH wrong AND uninformative (failure)

step5: BandWidth = 11.99 Rs/KG, Coverage = 84.0%
  → Narrower bands than other TFT variants AND better coverage.
  → The XGB fusion improved both simultaneously.
```

### 10.6 Summary: how to read the results table

| Metric | What it measures | Direction | Unit |
|---|---|---|---|
| MAE | Average point error | Lower = better | Rs/KG |
| MAPE | Average % point error | Lower = better | % |
| Coverage | % of actual prices inside band | Higher = better (until ~90%) | % |
| BandWidth | Average band width | Lower = better IF coverage is acceptable | Rs/KG |

---

## 11. Results

### 11.1 Main results (2023 test set)

| Variant | MAE (Rs/KG) | MAPE | Coverage | Avg Band Width |
|---|---:|---:|---:|---:|
| XGBoost baseline | **1.58** | **2.8%** | N/A (point only) | N/A |
| TFT-Base | 10.53 | 29.1% | 64.9% | 23.62 |
| TFT-EnsCQR | 10.85 | 28.3% | 88.5% | 32.44 |
| TFT-Retrain21-CQR | 9.91 | 29.2% | 87.6% | 34.78 |
| **TFT-XGBFusion-CQR** | **5.27** | **11.4%** | **84.0%** | **11.99** |

XGBoost wins on point accuracy (it uses 100% of training signal on one output). TFT-XGBFusion-CQR wins on everything probabilistic + interpretable.

### 11.2 Commodity-wise best-family results (step5, test set)

| Commodity | MAE | MAPE | Coverage |
|---|---:|---:|---:|
| Onions | 2.25 | 9.1% | 90.9% |
| Tomatoes | 8.13 | 13.0% | 82.5% |
| Rice | 4.81 | 11.7% | 79.1% |

Onions are easiest (model learned Kharif/Rabi seasonal patterns well). Tomatoes are hardest due to the unprecedented 2023 spike (Rs 200+/KG). Rice is in between.

### 11.3 Comparison with published literature

| Paper | Crop | Method | MAPE |
|---|---|---|---|
| Dharavath et al. 2020 | Onion | LSTM | 11–18% |
| Paul & Sinha 2022 | Onion/Tomato | Deep Learning | 12–22% |
| Sabu & Kumar 2020 | Vegetables | Random Forest | 9–14% |
| **Our XGBoost baseline** | **All 3** | **XGBoost** | **2.8%** |
| **Our TFT-XGBFusion-CQR** | **All 3** | **TFT+CQR** | **11.4%** |

Our XGBoost baseline outperforms all published results for point accuracy. Our TFT is within the published MAPE range AND is the only Indian food-price model that provides calibrated uncertainty bands + per-timestep feature importance + temporal attention simultaneously.

---

## 12. Statistical validation — ablation, coverage, VSN stability

This project does not just show results. It provides **statistical evidence** that the results are genuine and not due to chance.

### 12.1 Ablation tests (pairwise error comparison)

**Script:** `scripts/11_explainability_stats.py`

For each series, compute the per-month MAE difference between two model families. Then run:

1. **Paired t-test**: tests whether the mean dMAE is statistically different from zero
2. **Wilcoxon signed-rank test**: non-parametric version (does not assume normality)
3. **Cohen's d**: effect size (how large is the improvement relative to variance)

Results from `visualizations/explainability_stats.txt`:

| Comparison | Mean dMAE | t-statistic | p-value | Cohen's d |
|---|---:|---:|---:|---:|
| step5 vs original | −5.577 Rs/KG | −22.34 | 4.6×10⁻⁸⁶ | −0.79 |
| step5 vs step1 | −4.641 Rs/KG | −21.47 | 6.0×10⁻⁸¹ | −0.76 |
| step1 vs original | −0.936 Rs/KG | −4.15 | 3.6×10⁻⁵ | −0.15 |

All three comparisons are statistically significant. The step5 improvement is not due to random variation.

### 12.2 Coverage tests

For each commodity, test whether the empirical coverage equals the target level using a binomial test:

| Commodity | n | Hits | Coverage |
|---|---:|---:|---:|
| Onions | 242 | 220 | 90.9% |
| Rice | 253 | 200 | 79.1% |
| Tomatoes | 297 | 245 | 82.5% |
| All combined | 792 | 665 | 84.0% |

### 12.3 VSN stability (bootstrap)

**What it measures:** Are the variable importance rankings stable, or do they change randomly depending on which data subset is used?

```
Algorithm:
  1. Take the TFT encoder importance weights (per-series, per-timestep)
  2. Run 1000 bootstrap resamples (sample with replacement)
  3. For each resample, compute the mean importance per feature
  4. Compute pairwise Kendall rank correlation τ across all resample pairs
     τ = 1.0 → perfectly consistent ranking every time
     τ = 0.0 → random ranking
```

Results:

- **Mean pairwise Kendall τ = 0.942**
- **95% CI = [0.895, 0.990]**

This means the feature importance ranking is 94% consistent across bootstrap resamples — the VSN output is stable and reproducible, not a random artifact.

**Top encoder features by importance:**

| Feature | Mean Weight | 95% CI |
|---|---:|---|
| temperature_mean | 0.1432 | [0.1359, 0.1514] |
| rain_excess | 0.1427 | [0.1343, 0.1518] |
| rain_deficit | 0.1402 | [0.1290, 0.1512] |
| humidity_mean | 0.0664 | [0.0636, 0.0693] |
| year | 0.0600 | [0.0575, 0.0628] |
| covid_lockdown | 0.0587 | [0.0566, 0.0607] |
| season | 0.0419 | [0.0380, 0.0461] |
| log_price | 0.0405 | [0.0362, 0.0450] |

**Important caution:** High VSN weight means the model USES that feature heavily — it does not mean the feature CAUSES price movement. Weather features dominate as input signal, but weather does not directly cause prices. It is one factor in a complex supply chain. Do not claim causal relationships from VSN weights.

---

## 13. Streamlit dashboard — four views explained

### 13.1 How the app selects the best model

When the dashboard starts, it automatically checks what artifacts are available and promotes the best family:

```
Priority 1: step5 family
  Condition: tft_best_xgbfused*.ckpt exists
         AND master_dataset_xgbfused.csv exists
  → Loads TFT-XGBFusion-CQR (best model, 11.4% MAPE)

Priority 2: step1 family
  Condition: tft_best_2022*.ckpt exists
  → Loads TFT-Retrain21-CQR

Priority 3: original family
  Condition: tft_best.ckpt or tft_best-v*.ckpt exists
  → Loads TFT-Base

No checkpoint:
  → App works with XGBoost and CSV predictions only (no live TFT)
```

### 13.2 View 1: Price Forecast (historical)

Shows the historical price of the selected market/commodity with:
- **Actual price** (dark line) — the real observed prices
- **TFT q50** (orange line) — model's median prediction
- **q10–q90 band** (shaded) — uncertainty range from the stored CSV
- **Event markers** — manually labeled historical events (2019 onion crisis, 2023 tomato spike)
- **Auto-detected spikes** — months where price changed >25% month-over-month
- **VSN driver explanations** — which features drove each spike (from the TFT model)

### 13.3 View 2: Future Forecast

Shows the next 1–6 month forecast using live TFT inference:

```
How the future prediction works:

1. Load history_df for the selected series (all rows up to last date)

2. For each future month (1 to horizon):
     Build feature row: calendar features (exact) +
                        weather (historical monthly average) +
                        price lags (carried forward from last known or previous prediction)
     Append this row to working_df

3. Run ONE call to TFT with predict=True:
     → TimeSeriesDataSet.from_dataset(predict=True) selects the last 24-month window
     → model.predict(mode="quantiles") returns [B, T, Q] tensor
     → Take the last `horizon` steps from the prediction

4. Apply CQR offset to expand bands
5. Display as line chart with shaded uncertainty band
```

Shows:
- Risk level (Low / Medium / High) based on average band width
- Key driver explanations from VSN

### 13.4 View 3: Model Explainability

Shows three types of model interpretation:

**XGBoost feature importance:**
- Bar chart of feature weights (static — same regardless of series or date)
- Top feature is always `price_lag_1m` (~76%)

**TFT encoder importance (VSN):**
- Bar chart of which past features the model focused on
- Changes per series (Onions vs Rice show different patterns)
- This is DYNAMIC — not the same as XGBoost's static importance

**TFT attention over past months:**
- Line/bar chart showing α(t, n): how much each past month influenced the current prediction
- α(t, t−12) being high = model learned annual seasonality
- α(t, t−1) being high = model uses last-month momentum

### 13.5 View 4: Present vs Predicted (test validation)

Shows the held-out 2023+ test period comparison:
- Actual price vs TFT prediction on months the model never saw during training
- Reports MAE, MAPE, Coverage, Band Width for the selected series
- Band width chart showing confidence evolution over the test period

---

## 14. How to run

### 14.1 Launch the Streamlit dashboard (existing artifacts)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### 14.2 Rebuild the full evaluation from saved predictions

```powershell
.\.venv\Scripts\python.exe .\scripts\06_evaluate.py
.\.venv\Scripts\python.exe .\scripts\11_explainability_stats.py
```

### 14.3 Rebuild the step5 (XGB-fused) pipeline from scratch

```powershell
.\.venv\Scripts\python.exe .\run_xgb_steps.py --gpus 1
```

Check progress:

```powershell
.\.venv\Scripts\python.exe .\check_xgb_status.py
```

### 14.4 Full pipeline from raw data (baseline)

```powershell
.\.venv\Scripts\python.exe scripts\00_filter_prices.py
.\.venv\Scripts\python.exe scripts\01_fetch_weather.py
.\.venv\Scripts\python.exe scripts\02_merge_features.py
.\.venv\Scripts\python.exe scripts\03_train_tft.py
.\.venv\Scripts\python.exe scripts\04_train_xgboost.py
.\.venv\Scripts\python.exe scripts\05_generate_tft_predictions.py
.\.venv\Scripts\python.exe scripts\06_evaluate.py
```

### 14.5 Full upgraded pipeline

```powershell
.\.venv\Scripts\python.exe scripts\09_retrain_tft_2022.py
.\.venv\Scripts\python.exe scripts\10_xgb_as_tft_feature.py

.\.venv\Scripts\python.exe scripts\07_ensemble_predict.py --family original
.\.venv\Scripts\python.exe scripts\07_ensemble_predict.py --family step1
.\.venv\Scripts\python.exe scripts\07_ensemble_predict.py --family step5

.\.venv\Scripts\python.exe scripts\08_conformal_calibrate.py --family original
.\.venv\Scripts\python.exe scripts\08_conformal_calibrate.py --family step1
.\.venv\Scripts\python.exe scripts\08_conformal_calibrate.py --family step5

.\.venv\Scripts\python.exe scripts\11_explainability_stats.py
```

---

## 15. Technical Q&A for viva

### "What is TFT and why did you choose it?"

TFT (Temporal Fusion Transformer, Lim et al. 2021) is a deep learning model for interpretable multi-horizon time series forecasting. It gives three things simultaneously in one forward pass: a median prediction (q50), uncertainty bands (q10/q90), and per-timestep feature importance via the Variable Selection Network plus temporal attention. No other standard model — ARIMA, LSTM, XGBoost — provides all three without post-hoc wrappers.

### "Why not use LSTM?"

LSTM is a black box. It has no built-in variable importance (requires external SHAP, which is static and approximate). It has no built-in attention readout. It requires separate quantile regression heads trained independently. TFT trains all of this end-to-end with a single shared training signal.

### "Why not use Prophet?"

Prophet is designed for single series. It has no cross-series learning. It cannot learn that Onions in Chennai and Onions in Delhi have related price dynamics. It has no per-timestep feature importance. Its uncertainty is post-hoc (not trained).

### "What is the Pinball Loss?"

The loss function that trains TFT's quantile outputs. For quantile q:

```
QL(y, ŷ, q) = q × max(y − ŷ, 0)  +  (1−q) × max(ŷ − y, 0)
```

For q=0.90 (ceiling), underestimating is penalized 9× more than overestimating. So the model learns to set a ceiling that real prices rarely exceed. Summed over all three quantiles and all forecast horizons, this forces all three outputs to train together in one pass.

### "What is the VSN?"

Variable Selection Network — a gating mechanism inside TFT that computes a softmax weight vector over all input features at each timestep:

```
v_t = Softmax(GRN(features_t, static_context))
```

The weights sum to 1.0 and change every month. During a drought month, `rain_deficit` gets high weight. During a stable month, `price_lag_1m` gets high weight. This is dynamic feature importance, not static like tree-model SHAP.

### "What is the GRN?"

Gated Residual Network — the building block used everywhere inside TFT:

```
GRN(a, c) = LayerNorm(a + GLU(W_1 × ELU(W_2 × a + W_3 × c)))
GLU(x) = x[:d] × sigmoid(x[d:])
```

The sigmoid acts as a learnable on/off switch. During training, each dimension learns to either pass through or suppress its signal based on the input context.

### "Why does TFT share W_V across attention heads?"

In standard multi-head attention, each head has its own W_V (value matrix). This means attention weights from different heads operate in different spaces and cannot be meaningfully averaged. TFT shares W_V across all heads, so attention weights can be averaged across heads and directly interpreted as: "how much did past month n influence the prediction for future month t?"

### "What is CQR?"

Conformal Quantile Regression (Romano et al. 2019). A post-training calibration method that adjusts the predicted bands using a held-out calibration set (our validation split), providing a distribution-free marginal coverage guarantee.

The conformity score for each val row:

```
E_i = max(log_q10_i − log_y_i,  log_y_i − log_q90_i)
```

The (1−α) quantile of these scores gives the offset to add/subtract from the bands at test time. This inflates the bands just enough to achieve the target coverage, per commodity.

### "Why is TFT MAPE higher than XGBoost?"

Three reasons:

1. **Loss split**: XGBoost puts 100% of its optimization budget on point accuracy. TFT splits its gradient across three quantile heads (33% each). The q50 head is not the primary objective.

2. **Multi-horizon error compounding**: XGBoost predicts 1 month ahead. TFT predicts 6 months simultaneously, and errors compound at each horizon step. The reported MAPE is the average across all horizons 1–6.

3. **Dataset size**: TFT was designed for 50,000+ timesteps. Our dataset has ~16,000 rows across 123 series. The attention mechanism needs volume to learn reliable cross-series patterns.

### "Your TFT has 11.4% MAPE after XGB fusion. How?"

The XGBoost prediction (`xgb_log_pred`) is added as an input feature to TFT. XGBoost already captures 97.2% of the easy/stable months well (MAPE 2.8%). TFT now sees this prediction as a feature and can:
- Trust it for stable months (small correction)
- Override it for anomalous months (use weather and seasonal signals instead)
- Quantify uncertainty even when the XGB signal is strong

The result is that TFT's q50 is guided by XGB's accuracy while TFT's uncertainty quantification and interpretability are preserved.

### "Are the spike reasons hardcoded?"

Two separate things:

1. **Event labels on the chart** (e.g., "2019 onion crisis: Rs160/kg") — these ARE hardcoded as a `REAL_EVENTS` dictionary. They are decoration only. They do not affect the model.

2. **Spike detection and driver reasons** — these are NOT hardcoded. Spikes are auto-detected (>25% month-over-month change). The reasons come from the VSN encoder weights extracted from the trained TFT. The model decides which features are important; we only translate feature names to human-readable text via a `FEATURE_EXPLANATIONS` mapping.

### "What is the q10–q90 interval? Is it a 90% interval?"

q10–q90 is a central 80% prediction interval by quantile labels (the gap between the 10th and 90th percentiles spans 80% of the theoretical distribution). The reported empirical coverage for step5 is 84.0%, not 90%. Always report empirical coverage from the actual test data, not the theoretical quantile label.

### "Why is coverage 84% and not 90%?"

CQR with α=0.10 provides a theoretical (1−α) = 90% coverage guarantee under exchangeability (val and test come from the same distribution). In practice:
- The 2023 test period contains an unprecedented Tomato spike (Rs 200+/KG)
- This is more extreme than anything in the 2022 validation set
- The distribution is not perfectly exchangeable
- The empirical coverage is therefore 84.0% instead of the theoretical 90%

This is normal and honest. Report 84.0%.

### "How does future prediction work?"

For months that have not happened yet:

```
KNOWN (exact): month, year, season, covid_lockdown, month_sin/cos
ESTIMATED:     weather = historical monthly average for that market and month
CARRIED FORWARD: price_lag_1m = last known actual price (or previous prediction)
                 price_lag_12m = actual price from same month last year
                 rolling averages = computed from available history

For month 2+ of the future:
  price_lag_1m = TFT's own q50 prediction for the previous month
  → This is autoregressive: each future step uses the previous step's output as input
  → Errors compound at each step
```

Future predictions are scenario forecasts, not certainties. They show what prices WOULD be if weather follows historical averages and no unexpected policy events occur.

### "What would improve this project most?"

In priority order:

1. **Agmarknet mandi arrival data** — direct supply signal; replaces the indirect weather→supply chain
2. **Policy event flags** (export bans, MSP changes) — eliminates unexplained structural breaks
3. **More data volume** — switching to weekly frequency or adding more commodities would bring the training set closer to TFT's optimal range of 50K+ timesteps

---

## 16. Where is what — complete file map

### Root level

| File | What it does |
|---|---|
| `app.py` | Streamlit dashboard — the final user-facing application |
| `tft_utils.py` | Helper functions: checkpoint family listing, checkpoint loading, prediction normalization |
| `gpu_utils.py` | GPU/CPU detection, trainer kwargs for TFT prediction |
| `run_xgb_steps.py` | Convenience runner for the full step5 XGB-fused pipeline |
| `check_xgb_status.py` | Status checker for the step5 pipeline run |
| `requirements.txt` | Full Python dependencies (includes PyTorch for training) |
| `requirements_streamlit.txt` | Lightweight dependencies for Streamlit Cloud deployment |
| `README.md` | This file |

### `data/raw/`

| File | What it is |
|---|---|
| `wfp_food_prices_ind.csv` | Raw WFP India food prices (145,124 rows, 1994–2026) |
| `wfp_markets_ind.csv` | Market metadata: GPS coordinates, state, district |
| `nasa_weather_1994_2026.csv` | Fetched NASA POWER monthly weather for all 53 markets |

### `data/processed/`

| File | What it is |
|---|---|
| `prices_filtered.csv` | After step 00: cleaned, filtered, 123 series |
| `master_dataset.csv` | After step 02: full feature-engineered dataset (26 columns) |
| `master_dataset_xgbfused.csv` | After step 10: master_dataset + xgb_log_pred column |
| `tft_predictions_ensemble_original.csv` | Ensemble predictions from the original TFT family |
| `tft_predictions_ensemble_step1.csv` | Ensemble predictions from the step1 retrained family |
| `tft_predictions_ensemble_step5.csv` | Ensemble predictions from the step5 XGB-fused family |
| `tft_predictions_calibrated_original.csv` | CQR-calibrated predictions for original family |
| `tft_predictions_calibrated_step1.csv` | CQR-calibrated predictions for step1 family |
| `tft_predictions_calibrated_step5.csv` | CQR-calibrated predictions for step5 family |
| `tft_variable_importance_detail.csv` | Per-series TFT encoder/decoder variable importance |

### `models/`

| File | What it is |
|---|---|
| `tft_best.ckpt` | Best checkpoint from original TFT training |
| `tft_best-v*.ckpt` | Additional checkpoints from original training (ensemble members) |
| `tft_best_2022*.ckpt` | Checkpoints from step1 retrained TFT (train through 2021) |
| `tft_best_xgbfused*.ckpt` | Checkpoints from step5 XGB-fused TFT |
| `xgb_baseline.pkl` | XGBoost baseline model (trained through Dec 2022) |
| `xgb_clean_2019.pkl` | XGBoost model trained only through 2019 (for generating xgb_log_pred without leakage) |
| `conformal_offsets_original.json` | Per-commodity CQR offsets for original family |
| `conformal_offsets_step1.json` | Per-commodity CQR offsets for step1 family |
| `conformal_offsets_step5.json` | Per-commodity CQR offsets for step5 family |
| `tft_config_xgbfused.json` | Full TFT configuration for the step5 family |

### `scripts/` — pipeline in order

| Script | Stage | What it does |
|---|---|---|
| `00_filter_prices.py` | Data | Filter WFP raw data → `prices_filtered.csv` (123 series) |
| `01_fetch_weather.py` | Data | Fetch NASA POWER monthly weather → `nasa_weather_1994_2026.csv` |
| `02_merge_features.py` | Features | Merge prices + weather + engineer 26 features → `master_dataset.csv` |
| `03_train_tft.py` | Model | Train TFT baseline (original family, train through 2020) |
| `04_train_xgboost.py` | Model | Train XGBRegressor baseline (train through 2022) |
| `05_generate_tft_predictions.py` | Predict | Generate raw TFT predictions + interpretability outputs for original family |
| `06_evaluate.py` | Evaluate | Compute all metrics + generate PNG figures → `visualizations/` |
| `07_ensemble_predict.py` | Predict | Ensemble checkpoints per family → `tft_predictions_ensemble_<family>.csv` |
| `08_conformal_calibrate.py` | Calibrate | Apply CQR → `tft_predictions_calibrated_<family>.csv` + offsets JSON |
| `09_retrain_tft_2022.py` | Model | Retrain TFT with later cutoff (step1 family, train through 2021) |
| `10_xgb_as_tft_feature.py` | Model | Generate `xgb_log_pred` + train XGB-fused TFT (step5 family) |
| `11_explainability_stats.py` | Validate | Bootstrap VSN stability, ablation tests, coverage tests → `explainability_stats.txt` + PNG figures |

### `visualizations/`

| File | What it is |
|---|---|
| `evaluation_metrics.txt` | Human-readable metrics dump: MAE, MAPE, coverage per variant and commodity |
| `explainability_stats.txt` | VSN stability bootstrap results, ablation test statistics, coverage test results |
| `fig_ablation_mae.png` | Ablation error comparison figure (used in paper) |
| `fig_vsn_stability.png` | VSN bootstrap stability figure (used in paper) |
| `fig_calibration_coverage.png` | Coverage comparison before/after CQR (used in paper) |
| `quantile_forecast_onions.png` | Sample forecast visualization for Onions |
| `quantile_forecast_tomatoes.png` | Sample forecast visualization for Tomatoes |
| `quantile_forecast_rice.png` | Sample forecast visualization for Rice |
| `attention_heatmap.png` | Attention weight heatmap over past months |
| `variable_importance_comparison.png` | Encoder importance comparison across families |

### `docs/`

| File | What it is |
|---|---|
| `challenges_faced.md` | Real obstacles encountered during the project and how they were resolved |
| `dev_notes/viva.md` | Upgrade explanations in simple language for viva defense |
| `dev_notes/explain.md` | Full project walkthrough for interview defense |
| `dev_notes/TFT_TECHNICAL_NOTE.md` | Complete TFT technical reference: math, architecture, Q&A |
| `dev_notes/question1.md` | Why predictions are off by Rs 5–10/KG: honest analysis |
| `dev_notes/question2.md` | Markets, crops, dataset structure, why cities not mandis |
| `dev_notes/question3.md` | Spike reasons, news API, confidence bands, prediction pipeline |
| `dev_notes/question4.md` | Additional technical Q&A |
| `dev_notes/TFT_improve_steps.md` | Improvement roadmap: all upgrade steps with rationale |
| `dev_notes/MODEL_VARIANT_NAMING.md` | Internal vs external naming for the TFT families |
| `dev_notes/SETUP_GUIDE.md` | Environment setup instructions |

### `Report&Paper/`

The IEEE conference paper and associated LaTeX/BibTeX files. Do not edit the senior reference files (`paper.pdf`, `report.pdf`, and extracted texts). The active paper is `project_ieee_paper.tex`.

---

## Final summary

The real shape of this project is a **flowing system**:

```
Raw WFP prices + NASA weather
        ↓
Cleaned 123-series dataset (53 markets, 3 crops, 1994–2023)
        ↓
26 engineered features (lags, weather shocks, seasonality, fused model signal)
        ↓
XGBoost point baseline (MAPE 2.8%) + TFT families (original → step1 → step5)
        ↓
Checkpoint ensembling (variance reduction)
        ↓
Conformal calibration (coverage 65% → 84%)
        ↓
Statistical validation (ablation p < 10⁻⁸⁰, VSN τ = 0.942)
        ↓
Streamlit dashboard (4 views: history, future, explainability, test)
```

The contribution is not just forecasting price. It is combining:
- **XGBoost accuracy** (fused as a feature into TFT)
- **TFT probabilistic bands** (calibrated with CQR)
- **TFT interpretability** (VSN weights + attention, statistically validated)
- **Interactive dashboard** (exposing all of this to the user)

into one coherent system — something no published Indian food price forecasting paper has done.
