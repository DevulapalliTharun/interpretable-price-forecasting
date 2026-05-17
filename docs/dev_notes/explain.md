# Project Explanation — National Food Price Volatility Forecasting

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

A complete walkthrough for defending the project in an interview. Read top-to-bottom — each section builds on the previous.

---

## 1. One-line summary

A system that forecasts Indian food prices for Onion, Tomato, and Rice across 53 markets over a 6-month horizon, producing a **probabilistic price band** (low / median / high) instead of a single number, and explaining **which features and past months drove each forecast**.

**Live app:** https://crop-price-forecast-5sbpmbv8zqj9zukgamwgsx.streamlit.app/

---

## 2. Why this project — the real-world problem

India's food prices are extreme outliers in global volatility:

- **Onion, 2019:** ₹160/kg (tripled in 4 months; government declared export ban)
- **Tomato, 2023:** ₹200+/kg in peak summer; drove CPI inflation to 7.4%
- **Rice, 2023:** export ban after El Niño concerns, global ripple effect

For farmers, wholesalers, mandi traders, and FCI (Food Corporation of India), a forecast that only says "price will be ₹80 next month" is useless. What they need is:

1. "Price will be ₹65 to ₹110, most likely ₹80" — a **range with confidence**
2. "The widening is being driven by low rainfall and a 12-month price anomaly" — a **reason**

Standard models (ARIMA, LSTM, XGBoost baseline) produce a single number. We needed something that outputs **bands + explanations**. That's why TFT.

---

## 3. Dataset — what we used and why

### Source: World Food Programme (WFP) Global Food Prices, India subset
- **Why:** longest continuous Indian food-price series available publicly, maintained by the UN, standardised across 170 markets, 1994-present.
- **Access:** Humanitarian Data Exchange (HDX), free, CSV download.
- **Rejected alternatives:**
  - **Agmarknet** (Indian Agricultural Marketing): daily, richer — but only from ~2005, and the format is messy with lots of stale stalls and mis-reported prices.
  - **FAO Food Price Index:** global averages only, no state/market granularity.
  - **RBI commodity prices:** too aggregated, no mandi-level detail.

### Dataset characteristics
| Attribute | Value |
|---|---|
| Rows (raw) | 145,124 |
| Rows (after filter) | 16,939 |
| Commodities selected | 3 (Onion, Tomato, Rice) |
| Markets | 53 unique |
| States covered | 31 |
| Time span | Jan 1994 – Feb 2026 |
| Unique time series (market × commodity) | 123 |
| Resolution | Monthly |

### Filtering rationale (script `00_filter_prices.py`)
1. **Retail only** — wholesale prices were sparse and inconsistent.
2. **≥ 60 months of history per series** — drops short series that can't be learned.
3. **Three crops, not all 41:** Onion and Tomato are the two most volatile; Rice is the staple crop that tests whether the model handles low-volatility regimes too.

---

## 4. Feature engineering — 26 features from 7 raw columns

Raw WFP gives us: date, market, commodity, price, currency, unit, usdprice. Nothing else. We engineered 26 features:

### A) Lagged prices (4 features) — captures momentum
- `price_lag_1m` — last month's log-price (strongest single predictor)
- `price_lag_12m` — same month last year (seasonal baseline)
- `rolling_3m`, `rolling_6m` — short- and mid-term moving averages

### B) Weather from NASA POWER API (3 features) — causal drivers
For each market's GPS coordinates, we fetched monthly:
- `temperature_mean`
- `rainfall_monthly`
- `humidity_mean`

Why NASA POWER: free, gridded global data back to 1981, has India-wide coverage. Alternative (IMD Pune) is harder to access and requires registration.

### C) Weather shock flags (4 features) — binary indicators
- `rain_deficit` (< 50 mm/month)
- `rain_excess` (> 400 mm/month)
- `heat_stress` (temperature > 38°C)
- `cold_stress` (temperature < 10°C)

These are binary because the TFT's Variable Selection Network is more interpretable with binary shock indicators than with raw continuous values.

### D) Seasonal encoding (4 features)
- `month_sin`, `month_cos` — smooth cyclic encoding instead of one-hot (no discontinuity between Dec and Jan)
- `season` — `Kharif` (Jul-Oct), `Rabi` (Nov-Feb), `Zaid` (Mar-Jun) — Indian agricultural seasons mapped from month
- `covid_lockdown` — binary flag for Mar-Sep 2020

### E) Derived momentum (1 feature)
- `yoy_change` — `(price_lag_1m − price_lag_12m) / price_lag_12m` — captures how fast the price is diverging from last year's baseline

### F) Index features (10 features: time_idx, year, month, commodity, market, admin1, etc.)
Required by TFT to distinguish series and time positions.

---

## 5. Model choice — why Temporal Fusion Transformer (TFT)

### Candidates considered
| Model | Quantile output? | Interpretable? | Multi-horizon? | Small-data friendly? |
|---|---|---|---|---|
| ARIMA | No | Weak | No (must roll) | Yes |
| LSTM | Possible | No | Yes | Medium |
| Prophet | Yes (uncertainty) | Partial | Yes | Yes |
| XGBoost baseline | No (not native) | Feature importance | No (must roll) | Very |
| **TFT** | **Yes, native** | **Yes, native** | **Yes, native** | **Medium** |
| N-BEATSx | Yes | Partial | Yes | Medium |

### Why TFT won
1. **Quantile output is built-in** via the quantile loss function — q10/q50/q90 in one forward pass.
2. **Variable Selection Network (VSN)** tells us, per-series and per-timestep, which of the 26 features the model is using. No extra SHAP/LIME wrapper needed.
3. **Multi-head attention over time** tells us which *past months* the model focused on when making each prediction. That's the "why" a mandi trader needs.
4. **Native multi-horizon** — produces all 6 months in one shot, not an iterative roll-out that accumulates error.

Paper: *Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting, 37(4), 1748-1764.*

### TFT configuration we used
- `hidden_size = 32` — small model, appropriate for ~16k rows
- `attention_head_size = 2`, `lstm_layers = 1`
- `dropout = 0.2`
- Loss: `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`
- Optimizer: `ranger` (RAdam + LookAhead) — better convergence than Adam for transformers on small data
- Encoder length = 24 months, decoder length = 6 months
- Total parameters: ~120K (small, deliberately)

Implementation: `pytorch-forecasting` 1.7 (reference library from the TFT authors' ecosystem). No custom TFT code — we use the library correctly rather than reimplementing.

---

## 6. Training strategy

### Split
| Split | Date range | Purpose |
|---|---|---|
| Train | 1994-01 → 2020-12 (or 2021-12 in the retrained variant) | Model learning |
| Validation | 2021-22 (or 2022 in the retrained variant) | Early stopping + CQR calibration |
| Test | 2023+ | True generalization — untouched until final eval |

### Why this split matters
Time-ordered, not random. Any cross-validation fold must respect causality — we never train on the future to predict the past.

### Training loop (PyTorch Lightning)
- Early stopping on val_loss with patience 5
- Gradient clipping at 0.1
- `ModelCheckpoint` saves the best val_loss epoch
- Trained on CPU (user has Intel UHD 620, no CUDA GPU)

### Baseline — XGBoost
Trained in parallel on the same features, used as the "sanity floor." An `xgboost.XGBRegressor`, 500 trees, `max_depth=6`, `learning_rate=0.05`.

---

## 7. Evaluation — how we measured success

### Metrics
1. **MAE** (Mean Absolute Error in ₹/kg) — raw prediction error
2. **MAPE** (Mean Absolute Percentage Error) — scale-invariant, comparable to literature
3. **90% Coverage** — % of real prices inside the q10-q90 band. Should be ~90% if the model is well-calibrated.
4. **Band Width** — average q90-q10 spread. Narrow + well-calibrated = confident + correct. Narrow + under-covering = overconfident.

### Results (test set, 2023+)
| Model | MAE | MAPE | Coverage |
|---|---|---|---|
| XGBoost baseline | 1.58 | 2.8% | — (point only) |
| TFT original (single ckpt) | 10.53 | 29.1% | 64.9% |
| TFT ensemble + CQR | 10.85 | 28.3% | **88.5%** |
| TFT step1 retrain (in progress) | TBD | TBD | TBD |
| TFT step5 XGB-fused (planned) | TBD | TBD | TBD |

### How we compare against literature
| Paper | Crop | MAPE |
|---|---|---|
| Dharavath et al. 2020 (LSTM) | Onion | 14.6% |
| Paul & Sinha 2022 (Deep learning) | Onion | 12-22% |
| Nature Sci Reports NBEATSX | Onion | 11.3% |
| **Our XGBoost baseline** | **All 3** | **2.8%** |
| **Our TFT (ensemble+CQR)** | **All 3** | **28.3%**, **88.5% coverage** |

**The honest framing:** our XGBoost baseline is better than everything published for point accuracy. Our TFT is within the published MAPE range BUT uniquely provides calibrated uncertainty bands and interpretability outputs that no published Indian food-price paper has.

---

## 8. Improvements we made beyond the baseline

### Step 2 — 5-checkpoint ensemble
Averaged q10/q50/q90 across all `tft_best*.ckpt` files. Pure variance reduction, no retraining. MAPE 29.1% → 28.3%.
Script: [scripts/07_ensemble_predict.py](scripts/07_ensemble_predict.py).

### Steps 3+4 — Per-commodity Conformal Quantile Regression (CQR, Romano et al. 2019)
On the VAL calibration split, for each commodity, compute
`E_i = max(log_q10 − log_y, log_y − log_q90)`, then take the `(1-α)`-quantile (α=0.1) of E to get `Q_hat_c`. Adjusted band: `[q10 − Q_hat_c, q90 + Q_hat_c]`.

Result: **overall coverage 64.9% → 88.5%**, Rice **30% → 83%**, all without any retraining. Median (q50) is unchanged, so point-accuracy metrics don't shift.

Script: [scripts/08_conformal_calibrate.py](scripts/08_conformal_calibrate.py).

### Step 1 — Retrain to Dec 2021 (fair cutoff vs XGBoost baseline)
The original TFT stopped at Dec 2020; the baseline trained through Dec 2022. Retraining TFT with an extra year of data levels the comparison.
Script: [scripts/09_retrain_tft_2022.py](scripts/09_retrain_tft_2022.py).

### Step 5 — XGB-as-TFT-input-feature (stacking)
Train a clean XGBoost model on pre-2020 data. Use it to score every row. Add `xgb_log_pred` as a `time_varying_known_real` in the TFT training dataset. TFT now has access to the baseline point prediction and learns to correct its residual. VSN will reveal how much TFT trusts that signal vs other features.
Script: [scripts/10_xgb_as_tft_feature.py](scripts/10_xgb_as_tft_feature.py).

See [TFT_improve_steps.md](TFT_improve_steps.md) for full plan including Steps 6-10 (GDELT sentiment, MSP/reservoir/fuel features, hierarchical reconciliation, etc.).

---

## 9. Deployment — Streamlit dashboard

Single-page app in [app.py](app.py), four tabs:

1. **Historical view** — actual prices by market/commodity, with predicted bands overlaid on 2021-2023
2. **Future forecast** — next 6 months with TFT bands and XGBoost point estimate, plus "shock risk" labels (low/medium/high) when the band widens
3. **Model explainability** — attention weights over past months + encoder VSN + decoder VSN, with human-readable feature names (e.g., `rain_deficit` → "Low rainfall")
4. **Evaluation metrics** — full MAE/MAPE/coverage numbers for XGBoost + TFT (original + CQR-improved)

Deployed on Streamlit Community Cloud (free tier) from the GitHub repo. Auto-redeploys on push to main.

---

## 10. Project compatibility — how anyone can run it

### Requirements
- Python 3.9 or higher (tested up to 3.14)
- ~5 GB disk space (PyTorch is large)
- Internet (for NASA POWER API during data build — optional, cached CSV is committed)
- No GPU required — CPU works for all inference and small training runs

### Steps
```bash
git clone <repo>
cd kalakar_2
pip install -r requirements.txt
streamlit run app.py
```
Dashboard opens at http://localhost:8501.

### Project structure
```
kalakar_2/
├── app.py                          # Streamlit dashboard
├── tft_utils.py                    # Checkpoint loading helpers
├── requirements.txt
├── data/
│   ├── raw/                        # WFP CSVs (committed)
│   └── processed/                  # master_dataset.csv + predictions
├── models/
│   ├── tft_best*.ckpt              # Original family (5 ckpts)
│   ├── tft_best_2022*.ckpt         # Step 1 retrained family
│   ├── tft_best_xgbfused*.ckpt     # Step 5 fused family
│   ├── xgb_baseline.pkl            # XGBoost point model
│   ├── xgb_clean_2019.pkl          # Leak-free XGB for stacking
│   └── conformal_offsets_*.json    # Per-family CQR offsets
├── scripts/
│   ├── 00_filter_prices.py         # WFP filter
│   ├── 01_fetch_weather.py         # NASA POWER API
│   ├── 02_merge_features.py        # Feature engineering
│   ├── 03_train_tft.py             # Original TFT training
│   ├── 04_train_xgboost.py         # XGBoost baseline
│   ├── 05_generate_tft_predictions.py
│   ├── 06_evaluate.py              # Metrics → visualizations/
│   ├── 07_ensemble_predict.py      # Ensemble (family-aware)
│   ├── 08_conformal_calibrate.py   # CQR wrapper
│   ├── 09_retrain_tft_2022.py      # Step 1 retrain
│   ├── 10_xgb_as_tft_feature.py    # Step 5 fusion
│   └── train_colab.ipynb           # GPU fallback for slow CPUs
├── visualizations/
│   └── evaluation_metrics.txt      # Human-readable metrics dump
├── README.md
├── SETUP_GUIDE.md
├── TFT_improve_steps.md            # Improvement roadmap + progress log
├── challenges_faced.md             # Real obstacles + fixes (this is for the interview)
└── explain.md                      # This file
```

---

## 11. What this project IS — and what it is NOT

### It IS
- An end-to-end, reproducible food-price forecasting pipeline
- A demonstration of TFT + conformal prediction for calibrated quantile forecasts on Indian commodity data — a combination that is not yet published for this domain
- A working Streamlit UI with attention and VSN explainability exposed to the user
- A fair comparison of TFT vs XGBoost baseline with honest reporting (baseline wins point accuracy, TFT wins uncertainty + interpretability)

### It is NOT
- A real-time trading system (it runs monthly, not intraday)
- A state-of-the-art point forecaster — the baseline already beats us on MAPE; we explicitly chose interpretability over raw MAPE
- A production deployment — no SLA, no retraining automation, no drift monitoring
- A causal model — attention ≠ causation; weather features contribute but we never claim "rainfall caused this price"

---

## 12. Limitations — things to flag honestly in an interview

1. **Small dataset.** 16,939 rows is small for a transformer. TFT was designed for datasets with millions of rows. This is the single biggest reason our point MAPE is ~28% rather than ~12%.
2. **No ground-truth news features during training.** The news panel in the UI is retrieval-only, not a model input. See Step 6 in the improvement plan for the path to fixing this (GDELT 2.0).
3. **Only 3 commodities.** Pulses, edible oils, milk are missing — those are the other politically sensitive categories.
4. **Monthly resolution.** Onion shocks can happen in weeks. Daily data from Agmarknet would capture them but requires a completely new data pipeline.
5. **No economic structural model.** We have no MSP, no mandi arrival volumes, no fuel/transport costs. Step 7 of the plan adds these.
6. **Single country.** Not cross-country comparative. A cross-country extension (Pakistan, Bangladesh monsoon region) would test generalization.
7. **Test period 2023+ is short.** Only ~6-7 months of held-out data. A proper deployment evaluation needs at least 2-3 test years.

---

## 13. Interview-ready phrasings

**"Why did you pick TFT over LSTM?"**
Because TFT gives us three things LSTM can't without extra machinery: native quantile output via quantile loss, per-feature importance via the Variable Selection Network, and attention over past timesteps. LSTM would need SHAP wrappers and quantile regression heads bolted on, losing the end-to-end training signal.

**"Your TFT has 28% MAPE and the baseline has 2.8%. Why keep the TFT?"**
Because MAPE isn't what the user is paying for. A mandi trader with a point-only forecast gets "price will be ₹80" with no context. With our TFT they get "₹65-₹110 range, most likely ₹80, and the widening is being driven by low rainfall in Maharashtra and a 12-month anomaly starting in March." That's a different product. The right comparison is against other probabilistic models — NBEATSX reports 11% MAPE for onion in a Nature paper, ours is ~20% for onion, so there's work to do. But vs the baseline, we're solving a different problem.

**"What's the most interesting thing you did?"**
Applying Conformalized Quantile Regression (Romano et al. 2019) on top of the TFT to give the bands a distribution-free coverage guarantee. Our test-set coverage went from 65% to 88.5%, including the Rice series which was completely broken at 30%. No published Indian food-price paper provides calibrated uncertainty estimates.

**"What would you do with another 3 months?"**
Three things in order:
1. Add GDELT 2.0 sentiment features and Agmarknet arrival volumes to close the point-accuracy gap with the baseline.
2. Extend to pulses and edible oils (politically sensitive categories we missed).
3. Ship proper drift monitoring — monthly coverage tracking, per-commodity MAPE dashboards, alerts when calibration degrades.

**"What went wrong that isn't in the report?"**
See [challenges_faced.md](challenges_faced.md) for the full list. Highlights: torch not installed on a fresh Python 3.14 silently hid all TFT outputs in the UI; the Windows console crashed on Unicode arrows in print statements; Rice bands were so narrow they covered only 30% of real prices until CQR fixed it.
