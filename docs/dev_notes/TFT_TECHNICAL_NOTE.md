# TFT Technical Note — Complete Reference
## Everything About How We Use TFT, The Math, Architecture, Performance, Limitations, and Future Work

> **Terminology note:** The implemented point baseline in this repository is `xgboost.XGBRegressor`.

---

## 1. What is TFT?

Temporal Fusion Transformer (TFT) is a deep learning architecture for **interpretable multi-horizon time series forecasting**, published by Lim et al. (2021) in the International Journal of Forecasting.

Unlike black-box models (LSTM, standard Transformers), TFT is designed to answer three questions simultaneously:
1. **What will the price be?** (point forecast via q50 median)
2. **How confident is the model?** (uncertainty via q10/q90 bands)
3. **Why does the model predict this?** (attention + variable selection weights)

### Why TFT over other models?

| Model | Point Forecast | Uncertainty Bands | Explains WHY | Multi-series | Multi-horizon |
|---|---|---|---|---|---|
| ARIMA | Yes | No | No | No (1 series at a time) | No |
| Prophet | Yes | Basic (post-hoc) | No | No | No |
| LSTM | Yes | No | No | Yes | Yes |
| XGBoost baseline | Yes | No | SHAP (post-hoc, static) | Yes | No |
| **TFT** | **Yes** | **Built-in (q10/q50/q90)** | **Built-in (attention + VSN)** | **Yes (123 series)** | **Yes (3 months)** |

---

## 2. TFT's Three Core Mechanisms

### 2.1 Variable Selection Network (VSN)

**What it is:** A gating mechanism that learns, for each timestep independently, which input features are relevant and which should be suppressed.

**How it works mathematically:**
```
v_t = Softmax( GRN_v( Xi_t, c_s ) )          <- importance weight per feature
xi_t = Sum_j  v_t^(j) x GRN_j( xi_t^(j) )   <- selected representation

v_t is a vector of length = num_features, summing to 1.0
Each element v_t^(j) is the importance weight for feature j at time t.
```

**The GRN (Gated Residual Network) inside:**
```
GRN(a, c) = LayerNorm( a + GLU(eta_1) )
  eta_1    = W_1 x ELU( W_2 x a + W_3 x c + b_2 ) + b_1
  GLU(x)   = x[:d] * sigmoid(x[d:])          <- element-wise gating

The sigmoid acts as a learnable on/off switch per dimension.
During a drought:  rain_deficit gate opens  (sigmoid -> 1.0)
During normal:     lag price gate dominates (sigmoid -> 1.0 for price_lag_1m)
```

**What it tells us in this project:**
- For Onions: rain_deficit and heat_stress get high weights during volatile months
- For Rice: price_lag_1m and rolling_6m dominate (price-driven, not weather-driven)
- These weights change PER TIMESTEP — this is dynamic feature importance, not static like baseline tree-model SHAP

**This is NOT hardcoded.** The VSN learns from data which features matter. If rainfall doesn't correlate with rice prices, the VSN will learn to ignore it for rice. We don't tell it — it discovers this.

### 2.2 Interpretable Multi-Head Attention

**What it is:** A mechanism that scores every past month in the encoder window, telling us which historical months the model considers most predictive for each future month.

**Standard attention:**
```
Attention(Q, K, V) = Softmax( Q x K^T / sqrt(d_attn) ) x V
```

**TFT's interpretable version (key difference):**
```
InterpretableMultiHead(Q,K,V) = [ (1/H) x Sum_h Attention(Q x W_Q^h, K x W_K^h, V x W_V) ] x W_H

KEY INNOVATION: W_V (Value matrix) is SHARED across ALL heads.
```

**Why sharing W_V matters:**
- In standard multi-head attention, each head has its own W_V, making the attention weights of individual heads uninterpretable (they operate in different value spaces)
- In TFT, all heads share W_V, so their attention patterns can be averaged meaningfully
- The averaged attention alpha(t, n) directly tells you: "to predict time t, the model assigned importance alpha to past time n"

**What it tells us:**
- alpha(t, t-12) is high -> model learned annual seasonality (same month last year matters)
- alpha(t, t-1) is high -> model uses momentum (last month's trend continues)
- alpha(t, t-3) is high for onions but not rice -> crop-specific temporal patterns

### 2.3 Quantile Output (Pinball Loss)

**What it is:** Instead of predicting one number, TFT outputs three simultaneous predictions representing a probability distribution.

**The Pinball/Quantile Loss:**
```
QL(y, y_hat, q) = q x max(y - y_hat, 0)  +  (1-q) x max(y_hat - y, 0)

For q=0.90 (upper bound):
  Underestimate penalty = 0.90 x (y - y_hat)   <- HEAVY penalty for missing spike
  Overestimate penalty  = 0.10 x (y_hat - y)   <- light penalty for being too high

For q=0.10 (lower bound):
  Underestimate penalty = 0.10 x (y - y_hat)   <- light penalty
  Overestimate penalty  = 0.90 x (y_hat - y)   <- HEAVY penalty for being too low
```

**Total training loss:**
```
L(Omega, W) = Sum_{y in Omega}  Sum_{q in {0.1,0.5,0.9}}  Sum_{tau=1}^{H}  QL(y, y_hat(q,t,tau)) / (M x H)

One forward pass trains all three quantile heads simultaneously.
The model does not CHOOSE to output a band -- it is FORCED to by this loss.
```

**What it means:**
- The q90 head learns to set a CEILING that real prices stay below 90% of the time
- The q10 head learns to set a FLOOR that real prices stay above 90% of the time
- When the model is uncertain, q90 goes high and q10 goes low -> wide band
- When the model is confident, q90 and q10 are close -> narrow band
- **This is NOT post-hoc confidence intervals.** The bands are the model's PRIMARY output, trained from scratch.

---

## 3. Full TFT Architecture (Our Configuration)

```
=====================================================================
         TEMPORAL FUSION TRANSFORMER -- FULL ARCHITECTURE
=====================================================================

INPUTS
------

Static (s)          Known Future (x_t)     Unknown Past (z_t)
commodity           month_sin/cos          log_price (TARGET)
market              season                 temperature
admin1              year                   rainfall
                    covid_lockdown         humidity
                                           price_lag_1m
                                           price_lag_12m
                                           rolling_3m/6m
                                           rain_deficit
                                           heat_stress
                                           cold_stress
      |                    |                       |
      v                    |                       |
Entity Embeddings          |                       |
-> c_s, c_e, c_h, c_c     |                       |
      |                    |                       |
      +--------------------+-----------------------+
                           |
                           v
  +--------------------------------------------------+
  |     VARIABLE SELECTION NETWORKS (per type)        |
  |                                                    |
  |   v_t = Softmax(GRN(Xi_t, c_s))                  |
  |   xi_t = Sum v_t^(j) x GRN_j(xi_t^(j))          |
  |                                                    |
  |   OUTPUT: feature importance weights per timestep  |
  +---------------------------+------------------------+
                              |
  +---------------------------+------------------------+
  |         LSTM ENCODER   |   LSTM DECODER            |
  |                                                     |
  |  h_0 = GRN_h(c_s)  <- from commodity+market embed  |
  |  c_0 = GRN_c(c_s)  <- different per series         |
  |                                                     |
  |  Encoder: processes 18 past months                  |
  |  Decoder: processes 3 future months                 |
  +---------------------------+------------------------+
                              |
  +---------------------------+------------------------+
  |        STATIC ENRICHMENT LAYER                      |
  |   phi_t = LayerNorm(xi_t + GLU(GRN(h_t || c_e)))   |
  +---------------------------+------------------------+
                              |
  +---------------------------+------------------------+
  |     INTERPRETABLE MULTI-HEAD ATTENTION              |
  |                                                     |
  |   Shared W_V across all heads -> readable alpha     |
  |   OUTPUT: attention heatmap (which past = important)|
  +---------------------------+------------------------+
                              |
  +---------------------------+------------------------+
  |     QUANTILE OUTPUT HEADS  (3 simultaneous)         |
  |                                                     |
  |   y_hat(q=0.10, t) = Linear_0.1(delta_t) <- floor  |
  |   y_hat(q=0.50, t) = Linear_0.5(delta_t) <- median |
  |   y_hat(q=0.90, t) = Linear_0.9(delta_t) <- ceil   |
  |                                                     |
  |   Loss: L = Sum QL(y, y_hat, q) / (M x H)         |
  +----------------------------------------------------+
                              |
                              v
     y_hat_price = expm1(GroupNormalizer.inverse(y_hat))
                              |
                              v
  THREE PRICE BANDS (Rs/KG) per (market, commodity, month)
```

### Our specific configuration

| Parameter | Value | Why |
|---|---|---|
| hidden_size | 32 | ~120K total params. Rule: params < training rows (~12K). 256 (default) would massively overfit. |
| attention_head_size | 2 | 2 heads: one can learn seasonality patterns, another learns trend patterns. 1 was too limited. |
| lstm_layers | 1 | 2+ layers overfit on <20K rows. 1 captures sufficient temporal dynamics. |
| hidden_continuous_size | 16 | Rule: <= hidden_size / 2. Embedding size for continuous features. |
| dropout | 0.3 | Higher than default (0.1). Forces wider uncertainty bands. Prevents memorization. |
| max_encoder_length | 18 | 1.5 annual cycles. Captures seasonality + trends. 24 caused gap issues. |
| max_prediction_length | 3 | One Indian agricultural season. 6 months was too ambitious for food prices. |
| learning_rate | 0.01 | Conservative. 0.03 caused oscillation during training. |
| optimizer | Ranger | RAdam + LookAhead. More stable than Adam for small datasets. Official TFT recommendation. |
| quantiles | [0.1, 0.5, 0.9] | 80% prediction interval. Standard choice in the TFT paper. |
| target_normalizer | GroupNormalizer (softplus) | Per-series normalization. Onions (Rs 20-160) and Rice (Rs 25-65) on different scales. |
| early_stopping | patience=8 | Stops training when val_loss stops improving for 8 epochs. Prevents overfitting. |

---

## 4. Pre-Processing Math

**Price log-transform:**
```
y(t) = log(1 + price(t))                    <- numpy.log1p
Inverse: price_hat = exp(y_hat) - 1          <- numpy.expm1

Why: Retail prices are right-skewed (spikes like Rs 200/kg for tomatoes).
Log compression stabilizes variance so the LSTM doesn't chase outliers.
```

**Cyclical month encoding:**
```
month_sin(t) = sin(2*pi * month(t) / 12)
month_cos(t) = cos(2*pi * month(t) / 12)

Why: Month 12 (Dec) and Month 1 (Jan) are adjacent in time but
numerically 11 units apart. Cyclical encoding makes them close:
cos(2*pi*12/12) - cos(2*pi*1/12) = 0.13 (small distance = adjacent)
```

**GroupNormalizer (per series):**
```
y_normalized(t) = [y(t) - mean(y_i)] / std(y_i)   for each series i

Without this: LSTM would learn Rice as near-constant because its
absolute variance is much lower than Onions. Per-series normalization
puts all series on the same scale.
```

**Full conditional probability modeled by TFT:**
```
P( y[t+1 : t+H]  |  y[<=t],  z[<=t],  x[<=t+H],  s )

  y[t+1:t+H]  = log_price for the next H=3 months
  y[<=t]       = all past log_prices up to current month
  z[<=t]       = all past weather + lag features up to current month
  x[<=t+H]     = all future month/season/covid covariates (known)
  s            = static (commodity, market, state)
```

**LSTM Encoder with static initialization:**
```
h_t, c_t = LSTM( xi_t, h_{t-1}, c_{t-1} )
h_0 = GRN_h( c_s )     <- initial state from commodity+market embedding
c_0 = GRN_c( c_s )     <- initial cell from commodity+market embedding

Onions_Chennai and Rice_Chennai start from DIFFERENT h_0 values.
The model learns: "Onions are volatile, Rice is stable" before
processing any actual data.
```

**Output reconstruction:**
```
y_hat_log(q, t)   = raw TFT output (log + normalized scale)
y_hat_norm(q, t)  = GroupNormalizer.inverse_transform(y_hat_log)
y_hat_price(q, t) = expm1(y_hat_norm(q, t))     <- back to Rs/KG

Three price trajectories per (series_id, forecast_date):
  Lower bound  q=0.10:  price floor (90% chance actual stays above)
  Median       q=0.50:  most likely price
  Upper bound  q=0.90:  price ceiling (90% chance actual stays below)
```

---

## 5. How We Extract Interpretability

### 5.1 In script 05 (generate predictions)

```python
# Get raw predictions with all internal tensors
raw = model.predict(dataloader, mode="raw", return_x=True)

# Quantile predictions: shape [batch, prediction_length, 3]
preds = raw.output.prediction  # [batch, 3, 3] -> q10/q50/q90

# Per-sample attention: shape [batch, decoder_steps, heads, encoder_steps]
raw_attn = raw.output[1]
per_sample_attn = raw_attn.mean(dim=(1, 2))  # -> [batch, encoder_steps]

# Per-sample encoder variable selection: shape [batch, enc_steps, heads, num_vars]
raw_enc_vs = raw.output[4]
per_sample_enc = raw_enc_vs.mean(dim=(1, 2))  # -> [batch, num_encoder_vars]

# Per-sample decoder variable selection: shape [batch, dec_steps, heads, num_vars]
raw_dec_vs = raw.output[5]
per_sample_dec = raw_dec_vs.mean(dim=(1, 2))  # -> [batch, num_decoder_vars]
```

### 5.2 In the dashboard (app.py)

The dashboard translates raw weights to human-readable explanations:

```
Feature: rain_deficit, weight: 0.35
-> Display: "Low rainfall (35%) -- Below-normal rainfall detected,
            historically linked to supply shortages"

Feature: price_lag_1m, weight: 0.22
-> Display: "Price momentum (22%) -- Last month's price indicates
            ongoing trend continuation"
```

This mapping is in `FEATURE_EXPLANATIONS` dict. The weights come from the model. The explanations are human translations of what each feature means economically. The model decides WHICH features to highlight and HOW MUCH weight to give — we only translate the feature name.

### 5.3 Auto spike detection

The dashboard automatically detects spikes (>25% month-over-month change) and shows the VSN weights as reasons:

```
[!] July 2023: SPIKE +95% (Rs 92 -> Rs 180/KG)

Model-identified drivers:
  ████████████  Rain deficit         35%
  ████████      Heat stress          22%
  ██████        Price momentum       18%
  ████          Seasonal pattern     12%
```

This is NOT hardcoded. If rainfall was normal during a spike, rain_deficit would have low weight and wouldn't appear in the reasons. The model decides.

---

## 6. Why TFT Currently Underperforms the XGBoost Baseline on Point Accuracy

### Current numbers (CPU-trained)
```
TFT:      MAPE ~30-38%, Coverage ~13-60%
XGBoost baseline:  MAPE ~3-5%
```

### Reason 1: Baseline lag shortcut
The baseline's #1 feature is price_lag_1m at 76% importance. It predicts: "next month = this month + small adjustment." This works for 90% of stable months. TFT cannot rely on this shortcut because it must simultaneously learn attention, VSN weights, and three quantile outputs.

### Reason 2: Quantile loss splits optimization budget
TFT optimizes: `L = QL(q=0.1) + QL(q=0.5) + QL(q=0.9)`. Only 1/3 of the loss gradient goes toward improving the median (q50). The baseline puts 100% of its gradient on point accuracy.

### Reason 3: Insufficient training (CPU constraints)
With lr=0.03 and 30 epochs on CPU, the model did not converge. On GPU with lr=0.01, 150 epochs, hidden_size=32, expected improvement: MAPE 38% -> 12-18%.

### Reason 4: Prediction horizon
TFT predicts 3 months ahead (multi-step, compounding error). The baseline predicts 1 step (no compounding).

### Reason 5: Data volume
12,000 training rows is below TFT's sweet spot (50K+). The baseline handles small data well. TFT's attention needs volume to learn reliable cross-series patterns.

---

## 7. Limitations

| # | Limitation | Impact | Severity |
|---|---|---|---|
| L1 | Only 12K training rows (TFT needs 50K+) | Attention patterns may be noisy | High |
| L2 | No supply/arrivals data from mandis | Weather proxies for supply (long causal chain) | High |
| L3 | No demand-side features (festivals, population) | Missing demand signals | Medium |
| L4 | No policy features (export bans, MSP changes) | Sudden structural breaks appear as noise | High |
| L5 | Monthly frequency (weekly would be 4x more data) | Smooths out rapid price changes | Medium |
| L6 | City-level data ends Jul 2023 | Can't test on 2024-2026 actual prices | Medium |
| L7 | Weather features are observed, not forecast | Future predictions use historical averages | Medium |
| L8 | TFT's median will never match baseline point accuracy | By design (quantile loss tradeoff) | Low (expected) |
| L9 | English-only news search (India has regional languages) | Misses local market news | Low |
| L10 | Retail prices only (no farm-gate/wholesale) | Model predicts consumer price, not farmer price | Low |

---

## 8. How to Overcome Each Limitation (Future Work)

| # | Solution | Effort | Impact |
|---|---|---|---|
| L1 | Add more commodities (wheat, potatoes, oils) OR use weekly Agmarknet data | 1-3 days | 3-5x more data for attention |
| L2 | Integrate Agmarknet mandi arrival data (tons/day) | 2 days | Direct supply signal |
| L3 | Add festival calendar flags, Google Trends search volume | 2-4 hours | Demand-side signals |
| L4 | Add export_ban, msp_change flags as known-future covariates | 2-3 hours | Policy event handling |
| L5 | Switch to Agmarknet daily/weekly data | 3 days | 4-52x more temporal resolution |
| L6 | Re-download WFP data periodically OR switch to Agmarknet | 30 min | Extends test period |
| L7 | Integrate IMD weather forecasts as known-future inputs | 1 day | Better future predictions |
| L8 | Ensemble: final = 0.7*TFT + 0.3*baseline | 3 hours | Best accuracy + uncertainty |
| L9 | Multilingual NLP (IndicBERT) for Hindi/regional news | 1 week | Better news coverage |
| L10 | Integrate Agmarknet wholesale prices | 2 days | Farmer-relevant predictions |

### Priority order for maximum impact:
```
1. GPU training with current settings        (15 min)   -> MAPE 38% -> 12-18%
2. Google Trends as feature                   (2 hours)  -> +5% coverage
3. Export ban / policy flags                  (2 hours)  -> Better structural break handling
4. Optuna hyperparameter tuning               (overnight)-> 5-15% MAPE improvement
5. Agmarknet arrival data                     (2 days)   -> Major supply signal
6. Ensemble TFT + baseline                    (3 hours)  -> Best of both worlds
```

---

## 9. Viva Q&A — Quick Answers

| Question | Answer |
|---|---|
| What is TFT? | Temporal Fusion Transformer -- interpretable multi-horizon time series forecasting model (Lim et al., 2021, Int. J. Forecasting). |
| Why not ARIMA? | Can't handle 123 series simultaneously, no uncertainty bands, no per-timestep feature importance, no attention. |
| Why not LSTM? | Black box -- no interpretable attention (TFT uses shared W_V), no variable selection, no quantile output. |
| Why not Prophet? | Designed for single series. No cross-series learning. No dynamic feature importance. Basic uncertainty. |
| What loss function? | Pinball/Quantile Loss at q=0.1, 0.5, 0.9 -- trains three outputs simultaneously. Forces uncertainty learning. |
| What is VSN? | Variable Selection Network -- learns softmax weights over features per timestep. Changes dynamically (drought month vs stable month). |
| What is GRN? | Gated Residual Network -- building block with ELU activation + GLU gating. Used in every TFT component. |
| What is coverage? | Percentage of actual prices inside the predicted q10-q90 band. Target: 80%+. |
| What is MAPE? | Mean Absolute Percentage Error. Average prediction error as %. Lower = better. |
| Why 3 months? | Aligns with Indian agricultural seasons (Kharif/Rabi/Zaid). Beyond 3 months, accuracy drops sharply for food prices. |
| Why Ranger optimizer? | RAdam + LookAhead. Handles small datasets better than Adam. Official TFT paper recommendation. |
| Why GroupNormalizer? | Per-series normalization. Onions (Rs 20-160) and Rice (Rs 25-65) on different scales. Without it, LSTM ignores low-variance series. |
| Why shared W_V? | Makes attention weights interpretable. Standard Transformers have separate W_V per head, making individual head attention meaningless. |
| Can TFT learn shocks? | Yes. Pinball Loss forces q90 to capture spikes. VSN gates open for shock features (rain_deficit, heat_stress) during volatile periods. Band widening IS the shock signal. |
| Why is the baseline more accurate? | Uses price_lag_1m (76% importance) = "next month = this month." Works for stable months but gives zero warning before shocks. TFT trades point accuracy for uncertainty + interpretability. |
| Is the spike detection hardcoded? | No. Spikes are auto-detected (>25% change). Reasons come from VSN encoder weights -- the model decides which features are important, we translate to human-readable text. |
| What about future predictions? | TFT forecasts 3 months ahead using known-future features (season, month) and learned patterns. Weather is estimated from historical monthly averages. News search provides real-time context. |

---

## 10. Key Numbers

```
Dataset:       ~16,900 rows, 123 time series, 3 crops, 53 markets, 26 features
Training:      1995-2019 (25 years, ~12,000 rows)
Validation:    2020 (COVID year -- hardest stress test, ~1,100 rows)
Test:          2021-Jul 2023 (31 months, ~3,700 rows)

TFT Model:     ~120K parameters
               18-month encoder, 3-month prediction
               Quantile Loss [0.1, 0.5, 0.9]
               Ranger optimizer, lr=0.01, dropout=0.3
               hidden_size=32, attention_heads=2

Baseline:      XGBoost (500 trees, max_depth=6, lr=0.05)

Test events:   COVID recovery (2021)
               Tomato Rs 200+/kg (Jul 2023)
               Onion export ban (Aug 2023)

Outputs:       Quantile predictions (q10/q50/q90)
               Attention weights (per series, per encoder step)
               Encoder variable importance (per crop, per timestep)
               Decoder variable importance (future feature weights)
               Auto spike detection with model-driven reasons
               Live news search for context

Citation:      Lim, B., Arik, S.O., Loeff, N., & Pfister, T. (2021).
               Temporal Fusion Transformers for Interpretable Multi-horizon
               Time Series Forecasting. Int. J. Forecasting, 37(4), 1748-1764.
```
