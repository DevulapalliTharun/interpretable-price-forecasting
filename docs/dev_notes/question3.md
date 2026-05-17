# Question 3: Reasons Behind Prices, News API, Confidence Bands, and How Prediction Actually Works

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

---

## 1. Are the Price Spike Reasons Hardcoded?

### Two separate things on the dashboard — one is hardcoded, one is NOT:

**HARDCODED (event labels on the chart):**
```python
REAL_EVENTS = {
    "2010-12-01": "Onion crisis: Rs85/kg",
    "2019-12-01": "Onion: Rs160/kg, imports",
    "2023-07-15": "Tomato: Rs200+/kg",
    ...
}
```
These are just **labels drawn on the chart** for visual reference. They don't affect the model. They are there so the viewer can see "oh, this spike was the 2019 onion crisis." We added them manually because we know these events happened.

**NOT HARDCODED (spike detection + reasons):**
```
[!] July 2023: SPIKE +95% (Rs 92 -> Rs 180/KG)

Model-identified drivers:
  ████████████  Rain deficit         35%
  ████████      Heat stress          22%
  ██████        Price momentum       18%
  ████          Seasonal pattern     12%
```

This part is **entirely model-driven**. Here's exactly how:

### Step 1: Auto spike detection (no hardcoding)
```python
def detect_spikes(price_series, threshold=0.25):
    # For each month, calculate: (this_month - last_month) / last_month
    # If change > 25%, flag as spike
    # If change < -25%, flag as drop
```
The code scans ALL prices automatically. It doesn't know about the 2019 crisis or 2023 spike. It just finds months where price changed more than 25%. If we add new data with new spikes, they'll be detected automatically.

### Step 2: VSN weight extraction (from the trained model)
```python
# The TFT model has internal weights called "encoder_variables"
# These weights are LEARNED during training, not set by us
interp = model.interpret_output(raw.output, reduction="sum")
encoder_weights = interp["encoder_variables"]
# Returns: [0.35, 0.22, 0.18, 0.12, 0.05, ...]
#           rain    heat   lag    season  ...
```
These weights come from the **Variable Selection Network (VSN)** inside TFT. During training, the VSN learned:
- "When rainfall is low, rain_deficit should get high weight"
- "When prices are already rising, price_lag_1m should get high weight"

We did NOT tell it this. It learned from 25 years of price-weather patterns.

### Step 3: Human-readable translation (mapping table)
```python
FEATURE_EXPLANATIONS = {
    "rain_deficit": ("Low rainfall", "Below-normal rainfall — historically linked to supply shortages"),
    "heat_stress":  ("Extreme heat", "Temperature exceeded 38C — causes crop stress"),
    "price_lag_1m": ("Price momentum", "Last month's price — ongoing trend continuation"),
    ...
}
```
This mapping converts `rain_deficit` to `"Low rainfall"`. The mapping IS manually written by us — but the **weight (35%)** comes from the model. We just translate the feature name to English. If the model gives rain_deficit weight = 0.01, it won't show up in the reasons — the model decides what matters, not us.

### Summary: What's hardcoded vs learned

| Component | Source | Hardcoded? |
|---|---|---|
| Event labels ("2019 onion crisis") | Manual | YES (for chart decoration only) |
| Spike detection (>25% change) | Algorithm | NO (auto-scans all prices) |
| Which features are important | TFT VSN weights | NO (learned from data) |
| How much weight each feature gets | TFT VSN weights | NO (learned from data) |
| Feature name translations | Manual mapping | YES (just English labels) |

---

## 2. How the News API Works

### What we use
```python
from gnews import GNews
```
GNews is a Python library that searches **Google News** programmatically. No API key needed. Free.

### How we search
```python
def search_news(commodity, market):
    gn = GNews(language="en", country="IN", max_results=3)
    query = f"{commodity} price {market} India"
    articles = gn.get_news(query)
```

### What attributes we use to search

| Attribute | Value | Example |
|---|---|---|
| `language` | "en" | English articles only |
| `country` | "IN" | India-specific news |
| `max_results` | 3 | Top 3 articles |
| `query` | `"{commodity} price {market} India"` | `"Onions price Chennai India"` |

### What we get back

```python
# Each article has:
{
    "title": "Onion prices soar to Rs 160 as rains destroy Maharashtra crop",
    "publisher": {"title": "The Hindu"},
    "published date": "2023-07-15",
    "url": "https://..."
}
```

### What we display
```
📰 Related News:
  "Onion prices soar to Rs 160 as rains destroy Maharashtra crop"
   — The Hindu | Jul 15, 2023
```

### Important: News is for CONTEXT, not for the model

The news search happens **when the user clicks a button** on the dashboard. It searches Google News LIVE at that moment. The results are NOT used by the model for prediction. They are shown so the user can see "oh, this is what was actually happening in the news during this price spike."

The model predicts based on weather + price history. The news gives human-readable context for WHY.

### Limitations of news search
- Only English articles (misses Hindi/regional news)
- GNews is unofficial — Google may rate-limit
- Searches current Google News index — very old articles (2010) may not be indexed
- No sentiment analysis on the results (we just show headlines)

---

## 3. What is Special About TFT That It Can Reason?

### The key: Variable Selection Network (VSN)

Most ML models treat all features equally:
```
LSTM:     input = [rain, temp, lag_price, season, ...]  → all go into same LSTM cell
XGBoost:  input = [rain, temp, lag_price, season, ...]  → tree splits on any feature
```

TFT adds a **gating step BEFORE** processing:
```
TFT:      input = [rain, temp, lag_price, season, ...]
              ↓
          VSN gate: [0.35, 0.05, 0.22, 0.12, ...]  ← learned weights, sum to 1.0
              ↓
          weighted_input = 0.35*rain + 0.05*temp + 0.22*lag + 0.12*season + ...
              ↓
          LSTM processes this weighted version
```

The VSN gate **changes per timestep**:
```
January (stable month):    lag_price=0.45, rolling_6m=0.20, season=0.15, rain=0.02
July (volatile month):     rain_deficit=0.35, heat_stress=0.22, lag=0.18, season=0.12
COVID lockdown (Mar 2020): covid_flag=0.40, lag_price=0.25, season=0.10, rain=0.05
```

This is WHY TFT can "reason" — it literally assigns a numerical importance to each feature for each time point. No other standard model does this per-timestep.

### How attention adds temporal reasoning

After VSN selects features, the attention mechanism asks: "which PAST MONTHS should I focus on?"

```
To predict Onions_Chennai for Aug 2023:
  Attention weights over past 24 months:
    Aug 2022: 0.15  ← "same month last year had similar pattern"
    Jul 2023: 0.25  ← "last month's price is most relevant"
    Mar 2023: 0.10  ← "Rabi harvest month influenced current supply"
    Jul 2019: 0.08  ← "2019 crisis year had similar weather pattern"
    Other months:    ← low weights (not very relevant)
```

This gives TEMPORAL reasoning — not just "which features" but "which time periods."

### Combined: TFT's reasoning output

```
PREDICTION: Onions_Chennai, Aug 2023 → Rs 45/KG [Rs 28 - Rs 68]

WHY this price? (VSN weights):
  - Rain deficit is the dominant factor (35%)
  - Heat stress contributing (22%)
  - Prices were already rising (18%)

WHEN did similar patterns occur? (Attention):
  - Model focused on Aug 2022 and Jul 2019
  - Both were pre-monsoon shortage periods

HOW confident? (Quantile band):
  - Band width = Rs 40 (68 - 28)
  - This is WIDE → model is UNCERTAIN
  - Risk level: HIGH
```

No other model gives you all three simultaneously.

---

## 4. Confidence Band — What Are q10 and q90?

### Simple explanation

Imagine you predict "tomorrow's temperature will be 35 degrees." That's one number. But how confident are you?

TFT says: "tomorrow's temperature will be **between 30 and 40 degrees**, most likely **35 degrees**."

```
q10 = 30 degrees  (10% chance it's BELOW this — very unlikely to be colder)
q50 = 35 degrees  (median — most likely value)
q90 = 40 degrees  (10% chance it's ABOVE this — very unlikely to be hotter)
```

For food prices:
```
Onions_Chennai, Aug 2023:
  q10 = Rs 28/KG   (floor — 90% chance actual price is above this)
  q50 = Rs 45/KG   (median — most likely price)
  q90 = Rs 68/KG   (ceiling — 90% chance actual price is below this)
  
  Band width = 68 - 28 = Rs 40 → HIGH UNCERTAINTY
```

### How TFT learns these bands

The model has **three separate output heads** (three linear layers at the end):
```
Linear_q10(features) → predicts the floor
Linear_q50(features) → predicts the median
Linear_q90(features) → predicts the ceiling
```

Each head is trained with a different **Pinball Loss**:

```
For q90 head (ceiling):
  If actual price = Rs 70, and q90 predicted Rs 50:
    PENALTY = 0.90 × (70 - 50) = Rs 18  ← HEAVY penalty for underestimating
  
  If actual price = Rs 40, and q90 predicted Rs 50:
    PENALTY = 0.10 × (50 - 40) = Rs 1   ← light penalty for overestimating

Result: q90 head learns to OVERESTIMATE — sets a high ceiling.
        It's heavily punished for missing spikes, lightly punished for being too high.

For q10 head (floor):
  Opposite — heavily punished for overestimating, lightly for underestimating.
  Learns to set a low floor.
```

### What makes the band wide or narrow

The band width (q90 - q10) is determined by the **variance the model learned** for that specific situation:

```
NARROW BAND (model is confident):
  Rice_Chennai, Jan 2023: Rs 42 [Rs 38 - Rs 46]  Band = Rs 8
  Why? Rice prices are stable. MSP sets a floor. Low variance in training data.

WIDE BAND (model is uncertain):
  Onions_Chennai, Jul 2023: Rs 45 [Rs 28 - Rs 68]  Band = Rs 40
  Why? Onion prices historically spike in Jul-Aug (pre-Kharif shortage).
       Training data showed high variance in this month. Model learned to hedge.
```

The features that influence band width:
- **Season**: Kharif months → wider bands (monsoon uncertainty)
- **Recent volatility**: If rolling_3m shows high variance → wider bands
- **Weather shocks**: If rain_deficit=1 → wider bands (supply risk)
- **Commodity type**: Onions always wider than Rice (learned from data)

### What "90% coverage" means

```
If we make 100 predictions with 90% bands:
  - IDEAL: 90 of 100 actual prices fall inside the band
  - Our model: ~60-80 of 100 fall inside (varies by crop)
  
Coverage < 80%: bands are too NARROW (model is overconfident)
Coverage > 95%: bands are too WIDE (model is too cautious, bands are useless)
Coverage 80-90%: GOOD calibration
```

---

## 5. What Features Are Used for Prediction (Test Time AND Future)

### Three categories of features

**Category 1: STATIC (same for entire series)**
```
commodity = "Onions"          ← which crop
market    = "Chennai"         ← which city
admin1    = "Tamil Nadu"      ← which state

These initialize the LSTM memory. Onions_Chennai starts from a different
internal state than Rice_Delhi. The model learned different "priors" for
each crop-market combination.
```

**Category 2: KNOWN FUTURE (we know these for any future month)**
```
time_idx       = 354           ← month counter (Jan 1994 = 0)
year           = 2023          ← calendar year
month          = 8             ← August
month_sin      = 0.866         ← sin(2π × 8/12) — cyclical encoding
month_cos      = -0.5          ← cos(2π × 8/12) — cyclical encoding
season         = "Kharif"      ← Indian agricultural season
covid_lockdown = 0             ← binary flag (only 1 for Mar-Sep 2020)

We know these exactly for any future month because they are calendar-based.
The model uses these to understand WHEN it's predicting:
  "August = Kharif = monsoon = historically volatile for vegetables"
```

**Category 3: UNKNOWN PAST (only available from history, not for future)**
```
log_price         = 3.24       ← log(1 + Rs 24.5)
temperature_mean  = 29.5°C     ← from NASA POWER (observed)
rainfall_monthly  = 2.9mm      ← from NASA POWER (observed)
humidity_mean     = 68.3%      ← from NASA POWER (observed)
price_lag_1m      = 3.18       ← last month's log_price
price_lag_12m     = 3.56       ← same month last year's log_price
rolling_3m        = 3.22       ← 3-month average log_price
rolling_6m        = 3.35       ← 6-month average log_price
yoy_change        = -0.11      ← year-over-year price change rate
rain_deficit      = 1          ← binary: rainfall < 50mm? (YES)
rain_excess       = 0          ← binary: rainfall > 400mm? (NO)
heat_stress       = 0          ← binary: temperature > 38°C? (NO)
cold_stress       = 0          ← binary: temperature < 10°C? (NO)
```

### How prediction works for TEST DATA (2023-Jul 2023)

For the test period, we have ACTUAL values for all features because these months already happened:

```
Test month: Onions_Chennai, Mar 2023

KNOWN:   month=3, season=Zaid, year=2023, covid=0
ACTUAL:  temperature=28.4°C, rainfall=0.5mm, price_lag_1m=Rs 22
         (we KNOW these because March 2023 already happened)

TFT receives ALL of these as input → produces prediction.
We compare prediction vs actual price.
```

The model is NOT cheating — it was trained on data up to Dec 2020. It has never seen 2023 prices during training. But it does see 2023 weather and lag prices at test time because those are "unknown past" features that become known once the month passes.

### How prediction works for FUTURE (beyond Jul 2023)

For months that haven't happened yet, we DON'T have actual weather or prices. Here's what the dashboard does:

```
Future month: Onions_Chennai, Aug 2023 (beyond last data)

KNOWN (exact):
  month = 8
  season = "Kharif"
  year = 2023
  covid_lockdown = 0
  month_sin = sin(2π × 8/12) = 0.866
  month_cos = cos(2π × 8/12) = -0.5

ESTIMATED (from historical averages):
  temperature = 28.9°C   ← average of all past Augusts in Chennai
  rainfall    = 4.4mm    ← average of all past Augusts in Chennai
  humidity    = 72.8%    ← average of all past Augusts in Chennai

CARRIED FORWARD:
  price_lag_1m  = Jul 2023 actual price (Rs 25.2 — last known)
  price_lag_12m = Aug 2022 actual price (known from history)
  rolling_3m    = average of May, Jun, Jul 2023 prices
  rolling_6m    = average of Feb-Jul 2023 prices
  yoy_change    = (Jul 2023 price - Jul 2022 price) / Jul 2022 price

DERIVED:
  rain_deficit  = 1 if estimated rainfall < 50mm, else 0
  heat_stress   = 1 if estimated temperature > 38°C, else 0
```

For month 2 of the future forecast (Sep 2023):
```
  price_lag_1m  = TFT's own prediction for Aug 2023 (q50)
                  ← uses its own previous output as input!
  This is called AUTOREGRESSIVE forecasting.
  Error compounds: month 1 error feeds into month 2.
```

### The key honest point for the professor

```
TEST TIME:     Model sees ACTUAL weather + ACTUAL lag prices → fair evaluation
FUTURE TIME:   Model sees ESTIMATED weather + ITS OWN predictions → scenario forecast

Future predictions are NOT as reliable as test predictions because:
1. Weather is estimated from historical averages, not actual observations
2. Each month uses the previous month's prediction as input (error compounds)
3. We don't know about future export bans, strikes, or policy changes

This is why we call it a "scenario forecast" — it shows what prices WOULD be
if weather follows historical patterns and no unexpected policy events occur.
```

---

## 6. The Complete Prediction Pipeline (Visual)

```
HISTORICAL DATA (known)               FUTURE (estimated)
═══════════════════════                ═══════════════════
Jan 2022 ─── weather + price ──┐
Feb 2022 ─── weather + price ──┤
Mar 2022 ─── weather + price ──┤
...                            ├──→ ENCODER (18 months lookback)
Dec 2022 ─── weather + price ──┤      ↓
Jan 2023 ─── weather + price ──┤    LSTM processes all 18 months
...                            ├    VSN selects important features
Jun 2023 ─── weather + price ──┤    Attention focuses on key months
Jul 2023 ─── weather + price ──┘      ↓
                                    DECODER (6 months ahead)
                                      ↓
Aug 2023 ─── month=8, Kharif ──────→ Prediction 1: Rs 45 [28-68]
Sep 2023 ─── month=9, Kharif ──────→ Prediction 2: Rs 48 [25-72]
Oct 2023 ─── month=10, Kharif ─────→ Prediction 3: Rs 42 [22-65]
Nov 2023 ─── month=11, Rabi ───────→ Prediction 4: Rs 35 [20-52]
Dec 2023 ─── month=12, Rabi ───────→ Prediction 5: Rs 30 [18-45]
Jan 2024 ─── month=1, Rabi ────────→ Prediction 6: Rs 28 [16-42]
                                      ↑
                               Known future features only
                               + historical weather averages
                               + model's own previous predictions

Each prediction comes with:
  q10 (floor) ─── q50 (median) ─── q90 (ceiling)
                     ↑
              "Most likely price"
```

---

## 7. Quick Answers for the Professor

| Question | Answer |
|---|---|
| Are spike reasons hardcoded? | Event LABELS are hardcoded (for chart decoration). Spike DETECTION and REASONS are model-driven (VSN weights). |
| What does the news API search? | Google News for `"{crop} price {city} India"`. Shows top 3 headlines. For context only — not used by the model. |
| Why can TFT give reasons? | Variable Selection Network assigns learned softmax weights to each feature per timestep. Changes dynamically. |
| What is q10? | 10th percentile — price floor. 90% chance actual is above this. |
| What is q90? | 90th percentile — price ceiling. 90% chance actual is below this. |
| What determines band width? | Learned variance from training data. Volatile crops/months → wide. Stable → narrow. |
| What features for test prediction? | All 26 features (actual weather + actual lag prices). Fair evaluation. |
| What features for future? | Known future (calendar) + estimated weather (historical avg) + autoregressive lag prices. Scenario forecast. |
| Why is future less accurate? | Weather is estimated, lag prices are the model's own predictions (error compounds), no policy events. |
| Is the model predicting from weather? | Partially. Weather is one input. Price lags, season, and trend are equally or more important depending on the crop and month. |
