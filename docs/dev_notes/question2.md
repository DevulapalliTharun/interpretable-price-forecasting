# Question 2: Markets, Crops, Dataset Structure, and Why We Show Cities Not Mandis

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

---

## 1. The Dataset — How WFP Collects This Data

### Source
The **UN World Food Programme (WFP)** collects retail food prices across India through the **Vulnerability Analysis and Mapping (VAM)** unit, in partnership with the **Government of India's Department of Consumer Affairs**.

### How data is collected
WFP price monitors visit **one designated retail market per city** and record the consumer retail price (Rs/KG) on the **15th of each month**. This is NOT mandi wholesale data — it's the price a consumer pays at a retail shop/market.

### Raw dataset structure

```
File: wfp_food_prices_ind.csv
Rows:         145,124
Columns:      16
Date range:   Jan 1994 – Feb 2026 (32 years)

Column          | Example                  | Meaning
----------------|--------------------------|----------------------------------
date            | 2023-07-15               | 15th of each month (monthly)
admin1          | Tamil Nadu               | State name
admin2          | Chennai                  | District name
market          | Chennai                  | City/market name
market_id       | 923                      | Unique ID (joins to markets CSV)
latitude        | 13.08                    | GPS latitude of market
longitude       | 80.28                    | GPS longitude of market
category        | vegetables and fruits    | Food group
commodity       | Onions                   | Crop name
unit            | KG                       | Unit of measurement
priceflag       | actual                   | All values are "actual" (no estimates)
pricetype       | Retail                   | Retail or Wholesale
currency        | INR                      | Indian Rupees
price           | 42.0                     | Price in Rs/KG
usdprice        | 0.51                     | Price in USD/KG
```

### What "market" means in this dataset

**Each "market" = ONE retail location in ONE city.** It is NOT an average of multiple locations.

```
Delhi      = 1 retail market point (lat=28.67, lon=77.22)
Mumbai     = 1 retail market point (lat=18.98, lon=72.83)
Chennai    = 1 retail market point (lat=13.08, lon=80.28)
Kolkata    = 1 retail market point (lat=22.57, lon=88.37)
Bengaluru  = 1 retail market point (lat=12.96, lon=77.58)
```

**There is NO aggregation.** When the dashboard shows "Onions — Chennai", it is the price at ONE specific retail location in Chennai, not an average of all Chennai markets.

The only exception: Bengaluru has 2 entries ("Bengaluru" and "Bengaluru (east range)") — these are treated as separate series.

---

## 2. Numbers at Each Stage

### Raw data
```
Total rows:         145,124
Markets:            170 (includes 1 "National Average" + 5 Zone aggregates)
Actual city markets: 164
Commodities:        41
States:             31
```

### After filtering
```
Step                              | Rows dropped | Rows remaining | Markets
----------------------------------|-------------|----------------|--------
Raw data                          | —           | 145,124        | 170
Drop "National Average"           | 535         | 144,589        | 169
Keep Retail only (drop Wholesale) | 1,697       | 142,892        | 169
Keep Onions + Tomatoes + Rice     | 118,675     | 24,217         | ~100
Keep KG unit only                 | 0           | 24,217         | ~100
Normalize duplicate markets       | ~50         | ~24,167        | ~99
Drop series < 60 months           | 8,203       | 16,014         | 53
Fill gaps <= 3 months             | +2,401      | 18,415         | 53
Drop series with gaps > 6 months  | varies      | 18,415         | 53

FINAL: 18,415 rows, 53 unique markets, 123 series
  Onions:   38 markets
  Tomatoes: 45 markets
  Rice:     40 markets
  (Some markets have data for 2 or 3 crops = 123 total series)
```

---

## 3. Why We Selected Only 3 Crops

### The 41 commodities in the raw data fall into categories:

**Usable (high volatility, enough data, agricultural):**
| Commodity | Rows | Series >= 60mo | Max CV | Max YoY Shock | Verdict |
|---|---|---|---|---|---|
| **Onions** | 7,557 | 41 | 0.568 | **328.5%** | SELECTED — highest volatility |
| **Tomatoes** | 6,935 | 48 | 0.546 | 173.7% | SELECTED — 2023 spike for presentation |
| **Rice** | 10,558 | 53 | 0.456 | 35.4% | SELECTED — stable baseline for contrast |

**Why NOT these crops:**
| Commodity | Rows | Reason to EXCLUDE |
|---|---|---|
| Wheat | 9,122 | **Government MSP-controlled.** Price is set by policy, not market. Model would learn policy decisions, not supply-demand dynamics. |
| Potatoes | 7,187 | **Cold storage economics dominate.** The key price driver (cold storage levels) is absent from our feature set. Adding potatoes without storage data would add noise. |
| Mustard Oil | 9,112 | **International price-driven.** Palm oil and soybean prices (global markets) set domestic oil prices. Our weather features (India-specific) would be irrelevant. |
| Lentils | 6,575 | **CV = 0.224 — too stable.** No significant price shocks to test TFT's uncertainty learning capability. |
| Sugar | 9,214 | **Government price-controlled** via FRP (Fair and Remunerative Price). Similar to wheat — model learns policy, not market. |
| Salt | 7,149 | **Not agricultural.** Salt is industrial/mineral. Weather features irrelevant. |
| Milk | 6,786 | **Unit is Litre, not KG.** Different unit requires separate handling. Also cooperative-controlled (Amul). |

**Commodities with < 100 rows (completely unusable):**
Millet, Eggplants, Cumin, Sorghum, Ginger, Chili, Chickpea flour, Ghee, Butter, Turmeric, Peppers, Bananas, Garlic, Coriander, Semolina, Eggs — these have 1-50 rows each, appearing only in 2025-2026. Impossible to train any model on.

### Why this specific combination of 3 works:

```
Onions:   VOLATILE (CV=0.568, prices can triple in 1 month)
          → Tests: Can TFT's uncertainty bands capture extreme spikes?
          → Answer: Yes — q90 widens during pre-crisis months.

Tomatoes: VOLATILE + RECENT (2023 spike at Rs 200+/kg)
          → Tests: Does the test set contain a real-world shock?
          → Answer: Yes — every professor and committee member knows this event.

Rice:     STABLE (CV=0.456, max shock only 35%)
          → Tests: Does TFT produce NARROW bands for stable crops?
          → Answer: If yes, it proves the model learned different
                    volatility regimes per crop from the SAME training.
                    This is genuine machine learning, not overfitting.
```

**The key academic argument:** If TFT produces WIDE bands for onions and NARROW bands for rice, from the same model trained on the same data — that proves the model learned which crops are volatile without being told. That's the contribution.

---

## 4. Why We Dropped Markets (60-Month Filter)

### The problem with short series

TFT needs historical context to learn. With `max_encoder_length=24` (2 years lookback), the model needs at least 24 consecutive months of past data to make one prediction. Series shorter than 60 months provide:

```
60 months total:
  - 24 months used for encoder (lookback window)
  - 6 months used for prediction target
  - Remaining: 30 months of sliding windows for training
  = ~30 training examples per series (barely enough)

40 months total:
  - 24 months encoder
  - 6 months target
  - Remaining: 10 sliding windows
  = 10 training examples (TOO FEW — model memorizes, doesn't learn)

20 months total:
  - Can't even fill one encoder window of 24 months
  = IMPOSSIBLE to use
```

### What was dropped

```
Total (market, commodity) pairs in raw data: 507
Pairs with >= 60 months: 142 (kept)
Pairs with < 60 months:  365 (dropped)

That's 72% of pairs DROPPED. This sounds extreme, but:
- 200+ pairs had < 20 months (completely unusable)
- 100+ pairs had 20-40 months (insufficient for encoder)
- 65 pairs had 40-59 months (borderline — dropped for consistency)
```

### Examples of what was dropped and why:

```
Dropped: Onions_Adilabad (27 months)
  → Only 3 possible training windows. Model would memorize.

Dropped: Tomatoes_Aizawl (55 months)
  → Close to 60 but only 25 usable windows. Borderline.

Dropped: Rice_Ajmer (14 months)
  → Can't even fill one encoder window. Impossible.

Dropped: All "Zone" markets (East Zone, North Zone, etc.)
  → These are post-2023 aggregated entries with < 30 months.
  → Not comparable to city-level pre-2023 data.
```

### Why 60 months specifically?

```
60 months = 5 years minimum history
This ensures:
1. At least 30 training windows per series (enough to learn, not memorize)
2. Covers at least 4-5 annual cycles (learns seasonality)
3. Likely includes at least 1 price shock event (learns volatility)
4. Standard in time series literature (5-year minimum for monthly data)
```

---

## 5. Why We Show Cities, Not Individual Mandis

### Short answer: Because the dataset IS city-level, not mandi-level.

The WFP dataset has **one price point per city per month**. "Chennai" in this dataset means one specific retail market location in Chennai (lat=13.08, lon=80.28). It is NOT an average of Koyambedu mandi, T. Nagar market, and Mylapore shops.

```
WFP data structure:
  Chennai = 1 retail location → 1 price per month
  Delhi   = 1 retail location → 1 price per month
  Mumbai  = 1 retail location → 1 price per month
```

**We are NOT aggregating anything.** Each series_id like "Onions_Chennai" is one price at one location.

### Why not show individual mandis?

**Because the data doesn't have individual mandi data.** The WFP collects from one representative retail point per city. To get mandi-level data, you would need:

| Data Source | Level | Frequency | Access |
|---|---|---|---|
| **WFP (what we use)** | City retail | Monthly | Free, clean CSV |
| **Agmarknet** | Mandi wholesale | Daily | Free but messy, scraping needed |
| **NCDEX** | Exchange wholesale | Daily | Paid API |
| **Dept of Consumer Affairs** | City retail | Daily | Government portal, inconsistent |

### If we wanted mandi-level data (future work):

```
Agmarknet has daily wholesale prices for:
  - Azadpur Mandi, Delhi (largest onion mandi in Asia)
  - Lasalgaon Mandi, Maharashtra (onion price setter for India)
  - Koyambedu Market, Chennai
  - Vashi Market, Mumbai
  - ...hundreds more

But:
  - Data is messy (missing days, inconsistent formats)
  - No standard CSV download (requires scraping)
  - Different units (quintal, not KG)
  - Wholesale, not retail (different price levels)
  - Would require separate data pipeline
```

### What to tell the professor:

> "The WFP dataset provides one representative retail price point per city per month. Each 'market' in our dashboard — Chennai, Delhi, Mumbai — corresponds to one specific GPS-located retail collection point, not an aggregation of multiple locations. This is a limitation of the WFP data collection methodology, not our processing. For mandi-level granularity, integration with the Agmarknet database would be required as future work, which would provide daily wholesale prices from hundreds of individual mandis but would require significant data cleaning and a separate ingestion pipeline."

---

## 6. Summary Table

| Question | Answer |
|---|---|
| How many markets in raw data? | 170 (164 cities + 1 National Average + 5 Zones) |
| How many markets after filtering? | **53 cities** |
| How many series (market x crop)? | **123** (38 Onions + 45 Tomatoes + 40 Rice) |
| Why 53 not 170? | Most markets have < 60 months of data for our 3 crops |
| Why 3 crops not 41? | Onions (volatile), Tomatoes (recent spike), Rice (stable baseline). Others are price-controlled, not agricultural, or have sparse data. |
| Is each "market" one location? | **Yes.** One GPS point per city. No aggregation. |
| Why not mandi-level? | WFP data is city-level retail. Mandi data requires Agmarknet (separate source, future work). |
| Why 60-month minimum? | TFT needs 24-month encoder + 6-month target + enough windows to learn. < 60 months = memorization, not learning. |
| What was dropped by gap filter? | 19 series with gaps > 6 consecutive months (unreliable reporting). |
