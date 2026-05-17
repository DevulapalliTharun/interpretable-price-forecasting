# Presentation Speech Guide

## Interpretable and Uncertainty-Aware Food Price Forecasting

**How to use this guide:**
Read each slide section before your evaluation. The "Say this" block is your actual speech — you can read it naturally or use it as a base. "What it means" explains the concept simply if you get a question. "Point to" tells you which part of the slide to gesture toward while speaking.

---

---

## SLIDE 1 — Cover

**What is on the slide:**
Your name, roll number, programme, institute, guide name, mentor name, and the full project title.

**Say this:**

> "Good morning. My name is Devulapalli Tharun, Roll Number 252CS009, M.Tech CSE at NIT Karnataka, Surathkal. My project is guided by Dr. Shashidhar G Koolagudi.
>
> The project is titled: *InterpreOnly what u have done and what u achieved u should demonstratetable and Uncertainty-Aware Food Price Forecasting for Indian Markets using Temporal Fusion Transformers.*
>
> In simple words — I built a system that not only predicts food prices, but also tells you how confident it is and why it made that prediction."

**What it means (simple):**

- "Interpretable" means the model explains why it gave that prediction — not a black box.
- "Uncertainty-aware" means it gives a price *range*, not just one number. So you know best case and worst case.
- "Temporal Fusion Transformers" is the deep learning model at the core of the system.

---

---

## SLIDE 2 — Motivation

**What is on the slide:**
Title: *The Problem Worth Solving*. Four bullet points about the problem. Four stat boxes: ₹130 peak onion price, 1.4 billion people food-dependent, 10+ years of data, 3 staple commodities. A summary box at the bottom.

**Say this:**

> "Let me start with why this problem matters.
>
> Indian food prices — especially onions, tomatoes, and rice — are extremely volatile. In 2019, onion prices jumped from around ₹20 to ₹130 per kilogram within a few months, just because of a monsoon failure and an export ban. That kind of sudden jump affects hundreds of millions of people — farmers lose income, consumers face hardship, and procurement agencies have no warning.
>
> The bigger issue is this: most forecasting tools give you only *one number* — a single predicted price. But one number alone is risky. If you are a government buyer or a retailer, you need to know: what is the *range* of expected prices? And *why* is the model predicting that?
>
> Existing tools lack two things: first, a reliable price band — not just a point. Second, an explanation of which factors are actually driving the forecast.
>
> That is what this project solves."

**Point to:**

- ₹130 box → "This spike happened in 2019 — real data."
- 1.4B box → "The scale of the problem."
- Summary box at bottom → "These are the three things that were missing, and this project delivers all three."

---

---

## SLIDE 3 — Problem Statement

**What is on the slide:**
Title: *Problem Statement*. A large box with Input / Task / Constraints / Evaluation sections. A table showing scope (commodities, markets, training period, forecast outputs). A flow showing: Price history + weather → Model → Forecast bands → Calibrated output.

**Say this:**

> "Let me now precisely define what I built.
>
> The **input** is two things: monthly food price data from the World Food Programme for 3 commodities — onions, tomatoes, rice — across 53 markets in India, from 2010 to 2022. And alongside that, matched weather data from NASA — rainfall, temperature, and humidity for the same locations and months.
>
> The **task** is to forecast not just one price, but three — the 10th percentile, the 50th percentile, and the 90th percentile — for 1 to 6 months into the future. So you get a low estimate, a middle estimate, and a high estimate for each market and commodity.
>
> The **constraints** are important. The uncertainty bands must be *empirically calibrated* — meaning they should actually contain the real price the right percentage of the time, not just look good on paper. And the feature attribution must be statistically stable, not just visually plausible.
>
> **Evaluation** is done on a completely held-out test set — all data from 2023 onwards, 792 rows total. The model never sees this during training."

**What it means (simple):**

- q10, q50, q90 → Think of it as: q10 is the optimistic low price, q50 is the most likely price, q90 is the high-risk price. Together they form a band.
- "No look-ahead leakage" → The model is not allowed to use future data to predict the past. It is evaluated fairly, like in the real world.
- "Held-out test set" → Data locked away and never shown to the model during training. Only used for final evaluation.

**Point to:**

- The formulation box → walk through Input → Task → Constraints → Evaluation one by one.
- The flow at the bottom → "This is the simple version of the whole pipeline."

---

---

## SLIDE 4 — Overview

**What is on the slide:**
Title: *Project Question and Design Answer*. The central question: "Can we forecast Indian food prices with accuracy, uncertainty, and explanation?" Four stat boxes: 11.4% MAPE, 84.0% coverage, 792 test rows, 123 series. The system answer pipeline box.

**Say this:**

> "The central question of this project is: can we forecast Indian food prices with accuracy, uncertainty *and* explanation — all three together?
>
> The answer is yes, and the system achieves: 11.4% MAPE on the test set — that means on average the predicted price is within about 11% of the real price. The uncertainty band covers 84% of real prices — meaning 84 out of every 100 actual prices fall inside the predicted range. And this is evaluated on 792 real test rows.
>
> The design answer is a pipeline: we take WFP price data plus NASA weather, engineer 26 features, train an XGBoost point model, feed its output into a Temporal Fusion Transformer that produces quantile forecasts, calibrate those bands per commodity using conformal calibration, and expose everything through a Streamlit dashboard."

**What it means (simple):**

- MAPE = Mean Absolute Percentage Error. 11.4% means if the real price is ₹100, the model is usually within ₹11.4.
- 84% coverage means the band we predict is reliable — 84 out of 100 actual prices land inside it.
- 123 series means 123 unique market-commodity combinations.

**Point to:**

- Stat boxes → mention each one.
- System answer pipeline → "This is the full pipeline at a glance."

---

---

## SLIDE 5 — Data

**What is on the slide:**
Title: *Dataset and Modeling Scope*. Five stat boxes: 145,124 raw rows, 16,939 final rows, 53 markets, 3 commodities, 2023+ test period. Two tables — one showing data sources, one showing commodity breakdown. Text about train/validation/test split.

**Say this:**

> "The data comes from two main sources.
>
> First, the World Food Programme's open food price dataset for India — this had 145,124 raw rows. After cleaning and filtering for the three commodities and matching weather data, we get 16,939 rows forming a monthly panel.
>
> Second, NASA's POWER API, which gives monthly weather data — rainfall, temperature, and humidity — matched to each of the 53 market locations.
>
> For the three commodities: Onions have 5,003 rows across 38 markets and are highly seasonal and crisis-prone. Tomatoes have 5,245 rows across 45 markets and show sudden spikes — the hardest to forecast. Rice has 6,691 rows across 40 markets and is the most stable baseline.
>
> The time split is: training data goes up to December 2021. The validation set is all of 2022 — this is used for early stopping and for calibrating the uncertainty bands. The test set is 2023 onwards — held out completely and never touched during training."

**What it means (simple):**

- Why log-price? Price data has big spikes. Log transformation compresses those spikes so the model is not dominated by extreme values.
- Why 24 months encoder window? The model looks at the last 24 months to predict the next 6 months.
- Train / validation / test split → Think of it like studying for an exam: training is study time, validation is mock tests, test is the actual exam.

**Point to:**

- Raw rows vs final rows → "This shows how much cleaning happened."
- Commodity table → "Three very different behaviors."
- Train/validation/test bullet points → "Strict discipline — the model never saw 2023 data."

---

---

## SLIDE 6 — Pipeline

**What is on the slide:**
Title: *End-to-End System Flow*. A horizontal flow diagram: Raw data → Cleaning → Feature panel (26 columns) → XGBoost and TFT in parallel → CQR calibration → Dashboard. An artifact list. A final family note box.

**Say this:**

> "This slide shows the complete pipeline from raw data to the final dashboard.
>
> We start with raw data from WFP and NASA. After cleaning and filtering, we build a feature panel with 26 columns. This panel goes to two models in parallel: XGBoost, which gives us a strong point prediction, and the Temporal Fusion Transformer, which gives us quantile predictions — the low, middle, and high estimates.
>
> The key innovation here — the XGBoost prediction is *fed into* TFT as an additional input feature, not just compared against it separately. This is what makes the fusion work.
>
> After TFT produces its quantile predictions, we apply per-commodity conformal calibration, which adjusts the bands so they reliably cover real prices. The final calibrated predictions go into the Streamlit dashboard.
>
> The final model family used is called TFT-XGBFusion-CQR — step 5 of the ablation."

**What it means (simple):**

- XGBoost = a powerful tree-based machine learning model. Very good at tabular data. Gives one number.
- TFT = a deep learning model designed for time series. Gives three numbers (low, middle, high).
- CQR = Conformal Quantile Regression. A technique to make the bands statistically reliable. Like adding safety margins that are calibrated from real errors.
- Artifacts = the saved model files. The dashboard loads these to work.

**Point to:**

- The flow arrows → walk left to right.
- "xgb_log_pred" arrow → "This is the fusion — XGBoost output going into TFT as an input."
- Final family box → "This is what runs in the dashboard."

---

---

## SLIDE 7 — Preprocessing / Feature Engineering

**What is on the slide:**
Title: *Feature Engineering*. A table with 6 feature groups: Static, Calendar, Price memory, Weather, Shock flags, Fusion. A formula box showing key transformations. Four bullet points about design choices.

**Say this:**

> "Before training any model, we engineer 26 features from the raw data. These fall into six groups.
>
> Static features: the commodity type, market name, and state — these tell the model which series it is looking at.
>
> Calendar features: month, season, and sine/cosine encodings of month. Why sine and cosine? Because month 12 and month 1 are adjacent in time, but if you just use numbers, the model thinks 12 and 1 are far apart. Sine/cosine encoding wraps the calendar correctly.
>
> Price memory features: price from 1 month ago, price from 12 months ago, rolling 3-month and 6-month averages. These give the model memory of recent trend and annual seasonality.
>
> Weather features: rainfall, temperature, humidity. These are supply-side signals — a drought directly affects onion supply.
>
> Shock flags: binary indicators — rain deficit means rainfall was below the 25th percentile, heat stress means temperature was above the 75th percentile. These help the model detect crisis regimes.
>
> And finally, the fusion feature — XGBoost's log-space prediction. This is a known future covariate in TFT.
>
> All 26 features are shared between XGBoost and TFT so the comparison is fair."

**What it means (simple):**

- "Log-price" → We use log(1 + price) instead of raw price. This compresses big spikes. Like how doubling a price is the same percentage change whether the base is ₹10 or ₹100.
- "Known future covariate" → TFT distinguishes between things it can know about the future (like month, season) and things it can only know from the past (like actual price). XGBoost prediction is treated as known future — because once XGBoost is trained, it can predict any future date.

**Point to:**

- Feature table → go through each group briefly.
- Formula box → "These are some of the key transformations used."
- Bottom bullet: "The fused XGBoost feature is created using a leakage-controlled pre-2020 model" → this is critical to explain.

---

---

## SLIDE 8 — Method Stack

**What is on the slide:**
Title: *Model Stack: Point + Distribution + Calibration*. Four boxes in a chain: XGBoost → TFT → CQR → Dashboard. A table showing what each layer solves. A box explaining the prediction object. A key design choice note.

**Say this:**

> "The system has four layers, each solving a specific problem.
>
> Layer one: XGBoost. It is a gradient boosting model with 500 trees. It gives a single predicted price — very accurate for tabular data. But it gives no uncertainty and no explanation.
>
> Layer two: TFT — Temporal Fusion Transformer. It takes the same features plus the XGBoost prediction as input, and outputs three quantiles — q10, q50, q90 — for each of the 6 future months. It also gives interpretable attention weights and variable importance scores.
>
> Layer three: CQR — Conformal Quantile Regression. This takes the TFT bands and calibrates them using 2022 validation errors. It finds the right amount to widen the bands so they reliably cover real prices.
>
> Layer four: the Streamlit dashboard. It presents all of this — the forecast band, the feature importance, and the test validation — in a clean, usable interface.
>
> The key design choice on this slide: XGBoost is not just averaged with TFT. Its prediction becomes a *known future input* to TFT. So TFT learns when to trust XGBoost and when to override it."

**What it means (simple):**

- Why not just use XGBoost? XGBoost is a point model — one number. No uncertainty, no time-aware attention, no interpretable feature gates.
- Why not just use TFT alone? TFT is weaker at raw point accuracy than XGBoost. By fusing XGBoost's signal into TFT, we get the best of both.
- Each forecast month gives you: a lower bound (q10), most likely (q50), upper bound (q90), and a risk label based on how wide the band is.

**Point to:**

- Four boxes in chain → walk through each.
- Table → "Each layer solves one limitation."
- Key design choice at bottom → "This is what makes our approach different from simple ensembling."

---

---

## SLIDE 9 — TFT Architecture

**What is on the slide:**
Title: *TFT Architecture in This Project*. A flow diagram: three input types (Static IDs, Known future, Observed past) → VSN gates → LSTM encoder (24 months) → LSTM decoder (6 months) → Static enrichment → Interpretable attention → Quantile heads (q10, q50, q90). Configuration box. Three key questions answered.

**Say this:**

> "This slide shows how the Temporal Fusion Transformer is structured internally — specifically how it is configured for this project.
>
> There are three types of inputs. Static inputs are things that don't change over time — like which commodity and which market. Known future inputs are things we can know about the future — like month, season, and the XGBoost prediction. Observed past inputs are things we only know from history — like actual past prices, weather, and shock flags.
>
> All three types go through a Variable Selection Network — VSN. This is a trainable gating mechanism. It assigns a weight to each variable for each time step. So the model itself learns which features matter most at any given time.
>
> The past 24 months go through an LSTM encoder. The next 6 months go through an LSTM decoder. The encoder's output is enriched with static context — so the model knows it's dealing with onions in Delhi versus rice in Chennai.
>
> Then there is interpretable multi-head attention — it looks at which past months are most relevant for each future prediction step.
>
> Finally, three separate output heads produce q10, q50, and q90.
>
> The configuration: hidden size 32, 2 attention heads, 1 LSTM layer — kept small intentionally to avoid overfitting on our data size."

**What it means (simple):**

- VSN (Variable Selection Network) → Think of it as the model's automatic feature selector. Instead of you telling it which features matter, it learns that on its own. And it can change the selection per time step and per series.
- LSTM → Long Short-Term Memory. A type of neural network layer designed for sequences. It remembers relevant information over time and forgets irrelevant things.
- Attention → The model learns to look at specific past months when making each future prediction. Like how when predicting next January's price, it automatically looks at last January.
- Three quantile heads → Three independent prediction heads at the end of the network, each trained to estimate a different percentile.

**Point to:**

- Three input boxes → "Static, known future, observed past — TFT is specifically designed for this separation."
- VSN → "This is what gives us the feature importance scores."
- Attention box → "This is what gives us the time-step attribution."
- Configuration box → "These are the actual settings used in training."

---

---

## SLIDE 10 — XGBoost Fusion (Novelty)

**What is on the slide:**
Title: *XGBoost Fusion Without Leakage*. Five numbered steps. An ablation table showing four model families. A flow: XGB prediction → TFT VSN → Quantile output → CQR band. An explanation of why pre-2020 XGB.

**Say this:**

> "This is the core novelty of the project — how we fuse XGBoost into TFT without data leakage.
>
> The process is five steps. Step one: train XGBoost only on data up to December 2019. This is important. Step two: use this XGBoost to generate predictions for all months after 2019 — including the TFT training period 2020-2021. Step three: add those XGBoost predictions as a known covariate in the TFT dataset. Step four: train TFT up to December 2021, validate on 2022. Step five: test on 2023 and beyond with calibrated bands.
>
> Why the strict 2019 cutoff for XGBoost? If we trained XGBoost on data that includes 2020-2021, and then used its predictions inside TFT — those predictions would contain information from the TFT validation and test period. That is leakage. The model would effectively be cheating.
>
> By training XGBoost only up to 2019, its predictions for 2020 onwards are generated without ever seeing 2022-2023 data. So when TFT uses those predictions, there is no leakage.
>
> The ablation table shows the four model families we tested. Each step adds one component. TFT-XGBFusion-CQR is the final best model."

**What it means (simple):**

---

### FULL EXPLANATION — Why XGBoost is trained only until 2019, and TFT until 2021

Read this carefully before your presentation. This is the part that confuses most people, including evaluators.

---

#### The data timeline — who uses what data

Think of the entire dataset as a single long timeline from 2010 to 2023. Different models are allowed to see different portions of it, for a specific reason. Here is the exact split:

| Period | What it is called | Who uses it | Purpose |
|---|---|---|---|
| 2010 – 2019 | XGBoost training window | XGBoost only | XGBoost trains here |
| 2020 – 2021 | TFT training window | TFT only | TFT trains here, using XGB's predictions as input |
| 2022 | Validation + CQR calibration | Neither model trains here | Used to calibrate the confidence bands |
| 2023+ | Test set | Nobody trains here | Final honest evaluation |

The key rule: **No model is ever allowed to train on data it will later be asked to predict.** The 2022 and 2023+ data are kept completely separate from training.

---

#### Why does XGBoost stop at December 2019? — The leakage firewall

The whole point of XGBoost in this project is to give TFT a helper feature — the XGBoost price prediction for each month. But here is the problem: **TFT trains on 2020-2021 data.** So if XGBoost had also trained on 2020-2021 data, its predictions for those months would be "in-sample" — meaning it already saw the answers before making the prediction.

Simple analogy: Imagine you have a tutor who is also going to be your examiner. If the tutor saw all the exam questions while preparing them for you, and then "predicted" what answers they would ask — their predictions would be perfect. But those perfect predictions would be useless to you as a student because the tutor cheated. The moment the real unseen exam comes (2023), the tutor fails because they only memorized, not learned.

In this project:
- **XGBoost is the tutor / helper model.**
- **TFT is the main student / forecasting model.**
- **2020-2021 is TFT's study material.**
- **2022-2023 is the real exam.**

If XGBoost sees 2020-2021 data while training, its predictions for those months are "cheating." TFT would learn from cheated predictions and appear to do well — but fail on 2022-2023 because it learned from contaminated data.

By stopping XGBoost at December 2019:
- XGBoost **never sees 2020-2021 data during training.**
- Its predictions for 2020-2021 are made **blindly** — it has not seen those months yet.
- These blind, honest predictions are what TFT uses as a feature.
- This is called an **out-of-sample prediction** — predicted without seeing the actual value.

---

#### What happens between 2019 and 2021? — Step by step

This is the exact sequence of events:

**Step 1:** Train XGBoost on 2010–2019 data. XGBoost learns price patterns from 10 years of data. At this point, 2020-2021 does not exist for XGBoost.

**Step 2:** Run XGBoost forward in time over 2020-2023. For each month — January 2020, February 2020, ... December 2021, 2022, 2023 — XGBoost produces a predicted price. It does this by using known inputs like rainfall, temperature, and lagged prices. It never uses the actual price of that month. These predictions are called **`xgb_log_pred`** in the code — one number per month, per commodity.

**Step 3:** Now TFT starts its work. TFT reads the 2020-2021 data — real weather, real lagged prices — but also one extra column: `xgb_log_pred`. This column is XGBoost's honest, never-cheated prediction for each month. TFT uses this as one of its input features.

**Step 4:** TFT trains on 2020-2021. It learns to combine all the features — including the XGBoost signal — to produce its own q10, q50, q90 quantile predictions.

**Step 5:** In 2022, TFT is never trained again. The 2022 data is used only to calibrate the CQR band offsets.

**Step 6:** 2023 onwards is the test. TFT predicts, and we measure MAE, MAPE, and coverage. Nobody touched this data during training.

---

#### Why does this design work so much better?

XGBoost alone: Very accurate at point prediction (MAE 1.58) but gives no price range and cannot explain why.

TFT alone: Can give a price range and explain which features matter — but it is less accurate at point prediction (MAE 10.53, MAPE 29.1%).

TFT + XGBoost fusion: TFT now has a strong numerical signal from XGBoost as one of its inputs. It combines XGBoost's precision with TFT's range-prediction and explainability. The result is MAE 5.27, MAPE 11.4%, coverage 84% — significantly better than TFT alone while keeping the range and explanation.

The fusion design is novel because it is done **without leakage.** Most papers that combine tree models with neural networks simply concatenate their outputs at test time — they do not use one as a real-time input feature to the other's training. This project integrates XGBoost predictions as a known covariate inside TFT's Variable Selection Network, so TFT learns to weight it alongside weather, season, and lag features.

---

#### Quick one-sentence answers if the evaluator asks

- *"Why did you stop XGBoost at 2019?"* → "Because TFT trains on 2020-2021. If XGBoost also trained on 2020-2021, its predictions there would be in-sample — that would leak information into TFT's training and make the final test evaluation dishonest."
- *"What is between 2019 and 2021?"* → "That is TFT's training window. XGBoost's predictions for 2020-2021 are out-of-sample predictions made honestly — XGBoost never saw that data. TFT then trains using those honest predictions as one of its input features."
- *"What is 2022 used for?"* → "Validation and CQR calibration. We measure how much the prediction bands miss on 2022 data and compute an offset to correct them. The model never trains on 2022."
- *"Why not retrain on 2022 data?"* → "We ran that experiment — TFT-Retrain21-CQR. It did not significantly improve over the fused approach. The fusion with XGBoost gave a bigger gain than retraining."

---

**Point to:**

- Five numbered steps → walk through them one by one.
- Why pre-2020 XGB note → read it out and explain.
- Table → "These are the four model families. The final row is our best model."

---

---

## SLIDE 11 — Math (Quantile Loss and CQR)

**What is on the slide:**
Title: *Quantile Loss and CQR Calibration*. Two formula boxes — pinball loss and CQR. A table with per-commodity CQR offsets and test coverage. A plain-meaning explanation at the bottom.

**Say this:**

> "This slide covers the two mathematical foundations of the system.
>
> First, quantile loss — also called pinball loss. This is the loss function TFT uses during training. For each quantile q, the loss penalizes over-prediction and under-prediction asymmetrically. At q=0.1 — the 10th percentile — the loss penalizes over-prediction more. At q=0.9 — the 90th percentile — it penalizes under-prediction more. This is what makes the model produce calibrated quantiles.
>
> This formula is from Lim et al. 2021, the original TFT paper.
>
> Second, conformal quantile regression — CQR. After TFT is trained, we take the 2022 validation set and compute a conformity score for each prediction: the score is how much the real price falls outside the predicted band. Specifically, it is the maximum of: how much q10 overshoots the real price on the low side, or how much q90 undershoots it on the high side.
>
> We then take the 90th percentile of all these scores across the validation set. That value is the offset c. We subtract c from q10 and add c to q90 — widening the bands just enough so they are reliable.
>
> This is done separately for onions, tomatoes, and rice. As you can see in the table, onions needed a small negative offset — meaning the bands were already slightly too wide for them. Tomatoes needed a larger positive offset. Overall test coverage is 84%.
>
> The plain meaning: CQR widens the band only as much as the 2022 validation errors say is needed."

**What it means (simple):**

---

### FULL EXPLANATION — Pinball Loss formula

```
QL(y, yhat, q) = q * max(y - yhat, 0)
               + (1-q) * max(yhat - y, 0)
```

- `y` = actual real price
- `yhat` = model's predicted price
- `q` = which quantile you want (0.1, 0.5, or 0.9)

**Two cases:**

**Case 1 — Model predicted too low** (actual price is higher than predicted, y > yhat):
- `max(y - yhat, 0)` is positive → penalty = `q × (y - yhat)`
- `max(yhat - y, 0)` is zero → second term drops out

**Case 2 — Model predicted too high** (actual price is lower than predicted, yhat > y):
- `max(y - yhat, 0)` is zero → first term drops out
- `max(yhat - y, 0)` is positive → penalty = `(1-q) × (yhat - y)`

**How this forces a specific percentile:**

| Quantile | Under-predict penalty | Over-predict penalty | What the model learns |
|---|---|---|---|
| q = 0.1 | 0.1 × error (small) | 0.9 × error (large) | Learns to predict LOW — 10th percentile |
| q = 0.5 | 0.5 × error | 0.5 × error | Learns median — 50th percentile |
| q = 0.9 | 0.9 × error (large) | 0.1 × error (small) | Learns to predict HIGH — 90th percentile |

At q=0.1: predicting too high is punished 9× more than predicting too low → model prefers to predict low → it naturally learns the 10th percentile.

At q=0.9: predicting too low is punished 9× more → model prefers to predict high → it naturally learns the 90th percentile.

**TFT trains all three quantiles at the same time** by summing the loss over q ∈ {0.1, 0.5, 0.9} and over all 6 horizon steps (H=6). This gives the model three separate output heads — one per quantile.

Simple analogy: Normal loss (MSE) is like a judge who punishes you equally for arriving too early or too late. Pinball loss is like a judge who says "for the early train, being late is 9× worse than being early" — so you always plan to arrive early. Different rules create different habits.

---

### FULL EXPLANATION — CQR formula

CQR runs **after** TFT training is complete. It uses the 2022 validation data to measure how often the predicted band misses, and then corrects the band width.

**Step 1 — Compute conformity score for each validation point:**

```
s_i = max(q10_i - y_i,   y_i - q90_i)
```

For each month `i` in the 2022 validation set:
- `q10_i - y_i` → how much the actual price fell **below** the lower bound. Positive = actual is below q10 (bad — lower bound is too high).
- `y_i - q90_i` → how much the actual price went **above** the upper bound. Positive = actual is above q90 (bad — upper bound is too low).
- Take the max of the two → this is the "miss distance" for that point.
- If the actual price is **inside** the band → both terms are negative → score is negative (the band covered it, no miss).

So `s_i` tells you: for this month, how far outside the band did the real price fall? Positive = missed. Negative = covered.

**Step 2 — Find the correction offset:**

```
c = quantile_{1-alpha}(s_i)
```

Collect all `s_i` scores from 2022. Take the 90th percentile (since alpha=0.10, 1-alpha=0.90). That value is `c`.

`c` represents: "the band needs to be this much wider to cover 90% of validation misses."

**Step 3 — Adjust both bounds:**

```
q10_cal = q10 - c    ← push lower bound DOWN by c
q90_cal = q90 + c    ← push upper bound UP by c
```

The band is now wider by `c` on both sides. Applied separately for onions, tomatoes, and rice — because each commodity has a different error pattern.

If onions already had wide-enough bands, `c` is negative → `q10_cal` actually goes up and `q90_cal` goes down → band shrinks slightly.

Simple analogy: CQR is like a tailor measuring how much a shirt rode up when you moved around. They add exactly that much fabric at the hem. Not more, not less — just enough based on what they measured.

---

- 84% coverage → Out of 792 test predictions, 665 real prices fell inside our calibrated band. This is an empirical measurement — we counted it. It is not a theoretical claim.

**Point to:**

- Pinball loss formula → "This is the training objective. Different q values force the model to learn different percentiles."
- CQR formula → "This is the post-training calibration. It measures misses on 2022 and widens the band by exactly that much."
- Table → "These are the actual per-commodity offsets. Negative for onions means their band was already slightly too wide — CQR trimmed it."

---

---

## SLIDE 12 — Evaluation Protocol

**What is on the slide:**
Title: *Evaluation Protocol*. A metrics table (MAE, MAPE, Coverage, Band width). A box with test set discipline details. Five stat boxes: 792 rows, 1000 bootstrap resamples, 80% coverage target, 1-6 month horizon, 24-month encoder. Three bullet points about honesty.

**Say this:**

> "Before showing results, let me explain exactly how we evaluated the system — because evaluation discipline matters.
>
> The metrics we use: MAE is the average absolute error in rupees per kilogram. MAPE is the percentage error — scale-free. Coverage is the fraction of real prices that fell inside the predicted band. Band width is the average width of the band — smaller is better if coverage is still maintained.
>
> The test discipline: TFT training goes up to December 2021. 2022 is used as the calibration window for CQR — meaning CQR offsets are derived from 2022 errors. The held-out test set is 2023 and beyond — 792 rows. This data was never shown to any model during training or calibration.
>
> On top of basic metrics, we run statistical significance tests: paired t-test, Wilcoxon signed-rank test, Cohen's d for effect size. For the uncertainty bands, we run a binomial coverage test. For the feature importance rankings, we run 1000 bootstrap resamples.
>
> We also report XGBoost separately and honestly — XGBoost has a lower point error than TFT alone. We don't hide that. The claim is that TFT-XGBFusion-CQR wins the *complete* objective: accuracy plus calibrated uncertainty plus interpretability."

**What it means (simple):**

- Why not just use accuracy? A model could cheat by giving very wide bands — then coverage is always 100% but the band is useless. Band width keeps that honest.

---

### FULL EXPLANATION — All Statistical Tests

If the evaluator asks "what are these tests?" — here is a clear, simple explanation of each one.

---

#### 1. Paired t-test

**What it is:**
A test that checks whether the difference in error between two models is real, or just random chance.

**Why "paired"?**
Both models are tested on the exact same 792 rows — the same months, the same commodities. For each row, you subtract: `error_of_model_A - error_of_model_B`. This gives 792 difference values. The paired t-test asks: is the average of these 792 differences significantly different from zero?

**Why paired instead of a regular t-test?**
Some months are genuinely harder to predict (e.g., October 2023 had a spike). A regular t-test would confuse "model B is better" with "model B happened to get easier months." Pairing removes that confusion because both models faced the same hard months.

**What the p-value means:**
p-value is the probability that the observed difference happened purely by random chance, assuming the two models are actually equal. Very small p-value (e.g., p = 4.6 × 10⁻⁸⁶) means: if the models were truly equal, the chance of seeing this result is essentially zero. So the difference is real.

**Our actual results:**
- TFT-XGBFusion-CQR (step5) vs plain TFT (original): mean error reduced by **5.577 Rs/kg**, t = -22.34, **p = 4.6 × 10⁻⁸⁶**
- That p-value is astronomically small — the improvement is definitively real, not random.

Simple analogy: You flip a coin 792 times. If you get heads 700 times, a t-test tells you — "that's not luck, the coin is biased." Here, the coin is the model comparison.

---

#### 2. Wilcoxon Signed-Rank Test

**What it is:**
A non-parametric alternative to the paired t-test. It checks the same question — is the difference real? — but without assuming the data is normally distributed.

**Why run this alongside the t-test?**
The paired t-test assumes the 792 differences follow a bell curve (normal distribution). Food prices have occasional sharp spikes — so the differences might not be perfectly normal. The Wilcoxon test makes no such assumption. It works by ranking the absolute differences and checking whether one model consistently wins or loses.

**Why does it matter?**
If both the t-test and Wilcoxon agree (both show tiny p-values), the result is doubly confirmed. It means the improvement is robust — it holds regardless of which test assumption you use.

**Our actual results:**
- step5 vs original: Wilcoxon p = 1.245 × 10⁻⁹⁵ (even more extreme than the t-test)
- step5 vs step1: Wilcoxon p = 2.303 × 10⁻⁸²
- Both tests agree. The improvement is not a statistical artifact.

Simple analogy: The t-test is like asking "on average, did model A score higher?" The Wilcoxon test is like asking "in how many individual rounds did model A beat model B?" Both questions lead to the same answer here.

---

#### 3. Cohen's d (Effect Size)

**What it is:**
A number that tells you how *large* the improvement is in practical terms — not just whether it is statistically significant.

**Why do we need this?**
A p-value only tells you "the difference is real." But with 792 data points, even a 0.001 Rs/kg improvement would have a tiny p-value — and that would be statistically significant but completely useless in practice. Cohen's d answers: "is the difference big enough to matter?"

**How it is calculated:**
```
Cohen's d = mean difference / standard deviation of differences
```

**Standard interpretation:**

| Cohen's d | Practical meaning |
|---|---|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| > 0.8 | Very large effect |

**Our actual results:**
- step5 vs original: Cohen's d = **-0.79** (close to large — nearly 0.8)
- step5 vs step1: Cohen's d = **-0.76** (medium-large)
- step1 vs original: Cohen's d = **-0.15** (small — XGB feature alone, without CQR, has a smaller practical gain)

The negative sign just means the error went down (improvement). The magnitude is what matters.

Simple analogy: p-value is like saying "the taller group is definitely taller (p < 0.001)." Cohen's d is like saying "they are taller by 2 standard deviations — that is a massive difference, not just 1 cm."

---

#### 4. Binomial Coverage Test

**What it is:**
A test that checks whether the 84% empirical coverage is meaningfully above the minimum acceptable threshold.

**How it works:**
Each of the 792 test predictions either hits (real price inside band) or misses (real price outside band). This is a binary outcome — hit or miss — for each row. A binomial test asks: given 792 trials and 665 hits (84% coverage), is this coverage rate statistically above the target threshold?

**Why run this?**
84% might sound good. But is it significantly above, say, 75%? Or could 84% happen by random chance even if the true coverage is only 75%? The binomial test gives a formal answer.

**Our actual results:**

| Commodity | n | Hits | Coverage |
|---|---|---|---|
| Onions | 242 | 220 | 90.9% |
| Tomatoes | 297 | 245 | 82.5% |
| Rice | 253 | 200 | 79.1% |
| Overall | 792 | 665 | 84.0% |

All commodities show coverage well above 75%, and the binomial test confirms this is statistically significant — not random.

Simple analogy: You claim your umbrella keeps you dry 84% of the time. The binomial test is like checking — if someone stood in the rain 792 times with your umbrella, and got wet only 127 times — is that genuinely better than a bad umbrella that covers only 75% of the time? Yes, significantly so.

---

#### 5. Bootstrap Resampling (for VSN stability)

**What it is:**
A method to measure how stable/reliable the Variable Selection Network (VSN) feature rankings are.

**How it works:**
1. Take the 792 test predictions.
2. Randomly sample 792 rows with replacement (some rows repeat, some are left out). This is called one bootstrap resample.
3. Compute VSN feature importance rankings on this resample.
4. Repeat 1000 times → get 1000 slightly different ranking lists.
5. Measure how similar all 1000 ranking lists are to each other using Kendall tau (a correlation score for rankings).

**What Kendall tau means:**
Kendall tau ranges from -1 (completely reversed ranking) to +1 (identical ranking). A value of 0.942 means the rankings are almost always in the same order across all 1000 resamples.

**Our actual results:**
- Mean pairwise Kendall tau = **0.942**
- 95% confidence interval = **[0.895, 0.990]**
- Interpretation: the feature importance ranking is highly stable — temperature, rainfall, and covid_lockdown are consistently the top features regardless of which subset of data you look at.

**Why does this matter?**
If VSN rankings were unstable — e.g., temperature was top in some resamples but bottom in others — it would mean the explainability is unreliable. The high Kendall tau proves the explanation is trustworthy, not random.

Simple analogy: You ask 1000 different groups of students to rank their top 3 subjects. If 950 groups say "Maths, Physics, Chemistry" in the same order, the ranking is stable. If every group gives a different answer, the ranking is meaningless.

---

#### Quick one-sentence answers if the evaluator asks

- *"Why paired t-test and not regular t-test?"* → "Because both models were evaluated on the exact same test rows. Pairing removes the effect of month difficulty — it isolates the model difference."
- *"Why also run Wilcoxon?"* → "As a robustness check. Food price data has spikes, so the differences may not be normally distributed. Wilcoxon makes no normality assumption. Both tests agree."
- *"What does Cohen's d = 0.79 mean?"* → "It means the improvement is close to 'large' by standard benchmarks — not just statistically significant, but practically meaningful."
- *"What is 84% coverage — is that good enough?"* → "The binomial test confirms it is significantly above 75%. And 84% with a narrow band width of 11.99 Rs/kg is better than 88.5% with a 32 Rs/kg band — wider is not better."
- *"Why 1000 bootstrap resamples?"* → "1000 is the standard for stable confidence interval estimation. Kendall tau of 0.942 with CI [0.895, 0.990] means the feature rankings are consistent across all resamples."

---

**Point to:**

- Metrics table → go through each one.
- Test set discipline box → "Three separate time windows — this is why the evaluation is trustworthy."
- Three bullet points at bottom → "We report XGBoost honestly. We don't claim TFT wins on everything."

---

---

## SLIDE 13 — Main Results

**What is on the slide:**
Title: *Main Results on 2023+ Test Set*. A results table with 5 models. Four stat boxes: 5.27 MAE, 11.4% MAPE, 84.0% Coverage, 11.99 Band width. Two figures (ablation bar chart, calibration chart). A result story summary.

**Say this:**

> "These are the main results on the 2023 test set.
>
> The table compares five model variants. XGBoost point baseline has the lowest MAE at 1.95 and only 3.5% MAPE — but it has no uncertainty band at all, so coverage and band width are not applicable.
>
> Plain TFT-Base has high error — 10.53 MAE and 29.1% MAPE — and only 64.9% coverage. This tells us that without calibration, the bands are too narrow.
>
> As we add components — ensemble CQR, retrained cutoff, and finally XGB fusion — the MAE drops dramatically. The final model, TFT-XGBFusion-CQR, achieves 5.27 MAE and 11.4% MAPE, which is a 50% drop in error compared to plain TFT. Coverage is 84% with a band width of only 11.99 — the narrowest among all probabilistic models.
>
> The story here: XGBoost wins raw point error. Our final TFT model wins the *complete* forecasting objective — accuracy, reliable uncertainty band, and interpretable explanation together.
>
> The figures on this slide: on the left is the ablation chart showing how MAE drops at each step. On the right is the calibration coverage chart showing that our bands actually cover the right fraction of real prices."

**What it means (simple):**

- Why is XGBoost MAE lower? XGBoost is a pure point model optimized only for accuracy. TFT is optimized for quantile loss across three percentiles simultaneously, which is a harder objective.
- But why use TFT then? Because XGBoost gives you *one number*. TFT gives you a range — and with the fusion, the range is tight (11.99 width) and reliable (84% coverage). For decision-making, a reliable range is more useful than a single possibly-wrong number.
- Band width 11.99 means: the gap between the low estimate and high estimate is on average about ₹12 per KG. For a commodity like onions that was trading around ₹50-₹130, this is informative.

**Point to:**

- Results table → walk row by row.
- Stat boxes (5.27, 11.4%, 84.0%, 11.99) → "These four numbers summarize our final model's performance."
- Figures → "The ablation chart shows the improvement step by step. The calibration chart confirms the bands are real."

---

---

## SLIDE 14 — Commodity Results

**What is on the slide:**
Title: *Commodity-Level Behavior*. A table with Onions, Tomatoes, Rice — MAE, MAPE, Coverage, and interpretation. Three figures (one per commodity showing forecast bands). Three bullet points explaining each commodity's behavior.

**Say this:**

> "This slide breaks down the results by commodity.
>
> Onions perform the best: 2.25 MAE, 9.1% MAPE, and 91% coverage. Onions have a strong annual pattern — the model picks this up well through the lag-12-month and rolling features.
>
> Tomatoes are the hardest: 8.13 MAE, 13.0% MAPE, and 82% coverage. Tomatoes can spike and crash in a single week due to a local crop failure. Monthly data cannot see those rapid changes. So the model's error is naturally higher.
>
> Rice is in the middle: 4.81 MAE, 11.7% MAPE, and 79% coverage. Rice is the most stable commodity, so the model produces narrower bands. But because the true variation is small, even small misses show up in coverage.
>
> The three figures show the actual forecast bands versus real prices for each commodity. You can see the band tracking the real price most of the time, with the shaded region showing the uncertainty."

**What it means (simple):**

- Why does coverage differ? Each commodity behaves differently. CQR is calibrated per commodity to account for this. But the calibration is done on 2022 data — if 2023 regime is different from 2022, there will be some mismatch.
- Onion 91% coverage > target 80% → The bands are slightly conservative for onions. This is safer than being too narrow.
- Rice 79% coverage < target 80% → Very close to target. Rice is stable so small prediction errors still cause occasional misses.

**Point to:**

- Table → go through each commodity row.
- Figures → "These are actual model outputs on real test data. The shaded area is the predicted band, the line is the real price."

---

---

## SLIDE 15 — Explainability: VSN Feature Rankings

**What is on the slide:**
Title: *Explainability: Stable Feature Rankings*. Three stat boxes: Kendall tau 0.942, 95% CI low 0.895, 95% CI high 0.990. Two figures (VSN weight bar chart, stability chart). A table with top 5 encoder features. A note at the bottom about bootstrap testing.

**Say this:**

> "This slide is about interpretability — specifically, whether the model's feature importance rankings are trustworthy.
>
> The Variable Selection Network inside TFT assigns a weight to each input feature at each time step. We can look at the average weights across the test set to see which features the model considers most important.
>
> The top features are: temperature mean, rain excess, and rain deficit — all weather-related — each with a weight around 0.14. Then humidity and year. Then shock events like covid lockdown. Then seasonal and price memory features.
>
> This is consistent with domain knowledge: weather directly affects agricultural supply, so it should be important.
>
> But the key question is: are these rankings *stable*? Maybe on a different subset of data, the ranking would flip completely. To test this, we run 1000 bootstrap resamples — randomly resample the test series 1000 times and compute rankings each time. Then we measure the consistency using Kendall's tau — a correlation measure for rankings.
>
> The mean Kendall tau is 0.942, with a 95% confidence interval from 0.895 to 0.990. A value of 1.0 means perfect stability. 0.942 is very high — the rankings are stable regardless of which subset we test on.
>
> This means the explanation is not just visual decoration. It is statistically validated."

**What it means (simple):**

- VSN weight → A number between 0 and 1 showing how much the model uses that feature at a given time. Weights across all features sum to 1.
- Kendall's tau → A statistic between -1 and 1 measuring how consistent two rankings are. 0.942 means the top features stay at the top almost regardless of which data subset you use.
- Bootstrap resamples → Repeat the analysis 1000 times on random subsets. If results are consistent, we trust them.
- This is NOT causal → The model learned these weights from data. Weather being important doesn't prove weather *causes* price changes — it means price and weather co-move, and the model picked that up.

**Point to:**

- Stat boxes → "0.942 Kendall tau, confidence interval 0.895 to 0.990."
- Table → "Top 5 features by average weight."
- Figures → "Left shows the average weights. Right shows the stability across bootstraps."
- Bottom note → "The explanation is statistically tested, not just visual."

---

---

## SLIDE 16 — Attention and Forecast Evidence

**What is on the slide:**
Title: *Attention and Forecast Evidence*. Two figures (attention heatmap, forecast with band). An interpretation rule box: high attention at t-1 = recent momentum, t-12 = annual seasonality, spread = uncertain regime. Three bullet points.

**Say this:**

> "This slide covers the second interpretability mechanism — attention weights.
>
> TFT uses interpretable multi-head attention. For each future prediction step, the attention mechanism assigns weights to each of the 24 past months in the encoder window. High attention at the most recent month means the model is responding to recent price momentum. High attention at 12 months ago means the model is picking up annual seasonality. Spread attention across many months means the model is uncertain and averaging over a long history.
>
> The left figure is an attention heatmap — rows are future prediction steps, columns are past months. The color shows where attention is concentrated.
>
> The right figure shows the actual forecast band on the test set — the predicted band versus the real price over time.
>
> Together, VSN and attention answer two different questions. VSN answers: which input variable matters most? Attention answers: which past time period matters most? Together, the dashboard can tell a user not just what the price will be, but *why* the model thinks that."

**What it means (simple):**

- Attention heatmap → Think of it as the model's "focus map." For each future month it predicts, you can see which past month it was looking at the most.
- t-1 = last month, t-12 = same month last year. High attention at t-12 means the model is using last year's seasonal pattern strongly.
- VSN vs attention → VSN is about which *variable* (feature column) matters. Attention is about which *time step* (past month) matters. They are complementary explanations.

**Point to:**

- Attention heatmap → "Each row is a future month, each column is a past month."
- Forecast figure → "The actual model output on test data."
- Interpretation rule box → read it out.

---

---

## SLIDE 17 — Dashboard

**What is on the slide:**
Title: *Dashboard: Four Operational Views*. Four screenshots of the Streamlit dashboard. A description at the bottom.

**Say this:**

> "This slide shows the actual Streamlit dashboard that was built as part of this project.
>
> The dashboard has four views.
>
> View one: Historical Forecast — shows how the model's predictions compare with actual recorded prices over the historical period. This lets the user see how well the model tracks real price movements.
>
> View two: Future Forecast — shows the 6-month ahead price band for any selected market and commodity. The user sees the median prediction plus the calibrated q10 and q90 band.
>
> View three: Model Explainability — shows the VSN feature importance weights for the selected series and the attention heatmap. The user can see which features and which past months the model is paying attention to.
>
> View four: Test Validation — shows how the model performed on the held-out 2023 test data for the selected commodity. The user can verify the model's real-world accuracy.
>
> The dashboard automatically selects the best available model family. If the full TFT-XGBFusion-CQR artifacts are present, it uses those. Otherwise it falls back gracefully."

**What it means (simple):**

- Why a dashboard? Models are not useful if only researchers can access them. The dashboard makes the forecast accessible to procurement officers, policy analysts, or any domain user.
- Streamlit → A Python library that creates web apps directly from Python code. No HTML or JavaScript needed.
- The four views cover the full lifecycle: what happened (historical), what will happen (future), why the model thinks that (explainability), and proof it works on new data (test validation).

**Point to:**

- Each screenshot → describe what it shows.
- "The dashboard is live and can be demonstrated."

---

---

## SLIDE 18 — Limitations and Future Work

**What is on the slide:**
Title: *Limitations and Future Work*. A three-column table: Current limitation | Effect | Future improvement. Five rows covering monthly data granularity, weather resolution, commodity scope, static calibration, and prototype dashboard.

**Say this:**

> "Every project has limitations, and it is important to state them honestly.
>
> First: monthly data granularity. We use monthly price data. Tomato prices can spike and recover within a single week. Monthly averages smooth those spikes out, so the model cannot capture them. The future improvement is to use daily or weekly data from Agmarknet — the Agriculture Marketing department's API which has more granular data.
>
> Second: monthly weather means. Similarly, a three-day heat event in the middle of a month is invisible in monthly average temperature. Higher frequency weather features would help.
>
> Third: three commodities. We tested on onions, tomatoes, and rice. The generality of the approach is not yet proven for pulses, edible oils, or vegetables. Expanding the commodity scope is straightforward given the pipeline.
>
> Fourth: static calibration. The CQR offsets are computed once from 2022 data. If market regimes shift — say, due to a new government policy or a structural change in supply chains — the calibration may drift. Rolling conformal recalibration would address this.
>
> Fifth: prototype dashboard. Currently the dashboard loads pre-computed artifacts. For production use, there should be a scheduled job that fetches new data, updates the model, and refreshes the predictions automatically."

**Point to:**

- Table → go through each row: limitation, effect, future improvement.

---

---

## SLIDE 19 — Conclusion

**What is on the slide:**
Title: *Conclusion*. A two-column contributions table with five rows. A key results box (5.27 MAE, 11.4% MAPE, 84.0% coverage). A Thank You with guide name.

**Say this:**

> "To conclude, this project makes five contributions.
>
> One: leakage-controlled XGBoost-to-TFT fusion. We trained XGBoost strictly on data up to 2019, and its prediction is injected as a known future covariate into TFT. This avoids data leakage while giving TFT a strong point signal to learn from.
>
> Two: quantile forecasting. TFT outputs q10, q50, and q90 for 6 months ahead across 53 markets and 3 commodities — giving users a low, middle, and high estimate.
>
> Three: per-commodity CQR calibration. The uncertainty bands are calibrated separately for each commodity using 2022 validation errors, achieving 84% empirical coverage on 2023 test data.
>
> Four: statistical explainability. Feature importance from the Variable Selection Network is validated through 1000 bootstrap resamples. The mean Kendall tau stability is 0.942, confirming the explanation is not just visual — it is statistically robust.
>
> Five: a working Streamlit decision dashboard with four views — historical, future forecast, explainability, and test validation.
>
> The key results: 5.27 Rs/KG MAE, 11.4% MAPE, 84.0% empirical coverage on the 2023 test set.
>
> The system turns volatile food-price prediction into a decision-support workflow: expected price, calibrated band, and model evidence — on one screen.
>
> Thank you. I am happy to take questions."

**Point to:**

- Contributions table → go through each of the five rows.
- Results box → "These are the final numbers on held-out 2023 test data."

---

---

## SLIDE 20 — References

**What is on the slide:**
Title: *Key References*. Five reference bullets. A core claim box.

**Say this (only if asked, or to show you know your sources):**

> "The two foundational papers for this project are:
>
> Lim et al. 2021 — this is the original Temporal Fusion Transformer paper, which introduced the VSN and interpretable attention architecture we build on.
>
> Romano et al. 2019 — this introduced Conformalized Quantile Regression, the post-hoc calibration technique we use for the uncertainty bands.
>
> The data sources are the World Food Programme's open India food price dataset from the Humanitarian Data Exchange, and the NASA POWER monthly point API for weather covariates."

**Note:** You do not need to dwell on this slide. Just point to it briefly and say the sources are cited. The slide is there to show you know where your methods come from.

---

---

## Common Questions and Answers

**Q: Why not just use XGBoost since it has lower error?**

> XGBoost gives one number. If you are a buyer or a policy official, one number is not enough. You need to know: what is the realistic range? And what is driving this forecast? XGBoost gives neither. Our final system gives both — at only 11.4% MAPE which is still practically accurate.

**Q: Is 84% coverage enough? You were targeting 80%.**

> Yes. We targeted an 80% coverage level. Achieving 84% means the bands are slightly conservative, which is safer for decision-making than undershooting. We report the empirical number honestly — we do not claim 90% just because we used q10 and q90 labels.

**Q: Does weather cause price changes?**

> We cannot claim that from this model. VSN weights show that weather features are highly associated with price movements — but association is not causation. The model learned from historical co-movement of weather and prices. Domain knowledge supports that monsoon affects supply, which affects price — but our model does not prove causality.

**Q: What happens if the model is wrong?**

> That is what the uncertainty band is for. Even when the median prediction is off, 84% of the time the real price falls inside the predicted range. The dashboard also shows test validation so users can see the model's historical accuracy before trusting a future prediction.

**Q: Can this be used in real time?**

> Currently it is a prototype with pre-computed artifacts. The limitation is the offline retraining requirement. Future work includes a scheduled pipeline for automatic data refresh and model update.

**Q: Why 53 markets? Can it work for all markets in India?**

> We are limited by the overlap between WFP price data and NASA weather coverage. The 53 markets are those with consistent monthly price records. Expanding to AGMARKNET data would cover more markets but requires data integration work.

---

---

## DEEP EXPLANATION — How TFT Works Internally (After XGBoost Signal is Added)

Read this if the evaluator asks "explain how TFT processes the data" or "how does the XGBoost feature actually flow through TFT?"

---

### The dataset at this point

Every row now has columns like:

```
month | commodity | temperature | rainfall | price_lag_1m | ... | xgb_log_pred | actual_price
```

`xgb_log_pred` is just one more number in each row. TFT does not know it came from XGBoost — it treats it like any other feature.

---

### Step 1 — Input Embedding

TFT takes every feature and converts it into a vector (a list of numbers). This is called embedding.

- `temperature_mean` (a number) → 32-dimensional vector
- `xgb_log_pred` (a number) → 32-dimensional vector
- `commodity` (a category) → 32-dimensional vector
- ... and so on for all features

All features are now in the same "language" — all 32-dimensional vectors. TFT can now compare and combine them mathematically.

---

### Step 2 — Variable Selection Network (VSN)

VSN looks at all the embedded feature vectors and asks: **"which features matter right now?"**

It produces a weight (0 to 1) for each feature. For example:
- `xgb_log_pred` → weight ~0.14 (high — very useful)
- `temperature_mean` → weight ~0.14 (high)
- `month_sin` → weight ~0.02 (low — less useful)

Each feature vector is then **multiplied by its weight**. Low-weight features are almost zeroed out. High-weight features pass through strongly.

The output of VSN is a single combined vector — a weighted blend of all features — for each timestep.

This is where `xgb_log_pred` either gets listened to or ignored. VSN decides how much TFT should trust the XGBoost signal at each step.

---

### Step 3 — LSTM Encoder (past 24 months)

TFT now feeds the VSN output vectors into an LSTM, one month at a time, going through the past 24 months.

```
Month t-24 → Month t-23 → ... → Month t-1 → Month t
```

The LSTM has three gates:
- **Forget gate** → decides what to erase from memory
- **Input gate** → decides what new information to write into memory
- **Output gate** → decides what to read out at each step

After processing all 24 months, the LSTM produces:
- A **hidden state** for each of the 24 months (one vector per month)
- A **final cell state** — the compressed memory of the entire history

This captures patterns like: "prices were rising for 6 months, then a drought hit, then they spiked."

---

### Step 4 — LSTM Decoder (future 6 months)

Now TFT looks at the next 6 months it needs to predict.

For these future months, TFT has **no actual price** — that is what it is trying to predict. But it does have:
- Known future weather forecasts
- Known calendar features (month, season)
- `xgb_log_pred` for those future months — because XGBoost can generate predictions for any future month

These future known features go through VSN again → then into the LSTM decoder, initialised with the encoder's final hidden state. The decoder produces 6 hidden state vectors — one per future month.

---

### Step 5 — Multi-Head Self-Attention

Now TFT has 30 vectors total: 24 from the encoder + 6 from the decoder.

Self-attention lets every position look at every other position and ask: **"which past months are most relevant to predicting this future month?"**

For example, when predicting June 2024:
- Attention may focus heavily on June 2022 and June 2023 (same season, same patterns)
- Attention may also focus on months with extreme weather events
- `xgb_log_pred` from those high-attention months gets indirectly amplified because VSN gave it high weight

The output is again 30 vectors — but now each one contains information from across the entire sequence, not just its own position.

Note: This is **self-attention**, not cross-attention. All 30 positions attend to each other within the same sequence — there is no separate query sequence.

---

### Step 6 — Gated Residual Network + Output

The attention output passes through Gated Residual Networks (GRN). These apply another round of selective filtering — keeping what matters, suppressing noise — with a skip connection so important signals are not lost.

Finally, three separate linear layers produce:
- **q10** — the lower bound (10th percentile price)
- **q50** — the median forecast
- **q90** — the upper bound (90th percentile price)

For each of the 6 future months, for each commodity.

---

### Full picture in one diagram

```
All features (including xgb_log_pred)
        ↓
   [Embedding] — convert every feature to a 32-dim vector
        ↓
   [VSN] — weight each feature (xgb_log_pred gets weight ~0.14)
        ↓
   [LSTM Encoder] — process past 24 months, build memory
        ↓
   [LSTM Decoder] — process future 6 months using known covariates
        ↓
   [Self-Attention] — every month attends to every other month
        ↓
   [GRN] — final gated filtering with residual connections
        ↓
   [3 output heads] → q10, q50, q90 for each of 6 future months
```

---

### Where exactly does xgb_log_pred help?

It enters at **VSN** and stays active through the whole sequence. Its main contribution: it gives TFT a strong numerical anchor — XGBoost's point estimate — which TFT then wraps with its own uncertainty band and seasonal/weather reasoning.

- Without `xgb_log_pred`: TFT figures out price level from raw weather and lags alone → MAE 10.53, MAPE 29.1%
- With `xgb_log_pred`: TFT starts from XGBoost's already-good estimate and refines it → MAE 5.27, MAPE 11.4%

---

### Quick one-sentence answers if the evaluator asks

- *"How does TFT use the XGBoost feature?"* → "It enters as a known covariate at the embedding step. VSN assigns it a weight of about 0.14, and it flows through the LSTM encoder and decoder alongside weather and lag features. The self-attention then decides which timesteps' XGBoost signals are most relevant for each future month."
- *"Why is xgb_log_pred called a known covariate?"* → "Because XGBoost can generate a prediction for any future month without needing the actual future price. So at prediction time, TFT already knows the XGBoost estimate for the next 6 months — it is a known input, not something TFT has to guess."
- *"Does TFT just copy the XGBoost prediction?"* → "No. TFT uses it as one of many inputs. VSN also weighs temperature, rainfall, and lagged prices. The self-attention further modulates which signals dominate based on the current temporal context. The final output is TFT's own quantile prediction, informed by but not equal to XGBoost."

---

*End of Speech Guide — 20 slides covered.*
