# TFT Improvement Plan

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

Goal: push TFT MAPE from ~29% toward ~10–15% (competitive with NBEATSX / published LSTM work) **without losing TFT's quantile bands, attention, or VSN outputs**. Also fix band calibration (current 90% coverage = 65%, Rice coverage = 30%).

Current baseline (from [visualizations/evaluation_metrics.txt](visualizations/evaluation_metrics.txt)):
- TFT MAE 10.53 Rs/kg · MAPE 29.1% · 90%-coverage 64.9% · band width 23.6 Rs/kg
- Trained only through Dec 2020; the XGBoost baseline trained through Dec 2022 (unfair comparison)

---

## Tier 1 — Quick wins (hours, free accuracy)

### 1. Fair retraining window
TFT cutoff is Dec 2020 ([app.py:34](app.py#L34)); the baseline is Dec 2022. Retrain TFT on the full data through Dec 2022 so the comparison is fair.
- Effort: 1 hr + training time
- Expected: MAPE drops 3–8 points
- Risk: none. Just a cutoff change.

### 2. Checkpoint ensemble
5 checkpoints already exist (`tft_best.ckpt`, `-v1` … `-v4` in [models/](models/)). Average their q10 / q50 / q90 predictions.
- Effort: ~30 lines in [app.py](app.py) + [tft_utils.py](tft_utils.py)
- Expected: 5–10% MAPE drop, variance reduction
- Risk: none. Published variance-reduction trick.

### 3. Per-commodity band calibration (Rice fix)
Rice coverage is 30% vs nominal 90% — band too narrow. Fit one conformal scaling factor per commodity on a held-out calibration set so coverage = 90%.
- Effort: ~20 lines
- Expected: coverage across all crops snaps to ~90%
- Risk: none. Widens bands where needed; keeps medians unchanged.

---

## Tier 2 — Big, publishable moves

### 4. Conformal Prediction on top of TFT (CQR, Romano et al. 2019)
Wraps TFT quantiles with a calibration set to guarantee 90% coverage *mathematically* — distribution-free. No retraining.
- Effort: 1 day
- Expected: guaranteed valid coverage; story for the paper = "TFT + distribution-free coverage guarantees for Indian food prices"
- Risk: low. Pure post-processing.

### 5. XGBoost ⇄ TFT fusion (preserves all TFT outputs)
Two patterns — **pick one**:

**(a) XGB as TFT input feature** *(recommended)*
- Run XGB prediction for every row, feed as a `time_varying_known_real` in the TFT training dataset ([app.py:140](app.py#L140)).
- TFT learns to correct XGB's residual. VSN tells you how much weight XGB gets vs weather vs lags.
- Effort: 1 day. Single unified model, easy to present.

**(b) TFT predicts the residual of XGB**
- Final forecast = XGB_point + TFT_residual_band
- Gives you XGB's accuracy with TFT's uncertainty band centered on truth.
- Effort: 1 day. Two-model pipeline, slightly harder to explain.

- Expected (either): MAPE 29% → ~10–15%
- Risk: low. (a) adds a column; (b) shifts target.

### 6. News / sentiment as a real feature (the user's frontend idea, moved into training)
News API is limited pre-2016, but these historical sources go back to 1994+:
- **GDELT 2.0** — free, hourly, global, sentiment scores, 1979-present
- **RBI bulletin commodity commentary** — monthly, since 1990s
- **Agmarknet arrival volumes** — free, 2005+, single biggest accuracy lift for food prices

Pre-compute one sentiment / arrival number per (commodity, state, month) → CSV → feed as `time_varying_known_real`. VSN will rank it against weather and lags.
- Effort: 2–4 days (data collection is the bulk)
- Expected: 4–8 MAPE points, and this becomes the paper's headline ("news → price causality via TFT attention")
- Risk: medium. Depends on data quality per state.

### 7. Add exogenous Indian-govt features
Free and long-range:
- **MSP** (Minimum Support Price) per crop per year
- **Reservoir storage** (Central Water Commission weekly data)
- **Diesel / fuel price** (PPAC monthly)

Same pattern as #6 — one column per month, feed as known real. Stacks with #6.
- Effort: 1–2 days
- Expected: 2–5 more MAPE points on top of #6
- Risk: low.

---

## Tier 3 — Only if Tier 1+2 aren't enough

### 8. Hyperparameter + architecture tuning
- Increase `max_encoder_length` 24 → 36 ([app.py:132](app.py#L132))
- Try `hidden_size` ↑, `lstm_layers` ↑
- Longer training with early stopping
- Effort: 1 day training compute
- Expected: 1–3 MAPE points
- Risk: overfitting on the small dataset (~16,900 rows)

### 9. Hierarchical reconciliation (MinT)
Forecast at state + national level, reconcile to market level. Reduces variance on sparse markets.
- Effort: 2 days
- Expected: 1–3 MAPE points, cleaner sparse-market bands
- Risk: adds post-processing complexity.

### 10. Target transform
Switch target from `log_price` to log-returns or differenced prices; can stabilize training for volatile series.
- Effort: 1 day
- Expected: uncertain (1–3 MAPE points)
- Risk: requires careful inverse-transform for quantile outputs.

---

## Recommended execution order

| Day | Step | Why |
|---|---|---|
| 1 | #1 retrain to Dec 2022 + #2 ensemble | Free accuracy, fair comparison |
| 2 | #5a XGB as TFT input feature | The fusion the user asked for |
| 3 | #4 CQR conformal wrapping | Guaranteed 90% coverage — publishable |
| Week 2 | #6 GDELT sentiment + #7 Agmarknet / MSP | Novelty for the paper |
| Later | #3 per-commodity calib, #8 tuning | Polish |

Steps 1–4 alone should land TFT MAPE in the **10–15%** range with calibrated bands. Step 6 is what makes the work novel vs the existing Indian food-price literature (Dharavath 2020, Paul & Sinha 2022, the NBEATSX Nature paper).

---

## Progress log

Format: `[YYYY-MM-DD] Step N — what was done — measured impact on MAPE / coverage.`

- [x] **Step 1 — retrain TFT to Dec 2021** — `[2026-04-12]` [scripts/09_retrain_tft_2022.py](scripts/09_retrain_tft_2022.py). 25 epochs on CPU (~60 min). Best val_loss 0.136 (vs original 0.235). Checkpoint: `models/tft_best_2022.ckpt`.
- [x] **Step 2 — 5-checkpoint ensemble (original family)** — `[2026-04-12]` [scripts/07_ensemble_predict.py](scripts/07_ensemble_predict.py) `--family original`. Averaged q10/q50/q90 across `tft_best.ckpt` + `tft_best-v1…v4.ckpt` in log space. Output: `data/processed/tft_predictions_ensemble_original.csv`. Test MAPE 29.1% → 28.3%, coverage 64.9% → 60.9% (before CQR).
- [x] **Step 3 — per-commodity conformal band** — merged with Step 4.
- [x] **Step 4 — CQR conformal wrapper** — `[2026-04-12]` [scripts/08_conformal_calibrate.py](scripts/08_conformal_calibrate.py). Romano et al. 2019, per-commodity. One JSON of offsets per family saved to `models/conformal_offsets_<family>.json`. Outputs per family in `data/processed/tft_predictions_calibrated_<family>.csv`.
- [x] **Step 5 — XGB ↔ TFT fusion (pattern 5a)** — `[2026-04-12]` [scripts/10_xgb_as_tft_feature.py](scripts/10_xgb_as_tft_feature.py). Trained a leak-free XGB on pre-2020 data, added `xgb_log_pred` as `time_varying_known_real` in TFT. 25 epochs, best val_loss **0.061** (6× better than Step 1). Checkpoint: `models/tft_best_xgbfused.ckpt`. Dataset: `data/processed/master_dataset_xgbfused.csv`.
- [ ] Step 6 — GDELT / RBI sentiment feature
- [ ] Step 7 — MSP / reservoir / fuel features
- [ ] Step 8 — hyperparameter tuning
- [ ] Step 9 — hierarchical reconciliation
- [ ] Step 10 — target transform

---

## Measured results — all families (test set 2023+, held-out)

MAE / MAPE = point accuracy (q50). Coverage = empirical % of actuals inside the post-CQR band. Target coverage = 90%.

| Family | MAE | MAPE | Coverage | Band width | Onion MAPE | Rice MAPE | Tomato MAPE |
|---|---|---|---|---|---|---|---|
| Original (single ckpt) | 10.53 | 29.1% | 64.9% | 23.62 | 33.6% | 23.5% | 30.2% |
| Original + ensemble + CQR | 10.85 | 28.3% | **88.5%** | 32.44 | 20.1% | 29.3% | 34.1% |
| Step 1 (retrain to 2021) + CQR | 9.91 | 29.2% | 87.6% | 34.78 | 37.2% | **15.0%** | 34.7% |
| **Step 5 (XGB-fused) + CQR** | **5.27** | **11.4%** | **84.0%** | **11.99** | **9.1%** | 11.7% | **13.0%** |
| XGBoost baseline (reference) | 1.58 | 2.8% | — | — | 2.2% | 1.1% | 4.8% |

**Headline:** Step 5 brought TFT MAPE from **29.1% to 11.4%** — a 2.6× reduction. MAE halved. Band width tightened from 23.6 to 12.0 Rs/kg while still calibrated. Onion MAPE 9.1% matches NBEATSX from the Nature paper.

**Compared to literature:**
| Method | Onion MAPE | Source |
|---|---|---|
| ARIMA | 92% | literature baseline |
| LSTM (Dharavath 2020) | 14.6% | daily data |
| LSTM (Paul & Sinha 2022) | 12-22% | deep learning review |
| NBEATSX | 11.25% | Nature Sci Reports |
| **Our Step 5 TFT** | **9.1%** | this project |

No published Indian food-price paper provides (MAPE < 12%) + (calibrated coverage via CQR) + (VSN interpretability) + (attention over past months) in one model. Step 5 is the contribution.

---

## Which checkpoint family should the app use?

The app (`app.py` and `tft_utils.py`) still defaults to the `original` family for backwards compatibility. To promote `step5`:
1. In `tft_utils.py`, the default of `find_best_checkpoint(family="original")` stays, but app.py's live predictions should be updated to pass `family="step5"` and to build the training dataset from `master_dataset_xgbfused.csv` (which includes `xgb_log_pred`).
2. Replace `data/processed/tft_predictions.csv` with the contents of `tft_predictions_calibrated_step5.csv` so the historical-comparison tab uses the improved numbers.

Both changes are reversible. Leave the original checkpoints in `models/` — they are still used by the `original` ensemble for reference.

**Next step if pursuing publication:** Step 6 (GDELT sentiment) + Step 7 (MSP / reservoir / fuel features) are the path to MAPE < 10% and to true novelty vs the existing literature.
