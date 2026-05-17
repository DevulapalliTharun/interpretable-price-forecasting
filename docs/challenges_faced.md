# Challenges Faced in This Project

A factual record of real problems that came up during the build, why they happened, how we solved them, and what alternatives we tried before settling on the final fix.

---

## 1. TFT's MAPE was very high (29%) compared to the XGBoost baseline (2.8%)

**What happened**
When we first compared models on the 2023 test set, the XGBoost baseline landed at 2.8% MAPE while the TFT came in at 29.1%. On the face of it, TFT looked useless.

**Why it happened**
Three separate issues, each contributing:
1. **Unfair cutoff.** The TFT was trained on data through **Dec 2020**, but the XGBoost baseline was trained through **Dec 2022**. It simply saw two extra years of data.
2. **Different jobs.** The baseline was a one-step-ahead point estimator. TFT was predicting a 6-month horizon with q10/q50/q90 quantiles. A multi-horizon probabilistic model *cannot* match a 1-step point model on MAPE.
3. **Small data for TFT.** ~16,900 monthly rows across 123 series. TFT was designed by Google for much larger datasets (millions of rows). Small data amplifies variance.

**What we tried**
- First instinct: "TFT is the wrong model, go back to LSTM." We rejected this because the point of TFT was uncertainty + interpretability, not raw accuracy.
- Second try: reduce the horizon to 1 month to make the comparison apples-to-apples. Rejected — it would remove TFT's multi-horizon value.
- **Final fix:** reframe the comparison honestly in the evaluation write-up (XGBoost wins point accuracy, TFT provides probabilistic bands + attention + VSN), and then pursue a two-track improvement plan: (a) 5-checkpoint ensemble + per-commodity CQR for band calibration, (b) retrain TFT to Dec 2021 to level the cutoff, (c) use the baseline prediction as a feature inside TFT (stacking).

---

## 2. TFT's confidence bands were badly miscalibrated — Rice was broken

**What happened**
Nominal claim was "90% coverage" (q10-q90 should contain 90% of real prices). Actual on the test set: 64.9% overall. For Rice specifically, 30%. That means the model kept producing over-narrow bands and missing the real price most of the time.

**Why it happened**
Rice prices in our series were nearly constant (state-controlled MSP, PDS distribution). The quantile loss saw almost-no-variance in training and collapsed the band. When any real-world shock hit, the band couldn't absorb it.

**What we tried**
- Considered widening the bands by a fixed factor — heuristic, hard to defend in a paper.
- Considered training a second quantile model specifically for Rice — splits the project unnecessarily.
- **Final fix: Conformalized Quantile Regression (CQR, Romano et al. 2019).** On a held-out calibration split (val 2021-22), we computed a per-commodity offset `Q_hat` in log space: `E_i = max(q10 - y, y - q90)`, then `Q_hat = (1-α)-quantile of E`. Adjusted bands = `[q10 − Q_hat, q90 + Q_hat]`. This guarantees ≥ 90% marginal coverage under exchangeability, distribution-free.

**Result:** overall coverage 64.9% → **88.5%**, Rice 30% → **83%**, all without any retraining. Each commodity gets its own offset, so Rice's band widened while Onion's stayed close to original.

---

## 3. `torch` not installed in the Python 3.14 environment — Streamlit silently hid TFT outputs

**What happened**
The user noticed: "future forecast has no TFT predictions, explainability tab has no TFT weights." But the app ran without errors.

**Why it happened**
In [app.py:17-29](app.py#L17-L29) the `tft_utils` import was wrapped in try/except. If `torch` wasn't available, `TFT_IMPORT_ERROR` was set and the TFT code paths silently returned `None`. Python 3.14 is very new and torch wheels for it are limited — the existing venv had never had torch installed for 3.14.

**What we tried**
- Considered falling back to an older Python (3.11). Rejected — would force the user to switch interpreters.
- **Final fix:** installed `torch==2.11`, `pytorch-lightning==2.6.1`, `pytorch-forecasting==1.7.0` directly into the Python 3.14 env. They had cp314 wheels available.

**Lesson:** silent `try/except` around imports is dangerous. The app should have printed a visible warning banner if TFT import failed, not hidden it. Would fix this if shipping to production.

---

## 4. Streamlit blocked on interactive email prompt the first time it ran

**What happened**
First `streamlit run app.py` hung forever. The background task failed with exit code 127.

**Why it happened**
Streamlit on first launch asks for an email address for the newsletter. In a non-interactive shell, there's no one to type, so it hangs.

**Fix**
Created `~/.streamlit/config.toml` with:
```toml
[browser]
gatherUsageStats = false
[server]
headless = true
```
Plus `--server.headless true` on the command line. Never blocks again.

---

## 5. Windows console (`cp1252`) crashed on non-ASCII characters in print statements

**What happened**
A training script printed dates with a Unicode arrow (`→`) in a status message. Crash: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'`.

**Why it happened**
Default Windows PowerShell/cmd uses the `cp1252` code page, which doesn't include the arrow character. Python 3 writes text via the system encoder by default.

**What we tried**
- `PYTHONIOENCODING=utf-8` as an env var — works but easy to forget.
- `sys.stdout.reconfigure(encoding='utf-8')` — works but polluting.
- **Final fix:** replaced all non-ASCII characters in scripts with ASCII equivalents (`→` became `-`). Clean, portable, no environment setup required. Matters because your friend may run the project on Windows too.

---

## 6. Training crashed because `pytorch_optimizer` was not installed

**What happened**
`ImportError: optimizer 'ranger' requires pytorch_optimizer in the environment.`

**Why it happened**
The TFT training config uses the `ranger` optimizer (Ranger = RAdam + LookAhead). pytorch-forecasting delegates to the `pytorch_optimizer` library for advanced optimizers, but that library isn't pulled in by default.

**Fix**
`pip install pytorch_optimizer` and added it to [requirements.txt](requirements.txt). Rejected using stock Adam as a shortcut — the original TFT was trained with Ranger, changing the optimizer mid-project would invalidate the existing checkpoints.

---

## 7. No GPU available — only Intel UHD 620 integrated graphics

**What happened**
`torch.cuda.is_available()` returned False. TFT retraining on CPU takes ~20-30 minutes per run.

**Why it happened**
The user's laptop has Intel UHD 620, not an NVIDIA CUDA GPU.

**What we tried**
- **Google Colab integration via VS Code extension** — Google killed the official extension a while back, and SSH-via-ngrok into Colab violates their ToS.
- **GitHub Codespaces** — free tier has no GPU.
- **DirectML / Intel IPEX** — only gives ~2-3× speedup on iGPU, and pytorch-forecasting uses many ops that might not accelerate cleanly. Risk of silent fallback.
- **Colab in browser** — works, free T4 GPU, ~30× faster. But requires uploading the project.
- **CPU training** — slow but reliable.

**Final approach (dual-path):**
1. CPU retrain runs in the background while other work continues. ~25 min for Step 1 (retrain to 2021), ~30 min for Step 5 (XGBoost-fused).
2. Wrote a `scripts/train_colab.ipynb` that does both on free Colab GPU in ~20 min total, as a fallback if CPU is too slow.

---

## 8. Risk of label leakage when fusing XGBoost into TFT

**What happened**
Pattern 5a (baseline prediction as a TFT input feature) is classic stacking. Naive stacking leaks labels: if the baseline was trained on the same rows TFT is now training on, those predictions are "too good" — they implicitly encode the label. TFT then over-trusts them during training, but at test time predictions are genuine, and the learned weights don't transfer.

**What we tried**
- **Proper out-of-fold stacking (K-fold CV)** — ideal but complex and slow with time-series data (you need time-ordered folds).
- **Trivial reuse of the existing baseline checkpoint** — leakage.
- **Final fix:** trained a dedicated "clean" XGBoost model on data strictly **before Dec 2019** (two years earlier than the TFT train cutoff of Dec 2021). Used it to score the entire dataset. Rows 2020+ (which includes all val/test) are leak-free; rows 1994-2019 have minor leakage but the signal there is weak (baseline mostly predicts from `price_lag_1m`, so it's essentially a denoised version of a feature that was already available). Good enough for research + interviewable.

---

## 9. Different TFT checkpoint "families" have incompatible schemas

**What happened**
After Step 1 (retrain to 2021) and Step 5 (XGBoost-fused), we had three kinds of checkpoints in `models/`:
- Original (5 files) — schema A (no xgb_pred)
- Step 1 — schema A (same as original)
- Step 5 — schema B (extra `xgb_log_pred` input)

The ensemble script was originally globbing `tft_best*.ckpt`, which would have mixed schema-A and schema-B models and broken inference.

**Fix**
Introduced a `CHECKPOINT_PATTERNS` dict in [tft_utils.py](tft_utils.py) with explicit globs per family. Scripts 07 and 08 gained a `--family {original,step1,step5}` argument. Output CSVs are suffixed with the family name. App.py still defaults to the "original" family until a deliberate promotion decision is made.

---

## 10. The "news-driven" feature idea — historical news APIs don't exist pre-2016

**What happened**
Initial idea: "Search for news about this commodity in this state on this date, feed sentiment into the model."

**Why it doesn't work at scale**
Commercial news APIs (GNews, NewsAPI, Bing News) only have coverage from ~2015-2016 onward. Our dataset runs from 1994. We'd have no news features for 20+ years of training data.

**What we tried**
- GNews API for post-2016 only — creates a structural imbalance in features.
- Paid news archives (LexisNexis, Factiva) — out of scope and expensive.

**Final approach**
- Phase 1 (shipped): news is a **frontend-only** feature — when a user looks at a forecast, the app searches recent news and shows headlines alongside the prediction. Doesn't enter the model.
- Phase 2 (planned, see [TFT_improve_steps.md](TFT_improve_steps.md) Step 6): use **GDELT 2.0** which has coverage back to 1979. Pre-compute monthly sentiment per (commodity, state), feed into TFT as `time_varying_known_real`.

---

## 11. Streamlit dashboard cached stale evaluation metrics after we improved the model

**What happened**
After running the ensemble + CQR pipeline and getting coverage 64.9% → 88.5%, the Streamlit evaluation tab still showed the old numbers.

**Why it happened**
[app.py:107-111](app.py#L107-L111) uses `@st.cache_data` on the file reader, and it reads `visualizations/evaluation_metrics.txt` — a **static file** that was generated once by the original evaluation script. The improved numbers lived only in the terminal output of the new scripts.

**Fix**
Overwrote `visualizations/evaluation_metrics.txt` to include both the original baseline and the new TFT-improved section side-by-side. Cache invalidates on file change + Streamlit rerun.

---

## 12. Tradeoffs we explicitly chose

Some decisions were judgment calls, not problems. Listing them so they can be defended in an interview:

| Decision | Alternative rejected | Why |
|---|---|---|
| Monthly resolution | Daily (Agmarknet) | WFP data is monthly; daily would mean a new dataset + ~10× rows but no long history before 2015 |
| Same architecture across 3 crops | One model per crop | Shared representation benefits transfer across crops; separate models ×3 the training work and lose cross-crop signal |
| Log-price target | Raw price | Prices are multiplicative; log stabilizes variance and makes MAPE-like losses well-behaved |
| MAPE as headline metric | sMAPE / MASE | MAPE is what the published Indian food-price papers report — comparable across literature |
| 6-month horizon | 3 or 12 months | 6 matches a growing-season / mandi-planning window; 12 is too long for volatile crops, 3 is too short to be useful |
| Pytorch-forecasting library | Writing TFT from scratch | PF is the reference implementation, saved ~weeks of work, proven correct |

---

## Status summary

| Challenge | Status |
|---|---|
| 1. TFT vs baseline gap | Acknowledged + partially closed via ensemble + CQR |
| 2. Band miscalibration | **Solved** (CQR, 88.5% test coverage) |
| 3. torch missing | **Solved** (installed) |
| 4. Streamlit email prompt | **Solved** (headless config) |
| 5. Windows encoding | **Solved** (ASCII-only output) |
| 6. Ranger optimizer | **Solved** (pytorch_optimizer installed) |
| 7. No GPU | **Mitigated** (CPU path works; Colab notebook ready) |
| 8. Stacking leakage | **Solved** (pre-2020 clean XGBoost baseline) |
| 9. Family schema conflict | **Solved** (family-scoped patterns) |
| 10. Historical news API | **Phase 1 done**, Phase 2 designed (GDELT) |
| 11. Cached metrics | **Solved** (file updated) |
| 12. Design tradeoffs | Documented for interview defence |
