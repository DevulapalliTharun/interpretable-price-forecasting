# Setup Guide — Food Price Forecasting (TFT)
## Step-by-Step Instructions for Running from ZIP

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`. Core filenames remain `xgb_*` because they now match the implementation (`04_train_xgboost.py`, `xgb_baseline.pkl`, `xgb_log_pred`).

---

## STEP 0: What You Need Before Starting

- A laptop/PC with **Windows 10/11** (or Linux/Mac)
- **Python 3.9 or higher** installed
- **Internet connection** (for NASA weather API in Step 7)
- **(Recommended)** NVIDIA GPU with CUDA support (GTX 1650 or better)
- **~5 GB free disk space** (PyTorch is large)

### Check if Python is installed:
```bash
python --version
```
Should show `Python 3.9.x` or higher. If not installed, download from https://www.python.org/downloads/

### Check if you have an NVIDIA GPU:
```bash
nvidia-smi
```
If this shows a table with your GPU name and CUDA version, you have GPU support.
Note down the **CUDA Version** number (e.g., 12.1, 11.8) — you need it in Step 3.

---

## STEP 1: Extract the ZIP

1. Right-click the ZIP file
2. Click **Extract All**
3. Choose a location (e.g., `C:\Users\YourName\Desktop\`)
4. You should now have a folder with this structure:

```
wfp-india-tft-forecasting/
├── README.md
├── SETUP_GUIDE.md          ← You are reading this
├── requirements.txt
├── app.py
├── data/
│   └── raw/
│       ├── wfp_food_prices_ind.csv
│       └── wfp_markets_ind.csv
├── scripts/
│   ├── 00_filter_prices.py
│   ├── 01_fetch_weather.py
│   ├── 02_merge_features.py
│   ├── 03_train_tft.py
│   ├── 04_train_xgboost.py
│   ├── 05_generate_tft_predictions.py
│   └── 06_evaluate.py
├── models/                  (empty — will be filled by training)
└── visualizations/          (empty — will be filled by evaluation)
```

---

## STEP 2: Open Terminal in the Project Folder

### Windows:
1. Open the extracted folder in File Explorer
2. Click the address bar at the top
3. Type `cmd` and press Enter
4. A Command Prompt opens in the project folder

### OR use PowerShell:
```powershell
cd "C:\Users\YourName\Desktop\wfp-india-tft-forecasting"
```

### Linux/Mac:
```bash
cd ~/Desktop/wfp-india-tft-forecasting
```

---

## STEP 3: Create a Virtual Environment

A virtual environment keeps this project's packages separate from other Python projects.

```bash
python -m venv venv
```

### Activate it:

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```
If PowerShell blocks this, run first: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux/Mac:**
```bash
source venv/bin/activate
```

Your prompt should now show `(venv)` at the beginning. Example:
```
(venv) C:\Users\YourName\Desktop\wfp-india-tft-forecasting>
```

**IMPORTANT:** Every time you open a new terminal to work on this project,
you must activate the venv again. If you see `ModuleNotFoundError`, you
probably forgot to activate.

---

## STEP 4: Install PyTorch (GPU or CPU)

This is the most important step. Install the CORRECT version for your hardware.

### Option A: You have an NVIDIA GPU (RECOMMENDED)

First, check your CUDA version:
```bash
nvidia-smi
```
Look for `CUDA Version: XX.X` in the top-right of the output.

**For CUDA 12.1 or higher:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Option B: No GPU (CPU only)

```bash
pip install torch torchvision
```
Training will be slower (~30-40 min instead of ~5-10 min) but it will work.

### Verify PyTorch installation:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output (GPU):**
```
PyTorch version: 2.x.x+cu121
GPU available: True
GPU name: NVIDIA GeForce RTX 3060
```

**Expected output (CPU):**
```
PyTorch version: 2.x.x+cpu
GPU available: False
GPU name: None
```

If GPU is `False` but you have an NVIDIA GPU, you installed the wrong version.
Go back and reinstall with the `--index-url` flag.

---

## STEP 5: Install All Other Packages

Copy-paste this entire command:

```bash
pip install pytorch-forecasting pytorch-lightning pytorch-optimizer pandas numpy scikit-learn requests plotly streamlit joblib kaleido gnews nltk
```

Then download NLTK sentiment data:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

This installs:

| Package | What it does |
|---|---|
| `pytorch-forecasting` | TFT model implementation |
| `pytorch-lightning` | Training framework (handles epochs, callbacks, GPU) |
| `pytorch-optimizer` | Ranger optimizer (RAdam + LookAhead) |
| `pandas` | Data manipulation (CSVs, DataFrames) |
| `numpy` | Numerical operations |
| `scikit-learn` | Label encoding and preprocessing utilities |
| `xgboost` | XGBoost baseline and fused baseline signal |
| `requests` | NASA POWER API calls for weather data |
| `plotly` | Interactive charts in the dashboard |
| `streamlit` | Web dashboard framework |
| `joblib` | Saving/loading baseline model |
| `kaleido` | Exporting plotly charts as PNG images |
| `gnews` | Live news search for spike context |
| `nltk` | Sentiment analysis (VADER) for news |

**This will take 2-5 minutes.** Wait for it to finish completely.

### Verify all packages are installed:

```bash
python -c "import torch; import pytorch_forecasting; import pytorch_lightning; import plotly; import streamlit; import sklearn; print('All packages OK')"
```

Should print: `All packages OK`

If any package fails, install it individually:
```bash
pip install <package-name>
```

---

## STEP 6: Run Script 00 — Filter Prices

```bash
python scripts/00_filter_prices.py
```

**What it does:** Cleans the raw WFP CSV — drops National Average rows,
keeps only Retail + Onions/Tomatoes/Rice + KG unit, filters for series
with 60+ months of data.

**Expected output:**
```
Rows:      ~18,000
Series:    ~123
Onions:    ~38 series
Tomatoes:  ~45 series
Rice:      ~40 series
```

**Verify:** Check that `data/processed/prices_filtered.csv` exists.

---

## STEP 7: Run Script 01 — Fetch NASA Weather

```bash
python scripts/01_fetch_weather.py
```

**What it does:** Calls NASA POWER API to download monthly temperature,
rainfall, and humidity data for all market locations from 1994 to 2025.

**Time:** ~4-5 minutes (53 API calls with 2-second delay between each)

**Requires:** Internet connection

**Expected output:**
```
Rows:      ~20,352
Markets:   53
Date range: 1994-01 to 2025-12
```

**Verify:** Check that `data/raw/nasa_weather_1994_2026.csv` exists.

**If it fails:** NASA POWER API may be temporarily down. Wait 10 minutes and retry.

---

## STEP 8: Run Script 02 — Merge Features

```bash
python scripts/02_merge_features.py
```

**What it does:** Joins price data with weather data, creates lag features,
rolling averages, seasonal encoding, weather shock indicators, COVID flag.

**Expected output:**
```
Rows:      ~16,900
Columns:   26
Series:    ~123
```

**Verify:** Check that `data/processed/master_dataset.csv` exists.

---

## STEP 9: Run Script 03 — Train TFT Model

This is the main training step. Use the correct command for your hardware.

### With GPU (RECOMMENDED):

```bash
python scripts/03_train_tft.py --gpus 1
```

**Time:** ~10-15 minutes
**Settings:** hidden_size=32, attention_head_size=2, 150 epochs, batch_size=64, dropout=0.3, lr=0.01

### Without GPU (CPU only):

```bash
python scripts/03_train_tft.py --batch_size 128 --epochs 50
```

**Time:** ~30-40 minutes

### What to watch for during training:

```
Epoch 0:  train_loss=0.35  val_loss=0.30   ← Starting, loss is high
Epoch 10: train_loss=0.22  val_loss=0.20   ← Learning well
Epoch 25: train_loss=0.17  val_loss=0.15   ← Getting good
Epoch 40: train_loss=0.15  val_loss=0.14   ← Converging
...
Training stops when val_loss stops improving for 5 epochs (early stopping)
```

**Good signs:**
- val_loss steadily decreasing
- val_loss < 0.15 by end of training
- train_loss and val_loss are close (not overfitting)

**Bad signs:**
- val_loss increasing while train_loss decreasing = OVERFITTING
  Fix: Increase dropout (edit script, change 0.3 to 0.4)
- val_loss stuck above 0.25 = UNDERFITTING
  Fix: Increase epochs or hidden_size
- `CUDA out of memory` = batch_size too large
  Fix: Add `--batch_size 32` to the command

**Expected output:**
```
Best val_loss: < 0.15
Best model: models/tft_best.ckpt
```

### If you get CUDA out of memory:

```bash
python scripts/03_train_tft.py --gpus 1 --batch_size 32
```

If it still fails with batch_size=32, your GPU has very limited VRAM.
Fall back to CPU:
```bash
python scripts/03_train_tft.py --batch_size 128 --epochs 50
```

---

## STEP 10: Run Script 04 — Train XGBoost Baseline

```bash
python scripts/04_train_xgboost.py
```

**What it does:** Trains an XGBoost model as a comparison baseline.
XGBoost produces point predictions (no uncertainty bands).

**Time:** < 1 minute

**Expected output:**
```
Test MAE:  ~1.5 Rs/KG
Test MAPE: ~3%
```

**Verify:** Check that `models/xgb_baseline.pkl` exists.

---

## STEP 11: Run Script 05 — Generate TFT Predictions

```bash
python scripts/05_generate_tft_predictions.py
```

**What it does:** Loads the trained TFT model and generates:
1. Quantile predictions (q10/q50/q90) for all data splits
2. Attention weights (which past months the model focused on)
3. Variable importance (which features mattered per crop)

**Time:** 2-3 minutes

**Expected output:**
```
tft_predictions.csv              — quantile predictions
tft_attention.csv                — aggregated attention weights
tft_attention_detail.csv         — per-series attention
tft_variable_importance.csv      — encoder + decoder variable weights
tft_variable_importance_detail.csv — per-series variable importance
```

**Verify:** Check that all 5 CSV files exist in `data/processed/`.

**DO NOT SKIP THIS STEP.** Without it, the dashboard will not show TFT results,
spike explanations, or interpretability features.

---

## STEP 12: Run Script 06 — Evaluate

```bash
python scripts/06_evaluate.py
```

**What it does:** Computes final metrics for both models, generates
comparison plots, and saves everything to `visualizations/`.

**Time:** < 1 minute

**Output files created:**
- `visualizations/evaluation_metrics.txt` — All metrics in text format
- `visualizations/quantile_forecast_onions.png`
- `visualizations/quantile_forecast_tomatoes.png`
- `visualizations/quantile_forecast_rice.png`
- `visualizations/attention_heatmap.png` — Which past months TFT focuses on
- `visualizations/attention_distribution.png` — Attention weight curves per crop
- `visualizations/tft_encoder_importance.png` — Which historical features matter per crop
- `visualizations/tft_decoder_importance.png` — Which future features matter per crop
- `visualizations/variable_importance_comparison.png` — Onions vs Rice vs Tomatoes
- `visualizations/feature_importance_xgboost.png` — XGBoost baseline
- `visualizations/shock_events_overlay.png`

---

## STEP 13: Launch the Dashboard

```bash
streamlit run app.py
```

**First time:** It may ask for an email. Just press Enter to skip.

**Opens at:** http://localhost:8501

### Dashboard Features (4 Tabs):

**Tab 1 — Price Forecast:**
- Blue line: historical actual prices
- Red dashed: TFT median prediction
- Red shaded area: 90% confidence band (q10 to q90)
- Grey dotted: XGBoost baseline
- AUTO SPIKE DETECTION: expandable cards for each detected spike
  - Model-driven reasons from TFT VSN weights (NOT hardcoded)
  - Bar chart showing which features drove the spike
  - "Search news" button — live Google News search for context

**Tab 2 — Future Forecast:**
- Chart with recent history + forward projections
- MONTHLY BREAKDOWN CARDS for each forecast month:
  - Price prediction with confidence band range
  - Risk level: STABLE / MODERATE / HIGH RISK
  - Model-driven reasons from decoder weights
- "Search latest news" button for live articles

**Tab 3 — Model Explainability:**
- XGBoost feature importance (static)
- TFT encoder variable weights (which historical features matter)
- TFT attention over past months (which past months the model focused on)
- TFT decoder weights (which future features matter)
- Weather vs price correlation charts

**Tab 4 — Present vs Predicted:**
- Side-by-side comparison on test data (2021-Jul 2023)
- Metrics table: MAE, MAPE, RMSE, Coverage
- Uncertainty width chart

### To stop the dashboard:
Press `Ctrl+C` in the terminal.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
You forgot to activate the virtual environment.
```bash
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

### "CUDA out of memory"
Reduce batch size:
```bash
python scripts/03_train_tft.py --gpus 1 --batch_size 32
```

### "No objects to concatenate" in script 01
NASA API may be down. Wait and retry. Check your internet connection.

### "AssertionError: filters should not remove entries"
The dataset for that split is too short. This is handled automatically
in the prediction script — it includes context months.

### Dashboard shows no TFT results
You skipped script 05. Run it:
```bash
python scripts/05_generate_tft_predictions.py
```
Then refresh the browser.

### "torch.cuda.is_available() returns False"
You installed CPU-only PyTorch. Reinstall:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Training loss not decreasing
- Check that `master_dataset.csv` has ~16,000+ rows
- Try a different learning rate: `--lr 0.01`
- Make sure you did not skip scripts 00-02

### Port 8501 already in use
Another Streamlit instance is running. Kill it:
```bash
# Windows:
taskkill /F /IM streamlit.exe
# Linux/Mac:
pkill -f streamlit
```
Then relaunch: `streamlit run app.py`

---

## Complete Command Summary (Copy-Paste Ready)

```bash
# === ONE-TIME SETUP ===
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-forecasting pytorch-lightning pytorch-optimizer pandas numpy scikit-learn requests plotly streamlit joblib kaleido gnews nltk
python -c "import nltk; nltk.download('vader_lexicon')"

# === DELETE OLD MODELS (if any exist from previous CPU run) ===
del models\tft_best*.ckpt 2>nul
del models\xgb_baseline.pkl 2>nul
del data\processed\tft_predictions.csv 2>nul
del data\processed\tft_attention*.csv 2>nul
del data\processed\tft_variable_importance*.csv 2>nul

# === RUN FULL PIPELINE ===
python scripts/00_filter_prices.py
python scripts/01_fetch_weather.py
python scripts/02_merge_features.py
python scripts/03_train_tft.py --gpus 1
python scripts/04_train_xgboost.py
python scripts/05_generate_tft_predictions.py
python scripts/06_evaluate.py
streamlit run app.py
```

---

## What the GPU Run Does Differently

These settings are already configured in the scripts — no manual editing needed:

| Setting | CPU (old) | GPU (current) | Why changed |
|---|---|---|---|
| hidden_size | 16 | **32** | More capacity to learn |
| attention_heads | 1 | **2** | Learns seasonality + trends |
| hidden_continuous_size | 8 | **16** | Richer feature representation |
| dropout | 0.2 | **0.3** | Wider bands, better coverage |
| learning_rate | 0.03 | **0.01** | Smoother convergence |
| epochs | 30 | **150** | More training (early stopping handles it) |
| batch_size | 128 | **64** | Better gradient estimates |
| early_stopping | patience=5 | **patience=8** | More time to find optimum |
| encoder_length | 24 | **18** | Fewer gaps, more reliable |
| prediction_length | 6 | **3** | 1 season = realistic |
| Train split | up to 2020 | **up to 2019** | Keeps 2020 COVID for validation |
| Val split | 2021-2022 | **2020** | COVID = hard stress test |
| Test split | 7 months | **31 months** | Much more test data |
| Estimated time | ~30-40 min | **~10-15 min** | GPU acceleration |

---

## Expected Results After GPU Training

| Metric | CPU result | GPU expected |
|---|---|---|
| TFT MAPE | ~38% | **~12-18%** |
| TFT 90% Coverage | ~13% | **~75-85%** |
| XGBoost MAPE | ~3% | ~5% |

---

## If You Need to Retrain

```bash
# 1. Delete old checkpoints (REQUIRED before retraining)
del models\tft_best*.ckpt
del data\processed\tft_predictions.csv
del data\processed\tft_attention*.csv
del data\processed\tft_variable_importance*.csv

# 2. Retrain
python scripts/03_train_tft.py --gpus 1

# 3. Regenerate predictions + interpretability (MUST do after retraining)
python scripts/05_generate_tft_predictions.py

# 4. Re-evaluate
python scripts/06_evaluate.py

# 5. Relaunch dashboard
streamlit run app.py
```
