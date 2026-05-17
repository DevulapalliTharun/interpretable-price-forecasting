# Reproducibility Guide

**Paper:** Interpretable and Uncertainty-Aware Food Price Forecasting for Indian Markets using Temporal Fusion Transformers  
**Author:** Devulapalli Tharun, NITK Surathkal

---

## Environment

| Item | Version used |
|---|---|
| Python | 3.11 or 3.12 recommended (tested on 3.14) |
| OS | Windows 11 / Ubuntu 22.04 |
| GPU | NVIDIA (CUDA 12.8) — optional; CPU works but training is slow |

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd kalakar_2

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Install PyTorch (pick ONE)
# GPU (CUDA 12.8):
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# CPU-only (for reviewers without a GPU):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install all other dependencies
pip install -r requirements.txt
```

---

## Data

Raw data files are already included in `data/raw/`:

| File | Source | License |
|---|---|---|
| `wfp_food_prices_ind.csv` | [WFP HDX](https://data.humdata.org/dataset/wfp-food-prices-for-india) | CC BY |
| `wfp_markets_ind.csv` | WFP HDX | CC BY |
| `nasa_weather_1994_2026.csv` | [NASA POWER API](https://power.larc.nasa.gov) | Public domain |

No download step is needed — raw files are committed to the repository.

---

## Reproducing all results (full pipeline)

Run scripts in this exact order from the project root:

```bash
# Step 0 — Filter and clean WFP prices
python scripts/00_filter_prices.py

# Step 1 — Fetch NASA POWER weather (already cached in data/raw/)
python scripts/01_fetch_weather.py

# Step 2 — Merge prices + weather into master dataset
python scripts/02_merge_features.py

# Step 3 — Train original TFT (train ≤ 2020)
python scripts/03_train_tft.py

# Step 4 — Train XGBoost baseline
python scripts/04_train_xgboost.py

# Step 5 — Generate TFT predictions (original family)
python scripts/05_generate_tft_predictions.py

# Step 6 — Evaluate all models, generate visualizations
python scripts/06_evaluate.py

# Step 7 — Ensemble predictions across checkpoints
python scripts/07_ensemble_predict.py --family original
python scripts/07_ensemble_predict.py --family step1
python scripts/07_ensemble_predict.py --family step5

# Step 8 — Conformal Quantile Regression calibration
python scripts/08_conformal_calibrate.py

# Step 9 — Retrain TFT with extended window (train ≤ 2021)
python scripts/09_retrain_tft_2022.py

# Step 10 — XGBoost-as-TFT-feature fusion model
python scripts/10_xgb_as_tft_feature.py

# Step 11 — Statistical validation (ablation, coverage, VSN bootstrap)
python scripts/11_explainability_stats.py
```

**Training time estimate (GPU):** Steps 03, 09, 10 each take ~30–90 min on a single GPU.  
**Training time estimate (CPU):** ~4–8 hours per training step.

---

## Using pre-trained artifacts (skip training)

All trained model artifacts are included in `models/`:

| File | Description |
|---|---|
| `tft_best.ckpt` | Original TFT checkpoint (train ≤ 2020) |
| `tft_best_2022.ckpt` | Retrained TFT checkpoint (train ≤ 2021) |
| `tft_best_xgbfused.ckpt` | XGBoost-fused TFT checkpoint (train ≤ 2021) |
| `tft_best_xgbfused-v1.ckpt` | Alternate fused checkpoint |
| `tft_config.json` | TFT hyperparameters (original) |
| `tft_config_2022.json` | TFT hyperparameters (retrained) |
| `tft_config_xgbfused.json` | TFT hyperparameters (fused) |
| `xgb_baseline.pkl` | XGBoost baseline model |
| `xgb_clean_2019.pkl` | Auxiliary XGBoost model (train ≤ 2019, used as TFT input) |
| `conformal_offsets_original.json` | CQR calibration offsets (original family) |
| `conformal_offsets_step1.json` | CQR calibration offsets (retrained family) |
| `conformal_offsets_step5.json` | CQR calibration offsets (fused family) |

If you only want to reproduce the evaluation numbers (not retrain), run only Step 11 after ensuring `data/processed/` is populated (included in the repository).

---

## Reproducing the dashboard

```bash
python -m streamlit run app.py --server.port 8501
# Open http://localhost:8501 in your browser
```

---

## Key results (from paper)

These numbers should be reproduced exactly by running the full pipeline:

| Model | MAE (Rs/kg) | MAPE | Coverage | Band Width |
|---|---|---|---|---|
| XGBoost baseline | 1.58 | 2.8% | — | — |
| TFT-Base | 10.53 | 29.1% | 64.9% | 23.62 |
| TFT-EnsCQR | 10.85 | 28.3% | 88.5% | 32.44 |
| TFT-Retrain21-CQR | 9.91 | 29.2% | 87.6% | 34.78 |
| **TFT-XGBFusion-CQR** | **5.27** | **11.4%** | **84.0%** | **11.99** |

Statistical validation (from `scripts/11_explainability_stats.py`):
- Fusion vs TFT-Base: $t = -22.34$, $p = 4.6 \times 10^{-86}$, Cohen's $d = -0.79$
- VSN bootstrap Kendall $\tau = 0.942$, 95% CI [0.895, 0.990]

---

## Notes for reviewers

- **No API key needed.** The NASA POWER API is free and public. The weather CSV is already cached in `data/raw/` so Step 1 can be skipped.
- **No GPU required to run the dashboard** — only inference is needed at runtime, not training.
- **Results may vary slightly** due to PyTorch non-determinism across hardware. The paper reports results from the checkpoint files in `models/` which are included.
- **Python 3.11 or 3.12** is recommended for best compatibility. Python 3.10+ should work.
