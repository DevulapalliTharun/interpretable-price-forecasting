# Model Variant Naming (Professional)

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

Use the names below in report, viva, slides, and publication-style writing.

## Recommended naming convention

| Internal tag | Professional name | When to use |
|---|---|---|
| original | TFT-Base | Baseline TFT reference model |
| original + ensemble + CQR | TFT-EnsCQR | Calibrated ensemble baseline |
| step1 | TFT-Retrain21 | Retrained timeline variant |
| step1 + CQR | TFT-Retrain21-CQR | Retrained + calibrated |
| step5 | TFT-XGBFusion | XGBoost-feature fused TFT |
| step5 + CQR | TFT-XGBFusion-CQR | Best calibrated fused variant |

## One-line definitions for report writing

- TFT-Base: Single-checkpoint TFT baseline trained on the original cutoff timeline.
- TFT-EnsCQR: Ensemble of TFT-Base checkpoints with conformal quantile calibration.
- TFT-Retrain21-CQR: Extended-cutoff retrained TFT with calibrated prediction intervals.
- TFT-XGBFusion-CQR: XGBoost-feature fused TFT with conformal calibration (best overall variant).

## Suggested usage in tables/figures

- Use concise labels in tables: `TFT-Base`, `TFT-XGBFusion-CQR`.
- In caption or footnote, optionally map once: "TFT-XGBFusion-CQR corresponds to internal step5 + CQR."
- Avoid using step numbers directly in final report text.
