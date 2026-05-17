# Viva Notes: TFT Improvements Explained Simply

> **Terminology note:** The implemented point baseline is `xgboost.XGBRegressor`.

## Simple way to say the whole story

"My original TFT model was giving around 29% MAPE, so I improved it step by step. I kept the TFT because it gives not only a forecast, but also uncertainty bands, attention, and feature importance. My goal was to improve its accuracy without losing those TFT advantages."

---

## Upgrade 1: Fair TFT retraining

### What problem was there?

"At first, the comparison was unfair. My TFT was trained only up to December 2020, but my XGBoost baseline was trained up to December 2022. So naturally the baseline had more recent information."

### What I did

"I retrained the TFT using a later cutoff, so it could learn from more recent data. In the improved version, TFT training was extended up to December 2021, validation was on 2022, and testing was on 2023 data."

### Why I did it

"This made the comparison more fair and gave TFT more recent market behavior to learn from."

### Simple one-line version

"I first fixed the train-test setup so TFT was not handicapped by older training data."

---

## Upgrade 2: Checkpoint ensemble

### What problem was there?

"A single deep learning checkpoint can be unstable. Two runs of the same model can give slightly different predictions."

### What I did

"Instead of trusting only one TFT checkpoint, I used multiple saved checkpoints and averaged their predictions."

### Why I did it

"This reduces randomness and makes the model more stable."

### Simple one-line version

"I combined multiple TFT checkpoints so the final prediction is more stable than using just one saved model."

---

## Upgrade 3: Conformal calibration for prediction bands

### What problem was there?

"The TFT gives prediction bands, but initially those bands were not well calibrated. For example, for some crops, the true price was falling outside the band too often."

### What I did

"I added a calibration step after prediction. This step adjusts the upper and lower bands so that the uncertainty interval better matches the real errors seen on held-out data."

### Why I did it

"This improves reliability. The model is not only predicting a price, but also giving a more believable uncertainty range."

### Important point to say

"This step mainly improves the quality of the uncertainty band. It does not mainly target point MAPE."

### Simple one-line version

"I corrected the TFT confidence bands so that the uncertainty range matches reality more closely."

---

## Upgrade 4: XGBoost as an input feature inside TFT

### What problem was there?

"Even after the earlier improvements, TFT point accuracy was still behind the XGBoost baseline."

### What I did

"I trained a clean XGBoost model first. Then for every row in the dataset, I generated a baseline prediction and added it as a new input feature to TFT. So TFT does not work alone anymore. It learns using the original time-series features plus this baseline signal."

### Why I did it

"XGBoost is very strong at point prediction. TFT is strong at sequence learning, uncertainty bands, and interpretability. So I combined the strengths of both."

### Important clarification

"I did not simply average TFT and baseline outputs. I used the baseline prediction as an extra feature inside TFT."

### Why this was the biggest improvement

"Because TFT now gets a very strong machine learning signal from the baseline, and then it learns how to refine or correct that signal using its sequence model."

### Simple one-line version

"The biggest improvement was feeding the baseline prediction into TFT as an extra feature, so TFT could use both models' strengths together."

---

## Best short explanation to say to sir

"I improved TFT in four stages. First, I retrained it with a fairer and more recent data cutoff. Second, I used checkpoint ensembling to make predictions more stable. Third, I calibrated the prediction bands so the uncertainty range became more reliable. Fourth, I implemented baseline-informed TFT, where the XGBoost prediction is used as an input feature inside TFT. That last step gave the biggest accuracy gain."

---

## If sir asks: what exactly is implemented?

Say this:

"The implemented fusion is baseline-as-input-feature inside TFT. It is not just a side-by-side comparison, and it is not a simple average of both models. The XGBoost prediction is generated first and then given to TFT as one more input feature."

---

## If sir asks: what was the final improvement?

Say this:

"The best TFT variant is the fused TFT model (legacy artifact name: TFT-XGBFusion). Its overall test MAPE is about 11.4% across all test rows, covering all three commodities and all included markets in the 2023 test period."

---

## If sir asks: what do 9.1%, 13.0%, and 11.7% mean?

Say this:

"Those are the per-commodity test MAPEs of the improved TFT model: 9.1% for onions, 13.0% for tomatoes, and 11.7% for rice."

---

## Safest final summary

"So my contribution was not just improving accuracy. I improved TFT in a way that still preserves uncertainty bands, attention, and feature importance. The strongest upgrade was combining XGBoost's predictive power with TFT's interpretability and probabilistic forecasting."
