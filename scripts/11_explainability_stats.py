"""
11_explainability_stats.py

Statistical validation of the TFT pipeline's explainability & novelty claims.
Produces three pieces of evidence for the paper, each with a bar graph and
a p-value / effect size:

  A. VSN stability       - bootstrap the per-series VSN importances,
                           report mean +/- 95% CI per feature and the mean
                           pairwise Kendall tau of the rankings.
                           Answers: are the VSN explanations stable?

  B. Ablation t-test     - compare per-(market, date) absolute error across
                           the three families (original, step1, step5).
                           Paired two-sided t-test + Cohen's d.
                           Answers: does the fused XGBoost feature significantly
                           improve the model?

  C. CQR coverage test   - per-commodity binomial test of empirical coverage
                           of the calibrated [q10, q90] band vs the nominal
                           80% target.
                           Answers: are our uncertainty bands statistically
                           valid and per-commodity calibrated?

Inputs (already on disk):
    data/processed/tft_variable_importance_detail.csv
    data/processed/tft_predictions_calibrated_original.csv
    data/processed/tft_predictions_calibrated_step1.csv
    data/processed/tft_predictions_calibrated_step5.csv

Outputs:
    visualizations/fig_vsn_stability.png
    visualizations/fig_ablation_mae.png
    visualizations/fig_calibration_coverage.png
    visualizations/explainability_stats.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
VIZ = ROOT / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 1000
TEST_START = "2023-01-01"

report_lines = []
def log(s=""):
    print(s)
    report_lines.append(s)

log("=" * 72)
log("EXPLAINABILITY & NOVELTY - STATISTICAL VALIDATION")
log("=" * 72)

# -----------------------------------------------------------------------
# A. VSN stability via bootstrap
# -----------------------------------------------------------------------
log("")
log("[A] VSN STABILITY via series-level bootstrap (N={} resamples)".format(N_BOOT))
log("-" * 72)

vars_detail = pd.read_csv(PROC / "tft_variable_importance_detail.csv")
enc = vars_detail[vars_detail["type"] == "encoder"].copy()

pivot = enc.pivot_table(
    index="series_id",
    columns="variable",
    values="importance",
    aggfunc="mean",
).dropna(axis=0, how="any")

features = list(pivot.columns)
matrix = pivot.values  # (n_series, n_features)
n_series = matrix.shape[0]

boot_means = np.zeros((N_BOOT, len(features)))
for b in range(N_BOOT):
    idx = RNG.integers(0, n_series, n_series)
    boot_means[b] = matrix[idx].mean(axis=0)

mean_imp = boot_means.mean(axis=0)
ci_lo = np.quantile(boot_means, 0.025, axis=0)
ci_hi = np.quantile(boot_means, 0.975, axis=0)

# Pairwise Kendall tau of rankings across bootstraps (sample 100 pairs)
n_pairs = 100
pair_ids = RNG.integers(0, N_BOOT, size=(n_pairs, 2))
taus = []
for a, b in pair_ids:
    if a == b:
        continue
    t, _ = stats.kendalltau(boot_means[a], boot_means[b])
    taus.append(t)
mean_tau = float(np.mean(taus))
ci_tau_lo = float(np.quantile(taus, 0.025))
ci_tau_hi = float(np.quantile(taus, 0.975))

order = np.argsort(-mean_imp)
log("  Mean pairwise Kendall tau across bootstrap rankings: "
    "{:.3f} (95% CI [{:.3f}, {:.3f}])".format(mean_tau, ci_tau_lo, ci_tau_hi))
log("  Interpretation: tau close to 1 -> feature rankings are stable")
log("")
log("  Top-10 encoder features (mean +/- 95% CI):")
log("  {:<22s} {:>10s}   {}".format("feature", "mean", "95% CI"))
for i in order[:10]:
    log("  {:<22s} {:>10.4f}   [{:.4f}, {:.4f}]".format(
        features[i], mean_imp[i], ci_lo[i], ci_hi[i]))

# Plot
fig, ax = plt.subplots(figsize=(9, 6))
top = order[:12]
ax.barh(
    [features[i] for i in top][::-1],
    mean_imp[top][::-1],
    xerr=np.vstack([
        (mean_imp[top] - ci_lo[top])[::-1],
        (ci_hi[top] - mean_imp[top])[::-1],
    ]),
    color="#2b7a78",
    ecolor="#333",
    capsize=3,
)
ax.set_xlabel("VSN importance (bootstrap mean, 95% CI)")
ax.set_title(
    "VSN encoder-feature importance stability\n"
    f"N={N_BOOT} bootstraps, mean Kendall tau = {mean_tau:.3f}"
)
plt.tight_layout()
plt.savefig(VIZ / "fig_vsn_stability.png", dpi=160)
plt.close(fig)
log("  Saved: visualizations/fig_vsn_stability.png")

# -----------------------------------------------------------------------
# B. Ablation: paired per-row absolute error across families
# -----------------------------------------------------------------------
log("")
log("[B] ABLATION: paired t-test across checkpoint families (test split)")
log("-" * 72)

def load_test(path):
    d = pd.read_csv(path, parse_dates=["date"])
    d = d[d["date"] >= TEST_START].copy()
    d["abs_err"] = (d["price"] - d["tft_q50_cal"]).abs()
    return d[["series_id", "date", "commodity", "market", "price",
              "tft_q10_cal", "tft_q90_cal", "tft_q50_cal", "abs_err"]]

orig = load_test(PROC / "tft_predictions_calibrated_original.csv")
step1 = load_test(PROC / "tft_predictions_calibrated_step1.csv")
step5 = load_test(PROC / "tft_predictions_calibrated_step5.csv")

key = ["series_id", "date"]
merged = (
    step5.rename(columns={"abs_err": "err_step5", "tft_q50_cal": "q50_step5"})
    .merge(orig.rename(columns={"abs_err": "err_orig", "tft_q50_cal": "q50_orig"})[key + ["err_orig", "q50_orig"]], on=key)
    .merge(step1.rename(columns={"abs_err": "err_step1", "tft_q50_cal": "q50_step1"})[key + ["err_step1", "q50_step1"]], on=key)
)
log("  Paired rows across all three families: n = {}".format(len(merged)))

def paired_test(a, b, label_a, label_b):
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    # Wilcoxon signed-rank (non-parametric backup)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    log("  {:<18s} vs {:<18s}  mean dMAE={:+.3f} Rs/kg  "
        "t={:.2f} p={:.3e}  Wilcoxon p={:.3e}  Cohen's d={:.2f}".format(
            label_a, label_b, diff.mean(), t, p, w_p, d))
    return {"t": t, "p": p, "wilcoxon_p": w_p, "d": d, "mean_diff": diff.mean()}

results = {}
results["step5_vs_orig"] = paired_test(merged["err_step5"].values,
                                       merged["err_orig"].values,
                                       "step5", "original")
results["step5_vs_step1"] = paired_test(merged["err_step5"].values,
                                        merged["err_step1"].values,
                                        "step5", "step1")
results["step1_vs_orig"] = paired_test(merged["err_step1"].values,
                                       merged["err_orig"].values,
                                       "step1", "original")

log("")
log("  Per-commodity MAE by family:")
header = "  {:<10s} {:>10s} {:>10s} {:>10s}".format(
    "commodity", "original", "step1", "step5")
log(header)
mae_by_commodity = {}
for c in sorted(merged["commodity"].unique()):
    sub = merged[merged["commodity"] == c]
    m_o = sub["err_orig"].mean()
    m_1 = sub["err_step1"].mean()
    m_5 = sub["err_step5"].mean()
    mae_by_commodity[c] = (m_o, m_1, m_5)
    log("  {:<10s} {:>10.3f} {:>10.3f} {:>10.3f}".format(c, m_o, m_1, m_5))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
commodities = list(mae_by_commodity.keys())
x = np.arange(len(commodities))
w = 0.26
ax.bar(x - w, [mae_by_commodity[c][0] for c in commodities], w, label="original", color="#b33")
ax.bar(x,     [mae_by_commodity[c][1] for c in commodities], w, label="step1 (2022 retrain)", color="#d90")
ax.bar(x + w, [mae_by_commodity[c][2] for c in commodities], w, label="step5 (XGBoost-fused)", color="#2b7a78")
ax.set_xticks(x)
ax.set_xticklabels(commodities)
ax.set_ylabel("MAE (Rs/kg)")
ax.set_title(
    "Per-commodity MAE across ablation families (test 2023+)\n"
    f"step5 vs original: t={results['step5_vs_orig']['t']:.2f}, "
    f"p={results['step5_vs_orig']['p']:.1e}, d={results['step5_vs_orig']['d']:.2f}"
)
ax.legend()
plt.tight_layout()
plt.savefig(VIZ / "fig_ablation_mae.png", dpi=160)
plt.close(fig)
log("  Saved: visualizations/fig_ablation_mae.png")

# -----------------------------------------------------------------------
# C. CQR coverage: per-commodity binomial test vs nominal 80%
# -----------------------------------------------------------------------
log("")
log("[C] CQR CALIBRATION: per-commodity binomial test on step5 (nominal 80%)")
log("-" * 72)

step5_full = pd.read_csv(
    PROC / "tft_predictions_calibrated_step5.csv", parse_dates=["date"])
step5_test = step5_full[step5_full["date"] >= TEST_START].copy()
step5_test["covered"] = (
    (step5_test["price"] >= step5_test["tft_q10_cal"]) &
    (step5_test["price"] <= step5_test["tft_q90_cal"])
).astype(int)

NOMINAL = 0.80
log("  {:<10s} {:>5s} {:>8s} {:>10s} {:>10s}".format(
    "commodity", "n", "hits", "coverage", "p (binom)"))
cov_rows = []
for c in sorted(step5_test["commodity"].unique()):
    sub = step5_test[step5_test["commodity"] == c]
    n = len(sub)
    k = int(sub["covered"].sum())
    emp = k / n
    # Two-sided binomial test: is empirical coverage consistent with nominal?
    res = stats.binomtest(k, n, p=NOMINAL, alternative="two-sided")
    p_val = res.pvalue
    cov_rows.append((c, n, k, emp, p_val))
    flag = "" if p_val > 0.05 else "  <-- reject H0=nominal"
    log("  {:<10s} {:>5d} {:>8d} {:>9.1%} {:>10.3f}{}".format(
        c, n, k, emp, p_val, flag))

# Overall
n_all = len(step5_test)
k_all = int(step5_test["covered"].sum())
emp_all = k_all / n_all
p_all = stats.binomtest(k_all, n_all, p=NOMINAL, alternative="two-sided").pvalue
log("  {:<10s} {:>5d} {:>8d} {:>9.1%} {:>10.3f}".format(
    "ALL", n_all, k_all, emp_all, p_all))
log("  H0: empirical coverage == 80% nominal. p > 0.05 -> cannot reject H0")
log("       (i.e., the CQR band is statistically calibrated for that commodity).")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [r[0] for r in cov_rows] + ["ALL"]
values = [r[3] for r in cov_rows] + [emp_all]
pvals = [r[4] for r in cov_rows] + [p_all]
colors = ["#2b7a78" if p > 0.05 else "#b33" for p in pvals]
bars = ax.bar(labels, values, color=colors, edgecolor="#222")
ax.axhline(NOMINAL, color="black", linestyle="--", label=f"nominal {NOMINAL:.0%}")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Empirical coverage of [q10, q90] band")
ax.set_title(
    "CQR calibration per commodity (step5, test 2023+)\n"
    "green = consistent with nominal (p>0.05); red = significant deviation"
)
for bar, v, p in zip(bars, values, pvals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.1%}\np={p:.2f}", ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(VIZ / "fig_calibration_coverage.png", dpi=160)
plt.close(fig)
log("  Saved: visualizations/fig_calibration_coverage.png")

# -----------------------------------------------------------------------
# Write report
# -----------------------------------------------------------------------
log("")
log("=" * 72)
log("Done. See visualizations/ for 3 figures + explainability_stats.txt")
log("=" * 72)

(VIZ / "explainability_stats.txt").write_text("\n".join(report_lines), encoding="utf-8")
