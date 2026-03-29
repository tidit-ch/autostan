# TabPFN Baseline — 1D Regression (Heteroscedastic with Outliers)

Branch: `analysis/tabpfn-1d-regression`
Script: `analysis/tabpfn_1d_regression.py`
TabPFN version: 2.2.1 (Prior-Labs, `FullSupportBarDistribution`)

---

## Setup

The 1D regression datasets have a single continuous predictor and a continuous response with three independent sources of difficulty:

1. **Nonlinear mean**: $f(x) = 2\sin(1.2x) + 0.3x$ (arch-shaped, peak near $x\approx1.5$)
2. **Heteroscedastic noise**: $\sigma(x) = 0.3 + 0.8\exp(-0.5((x-3)/1.5)^2)$ — variance is locally elevated around $x\approx3$ and low elsewhere
3. **Extreme outliers**: ~6% of training observations shifted $\pm10$–15 units

Two dataset sizes:
- **Small**: 68 train / 30 test
- **Large**: 500 train / 200 test

TabPFN is fit with `n_estimators=8` and `ignore_pretraining_limits=True` (needed because the dataset is 1D, below TabPFN's default column-count guard).

---

## NLPD Computation

TabPFN's `FullSupportBarDistribution` represents a piecewise-constant density over 5000 non-uniformly spaced buckets. To compute NLPD in a way that is directly comparable to AutoStan's continuous Gaussian/mixture NLPD, we use a **CDF finite-difference** estimate:

$$\hat{p}(y_n \mid x_n) \approx \frac{\mathrm{CDF}(y_n + \delta) - \mathrm{CDF}(y_n - \delta)}{2\delta}, \quad \delta = 0.02$$

This gives a proper continuous density estimate, and NLPD = $-\frac{1}{N}\sum_n \log \hat{p}(y_n \mid x_n)$.

**Why not use `pdf()` directly?** The `BarDistribution.pdf()` method returns `exp(forward())`, where `forward()` is the bucket-level log-probability — not the continuous density. Integrating `pdf()` over a wide range gives infinity, confirming it is not a normalized density. The CDF approach avoids this issue.

---

## Results

| | Small (n=68) | Large (n=500) |
|---|---|---|
| Oracle (true model known) | 0.9443 | 1.1442 |
| AutoStan best | 1.1244 | 1.2256 |
| **TabPFN 2.2** | **1.1184** | **1.2501** |

RMSE (point prediction accuracy):

| | Small | Large |
|---|---|---|
| TabPFN mean | 0.807 | 0.880 |

---

## Findings

### Small dataset (n=68): TabPFN matches AutoStan

TabPFN achieves NLPD 1.1184, marginally *better* than AutoStan's 1.1244. With only 68 observations, the limited data constrains both methods similarly. TabPFN implicitly learns the rough shape of the mean function and places appropriate uncertainty, without explicitly representing the nonlinearity, heteroscedasticity, or outlier mechanism.

This is a notable result: AutoStan's 9-iteration search (including explicit contamination mixture and cubic log-sigma) achieves the same NLPD as TabPFN's zero-shot inference. On small data, the structural modeling effort of AutoStan does not pay off over a flexible non-parametric baseline.

### Large dataset (n=500): AutoStan pulls ahead

With 500 training observations, AutoStan (1.2256) clearly outperforms TabPFN (1.2501, gap: 0.024). The larger dataset gives AutoStan's explicit structural discoveries real traction:

- **Sinusoidal mean**: AutoStan discovered $\sin(\omega x)/\cos(\omega x)$ with learned frequency $\hat\omega \approx 1.2$ (true: 1.2). TabPFN approximates the arch shape but cannot represent the exact frequency.
- **Heteroscedastic noise**: AutoStan explicitly models $\log\sigma(x)$ as a cubic polynomial. TabPFN's intervals widen over the high-variance region ($x\approx3$) but less precisely.
- **Outlier mixture**: AutoStan uses an explicit two-component Gaussian mixture ($\hat\pi\approx6\%$, fixed $\sigma_\text{out}=10$). TabPFN treats the outliers as signal and broadens the predictive distribution globally, which hurts density at clean test points.

The gap to oracle also narrows more for AutoStan (0.081) than it would with TabPFN (0.108), reflecting AutoStan's ability to recover the generative structure.

### Takeaway for the paper

TabPFN serves as a strong zero-shot baseline, confirming that AutoStan's discoveries are not trivially available. The cross-over point appears somewhere between n=68 and n=500: structural modeling via NLPD optimization adds value once there is sufficient data to identify the model components. This matches the paper's broader finding that improvement quality scales with sample size.

---

## Figures

- `paper/figures/tabpfn_regression_1d_small.png` — predictions on the small dataset
- `paper/figures/tabpfn_regression_1d_large.png` — predictions on the large dataset
