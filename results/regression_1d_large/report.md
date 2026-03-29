# AutoStan Report: regression_1d_large

## 1. Best Model

**Iteration 11** — NLPD = **1.2256**

Key design choices:
- **Mean function**: `alpha + a1*sin(omega*x) + b1*cos(omega*x) + beta_lin*x` with learned frequency `omega` (prior centered at π/3)
- **Heteroscedastic noise**: `sigma(x) = exp(s0 + s1*x + s2*x² + s3*x³)` — cubic log-sigma polynomial
- **Outlier handling**: Two-component Gaussian mixture — inlier `Normal(mu, sigma(x))` + outlier `Normal(mu, 10)` with fixed `sigma_out=10` and estimated `pi_out ~ Beta(1,9)`
- **No divergences**, R-hat = 1.0002

## 2. Trajectory

**Baseline → Outlier model** (iters 0–8): The biggest single gain was replacing a Normal likelihood with Student-t (+0.84 NLPD improvement from iter 0 to iter 1). Recognizing that the data followed an arch-shaped nonlinear mean (peaks at x≈1.5, returns near zero at x≈3), a sine basis (`sin(πx/3)`, `cos(πx/3)`) was introduced in iter 2 and slightly outperformed the cubic polynomial.

**Heteroscedasticity** (iters 4–5): The dataset description explicitly mentions heteroscedastic noise. Adding a linear then quadratic log-sigma model improved NLPD from 1.3060 → 1.2952 → 1.2854. A 2nd Fourier harmonic and quadratic polynomial mean correction both failed to improve.

**Learnable frequency** (iter 7): Freeing omega gave a small improvement (1.2854 → 1.2844), suggesting the true frequency is close but not exactly π/3.

**Mixture model breakthrough** (iters 8–9): Replacing Student-t with an explicit two-component Gaussian mixture (inlier + wide outlier) gave the largest single improvement after iter 1. The first attempt (estimated sigma_out) suffered from R-hat=1.73 due to label-switching. Fixing sigma_out=10 resolved convergence and gave NLPD=1.2325.

**Cubic log-sigma** (iter 11): Extending from quadratic to cubic log-sigma gave another improvement (1.2325 → 1.2256).

**Plateau** (iters 12–14): Sine-based log-sigma, spatially-varying pi_out, and Student-t inlier all failed to improve. Stopped at 3 consecutive non-improving iterations.

## 3. NLPD Table

| Iter | NLPD   | Description |
|------|--------|-------------|
| 0    | 2.1589 | Baseline: linear mean, Normal likelihood |
| 1    | 1.3181 | Student-t + cubic polynomial mean |
| 2    | 1.3060 | Student-t + sine basis (1 harmonic) + linear |
| 3    | 1.3088 | Student-t + 2-harmonic Fourier (worse) |
| 4    | 1.2952 | Student-t + sine + linear log-sigma |
| 5    | 1.2854 | Student-t + sine + quadratic log-sigma |
| 6    | 1.2859 | Student-t + sine+quadratic mean + quadratic log-sigma (worse) |
| 7    | 1.2844 | Student-t + learnable omega + quadratic log-sigma |
| 8    | 1.2635 | Gaussian mixture + learnable omega + quadratic log-sigma (R-hat=1.73) |
| 9    | 1.2325 | Gaussian mixture (sigma_out=10 fixed) + learnable omega + quadratic log-sigma |
| 10   | 1.3529 | Gaussian mixture (sigma_out estimated) — R-hat=1.52, divergences (worse) |
| **11** | **1.2256** | **Gaussian mixture (sigma_out=10) + learnable omega + cubic log-sigma** ← BEST |
| 12   | 1.2262 | Same but sine-based log-sigma (worse) |
| 13   | 1.2258 | Same + spatially-varying pi_out (worse) |
| 14   | 1.2291 | Student-t inlier + Normal outlier mixture (worse) |

## 4. Key Insights

1. **Outlier mechanism matters more than tail shape**: The data contains bimodal noise — most observations have small residuals (σ≈1–2) while ~10% are extreme outliers (±10–17). An explicit two-component mixture with fixed outlier scale substantially outperformed Student-t, which applies heavy tails uniformly.

2. **Fixing sigma_out prevents label-switching**: Estimating sigma_out led to R-hat > 1.5 and degraded NLPD. Setting sigma_out=10 (informed by data inspection) gave stable inference. For mixture models with heavy-tailed components, fixing the outlier scale is essential.

3. **Sine basis > polynomial for this data**: The arch-shaped mean (peak at x≈1.5, returns to zero at x≈3) is naturally parameterized by sin(ωx)/cos(ωx). This outperformed a cubic polynomial and needed only one harmonic.

4. **Heteroscedastic variance follows a polynomial structure**: The variance was better described by a cubic polynomial in x than by a sine-matched basis. The two x-clusters ([0,3] and [3,5]) have different variance profiles that require more than linear or sine-based parameterization.

5. **Learnable frequency adds modest value**: The true oscillation frequency is close to π/3 (half-period ≈ 3), but freeing ω gave a small gain. The prior `ω ~ Normal(π/3, 0.3)` with constraint [0.3, 3.0] prevented divergence.
