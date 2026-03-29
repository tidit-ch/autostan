# AutoStan Report: regression_1d

## Best Model

**Iteration 5** — NLPD **1.1244**

Key design choices:
- **Gaussian contamination mixture**: `y ~ (1-π)*Normal(μ(x), σ(x)) + π*Normal(μ(x), σ_out)`
- **Cubic polynomial mean**: `μ(x) = α + β₁x + β₂x² + β₃x³`
- **Log-linear heteroscedastic noise**: `σ(x) = exp(ls₀ + ls₁·x)`
- **Priors**: `π_out ~ Beta(1, 10)` (prior mean ~9%), `σ_out ~ HalfNormal(0, 20)`

## Trajectory

**Iter 0 (baseline, NLPD 2.2482):** Simple linear regression with Normal errors. Poor fit: the 4 extreme outliers (|y| > 11) dominate and inflate σ, degrading predictions everywhere.

**Iter 1 (NLPD 1.5023, +0.75 improvement):** Added Student-t likelihood and quadratic mean. Enormous gain — the Student-t's heavy tails downweight the extreme outliers, and the quadratic captures the peaked mean shape.

**Iter 2 (NLPD 1.1558, +0.35 improvement):** Extended to cubic polynomial and added log-linear heteroscedastic σ(x). The cubic captures additional curvature; heteroscedastic noise better reflects the dataset's explicit noise structure.

**Iter 3 (NLPD 1.2247, worse):** Replaced polynomial mean with Fourier basis (sin/cos at 1x and 2x). Worse than cubic — the polynomial is more appropriate for the actual mean shape despite the data looking sinusoidal.

**Iter 4 (NLPD 1.2319, worse):** Tried quartic polynomial + quadratic log-σ without mixture. Quadratic heteroscedasticity caused numerical instability (R-hat 1.02); quartic alone didn't help.

**Iter 5 (NLPD 1.1244, +0.03 improvement):** Replaced Student-t with an explicit Gaussian contamination mixture. The contamination model explicitly identifies the ~6% outlier fraction, giving a tighter Normal distribution for clean observations — slightly better than Student-t's implicit heavy-tail handling.

**Iter 6 (NLPD 1.1520, worse):** Added Student-t for the clean component + sin(x) to the mean. Too many degrees of freedom; sin(x) and cubic are somewhat collinear.

**Iter 7 (NLPD 1.2131, R-hat 1.74, worse):** Removed Student-t but kept sin(x) with the mixture. R-hat failure confirms sin(x) is collinear with the cubic polynomial, causing multimodality.

**Iter 8 (NLPD 1.2592, R-hat 1.53, worse):** Quartic polynomial + mixture + centering. Still R-hat issues — quartic with a mixture is over-parameterized for N=68 training points.

**Stop:** 3 consecutive non-improving iterations after iter 5.

## NLPD Table

| Iter | NLPD   | Improved | Notes |
|------|--------|----------|-------|
| 0    | 2.2482 | —        | Linear regression, Normal errors (baseline) |
| 1    | 1.5023 | ✓        | Student-t + quadratic mean |
| 2    | 1.1558 | ✓        | Student-t + cubic + log-linear heteroscedastic σ |
| 3    | 1.2247 | ✗        | Student-t + Fourier basis (sin/cos 1x,2x) |
| 4    | 1.2319 | ✗        | Student-t + quartic + quadratic log-σ |
| 5    | 1.1244 | ✓        | **Contamination mixture + cubic + log-linear σ** |
| 6    | 1.1520 | ✗        | Mixture + Student-t clean + cubic+sin(x) |
| 7    | 1.2131 | ✗        | Mixture + Normal clean + cubic+sin(x) (R-hat 1.74) |
| 8    | 1.2592 | ✗        | Mixture + quartic centered (R-hat 1.53) |

## Key Insights

1. **Outlier handling is critical.** Switching from Normal to Student-t (iter 0→1) gave the largest single NLPD reduction (0.75). Four extreme outliers (|y| > 11) can completely destabilize a homoscedastic Normal model.

2. **Contamination mixture > Student-t.** Switching from Student-t to an explicit Gaussian contamination mixture (iter 2→5, same polynomial structure) improved NLPD from 1.1558 → 1.1244. The mixture explicitly separates clean and outlier processes, giving tighter predictive distributions for non-outlier test points.

3. **Cubic polynomial suffices for the mean.** The data's nonlinear mean is well captured by a cubic. Fourier basis and quartic extensions consistently failed — suggesting the mean is a smooth unimodal-ish curve without oscillatory structure, well suited to a low-degree polynomial over [0, 5].

4. **Log-linear heteroscedasticity works; quadratic does not.** A single linear trend `log σ(x) = ls₀ + ls₁·x` was sufficient. Quadratic log-σ introduced identification problems in Stan's sampler.

5. **Adding sin(x) to a cubic causes collinearity.** Both capture oscillatory variation; combining them leads to multimodality (R-hat up to 1.74), suggesting the sampler cannot distinguish the two components.

6. **Model complexity vs. N.** With only 68 training observations and a contamination mixture already having 8 parameters, adding a 9th (beta4, beta_sin, or nu) consistently overfits or causes sampling problems. The best model is parsimonious.
