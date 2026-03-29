# AutoStan Report: synthetic_regression

## Best Model

**Iteration 10** — NLPD **1.2738**

Key design choices:
- **Piecewise linear** with a knot fixed at x=0: each unit has separate slopes for negative and positive predictor values (`alpha_j + beta_neg_j * min(x,0) + beta_pos_j * max(x,0)`)
- **3D LKJ(2) correlated prior** on `(alpha_j, beta_neg_j, beta_pos_j)` via non-centered Cholesky parameterization, capturing cross-unit correlation between intercept and the two slopes
- **Unit-specific residual sigma** via hierarchical log-normal (`sigma_j = exp(mu_log_sigma + sigma_log_sigma * z_j)`)
- Normal likelihood

Saved at: `models/synthetic_regression/iter_010.stan`

---

## Trajectory

**Baseline (iter 0)**: Complete pooling (single intercept + slope for all units). NLPD = 1.8178. This served only to establish a floor.

**Iter 1 — Big leap**: Added hierarchical varying intercepts and slopes with NCP. NLPD dropped to 1.3091 — the single largest gain of the run. Each unit clearly has a different slope (unit 14 spans response ~−3 to +8; unit 15 has a positive slope; unit 7 is near zero).

**Iters 2–4 — Diminishing returns on the linear model**: Adding LKJ correlation between slopes and intercepts (iter 2: 1.3073), Student-t likelihood (iter 3: tied at 1.3073, confirming Gaussian residuals), and unit-specific sigma (iter 4: 1.3055) all gave marginal gains. The hierarchical linear model appeared to have plateaued.

**Iter 5 — Second breakthrough**: Added a hierarchical quadratic term `gamma_j * x²` correlated with `(alpha_j, beta_j)` via 3D LKJ. NLPD jumped to 1.2910. The data has nonlinear structure beyond simple varying slopes.

**Iters 6–7 — Refinement of 3D structure**: Removing the LKJ (iter 6: 1.2933) hurt — the correlation between parameters matters. Extending to 3D LKJ including the quadratic term (iter 7: 1.2839) improved over iter 5.

**Iter 8–9 — Prior tuning failure**: Tighter half-normal priors on tau (iter 8: 1.2861) and LKJ(4) instead of LKJ(2) (iter 9: 1.2890) both hurt. The original Exp(1) priors and LKJ(2) were better calibrated.

**Iter 10 — Third breakthrough**: Replaced polynomial with **piecewise linear** (knot at x=0). NLPD dropped to 1.2738. The data appears to have different linear slopes for negative vs. positive predictor values, which a polynomial cannot capture efficiently.

**Iters 11–13 — Complexity overload**: 3-segment piecewise (4D LKJ, iter 11: 1.2833, 15 divergences), predictor-dependent sigma (iter 12: 1.2949), and learned knot location (iter 13: 1.2937, 282 divergences, R-hat=1.63) all failed. The 2-segment model at x=0 was optimal — adding more parameters caused overfitting and sampling problems.

**Stopped**: 3 consecutive non-improving iterations (11, 12, 13).

---

## NLPD Table

| Iter | NLPD   | Model Description |
|------|--------|-------------------|
| 0    | 1.8178 | Baseline: complete pooling |
| 1    | 1.3091 | Hierarchical varying intercepts+slopes, NCP |
| 2    | 1.3073 | + LKJ correlation (alpha, beta) |
| 3    | 1.3073 | + Student-t likelihood (tied, nu→∞ confirming Gaussian) |
| 4    | 1.3055 | + Unit-specific sigma (lognormal hierarchy) |
| 5    | 1.2910 | + Hierarchical quadratic term gamma_j (3D LKJ) |
| 6    | 1.2933 | Independent slopes (no LKJ) — worse, LKJ matters |
| 7    | 1.2839 | 3D LKJ including gamma in correlation structure |
| 8    | 1.2861 | Tighter HN(0.5) priors on tau[3] — worse |
| 9    | 1.2890 | LKJ(4) instead of LKJ(2) — worse |
| **10** | **1.2738** | **Piecewise linear at x=0: (alpha, beta_neg, beta_pos) 3D LKJ + unit sigma** |
| 11   | 1.2833 | 3-segment piecewise (knots at ±1), 4D LKJ — 15 divergences |
| 12   | 1.2949 | + Predictor-dependent sigma sigma_j*exp(xi_j*x²) |
| 13   | 1.2937 | Piecewise with learned shared knot k — 282 divergences |

---

## Key Insights

1. **Varying slopes dominate**: The biggest single gain (0.51 NLPD) came from allowing unit-specific slopes. The dataset genuinely has 15 different linear relationships.

2. **Nonlinearity is present but asymmetric**: A polynomial quadratic (iter 5) helped, but piecewise linear (iter 10) was better — the data has fundamentally different slopes for x<0 vs x>0 rather than smooth curvature.

3. **Correlation structure matters**: The 3D LKJ coupling (alpha, beta_neg, beta_pos) captured useful cross-unit shrinkage patterns. Removing the LKJ always hurt.

4. **Unit-specific sigma is useful**: Different units have different residual variances. The lognormal hierarchy shrinks these appropriately.

5. **Complexity has a sharp cliff**: Adding a 4th dimension to the LKJ (4D) or learning the knot position immediately caused divergences and worse NLPD. With only J=15 groups and 20 observations each, the model is near its information capacity.

6. **LKJ concentration**: LKJ(2) was consistently better than LKJ(4) — the data supports moderate correlations between hierarchical parameters.
