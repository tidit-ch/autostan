# AutoStan Report: synthetic_hierarchical_small

## 1. Best Model

**Iteration:** 0 (baseline)
**NLPD:** 1.4999
**Key design choices:**
- Standard hierarchical normal model with partial pooling
- Global mean `mu` and between-group SD `tau` governing group-level means
- Single within-group SD `sigma` shared across all groups
- Group means `alpha[j] ~ normal(mu, tau)` (centered parameterization)
- Observation model: `effect ~ normal(alpha[unit], sigma)`
- Weakly informative priors: `mu ~ normal(0, 5)`, `tau ~ normal(0, 2)`, `sigma ~ normal(0, 2)`

## 2. Trajectory

The dataset consists of 160 training observations from 20 groups (8 obs/group) with a held-out test set of 40 observations (2/group). Visual inspection of the data showed strong group-level structure: groups ranged from very negative means (group 16: ~-1.9) to very positive means (group 5: ~1.8), with within-group spreads that varied.

**Iteration 0 (baseline):** A standard centered hierarchical normal model achieved NLPD = 1.4999. The model correctly identified group-level variation through partial pooling, giving a strong starting point.

**Iteration 1 (non-centered):** Switching to a non-centered parameterization (z_alpha ~ std_normal, alpha = mu + tau * z_alpha) did not improve sampling — NLPD = 1.5014. This is typical when tau is reasonably large relative to the observation noise; centered parameterizations can actually perform better in that regime.

**Iteration 2 (Student-t likelihood):** Replacing the normal likelihood with a Student-t (nu estimated from data) aimed to handle potential heavy tails or outliers. NLPD = 1.5035, slightly worse. The data appears to be generated from a normal distribution, making the extra flexibility of t-tails unnecessary and adding estimation uncertainty.

**Iteration 3 (heteroscedastic):** Allowing group-specific within-group variances (sigma_j ~ log-normal hyperprior) was tried to capture the apparent variation in within-group spread. NLPD = 1.5019, still worse. With only 8 training observations per group, there is insufficient data to reliably estimate group-specific variances; the added complexity hurt generalization.

**Stopping rule met:** 3 consecutive non-improving iterations (iterations 1, 2, 3 all worse than iteration 0).

## 3. NLPD Table

| Iteration | NLPD   | Improved | Description |
|-----------|--------|----------|-------------|
| 0         | 1.4999 | —        | Baseline hierarchical normal model, centered parameterization |
| 1         | 1.5014 | No       | Non-centered parameterization (z-scores for group effects) |
| 2         | 1.5035 | No       | Student-t likelihood with estimated degrees of freedom |
| 3         | 1.5019 | No       | Heteroscedastic: group-specific within-group variances |

## 4. Key Insights

1. **Partial pooling is effective:** The baseline hierarchical model with partial pooling immediately captured the strong group structure. The NLPD of 1.4999 reflects well-calibrated predictions leveraging shrinkage.

2. **Centered vs. non-centered parameterization:** The centered parameterization outperformed non-centered here. This is consistent with theory: when the group-level effects are well-identified (large tau relative to noise, reasonable amount of data per group), the centered parameterization can be more efficient.

3. **Normal likelihood is sufficient:** The data appears to be generated from a normal distribution — t-distribution tails added noise to the estimation without improving predictions.

4. **Variance homogeneity:** Despite visual differences in within-group spread, the shared within-group sigma model generalized better than group-specific sigmas. With only 8 training observations per group, estimating 20 separate sigmas overfits.

5. **Small dataset regime:** With 8 training and 2 test observations per group, model complexity must be carefully controlled. The simplest well-specified model won, confirming Occam's Razor in the small-data regime.
