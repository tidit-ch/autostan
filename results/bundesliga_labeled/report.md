# AutoStan Report: bundesliga_labeled

## 1. Best Model

**Iteration:** 9
**NLPD:** 1.5432
**Key design choices:**
- Hierarchical Poisson with NCP for attack and defense
- **Team-specific home advantage** `delta[j] = delta_mu + sigma_delta * delta_raw[j]`
- Weakly informative priors on global intercept and home advantage based on training data means
- Tight sigma priors (`N+(0, 0.5)`) for regularization

```
lambda_home = exp(mu + delta[home] + attack[home] - defense[away])
lambda_away = exp(mu + attack[away] - defense[home])
```

## 2. Trajectory

### Phase 1: Establishing the hierarchy (iters 0–3)
The baseline (iter 0, NLPD=1.5663) used independent `N(0,1)` priors. Adding **hierarchical priors** (iter 1) gave the largest single improvement (+0.02 NLPD), shrinking 18 teams toward a global mean with only ~11 games per team. NCP (iter 3, +0.0002) improved geometry slightly. Negative binomial likelihood (iter 2) and Dixon-Coles correction (iter 4) were both tried and failed — goals are not significantly overdispersed beyond what team heterogeneity explains, and the Dixon-Coles low-score correction (15 observed 0-0 matches vs ~8.7 expected) does not persist in the test set.

### Phase 2: Refining priors (iters 5–8)
Bivariate attack/defense correlation (iter 5, -0.0005), asymmetric NB for home goals only (iter 7, -0.0071), and a single Bradley-Terry quality parameter (iter 8, -0.0213) all failed. The last result confirmed that separate attack and defense parameters capture genuinely different dimensions — teams that attack well do not necessarily defend well (or poorly) in a predictable way with only ~11 observations per team. Informative priors on `mu` and `home_adv` from training means gave a tiny gain (iter 6, +0.0003).

### Phase 3: Team-specific home advantage (iter 9)
Adding a hierarchical team-specific home advantage gave the second largest improvement (+0.003 over iter 6). Bundesliga teams differ substantially in how much they benefit from home support — some stadiums are famously intimidating. With ~8-9 home games per team in training, the hierarchical prior (sigma_delta ~ N+(0,0.3)) effectively pools information while allowing meaningful variation.

### Phase 4: Extending home advantage (iters 10–12)
Further attempts to extend the home advantage (symmetric for defense in iter 10, separate defense in iter 11, zero-inflation in iter 12) all failed or tied. The offensive home advantage captures the main signal; defensive home advantages add noise, not signal.

## 3. NLPD Table

| Iter | NLPD   | Notes |
|------|--------|-------|
| 0    | 1.5663 | Baseline: Poisson, independent N(0,1) priors |
| 1    | 1.5465 | Hierarchical priors, sigma~N+(0,0.5) |
| 2    | 1.5557 | NB likelihood (both), phi~N(0,5) |
| 3    | 1.5463 | NCP parameterization, sigma~N+(0,1) |
| 4    | 1.5472 | Dixon-Coles correction (rho~beta(2,10)) |
| 5    | 1.5468 | Bivariate attack/defense (corr_ad~N(-0.3,0.4)) |
| 6    | 1.5460 | NCP + sigma~N+(0,0.5) + informative mu/home_adv priors |
| 7    | 1.5544 | Asymmetric: NB home + Poisson away |
| 8    | 1.5645 | Single quality per team (Bradley-Terry) |
| **9** | **1.5432** | **Team-specific home advantage (hierarchical NCP)** |
| 10   | 1.5443 | Symmetric home advantage (same delta for offense + defense) |
| 11   | 1.5432 | Separate delta_att and delta_def (tied best) |
| 12   | 1.5460 | ZIP for both goals + team home advantage |

## 4. Key Insights

1. **Hierarchy is essential with few observations per team.** With only ~11 games per team, independent priors overfit badly. Hierarchical priors improved NLPD by ~0.02, the largest single gain.

2. **Teams have genuinely different home advantage profiles.** The second largest gain (+0.003) came from allowing team-specific home advantages. This reflects real variation in Bundesliga: some clubs have more intimidating home atmospheres than others.

3. **Home advantage is mainly offensive, not defensive.** Adding the same delta to `lambda_away` (defensive home advantage) consistently hurt or tied. The crowd effect boosts the home team's attack; the away team's scoring rate is dominated by the home team's defense quality, not the crowd.

4. **The goal distribution is essentially Poisson given team effects.** Home goals show var/mean = 1.26 and excess zeros (51 vs ~36 expected), but NB, ZIP, and Dixon-Coles corrections all failed. The overdispersion is mostly explained by between-team attack/defense variation already captured by the hierarchical model.

5. **Separate attack and defense are necessary.** The single-quality (Bradley-Terry) model gave NLPD=1.5645 — by far the worst. Teams that score many goals do not reliably concede fewer, so collapsing to one dimension is a serious misspecification.

6. **NCP only marginally improves geometry here.** With ~207 training observations, the data is informative enough that centered vs non-centered parameterization makes little difference (1.5465 vs 1.5463). The main benefit came from the hierarchical structure, not the sampling geometry.
