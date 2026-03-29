#!/usr/bin/env python3
"""Generate synthetic varying-slopes regression dataset.

Generative process:
    x_ij ~ Uniform(-2, 2)
    alpha_j ~ Normal(mu_alpha, tau_alpha)    for j = 1..J
    beta_j  ~ Normal(mu_beta, tau_beta)      for j = 1..J
    y_ij ~ Normal(alpha_j + beta_j * x_ij, sigma)

True parameters (hidden from agent):
    mu_alpha = 2.0
    tau_alpha = 1.0
    mu_beta = -0.5
    tau_beta = 0.7
    sigma = 0.8
    J = 15
    N_per_group = 25

80/20 train/test split at observation level (stratified by group).

The agent must discover:
1. That the predictor has an effect on the response
2. That the effect (slope) varies by group
3. Optionally, correlation between intercepts and slopes
"""

import csv
from pathlib import Path

import numpy as np

SEED = 20260328
J = 15
N_PER_GROUP = 25
MU_ALPHA = 2.0
TAU_ALPHA = 1.0
MU_BETA = -0.5
TAU_BETA = 0.7
SIGMA = 0.8
TRAIN_FRAC = 0.8

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

rng = np.random.default_rng(SEED)

# Generate group-level parameters
alpha = rng.normal(MU_ALPHA, TAU_ALPHA, size=J)
beta = rng.normal(MU_BETA, TAU_BETA, size=J)

# Generate observations
rows = []
for j in range(J):
    for i in range(N_PER_GROUP):
        x = round(rng.uniform(-2, 2), 4)
        mu = alpha[j] + beta[j] * x
        y = round(rng.normal(mu, SIGMA), 4)
        rows.append({
            "unit_id": j + 1,
            "predictor": x,
            "response": y,
        })

# Stratified 80/20 split: within each group, 20 train / 5 test
train_rows = []
test_rows = []
for j in range(J):
    group_rows = [r for r in rows if r["unit_id"] == j + 1]
    rng.shuffle(group_rows)
    n_train = int(N_PER_GROUP * TRAIN_FRAC)
    train_rows.extend(group_rows[:n_train])
    test_rows.extend(group_rows[n_train:])


def write_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


write_csv(DATASET_DIR / "train.csv", train_rows)
write_csv(SCRIPT_DIR / "test.csv", test_rows)

print(f"Generated {len(train_rows)} train + {len(test_rows)} test observations")
print(f"Groups: {J}, Observations per group: {N_PER_GROUP}")
print(f"True alpha: {alpha.round(3)}")
print(f"True beta: {beta.round(3)}")
print(f"True sigma: {SIGMA}")


# Compute oracle NLPD
# For each test point (x*, j), the oracle posterior predictive is:
#   y* | x*, y_train_j ~ N(alpha_j_post + beta_j_post * x*, var_pred)
# With known hyperparameters, the posterior for (alpha_j, beta_j) given
# group j's training data is a bivariate normal. The predictive variance
# includes both posterior uncertainty on (alpha_j, beta_j) and sigma^2.

from scipy.stats import norm

train_by_group = {}
for r in train_rows:
    g = r["unit_id"]
    if g not in train_by_group:
        train_by_group[g] = {"x": [], "y": []}
    train_by_group[g]["x"].append(r["predictor"])
    train_by_group[g]["y"].append(r["response"])

log_preds = []
for r in test_rows:
    j = r["unit_id"]
    x_star = r["predictor"]
    y_star = r["response"]

    x_train = np.array(train_by_group[j]["x"])
    y_train = np.array(train_by_group[j]["y"])
    n_j = len(x_train)

    # Prior precision matrix for (alpha_j, beta_j)
    prior_prec = np.diag([1.0 / TAU_ALPHA**2, 1.0 / TAU_BETA**2])
    prior_mean = np.array([MU_ALPHA, MU_BETA])

    # Design matrix for group j training data
    X_j = np.column_stack([np.ones(n_j), x_train])

    # Likelihood precision
    lik_prec = (X_j.T @ X_j) / SIGMA**2

    # Posterior precision and mean
    post_prec = prior_prec + lik_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (prior_prec @ prior_mean + X_j.T @ y_train / SIGMA**2)

    # Predictive for y* at x*
    x_star_vec = np.array([1.0, x_star])
    pred_mean = x_star_vec @ post_mean
    pred_var = x_star_vec @ post_cov @ x_star_vec + SIGMA**2

    log_preds.append(norm.logpdf(y_star, pred_mean, np.sqrt(pred_var)))

oracle_nlpd = -np.mean(log_preds)
print(f"\nOracle NLPD (true hyperparameters known): {oracle_nlpd:.4f}")
