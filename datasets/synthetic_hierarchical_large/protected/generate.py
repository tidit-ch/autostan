#!/usr/bin/env python3
"""Generate synthetic hierarchical dataset (large).

Same generative process as synthetic_hierarchical_small but with more
observations per group, making priors nearly irrelevant.

Generative process:
    mu_j ~ Normal(mu0, tau)    for j = 1..J
    y_ij ~ Normal(mu_j, sigma) for i = 1..N_per_group

True parameters (hidden from agent):
    mu0 = 0.0
    tau = 1.0
    sigma = 1.0
    J = 20
    N_per_group = 50

80/20 train/test split at observation level (stratified by group).
"""

import csv
from pathlib import Path

import numpy as np

SEED = 20260327
J = 20
N_PER_GROUP = 50
MU0 = 0.0
TAU = 1.0
SIGMA = 1.0
TRAIN_FRAC = 0.8

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

rng = np.random.default_rng(SEED)

# Generate group means
mu = rng.normal(MU0, TAU, size=J)

# Generate observations
rows = []
for j in range(J):
    for i in range(N_PER_GROUP):
        rows.append({
            "unit_id": j + 1,
            "effect": round(rng.normal(mu[j], SIGMA), 4),
        })

# Stratified 80/20 split: within each group, 40 train / 10 test
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
        writer = csv.DictWriter(f, fieldnames=["unit_id", "effect"])
        writer.writeheader()
        writer.writerows(data)


write_csv(DATASET_DIR / "train.csv", train_rows)
write_csv(SCRIPT_DIR / "test.csv", test_rows)

print(f"Generated {len(train_rows)} train, {len(test_rows)} test observations.")
print(f"Groups: {J}, Obs per group: {N_PER_GROUP}")
print(f"True parameters: mu0={MU0}, tau={TAU}, sigma={SIGMA}")
print(f"Group means: {np.round(mu, 3).tolist()}")
