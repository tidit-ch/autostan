#!/usr/bin/env python3
"""Generate heteroscedastic 1D regression dataset with outliers.

Generative process:
    f(x) = 2 * sin(1.2 * x) + 0.3 * x
    sigma(x) = 0.3 + 0.8 * exp(-0.5 * ((x - 3) / 1.5)^2)
    y ~ Normal(f(x), sigma(x))

    Plus contamination: 4 training points shifted by +/- Uniform(10, 15).

True parameters (hidden from agent):
    Mean function: smooth nonlinear (sinusoidal + linear trend)
    Noise: heteroscedastic, tight at edges, wide in middle
    Outliers: 4 extreme values in training only

x density: dense in [0, 3] (60 points), sparse in [3, 5] (8 points)
Test set: same x density (25 dense + 5 sparse), no outlier contamination.

The agent must discover:
1. Nonlinear mean function (not just a line)
2. Heteroscedastic noise (variance depends on x)
3. Robust likelihood (outliers break Gaussian)
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED = 20260327

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent


def f(x):
    return 2 * np.sin(1.2 * x) + 0.3 * x


def sigma(x):
    return 0.3 + 0.8 * np.exp(-0.5 * ((x - 3) / 1.5) ** 2)


def main():
    rng = np.random.default_rng(SEED)

    # --- Training data ---
    x_dense = rng.uniform(0, 3, size=60)
    x_sparse = rng.uniform(3, 5, size=8)
    x_train = np.concatenate([x_dense, x_sparse])
    y_train = f(x_train) + sigma(x_train) * rng.standard_normal(len(x_train))

    # Add outliers: 4 points shifted by +/- 10-15 units
    n_outliers = 4
    outlier_idx = rng.choice(len(x_train), n_outliers, replace=False)
    outlier_signs = rng.choice([-1, 1], n_outliers)
    outlier_shifts = rng.uniform(10, 15, n_outliers)
    y_train[outlier_idx] += outlier_signs * outlier_shifts

    # Round for CSV
    x_train = np.round(x_train, 4)
    y_train = np.round(y_train, 4)

    # --- Test data: same x density, no outliers ---
    x_test_dense = rng.uniform(0, 3, size=25)
    x_test_sparse = rng.uniform(3, 5, size=5)
    x_test = np.concatenate([x_test_dense, x_test_sparse])
    y_test = f(x_test) + sigma(x_test) * rng.standard_normal(len(x_test))

    x_test = np.round(x_test, 4)
    y_test = np.round(y_test, 4)

    # --- Write CSVs ---
    def write_csv(path, x, y):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["predictor", "response"])
            for xi, yi in zip(x, y):
                writer.writerow([xi, yi])

    write_csv(DATASET_DIR / "train.csv", x_train, y_train)
    write_csv(SCRIPT_DIR / "test.csv", x_test, y_test)

    print(f"Generated {len(x_train)} train + {len(x_test)} test observations")
    print(f"Outliers: {n_outliers} (indices {outlier_idx.tolist()})")
    print(f"Outlier x values: {x_train[outlier_idx].tolist()}")
    print(f"Outlier y values: {y_train[outlier_idx].tolist()}")

    # --- Oracle NLPD ---
    # The oracle knows the true f(x) and sigma(x), ignoring outliers.
    # Test set was generated without outliers, so oracle evaluates cleanly.
    log_preds = []
    for xi, yi in zip(x_test, y_test):
        log_preds.append(norm.logpdf(yi, f(xi), sigma(xi)))
    oracle_nlpd = -np.mean(log_preds)
    print(f"\nOracle NLPD (true f and sigma known): {oracle_nlpd:.4f}")

    # Save oracle and outlier info
    import json
    with open(SCRIPT_DIR / "ground_truth.json", "w") as fh:
        json.dump({
            "oracle_nlpd": round(float(oracle_nlpd), 4),
            "outlier_indices": outlier_idx.tolist(),
            "mean_function": "2 * sin(1.2 * x) + 0.3 * x",
            "sigma_function": "0.3 + 0.8 * exp(-0.5 * ((x - 3) / 1.5)^2)",
            "n_train": len(x_train),
            "n_test": len(x_test),
        }, fh, indent=2)


if __name__ == "__main__":
    main()
