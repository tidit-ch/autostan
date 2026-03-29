#!/usr/bin/env python3
"""Generate plots for the 1D regression experiment.

1. Learning curve (NLPD vs iteration)
2. Multi-panel: ground truth + data vs model fits for key iterations
   (baseline, iter 1, iter 2, best/iter 5)

Each model panel shows posterior predictive mean ± 2σ bands.
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel

REPO = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO / "datasets" / "regression_1d" / "train.csv"
TEST_CSV = REPO / "datasets" / "regression_1d" / "protected" / "test.csv"
MODELS_DIR = REPO / "models" / "regression_1d"
LOG_FILE = REPO / "results" / "regression_1d" / "log.jsonl"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# True generative process
def f_true(x):
    return 2 * np.sin(1.2 * x) + 0.3 * x

def sigma_true(x):
    return 0.3 + 0.8 * np.exp(-0.5 * ((x - 3) / 1.5) ** 2)


def load_data():
    def read_csv(path):
        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        x = np.array([float(r["predictor"]) for r in rows])
        y = np.array([float(r["response"]) for r in rows])
        return x, y

    x_train, y_train = read_csv(TRAIN_CSV)
    x_test, y_test = read_csv(TEST_CSV)
    return x_train, y_train, x_test, y_test


def load_log():
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def fit_model(stan_file, x_train, y_train, x_test, y_test):
    """Compile and sample a model, return the fit object."""
    # Copy to temp location to avoid clobbering
    tmp = REPO / f"_tmp_plot_{stan_file.stem}.stan"
    shutil.copy(stan_file, tmp)
    model = CmdStanModel(stan_file=str(tmp))
    fit = model.sample(
        data={
            "N_train": len(x_train),
            "N_test": len(x_test),
            "predictor_train": x_train.tolist(),
            "predictor_test": x_test.tolist(),
            "response_train": y_train.tolist(),
            "response_test": y_test.tolist(),
        },
        chains=4, iter_sampling=2000, iter_warmup=1000, seed=42,
        show_console=False,
    )
    tmp.unlink(missing_ok=True)
    for p in REPO.glob(f"_tmp_plot_{stan_file.stem}*"):
        p.unlink(missing_ok=True)
    return fit


def predict_on_grid(fit, stan_file, x_grid, x_train, y_train, x_test, y_test):
    """Get posterior predictive mean and bands on a grid.

    We re-fit with x_grid as 'test' points to get predictions there.
    """
    tmp = REPO / f"_tmp_grid_{stan_file.stem}.stan"
    shutil.copy(stan_file, tmp)
    model = CmdStanModel(stan_file=str(tmp))

    # Use x_grid as test points, with dummy y values
    y_grid_dummy = np.zeros_like(x_grid)

    fit_grid = model.sample(
        data={
            "N_train": len(x_train),
            "N_test": len(x_grid),
            "predictor_train": x_train.tolist(),
            "predictor_test": x_grid.tolist(),
            "response_train": y_train.tolist(),
            "response_test": y_grid_dummy.tolist(),
        },
        chains=4, iter_sampling=2000, iter_warmup=1000, seed=42,
        show_console=False,
    )
    tmp.unlink(missing_ok=True)
    for p in REPO.glob(f"_tmp_grid_{stan_file.stem}*"):
        p.unlink(missing_ok=True)

    return fit_grid


def extract_predictive(fit_grid, stan_file, x_grid):
    """Extract mean and sigma from posterior draws to compute predictive bands."""
    draws = {}
    for var in fit_grid.stan_variables():
        draws[var] = fit_grid.stan_variable(var)

    n_draws = list(draws.values())[0].shape[0] if draws else 0
    n_grid = len(x_grid)

    # We need to reconstruct mu and sigma from parameters
    # This depends on the model. Let's do it per model type.
    stem = stan_file.stem

    y_samples = np.zeros((n_draws, n_grid))

    if stem == "baseline":
        alpha = draws["alpha"]
        beta = draws["beta"]
        sigma = draws["sigma"]
        for i in range(n_draws):
            mu = alpha[i] + beta[i] * x_grid
            y_samples[i] = np.random.normal(mu, sigma[i])

    elif stem == "iter_001":
        alpha = draws["alpha"]
        beta1 = draws["beta1"]
        beta2 = draws["beta2"]
        sigma = draws["sigma"]
        nu = draws["nu"]
        for i in range(n_draws):
            mu = alpha[i] + beta1[i] * x_grid + beta2[i] * x_grid**2
            # Use normal approximation for predictive (t is close to normal for large nu)
            y_samples[i] = np.random.normal(mu, sigma[i])

    elif stem == "iter_002":
        alpha = draws["alpha"]
        beta1 = draws["beta1"]
        beta2 = draws["beta2"]
        beta3 = draws["beta3"]
        ls0 = draws["log_sigma0"]
        ls1 = draws["log_sigma1"]
        for i in range(n_draws):
            mu = alpha[i] + beta1[i] * x_grid + beta2[i] * x_grid**2 + beta3[i] * x_grid**3
            sig = np.exp(ls0[i] + ls1[i] * x_grid)
            y_samples[i] = np.random.normal(mu, sig)

    elif stem == "best":
        alpha = draws["alpha"]
        beta1 = draws["beta1"]
        beta2 = draws["beta2"]
        beta3 = draws["beta3"]
        ls0 = draws["log_sigma0"]
        ls1 = draws["log_sigma1"]
        pi_out = draws["pi_out"]
        sigma_out = draws["sigma_out"]
        for i in range(n_draws):
            mu = alpha[i] + beta1[i] * x_grid + beta2[i] * x_grid**2 + beta3[i] * x_grid**3
            sig = np.exp(ls0[i] + ls1[i] * x_grid)
            # Mixture: draw from clean or outlier component
            is_outlier = np.random.random(n_grid) < pi_out[i]
            y_clean = np.random.normal(mu, sig)
            y_out = np.random.normal(mu, sigma_out[i])
            y_samples[i] = np.where(is_outlier, y_out, y_clean)

    return y_samples


def compute_mu_bands(fit_grid, stan_file, x_grid):
    """Compute posterior mean function and ±2σ bands (without sampling noise)."""
    draws = {}
    for var in fit_grid.stan_variables():
        draws[var] = fit_grid.stan_variable(var)

    n_draws = list(draws.values())[0].shape[0]
    n_grid = len(x_grid)
    stem = stan_file.stem

    mu_all = np.zeros((n_draws, n_grid))
    sigma_all = np.zeros((n_draws, n_grid))

    if stem == "baseline":
        for i in range(n_draws):
            mu_all[i] = draws["alpha"][i] + draws["beta"][i] * x_grid
            sigma_all[i] = draws["sigma"][i]

    elif stem == "iter_001":
        for i in range(n_draws):
            mu_all[i] = draws["alpha"][i] + draws["beta1"][i] * x_grid + draws["beta2"][i] * x_grid**2
            sigma_all[i] = draws["sigma"][i]

    elif stem == "iter_002":
        for i in range(n_draws):
            mu_all[i] = (draws["alpha"][i] + draws["beta1"][i] * x_grid +
                         draws["beta2"][i] * x_grid**2 + draws["beta3"][i] * x_grid**3)
            sigma_all[i] = np.exp(draws["log_sigma0"][i] + draws["log_sigma1"][i] * x_grid)

    elif stem == "best":
        for i in range(n_draws):
            mu_all[i] = (draws["alpha"][i] + draws["beta1"][i] * x_grid +
                         draws["beta2"][i] * x_grid**2 + draws["beta3"][i] * x_grid**3)
            sigma_all[i] = np.exp(draws["log_sigma0"][i] + draws["log_sigma1"][i] * x_grid)

    mu_mean = mu_all.mean(axis=0)
    # ±2σ bands: mean of sigma across draws
    sigma_mean = sigma_all.mean(axis=0)

    return mu_mean, sigma_mean


def main():
    x_train, y_train, x_test, y_test = load_data()
    log_entries = load_log()
    x_grid = np.linspace(-0.2, 5.2, 200)

    # ===== Plot 1: Learning curve =====
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = [e["iter"] for e in log_entries]
    nlpds = [e["nlpd"] for e in log_entries]
    improved = [e["improved"] for e in log_entries]

    ax.plot(iters, nlpds, "o-", color="#4A90D9", markersize=6, linewidth=1.5, zorder=5)

    # Mark improvements
    for i, (it, nl, imp) in enumerate(zip(iters, nlpds, improved)):
        if imp is True:
            ax.plot(it, nl, "o", color="green", markersize=9, zorder=6)
        elif imp is False:
            ax.plot(it, nl, "o", color="red", markersize=6, zorder=6, alpha=0.5)

    # Oracle line
    ax.axhline(y=0.9443, color="gray", linestyle="--", linewidth=1, label="Oracle (0.9443)")

    # Best NLPD line
    best_nlpd = min(nlpds)
    ax.axhline(y=best_nlpd, color="green", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("NLPD", fontsize=11)
    ax.set_title("NLPD Learning Curve — 1D Regression", fontsize=12)
    ax.set_xticks(iters)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regression_1d_learning_curve.png", dpi=150)
    plt.close()
    print(f"Saved learning curve")

    # ===== Plot 2: Multi-panel model fits =====
    models_to_plot = [
        ("baseline", "Iter 0: Linear + Gaussian\nNLPD = 2.2482"),
        ("iter_001", "Iter 1: Quadratic + Student-t\nNLPD = 1.5023"),
        ("iter_002", "Iter 2: Cubic + Heteroscedastic + Student-t\nNLPD = 1.1558"),
        ("best", "Iter 5 (best): Cubic + Heteroscedastic + Mixture\nNLPD = 1.1244"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (model_name, title) in enumerate(models_to_plot):
        ax = axes[idx]
        stan_file = MODELS_DIR / f"{model_name}.stan"
        print(f"Fitting {model_name}...")

        fit_grid = predict_on_grid(None, stan_file, x_grid, x_train, y_train, x_test, y_test)
        mu_mean, sigma_mean = compute_mu_bands(fit_grid, stan_file, x_grid)

        # True function
        ax.plot(x_grid, f_true(x_grid), "k-", linewidth=1.2, alpha=0.4, label="True $f(x)$")
        ax.fill_between(x_grid, f_true(x_grid) - 2 * sigma_true(x_grid),
                        f_true(x_grid) + 2 * sigma_true(x_grid),
                        alpha=0.07, color="gray")

        # Model fit
        ax.plot(x_grid, mu_mean, color="C3", linewidth=2, label="Model $\\hat{f}(x)$")
        ax.fill_between(x_grid, mu_mean - 2 * sigma_mean, mu_mean + 2 * sigma_mean,
                        alpha=0.2, color="C3", label="$\\pm 2\\hat{\\sigma}(x)$")

        # Data
        ax.scatter(x_train, y_train, s=15, c="C0", alpha=0.6, zorder=4)
        ax.scatter(x_test, y_test, s=20, c="C1", marker="x", alpha=0.7, zorder=4)

        ax.set_title(title, fontsize=10)
        ax.set_ylim(-6, 8)  # Clip to exclude extreme outliers from dominating view

        if idx >= 2:
            ax.set_xlabel("predictor", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("response", fontsize=10)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Model Evolution — 1D Regression", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regression_1d_model_fits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model fits")


if __name__ == "__main__":
    main()
