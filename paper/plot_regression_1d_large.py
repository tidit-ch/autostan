#!/usr/bin/env python3
"""Generate plots for the 1D regression (large) experiment.

1. Learning curve (NLPD vs iteration)
2. Multi-panel: ground truth + data vs model fits for key iterations

Auto-detects improving iterations from log.jsonl. Extracts mean/sigma
from posterior draws by detecting parameter names in each Stan model.
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel

REPO = Path(__file__).resolve().parent.parent
DATASET = "regression_1d_large"
TRAIN_CSV = REPO / "datasets" / DATASET / "train.csv"
TEST_CSV = REPO / "datasets" / DATASET / "protected" / "test.csv"
MODELS_DIR = REPO / "models" / DATASET
LOG_FILE = REPO / "results" / DATASET / "log.jsonl"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

ORACLE_NLPD = 1.1442


def f_true(x):
    return 2 * np.sin(1.2 * x) + 0.3 * x


def sigma_true(x):
    return 0.3 + 0.8 * np.exp(-0.5 * ((x - 3) / 1.5) ** 2)


def load_data():
    def read_csv(path):
        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        return (np.array([float(r["predictor"]) for r in rows]),
                np.array([float(r["response"]) for r in rows]))
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


def fit_on_grid(stan_file, x_train, y_train, x_grid):
    """Fit model with x_grid as test points."""
    tmp = REPO / f"_tmp_lg_{stan_file.stem}.stan"
    shutil.copy(stan_file, tmp)
    model = CmdStanModel(stan_file=str(tmp))
    fit = model.sample(
        data={
            "N_train": len(x_train),
            "N_test": len(x_grid),
            "predictor_train": x_train.tolist(),
            "predictor_test": x_grid.tolist(),
            "response_train": y_train.tolist(),
            "response_test": np.zeros_like(x_grid).tolist(),
        },
        chains=4, iter_sampling=2000, iter_warmup=1000, seed=42,
        show_console=False,
    )
    tmp.unlink(missing_ok=True)
    for p in REPO.glob(f"_tmp_lg_{stan_file.stem}*"):
        p.unlink(missing_ok=True)
    return fit


def extract_mu_sigma(fit, x_grid):
    """Extract posterior mean and sigma from draws by detecting parameter names."""
    draws = {v: fit.stan_variable(v) for v in fit.stan_variables()}
    n_draws = list(draws.values())[0].shape[0]
    n_grid = len(x_grid)

    mu_all = np.zeros((n_draws, n_grid))
    sigma_all = np.zeros((n_draws, n_grid))

    for i in range(n_draws):
        mu = np.zeros(n_grid)
        if "alpha" in draws:
            mu += draws["alpha"][i]
        if "beta" in draws and draws["beta"].ndim == 1:
            mu += draws["beta"][i] * x_grid
        if "beta_lin" in draws:
            mu += draws["beta_lin"][i] * x_grid
        if "beta1" in draws:
            mu += draws["beta1"][i] * x_grid
        if "beta2" in draws:
            mu += draws["beta2"][i] * x_grid**2
        if "beta3" in draws:
            mu += draws["beta3"][i] * x_grid**3
        if "beta4" in draws:
            mu += draws["beta4"][i] * x_grid**4
        if "beta_sin" in draws:
            omega_fixed = np.pi / 3.0
            mu += draws["beta_sin"][i] * np.sin(omega_fixed * x_grid)
        if "beta_cos" in draws:
            omega_fixed = np.pi / 3.0
            mu += draws["beta_cos"][i] * np.cos(omega_fixed * x_grid)
        if "a1" in draws and "b1" in draws:
            omega_i = draws["omega"][i] if "omega" in draws else np.pi / 3.0
            mu += draws["a1"][i] * np.sin(omega_i * x_grid) + draws["b1"][i] * np.cos(omega_i * x_grid)
        if "a2" in draws and "b2" in draws:
            omega_i = draws["omega"][i] if "omega" in draws else np.pi / 3.0
            mu += draws["a2"][i] * np.sin(2 * omega_i * x_grid) + draws["b2"][i] * np.cos(2 * omega_i * x_grid)
        mu_all[i] = mu

        if "s0" in draws and "s1" in draws:
            ls = draws["s0"][i] + draws["s1"][i] * x_grid
            if "s2" in draws:
                ls += draws["s2"][i] * x_grid**2
            if "s3" in draws:
                ls += draws["s3"][i] * x_grid**3
            sigma_all[i] = np.exp(ls)
        elif "log_sigma0" in draws and "log_sigma1" in draws:
            ls = draws["log_sigma0"][i] + draws["log_sigma1"][i] * x_grid
            if "log_sigma2" in draws:
                ls += draws["log_sigma2"][i] * x_grid**2
            sigma_all[i] = np.exp(ls)
        elif "sigma" in draws and draws["sigma"].ndim == 1:
            sigma_all[i] = draws["sigma"][i]
        else:
            sigma_all[i] = 1.0

    return mu_all.mean(axis=0), sigma_all.mean(axis=0)


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
    for it, nl, imp in zip(iters, nlpds, improved):
        if imp is True:
            ax.plot(it, nl, "o", color="green", markersize=9, zorder=6)
        elif imp is False:
            ax.plot(it, nl, "o", color="red", markersize=6, zorder=6, alpha=0.5)

    ax.axhline(y=ORACLE_NLPD, color="gray", linestyle="--", linewidth=1, label=f"Oracle ({ORACLE_NLPD})")
    best_nlpd = min(nlpds)
    ax.axhline(y=best_nlpd, color="green", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("NLPD", fontsize=11)
    ax.set_title("NLPD Learning Curve — 1D Regression (Large, n=500)", fontsize=12)
    ax.set_xticks(iters)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regression_1d_large_learning_curve.png", dpi=150)
    plt.close()
    print("Saved learning curve")

    # ===== Plot 2: Multi-panel model fits =====
    # Auto-select: baseline, first 2 improvements, best
    improving = [e for e in log_entries if e["improved"] is True]
    best_entry = min(log_entries, key=lambda e: e["nlpd"])

    panels = [log_entries[0]]  # baseline always first
    for e in improving[:2]:
        if e["iter"] != best_entry["iter"] and e not in panels:
            panels.append(e)
    if best_entry not in panels:
        panels.append(best_entry)
    # Fill to 4 panels if we have more improvements
    if len(panels) < 4:
        for e in improving:
            if e not in panels and len(panels) < 4:
                panels.append(e)
    panels = sorted(panels, key=lambda e: e["iter"])[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    show_idx = np.arange(len(x_train))  # show all training points

    for idx, entry in enumerate(panels):
        ax = axes_flat[idx]
        it = entry["iter"]
        nlpd_val = entry["nlpd"]
        notes = entry["notes"]
        short = notes[:45] + "..." if len(notes) > 45 else notes

        if it == 0:
            stan_file = MODELS_DIR / "baseline.stan"
        elif it == best_entry["iter"]:
            stan_file = MODELS_DIR / "best.stan"
        else:
            stan_file = MODELS_DIR / f"iter_{it:03d}.stan"

        if not stan_file.exists():
            print(f"Skipping iter {it}: not found")
            continue

        print(f"Fitting iter {it}...")
        fit = fit_on_grid(stan_file, x_train, y_train, x_grid)
        mu_mean, sigma_mean = extract_mu_sigma(fit, x_grid)

        # True function
        ax.plot(x_grid, f_true(x_grid), "k-", linewidth=1.2, alpha=0.4, label="True $f(x)$")
        ax.fill_between(x_grid, f_true(x_grid) - 2 * sigma_true(x_grid),
                        f_true(x_grid) + 2 * sigma_true(x_grid),
                        alpha=0.07, color="gray")

        # Model fit
        ax.plot(x_grid, mu_mean, color="C3", linewidth=2, label="Model $\\hat{f}(x)$")
        ax.fill_between(x_grid, mu_mean - 2 * sigma_mean, mu_mean + 2 * sigma_mean,
                        alpha=0.2, color="C3", label="$\\pm 2\\hat{\\sigma}(x)$")

        # Data (all points)
        ax.scatter(x_train, y_train, s=6, c="C0", alpha=0.3, zorder=4, label="Train")
        ax.scatter(x_test, y_test, s=10, c="C1", marker="x", alpha=0.5, zorder=4, label="Test")

        is_best = (it == best_entry["iter"])
        ax.set_title(f"Iter {it}{' (best)' if is_best else ''}: {short}\nNLPD = {nlpd_val}", fontsize=9)
        ax.set_ylim(-20, 22)  # show all points including outliers (±15)

        if idx >= 2:
            ax.set_xlabel("predictor", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("response", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for idx in range(len(panels), 4):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Model Evolution — 1D Regression (Large, n=500)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regression_1d_large_model_fits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model fits")


if __name__ == "__main__":
    main()
