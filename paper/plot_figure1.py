#!/usr/bin/env python3
"""Generate the combined Figure 1 for the paper.

2×2 panel:
  (a) NLPD trajectory with markers linking to panels (b), (c), (d)
  (b) Iter 0: Linear + Gaussian baseline
  (c) Iter 1: Quadratic + Student-t
  (d) Iter 5: Cubic + heteroscedastic + contamination mixture (best)
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from cmdstanpy import CmdStanModel

REPO = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO / "datasets" / "regression_1d" / "train.csv"
TEST_CSV = REPO / "datasets" / "regression_1d" / "protected" / "test.csv"
MODELS_DIR = REPO / "models" / "regression_1d"
LOG_FILE = REPO / "results" / "regression_1d" / "log.jsonl"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

PANEL_COLORS = {0: "#D32F2F", 1: "#F57C00", 5: "#2E7D32"}
PANEL_LABELS = {0: "(b)", 1: "(c)", 5: "(d)"}


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
    return (*read_csv(TRAIN_CSV), *read_csv(TEST_CSV))


def load_log():
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def predict_on_grid(stan_file, x_grid, x_train, y_train):
    tmp = REPO / f"_tmp_fig1_{stan_file.stem}.stan"
    shutil.copy(stan_file, tmp)
    model = CmdStanModel(stan_file=str(tmp))
    y_dummy = np.zeros_like(x_grid)
    fit = model.sample(
        data={
            "N_train": len(x_train), "N_test": len(x_grid),
            "predictor_train": x_train.tolist(),
            "predictor_test": x_grid.tolist(),
            "response_train": y_train.tolist(),
            "response_test": y_dummy.tolist(),
        },
        chains=4, iter_sampling=2000, iter_warmup=1000, seed=42,
        show_console=False,
    )
    tmp.unlink(missing_ok=True)
    for p in REPO.glob(f"_tmp_fig1_{stan_file.stem}*"):
        p.unlink(missing_ok=True)
    return fit


def compute_mu_bands(fit, stan_file, x_grid):
    draws = {v: fit.stan_variable(v) for v in fit.stan_variables()}
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
            mu_all[i] = (draws["alpha"][i] + draws["beta1"][i] * x_grid
                         + draws["beta2"][i] * x_grid**2)
            sigma_all[i] = draws["sigma"][i]
    elif stem == "best":
        for i in range(n_draws):
            mu_all[i] = (draws["alpha"][i] + draws["beta1"][i] * x_grid
                         + draws["beta2"][i] * x_grid**2
                         + draws["beta3"][i] * x_grid**3)
            sigma_all[i] = np.exp(draws["log_sigma0"][i]
                                  + draws["log_sigma1"][i] * x_grid)

    return mu_all.mean(axis=0), sigma_all.mean(axis=0)


def plot_trajectory(ax, log_entries):
    iters = [e["iter"] for e in log_entries]
    nlpds = [e["nlpd"] for e in log_entries]
    improved = [e["improved"] for e in log_entries]

    ax.plot(iters, nlpds, "-", color="#4A90D9", linewidth=1.5, zorder=3, alpha=0.5)

    for it, nl, imp in zip(iters, nlpds, improved):
        if it in PANEL_COLORS:
            ax.plot(it, nl, "o", color=PANEL_COLORS[it], markersize=11,
                    zorder=7, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(PANEL_LABELS[it], xy=(it, nl),
                        xytext=(it + 0.35, nl + (0.06 if it != 0 else -0.06)),
                        fontsize=9, fontweight="bold", color=PANEL_COLORS[it],
                        ha="left", va="bottom" if it != 0 else "top",
                        arrowprops=dict(arrowstyle="->", color=PANEL_COLORS[it],
                                        lw=1.3))
        elif imp is True:
            ax.plot(it, nl, "o", color="#2E7D32", markersize=7, zorder=6)
        elif imp is False:
            ax.plot(it, nl, "o", color="#C62828", markersize=5, zorder=6, alpha=0.35)
        else:
            ax.plot(it, nl, "o", color="#4A90D9", markersize=7, zorder=6)

    ax.axhline(y=0.9443, color="gray", linestyle="--", linewidth=0.9,
               label="Oracle (0.94)")
    ax.axhline(y=1.1184, color="#E65100", linestyle=":", linewidth=0.9,
               label="TabPFN (1.12)")

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("NLPD", fontsize=10)
    ax.set_xticks(iters)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(0.85, 2.35)
    ax.set_title("(a) NLPD trajectory", fontsize=10, fontweight="bold")


def plot_model_fit(ax, stan_file, x_grid, x_train, y_train, x_test, y_test,
                   title, panel_color):
    print(f"  Fitting {stan_file.stem}...")
    fit = predict_on_grid(stan_file, x_grid, x_train, y_train)
    mu_mean, sigma_mean = compute_mu_bands(fit, stan_file, x_grid)

    ax.plot(x_grid, f_true(x_grid), "k-", linewidth=1, alpha=0.35,
            label="True $f(x)$")
    ax.fill_between(x_grid,
                    f_true(x_grid) - 2 * sigma_true(x_grid),
                    f_true(x_grid) + 2 * sigma_true(x_grid),
                    alpha=0.06, color="gray")

    ax.plot(x_grid, mu_mean, color="C3", linewidth=1.8,
            label="Model $\\hat{f}(x)$")
    ax.fill_between(x_grid, mu_mean - 2 * sigma_mean,
                    mu_mean + 2 * sigma_mean,
                    alpha=0.18, color="C3", label="$\\pm 2\\hat{\\sigma}(x)$")

    ax.scatter(x_train, y_train, s=12, c="C0", alpha=0.5, zorder=4)
    ax.scatter(x_test, y_test, s=16, c="C1", marker="x", alpha=0.6, zorder=4)

    ax.set_ylim(-6, 8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=panel_color)


def main():
    x_train, y_train, x_test, y_test = load_data()
    log_entries = load_log()
    x_grid = np.linspace(-0.2, 5.2, 200)

    fig = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.25)

    ax_traj = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Trajectory
    plot_trajectory(ax_traj, log_entries)

    models = [
        (ax_b, "baseline",
         "(b) Iter 0: Linear + Gaussian — NLPD 2.25",
         PANEL_COLORS[0]),
        (ax_c, "iter_001",
         "(c) Iter 1: Quadratic + Student-$t$ — NLPD 1.50",
         PANEL_COLORS[1]),
        (ax_d, "best",
         "(d) Iter 5: Mixture + heteroscedastic — NLPD 1.12",
         PANEL_COLORS[5]),
    ]

    for ax, model_name, title, color in models:
        stan_file = MODELS_DIR / f"{model_name}.stan"
        plot_model_fit(ax, stan_file, x_grid, x_train, y_train,
                       x_test, y_test, title, color)

    # Share axes for model-fit panels
    for ax in [ax_c, ax_d]:
        ax.set_xlabel("predictor", fontsize=9)
    for ax in [ax_b, ax_c]:
        ax.set_ylabel("response", fontsize=9)
    ax_d.set_ylabel("")

    # Legend only on first model panel
    ax_b.legend(fontsize=6.5, loc="upper right")

    fname = FIG_DIR / "figure1_combined.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
