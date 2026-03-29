#!/usr/bin/env python3
"""
TabPFN baseline for the 1D Regression (heteroscedastic with outliers) datasets.

Computes NLPD on the held-out test set using a CDF finite-difference density
estimate from TabPFN's FullSupportBarDistribution. This gives a continuous
density estimate that is directly comparable to AutoStan's Gaussian/mixture NLPD.

TabPFN version: 2.2.1  (Prior-Labs, FullSupportBarDistribution)
"""

import csv
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor

REPO = Path(__file__).resolve().parent.parent


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r["predictor"]) for r in rows], dtype=np.float32)
    y = np.array([float(r["response"]) for r in rows], dtype=np.float32)
    return x, y


def compute_nlpd(regressor, X_test, y_test, delta=0.02):
    """
    NLPD via CDF finite differences:  density(y) ≈ [CDF(y+δ) - CDF(y-δ)] / 2δ

    Using the diagonal of crit.cdf(logits, y) to pair each test point with
    its own predictive CDF row.
    """
    out = regressor.predict(X_test, output_type="full")
    crit  = out["criterion"]
    logits = out["logits"]          # [N_test, num_bars]
    device = logits.device

    y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    cdf_hi = crit.cdf(logits, y_t + delta).diag()
    cdf_lo = crit.cdf(logits, y_t - delta).diag()
    density   = ((cdf_hi - cdf_lo) / (2 * delta)).clamp(min=1e-12)
    log_lik   = torch.log(density)
    per_obs   = -log_lik.cpu().numpy()
    nlpd      = per_obs.mean()
    return float(nlpd), per_obs


def run_dataset(name, n_estimators=8):
    base    = REPO / "datasets" / name
    x_train, y_train = load_csv(base / "train.csv")
    x_test,  y_test  = load_csv(base / "protected" / "test.csv")

    X_train = x_train.reshape(-1, 1)
    X_test  = x_test.reshape(-1, 1)

    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"  Train : {len(x_train)}   Test : {len(x_test)}")

    reg = TabPFNRegressor(n_estimators=n_estimators,
                          ignore_pretraining_limits=True)
    reg.fit(X_train, y_train)

    nlpd, per_obs = compute_nlpd(reg, X_test, y_test)
    mean_pred = reg.predict(X_test, output_type="mean")
    rmse = float(np.sqrt(np.mean((mean_pred - y_test) ** 2)))

    print(f"  NLPD  : {nlpd:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    return nlpd, per_obs, mean_pred, x_train, y_train, x_test, y_test, reg


def make_plots(tag, x_tr, y_tr, x_te, y_te, reg, nlpd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(exist_ok=True)

    x_grid = np.linspace(float(x_tr.min()) - 0.2,
                         float(x_tr.max()) + 0.2, 200).astype(np.float32)
    X_grid = x_grid.reshape(-1, 1)

    mu    = reg.predict(X_grid, output_type="mean")
    q_low = np.array(reg.predict(X_grid, output_type="quantiles",
                                 quantiles=[0.05, 0.95]))[0]
    q_hig = np.array(reg.predict(X_grid, output_type="quantiles",
                                 quantiles=[0.05, 0.95]))[1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_tr, y_tr, s=6,  c="C0", alpha=0.4, zorder=4, label="Train")
    ax.scatter(x_te, y_te, s=14, c="C1", marker="x",
               alpha=0.7, zorder=5, label="Test")
    ax.plot(x_grid, mu, "C3", lw=2, label="TabPFN mean")
    ax.fill_between(x_grid, q_low, q_hig, alpha=0.2, color="C3",
                    label="90% prediction interval")
    ax.set_xlabel("predictor")
    ax.set_ylabel("response")
    ax.set_title(f"TabPFN — 1D Regression ({tag}, NLPD = {nlpd:.4f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = fig_dir / f"tabpfn_regression_1d_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


def main():
    results = {}

    # ── Small dataset ──────────────────────────────────────────────────────
    nlpd_s, per_s, pred_s, x_tr_s, y_tr_s, x_te_s, y_te_s, reg_s = \
        run_dataset("regression_1d")
    results["small"] = dict(tabpfn=nlpd_s, autostan_best=1.1244, oracle=0.9443)

    # ── Large dataset ──────────────────────────────────────────────────────
    nlpd_l, per_l, pred_l, x_tr_l, y_tr_l, x_te_l, y_te_l, reg_l = \
        run_dataset("regression_1d_large")
    results["large"] = dict(tabpfn=nlpd_l, autostan_best=1.2256, oracle=1.1442)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("NLPD Comparison (lower = better)")
    print(f"{'':22s} {'Small (n=68)':>14s} {'Large (n=500)':>14s}")
    print("-" * 52)
    for label, key in [("Oracle (true model)", "oracle"),
                       ("AutoStan best",        "autostan_best"),
                       ("TabPFN 2.2",           "tabpfn")]:
        s, l = results["small"][key], results["large"][key]
        print(f"{label:22s} {s:>14.4f} {l:>14.4f}")
    print("=" * 60)

    # ── Plots ──────────────────────────────────────────────────────────────
    for tag, x_tr, y_tr, x_te, y_te, reg, nlpd in [
        ("small", x_tr_s, y_tr_s, x_te_s, y_te_s, reg_s, nlpd_s),
        ("large", x_tr_l, y_tr_l, x_te_l, y_te_l, reg_l, nlpd_l),
    ]:
        make_plots(tag, x_tr, y_tr, x_te, y_te, reg, nlpd)


if __name__ == "__main__":
    main()
