#!/usr/bin/env python3
"""Generate Figure 1 (large dataset, n=500) for the AutoStan paper.

Layout  (2 rows × 6 columns via GridSpec):
  (a) NLPD trajectory  [0, 0:3]   (b) Iter 0: baseline          [0, 3:6]
  (c) Iter 1: Student-t [1, 0:2]   (d) Iter 11: AutoStan best    [1, 2:4]
  (e) TabPFN            [1, 4:6]

Panels (b–d) show posterior predictive mean ± 2σ(x).
Panel (e) shows TabPFN mean + 90 % interval.

Caching: Stan sampling & TabPFN results are cached to .npz so that
re-running only regenerates the figure.  Delete the cache to resample.
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASET = "regression_1d_large"
TRAIN_CSV = REPO / "datasets" / DATASET / "train.csv"
TEST_CSV  = REPO / "datasets" / DATASET / "protected" / "test.csv"
MODELS_DIR = REPO / "models" / DATASET
LOG_FILE  = REPO / "results" / DATASET / "log.jsonl"
FIG_DIR   = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)
CACHE_FILE = FIG_DIR / "figure1_large_cache.npz"

HIGHLIGHT = {0: "(b)", 1: "(c)", 11: "(d)"}
PANEL_COLORS = {0: "#D32F2F", 1: "#F57C00", 11: "#2E7D32"}
TABPFN_COLOR = "#1565C0"
TABPFN_NLPD = 1.2501
ORACLE_NLPD = 1.1442

# ── publication rcParams ──────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif":  ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def f_true(x):
    return 2 * np.sin(1.2 * x) + 0.3 * x

def sigma_true(x):
    return 0.3 + 0.8 * np.exp(-0.5 * ((x - 3) / 1.5) ** 2)


# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([float(r["predictor"]) for r in rows]),
            np.array([float(r["response"]) for r in rows]))


def load_log():
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


# ── Stan sampling ─────────────────────────────────────────────────────────────

def sample_on_grid(stan_file, x_grid, x_train, y_train):
    from cmdstanpy import CmdStanModel
    tmp = REPO / f"_fig1L_{stan_file.stem}.stan"
    shutil.copy(stan_file, tmp)
    model = CmdStanModel(stan_file=str(tmp))
    fit = model.sample(
        data={
            "N_train": len(x_train), "N_test": len(x_grid),
            "predictor_train": x_train.tolist(),
            "predictor_test":  x_grid.tolist(),
            "response_train":  y_train.tolist(),
            "response_test":   np.zeros_like(x_grid).tolist(),
        },
        chains=4, iter_warmup=500, iter_sampling=500, seed=42,
        show_console=False,
    )
    tmp.unlink(missing_ok=True)
    for p in REPO.glob(f"_fig1L_{stan_file.stem}*"):
        p.unlink(missing_ok=True)
    return fit


def mu_sigma(fit, stem, x_grid):
    d = {v: fit.stan_variable(v) for v in fit.stan_variables()}
    n, g = list(d.values())[0].shape[0], len(x_grid)
    mu_all  = np.zeros((n, g))
    sig_all = np.zeros((n, g))

    if stem == "baseline":
        for i in range(n):
            mu_all[i]  = d["alpha"][i] + d["beta"][i] * x_grid
            sig_all[i] = d["sigma"][i]
    elif stem == "iter_001":
        for i in range(n):
            mu_all[i] = (d["alpha"][i]
                         + d["beta1"][i] * x_grid
                         + d["beta2"][i] * x_grid**2
                         + d["beta3"][i] * x_grid**3)
            sig_all[i] = d["sigma"][i]
    else:  # best / iter_011
        for i in range(n):
            om = d["omega"][i]
            mu_all[i] = (d["alpha"][i]
                         + d["a1"][i] * np.sin(om * x_grid)
                         + d["b1"][i] * np.cos(om * x_grid)
                         + d["beta_lin"][i] * x_grid)
            sig_all[i] = np.exp(d["s0"][i]
                                + d["s1"][i] * x_grid
                                + d["s2"][i] * x_grid**2
                                + d["s3"][i] * x_grid**3)

    return mu_all.mean(0), sig_all.mean(0)


def tabpfn_predictions(x_train, y_train, x_grid):
    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=True)
    reg.fit(x_train.reshape(-1, 1), y_train)
    X_grid = x_grid.reshape(-1, 1).astype(np.float32)
    mu  = reg.predict(X_grid, output_type="mean")
    qs  = reg.predict(X_grid, output_type="quantiles", quantiles=[0.05, 0.95])
    q_lo = np.array(qs)[0]
    q_hi = np.array(qs)[1]
    return mu, q_lo, q_hi


# ── caching ───────────────────────────────────────────────────────────────────

def compute_or_load_cache(x_train, y_train):
    """Return dict with x_grid, mu/sig for baseline/iter1/best, TabPFN."""
    if CACHE_FILE.exists():
        print(f"Loading cached predictions from {CACHE_FILE.name}")
        return dict(np.load(CACHE_FILE))

    print("No cache found — running Stan + TabPFN (this takes ~90 s)…")
    x_grid = np.linspace(-0.2, 5.3, 300)

    print("  Fitting baseline…")
    fit_b = sample_on_grid(MODELS_DIR / "baseline.stan", x_grid, x_train, y_train)
    mu_b, sg_b = mu_sigma(fit_b, "baseline", x_grid)

    print("  Fitting iter_001…")
    fit_1 = sample_on_grid(MODELS_DIR / "iter_001.stan", x_grid, x_train, y_train)
    mu_1, sg_1 = mu_sigma(fit_1, "iter_001", x_grid)

    print("  Fitting best (iter_011)…")
    fit_d = sample_on_grid(MODELS_DIR / "best.stan", x_grid, x_train, y_train)
    mu_d, sg_d = mu_sigma(fit_d, "best", x_grid)

    print("  Running TabPFN…")
    mu_t, q_lo, q_hi = tabpfn_predictions(x_train, y_train, x_grid)

    cache = dict(x_grid=x_grid,
                 mu_b=mu_b, sg_b=sg_b,
                 mu_1=mu_1, sg_1=sg_1,
                 mu_d=mu_d, sg_d=sg_d,
                 mu_t=mu_t, q_lo=q_lo, q_hi=q_hi)
    np.savez_compressed(CACHE_FILE, **cache)
    print(f"  Cached to {CACHE_FILE.name}")
    return cache


# ── drawing helpers ───────────────────────────────────────────────────────────

YLIM = (-5, 7)

TRAIN_KW = dict(s=3, color="#5B8DB8", alpha=0.30, linewidths=0.2,
                edgecolors="white", zorder=4, rasterized=True)
TEST_KW  = dict(s=12, color="#E07B39", alpha=0.60, linewidths=0.5,
                zorder=5, marker="x", rasterized=True)


def scatter_clamped(ax, x, y, ylim, **kw):
    """Plot in-range points normally; outliers as edge-pinned triangles."""
    lo, hi = ylim
    x, y = np.asarray(x), np.asarray(y)
    mask_in = (y >= lo) & (y <= hi)
    mask_hi = y > hi
    mask_lo = y < lo

    if mask_in.any():
        ax.scatter(x[mask_in], y[mask_in], **kw)
    if mask_hi.any():
        ax.scatter(x[mask_hi], np.full(mask_hi.sum(), hi - 0.25),
                   **{**kw, "marker": "^"})
    if mask_lo.any():
        ax.scatter(x[mask_lo], np.full(mask_lo.sum(), lo + 0.25),
                   **{**kw, "marker": "v"})


MODEL_LINE_COLOR = "#B71C1C"
BAND_COLOR = "#E57373"
TRUE_COLOR = "#555555"


def draw_model_fit(ax, x_grid, mu, sig, x_train, y_train, x_test, y_test,
                   title, panel_color, show_legend=False):
    ax.fill_between(x_grid,
                    f_true(x_grid) - 2 * sigma_true(x_grid),
                    f_true(x_grid) + 2 * sigma_true(x_grid),
                    alpha=0.07, color=TRUE_COLOR, zorder=1)
    ax.plot(x_grid, f_true(x_grid), color=TRUE_COLOR, lw=0.8, alpha=0.5,
            zorder=2, label=r"True $f(x)$")

    ax.fill_between(x_grid, mu - 2 * sig, mu + 2 * sig,
                    alpha=0.20, color=BAND_COLOR, zorder=3,
                    label=r"$\pm\, 2\hat\sigma(x)$")
    ax.plot(x_grid, mu, color=MODEL_LINE_COLOR, lw=1.8, zorder=6,
            label=r"Model $\hat f(x)$")

    scatter_clamped(ax, x_train, y_train, ylim=YLIM, **TRAIN_KW)
    scatter_clamped(ax, x_test, y_test, ylim=YLIM, **TEST_KW)

    ax.set_ylim(*YLIM)
    ax.set_title(title, fontsize=9, fontweight="bold", color=panel_color)
    if show_legend:
        leg = ax.legend(fontsize=6, loc="upper right", ncol=1,
                        handlelength=1.5, handletextpad=0.4)
        leg.get_frame().set_linewidth(0.4)


def draw_tabpfn(ax, x_grid, mu, q_lo, q_hi,
                x_train, y_train, x_test, y_test):
    ax.fill_between(x_grid,
                    f_true(x_grid) - 2 * sigma_true(x_grid),
                    f_true(x_grid) + 2 * sigma_true(x_grid),
                    alpha=0.07, color=TRUE_COLOR, zorder=1)
    ax.plot(x_grid, f_true(x_grid), color=TRUE_COLOR, lw=0.8, alpha=0.5,
            zorder=2)

    ax.fill_between(x_grid, q_lo, q_hi, alpha=0.20, color=BAND_COLOR,
                    zorder=3, label=r"90\% interval")
    ax.plot(x_grid, mu, color=MODEL_LINE_COLOR, lw=1.8, zorder=6)

    scatter_clamped(ax, x_train, y_train, ylim=YLIM, **TRAIN_KW)
    scatter_clamped(ax, x_test, y_test, ylim=YLIM, **TEST_KW)

    ax.set_ylim(*YLIM)
    ax.set_title(r"\textbf{(e) TabPFN} --- NLPD %.4f" % TABPFN_NLPD,
                 fontsize=9, fontweight="bold", color=TABPFN_COLOR)


def draw_trajectory(ax, log_entries):
    iters    = [e["iter"]     for e in log_entries]
    nlpds    = [e["nlpd"]     for e in log_entries]
    improved = [e["improved"] for e in log_entries]

    ax.plot(iters, nlpds, "-", color="#4A90D9", lw=1.2, alpha=0.45, zorder=3)

    for it, nl, imp in zip(iters, nlpds, improved):
        if it in HIGHLIGHT:
            ax.plot(it, nl, "o", color=PANEL_COLORS[it], ms=10,
                    zorder=7, markeredgecolor="white", mew=1.3)
        elif imp is True:
            ax.plot(it, nl, "o", color="#2E7D32", ms=5, zorder=6)
        elif imp is False:
            ax.plot(it, nl, "o", color="#C62828", ms=4, zorder=6, alpha=0.30)
        else:
            ax.plot(it, nl, "o", color="#4A90D9", ms=5, zorder=6)

    offsets = {
        0:  dict(xytext=(0.6, 2.05), ha="left",  va="bottom"),
        1:  dict(xytext=(2.0, 1.38), ha="left",  va="bottom"),
        11: dict(xytext=(12.0, 1.18), ha="left",  va="top"),
    }
    for it, kw in offsets.items():
        nl = nlpds[iters.index(it)]
        ax.annotate(HIGHLIGHT[it],
                    xy=(it, nl), xytext=kw["xytext"],
                    fontsize=9, fontweight="bold", color=PANEL_COLORS[it],
                    ha=kw["ha"], va=kw["va"],
                    arrowprops=dict(arrowstyle="->", color=PANEL_COLORS[it],
                                    lw=1.0, shrinkB=3))

    ax.axhline(ORACLE_NLPD, color="0.45", ls="--", lw=0.7,
               label=r"Oracle (%.4f)" % ORACLE_NLPD)
    ax.axhline(TABPFN_NLPD, color=TABPFN_COLOR, ls=":", lw=0.7,
               label=r"TabPFN (%.4f)" % TABPFN_NLPD)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("NLPD")
    ax.set_xticks(iters)
    ax.legend(loc="upper right")
    ax.set_title(r"\textbf{(a)} NLPD trajectory ($n{=}500$)", fontsize=10)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    x_train, y_train = load_csv(TRAIN_CSV)
    x_test,  y_test  = load_csv(TEST_CSV)
    log_entries = load_log()

    C = compute_or_load_cache(x_train, y_train)
    x_grid = C["x_grid"]

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 6, hspace=0.38, wspace=0.35,
                            left=0.06, right=0.98, top=0.93, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    for a in [ax_b, ax_d, ax_e]:
        a.sharey(ax_c)

    draw_trajectory(ax_a, log_entries)

    draw_model_fit(
        ax_b, x_grid, C["mu_b"], C["sg_b"], x_train, y_train, x_test, y_test,
        title=r"\textbf{(b)} Iter 0: Linear $+$ Gaussian --- NLPD 2.16",
        panel_color=PANEL_COLORS[0], show_legend=True)

    draw_model_fit(
        ax_c, x_grid, C["mu_1"], C["sg_1"], x_train, y_train, x_test, y_test,
        title=r"\textbf{(c)} Iter 1: Cubic $+$ Student-$t$ --- NLPD 1.32",
        panel_color=PANEL_COLORS[1])

    draw_model_fit(
        ax_d, x_grid, C["mu_d"], C["sg_d"], x_train, y_train, x_test, y_test,
        title=r"\textbf{(d)} Iter 11: Sine $+$ mixture $+$ het.\ --- NLPD 1.23",
        panel_color=PANEL_COLORS[11])

    draw_tabpfn(
        ax_e, x_grid, C["mu_t"], C["q_lo"], C["q_hi"],
        x_train, y_train, x_test, y_test)

    for ax in [ax_c, ax_d, ax_e]:
        ax.set_xlabel("predictor")
    for ax in [ax_b, ax_c]:
        ax.set_ylabel("response")
    ax_d.tick_params(labelleft=False)
    ax_e.tick_params(labelleft=False)

    # save both PDF (vector) and PNG (preview)
    pdf_path = FIG_DIR / "figure1_large_combined.pdf"
    png_path = FIG_DIR / "figure1_large_combined.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close()
    print(f"\nSaved {pdf_path.name}  +  {png_path.name}")


if __name__ == "__main__":
    main()
