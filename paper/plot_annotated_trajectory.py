#!/usr/bin/env python3
"""Generate the annotated NLPD trajectory figure for the paper.

Central figure: NLPD over iterations for the 1D regression (small),
annotated with which model the agent chose at each step.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

ANNOTATIONS = {
    0: "Linear +\nGaussian",
    1: "Quadratic +\nStudent-t",
    2: "Cubic + het.\nσ(x) + Student-t",
    5: "Cubic + het. σ(x)\n+ mixture",
}

REJECTED_LABELS = {
    3: "Fourier",
    4: "Quartic",
    6: "t-clean",
    7: "sin(x)",
    8: "Quartic",
}


def load_log(dataset):
    path = REPO / "results" / dataset / "log.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def plot_small():
    entries = load_log("regression_1d")
    iters = [e["iter"] for e in entries]
    nlpds = [e["nlpd"] for e in entries]
    improved = [e["improved"] for e in entries]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(iters, nlpds, "-", color="#4A90D9", linewidth=1.5, zorder=3, alpha=0.6)

    for it, nl, imp in zip(iters, nlpds, improved):
        if imp is True or imp is None:
            ax.plot(it, nl, "o", color="#2E7D32", markersize=9, zorder=6)
        else:
            ax.plot(it, nl, "o", color="#C62828", markersize=6, zorder=6, alpha=0.4)

    ax.axhline(y=0.9443, color="gray", linestyle="--", linewidth=1,
               label="Oracle (0.9443)")
    ax.axhline(y=1.1184, color="#E65100", linestyle=":", linewidth=1,
               label="TabPFN (1.1184)")

    for it, label in ANNOTATIONS.items():
        nl = nlpds[iters.index(it)]
        offset_y = 0.07
        if it == 0:
            ax.annotate(label, xy=(it, nl), xytext=(it + 0.3, nl - 0.08),
                        fontsize=7.5, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#2E7D32", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#2E7D32",
                                        lw=1.2))
        elif it == 1:
            ax.annotate(label, xy=(it, nl), xytext=(it + 0.35, nl + 0.12),
                        fontsize=7.5, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#2E7D32", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#2E7D32",
                                        lw=1.2))
        elif it == 2:
            ax.annotate(label, xy=(it, nl), xytext=(it - 0.3, nl + 0.15),
                        fontsize=7.5, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#2E7D32", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#2E7D32",
                                        lw=1.2))
        elif it == 5:
            ax.annotate(label, xy=(it, nl), xytext=(it + 0.35, nl + 0.05),
                        fontsize=7.5, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9",
                                  ec="#2E7D32", lw=1.5, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#2E7D32",
                                        lw=1.5))

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("NLPD", fontsize=11)
    ax.set_title("Agent Trajectory — 1D Regression ($n=68$)", fontsize=12)
    ax.set_xticks(iters)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0.85, 2.35)
    plt.tight_layout()
    fname = FIG_DIR / "regression_1d_learning_curve.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    plot_small()
