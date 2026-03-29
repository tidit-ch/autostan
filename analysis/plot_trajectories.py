#!/usr/bin/env python3
"""Plot NLPD trajectories for all AutoStan runs.

Usage:
    uv run python analysis/plot_trajectories.py

Reads analysis/all_runs.jsonl (produced by collect_results.py)
and generates trajectory plots per dataset.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent

# Bayes-optimal NLPD for each dataset (computed from true generative process)
OPTIMAL_NLPD = {
    "eight_schools": None,  # contaminated, no meaningful optimal
    "synthetic_hierarchical_small": 1.4935,
    "synthetic_hierarchical_large": 1.4039,
}


def load_runs():
    entries = []
    with open(ANALYSIS_DIR / "all_runs.jsonl") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def plot_dataset(entries, dataset, ax):
    """Plot NLPD trajectory for one dataset."""
    iters = [e["iter"] for e in entries]
    nlpds = [e["nlpd"] for e in entries]

    # Main trajectory
    ax.plot(iters, nlpds, "o-", color="#2563eb", markersize=6, linewidth=2, label="NLPD")

    # Mark best
    best_idx = np.argmin(nlpds)
    ax.plot(iters[best_idx], nlpds[best_idx], "s", color="#16a34a", markersize=10,
            zorder=5, label=f"Best: {nlpds[best_idx]:.4f} (iter {iters[best_idx]})")

    # Mark non-improving iterations
    for e in entries:
        if e["improved"] is False:
            ax.plot(e["iter"], e["nlpd"], "x", color="#dc2626", markersize=8, markeredgewidth=2)

    # Optimal line
    optimal = OPTIMAL_NLPD.get(dataset)
    if optimal is not None:
        ax.axhline(y=optimal, color="#9333ea", linestyle="--", linewidth=1.5,
                   label=f"Bayes-optimal: {optimal:.4f}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("NLPD")
    ax.set_title(dataset.replace("_", " ").title())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    entries = load_runs()

    # Group by dataset
    datasets = {}
    for e in entries:
        datasets.setdefault(e["dataset"], []).append(e)

    # Create figure
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, (dataset, data) in enumerate(sorted(datasets.items())):
        data.sort(key=lambda e: e["iter"])
        plot_dataset(data, dataset, axes[0, i])

    fig.suptitle("AutoStan: NLPD Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = ANALYSIS_DIR / "trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
