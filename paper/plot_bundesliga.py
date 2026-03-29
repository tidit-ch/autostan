#!/usr/bin/env python3
"""Generate Bundesliga appendix plots from the best model (iter 9).

Creates two figures:
1. Team-specific home advantage (delta parameters)
2. Attack vs Defense strength per team

Requires: cmdstanpy, matplotlib, numpy, pandas
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

REPO = Path(__file__).resolve().parent.parent
BEST_STAN = REPO / "models" / "bundesliga_labeled" / "best.stan"
TRAIN_CSV = REPO / "datasets" / "bundesliga_labeled" / "train.csv"
# Test data is protected, but we only need it for the data block (evaluate.py handles scoring)
TEST_CSV = REPO / "datasets" / "bundesliga_labeled" / "protected" / "test.csv"
TEAM_MAP = REPO / "datasets" / "bundesliga" / "protected" / "team_mapping.json"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    with open(TEAM_MAP) as f:
        team_map = json.load(f)

    J = max(train["home_team_id"].max(), train["away_team_id"].max())

    data = {
        "N_train": len(train),
        "N_test": len(test),
        "J": J,
        "home_train": train["home_team_id"].tolist(),
        "away_train": train["away_team_id"].tolist(),
        "goals_home_train": train["home_goals"].tolist(),
        "goals_away_train": train["away_goals"].tolist(),
        "home_test": test["home_team_id"].tolist(),
        "away_test": test["away_team_id"].tolist(),
        "goals_home_test": test["home_goals"].tolist(),
        "goals_away_test": test["away_goals"].tolist(),
    }
    return data, team_map, J


def shorten_name(name):
    """Shorten team names for plot labels."""
    replacements = {
        "1. FC Heidenheim 1846": "Heidenheim",
        "1. FC Union Berlin": "Union Berlin",
        "1. FSV Mainz 05": "Mainz",
        "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund",
        "Borussia Mönchengladbach": "Gladbach",
        "Eintracht Frankfurt": "Frankfurt",
        "FC Augsburg": "Augsburg",
        "FC Bayern München": "Bayern",
        "FC St. Pauli": "St. Pauli",
        "Holstein Kiel": "Kiel",
        "RB Leipzig": "Leipzig",
        "SC Freiburg": "Freiburg",
        "SV Werder Bremen": "Bremen",
        "TSG Hoffenheim": "Hoffenheim",
        "VfB Stuttgart": "Stuttgart",
        "VfL Bochum": "Bochum",
        "VfL Wolfsburg": "Wolfsburg",
    }
    return replacements.get(name, name)


def main():
    data, team_map, J = load_data()

    # Copy best model and compile
    model_path = REPO / "model_plot.stan"
    shutil.copy(BEST_STAN, model_path)
    model = CmdStanModel(stan_file=str(model_path))
    fit = model.sample(data=data, chains=4, iter_sampling=2000, iter_warmup=1000, seed=42)

    # Extract posteriors
    delta_draws = fit.stan_variable("delta")  # (n_draws, J)
    attack_draws = fit.stan_variable("attack")
    defense_draws = fit.stan_variable("defense")

    delta_mean = delta_draws.mean(axis=0)
    delta_q05 = np.percentile(delta_draws, 5, axis=0)
    delta_q95 = np.percentile(delta_draws, 95, axis=0)

    attack_mean = attack_draws.mean(axis=0)
    attack_q05 = np.percentile(attack_draws, 5, axis=0)
    attack_q95 = np.percentile(attack_draws, 95, axis=0)

    defense_mean = defense_draws.mean(axis=0)
    defense_q05 = np.percentile(defense_draws, 5, axis=0)
    defense_q95 = np.percentile(defense_draws, 95, axis=0)

    # Team labels
    labels = [shorten_name(team_map[str(j + 1)]) for j in range(J)]

    # --- Plot 1: Team-specific home advantage ---
    fig, ax = plt.subplots(figsize=(8, 6))
    order = np.argsort(delta_mean)
    y_pos = np.arange(J)

    ax.barh(y_pos, delta_mean[order], color="#4A90D9", alpha=0.7, height=0.7)
    ax.errorbar(
        delta_mean[order], y_pos,
        xerr=[delta_mean[order] - delta_q05[order], delta_q95[order] - delta_mean[order]],
        fmt="none", ecolor="black", capsize=2, linewidth=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax.set_xlabel("Home advantage (log-rate scale)", fontsize=11)
    ax.set_title("Team-Specific Home Advantage — Bundesliga 2024/25", fontsize=12)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    # Add global mean line
    delta_mu_mean = fit.stan_variable("delta_mu").mean()
    ax.axvline(x=delta_mu_mean, color="red", linestyle=":", linewidth=1, label=f"Global mean ({delta_mu_mean:.2f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "bundesliga_home_advantage.png", dpi=150)
    plt.close()
    print(f"Saved {FIG_DIR / 'bundesliga_home_advantage.png'}")

    # --- Plot 2: Attack vs Defense ---
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(attack_mean, defense_mean, s=60, color="#4A90D9", zorder=5, alpha=0.8)

    # Error bars (90% CI)
    for j in range(J):
        ax.plot(
            [attack_q05[j], attack_q95[j]], [defense_mean[j], defense_mean[j]],
            color="gray", linewidth=0.5, alpha=0.5, zorder=3,
        )
        ax.plot(
            [attack_mean[j], attack_mean[j]], [defense_q05[j], defense_q95[j]],
            color="gray", linewidth=0.5, alpha=0.5, zorder=3,
        )

    # Label each point
    for j in range(J):
        ax.annotate(
            labels[j], (attack_mean[j], defense_mean[j]),
            textcoords="offset points", xytext=(6, 4), fontsize=7.5,
            ha="left", va="bottom",
        )

    ax.set_xlabel("Attack strength (higher = more goals scored)", fontsize=11)
    ax.set_ylabel("Defense strength (higher = fewer goals conceded by opponent)", fontsize=11)
    ax.set_title("Attack vs Defense — Bundesliga 2024/25", fontsize=12)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1] * 0.7, ylim[1] * 0.85, "Strong overall", fontsize=8, color="green", alpha=0.5, ha="center")
    ax.text(xlim[0] * 0.7, ylim[0] * 0.85, "Weak overall", fontsize=8, color="red", alpha=0.5, ha="center")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "bundesliga_attack_defense.png", dpi=150)
    plt.close()
    print(f"Saved {FIG_DIR / 'bundesliga_attack_defense.png'}")

    # Cleanup temp model file
    model_path.unlink(missing_ok=True)
    for p in REPO.glob("model_plot*"):
        p.unlink(missing_ok=True)

    # Print summary
    print("\n=== Team Parameter Summary ===")
    print(f"{'Team':<20} {'Attack':>8} {'Defense':>8} {'Home Adv':>8}")
    print("-" * 50)
    for j in range(J):
        print(f"{labels[j]:<20} {attack_mean[j]:>8.3f} {defense_mean[j]:>8.3f} {delta_mean[j]:>8.3f}")


if __name__ == "__main__":
    main()
