#!/usr/bin/env python3
"""Evaluate model.stan on the Synthetic Hierarchical dataset.

Handles compilation, sampling, NLPD computation, and all bookkeeping.
"""

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATASET_DIR.parent.parent
DATASET_NAME = DATASET_DIR.name

MODEL_PATH = PROJECT_ROOT / "model.stan"
MODELS_DIR = PROJECT_ROOT / "models" / DATASET_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / DATASET_NAME
LOG_FILE = RESULTS_DIR / "log.jsonl"
TEST_CSV = SCRIPT_DIR / "test.csv"
TRAIN_CSV = DATASET_DIR / "train.csv"


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model.stan on Synthetic Hierarchical dataset")
    parser.add_argument("--notes", type=str, default="", help="One-line description of this iteration's changes")
    parser.add_argument("--rationale", type=str, default="", help="Why this change was made, referencing previous iterations")
    args = parser.parse_args()

    import cmdstanpy

    if not MODEL_PATH.exists():
        print("ERROR: model.stan not found at project root.")
        sys.exit(1)

    # Load data
    train_rows = load_csv(TRAIN_CSV)
    test_rows = load_csv(TEST_CSV)

    # Get unique group IDs
    all_unit_ids = sorted(set(int(r["unit_id"]) for r in train_rows + test_rows))
    J = len(all_unit_ids)

    stan_data = {
        "N_train": len(train_rows),
        "N_test": len(test_rows),
        "J": J,
        "unit_train": [int(r["unit_id"]) for r in train_rows],
        "unit_test": [int(r["unit_id"]) for r in test_rows],
        "effect_train": [float(r["effect"]) for r in train_rows],
        "effect_test": [float(r["effect"]) for r in test_rows],
    }

    # Compile
    print("Compiling model.stan...")
    try:
        model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_PATH))
    except Exception as e:
        print(f"COMPILATION ERROR:\n{e}")
        sys.exit(1)

    # Sample: 4 chains x 1000 iterations
    print("Sampling: 4 chains x 1000 iterations...")
    start_time = time.time()
    try:
        fit = model.sample(
            data=stan_data,
            chains=4,
            iter_sampling=1000,
            iter_warmup=1000,
            seed=42,
            show_console=False,
        )
    except Exception as e:
        print(f"SAMPLING ERROR:\n{e}")
        sys.exit(1)
    runtime_sec = round(time.time() - start_time, 1)

    # Extract log_lik
    try:
        log_lik = fit.stan_variable("log_lik")
    except Exception:
        print("ERROR: model.stan must output 'log_lik' in generated quantities block.")
        print("log_lik should have length N_test (one entry per test observation).")
        sys.exit(1)

    # Compute NLPD
    nlpd = -np.mean(logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0]))

    # Diagnostics
    diagnostics = fit.diagnose()

    summary = fit.summary()
    r_hat_max = float(summary["R_hat"].max()) if "R_hat" in summary.columns else float("nan")

    # Count divergences
    n_divergences = 0
    try:
        sampler_vars = fit.method_variables()
        if "divergent__" in sampler_vars:
            n_divergences = int(sampler_vars["divergent__"].sum())
    except Exception:
        pass

    # Bookkeeping: create dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine iteration number
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            lines = [l for l in f if l.strip()]
        iteration = len(lines)
    else:
        iteration = 0

    # Check improvement against best so far
    best_nlpd = None
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if best_nlpd is None or entry["nlpd"] < best_nlpd:
                        best_nlpd = entry["nlpd"]

    improved = bool(nlpd < best_nlpd) if best_nlpd is not None else None

    # Save model snapshot
    iter_label = f"iter_{iteration:03d}"
    if iteration == 0:
        model_snapshot = MODELS_DIR / "baseline.stan"
    else:
        model_snapshot = MODELS_DIR / f"{iter_label}.stan"
    shutil.copy2(MODEL_PATH, model_snapshot)

    # Update best model
    if improved or improved is None:
        shutil.copy2(MODEL_PATH, MODELS_DIR / "best.stan")

    # Save stan.log
    iter_results_dir = RESULTS_DIR / iter_label
    iter_results_dir.mkdir(parents=True, exist_ok=True)
    with open(iter_results_dir / "stan.log", "w") as f:
        f.write(diagnostics)

    # Append to log.jsonl
    log_entry = {
        "iter": iteration,
        "nlpd": round(float(nlpd), 4),
        "improved": improved,
        "machine": platform.node(),
        "runtime_sec": runtime_sec,
        "n_divergences": n_divergences,
        "r_hat_max": round(float(r_hat_max), 4),
        "notes": "baseline" if iteration == 0 else args.notes,
        "rationale": "" if iteration == 0 else args.rationale,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Print results for agent
    print()
    print("=" * 50)
    print(f"DATASET: {DATASET_NAME}")
    print(f"ITERATION: {iteration}")
    print(f"NLPD: {nlpd:.4f}")
    if best_nlpd is not None:
        print(f"BEST NLPD: {best_nlpd:.4f}")
        print(f"IMPROVED: {improved}")
    print(f"DIVERGENCES: {n_divergences}")
    print(f"MAX R-HAT: {r_hat_max:.4f}")
    print(f"RUNTIME: {runtime_sec}s")
    print(f"MODEL SAVED: {model_snapshot}")
    print(f"STAN LOG: {iter_results_dir / 'stan.log'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
