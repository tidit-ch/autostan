#!/usr/bin/env python3
"""Collect log.jsonl files from run branches into a single results file.

Usage:
    uv run python analysis/collect_results.py

Reads log.jsonl from each run branch (via git show) and writes
analysis/all_runs.jsonl with dataset and run fields added.
"""

import json
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent

# Define runs to collect: (branch, dataset, label)
RUNS = [
    ("run/eight-schools-v2", "eight_schools", "eight_schools_v2"),
    ("run/synthetic-hierarchical-small-v1", "synthetic_hierarchical_small", "synth_hier_small_v1"),
    ("run/synthetic-hierarchical-large-v1", "synthetic_hierarchical_large", "synth_hier_large_v1"),
]


def get_log_from_branch(branch, dataset):
    """Read log.jsonl from a run branch via git show."""
    path = f"results/{dataset}/log.jsonl"
    try:
        result = subprocess.run(
            ["git", "show", f"{branch}:{path}"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print(f"WARNING: could not read {path} from branch {branch}", file=sys.stderr)
        return ""


def main():
    all_entries = []

    for branch, dataset, label in RUNS:
        log_text = get_log_from_branch(branch, dataset)
        for line in log_text.strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                entry["dataset"] = dataset
                entry["run"] = label
                all_entries.append(entry)

    # Write combined file
    out_path = ANALYSIS_DIR / "all_runs.jsonl"
    with open(out_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Collected {len(all_entries)} entries from {len(RUNS)} runs → {out_path}")

    # Also write a summary
    for branch, dataset, label in RUNS:
        entries = [e for e in all_entries if e["run"] == label]
        if entries:
            best = min(entries, key=lambda e: e["nlpd"])
            print(f"  {label}: {len(entries)} iters, best NLPD={best['nlpd']:.4f} (iter {best['iter']})")


if __name__ == "__main__":
    main()
