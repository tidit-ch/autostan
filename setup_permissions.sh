#!/usr/bin/env bash
# Re-apply runtime permissions that git does not preserve.
# These prevent the agent from reading test data and protected scripts,
# while still allowing humans to read and reproduce experiments.

set -euo pipefail

for dir in datasets/*/protected; do
    # evaluate.py: executable but not readable by the agent
    chmod 111 "$dir/evaluate.py"

    # test data and generation scripts: no access for the agent
    [[ -f "$dir/test.csv" ]]          && chmod 000 "$dir/test.csv"
    [[ -f "$dir/generate.py" ]]       && chmod 000 "$dir/generate.py"
    [[ -f "$dir/ground_truth.json" ]] && chmod 000 "$dir/ground_truth.json"
done

echo "Permissions restored."
