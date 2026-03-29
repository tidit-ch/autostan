# Dataset: Synthetic Hierarchical (Large)

We measured a continuous outcome ("effect") for 1000 units belonging to 20 groups (50 units per group). We suspect there is variation both within and between groups — units in the same group may share a common tendency.

## Data Format
`train.csv` columns:
- `unit_id`: integer, group identifier (1–20)
- `effect`: float, observed outcome

There are 800 training observations (40 per group). A held-out test set of 200 observations (10 per group) is used for evaluation.

## Data Interface
Your `model.stan` must declare this data block:
```stan
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> unit_train;
  array[N_test] int<lower=1,upper=J> unit_test;
  vector[N_train] effect_train;
  vector[N_test] effect_test;
}
```

## Goal
Minimize NLPD on the held-out test set by improving `model.stan`.
Lower NLPD = better model.

## Evaluation
Run: `uv run python datasets/synthetic_hierarchical_large/protected/evaluate.py --notes "what changed" --rationale "why, referencing previous iterations"`
Do not attempt to read any files in `protected/`.

## log_lik Contract
Your `model.stan` must output a vector `log_lik` of length `N_test` in the `generated quantities` block, containing the log-likelihood of each test observation under the model.
