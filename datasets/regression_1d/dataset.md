# Dataset: 1D Regression

## Overview

Observations of a continuous predictor and a continuous response. The goal is to predict `response` for held-out test observations.

## Data Format

**`train.csv`** columns:
- `predictor` — continuous predictor variable
- `response` — continuous response variable (target)

68 training observations. 30 test observations (held out, not visible to the agent).

## Data Interface

The evaluation script passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] predictor_train;
vector[N_test] predictor_test;
vector[N_train] response_train;
vector[N_test] response_test;
```

## Evaluation

```bash
python datasets/regression_1d/protected/evaluate.py --notes "..." --rationale "..."
```

Your model must output a `log_lik` vector of length `N_test` in the `generated quantities` block.
