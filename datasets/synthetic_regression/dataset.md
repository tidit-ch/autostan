# Dataset: Synthetic Regression

## Overview

Observations from 15 units, each with a continuous predictor and a continuous response. The goal is to predict `response` for held-out test observations.

## Data Format

**`train.csv`** columns:
- `unit_id` — integer, group identifier (1–15)
- `predictor` — continuous predictor variable
- `response` — continuous response variable (target)

300 training observations (20 per unit).

75 test observations (5 per unit, held out, not visible to the agent).

## Data Interface

The evaluation script passes the following to Stan:

```
int<lower=0> N_train;
int<lower=0> N_test;
int<lower=0> J;
array[N_train] int<lower=1,upper=J> unit_train;
array[N_test] int<lower=1,upper=J> unit_test;
vector[N_train] predictor_train;
vector[N_test] predictor_test;
vector[N_train] response_train;
vector[N_test] response_test;
```

## Evaluation

```
python datasets/synthetic_regression/protected/evaluate.py --notes "..." --rationale "..."
```

Your model must output a `log_lik` vector of length `N_test` in the `generated quantities` block.
