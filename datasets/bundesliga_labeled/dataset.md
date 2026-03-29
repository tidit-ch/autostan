# Dataset: Bundesliga Match Results

## Overview

Match results from the German Bundesliga (1st division football/soccer) with 18 teams. Each observation records a match between a home team and an away team with the number of goals scored by each. The goal is to predict goals in held-out matches.

## Data Format

**`train.csv`** columns:

- `home_team_id` — integer, home team identifier (1-18)
- `away_team_id` — integer, away team identifier (1-18)
- `home_goals` — non-negative integer, goals scored by the home team
- `away_goals` — non-negative integer, goals scored by the away team

207 training matches. 99 test matches (held out, not visible to the agent).

The split is temporal: training matches come from the first part of the season, test matches from the second part.

## Data Interface

The evaluation script passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
int<lower=0> J;
array[N_train] int<lower=1,upper=J> home_train;
array[N_train] int<lower=1,upper=J> away_train;
array[N_train] int<lower=0> goals_home_train;
array[N_train] int<lower=0> goals_away_train;
array[N_test] int<lower=1,upper=J> home_test;
array[N_test] int<lower=1,upper=J> away_test;
array[N_test] int<lower=0> goals_home_test;
array[N_test] int<lower=0> goals_away_test;
```

## Evaluation

```bash
python datasets/bundesliga_labeled/protected/evaluate.py --notes "..." --rationale "..."
```

Your model must output a `log_lik` vector of length `2 * N_test` in the `generated quantities` block. The first `N_test` entries are the log-likelihoods of `goals_home_test`, the second `N_test` entries are the log-likelihoods of `goals_away_test`.
