# AutoStan — Agent Instructions

You are an autonomous Bayesian modeling agent. Your task is to iteratively improve a Stan model to minimize NLPD (negative log predictive density) on a held-out test set.

## Workflow

1. **Read the dataset description**: Read `datasets/<dataset>/dataset.md` to understand the data, format, and evaluation procedure.
2. **Check history**: Before proposing any change, read `results/<dataset>/log.jsonl` to see the full NLPD history and what has been tried.
3. **Read training data**: Read `datasets/<dataset>/train.csv` to understand the data structure and values.
4. **Edit `model.stan`**: Modify the model — you can change priors, likelihood, parameterization, model structure, anything.
5. **Evaluate**: Run the evaluation script as specified in `dataset.md`. Always pass `--notes` (what you changed) and `--rationale` (why, referencing previous iterations), e.g.:
   ```
   python datasets/<dataset>/protected/evaluate.py --notes "NCP, tau~Exp(1)" --rationale "iter 2 had 47 divergences from centered funnel; NCP should fix geometry, Exp(1) prior keeps tau small"
   ```
6. **Interpret results**: Read the printed NLPD and diagnostics. Decide whether to keep or revert the change.
7. **Repeat**: Propose the next change based on what you've learned.

## Rules

- **NLPD is the only metric.** Lower NLPD = better model.
- **`log_lik` is the only interface contract.** Your `model.stan` must always output a `log_lik` vector in the `generated quantities` block. The likelihood family, priors, and parameterization are all your choice.
- **Do NOT read any files in `protected/`.** Only execute the evaluation script.
- **Do NOT randomly perturb.** Think like a statistician reasoning from evidence. Reference previous iterations when explaining your changes.
- **The filesystem is your memory.** All history is in `results/` — read it before acting.
- **Reason briefly** about *why* you make each change, referencing NLPD history, diagnostics, and previous iterations.

## Strategies to Consider

- Non-centered vs centered parameterization
- Prior tightening or loosening
- Different likelihood families (Normal, Student-t, etc.)
- Adding or removing hierarchy levels
- Reparameterization for better sampling efficiency
- Covariate effects, interactions
- Regularizing priors

## Stopping Rule

Stop iterating when **any** of these conditions is met:
- **3 consecutive non-improving iterations** (NLPD did not decrease)
- **20 total iterations** (including baseline)

When you stop, write a report (see below).

## Report

After stopping, write `results/<dataset>/report.md` summarizing:
1. **Best model**: which iteration, NLPD, key design choices
2. **Trajectory**: brief narrative of what was tried and why it worked or didn't
3. **NLPD table**: all iterations with NLPD and one-line description
4. **Key insights**: what the agent learned about this dataset

## Getting Started

If no `results/<dataset>/log.jsonl` exists yet, write a deliberately simple baseline model. Run it to establish the baseline NLPD, then start improving.
