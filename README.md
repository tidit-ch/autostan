# AutoStan

Autonomous Bayesian model improvement via scalar reward. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

A coding agent iteratively edits a Stan model, evaluates it via NLPD on held-out data, and keeps or reverts changes — no explicit search algorithm, no critic, just a single scalar reward.

> **For practical use on your own data**, see the [AutoStan skill](https://github.com/tidit-ch/autostan-skill) — a self-contained Claude Code skill that runs the full AutoStan loop without requiring a protected evaluation script.
> This repository contains the code to reproduce the experiments **in the preprint**.

## Setup

```bash
uv sync
uv run install_cmdstan  # downloads and builds CmdStan (one-time)
bash setup_permissions.sh  # restore agent-blocking permissions (not preserved by git)
```

## Running an Experiment

1. **Activate the environment and start Claude Code** (or any other CLI coding agent):
   ```bash
   source .venv/bin/activate
   claude  # requires Claude Code CLI: https://claude.ai/download
   ```

2. **Give it the launch prompt:**
   ```
   Read program.md for your instructions. Your dataset is datasets/regression_1d_large.
   ```

   That's it. The agent reads `program.md`, writes `model.stan`, runs `evaluate.py`, sees the NLPD, and iterates autonomously.

3. **Watch results** accumulate in:
   - `results/<dataset>/log.jsonl` — NLPD per iteration
   - `results/<dataset>/iter_XXX/stan.log` — Stan diagnostics
   - `models/<dataset>/` — model snapshots (baseline, each iteration, best)

## Available Datasets

| Dataset | Description |
|---|---|
| `regression_1d` | 1D regression with outliers, small ($n=68$) |
| `regression_1d_large` | 1D regression with outliers, large ($n=500$) |
| `synthetic_hierarchical_small` | Hierarchical partial pooling, small ($n_j=8$, 20 groups) |
| `synthetic_hierarchical_large` | Hierarchical partial pooling, large ($n_j=40$, 20 groups) |
| `synthetic_regression` | Varying slopes with correlated random effects ($J=15$ groups) |
| `bundesliga_labeled` | Bundesliga 2024/25 match results, Poisson attack/defense |

## How It Works

```
Agent reads program.md
  → writes/edits model.stan
  → runs: python datasets/<name>/protected/evaluate.py
  → sees NLPD + diagnostics
  → keeps or reverts
  → repeats
```

`evaluate.py` handles everything: compilation, sampling (4 chains × 1000 iterations), NLPD computation, model snapshots, logging. The agent just edits `model.stan` and reads the output.

**The only contract:** `model.stan` must output a `log_lik` vector in `generated quantities`.

## Project Structure

```
autostan/
├── program.md              # Agent instructions
├── model.stan              # The file the agent edits
├── setup_permissions.sh    # Restore agent-blocking file permissions after clone
├── datasets/<name>/
│   ├── dataset.md          # Data description (agent reads this)
│   ├── train.csv           # Training data (agent can read)
│   └── protected/          # Agent cannot read (chmod 000/111); humans can
│       ├── evaluate.py     # Evaluation + bookkeeping (chmod 111: execute only)
│       ├── test.csv        # Held-out test data (chmod 000)
│       └── generate.py     # Data generation script (chmod 000)
├── results/<name>/         # Iteration logs and diagnostics
├── models/<name>/          # Model snapshots per iteration
└── paper/                  # Figure scripts and rendered figures
```
