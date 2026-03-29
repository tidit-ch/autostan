# Models Directory

Each subdirectory contains the Stan models produced by one AutoStan experiment run.
Files follow the naming convention:
- `baseline.stan` — the hand-written starting model
- `iter_NNN.stan` — the model submitted at iteration NNN
- `best.stan` — copy of the best-performing model (lowest test NLPD)

Corresponding evaluation logs and reports live in `results/<dataset>/`.

## Datasets and Source Branches

| Directory | Source Branch | Iters | Best NLPD | Notes |
|-----------|---------------|-------|-----------|-------|
| `bundesliga_labeled/` | `run/bundesliga-labeled-v1` | 13 | 1.5432 | Bundesliga match outcome prediction |
| `eight_schools/` | `run/eight-schools-v3` | 8 | 3.9544 | Eight Schools (v3): principled run with current `program.md`; smoke-test / contamination study |
| `eight_schools_v2/` | `run/eight-schools-v2` | 15 | 3.5500 | Eight Schools (v2): run with old `program.md`; exhibits explicit prior-gaming contamination |
| `regression_1d/` | `run/regression-1d-v1` | 9 | 1.1244 | 1D regression, small dataset |
| `regression_1d_large/` | `run/regression-1d-large-v1` | 15 | 1.2256 | 1D regression, large dataset (n=500), main benchmark |
| `synthetic_hierarchical_small/` | `run/synthetic-hierarchical-small-v2` | 4 | 1.4999 | Synthetic hierarchical model, small N |
| `synthetic_hierarchical_large/` | `run/synthetic-hierarchical-large-v2` | 8 | 1.4014 | Synthetic hierarchical model, large N |
| `synthetic_regression/` | `run/synthetic-regression-v1` | 14 | 1.2738 | Synthetic regression (varying slopes) |


