# Per-study Optuna outputs (3M / 10M / 50M / 180M joint sweeps)

One directory per rung, copied from each study's `_studies/<study_id>/` folder.
The rolled-up cross-rung tables (`../data/*.csv`) and figures (`../figures/`)
are derived from these.

| Rung | Study id | Trials (rows in trials.csv) |
|---|---|---|
| 3m   | 3m-joint-v1.9229da2d.c1d3441f   | 3,022 |
| 10m  | 10m-joint-v1.125c7a71.35bd1d8c  | 271 |
| 50m  | 50m-joint-v1.3deabb22.435a057f  | 63 |
| 180m | 180m-joint-v1.a6c607b1.f2785b3f | 39 |

Row counts include pruned/failed/duplicate trials; `../data/rungs.csv` gives the
completed counts the analysis used (1,041 / 177 / 30 / 14).

## Files in each directory

- `trials.csv` -- every trial: sampled `param_*`, resolved `attr_*` (param count,
  ska_delta, throughput, peak memory, run_id), objective, and the full loss curve
  in `intermediate_values`.
- `best_trial.json`, `top_by_loss.csv`, `top_by_ska_delta.csv`, `top_trials.md`,
  `shortlist.csv` -- the winners.
- `importances.csv`, `main_effects.csv`, `conditional_effects.csv`,
  `interactions.md`, `macro_pairwise.json` -- what mattered and how.
- `noise_floor.csv` -- seed-noise sigma; effects smaller than this are not resolved.
- `anchor_contrasts.csv`, `rank_curve.csv`, `objective_vs_params.csv`,
  `pareto.csv`, `throughput_pareto.csv`, `sampler_correlations.csv`,
  `promotions.yaml`, `summary.json` -- supporting analyses.

## Not included

`optuna_journal.log` (the raw Optuna storage, ~7 MB per study) stays on scratch at
`/scratch/m000151/cody1212/pm06-migration/study-<rung>-joint-v1/_studies/<id>/`.

## Caveats

- Llama-2 tokenizer, ~65 tokens/parameter -- not the 100B Llama-3.1 protocol.
- The 3M learning-rate optimum is censored at the top of its search range.
- 180M: 24 of the 39 sampled trials are copies of one configuration (sampler-seed
  collision across single-worker jobs, fixed in 844b389), and architecture was
  not searched at that rung.
