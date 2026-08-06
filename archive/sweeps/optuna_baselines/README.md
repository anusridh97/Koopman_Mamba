# Optuna baseline sweeps (harvested from `cody-sweep-baselines`)

Hyperparameter sweeps for the **baseline** models (mamba_only / mamba_attn /
transformer) at 1M, 5M, and 20M parameter scales, run against the old
`koopman-lm-fast` layout in June 2026. Harvested here so the branch can be
deleted; the interactive HTML plots were left behind (regenerate from
`trials.csv` with `optuna.visualization` if needed).

## Contents

| Path | What |
|---|---|
| `optuna_baseline_{1m,5m,20m}/` | plain studies — objectives: max `final_mqar_acc`, min `final_wikitext_ppl` |
| `optuna_baseline_{1m,5m,20m}_mqar_mix/` | mixed-objective studies (MQAR mixed into the training data) |
| `*/trials.csv` | every trial with full hyperparams + metrics |
| `*/pareto_front*.csv` | the non-dominated set (`_ppl_le_5000` = filtered to trials with wikitext PPL ≤ 5000) |
| `pareto_best.yaml` | all 11 Pareto-optimal trials across the 6 studies, distilled into one file |
| `optuna_baseline_search.py` | the sweep driver (targets the old layout; reference only) |

## Caveat

The hyperparameter keys (`attention_fraction`, `attention_placement`,
`mamba_expand`, …) parameterize the **old** `koopman-lm-fast` config, not the
current `koopman_lm.globals.config.KoopmanLMConfig`. Translate deliberately
before reusing — names overlap but defaults and wiring differ.
