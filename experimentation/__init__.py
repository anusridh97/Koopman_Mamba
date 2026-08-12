"""Research and infrastructure code that is NOT part of the koopman_lm package.

koopman_lm/ is the publishable model: config, models, modules, kernels. It has
no knowledge of Slurm, wandb, sweep grids, or any particular study. Everything
here depends on koopman_lm; nothing in koopman_lm depends on anything here, and
code-tests/test_package_boundary.py enforces that direction.

    training/     the trainer, optimizer policy, exact resume, data prep
    run/          RunSpec, run identity, materialization, launchers
    sweep/        sweep grids expanded into RunSpecs (declared exactly once)
    evaluation/   eval harnesses and the result envelope
    experiments/  study-specific code (table2, MQAR fine-tuning, curricula)
    retrieval/    Phase-2 dense-retrieval adaptation
    results.py    the spec.yaml + eval/**/*.json aggregation walk

Deliberately NOT installed: pyproject's packages.find is scoped to koopman_lm*,
so `pip install koopman-lm` ships the model and none of this. Run these as
modules from the repo root, e.g. `python -m experimentation.run <spec.yaml>`.
"""
