"""Adaptive search over the same RunSpecs a static sweep produces.

Lives inside `experimentation/sweep/` rather than beside it because both halves
answer one question -- how do many runs get made -- and both drive the same
primitive, `experimentation.sweep.launch.materialize_cell`. A sibling package
would have had to import that from a `__main__` module, duplicate it, or force
the same extraction anyway.

`sweep/spec.py`'s rule is that a sweep's *grid* is declared exactly once. The
rule here is the same one level up: a search's *space* is declared exactly once,
in `space.py`. Adaptive search extends that principle rather than contradicting
it -- the difference is only that the grid is enumerated up front while the
space is sampled from as results arrive.

    python -m experimentation.sweep <sweep.yaml>          static grid
    python -m experimentation.sweep.search <study.yaml>   adaptive

The modules split along the optuna-dependency boundary, deliberately:

    geometry.py   pure arithmetic over layer indices        no optuna
    space.py      the space, and params -> RunSpec overrides  no optuna
    anchors.py    curated designs -> concrete params        no optuna
    ---------------------------------------------------------------
    study.py      sampler / storage / pruner / enqueue      imports optuna
    driver.py     ask -> materialize -> submit -> tell      imports optuna
    metrics.py    read a finished run's eval JSON
    report.py     trials.csv, top_trials.md, promotions

Everything above the line is importable, testable and useful with no optuna
installed, which is what lets a curated set of anchor designs ship as an
ordinary `cells:` sweep before the adaptive machinery exists. Nothing in this
package's `__init__` imports its submodules, so `import experimentation.sweep`
stays free of the optional dependency.
"""
