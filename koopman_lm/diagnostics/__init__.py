"""Measurement of a model, as opposed to measurement of a training run.

Everything here is read-only, `no_grad`, and off the training path. Nothing in
`koopman_lm/` imports this package, so it can never change what a model does.

    ska.py           ska_health(ska_module, x)  -- operator spectral radius,
                     Gram lambda_min, power-filter gap, gate/beta/eta/gamma
    koopman_mlp.py   gain_stats / dead_pair_stats / utilization_report(model, x)
                     -- lifted-dimension gain and dead-pair fractions

The split against `experimentation/training/diagnostics.py` is the same one
`pyproject.toml` already enforces between the package and the research tree:
intrinsic properties of a trained model ship with the model, so a checkpoint can
be inspected by anyone who installed it. Instrumenting a *run* -- forward hooks,
`capture()`, wandb histograms, gradient-flow probes, the SKA-vs-Mamba residual
ratio -- needs the training loop, and stays there.

`koopman_mlp.py` was `modules/mlp/koopman_diag.py`. It never belonged there:
`modules/` holds layer components, and these functions take a whole model and
walk `model.modules()`.
"""
from koopman_lm.diagnostics.ska import ska_health, spectral_radius
from koopman_lm.diagnostics.koopman_mlp import (
    iter_koopman_mlps, gain_stats, dead_pair_stats, utilization_report,
    format_report,
)

__all__ = [
    "ska_health",
    "spectral_radius",
    "iter_koopman_mlps",
    "gain_stats",
    "dead_pair_stats",
    "utilization_report",
    "format_report",
]
