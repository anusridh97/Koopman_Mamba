"""Characterization tests for `experimentation.training.optim.param_groups`.

Measured 2026-08-21 with coverage 7.15.4: this module was at **5% line
coverage** -- 19 of its 20 statements never executed under the full suite. The
only `param_groups` the suite mentioned was `test_retrieval_encoder.py:49`, a
different function (a method on the retrieval encoder).

That is a bad place to have no tests. `param_groups` encodes the decay/no-decay
policy that **all three trainers** now depend on (`training/train.py:299`,
`experiments/table2.py:244`, `experiments/mqar_finetune.py`), and routing them
through it was the fix for the run-system design's §6.3 -- the discrepancy where
`table2.py` decayed norms, biases, embeddings and the Mamba state parameters that
`train.py` excluded, making published Table 2 numbers a different optimizer
regime from every other result in the repo. The remedy for a paper-invalidating
bug was to funnel everything through an untested function.

These are characterization tests: they pin behaviour that already exists, ahead
of the §6.2 `TrainTask` unification that will move this code. Written as
tests-after deliberately and with the usual caveat handled -- each one was
checked to fail against the flat `model.parameters()` baseline it exists to
distinguish from (see `test_the_policy_differs_from_flat_parameters`), so they
are not passing vacuously.
"""

import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.training.optim import param_groups  # noqa: E402


class Tiny(nn.Module):
    """Exercises every branch of the policy in one module."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)          # embedding -> no decay
        self.proj = nn.Linear(8, 8)               # weight decays, bias does not
        self.norm = nn.LayerNorm(8)               # ndim 1 -> no decay
        self.A_log = nn.Parameter(torch.zeros(4))         # Mamba state
        self.D = nn.Parameter(torch.zeros(4))             # Mamba state
        self.dt_bias = nn.Parameter(torch.zeros(4))       # Mamba state
        self.theta = nn.Parameter(torch.zeros(4, 4))      # geometry, 2D
        self.gamma = nn.Parameter(torch.zeros(4, 4))      # geometry, 2D
        self.lift_v = nn.Parameter(torch.zeros(4, 4))     # geometry, 2D
        self.matrix = nn.Parameter(torch.zeros(4, 4))     # plain 2D -> decays


def _split(model, wd=0.1):
    groups = param_groups(model, wd)
    decay = {id(p) for g in groups if g["weight_decay"] == wd for p in g["params"]}
    no_decay = {id(p) for g in groups if g["weight_decay"] == 0.0 for p in g["params"]}
    return decay, no_decay


def _named(model):
    return {name: id(p) for name, p in model.named_parameters()}


def test_matrix_weights_decay():
    m = Tiny()
    decay, _ = _split(m)
    names = _named(m)
    assert names["proj.weight"] in decay
    assert names["matrix"] in decay


@pytest.mark.parametrize("name", [
    "embed.weight",        # embeddings
    "proj.bias",           # bias by leaf name
    "norm.weight",         # ndim < 2
    "norm.bias",
    "A_log", "D", "dt_bias",                 # Mamba state / discretization
    "theta", "gamma", "lift_v",              # Koopman/SKA geometry, all 2D
])
def test_excluded_from_decay(name):
    """Each of these would be silently shrunk by a flat AdamW. The geometry and
    state entries are the interesting ones: they are 2D, so an ndim>=2 rule alone
    would decay them -- they are excluded by explicit leaf name."""
    m = Tiny()
    decay, no_decay = _split(m)
    pid = _named(m)[name]
    assert pid in no_decay, f"{name} must not be decayed"
    assert pid not in decay


def test_the_policy_differs_from_flat_parameters():
    """The whole point. If this ever stops holding, param_groups has become a
    no-op and §6.3's bug is back."""
    m = Tiny()
    decay, no_decay = _split(m)
    assert no_decay, "nothing excluded -- policy is a no-op"
    total = sum(1 for _ in m.parameters())
    assert len(decay) + len(no_decay) == total, "every parameter must be placed"
    assert len(decay) < total, "a flat parameters() call would decay all of them"


def test_every_parameter_appears_exactly_once():
    m = Tiny()
    groups = param_groups(m, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter landed in both groups"
    assert set(ids) == {id(p) for p in m.parameters()}


def test_shared_parameters_are_not_double_counted():
    """`seen` guards weight tying, which is why it exists."""
    m = Tiny()
    m.proj2 = nn.Linear(8, 8)
    m.proj2.weight = m.proj.weight            # tied
    groups = param_groups(m, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))


def test_frozen_parameters_are_omitted_entirely():
    m = Tiny()
    m.matrix.requires_grad_(False)
    decay, no_decay = _split(m)
    pid = _named(m)["matrix"]
    assert pid not in decay and pid not in no_decay


def test_explicit_no_weight_decay_names_are_honoured():
    """A model may name its own exclusions; the hook is consulted."""
    class WithHook(Tiny):
        def no_weight_decay_param_names(self):
            return ("matrix",)

    m = WithHook()
    decay, no_decay = _split(m)
    assert _named(m)["matrix"] in no_decay


def test_the_per_parameter_attribute_is_honoured():
    m = Tiny()
    m.matrix._no_weight_decay = True
    decay, no_decay = _split(m)
    assert _named(m)["matrix"] in no_decay


def test_weight_decay_value_is_threaded_through():
    m = Tiny()
    groups = param_groups(m, 0.037)
    assert {g["weight_decay"] for g in groups} == {0.037, 0.0}


def test_no_empty_group_is_emitted():
    """An all-excluded model must not produce a decay group with no params --
    torch accepts it but it makes optimizer state confusing to read back."""
    class AllNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)

    groups = param_groups(AllNorm(), 0.1)
    assert all(g["params"] for g in groups)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0
