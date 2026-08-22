"""The one contract every synthetic generator in `curricula` shares: labels sit
ON the answer token, not one ahead of it.

`curricula`'s module docstring states it ("labels sit ON the echoed answer
token, so the position right before it is what's actually supervised"), and
every caller depends on it -- `eval_mqar` and `eval_niah` both score
`logits[:, :-1]` against `labels[:, 1:]`, and the shared training loop's
synthetic path shifts the same way. If a generator ever emitted pre-shifted
labels instead, that shift would silently score the wrong position: the loss
would still be finite and the curve would still descend, so nothing else in
the suite would notice.

This is the same-position convention, and it is NOT interchangeable with the
memmap path's convention, where `MemmapPackedDataset` pre-shifts
(`input_ids = chunk[:-1]`, `labels = chunk[1:]`) and the model computes loss
internally with no shift. `test_loss_alignment.py` covers what goes wrong when
the two are crossed; this file pins the upstream half -- that the generators
really do produce what that analysis assumed.

The generator set is DISCOVERED from the module rather than enumerated, so a
fifth generator is covered the day it lands. `test_every_generator_is_covered`
is the guard-the-guard: without it, a discovery bug that matched nothing would
make every parametrised case below vacuously green.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from experimentation.experiments import curricula

pytestmark = pytest.mark.correctness

BATCH, SEQ_LEN, VOCAB = 6, 64, 128

# Values for parameters the generators require but do not default. Keyed by
# parameter name, not by generator, so generators sharing a parameter share the
# value -- and so an unrecognised required parameter fails loudly (see
# `_call_kwargs`) instead of being guessed at.
REQUIRED_PARAM_VALUES = {
    "batch": BATCH,
    "seq_len": SEQ_LEN,
    "vocab_size": VOCAB,
    "num_kv_pairs": 4,
}


def _discover_generators():
    """Every public `make_*` callable in `curricula`, by name."""
    return {
        name: obj
        for name, obj in vars(curricula).items()
        if name.startswith("make_") and callable(obj)
    }


def _call_kwargs(fn):
    """Fill in every parameter this generator has no default for."""
    kwargs = {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue
        if name not in REQUIRED_PARAM_VALUES:
            raise AssertionError(
                f"{fn.__name__} requires a parameter this test does not know how "
                f"to supply: {name!r}. Add it to REQUIRED_PARAM_VALUES."
            )
        kwargs[name] = REQUIRED_PARAM_VALUES[name]
    return kwargs


GENERATORS = _discover_generators()


def test_every_generator_is_covered():
    """Guard the guard: discovery must find the generators that exist.

    Compares against the module's own source, so this cannot drift with the
    dict above. A parametrised suite over an empty set passes trivially --
    this is what makes that failure mode visible.
    """
    from_source = {
        name
        for name, obj in inspect.getmembers(curricula, inspect.isfunction)
        if name.startswith("make_") and obj.__module__ == curricula.__name__
    }
    assert from_source, "no make_* generators found in curricula -- discovery is broken"
    assert set(GENERATORS) == from_source


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_labels_sit_on_the_answer_position_not_one_ahead(name):
    """At every supervised position, `labels[p] == input_ids[p]`.

    The label echoes the token at its own index. It is emphatically not
    `input_ids[p + 1]`, which is what a pre-shifted convention would give.
    """
    fn = GENERATORS[name]
    inputs, labels = fn(**_call_kwargs(fn))

    supervised = labels != -100
    assert supervised.any(), f"{name} supervised no position at all"
    assert torch.equal(inputs[supervised], labels[supervised])


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_the_preshifted_reading_is_actually_wrong_here(name):
    """The convention is a real choice, so the alternative must not also hold.

    If `labels[p] == input_ids[p + 1]` were true too, the assertion above
    would be satisfiable by data of either convention and would prove nothing.
    """
    fn = GENERATORS[name]
    inputs, labels = fn(**_call_kwargs(fn))

    # Supervised positions with a successor to compare against.
    supervised = labels != -100
    supervised[:, -1] = False
    if not supervised.any():
        pytest.skip(f"{name} supervises only the final position")

    next_token = torch.zeros_like(inputs)
    next_token[:, :-1] = inputs[:, 1:]
    assert not torch.equal(next_token[supervised], labels[supervised])


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_shapes_dtypes_and_unsupervised_fill(name):
    """The rest of the shared contract: paired (batch, seq_len) long tensors,
    with -100 everywhere the generator does not supervise."""
    fn = GENERATORS[name]
    inputs, labels = fn(**_call_kwargs(fn))

    assert inputs.shape == (BATCH, SEQ_LEN)
    assert labels.shape == (BATCH, SEQ_LEN)
    assert inputs.dtype is torch.long
    assert labels.dtype is torch.long
    # Every non-supervised entry is exactly the ignore_index, not 0 or -1.
    assert ((labels == -100) | (labels != -100)).all()
    assert (labels[labels != -100] >= 0).all()


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_the_caller_side_shift_lands_on_the_position_before_each_answer(name):
    """Why the convention matters, stated as the callers state it.

    `eval_mqar`/`eval_niah` score `logits[:, :-1]` against `labels[:, 1:]`.
    Under this convention that pairs the answer token with the position
    *before* it -- a causally blind prediction. Any supervised label in the
    final column is unreachable by that slice and is dropped by design.
    """
    fn = GENERATORS[name]
    inputs, labels = fn(**_call_kwargs(fn))

    target = labels[:, 1:]
    scored = target != -100
    if not scored.any():
        pytest.skip(f"{name} supervises only the final position")

    # The token being predicted is the answer itself...
    assert torch.equal(target[scored], inputs[:, 1:][scored])
    # ...read from the slot one earlier, which holds a different token.
    assert not torch.equal(target[scored], inputs[:, :-1][scored])
