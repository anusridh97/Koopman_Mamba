"""Zeroing the SKA branch of ANY model, including the ones without `.ablate`.

## Why this is not just `KoopmanLM.ablate`

`models/koopman_lm.py::KoopmanLM.ablate` is a method on that class. The MQAR arm
does not use that class -- `experiments/mqar_finetune.py` builds through
`models/baselines.py::build_mamba_ska_swiglu`, whose `_build_model` returns a
locally-defined `_Model` with no `ablate` at all. `quick_eval` already notices
this and degrades gracefully: `if ska_ablation and hasattr(model, "ablate")`,
which on a baselines model silently means NO ablation is measured and the payload
says `supported: False`.

That is fine for LM eval and useless here, because the SKA-on vs SKA-zeroed
retrieval delta is the PRIMARY discriminator of the exponent arm. Held-out LM
loss is proven non-responsive to SKA damage in this repo -- 18 trials destroyed
44% of SKA's contribution and moved mean loss by -1.6e-4, far below the 7.54e-3
floor -- while the ablation delta separates the same conditions at t = 7-22. So
the arm needs a zeroing that works on the model the arm actually builds.

## What it must get right

`SKABlock` has one flag, `_ablate`, and a zeroed block is a pure residual
passthrough. The two failure modes worth testing are both silent:

  * **Missing a block.** A model whose SKA blocks are not reached by the walk
    would report a smaller delta than the truth, which is the direction that
    makes a policy look less load-bearing than it is.
  * **Leaving the flag set.** A flag not restored on exit turns every subsequent
    measurement into an ablated one, so the FULL accuracy of the next run in the
    same process would silently be the zeroed accuracy. Restoration has to
    survive an exception in the body, not just a clean exit.
"""
import pytest
import torch
import torch.nn as nn

import dataclasses

from experimentation.evaluation.ska_ablation import ablate_ska, ska_blocks
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.seq.ska_block import SKABlock

pytestmark = pytest.mark.correctness


class _Stack(nn.Module):
    """A model shaped like the ones the MQAR arm builds: SKA blocks nested
    inside a ModuleList, no `.ablate` method, mixed with non-SKA layers."""

    def __init__(self, n_ska=3, d_model=32):
        super().__init__()
        cfg = dataclasses.replace(
            KoopmanLMConfig(), d_model=d_model, ska_n_heads=4, ska_rank=16,
            ska_inverse_cholesky=True, ska_layerscale=True,
            ska_short_conv=False)
        self.embed = nn.Linear(d_model, d_model)
        self.seq_layers = nn.ModuleList([SKABlock(cfg) for _ in range(n_ska)])
        self.mlp_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_ska)])

    def forward(self, x):
        h = self.embed(x)
        for seq, mlp in zip(self.seq_layers, self.mlp_layers):
            h = seq(h)
            h = h + mlp(h)
        return h


def _model(n_ska=3, seed=0):
    torch.manual_seed(seed)
    m = _Stack(n_ska=n_ska)
    # LayerScale (and out_proj) start near zero, so an untrained SKA branch
    # contributes almost nothing and an ablation delta would be ~0 for the
    # wrong reason. Push the gate up so the branch is actually load-bearing.
    with torch.no_grad():
        for b in ska_blocks(m):
            if b.ska.layerscale_gate is not None:
                b.ska.layerscale_gate.fill_(1.0)
            b.ska.out_proj.weight.normal_(0.0, 0.3)
    return m


def test_every_ska_block_is_found_including_nested_ones():
    """Missing a block understates the delta, which is the direction that makes
    a policy look less load-bearing than it is."""
    m = _model(n_ska=3)
    found = ska_blocks(m)
    assert len(found) == 3
    assert all(isinstance(b, SKABlock) for b in found)
    assert {id(b) for b in found} == {id(b) for b in m.seq_layers}


def test_a_model_with_no_ska_blocks_yields_an_empty_list_rather_than_raising():
    """`mamba_only` is a legitimate baseline. An empty list lets the caller
    report `supported: False` instead of crashing an eval sweep."""
    assert ska_blocks(nn.Sequential(nn.Linear(4, 4))) == []


def test_the_context_manager_actually_changes_the_output():
    """The check that the flag is wired to the forward, not merely set."""
    m = _model()
    x = torch.randn(2, 16, 32)
    full = m(x)
    with ablate_ska(m):
        zeroed = m(x)
    assert not torch.allclose(full, zeroed), (
        "zeroing SKA changed nothing -- either the flag is not read or the "
        "branch contributes nothing and the measurement is vacuous")


def test_the_flags_are_restored_on_a_clean_exit():
    m = _model()
    with ablate_ska(m):
        assert all(b._ablate for b in ska_blocks(m))
    assert all(not b._ablate for b in ska_blocks(m))


def test_the_flags_are_restored_when_the_body_raises():
    """A flag left set turns every LATER measurement in the same process into an
    ablated one, so the next run's `full` accuracy would silently be its
    `zeroed` accuracy. That is a wrong number, not a crash."""
    m = _model()

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with ablate_ska(m):
            raise _Boom
    assert all(not b._ablate for b in ska_blocks(m))


def test_the_output_is_restored_after_the_context_exits():
    """The property that matters to a caller: measuring the ablation must not
    change the model. Asserted on the OUTPUT rather than on the flag, because
    the flag is the mechanism and the output is the contract."""
    m = _model()
    x = torch.randn(2, 16, 32)
    before = m(x)
    with ablate_ska(m):
        m(x)
    after = m(x)
    assert torch.equal(before, after)


def test_a_zeroed_block_is_a_pure_residual_passthrough():
    """`KoopmanLM.ablate`'s documented contract -- "output == input" -- restated
    for this walk, since a caller comparing two accuracies is entitled to assume
    the zeroed model is exactly the model without the branch."""
    m = _model(n_ska=1)
    block = ska_blocks(m)[0]
    x = torch.randn(2, 16, 32)
    with ablate_ska(m):
        assert torch.equal(block(x), x)


def test_nested_contexts_do_not_leave_the_flag_set():
    """Two nested ablations happen when a caller wraps a helper that also
    ablates. The inner exit must not clear a flag the outer context still
    needs, and the outer exit must clear it."""
    m = _model()
    with ablate_ska(m):
        with ablate_ska(m):
            assert all(b._ablate for b in ska_blocks(m))
        assert all(b._ablate for b in ska_blocks(m)), (
            "the inner exit cleared a flag the outer context still owns")
    assert all(not b._ablate for b in ska_blocks(m))
