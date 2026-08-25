"""Zero the SKA branch of any model, including the ones without `.ablate`.

`models/koopman_lm.py::KoopmanLM.ablate` is a method on that class.
`experiments/mqar_finetune.py` does not use that class -- it builds through
`models/baselines.py::build_mamba_ska_swiglu`, whose `_build_model` returns a
locally-defined `_Model` with no `ablate` at all. `quick_eval` already handles
that by degrading (`if ska_ablation and hasattr(model, "ablate")`), which on a
baselines model silently means no ablation is measured.

That is acceptable for LM eval and not acceptable for the exponent arm, where
SKA-on versus SKA-zeroed retrieval is the PRIMARY discriminator: held-out LM loss
is proven non-responsive to SKA damage in this repo (18 trials destroyed 44% of
SKA's contribution and moved mean loss by -1.6e-4, far below the 7.54e-3 floor),
while the ablation delta separates the same conditions at t = 7-22.

So this is one walk over `model.modules()` that finds `SKABlock` wherever it is
nested, usable on `KoopmanLM`, on a baselines `_Model`, and on anything else.
`KoopmanLM.ablate` is left alone: it also handles `zero_mamba` and the four-mode
decomposition, and two implementations of the same flag would be one drift away
from disagreeing about what "ablated" means.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List

from koopman_lm.modules.seq.ska_block import SKABlock


def ska_blocks(model) -> List[SKABlock]:
    """Every `SKABlock` in `model`, however deeply nested. `[]` if there are none.

    An empty list rather than an error: `mamba_only` is a legitimate baseline, and
    a caller sweeping several model types should be able to report
    `supported: False` for it instead of crashing the sweep.
    """
    if not hasattr(model, "modules"):
        return []
    return [m for m in model.modules() if isinstance(m, SKABlock)]


@contextmanager
def ablate_ska(model):
    """Inside the block, every SKA layer is a pure residual passthrough.

    Restores the PREVIOUS value of each flag rather than hard-clearing it, so a
    nested ablation's exit cannot clear a flag the enclosing context still owns.
    Restoration is in a `finally`, because a flag left set turns every later
    measurement in the same process into an ablated one -- the next run's `full`
    accuracy would silently be its `zeroed` accuracy, which is a wrong number
    rather than a crash.

    Yields the blocks it touched, so a caller can assert it found any at all
    instead of reporting a 0.0 delta for a model that has no SKA in it.
    """
    blocks = ska_blocks(model)
    previous = [b._ablate for b in blocks]
    try:
        for b in blocks:
            b._ablate = True
        yield blocks
    finally:
        for b, was in zip(blocks, previous):
            b._ablate = was
