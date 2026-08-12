"""Regression: stream_write's lazy import must resolve.

`LastLayerRidgeMemory.stream_write` does a function-level
`from koopman_lm.kernels.cholesky_update import update_L_only`. That path is a
leftover from the old flat echo-ska-440m layout and does not exist in the
package layout. Because the import is lazy, nothing catches it until
stream_write is actually called -- and it IS called, from
models/koopman_lm.py and models/recurrent.py.

NOTE ON THE CURRENT FAILURE MODE: until the module reorg lands, this test
fails on `No module named 'koopman_lm.modules'` -- the NEW home of the
memory class, which does not exist yet. That is a different error from the
stale-import bug above. Both are resolved by the reorg. The import is done
inside the test body rather than at module scope so that this transitional
failure stays a single failing test instead of aborting collection for the
whole suite.
"""
import pytest
import torch


@pytest.mark.correctness
def test_stream_write_import_resolves():
    # Imported inside the test on purpose -- see the module docstring.
    from koopman_lm.modules.wip.memory import LastLayerRidgeMemory

    # Signatures verified against the real class:
    #   __init__(self, d_model, rank=64, ridge=0.01, ...)
    #   stream_reset(self, batch=1, device=None, dtype=torch.float32)
    #   stream_write(self, h, v, weight=1.0)
    # NOTE: the reset kwarg is `batch`, NOT `batch_size`.
    r, d, B = 4, 6, 2
    mem = LastLayerRidgeMemory(d_model=d, rank=r)
    mem.stream_reset(batch=B, device=torch.device("cpu"), dtype=torch.float32)

    h = torch.randn(B, d)
    v = torch.randn(B, d)

    # Before the fix this raises ModuleNotFoundError, not an assertion failure.
    mem.stream_write(h, v, weight=1.0)
