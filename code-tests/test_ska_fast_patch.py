"""Regression coverage for patch_ska_module (structural-review issue 1).

patch_ska_module (used by training/train.py's opt-in --ska_fast flag) had
ZERO test coverage before this: an untested monkey-patch is a failure mode
this codebase has already suffered (see
code-tests/test_streaming_memory_imports.py's docstring for another
instance). The investigation
(docs/superpowers/specs/2026-08-07-structural-review.md, issue 1) confirmed
by direct execution that patch_ska_module is bit-exact (forward and
state-dict round-trip) against a CPU-only SKAModule; this test locks that
in and must keep passing whether patch_ska_module lives in kernels/fast.py
or modules/seq/fast.py.
"""
import copy

import pytest
import torch

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness


def _make_module(seed=0):
    torch.manual_seed(seed)
    return SKAModule(
        d_model=32, n_heads=2, rank=8, head_dim=4, power_K=2,
        chunk_size=8, backend='pytorch', eta_learnable=False, eta_value=1.0,
        gamma_learnable=False, gamma_value=1.0, layerscale=True,
    )


def test_patch_ska_module_forward_is_bit_exact():
    from koopman_lm.modules.seq.fast import patch_ska_module

    torch.manual_seed(1)
    m_plain = _make_module(seed=0)
    m_patched = copy.deepcopy(m_plain)
    patch_ska_module(m_patched)

    x = torch.randn(2, 17, 32)
    with torch.no_grad():
        y_plain = m_plain(x)
        y_patched = m_patched(x)

    err = (y_plain - y_patched).abs().max().item()
    assert err == 0.0, f"patched vs unpatched forward max diff {err:.3e}"


def test_patch_ska_module_state_dict_round_trip():
    from koopman_lm.modules.seq.fast import patch_ska_module

    m_plain = _make_module(seed=0)
    m_patched = copy.deepcopy(m_plain)
    patch_ska_module(m_patched)

    # patched state_dict keeps the ORIGINAL key_proj/query_proj/value_proj
    # names (checkpoint compatibility), not fused_proj.
    sd = m_patched.state_dict()
    assert 'fused_proj.weight' not in sd
    for name in ('key_proj.weight', 'query_proj.weight', 'value_proj.weight'):
        assert name in sd

    m_reloaded = _make_module(seed=0)
    m_reloaded.load_state_dict(sd)

    x = torch.randn(2, 17, 32)
    with torch.no_grad():
        y_patched = m_patched(x)
        y_reloaded = m_reloaded(x)

    err = (y_patched - y_reloaded).abs().max().item()
    assert err == 0.0, f"reloaded-into-plain vs patched forward max diff {err:.3e}"
