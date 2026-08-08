"""Regression coverage for patch_ska_module (structural-review issue 1).

patch_ska_module (used by training/train.py's opt-in --ska_fast flag) had
ZERO test coverage before this: an untested monkey-patch is a failure mode
this codebase has already suffered (see
code-tests/test_streaming_memory_imports.py's docstring for another
instance). The investigation
(docs/superpowers/specs/2026-08-07-structural-review.md, issue 1) confirmed
by direct execution that patch_ska_module reproduces the plain SKAModule's
forward (and its state-dict round-trip) to within fp32 rounding; this test
locks that in and must keep passing whether patch_ska_module lives in
kernels/fast.py or modules/seq/fast.py.

NOTE on tolerance (was asserted bit-exact == 0.0, which is not a valid
invariant and failed nondeterministically in CI): _FusedProjSlice/
patch_ska_module concatenate the key/query/value weight matrices along
their OUTPUT dimension and slice the fused GEMM's output back apart -- the
contraction length (d_model) for any given output element is unchanged, so
this is a mathematical identity, not a different computation. What *can*
differ is which BLAS kernel/blocking a given GEMM shape dispatches to,
which can round a length-n dot product's accumulation differently by up to
~O(n) ULPs depending on summation order (n = d_model = 32 here, so a
pessimistic per-GEMM bound is n * eps(fp32) ~= 32 * 1.19e-7 ~= 3.8e-6
relative). That perturbation then flows through causal_normalize /
chunk_stats / ska_core -- a ridge-regularized pipeline with power_K=2, not
an ill-conditioned inverse -- so it should not blow up by more than a small
constant factor; empirically (see CI failure logs) the propagated relative
error is ~1e-6, i.e. the pipeline itself does not amplify much beyond the
single-GEMM rounding bound above.

The output tensors span several orders of magnitude in this small config
(elements from ~1e-8 to ~1e-4), so a plain elementwise-relative check is
unstable near zero (tiny denominators) and a fixed absolute tolerance would
either be meaningless at the top of the range or too tight at the bottom.
Instead we normalize the max abs difference by the reference tensor's own
peak magnitude -- the right scale for "did two GEMM orderings of the same
math produce the same answer". FUSED_PROJ_REL_TOL=1e-4 gives >1 order of
magnitude of margin over the pessimistic single-GEMM bound (3.8e-6) and
~2 orders of magnitude over the empirically observed CI value (~1e-6), so
it is a derived tolerance with margin, not "bumped until CI's number
passes".
"""
import copy

import pytest
import torch

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness

# See module docstring for the derivation.
FUSED_PROJ_REL_TOL = 1e-4


def _make_module(seed=0):
    torch.manual_seed(seed)
    return SKAModule(
        d_model=32, n_heads=2, rank=8, head_dim=4, power_K=2,
        chunk_size=8, backend='pytorch', eta_learnable=False, eta_value=1.0,
        gamma_learnable=False, gamma_value=1.0, layerscale=True,
    )


def _rel_err(reference, actual):
    """Max abs difference, normalized by the reference's own peak magnitude
    (see module docstring: elementwise-relative is unstable here since
    these outputs span ~1e-8 to ~1e-4)."""
    scale = reference.abs().max().clamp_min(torch.finfo(reference.dtype).eps)
    return ((reference - actual).abs().max() / scale).item()


def test_patch_ska_module_forward_matches_within_fp32_tolerance():
    from koopman_lm.modules.seq.fast import patch_ska_module

    torch.manual_seed(1)
    m_plain = _make_module(seed=0)
    m_patched = copy.deepcopy(m_plain)
    patch_ska_module(m_patched)

    x = torch.randn(2, 17, 32)
    with torch.no_grad():
        y_plain = m_plain(x)
        y_patched = m_patched(x)

    err = _rel_err(y_plain, y_patched)
    assert err < FUSED_PROJ_REL_TOL, (
        f"patched vs unpatched forward rel diff {err:.3e} "
        f"(tol {FUSED_PROJ_REL_TOL:.3e})"
    )


def test_patch_ska_module_state_dict_round_trip_matches_within_fp32_tolerance():
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

    err = _rel_err(y_patched, y_reloaded)
    assert err < FUSED_PROJ_REL_TOL, (
        f"reloaded-into-plain vs patched forward rel diff {err:.3e} "
        f"(tol {FUSED_PROJ_REL_TOL:.3e})"
    )
