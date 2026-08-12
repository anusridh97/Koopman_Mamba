"""The fused k/q/v projection in SKAModule.forward must equal three separate GEMMs.

Replaces test_ska_fast_patch.py. That file tested `patch_ska_module`, which
monkey-patched a SECOND forward over a live SKAModule; the fusion now lives in
`forward()` itself, so there is no second implementation left to compare against
and nothing that can drift out of sync.

What is still worth pinning is the one non-obvious part: the slice offsets into
the concatenated weight have to match the cat order, and `H*r` vs `H*P` differ
whenever head_dim != rank. Get that wrong and the model trains on a silently
permuted projection -- no exception, just worse loss. So assert the block-matmul
identity directly:

    F.linear(x, cat([W_k, W_q, W_v]))[..., slice_i]  ==  proj_i(x)

Why the old test could not catch the drift it was written for: its `_make_module`
built the module with the default prefix_scan=False, i.e. the chunked branch --
the only one of the four backends the patched forward actually implemented. Both
production configs (configs/50m.yaml:26, configs/180m.yaml:25) set
ska_prefix_scan: true. The parametrization below covers all four so that a future
change to the projection is checked against every path that consumes it.

Tolerance: derived, not exact -- and the reason is worth recording, because the
first version of this test asserted torch.equal and CI disproved it.

Splitting one (2*H*r + H*P, d_model) GEMM into three narrower ones changes which
BLAS kernel and blocking each shape dispatches to, which can reorder a
length-d_model accumulation. The bound is ~d_model * eps(fp32) = 32 * 1.19e-7
~= 3.8e-6 relative.

Two measurements:
  * On the Marlowe login node (CPU fp32, d_model=32) the fused forward and the
    pre-fusion three-GEMM forward are BIT-identical -- torch.equal True,
    max|diff| 0.0, on all four backends, with weights asserted equal first.
  * On GitHub's ubuntu-latest runner they are NOT: value_proj differs by
    max|diff| 4.768e-07 (2^-21), i.e. ~2.4e-7 relative on values of magnitude
    ~2. Same dtype, same device type, different BLAS.

So bit-identity is a property of a particular CPU + BLAS build, not of CPU fp32,
and asserting it makes this test machine-dependent. 1e-5 relative sits ~2.6x
above the observed CI value and below the 3.8e-6 analytic bound's own margin,
while staying orders of magnitude tighter than a wrong slice offset, which is
O(1) wrong rather than O(1e-7).
"""
import pytest
import torch
import torch.nn.functional as F

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness

# See module docstring: derived from d_model * eps(fp32), with margin over the
# 4.768e-07 observed on GitHub's runner. NOT exact -- bit-identity holds on some
# CPU/BLAS combinations and not others.
GEMM_SPLIT_REL_TOL = 1e-5

# head_dim != rank on purpose: makes a k/q vs v offset mix-up observable.
D_MODEL, N_HEADS, RANK, HEAD_DIM = 32, 2, 8, 4

BACKENDS = [
    pytest.param({}, id="chunked"),
    pytest.param({"prefix_scan": True, "prefix_scan_block_size": 8}, id="prefix_scan"),
    pytest.param({"inverse_cholesky": True}, id="inverse_cholesky"),
    pytest.param({"exact_intrachunk": True}, id="exact_intrachunk"),
]


def _make_module(seed=0, **kw):
    torch.manual_seed(seed)
    return SKAModule(
        d_model=D_MODEL, n_heads=N_HEADS, rank=RANK, head_dim=HEAD_DIM,
        power_K=2, chunk_size=8, backend='pytorch',
        eta_learnable=False, eta_value=1.0,
        gamma_learnable=False, gamma_value=1.0, layerscale=True, **kw)


def test_fused_projection_equals_three_separate_gemms():
    """The cat + single GEMM, sliced, reproduces each individual projection."""
    m = _make_module()
    H, r, P = m.H, m.rank, m.P
    x = torch.randn(2, 17, D_MODEL)

    with torch.no_grad():
        fused_w = torch.cat([m.key_proj.weight, m.query_proj.weight,
                             m.value_proj.weight], dim=0)
        combined = F.linear(x, fused_w)

        for name, proj, lo, hi in (
            ("key",   m.key_proj,   0,          H * r),
            ("query", m.query_proj, H * r,      2 * H * r),
            ("value", m.value_proj, 2 * H * r,  2 * H * r + H * P),
        ):
            got = combined[..., lo:hi]
            want = proj(x)
            scale = want.abs().max().clamp_min(torch.finfo(want.dtype).eps)
            err = ((got - want).abs().max() / scale).item()
            assert err < GEMM_SPLIT_REL_TOL, (
                f"{name}_proj slice [{lo}:{hi}] differs from {name}_proj(x) by "
                f"rel {err:.3e} (tol {GEMM_SPLIT_REL_TOL:.3e}). A value this "
                "large is a wrong slice offset or cat order, not GEMM reordering "
                "-- reordering lands around 1e-7 (see module docstring).")


def test_fused_slice_widths_cover_the_whole_projection_exactly():
    """No gap and no overlap: the three slices must tile [0, 2*H*r + H*P)."""
    m = _make_module()
    H, r, P = m.H, m.rank, m.P
    fused_w = torch.cat([m.key_proj.weight, m.query_proj.weight,
                         m.value_proj.weight], dim=0)
    assert fused_w.shape[0] == 2 * H * r + H * P
    assert m.key_proj.weight.shape[0] == H * r
    assert m.query_proj.weight.shape[0] == H * r
    assert m.value_proj.weight.shape[0] == H * P
    # the case a single shared width would mask
    assert H * r != H * P, "test config must keep rank != head_dim to be meaningful"


@pytest.mark.parametrize("kw", BACKENDS)
def test_forward_runs_and_is_finite_on_every_backend(kw):
    """All four forward strategies consume the fused projection.

    The retired patch implemented only the chunked branch, so a prefix_scan
    model silently fell back to the chunked approximation. One forward means
    every backend sees the same projection; this pins that they all still run.
    """
    m = _make_module(**kw)
    x = torch.randn(2, 24, D_MODEL)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 24, D_MODEL)
    assert torch.isfinite(y).all(), f"non-finite output on {kw or 'chunked'}"


def test_projections_remain_three_separate_parameters():
    """Checkpoint and decode-path compatibility.

    The fusion is a call-time cat, NOT a stored fused weight: models/recurrent.py
    calls ska.key_proj(h) directly for step-by-step decode, and every existing
    checkpoint has key_proj/query_proj/value_proj keys. If someone later replaces
    these with slices of one parameter, both break -- so pin it here.
    """
    m = _make_module()
    keys = set(m.state_dict())
    for name in ("key_proj.weight", "query_proj.weight", "value_proj.weight"):
        assert name in keys, f"{name} missing -- checkpoints and decode would break"
    assert not any("fused" in k for k in keys), (
        "a fused weight leaked into state_dict; the cat must stay call-time")
    # the decode path's actual call pattern
    x = torch.randn(1, 3, D_MODEL)
    with torch.no_grad():
        assert m.key_proj(x).shape == (1, 3, m.H * m.rank)
        assert m.value_proj(x).shape == (1, 3, m.H * m.P)
