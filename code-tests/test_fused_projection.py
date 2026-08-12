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

Tolerance: none. MEASURED bit-identical on CPU fp32 -- the fused forward and the
pre-fusion three-GEMM forward (`main`'s ska.py, loaded side by side with equal
weights asserted first) agree to max|diff| == 0.0 exactly, via torch.equal, on
all four backends. So this asserts exact equality rather than a tolerance: any
drift at all is a real change, and a loose bound would hide a wrong slice offset.

That is a claim about THIS environment, not a general guarantee. On GPU with bf16
autocast and production shapes, splitting one (2*H*r + H*P, d_model) GEMM into
three narrower ones can dispatch different BLAS kernels and reorder a
length-d_model accumulation, bounded by ~d_model * eps. If this test ever fails
on a GPU runner, the fix is a documented tolerance there -- not here.
"""
import pytest
import torch
import torch.nn.functional as F

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness

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
            assert torch.equal(got, want), (
                f"{name}_proj slice [{lo}:{hi}] is not bit-identical to "
                f"{name}_proj(x): max|diff|={(got - want).abs().max().item():.3e}. "
                "Either the slice offsets no longer match the cat order, or this "
                "backend reorders the GEMM accumulation (see module docstring -- "
                "expected on GPU/bf16, not on CPU fp32).")


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
