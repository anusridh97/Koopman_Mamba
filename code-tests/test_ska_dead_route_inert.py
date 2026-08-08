"""Characterization test: SKAModule's legacy post-Cholesky route is inert.

`SKAModule.forward` calls `kernels.chunk_stats.chunk_stats` directly (or one of
the prefix_scan / inverse_cholesky / exact_intrachunk kernels). It never routed
through the old `_get_chunk_stats` -> `_post_cholesky_{pytorch,triton}` chain,
so the knobs that only ever selected within that chain -- `chunk_strategy`,
`overlap_fraction`, `decay_alpha` -- could not affect any output.

That is exactly WHY that chain was deletable (see the commit that removed it).
This test locks the property in permanently: it is the tripwire for someone
re-wiring one of these knobs into the live forward, or resurrecting a second
post-Cholesky implementation, without saying so. Equality is exact, not
`allclose` -- an inert knob must be bit-inert, and any real re-wiring would
perturb the last bits at minimum.

Deliberately spans every forward branch, because "dead in the chunked path" is
a weaker claim than "dead in all four".

`backend` is NOT in the inert set: it looks like a sibling knob but it is still
forwarded raw to `ska_prefix_scan` (whose vocabulary is auto/cuda/cuda_prefix/
reference/pytorch -- `backend='triton'` raises there). Only the eagerly
resolved `self.backend` triton/pytorch string is vestigial, and it survives for
`extra_repr` alone; that is asserted separately below.
"""
from contextlib import nullcontext

import pytest
import torch

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness

# One per forward branch in SKAModule.forward.
BRANCHES = {
    "chunked": dict(),
    "prefix_scan": dict(prefix_scan=True, prefix_scan_block_size=8),
    "exact_intrachunk": dict(exact_intrachunk=True),
    "inverse_cholesky": dict(inverse_cholesky=True),
}

# Knob settings that used to steer the legacy chain. 'overlap'/'decay' warn at
# construction (they are documented as ignored), hence the warns filter below.
INERT_KNOBS = [
    dict(chunk_strategy="overlap", overlap_fraction=0.25),
    dict(chunk_strategy="overlap", overlap_fraction=0.75),
    dict(chunk_strategy="decay", decay_alpha=0.5),
    dict(overlap_fraction=0.9, decay_alpha=0.1),
]


def _forward(seed, **kw):
    """Build + run an SKAModule deterministically. Returns (output, grads).

    Init is reseeded per call so two builds with different inert knobs share
    identical weights -- otherwise the comparison would only test the RNG.
    """
    torch.manual_seed(1234)
    m = SKAModule(d_model=32, n_heads=2, rank=16, chunk_size=8, **kw)
    x = torch.randn(2, 20, 32, generator=torch.Generator().manual_seed(seed))
    y = m(x)
    y.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()
             if p.grad is not None}
    return y.detach(), grads


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_legacy_route_knobs_do_not_change_forward_or_grads(branch):
    base_out, base_grads = _forward(99, **BRANCHES[branch])
    # out_proj is zero-init only when layerscale=False; guard against a
    # vacuous pass where every tensor compared is identically zero.
    assert base_out.abs().sum().item() > 0.0

    for knobs in INERT_KNOBS:
        ctx = (pytest.warns(RuntimeWarning) if knobs.get("chunk_strategy")
               else nullcontext())
        with ctx:
            out, grads = _forward(99, **BRANCHES[branch], **knobs)
        assert torch.equal(out, base_out), \
            f"{branch}: {knobs} changed the forward output"
        assert grads.keys() == base_grads.keys()
        for name, g in grads.items():
            assert torch.equal(g, base_grads[name]), \
                f"{branch}: {knobs} changed grad of {name}"


def test_resolved_backend_string_is_reprs_only_consumer():
    """`self.backend` is vestigial but load-bearing for `extra_repr`.

    The raw constructor argument is still live (it reaches ska_prefix_scan via
    `_prefix_scan_backend`); it is only the triton/pytorch resolution of it that
    no longer selects any code. Pinning that keeps a future cleanup honest
    about which half of `backend` is actually dead.
    """
    m = SKAModule(d_model=32, n_heads=2, rank=16, backend="pytorch")
    assert m.backend == "pytorch"
    assert f"backend={m.backend}" in m.extra_repr()
    # raw string preserved unresolved, so prefix_scan gets its own 'auto'
    assert SKAModule(d_model=32, n_heads=2, rank=16)._prefix_scan_backend == "auto"


def test_removed_legacy_symbols_stay_removed():
    """The deleted cluster's only remaining trace should be absence.

    Re-adding any of these names is how the `M G^-1` variant (which the live
    kernels do NOT compute -- they use the whitened L^-1 M L^-T) would creep
    back in, along with its 6-iteration spectral normalization where the live
    path uses `lin_alg.spec_w`'s 20.
    """
    import koopman_lm.modules.seq.ska as ska

    for name in ("_post_cholesky_pytorch", "_post_cholesky_triton",
                 "_get_chunk_stats", "_spectral_normalize_power_iter",
                 "_compute_chunk_stats_and_cholesky", "_power_spectral_filter",
                 "_fused_filter_readout_kernel"):
        assert not hasattr(ska, name), (
            f"{name} is back in ska.py; it was removed as unreachable dead "
            "code computing a different matrix than the live path")
    assert not hasattr(SKAModule, "_get_chunk_stats")


def test_no_cholesky_solve_in_ska_module():
    """`torch.cholesky_solve` in ska.py meant the dead `M G^-1` form.

    The live path never inverts G explicitly -- it whitens (`lin_alg.whiten_M`,
    triangular solves against L). A cholesky_solve reappearing here is the
    specific numerics regression this deletion was meant to prevent.
    """
    from pathlib import Path

    import koopman_lm.modules.seq.ska as ska

    src = Path(ska.__file__).read_text()
    assert "cholesky_solve" not in src
