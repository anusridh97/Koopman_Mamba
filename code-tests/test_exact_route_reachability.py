"""Which configs on the chunked route COULD take an exact one -- as data.

`test_ska_contractivity_contract.py::test_which_committed_configs_still_take_a_
clamped_route` pins WHICH nine registry configs run the chunked approximation. It
does not ask the next question, and the next question is the one that decides what
can be done about it: for each of those nine, is there an exact route it could
actually run?

The answer is not "yes, flip a flag", and that matters enough to be machine-
checked rather than described. Three independent gates:

  `inverse_cholesky` asserts `rank <= 64`. It materialises per-token (r x r)
    statistics, so its memory is (B,T,H,r,r) -- QUADRATIC in rank and LINEAR in
    sequence length, against the chunked path's (B,T/S,H,r,r).
  the fused CUDA prefix scan requires rank == 24 AND value width == 64
    (`cuda_prefix_scan._RANK` / `._VALUE`). Off that geometry `ska_prefix_scan`
    falls back to the Python reference scan, measured at 137x-160x chunked.
  `exact_intrachunk` runs at any geometry and costs 57x-87x chunked (space.py,
    job 440135), which is not a training route.

So of the nine: four (`1m` and the three 180m variants) can take
`inverse_cholesky` at a few GiB; `370m` sits exactly at the rank cap and needs
tens of GiB of per-token statistics at batch 1; and four (`440m`, `880m`, `1p5b`,
`3b`) are REFUSED by the assert and are ineligible for the fused kernel too. For
those four there is no exact route that could train them at their current rank --
which makes "flip the ladder to invchol" unavailable as a uniform action, and
makes the ladder's RANK, not its route flag, the upstream question.

Pinned as data because it is exactly the kind of claim that goes stale silently: a
change to `_RANK`, to the assert's bound, or to any config's `ska_rank` moves it,
and the recommendation built on top of it would still read as though it held.
"""

import pathlib
import sys
import warnings

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from koopman_lm.config import CONFIG_REGISTRY, build_config  # noqa: E402
from koopman_lm.kernels.cuda_prefix_scan import _RANK, _VALUE  # noqa: E402

#: The documented bound in `SKAModule.__init__`'s assert. Duplicated here so a
#: change to it fails this test rather than silently widening the matrix below.
INVCHOL_RANK_CAP = 64


def _cfg(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_config(name)


def _chunked(name):
    c = _cfg(name)
    return not (c.ska_prefix_scan or c.ska_inverse_cholesky
                or c.ska_exact_intrachunk)


CHUNKED = sorted(n for n in CONFIG_REGISTRY if _chunked(n))


def test_the_assert_bound_this_file_reasons_about_is_still_the_real_one():
    """The matrix below is only true while the assert says 64."""
    src = (REPO / "koopman_lm/modules/seq/ska.py").read_text()
    assert f"rank <= {INVCHOL_RANK_CAP}" in src, (
        "SKAModule's inverse_cholesky rank assert changed; the reachability "
        "matrix in this file was reasoned about at <= 64 and must be redone")


def test_the_chunked_set_is_the_one_the_contractivity_contract_pins():
    """Two files must not disagree about which configs are affected."""
    assert set(CHUNKED) == {
        "180m_dense", "180m_gated", "180m_v2", "1m", "1p5b", "370m", "3b",
        "440m", "880m"}, CHUNKED


# --------------------------------------------------------- the three gates ----

#: config -> (can take inverse_cholesky, eligible for the FUSED prefix scan)
EXPECTED = {
    "1m":         (True,  False),   # rank 24 but value width 32
    "180m_dense": (True,  False),   # rank 48
    "180m_gated": (True,  False),
    "180m_v2":    (True,  False),
    "370m":       (True,  False),   # rank 64 -- exactly at the cap
    "440m":       (False, False),   # rank 96 -- REFUSED
    "880m":       (False, False),   # rank 96 -- REFUSED
    "1p5b":       (False, False),   # rank 128 -- REFUSED
    "3b":         (False, False),   # rank 128 -- REFUSED
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_exact_route_reachability(name):
    c = _cfg(name)
    invchol_ok = c.ska_rank <= INVCHOL_RANK_CAP
    fused_ok = (c.ska_rank == _RANK and c.d_model // c.ska_n_heads == _VALUE)
    assert (invchol_ok, fused_ok) == EXPECTED[name], (
        f"{name}: rank={c.ska_rank} value_width={c.d_model // c.ska_n_heads}; "
        f"got invchol={invchol_ok} fused={fused_ok}, expected {EXPECTED[name]}. "
        f"A config gaining an exact route is good news -- update this table and "
        f"the module docstring. Losing one is a regression.")


def test_four_of_the_nine_have_no_feasible_exact_route():
    """The finding that decides what can be done. `inverse_cholesky` is refused
    by the assert and the fused kernel does not apply, so the only exact options
    left are the reference prefix scan (137x-160x) and exact_intrachunk
    (57x-87x) -- neither of which trains a 440M-3B model."""
    stranded = sorted(n for n in CHUNKED
                      if not EXPECTED[n][0] and not EXPECTED[n][1])
    assert stranded == ["1p5b", "3b", "440m", "880m"], stranded
    for n in stranded:
        assert _cfg(n).ska_rank > INVCHOL_RANK_CAP


def test_no_chunked_config_can_use_the_fused_kernel():
    """The cheap exact route (1.01x chunked) is reachable by exactly the two
    configs that are ALREADY exact. That is why the chunked nine are chunked."""
    assert not [n for n in CHUNKED if EXPECTED[n][1]]
    fused_capable = sorted(
        n for n in CONFIG_REGISTRY
        if _cfg(n).ska_rank == _RANK
        and _cfg(n).d_model // _cfg(n).ska_n_heads == _VALUE)
    assert fused_capable == ["180m", "50m"], fused_capable
    for n in fused_capable:
        assert _cfg(n).ska_prefix_scan, f"{n} can use the fused kernel; it should"


# --------------------------- cost and memory, which the assert does not cover ----

#: MEASURED on an H100, one SKA layer, fwd+bwd, batch 1, each config at its OWN
#: max_seq_len -- job 446319, via scripts/measure_chunked_route_cost.py.
#: {config: (invchol time / chunked time, invchol peak GiB)}.
#:
#: An earlier version of this file ESTIMATED the memory as
#: `T*H*(3r^2 + P*r)*4` bytes, counting G, M, P=L^-1 and Cv. That undercounts by
#: roughly 2.5x -- `exact_stats` plus `SKACoreInvChol` keep the prefix-sum
#: contributions AND the sums AND the jittered copy AND the whitened W live for
#: the backward, nearer 7r^2 + 3Pr -- and two conclusions here rested on the
#: undercount. Measured peaks replace it: a number from the machine cannot be
#: wrong about which tensors autograd retains.
MEASURED_INVCHOL = {
    "1m":         (1.61, 0.46),
    "50m":        (5.60, 1.57),
    "180m":       (6.51, 2.58),
    "180m_dense": (10.20, 9.37),
    "180m_gated": (10.21, 9.37),
    "180m_v2":    (10.21, 9.37),
    "370m":       (12.57, 20.56),
}


def test_the_rank_cap_is_not_the_binding_constraint_at_370m():
    """`370m` passes the assert and still cannot realistically train: 20.6 GiB of
    invchol working set for ONE SKA layer at BATCH 1, and it has 7 SKA layers.
    `rank <= 64` is a correctness/indexing bound, not a feasibility one, and
    reading it as permission is the trap this test names."""
    c = _cfg("370m")
    assert c.ska_rank <= INVCHOL_RANK_CAP           # the assert lets it through
    _ratio, peak = MEASURED_INVCHOL["370m"]
    n_ska = len(c.ska_layer_indices) if c.ska_layer_indices else c.n_layers
    assert peak > 16.0, f"370m invchol peak {peak} GiB/layer at batch 1"
    assert peak * n_ska > 80.0, (
        f"370m: {n_ska} SKA layers x {peak} GiB = {peak*n_ska:.0f} GiB at batch "
        f"1, past an 80GB H100 before activations or optimizer state")


def test_only_1m_is_cheap_enough_that_nothing_has_to_be_traded():
    """The distinction the estimate blurred. `1m` is 1.61x chunked and 0.46 GiB
    for one SKA layer, of which it has 2 -- essentially free, and it is the Echo
    section 4.1 config both curve goldens record. The three 180m variants are
    legal but 10.2x and 9.4 GiB per SKA layer: affordable, not free. Calling all
    four "small" (as this file did while it estimated the memory) would have made
    a 10x slowdown sound like a flag flip."""
    assert MEASURED_INVCHOL["1m"] < (2.0, 1.0), MEASURED_INVCHOL["1m"]
    for name in ("180m_dense", "180m_gated", "180m_v2"):
        ratio, peak = MEASURED_INVCHOL[name]
        assert ratio > 5.0, f"{name} invchol ratio {ratio} -- not free"
        assert peak > 5.0, f"{name} invchol peak {peak} GiB -- not small"
    assert all(n in MEASURED_INVCHOL for n in CHUNKED
               if EXPECTED[n][0]), "every invchol-legal config needs a measurement"
    assert not [n for n in CHUNKED if not EXPECTED[n][0] and n in MEASURED_INVCHOL], (
        "a config the assert refuses cannot have an invchol measurement")


def test_the_rank_cap_is_enforced_behaviourally_not_just_in_prose():
    """`ska.py` now contains the string "rank <= 64" in five places and only one
    of them is the assert, so a textual guard would stay green if the assert
    alone changed. Exercise it."""
    from koopman_lm.modules.seq.ska import SKAModule
    kw = dict(d_model=1024, n_heads=16, inverse_cholesky=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SKAModule(rank=INVCHOL_RANK_CAP, **kw)          # at the cap: allowed
        with pytest.raises(AssertionError, match="not supported"):
            SKAModule(rank=INVCHOL_RANK_CAP + 1, **kw)  # one past it: refused
