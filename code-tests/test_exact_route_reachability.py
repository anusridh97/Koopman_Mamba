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


# ----------------------------------------- memory, which the assert does not ----

def _invchol_stat_gib(c, batch=1):
    """fp32 bytes for the per-token statistics `inverse_cholesky` materialises:
    G, M and P = L^-1 at (B,T,H,r,r) plus Cv at (B,T,H,P,r), per SKA layer."""
    H, r, T = c.ska_n_heads, c.ska_rank, c.max_seq_len
    P = c.d_model // H
    return batch * T * H * (3 * r * r + P * r) * 4 / 1024 ** 3


def test_the_rank_cap_is_not_the_binding_constraint_at_370m():
    """`370m` passes the assert and still cannot realistically run: rank 64 at
    seq 8192 is tens of GiB of per-token statistics per SKA layer at BATCH 1,
    before activations. `rank <= 64` is a correctness/indexing bound, not a
    feasibility one, and reading it as permission is the trap this test names."""
    c = _cfg("370m")
    assert c.ska_rank <= INVCHOL_RANK_CAP           # the assert lets it through
    per_layer = _invchol_stat_gib(c)
    assert per_layer > 4.0, (
        f"370m invchol stats measured at {per_layer:.1f} GiB/layer at batch 1; "
        f"this test asserts the constraint is real")
    # and it is dominated by rank, not by depth: the four flippable configs are
    # an order of magnitude cheaper per layer
    for cheap in ("1m", "180m_dense"):
        assert _invchol_stat_gib(_cfg(cheap)) < per_layer / 2


def test_the_four_flippable_configs_are_actually_cheap():
    """The other half of the same point: for `1m` and the 180m variants the
    exact route is not merely legal, it is small. If a flip happens, these are
    the configs where nothing has to be traded."""
    for name in ("1m", "180m_dense", "180m_gated", "180m_v2"):
        gib = _invchol_stat_gib(_cfg(name))
        assert gib < 8.0, f"{name}: {gib:.1f} GiB/layer at batch 1"
