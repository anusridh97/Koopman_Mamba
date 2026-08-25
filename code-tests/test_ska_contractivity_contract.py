"""The sqrt-beta contractivity contract: the acceptance artefact for the one
claim that licenses three of the four SKA backends to omit spectral
normalization entirely.

## Why this file exists

Three live code paths omit the spectral clamp entirely, on the strength of a
contractivity claim that until 2026-08-24 was stated in prose only -- and
stated WRONG, in a way this file's Part 1/Part 2 split is what corrected:

  * `kernels/inverse_cholesky.py` -- "the spectral clamp removed (contractive by
    the sqrt-beta convention)";
  * `kernels/incremental_transport.py` -- "Contractive A (||A||<=1) so no
    spectral normalization -- the sqrt-beta guarantee (do NOT clamp here)";
  * `kernels/prefix_scan.py` -- which omits the clamp with no comment at all.

All three attributed the bound to the SQUARE ROOT. It does not come from the
square root (Part 1 below), and the imprecision was not harmless: it told a
reader that `ska_beta_policy='linear'` or `'one'` would void the guarantee,
when both are equally contractive and equally legal on those routes. Those
comments now say "single key stream" and cite this file; the quotes above are
kept so the correction is legible rather than invisible.

That last one is the production path for `configs/50m.yaml`,
`configs/180m.yaml` and the three `configs/runs/50m-*.yaml` specs
(`ska_prefix_scan: true`); `configs/runs/proxy-256x17.yaml` takes the
inverse-Cholesky route. Six committed configs are therefore clamp-free.

An earlier version of this docstring said "no committed production config runs
`spec_w`". That is FALSE, and enumerating it is worth the space because the
answer is more interesting than the claim was. Ten committed configs still take
a clamped route -- meaning they set none of the three exact flags, i.e. they run
the CHUNKED APPROXIMATION:

    configs/1m.yaml            configs/370m.yaml     configs/1p5b.yaml
    configs/180m_dense.yaml    configs/440m.yaml     configs/3b.yaml
    configs/180m_gated.yaml    configs/880m.yaml
    configs/180m_v2.yaml       configs/runs/4m-golden.yaml

That is the entire large-scale ladder above 180m. They get the clamp, and they
also get the thing `SKAModule.__init__` warns about at construction: the chunked
route drops every within-chunk lag-1..lag-(S-1) cross-covariance term, measured
at ~100% relative error against a per-token-causal reference ON SHORT-RANGE
RECALL. So for those configs the clamp is not inert-but-harmless -- it is
attached to an operator that is not the one SKA is supposed to compute. Out of
scope here and flagged rather than fixed; verified by enumerating
`CONFIG_REGISTRY` plus `configs/runs/*.yaml` and checking
`ska_prefix_scan or ska_inverse_cholesky`.

Before this file the guarantee was pinned in exactly one place --
`test_inverse_cholesky.py::test_whitened_operator_is_contractive`, one path,
one seed, one beta distribution, and with no arm that could distinguish "the
convention is load-bearing" from "sigma happened to be below 1". A test with no
falsification arm cannot tell you the convention matters, only that it did not
hurt.

## The contract, in two parts

The bound is usually attributed to "sqrt-beta". That is imprecise, and the
imprecision matters because it hides which alternatives are safe. Written out:

    G_t = ridge*I + sum_{i<t} x_i x_i^T
    M_t =           sum_{i<t} x_i x_{i-1}^T
    A_w = L^-1 M L^-T,   G = L L^T

**Part 1 (the bound).** Let u_i = L^-1 x_i. Then
    sum_i u_i u_i^T = L^-1 (G - ridge*I) L^-T = I - ridge*G^-1 <= I,
so writing A_w = U_1 U_0^T with U_1 = [u_i]_{i in S}, U_0 = [u_{i-1}]_{i in S},
    sigma_max(A_w) <= sigma_max(U_1) sigma_max(U_0) <= 1
because each of U_1 U_1^T and U_0 U_0^T is a sub-sum of sum_i u_i u_i^T <= I
(every index appears at most once in each).

Note what Part 1 uses: **that both slots of M are drawn from the SAME key
stream whose Gram is G.** It says nothing about beta. So the bound holds for
sqrt(beta) weighting, for beta weighting, for beta == 1, and for no weighting
at all -- and it FAILS for the legacy two-stream form
M = sum beta_i z_i z_{i-1}^T against G = ridge*I + sum beta_i z_i z_i^T, where
the right slot's Gram is unweighted and can exceed G by 1/min(beta).

**Part 2 (why sqrt and not something else).** Single-stream is what buys the
bound; sqrt(beta) is what buys the SEMANTICS. With x = sqrt(beta) z,
    G = sum beta z z^T  and  C = sum beta v z^T,
i.e. the own-weight statistics are exactly the beta-weighted ones, so "beta is
the per-token write weight" remains literally true. Weighting both slots by
beta instead is equally contractive and silently changes the write weight to
beta^2.

Neither part alone picks the convention out. Both are tested below, because a
future author reading only Part 1 would conclude beta-in-both-slots is fine,
and a future author reading only Part 2 would conclude the bound comes from the
square root.

## The consequence for backends

Under any single-stream weighting the clamp is not merely unnecessary, it is
INERT: `spec_w` returns alpha == 1 identically (`test_spectral_clamp_is_inert`).
So the "spectral-normalization difference" between the four backends is a
difference in COST, not in the operator they compute. That is what makes the
clamp-free paths exact rather than approximate.

And the cost is not negligible. Measured at the proxy geometry (r=24, B=8,
T=1024, H=4, chunk 64), fp32 on 8 CPU threads -- so the RATIOS are indicative
and a GPU would redistribute them, but the shapes are the ones these routes
really build:

    route              n matrices   cholesky   whiten_M   spec_w    spec_w share
    chunked (B*nc*H)          512     0.97 ms    0.93 ms   2.76 ms       59%
    exact_intrachunk (B*T*H) 32768    68.7 ms    81.7 ms   72.2 ms       32%

`inverse_cholesky.py`'s header already records that "the 20-iteration power
iteration over B*T*H matrices was a dominant cost of the previous exact path",
which is the same observation from the other side. The recommendation that
follows is NOT "delete spec_w": it is the guard that would fire if a two-stream
weighting were ever reintroduced, and the falsification arm below shows what it
would be guarding against. It is that the clamped routes are paying a third to a
half of their whitening pipeline for a computation whose result is provably the
multiplicative identity, and that this is a cheap and safe thing to skip
BECAUSE the bound holds -- not in spite of not knowing whether it does.

All pure-torch fp64 linear algebra -> runs on CPU.
"""
import pytest
import torch

from koopman_lm.kernels.chunk_stats import (
    causal_normalize, chunk_stats, symmetric_key_value)
from koopman_lm.kernels.chunk_stats_exact import exact_stats
from koopman_lm.kernels.lin_alg import spec_w, whiten_M
from koopman_lm.kernels.prefix_scan import _advance_whitened_state

pytestmark = pytest.mark.correctness

DTYPE = torch.float64
RIDGE = 1e-2
CLIP = 4.0

#: The bound is exact, so the only slack allowed is fp64 roundoff in the
#: Cholesky + two triangular solves that produce A_w.
TOL = 1e-9


def _beta(kind, shape, generator):
    """The beta distributions worth separating.

    `ones` is not a curiosity: it is the `beta_policy: one` candidate, and the
    whole point of including it is that Part 1 predicts it is contractive.
    `bimodal` is the adversarial case -- beta alternating between ~0 and ~1 is
    what makes the two-stream form blow up hardest, because the ratio between
    G's weighting and the right slot's is maximal.
    """
    if kind == "ones":
        return torch.ones(*shape, dtype=DTYPE)
    if kind == "uniform":
        return torch.rand(*shape, generator=generator, dtype=DTYPE)
    if kind == "bimodal":
        draw = torch.rand(*shape, generator=generator, dtype=DTYPE)
        return (draw < 0.5).to(DTYPE) * (1.0 - 1e-8) + 1e-8
    if kind == "tiny":
        return torch.full(shape, 1e-6, dtype=DTYPE)
    raise AssertionError(f"unknown beta kind {kind!r}")


BETA_KINDS = ("ones", "uniform", "bimodal", "tiny")


def _stream(B=2, T=96, H=3, r=16, P=8, seed=0, clip=CLIP):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T, H, r, generator=g, dtype=DTYPE)
    zq = torch.randn(B, T, H, r, generator=g, dtype=DTYPE)
    v = torch.randn(B, T, H, P, generator=g, dtype=DTYPE)
    return causal_normalize(z, clip), causal_normalize(zq, clip), v, g


def _sigma_max(G, M):
    return float(torch.linalg.matrix_norm(whiten_M(torch.linalg.cholesky(G), M),
                                          ord=2).max())


# ---------------------------------------------------------------------------
# Part 1: the bound, on every path that relies on it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta_kind", BETA_KINDS)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_exact_prefix_stats_are_contractive(beta_kind, seed):
    """The inverse-Cholesky / exact-intrachunk statistics family.

    `exact_stats` is what `ska_inverse_cholesky: true` (the proxy-256x17
    reference route) and `ska_exact_intrachunk: true` both consume.
    """
    z_n, zq_n, v, g = _stream(seed=seed)
    beta = _beta(beta_kind, z_n.shape[:-1], g)
    x, vbar = symmetric_key_value(z_n, beta, v)
    G, M, _, _, _ = exact_stats(x, x, zq_n, vbar, RIDGE)
    sigma = _sigma_max(G, M)
    assert sigma <= 1.0 + TOL, (
        f"beta={beta_kind}: sigma_max(A_w) = {sigma:.9f} > 1. The "
        f"inverse-Cholesky and prefix-scan paths omit spec_w on the strength "
        f"of this bound, so a violation here means those paths are applying an "
        f"expansive operator K times with nothing to stop it.")


@pytest.mark.parametrize("beta_kind", BETA_KINDS)
@pytest.mark.parametrize("chunk_size", (8, 64))
def test_chunked_stats_are_contractive_including_the_boundary_term(
        beta_kind, chunk_size):
    """The chunked path, whose M is `within-chunk lag-1 + cross-chunk boundary`.

    Worth its own test rather than folding into the one above: `chunk_stats`
    builds M as two separate exclusive cumsums (`Mc` and `bnd`), and the bound
    holds only because their union covers each lag-1 pair exactly once. An
    off-by-one that double-counted a boundary pair would break the bound while
    leaving every other test in the suite green.
    """
    z_n, zq_n, v, g = _stream(T=200, seed=3)
    beta = _beta(beta_kind, z_n.shape[:-1], g)
    x, vbar = symmetric_key_value(z_n, beta, v)
    G, M, _, _, _ = chunk_stats(x, x, zq_n, vbar, RIDGE, chunk_size)
    sigma = _sigma_max(G, M)
    assert sigma <= 1.0 + TOL, (
        f"beta={beta_kind}, chunk={chunk_size}: sigma_max(A_w) = "
        f"{sigma:.9f} > 1 -- suspect the cross-chunk boundary term double "
        f"counts a lag-1 pair.")


@pytest.mark.parametrize("beta_kind", BETA_KINDS)
def test_prefix_scan_compact_state_is_contractive(beta_kind):
    """The compact whitened state {L, A, R} the prefix-scan path carries.

    This is the path with NO justifying comment and, before this file, no test
    -- and it is the one `configs/50m.yaml` and `configs/180m.yaml` run. It is
    also the strictest version of the claim: A is not recomputed from G and M
    at each token, it is TRANSPORTED by replaying Givens rotations
    (`_transport_operator`), so the bound has to survive r rotations per token
    for the whole stream rather than holding for one freshly-solved matrix.
    """
    N, T, r, P = 6, 96, 16, 8
    g = torch.Generator().manual_seed(4)
    z = causal_normalize(torch.randn(N, T, r, generator=g, dtype=DTYPE), CLIP)
    v = torch.randn(N, T, P, generator=g, dtype=DTYPE)
    beta = _beta(beta_kind, (N, T), g)
    x, vbar = symmetric_key_value(z, beta, v)

    L = (RIDGE ** 0.5) * torch.eye(r, dtype=DTYPE).expand(N, r, r).contiguous()
    A = torch.zeros(N, r, r, dtype=DTYPE)
    R = torch.zeros(N, P, r, dtype=DTYPE)
    hprev = torch.zeros(N, r, dtype=DTYPE)
    hasprev = torch.zeros(N, dtype=torch.bool)
    worst = 0.0
    for t in range(T):
        L, A, R, hprev, hasprev = _advance_whitened_state(
            L, A, R, hprev, hasprev, x[:, t], vbar[:, t])
        worst = max(worst, float(torch.linalg.matrix_norm(A, ord=2).max()))
    assert worst <= 1.0 + TOL, (
        f"beta={beta_kind}: worst sigma_max(A) = {worst:.9f} > 1 over the "
        f"stream. `_read_state` applies A power_k times with no clamp.")


# ---------------------------------------------------------------------------
# The falsification arm. Without this, the tests above are vacuous.
# ---------------------------------------------------------------------------

def _write_gate_sequence(B=2, T=200, H=3, r=16, P=8, delta=0.01, seed=5):
    """The sequence a TRAINED write gate produces: alternating near-zero and
    near-one beta along two noisy directions -- beta~0 distractors, beta~1
    facts.

    This shape is not decoration. `test_gated_blend_oracle.py::alt_sequence`
    records why, and the finding is reproduced below: under i.i.d. uniform beta
    the two-stream convention only reaches sigma ~ 4.6, because a random beta
    rarely puts a heavily-downweighted key next to a heavily-upweighted one.
    A trained gate does exactly that, systematically. So the adversarial case
    for the legacy convention is also the REALISTIC one, and a falsification
    arm built on uniform noise would understate the violation by more than an
    order of magnitude.
    """
    g = torch.Generator().manual_seed(seed)
    e1 = torch.zeros(r, dtype=DTYPE)
    e1[0] = 1.0
    e2 = torch.zeros(r, dtype=DTYPE)
    e2[1] = 1.0
    even = torch.arange(T) % 2 == 0
    base = torch.where(even.view(1, T, 1, 1), e1.view(1, 1, 1, r),
                       e2.view(1, 1, 1, r)).expand(B, T, H, r)
    z = base + 0.05 * torch.randn(B, T, H, r, generator=g, dtype=DTYPE)
    z = z / z.norm(dim=-1, keepdim=True)
    beta = torch.where(even.view(1, T, 1),
                       torch.tensor(delta, dtype=DTYPE),
                       torch.tensor(1.0, dtype=DTYPE)).expand(B, T, H)
    zq = causal_normalize(torch.randn(B, T, H, r, generator=g, dtype=DTYPE), CLIP)
    v = torch.randn(B, T, H, P, generator=g, dtype=DTYPE)
    return z, zq, v, beta.contiguous()


def test_legacy_two_stream_keys_violate_the_bound_and_worsen_as_the_gate_sharpens():
    """The legacy asymmetric convention: M = sum beta_i z_i z_{i-1}^T.

    THE POINT OF THIS FILE. Every assertion above is `sigma <= 1`, and a bound
    that is never approached from the other side tells you nothing about
    whether the thing being tested is doing any work. Here the SAME statistics
    kernel, fed the SAME keys under the pre-v1.1 calling convention
    (`zb = beta*z` in one slot, raw `z` in the other), produces sigma two
    orders of magnitude above 1 -- so the bound is a property of the
    CONVENTION and not of the kernel, the data, or the ridge.

    Related but not redundant: `test_gated_blend_oracle.py::
    test_c1_contractivity_and_sqrt_beta_dependency` makes the same sym-vs-asym
    comparison in a standalone NumPy reimplementation of the statistics. This
    one runs it through the PRODUCT kernel (`chunk_stats_exact.exact_stats`),
    which is what `ska_inverse_cholesky` and `ska_exact_intrachunk` actually
    call -- so a divergence between the oracle and the shipped kernel shows up
    here and nowhere else.

    This is also the concrete reason `koopman_lm/config.py` refuses a
    two-stream beta policy on a clamp-free backend: at these sigmas, A^K with
    K=1..2 and no clamp is an expansive map, and the run does not fail loudly,
    it just trains something else.
    """
    # Sharper gate -> larger violation, because the mechanism is the RATIO
    # between G's weighting and M's unweighted right factor. Asserting the
    # trend rather than one threshold: a single number invites tuning until it
    # passes, whereas "it diverges as the gate sharpens" is the mechanism, and
    # the only way to satisfy it is for the mechanism to be real.
    measured = {}
    for delta in (1e-1, 1e-2, 1e-3):
        z_n, zq_n, v, beta = _write_gate_sequence(delta=delta)

        # The SAME keys and the SAME beta, both conventions, one kernel.
        x_sym, v_sym = symmetric_key_value(z_n, beta, v)
        G_sym, M_sym, _, _, _ = exact_stats(x_sym, x_sym, zq_n, v_sym, RIDGE)
        sigma_sym = _sigma_max(G_sym, M_sym)

        # zb into the LEFT slot, raw z into the right: G stays sum beta z z^T
        # but M's right factor is now unweighted, so the Cauchy-Schwarz step
        # loses its second bound.
        zb = beta.unsqueeze(-1) * z_n
        G_asym, M_asym, _, _, _ = exact_stats(z_n, zb, zq_n, v, RIDGE)
        measured[delta] = (sigma_sym, _sigma_max(G_asym, M_asym))

        assert sigma_sym <= 1.0 + TOL, (
            f"delta={delta:.0e}: the symmetric convention must hold on the "
            f"ADVERSARIAL sequence too, not just on random beta; got "
            f"sigma_max = {sigma_sym:.9f}")

    sigmas = [measured[d][1] for d in (1e-1, 1e-2, 1e-3)]
    assert sigmas[0] < sigmas[1] < sigmas[2], (
        f"the two-stream violation must GROW as the write gate sharpens "
        f"(delta 1e-1 -> 1e-3); measured {sigmas}. If it does not, the "
        f"emulation of the legacy convention has drifted from what "
        f"chunk_stats' docstring describes.")
    assert sigmas[-1] > 20.0, (
        f"at delta=1e-3 the two-stream convention gave sigma_max = "
        f"{sigmas[-1]:.4f}, which is not a decisive violation. Either the "
        f"emulation has drifted, or the single-stream property is not what the "
        f"bound rests on -- and in the second case every clamp-free backend "
        f"needs its justification rewritten.")


def test_uniform_beta_understates_the_two_stream_violation():
    """Why the arm above uses a structured sequence, pinned as a fact.

    Under i.i.d. uniform beta the legacy convention violates the bound only
    mildly. Recorded as an assertion rather than a comment because it is the
    reason a reviewer should not "simplify" the adversarial sequence away: a
    falsification arm built on uniform noise still passes a `> 1` check, so the
    simplification would look harmless and would quietly cost most of the
    test's sensitivity.
    """
    z_n, zq_n, v, g = _stream(seed=5)
    beta = _beta("uniform", z_n.shape[:-1], g)
    G, M, _, _, _ = exact_stats(z_n, beta.unsqueeze(-1) * z_n, zq_n, v, RIDGE)
    sigma = _sigma_max(G, M)
    assert 1.0 < sigma < 10.0, (
        f"uniform-beta two-stream sigma_max = {sigma:.6f}; the point of this "
        f"test is that it is ABOVE 1 but far below what a trained gate's "
        f"alternating pattern produces.")


# ---------------------------------------------------------------------------
# Part 2: sqrt(beta) specifically, and what it buys that beta does not.
# ---------------------------------------------------------------------------

def test_beta_in_both_slots_is_also_contractive_but_squares_the_write_weight():
    """Single-stream is what buys the bound; sqrt is what buys the semantics.

    Weighting both slots by beta (rather than sqrt(beta)) is still a single key
    stream, so Part 1 still applies and the operator is still contractive. What
    it silently changes is G: the own-weight statistic becomes sum beta^2 z z^T,
    so "beta is the per-token write weight" stops being true and every
    downstream reading of beta -- the diagnostics' `beta_mean`, the decode
    path's state, any ablation of the gate -- is describing beta^2.

    Asserting BOTH halves in one test on purpose: they are the two premises of
    the same conclusion, and separating them invites someone to satisfy one and
    call the convention justified.
    """
    z_n, zq_n, v, g = _stream(T=48, seed=6)
    beta = _beta("uniform", z_n.shape[:-1], g)

    x_sqrt, v_sqrt = symmetric_key_value(z_n, beta, v)
    x_lin = beta.unsqueeze(-1) * z_n

    G_sqrt, M_sqrt, _, _, _ = exact_stats(x_sqrt, x_sqrt, zq_n, v_sqrt, RIDGE)
    G_lin, M_lin, _, _, _ = exact_stats(x_lin, x_lin, zq_n, v, RIDGE)

    # Half one: both are contractive, so the bound does NOT single out sqrt.
    assert _sigma_max(G_lin, M_lin) <= 1.0 + TOL
    assert _sigma_max(G_sqrt, M_sqrt) <= 1.0 + TOL

    # Half two: only sqrt(beta) makes G the beta-weighted Gram. Compare against
    # the explicit beta-weighted and beta^2-weighted Grams built without the
    # helper, so the test does not check symmetric_key_value against itself.
    G_beta, _, _, _, _ = exact_stats(
        z_n, beta.unsqueeze(-1) * z_n, zq_n, v, RIDGE)
    G_beta2, _, _, _, _ = exact_stats(
        z_n, (beta ** 2).unsqueeze(-1) * z_n, zq_n, v, RIDGE)
    assert torch.allclose(G_sqrt, G_beta, atol=1e-12), \
        "sqrt(beta) in both slots must give G = ridge*I + sum beta z z^T"
    assert torch.allclose(G_lin, G_beta2, atol=1e-12), \
        "beta in both slots gives G = ridge*I + sum beta^2 z z^T"


# ---------------------------------------------------------------------------
# The consequence: the clamp is inert, so the backends agree on the operator.
# ---------------------------------------------------------------------------

def test_which_committed_configs_still_take_a_clamped_route():
    """The enumeration in this module's docstring, pinned so it cannot go stale.

    A route is clamp-free iff `ska_prefix_scan or ska_inverse_cholesky`; setting
    neither (nor `ska_exact_intrachunk`) is the CHUNKED approximation, which both
    keeps `spec_w` and drops the within-chunk cross-covariance terms.

    Pinned as data rather than described in prose because it is exactly the kind
    of claim `test_docs_are_not_stale.py` exists for: a count and a membership
    list. If someone switches `440m.yaml` to the prefix scan, this fails with the
    old and new sets side by side, which is the right way to learn that the
    docstring above needs editing.
    """
    from koopman_lm.config import CONFIG_REGISTRY, build_config

    expected_clamped = {
        "180m_dense", "180m_gated", "180m_v2", "1m", "1p5b", "370m", "3b",
        "440m", "880m",
    }
    clamped = {
        name for name in CONFIG_REGISTRY
        if not (build_config(name).ska_prefix_scan
                or build_config(name).ska_inverse_cholesky)
    }
    assert clamped == expected_clamped, (
        f"the set of registry configs on a clamped (chunked) route changed.\n"
        f"  now:      {sorted(clamped)}\n"
        f"  expected: {sorted(expected_clamped)}\n"
        f"Update this test AND the enumeration in this module's docstring -- and "
        f"note that a config moving OFF this list is good news (it gains an "
        f"exact route), while one moving ON is a regression.")
    # And the complement, stated so the pass is not vacuous if the registry
    # shrinks: the canonical production sizes must stay clamp-free.
    assert {"50m", "180m"}.isdisjoint(clamped), (
        "50m and 180m are the canonical production configs and must keep an "
        "exact route")


@pytest.mark.parametrize("stats", ("chunked", "exact"))
@pytest.mark.parametrize("beta_kind", BETA_KINDS)
def test_spectral_clamp_is_inert_under_single_stream_keys(beta_kind, stats):
    """`spec_w` returns exactly alpha == 1, so the clamped and clamp-free
    backends compute the SAME operator.

    This is the load-bearing fact behind "the backend spectral-normalization
    differences are justified". They are not differences in what is computed;
    the chunked and exact-intrachunk routes pay for a 20-iteration power
    iteration over (B*nc*H) or (B*T*H) matrices whose result is provably 1.

    `alpha == 1` EXACTLY, not approximately: `spec_w` is
    `1/clamp(sigma, min=1)`, so any sigma <= 1 maps to exactly 1.0 in floating
    point. A failure here therefore means sigma > 1 -- the same contract
    violation the tests above check, observed through the clamp.
    """
    z_n, zq_n, v, g = _stream(T=200, seed=7)
    beta = _beta(beta_kind, z_n.shape[:-1], g)
    x, vbar = symmetric_key_value(z_n, beta, v)
    # BOTH statistics families, because the docstring above reasons about
    # (B*nc*H) AND (B*T*H) and the two routes that pay for the clamp are one of
    # each: `chunk_stats` feeds the chunked route's `ska_core`, `exact_stats`
    # feeds exact-intrachunk's `ska_core_given_L`. Logically the second follows
    # from the contractivity tests plus alpha = 1/clamp(sigma, min=1), but the
    # claim is about a specific call and it costs one parametrize to check it.
    if stats == "chunked":
        G, M, _, _, _ = chunk_stats(x, x, zq_n, vbar, RIDGE, 64)
    else:
        G, M, _, _, _ = exact_stats(x, x, zq_n, vbar, RIDGE)
    alpha = spec_w(whiten_M(torch.linalg.cholesky(G), M))
    assert torch.all(alpha == 1.0), (
        f"beta={beta_kind}, stats={stats}: spec_w fired on "
        f"{int((alpha != 1.0).sum())} of "
        f"{alpha.numel()} operators (min alpha "
        f"{float(alpha.min()):.9f}). Under a single key stream it cannot, so "
        f"either the convention broke or the clamp is now doing something the "
        f"clamp-free backends are not.")
