"""Decomposing the write gate into its KEY exponent and its VALUE exponent.

## What was conflated, and why it matters

`ska_beta_policy` had two learned-gate cells, and each moved TWO things at once:

    learned   key = sqrt(beta) z    value = sqrt(beta) v
    linear    key = beta      z     value = beta      v

So `learned` vs `linear` is a comparison of (sqrt, sqrt) against (linear,
linear) -- a diagonal of a 2x2, with the off-diagonal missing. Any difference
between them is attributable to the key exponent, or to the value exponent, or
to an interaction, and the design cannot tell those apart. That is the whole
reason `linear` grokked 3/3 on MQAR while `learned` grokked 1/3 and nobody could
say which half of the change did it.

This file adds the two mixed cells that complete the square:

    key_linear_value_sqrt   (C)   key = beta      z    value = sqrt(beta) v
    key_sqrt_value_linear   (D)   key = sqrt(beta) z   value = beta      v

with `one` (key = z, value = v) retained as the no-gate control.

`head_scalar` is deliberately NOT extended into this decomposition: seeds spent
re-litigating it are seeds not spent on the confirmation wave. Note that the
usual justification -- "0/3 grokked on MQAR" -- is job 446106 at ~4000 steps, and
job 446145 groks it at step 12000 with an SKA ablation delta of +0.9746. So the
exclusion is a budget decision, not a finding about the policy. It remains a
fully supported policy in the module; only the arm omits it.

## The mathematical claim these tests are here to VERIFY, not assume

Contractivity of `W = L^-1 M L^-T` (with `L L^T = G`) depends only on `G` and
`M`. Both are built ENTIRELY from the key stream:

    G = ridge*I + sum_i x_i x_i^T
    M =           sum_i x_i x_{i-1}^T

The value weighting enters only through

    C = sum_i vbar_i x_i^T

which `ska_core` applies AFTER the whitened operator. Therefore:

  * **The value exponent cannot affect the spectral bound at all.** It does not
    appear in either matrix the bound is a statement about.
  * **C and D are both contractive** whenever the key stream is single-sourced,
    which both are -- C puts `beta z` in both slots of M, D puts `sqrt(beta) z`
    in both slots. The bound follows from the Cauchy-Schwarz argument in
    `test_ska_contractivity_contract.py`, which never mentions beta.

So the square root is NOT what buys contractivity. What it buys is that
`G = sum beta z z^T` and `C = sum beta v z^T`, i.e. that beta is the LITERAL
per-token write weight. Under a linear key exponent `G` scales as `beta^2`
instead, which silently redefines the write weight as its square -- and, at the
shared `beta = 0.5` init, halves the own-weight against a fixed ridge. That last
consequence is a confound, not a feature, and `test_the_key_exponent_changes_the
_effective_ridge_at_init` below measures its size so the screening design can
control for it.

Every spectral assertion here reads a TRUE singular value
(`torch.linalg.matrix_norm(..., ord=2)`, which is an SVD), never a power
iteration: a power iteration that has not converged understates sigma_max, and
understating it is the direction that would make a broken policy look safe.

## The retired asymmetric convention stays retired

`test_ska_contractivity_contract.py` measures the two-stream form
(`zb = beta z` against raw `z`) at 22.8x over the bound. Neither policy added
here is that form: both put ONE stream into both slots of M. That is asserted
directly in `test_neither_mixed_policy_is_the_retired_two_stream_form`, because
"key exponent" and "asymmetric keys" are close enough in words to be confused in
code.
"""
import dataclasses
import math
from pathlib import Path

import pytest
import torch

from koopman_lm.config import (
    BETA_POLICIES as CONFIG_BETA_POLICIES, KoopmanLMConfig, build_config,
    config_hash)
from koopman_lm.kernels.chunk_stats import causal_normalize
from koopman_lm.kernels.chunk_stats_exact import exact_stats
from koopman_lm.kernels.lin_alg import whiten_M
from koopman_lm.modules.seq.ska import BETA_POLICIES, SKAModule

pytestmark = pytest.mark.correctness

REPO_ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float64

#: The two cells this file adds, and the exponent each slot gets. `1.0` means
#: "multiply by beta", `0.5` means "multiply by sqrt(beta)", `0.0` means
#: "leave alone".
MIXED = {
    "key_linear_value_sqrt": (1.0, 0.5),
    "key_sqrt_value_linear": (0.5, 1.0),
}

#: The complete 2x2 plus the control, as (key_exponent, value_exponent). This is
#: the design the file exists to make expressible.
EXPONENTS = {
    "learned": (0.5, 0.5),
    "linear": (1.0, 1.0),
    "key_linear_value_sqrt": (1.0, 0.5),
    "key_sqrt_value_linear": (0.5, 1.0),
    "one": (0.0, 0.0),
}


def _module(policy, d_model=32, n_heads=4, rank=16, seed=0):
    torch.manual_seed(seed)
    mod = SKAModule(d_model, n_heads, rank=rank, inverse_cholesky=True,
                    beta_policy=policy, precision='fp64')
    return mod.double()


def _hidden(B=2, T=12, d_model=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d_model, generator=g, dtype=DTYPE)


def _sharpened(policy, seed=5, scale=2.0):
    """A module whose gate is NOT at its 0.5 initialisation.

    Every policy here is zero-initialised so an untrained gate emits a constant
    0.5, under which `beta` and `sqrt(beta)` differ only by a constant factor
    and a per-token exponent test would pass for the wrong reason. `scale=2.0`
    on the projection drives beta across most of (0,1).
    """
    mod = _module(policy, seed=seed)
    with torch.no_grad():
        if mod.beta_proj is not None:
            mod.beta_proj.weight.normal_(0.0, scale)
            mod.beta_proj.bias.normal_(0.0, scale)
        if getattr(mod, "beta_logit", None) is not None:
            mod.beta_logit.normal_(0.0, scale)
    return mod


def _stream(mod, B=2, T=64, seed=6, clip=4.0):
    """(beta, z_n, zq_n, v) for `mod`, with the keys causally normalised."""
    h = _hidden(B=B, T=T, d_model=mod.beta_proj.in_features
                if mod.beta_proj is not None else 32)
    beta = mod._resolve_beta(h).detach()
    g = torch.Generator().manual_seed(seed)
    z = causal_normalize(
        torch.randn(B, T, mod.H, mod.rank, generator=g, dtype=DTYPE), clip)
    zq = causal_normalize(
        torch.randn(B, T, mod.H, mod.rank, generator=g, dtype=DTYPE), clip)
    v = torch.randn(B, T, mod.H, mod.P, generator=g, dtype=DTYPE)
    return beta, z, zq, v


def _dense_stats(x, vbar, ridge):
    """G, M, C by explicit exclusive-prefix summation. Returns (B,T,H,...).

    Written out with einsum rather than by calling `exact_stats`, so the tests
    below compare the production kernel against an independent statement of the
    definition instead of against itself.
    """
    B, T, H, r = x.shape
    P = vbar.shape[-1]
    eye = torch.eye(r, dtype=x.dtype)
    G = torch.zeros(B, T, H, r, r, dtype=x.dtype)
    M = torch.zeros(B, T, H, r, r, dtype=x.dtype)
    C = torch.zeros(B, T, H, P, r, dtype=x.dtype)
    for t in range(T):
        G[:, t] = ridge * eye
        if t == 0:
            continue
        xp = x[:, :t]                                        # (B,t,H,r)
        vp = vbar[:, :t]
        G[:, t] = G[:, t] + torch.einsum('bthr,bths->bhrs', xp, xp)
        if t > 1:
            M[:, t] = torch.einsum('bthr,bths->bhrs', xp[:, 1:], xp[:, :-1])
        C[:, t] = torch.einsum('bthp,bthr->bhpr', vp, xp)
    return G, M, C


# ---------------------------------------------------------------------------
# The cells exist, are declared in both places, and mean what they say.
# ---------------------------------------------------------------------------

def test_both_mixed_cells_are_declared_in_both_policy_lists():
    """`config.py` and `modules/seq/ska.py` each keep their own copy on purpose
    (config must import without the model code). A cell present in one and not
    the other fails at construction on a GPU rather than at validation."""
    for policy in MIXED:
        assert policy in BETA_POLICIES, f"{policy} missing from ska.BETA_POLICIES"
        assert policy in CONFIG_BETA_POLICIES, (
            f"{policy} missing from config.BETA_POLICIES")
    assert BETA_POLICIES == CONFIG_BETA_POLICIES


def test_the_two_by_two_is_complete_in_the_PRODUCTION_exponent_table():
    """The design claim, read off `ska.BETA_EXPONENTS` rather than off this
    file's own `EXPONENTS`.

    An earlier version of this test asserted properties of `EXPONENTS`, a
    constant defined thirty lines above it -- so it passed unchanged if the
    production table were wrong, or if both mixed cells were deleted. That is a
    test of a literal, not of the code.
    """
    from koopman_lm.modules.seq.ska import BETA_EXPONENTS
    gated = {p: BETA_EXPONENTS[p] for p in BETA_EXPONENTS if p != "one"}
    # Every corner of {sqrt, linear} x {sqrt, linear} is reachable in production.
    assert set(gated.values()) == {(0.5, 0.5), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)}
    # ... and each mixed corner is reachable by EXACTLY ONE policy, so a result
    # attributes to a named cell rather than to a set of aliases.
    for corner in ((1.0, 0.5), (0.5, 1.0)):
        owners = [p for p, e in gated.items() if e == corner]
        assert len(owners) == 1, f"{corner} has owners {owners}"
    assert BETA_EXPONENTS["key_linear_value_sqrt"] == (1.0, 0.5)
    assert BETA_EXPONENTS["key_sqrt_value_linear"] == (0.5, 1.0)
    # The production table must cover every declared policy, or a legal config
    # value would KeyError inside `_weight_key_value` on a GPU.
    assert set(BETA_EXPONENTS) == set(BETA_POLICIES)


def test_head_scalar_is_excluded_from_the_ARM_not_from_the_module():
    """The exclusion is a design decision about seed budget, not a claim that the
    policy is illegal. So `head_scalar` must remain a working policy in the
    module while being absent from the arm's cell list -- and the assertion
    belongs against the cell list, which is where the decision lives.
    """
    from experimentation.experiments.beta_exponent_cells import CELLS
    assert "head_scalar" in BETA_POLICIES, (
        "head_scalar is excluded from the arm, not removed from the repo")
    assert all(c.policy != "head_scalar" for c in CELLS)


def test_this_files_exponent_table_agrees_with_the_production_one():
    """`EXPONENTS` is a readable restatement of the design, used to parametrise
    the tests below. It must not be allowed to drift from what the module does --
    a test fixture that disagrees with production tests the fixture.

    `one` is the one deliberate difference and it is recorded as such: production
    maps it to (0.5, 0.5) because beta == 1 makes every exponent agree, while
    this file writes (0.0, 0.0) to say "no weighting at all". The equivalence is
    asserted rather than assumed.
    """
    from koopman_lm.modules.seq.ska import BETA_EXPONENTS
    for policy, pair in EXPONENTS.items():
        if policy == "one":
            assert BETA_EXPONENTS["one"] == (0.5, 0.5)
            mod = _module("one")
            beta = mod._resolve_beta(_hidden())
            assert torch.all(beta == 1.0), (
                "the (0.0,0.0) vs (0.5,0.5) discrepancy is only harmless while "
                "beta is identically 1")
            continue
        assert BETA_EXPONENTS[policy] == pair, policy


@pytest.mark.parametrize("policy,key_exp,val_exp",
                         [(p, *MIXED[p]) for p in sorted(MIXED)])
def test_each_mixed_policy_applies_its_declared_exponents(policy, key_exp, val_exp):
    """The whole content of the two new cells, checked against beta directly
    rather than against the other policy's helper."""
    mod = _sharpened(policy)
    beta, z, _, v = _stream(mod, T=16)
    x, vbar = mod._weight_key_value(z, beta, v)
    w_key = beta.clamp_min(0).pow(key_exp).unsqueeze(-1)
    w_val = beta.clamp_min(0).pow(val_exp).unsqueeze(-1)
    assert torch.allclose(x, w_key * z), f"{policy}: key exponent wrong"
    assert torch.allclose(vbar, w_val * v), f"{policy}: value exponent wrong"
    # And the gate really did move off 0.5, or the assertions above hold
    # trivially for both exponents at once.
    assert beta.std() > 0.05, "gate not sharpened; the test would be vacuous"


def test_the_mixed_policies_share_the_learned_gate_and_its_initialisation():
    """C and D must differ from `learned` ONLY in an exponent. If they built a
    different gate, or started from a different beta, the MQAR contrast would be
    confounded exactly the way the original learned-vs-linear contrast was."""
    ref = _module("learned", seed=3)
    for policy in MIXED:
        mod = _module(policy, seed=3)
        assert mod.beta_proj is not None, f"{policy} must keep the learned gate"
        assert mod.beta_proj.weight.shape == ref.beta_proj.weight.shape
        assert mod.beta_proj.bias.shape == ref.beta_proj.bias.shape
        # Zero-init -> beta = sigmoid(0) = 0.5, identical to `learned` and
        # `linear`, so all four cells start from the same beta.
        h = _hidden()
        assert torch.allclose(mod._resolve_beta(h),
                              torch.full_like(mod._resolve_beta(h), 0.5))
        assert torch.allclose(mod._resolve_beta(h), ref._resolve_beta(h))


def test_the_default_policy_is_still_learned():
    """Backward compatibility: adding cells must not change what an existing
    config means."""
    assert KoopmanLMConfig().ska_beta_policy == "learned"
    assert build_config("1m").ska_beta_policy == "learned"


# ---------------------------------------------------------------------------
# G, M and C under each policy -- the dense statement of the decomposition.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", sorted(EXPONENTS))
def test_G_M_and_C_match_a_dense_reference_under_every_policy(policy):
    """The production kernel against an independent einsum of the definition.

    This is the check that a policy wired into `_weight_key_value` but wrong in
    a kernel would fail, and it is run for the existing cells too so the
    reference is validated on known-good policies.
    """
    mod = _sharpened(policy)
    beta, z, zq, v = _stream(mod, T=24)
    x, vbar = mod._weight_key_value(z, beta, v)
    ridge = 1e-2
    Gf, Mf, Cf, _, (B, T, H, P) = exact_stats(x, x, zq, vbar, ridge)
    r = mod.rank
    # exact_stats symmetrises G and adds a fixed 1e-4 jitter; match it so this
    # compares the statistics rather than the regulariser.
    Gd, Md, Cd = _dense_stats(x, vbar, ridge + 1e-4)
    assert torch.allclose(Gf.reshape(B, T, H, r, r), Gd, atol=1e-10), f"{policy}: G"
    assert torch.allclose(Mf.reshape(B, T, H, r, r), Md, atol=1e-10), f"{policy}: M"
    assert torch.allclose(Cf.reshape(B, T, H, P, r), Cd, atol=1e-10), f"{policy}: C"


def test_G_and_M_depend_only_on_the_key_exponent():
    """The load-bearing half of the mathematical claim.

    C (key-linear/value-sqrt) and `linear` (key-linear/value-linear) share a key
    exponent, so they must produce IDENTICAL G and M -- and therefore an
    identical whitened operator -- differing only in C. Likewise D and `learned`.

    If this fails, the "value exponent cannot affect the spectral bound"
    argument is not about this code.
    """
    ridge = 1e-2
    for mixed, same_key in [("key_linear_value_sqrt", "linear"),
                            ("key_sqrt_value_linear", "learned")]:
        a, b = _sharpened(mixed, seed=7), _sharpened(same_key, seed=7)
        with torch.no_grad():
            b.beta_proj.weight.copy_(a.beta_proj.weight)
            b.beta_proj.bias.copy_(a.beta_proj.bias)
        beta, z, zq, v = _stream(a, T=32)
        assert torch.allclose(beta, b._resolve_beta(
            _hidden(B=2, T=32)).detach()), "gates must agree"
        xa, va = a._weight_key_value(z, beta, v)
        xb, vb = b._weight_key_value(z, beta, v)
        assert torch.allclose(xa, xb), f"{mixed} vs {same_key}: keys must match"
        Ga, Ma, Ca, _, _ = exact_stats(xa, xa, zq, va, ridge)
        Gb, Mb, Cb, _, _ = exact_stats(xb, xb, zq, vb, ridge)
        assert torch.allclose(Ga, Gb, atol=1e-12), f"{mixed} vs {same_key}: G"
        assert torch.allclose(Ma, Mb, atol=1e-12), f"{mixed} vs {same_key}: M"
        # ... and C is where they differ, or the pair is not a real contrast.
        assert not torch.allclose(Ca, Cb), (
            f"{mixed} vs {same_key}: C identical too -- these are the same cell")


def test_C_depends_only_on_the_value_exponent_given_the_key_stream():
    """The mirror claim. D (key-sqrt/value-linear) and `linear`
    (key-linear/value-linear) share a VALUE exponent, so for a FIXED key stream
    their C agrees; they differ because the key stream differs.

    Stated on a fixed key stream because C = sum vbar x^T touches both, so
    "depends only on the value exponent" is only true holding x fixed. Writing
    the test this way makes the conditional explicit instead of asserting
    something false.
    """
    d = _sharpened("key_sqrt_value_linear", seed=9)
    lin = _sharpened("linear", seed=9)
    with torch.no_grad():
        lin.beta_proj.weight.copy_(d.beta_proj.weight)
        lin.beta_proj.bias.copy_(d.beta_proj.bias)
    beta, z, zq, v = _stream(d, T=32)
    _, vbar_d = d._weight_key_value(z, beta, v)
    _, vbar_lin = lin._weight_key_value(z, beta, v)
    assert torch.allclose(vbar_d, vbar_lin), "shared value exponent -> same vbar"
    # Fix the key stream to `learned`'s, and C must then agree between the two.
    x_fixed = beta.clamp_min(0).sqrt().unsqueeze(-1) * z
    _, _, C_d, _, _ = exact_stats(x_fixed, x_fixed, zq, vbar_d, 1e-2)
    _, _, C_lin, _, _ = exact_stats(x_fixed, x_fixed, zq, vbar_lin, 1e-2)
    assert torch.allclose(C_d, C_lin, atol=1e-12)


# ---------------------------------------------------------------------------
# Contractivity, by true SVD, for every cell in the design.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", sorted(EXPONENTS))
def test_every_cell_keeps_the_whitened_operator_contractive(policy):
    """sigma_max(L^-1 M L^-T) <= 1, read off an SVD.

    The reference config runs `ska_inverse_cholesky: true`, which omits the
    spectral clamp entirely, so an expansive policy would not raise -- it would
    silently apply an expansive operator K times and report a loss.
    """
    mod = _sharpened(policy)
    beta, z, zq, v = _stream(mod, T=64)
    x, vbar = mod._weight_key_value(z, beta, v)
    G, M, _, _, _ = exact_stats(x, x, zq, vbar, 1e-2)
    W = whiten_M(torch.linalg.cholesky(G), M)
    sigma = float(torch.linalg.matrix_norm(W, ord=2).max())
    assert sigma <= 1.0 + 1e-9, f"{policy}: sigma_max = {sigma:.12f} > 1"


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_the_value_exponent_cannot_change_sigma_max(policy):
    """The claim stated as an experiment: swap ONLY the value exponent and
    sigma_max must be bit-identical, because neither G nor M sees vbar.

    A test that merely checked `sigma <= 1` for both would pass even if the
    value weighting leaked into G -- it would just report a different number
    under the bound. This one would not.
    """
    key_exp, _ = MIXED[policy]
    partner = "linear" if key_exp == 1.0 else "learned"
    a, b = _sharpened(policy, seed=11), _sharpened(partner, seed=11)
    with torch.no_grad():
        b.beta_proj.weight.copy_(a.beta_proj.weight)
        b.beta_proj.bias.copy_(a.beta_proj.bias)
    beta, z, zq, v = _stream(a, T=64)
    sigmas = []
    for mod in (a, b):
        x, vbar = mod._weight_key_value(z, beta, v)
        G, M, _, _, _ = exact_stats(x, x, zq, vbar, 1e-2)
        sigmas.append(torch.linalg.matrix_norm(
            whiten_M(torch.linalg.cholesky(G), M), ord=2))
    assert torch.equal(sigmas[0], sigmas[1]), (
        f"{policy} vs {partner}: identical key exponent gave different "
        f"sigma_max, so the value weighting is reaching G or M")


def test_neither_mixed_policy_is_the_retired_two_stream_form():
    """The retired asymmetric convention (`zb = beta z` against raw `z`) is
    22.8x over the bound. Both new cells put ONE key stream into BOTH slots of
    M, which is what the Cauchy-Schwarz argument needs, and "key exponent" is
    close enough to "asymmetric keys" in words to be worth asserting in code.
    """
    for policy in MIXED:
        mod = _sharpened(policy)
        beta, z, zq, v = _stream(mod, T=64)
        x, _ = mod._weight_key_value(z, beta, v)
        # exact_stats is called with x in BOTH key slots everywhere in
        # production; assert the asymmetric alternative is measurably worse so
        # this test fails if someone "generalises" the helper to two streams.
        G2, M2, _, _, _ = exact_stats(z, beta.unsqueeze(-1) * z, zq, v, 1e-2)
        sigma_two = float(torch.linalg.matrix_norm(
            whiten_M(torch.linalg.cholesky(G2), M2), ord=2).max())
        G1, M1, _, _, _ = exact_stats(x, x, zq, v, 1e-2)
        sigma_one = float(torch.linalg.matrix_norm(
            whiten_M(torch.linalg.cholesky(G1), M1), ord=2).max())
        assert sigma_one <= 1.0 + 1e-9, policy
        assert sigma_two > 1.0, (
            "the two-stream form is supposed to VIOLATE the bound; if it no "
            "longer does, this test has stopped being evidence")


# ---------------------------------------------------------------------------
# The ridge confound the screening design has to control for.
# ---------------------------------------------------------------------------

def test_the_key_exponent_changes_the_effective_ridge_at_init():
    """Why `learned` at 2x ridge is a mandatory screening cell, quantified.

    At the shared initialisation beta = 0.5, a unit-norm key contributes
    `beta = 0.5` to G under a sqrt key exponent and `beta^2 = 0.25` under a
    linear one. So against a FIXED ridge, a linear key exponent is twice as
    ridge-regularised at step 0 -- and any linear-vs-sqrt difference could be
    that and nothing to do with the exponent.

    The factor is asserted to be 2.0 rather than merely "> 1" because 2.0 is the
    number the ridge-matched control cell is built from.
    """
    ridge = 1e-2
    sqrt_mod = _module("learned", seed=13)     # NOT sharpened: beta == 0.5 init
    lin_mod = _module("linear", seed=13)
    B, T = 2, 32
    h = _hidden(B=B, T=T)
    beta = sqrt_mod._resolve_beta(h).detach()
    assert torch.allclose(beta, torch.full_like(beta, 0.5)), "init must be 0.5"
    g = torch.Generator().manual_seed(14)
    z = torch.randn(B, T, sqrt_mod.H, sqrt_mod.rank, generator=g, dtype=DTYPE)
    z = z / z.norm(dim=-1, keepdim=True)       # exactly unit norm
    v = torch.randn(B, T, sqrt_mod.H, sqrt_mod.P, generator=g, dtype=DTYPE)

    own = {}
    for name, mod in (("sqrt", sqrt_mod), ("linear", lin_mod)):
        x, _ = mod._weight_key_value(z, beta, v)
        # trace of the accumulated Gram at the final token, minus the ridge:
        # the total own-weight the operator has written by then.
        own[name] = float((x * x).sum())
    assert math.isclose(own["sqrt"] / own["linear"], 2.0, rel_tol=1e-9), (
        f"own-weight ratio {own['sqrt'] / own['linear']} != 2.0; the "
        f"ridge-matched control cell is derived from this factor")


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_a_mixed_policy_inherits_the_ridge_confound_from_its_key_exponent(policy):
    """C shares `linear`'s halved own-weight; D shares `learned`'s. So the
    ridge control is needed for C and not for D, and that is a property of the
    KEY exponent alone -- which is exactly the decomposition's point."""
    key_exp, _ = MIXED[policy]
    ratio_to_sqrt = 0.5 ** (2 * key_exp) / 0.5 ** (2 * 0.5)
    mod = _module(policy, seed=15)
    ref = _module("learned", seed=15)
    B, T = 2, 32
    h = _hidden(B=B, T=T)
    beta = mod._resolve_beta(h).detach()
    g = torch.Generator().manual_seed(16)
    z = torch.randn(B, T, mod.H, mod.rank, generator=g, dtype=DTYPE)
    z = z / z.norm(dim=-1, keepdim=True)
    v = torch.randn(B, T, mod.H, mod.P, generator=g, dtype=DTYPE)
    x_mixed, _ = mod._weight_key_value(z, beta, v)
    x_ref, _ = ref._weight_key_value(z, beta, v)
    got = float((x_mixed * x_mixed).sum()) / float((x_ref * x_ref).sum())
    assert math.isclose(got, ratio_to_sqrt, rel_tol=1e-9), (
        f"{policy}: own-weight ratio to `learned` is {got}, expected "
        f"{ratio_to_sqrt} from key exponent {key_exp}")


# ---------------------------------------------------------------------------
# Identity: new cells must not renumber anything, and must not be invisible.
# ---------------------------------------------------------------------------

def test_the_pinned_golden_identity_survives_the_new_cells():
    """Adding a CHOICE must not touch identity at all -- unlike adding a field,
    it changes no default. Asserted because `BETA_POLICIES` is imported by
    `config.py`'s validation, and a change there is one edit away from a change
    to the dataclass.
    """
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.spec import group_id, run_id

    spec = resolve_run_spec(REPO_ROOT / "configs/runs/4m-golden.yaml")
    assert spec.model.ska_beta_policy == "learned"
    assert run_id(spec) == "2e63f16e"
    assert group_id(spec) == "d812e412"


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_each_new_cell_gets_its_own_run_identity(policy):
    """Two scientifically different policies sharing a run_id would share a run
    directory and therefore one set of checkpoints."""
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.spec import group_id, run_id

    spec = resolve_run_spec(REPO_ROOT / "configs/runs/4m-golden.yaml")
    other = dataclasses.replace(
        spec, model=dataclasses.replace(spec.model, ska_beta_policy=policy))
    assert config_hash(other.model) != config_hash(spec.model), policy
    assert group_id(other) != group_id(spec), policy
    assert run_id(other) != run_id(spec), policy
    # ... and distinct from every OTHER policy, not merely from the default.
    seen = {}
    for p in sorted(BETA_POLICIES):
        s = dataclasses.replace(
            spec, model=dataclasses.replace(spec.model, ska_beta_policy=p))
        h = config_hash(s.model)
        assert h not in seen, f"{p} and {seen.get(h)} share a config_hash"
        seen[h] = p


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_a_mixed_policy_counts_the_learned_gates_parameters(policy):
    """`param_count` is reported in specs and trial metadata. C and D build the
    same Linear(d, H, bias=True) as `learned`, so they must be counted like it
    -- not like `head_scalar` (H scalars) and not like `one` (nothing)."""
    base = dataclasses.replace(build_config("1m"), ska_beta_policy="learned")
    mixed = dataclasses.replace(base, ska_beta_policy=policy)
    assert mixed.param_count_estimate() == base.param_count_estimate()
    one = dataclasses.replace(base, ska_beta_policy="one")
    assert mixed.param_count_estimate() > one.param_count_estimate(), (
        "a policy that builds a gate must count more parameters than `one`")


# ---------------------------------------------------------------------------
# Forward/backward and train/decode agreement.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", sorted(MIXED))
def test_each_mixed_policy_runs_a_full_forward_and_backward(policy):
    """Catches a cell wired into `_weight_key_value` but not into the gate
    construction, or vice versa."""
    mod = _module(policy)
    h = _hidden().requires_grad_(True)
    out = mod(h)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert mod.beta_proj.weight.grad is not None
    assert torch.isfinite(mod.beta_proj.weight.grad).all()


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_the_gate_receives_a_nonzero_gradient_under_each_mixed_policy(policy):
    """A cell whose beta gradient is identically zero is not a write gate, it is
    a constant. The chunked route already has this failure (`beta_proj.bias`
    gradient cosine ~0.00), which is why the launcher refuses that backend --
    but a bad exponent could reproduce it on the exact route.
    """
    mod = _module(policy)
    h = _hidden()
    mod(h).sum().backward()
    gnorm = float(mod.beta_proj.weight.grad.norm())
    assert gnorm > 0, f"{policy}: beta_proj weight gradient is exactly zero"


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_decode_weights_keys_and_values_as_the_training_forward_does(policy):
    """`models/recurrent.py::_proj_norm` is the decode path's only weighting
    site. A policy added to the training forward alone would decode as
    `learned`, which surfaces as degraded generation quality rather than an
    error -- the exact trap `symmetric_key_value`'s one-helper rule exists to
    close.

    The right-hand side is an INDEPENDENT restatement of what training does
    (projections, causal_normalize, then the policy weighting), not a second
    call to the same code path, so a convention change in `forward` makes this
    diverge rather than following along.
    """
    from koopman_lm.models.recurrent import RecurrentKoopmanLM
    torch.manual_seed(0)
    mod = SKAModule(32, 4, rank=16, inverse_cholesky=True,
                    beta_policy=policy, norm_clip_c=4.0)
    with torch.no_grad():
        mod.beta_proj.weight.normal_(0.0, 1.0)
        mod.beta_proj.bias.normal_(0.0, 0.5)
    g = torch.Generator().manual_seed(7)
    h = torch.randn(2, 24, 32, generator=g)

    x_dec, zq_dec, v_dec = RecurrentKoopmanLM._proj_norm(mod, h)

    B, T, _ = h.shape
    H, r, P = mod.H, mod.rank, mod.P
    beta = mod._resolve_beta(h)
    z_n = causal_normalize(mod.key_proj(h).reshape(B, T, H, r), mod.norm_clip_c)
    key_exp, val_exp = MIXED[policy]
    x_ref = beta.clamp_min(0).pow(key_exp).unsqueeze(-1) * z_n
    v_ref = (beta.clamp_min(0).pow(val_exp).unsqueeze(-1)
             * mod.value_proj(h).reshape(B, T, H, P))

    assert torch.equal(x_dec, x_ref.float()), f"{policy}: decode key exponent"
    assert torch.equal(v_dec, v_ref.float()), f"{policy}: decode value exponent"
    # The QUERY is never beta-weighted, under any policy. Weighting it would be
    # a plausible-looking mistake that no loss curve would flag.
    zq_ref = causal_normalize(
        mod.query_proj(h).reshape(B, T, H, r), mod.norm_clip_c)
    assert torch.equal(zq_dec, zq_ref.float()), f"{policy}: query must be raw"


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_recurrent_accumulation_reproduces_the_prefill_statistics(policy):
    """Prefill/decode parity at the level the operator actually consumes.

    `exact_stats` computes G, M and C for the whole sequence at once; decode
    accumulates them one token at a time. Under a mixed exponent the two could
    disagree only if a per-token path re-derived a weight, so this walks the
    recurrence explicitly and compares against the batched kernel at every
    token.
    """
    mod = _sharpened(policy)
    beta, z, zq, v = _stream(mod, T=20)
    x, vbar = mod._weight_key_value(z, beta, v)
    ridge = 1e-2
    Gf, Mf, Cf, _, (B, T, H, P) = exact_stats(x, x, zq, vbar, ridge)
    r = mod.rank
    eye = torch.eye(r, dtype=DTYPE)

    # Token-at-a-time accumulation, exactly what a decode loop does.
    G = (ridge + 1e-4) * eye.expand(B, H, r, r).clone()
    M = torch.zeros(B, H, r, r, dtype=DTYPE)
    C = torch.zeros(B, H, P, r, dtype=DTYPE)
    prev = None
    for t in range(T):
        assert torch.allclose(Gf.reshape(B, T, H, r, r)[:, t], G, atol=1e-11), \
            f"{policy}: G diverged at token {t}"
        assert torch.allclose(Mf.reshape(B, T, H, r, r)[:, t], M, atol=1e-11), \
            f"{policy}: M diverged at token {t}"
        assert torch.allclose(Cf.reshape(B, T, H, P, r)[:, t], C, atol=1e-11), \
            f"{policy}: C diverged at token {t}"
        xt, vt = x[:, t], vbar[:, t]
        G = G + torch.einsum('bhr,bhs->bhrs', xt, xt)
        if prev is not None:
            M = M + torch.einsum('bhr,bhs->bhrs', xt, prev)
        C = C + torch.einsum('bhp,bhr->bhpr', vt, xt)
        prev = xt


@pytest.mark.parametrize("policy", sorted(MIXED))
def test_a_search_space_can_OPT_IN_to_each_new_cell(policy):
    """Reachable from a study, but NOT in the default space.

    This test previously asserted the opposite -- that each new cell IS a default
    `beta_policy` choice -- and that assertion was the bug. Widening the declared
    default changes the recorded `CategoricalDistribution` of every study that
    sampled it, and optuna raises `does not support dynamic value space` on
    reattach: a HARD failure at `ask()`, after a worker has claimed a run root.
    One archived journal (`smoke-fanout.5f93952c`) was broken by it.

    So the requirement is the opt-in, not the default. `_check_beta_policy` must
    accept the policy (it is a legal value of the axis), and `search_axes` must be
    able to declare it, while `search_space()` must NOT offer it unasked. See
    `test_beta_policy_axis_stability.py`.
    """
    from experimentation.sweep.search.space import (
        _check_beta_policy, restrict_space, search_space)
    base = build_config("1m")
    space = search_space(base, base_lr=1e-3)
    assert policy not in space["beta_policy"]["choices"], (
        f"{policy} is in the DEFAULT search space; widening that default breaks "
        f"reattachment to every journal that sampled the narrower support")
    _check_beta_policy(policy, base)      # a legal value of the axis
    opted_in = restrict_space(
        space, base,
        axes={"beta_policy": {"kind": "categorical",
                              "choices": ["learned", policy]}})
    assert policy in opted_in["beta_policy"]["choices"]
