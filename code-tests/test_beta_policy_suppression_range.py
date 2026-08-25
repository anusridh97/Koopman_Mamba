"""What sqrt(beta) costs: the gate's reachable SUPPRESSION range.

Written before the retrieval arm's replicate seeds landed, deliberately, so that
the mechanism is a prediction rather than a story fitted to a result. Everything
here is arithmetic on the two weightings -- no training, no data.

## The asymmetry

For beta in (0, 1), sqrt(beta) > beta. The square root pulls small values UP:
sqrt(0.01) = 0.1, ten times the 0.01 that `linear` would apply. So the two
policies differ in a way that is invisible at beta near 1 and grows without
bound as the gate tries to suppress:

    achievable key weight w   `learned` needs beta = w^2   `linear` needs beta = w

Since beta = sigmoid(logit), that difference is a difference in how far the gate
has to push its logit for a given suppression:

    suppress to w = 0.1     learned logit = -4.60   linear logit = -2.20
    suppress to w = 0.01    learned logit = -9.21   linear logit = -4.60
    suppress to w = 0.001   learned logit = -13.8

i.e. **the sqrt convention costs a factor of ~2 in required logit magnitude
for any target suppression** (exactly, in the small-w limit: log(w^2) = 2 log w).

## Why that is the interesting number rather than a curiosity

The write gate exists to keep distractors out of the associative memory. Its
usefulness is bounded by how far it can push a distractor's key weight DOWN
relative to a fact's. Under `learned` that range is the square of what `linear`
reaches for the same logits -- and a projection initialised to zeros
(`beta_proj` is zero-initialised, so every gate starts at beta = 0.5) has to
travel twice as far to get there.

## What this does NOT establish

That `linear` is better. Selectivity is a capability, not a benefit: a sharper
gate could equally suppress things worth keeping, and `beta = 1` (no gate,
maximal key weight for everything) is a live competitor precisely because
indiscriminate writing is not obviously worse than badly-aimed suppression.
Whether any of it moves a metric is what scripts/analyze_beta_policy.py and
scripts/analyze_beta_policy_mqar.py measure.

What it DOES establish is that `learned` vs `linear` is not a cosmetic
reparameterisation. The two policies have materially different reachable
behaviour at fixed logit scale, so a difference between them in either arm has a
mechanism to point at -- and an ABSENCE of difference is then informative too,
because it says the gate is not operating in the regime where the two diverge.
"""
import math

import pytest
import torch

from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness


def _module(policy, d_model=32, n_heads=4, rank=16):
    torch.manual_seed(0)
    return SKAModule(d_model, n_heads, rank=rank, inverse_cholesky=True,
                     beta_policy=policy)


def _key_weight(policy, beta):
    """The multiplier the policy applies to a unit key, read off the shipped
    weighting helper rather than reimplemented -- so this cannot drift from
    what the model does."""
    mod = _module(policy)
    z = torch.ones(1, 1, mod.H, mod.rank, dtype=torch.float64)
    v = torch.zeros(1, 1, mod.H, mod.P, dtype=torch.float64)
    b = torch.full((1, 1, mod.H), float(beta), dtype=torch.float64)
    x, _ = mod._weight_key_value(z, b, v)
    return float(x[0, 0, 0, 0])


@pytest.mark.parametrize("beta", [0.5, 0.1, 0.01, 1e-3, 1e-4])
def test_sqrt_beta_suppresses_strictly_less_than_beta(beta):
    """The whole asymmetry, in one assertion."""
    learned = _key_weight("learned", beta)
    linear = _key_weight("linear", beta)
    assert learned == pytest.approx(math.sqrt(beta))
    assert linear == pytest.approx(beta)
    assert learned > linear, (
        f"beta={beta}: sqrt(beta) must exceed beta on (0,1) -- the sqrt "
        f"convention cannot suppress as hard at the same beta")


def test_the_two_policies_agree_at_beta_one():
    """`one` is where the conventions coincide, which is why `beta_policy=one`
    is the same model under either -- and why the LM/retrieval arms can use it
    as a common reference point."""
    for policy in ("learned", "linear"):
        assert _key_weight(policy, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("target", [0.1, 0.01, 1e-3])
def test_the_sqrt_convention_costs_exactly_a_factor_of_two_in_logit(target):
    """To reach key weight `target`, `learned` needs beta = target^2 and
    `linear` needs beta = target -- so the required sigmoid logit doubles.

    Asserted as the RATIO of logits rather than as two absolute numbers,
    because the ratio is the scale-free statement and it tends to exactly 2 in
    the small-target limit where suppression matters.

    It approaches 2 from ABOVE, not below. An earlier version of this test
    asserted `ratio <= 2.0` and failed at 2.0003 -- which is the test doing its
    job on the claim rather than on the code. The reason:
    logit(w) = ln w - ln(1-w) = ln w + w + O(w^2), so

        ratio = logit(w^2)/logit(w) ~ 2 ln w / (ln w + w)

    and since ln w < 0, adding w shrinks the denominator's magnitude, pushing
    the ratio slightly above 2. Measured: 2.0913 at w=0.1, 2.0044 at 0.01,
    2.0003 at 0.001.
    """
    def logit(p):
        return math.log(p / (1.0 - p))

    learned_logit = logit(target ** 2)
    linear_logit = logit(target)
    assert learned_logit < linear_logit < 0
    ratio = learned_logit / linear_logit
    assert 2.0 <= ratio < 2.5, (
        f"target={target}: logit ratio {ratio:.4f} outside the expected range; "
        f"the factor-of-two claim in this module's docstring is wrong")
    if target <= 0.01:
        assert ratio < 2.01, (
            f"target={target}: the ratio should be converging to 2 by here, "
            f"got {ratio:.4f}")


def test_every_policy_starts_at_the_same_gate_value():
    """All four begin at beta = 0.5 (zero-initialised projection / logits), or
    at 1 for `one`. Stated because it is what makes the arms a comparison of
    PARAMETERISATIONS rather than of starting points -- the four models differ
    in where the gate can GO, not in where it begins.
    """
    h = torch.zeros(1, 4, 32)
    for policy in ("learned", "linear", "head_scalar"):
        beta = _module(policy)._resolve_beta(h)
        assert torch.allclose(beta, torch.full_like(beta, 0.5)), policy
    assert torch.allclose(_module("one")._resolve_beta(h),
                          torch.ones(1, 4, 4))
    # ... but the KEY WEIGHT at that shared starting beta already differs, which
    # is the initial condition the arms actually see.
    assert _key_weight("learned", 0.5) == pytest.approx(math.sqrt(0.5))
    assert _key_weight("linear", 0.5) == pytest.approx(0.5)
