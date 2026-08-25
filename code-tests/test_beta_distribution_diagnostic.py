"""`ska_health` must report beta's DISTRIBUTION, not just its mean.

## Why a mean is the wrong summary

"Does the write gate develop meaningful token and head variation" is a
*necessary condition* for the gate to be doing anything, and `beta_mean` cannot
answer it. A gate pinned at 0.5 everywhere and a gate alternating between 0.01
and 0.99 have the same mean. So the one number the diagnostic reported is
precisely the one that cannot distinguish a working gate from a dead one.

Two variances are needed, not one, because they answer different questions and
the four `ska_beta_policy` choices separate them cleanly:

    token variation   sd across tokens, within a head, averaged over heads.
                      This is the CONTENT dependence -- the thing the gate is
                      for. `head_scalar` has exactly zero of it by
                      construction.
    head variation    sd of the per-head means. A per-head write SCALE, which
                      `out_proj` could in principle absorb, so a gate showing
                      only this is not obviously earning its parameters.

The policies make the test sharp rather than vague: `one` must show zero of
both, `head_scalar` must show zero token variation and nonzero head variation
once its logits differ, and `learned` must show both. A diagnostic that reported
a single pooled sd would pass for `head_scalar` and would be reporting head
variation as though it were content dependence.

## And this is NOT evidence the gate helps

Deliberately stated here as well as in the study file, because the whole reason
these numbers get reported separately from the loss is that a gate can vary a
great deal and help nothing. Nothing in this file measures benefit. Reading a
large `beta_sd_token` as "the gate is useful" is the specific error the
separation exists to prevent.
"""
import pytest
import torch

from koopman_lm.diagnostics.ska import ska_health
from koopman_lm.modules.seq.ska import SKAModule

pytestmark = pytest.mark.correctness

KEYS = ("beta_mean", "beta_sd_token", "beta_sd_head", "beta_min", "beta_max")


def _module(policy, d_model=32, n_heads=4, rank=16, seed=0):
    torch.manual_seed(seed)
    return SKAModule(d_model, n_heads, rank=rank, inverse_cholesky=True,
                     beta_policy=policy, chunk_size=16)


def _health(mod, B=2, T=48, seed=1):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, mod.d_model, generator=g)
    return ska_health(mod, h)


@pytest.mark.parametrize("policy", ["learned", "one", "head_scalar", "linear"])
def test_the_distribution_keys_are_present_and_finite(policy):
    health = _health(_module(policy))
    for key in KEYS:
        assert key in health, f"{policy}: ska_health does not report {key}"
        assert torch.isfinite(torch.as_tensor(health[key])).all(), (policy, key)


def test_an_untrained_learned_gate_is_a_constant_one_half():
    """`beta_proj` is zero-initialised, so a fresh `learned` gate has NO
    variation. Worth pinning: it is the baseline against which a trained gate's
    variation is read, and it means a nonzero sd in a trained checkpoint is
    learned rather than an artefact of initialisation.
    """
    health = _health(_module("learned"))
    assert float(health["beta_mean"]) == pytest.approx(0.5)
    assert float(health["beta_sd_token"]) == pytest.approx(0.0, abs=1e-12)
    assert float(health["beta_sd_head"]) == pytest.approx(0.0, abs=1e-12)
    assert float(health["beta_min"]) == pytest.approx(0.5)
    assert float(health["beta_max"]) == pytest.approx(0.5)


def test_the_no_gate_policy_reports_exactly_one_with_no_variation():
    health = _health(_module("one"))
    assert float(health["beta_mean"]) == pytest.approx(1.0)
    assert float(health["beta_sd_token"]) == pytest.approx(0.0, abs=1e-12)
    assert float(health["beta_sd_head"]) == pytest.approx(0.0, abs=1e-12)


def test_a_perturbed_learned_gate_shows_BOTH_token_and_head_variation():
    mod = _module("learned")
    with torch.no_grad():
        mod.beta_proj.weight.normal_(0.0, 1.0)
        mod.beta_proj.bias.normal_(0.0, 0.5)
    health = _health(mod)
    assert float(health["beta_sd_token"]) > 1e-3
    assert float(health["beta_sd_head"]) > 1e-3
    assert float(health["beta_min"]) < float(health["beta_max"])


def test_head_scalar_separates_the_two_variances():
    """THE test that shows the decomposition is doing real work.

    A per-head gate has head variation and, by construction, exactly zero token
    variation. A single pooled sd would be nonzero here and would be read as
    content dependence, which is the misreading this decomposition exists to
    make impossible.
    """
    mod = _module("head_scalar")
    with torch.no_grad():
        mod.beta_logit.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))
    health = _health(mod)
    assert float(health["beta_sd_token"]) == pytest.approx(0.0, abs=1e-12), \
        "head_scalar cannot depend on the token"
    assert float(health["beta_sd_head"]) > 0.1, \
        "four logits spanning [-2, 2] must show head variation"


def test_beta_mean_is_unchanged_by_the_addition():
    """The pre-existing key keeps its meaning, so any consumer reading
    `beta_mean` off an archived diagnostic still gets the same quantity."""
    mod = _module("learned")
    with torch.no_grad():
        mod.beta_proj.weight.normal_(0.0, 1.0)
    health = _health(mod)
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 48, mod.d_model, generator=g)
    assert float(health["beta_mean"]) == pytest.approx(
        float(mod._resolve_beta(h).mean().detach()), rel=1e-6)
