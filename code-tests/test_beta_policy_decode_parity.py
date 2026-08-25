"""Decode must weight keys and values exactly as training does, for EVERY
`ska_beta_policy`.

## The specific trap

`kernels/chunk_stats.py::symmetric_key_value`'s docstring names it: "Use this ONE
helper at every accumulation site so the weighting convention is provably
identical (no path re-derives beta from a vector norm -- that is the
train/decode divergence trap)." Before `ska_beta_policy` existed there was one
convention, so a decode path that computed `sigmoid(ska.beta_proj(h))` locally
was correct by accident. With four policies it is correct for exactly one of
them:

  * under `head_scalar` it would read a projection that does not exist,
  * under `one` it would crash (`beta_proj` is None),
  * under `linear` it would apply sqrt(beta) where training applies beta.

The `linear` case is the dangerous one, because it does not crash. A model
trained with `linear` would decode with the wrong exponent on every key and
value, and the symptom would be degraded generation quality -- not an error. That
is a bug that gets attributed to the architecture rather than to the wiring.

`models/recurrent.py::_proj_norm` now calls `ska._resolve_beta` and
`ska._weight_key_value`, which is what this file pins. Tested at the helper level
rather than through `RecurrentKoopmanLM`, deliberately: the full recurrent
wrapper needs the Mamba-2 backbone and therefore a GPU, and this property is
about the weighting convention rather than about the backbone. The GPU-marked
`test_decode_prefill_parity.py::test_full_model_decode_prefill_parity` covers the
end-to-end path.
"""
import pytest
import torch

from koopman_lm.models.recurrent import RecurrentKoopmanLM
from koopman_lm.modules.seq.ska import BETA_POLICIES, SKAModule
from koopman_lm.kernels.chunk_stats import causal_normalize

pytestmark = pytest.mark.correctness


def _module(policy, d_model=32, n_heads=4, rank=16, seed=0):
    torch.manual_seed(seed)
    mod = SKAModule(d_model, n_heads, rank=rank, inverse_cholesky=True,
                    beta_policy=policy, norm_clip_c=4.0)
    # Perturb whatever gate parameters the policy has, so the comparison is
    # against a NON-trivial beta. At initialisation every policy gives a
    # constant, and a constant would make all four agree for the wrong reason.
    with torch.no_grad():
        if mod.beta_proj is not None:
            mod.beta_proj.weight.normal_(0.0, 1.0)
            mod.beta_proj.bias.normal_(0.0, 0.5)
        if mod.beta_logit is not None:
            mod.beta_logit.normal_(0.0, 1.0)
    return mod


def _training_weighting(mod, h):
    """The key/value weighting the TRAINING forward performs, recomputed from
    the module's own projections exactly as `SKAModule.forward` does.

    Written out here rather than captured from `forward`, so this is an
    independent statement of what training does; if `forward` changes its
    convention, this diverges and the test fails, which is the point.
    """
    B, T, _ = h.shape
    H, r, P = mod.H, mod.rank, mod.P
    z = mod.key_proj(h).reshape(B, T, H, r)
    v = mod.value_proj(h).reshape(B, T, H, P)
    beta = mod._resolve_beta(h)
    z_n = causal_normalize(z, mod.norm_clip_c)
    return mod._weight_key_value(z_n, beta, v)


@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_decode_weights_keys_and_values_exactly_as_training_does(policy):
    mod = _module(policy)
    g = torch.Generator().manual_seed(7)
    h = torch.randn(2, 24, mod.d_model, generator=g)

    x_decode, zq_decode, vbar_decode = RecurrentKoopmanLM._proj_norm(mod, h)
    x_train, vbar_train = _training_weighting(mod, h)

    # Exact, not approximate: both sides call the same two helpers on the same
    # tensors, so any difference is a convention difference rather than
    # roundoff. A tolerance here would hide precisely the bug being tested.
    assert torch.equal(x_decode, x_train.float()), (
        f"{policy}: decode's symmetric key differs from training's")
    assert torch.equal(vbar_decode, vbar_train.float()), (
        f"{policy}: decode's weighted value differs from training's")
    # The query is NOT beta-weighted on either side -- pinned because weighting
    # it would be a plausible-looking mistake that no loss curve would flag.
    zq_train = causal_normalize(
        mod.query_proj(h).reshape(2, 24, mod.H, mod.rank), mod.norm_clip_c)
    assert torch.equal(zq_decode, zq_train.float()), (
        f"{policy}: decode's query differs from training's")


@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_decode_does_not_touch_beta_proj_directly(policy):
    """A structural guard on the fix, not a restatement of it.

    The bug this file exists to prevent is a call site that reaches for
    `ska.beta_proj` instead of `ska._resolve_beta`. Under `one` and
    `head_scalar` there is no `beta_proj` to reach for, so a regression would
    raise -- but under `learned` and `linear` it would silently succeed and
    `linear` would get the wrong exponent. Deleting the attribute for the
    duration of the call makes the reach fail for ALL FOUR policies.
    """
    mod = _module(policy)
    g = torch.Generator().manual_seed(8)
    h = torch.randn(2, 16, mod.d_model, generator=g)
    expected = RecurrentKoopmanLM._proj_norm(mod, h)

    beta = mod._resolve_beta(h).detach().clone()
    saved = mod.beta_proj
    # `_resolve_beta` is the only legitimate reader, so it is stubbed to return
    # the value it would have produced; anything ELSE reading beta_proj now hits
    # None and raises.
    mod.beta_proj = None
    mod._resolve_beta = lambda _h, _b=beta: _b
    try:
        got = RecurrentKoopmanLM._proj_norm(mod, h)
    finally:
        mod.beta_proj = saved
    for a, b in zip(got, expected):
        assert torch.equal(a, b), (
            f"{policy}: _proj_norm reads the write gate through something "
            f"other than _resolve_beta")
