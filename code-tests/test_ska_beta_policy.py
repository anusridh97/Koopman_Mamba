"""`ska_beta_policy`: the four write-gate parameterisations, and the identity
mechanism that lets a new model field exist without renumbering finished runs.

## The four policies and what each one is for

The SKA write gate is `beta = sigmoid(beta_proj(h))`, per token and per head,
and nothing in the repo has ever tested whether it earns its keep. These four
are the space in which that question is answerable, chosen so that no contrast
is confounded by the backend:

    learned      beta = sigmoid(W h + b)     x = sqrt(beta) z   (the default;
                                                                 unchanged)
    one          beta = 1                    x = z              no gate at all
    head_scalar  beta = sigmoid(b_h)         x = sqrt(beta) z   per-head scale,
                                                                no token
                                                                dependence
    linear       beta = sigmoid(W h + b)     x = beta z         the "simpler
                                                                alternative":
                                                                just multiply
                                                                by beta

Read as contrasts:

  * `learned` vs `one` -- is a write gate useful?
  * `learned` vs `head_scalar` -- is the TOKEN dependence useful, or is the gate
    only supplying a per-head write scale that out_proj could absorb?
  * `learned` vs `linear` -- is the square root the right exponent?

All four put the SAME key stream into both slots of M, so all four are
contractive (`test_every_policy_keeps_the_operator_contractive`). That is not a
detail: the reference config `configs/runs/proxy-256x17.yaml` runs
`ska_inverse_cholesky: true`, which omits the spectral clamp entirely, so a
policy that broke contractivity would silently train an expansive operator
there. It is also what makes this a clean 4-way comparison -- every cell can run
on the same backend, so no contrast is confounded with a change of kernel. See
`test_ska_contractivity_contract.py` for the bound and for the two-stream form
that violates it (which is why the legacy asymmetric convention is deliberately
NOT one of the four).

`linear` is the interesting near-miss. It is contractive, so it is safe, and it
silently redefines the write weight as beta^2 -- G becomes sum beta^2 z z^T.
That is the whole content of "symmetric sqrt-beta": the square root is not what
buys the bound, it is what keeps `G = sum beta z z^T`, i.e. what keeps "beta is
the per-token write weight" literally true.

## Why the identity tests are in this file

`KoopmanLMConfig` is hashed field-for-field into `config_hash`, `group_id` and
`run_id` (`run/spec.py::_scientific_payload` calls `dataclasses.asdict`). So
adding ANY model field renumbers every run ever recorded --
`configs/runs/4m-golden.yaml` would stop hashing to `run_id 2e63f16e` /
`group_id d812e412`, and `test_identity_baseline.py`'s own docstring says that
baseline may be regenerated only "when an intentional change to run identity has
been decided and the old->new mapping recorded -- never to make this go green".

Renumbering is not the intent here, and it should not be the price of adding a
knob. `_scientific_payload` already solves the same problem for `schedules` and
`optim.groups`: an absent section is OMITTED from the hash rather than emitted
empty, precisely so that adding the mechanism did not renumber runs that predate
it (REVIEW.md records 4m-golden surviving that change for exactly this reason).
A model field cannot be "absent" -- KoopmanLMConfig is total by design -- so the
same idea needs an explicit spelling: `IDENTITY_TRANSPARENT_DEFAULTS`, a
registry of fields added after identity was pinned, each mapped to the value an
archived run implicitly had. At that value the field contributes nothing to the
hash.

That is faithful rather than convenient: a run recorded before this field
existed really did have `learned` behaviour, so hashing it as
"absent == learned" says something true about it. The mechanism has one
requirement, tested below, without which it would be a hole rather than a
bridge: **a non-default value must change the hash.** A mechanism that made the
field invisible in both directions would let two scientifically different runs
share a run_id and a directory.
"""
import dataclasses
import json
import warnings
from pathlib import Path

import pytest
import torch

from koopman_lm.config import (
    IDENTITY_TRANSPARENT_DEFAULTS, KoopmanLMConfig, build_config, config_hash)
from koopman_lm.kernels.chunk_stats import causal_normalize
from koopman_lm.kernels.chunk_stats_exact import exact_stats
from koopman_lm.kernels.lin_alg import whiten_M
from koopman_lm.modules.seq.ska import BETA_POLICIES, SKAModule

pytestmark = pytest.mark.correctness

REPO_ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float64


def _module(policy, d_model=32, n_heads=4, rank=16, seed=0):
    torch.manual_seed(seed)
    mod = SKAModule(d_model, n_heads, rank=rank, inverse_cholesky=True,
                    beta_policy=policy, precision='fp64')
    # fp64 throughout: these tests read a spectral norm off a Cholesky, and the
    # `precision='fp64'` kwarg governs only the whitened CORE -- the projections
    # are still whatever dtype the module was constructed in.
    return mod.double()


def _hidden(B=2, T=12, d_model=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d_model, generator=g, dtype=DTYPE)


# ---------------------------------------------------------------------------
# The four policies produce the betas they claim to.
# ---------------------------------------------------------------------------

def test_the_default_policy_is_learned():
    """Omitting the field must reproduce the pre-field model exactly, or every
    committed config silently changes meaning."""
    assert KoopmanLMConfig().ska_beta_policy == "learned"
    assert IDENTITY_TRANSPARENT_DEFAULTS["ska_beta_policy"] == "learned"


def test_learned_policy_varies_across_tokens_and_heads():
    mod = _module("learned")
    # beta_proj is zero-initialised, so an untrained module gives a constant
    # 0.5 everywhere. Perturb the projection so the test observes the
    # PARAMETERISATION rather than the initialisation.
    with torch.no_grad():
        mod.beta_proj.weight.normal_(0.0, 1.0)
        mod.beta_proj.bias.normal_(0.0, 1.0)
    h = _hidden()
    beta = mod._resolve_beta(h)
    assert beta.shape == (h.shape[0], h.shape[1], mod.H)
    assert beta.std(dim=1).min() > 0, "learned beta must vary across TOKENS"
    assert beta.std(dim=-1).min() > 0, "learned beta must vary across HEADS"
    assert torch.all((beta > 0) & (beta < 1))


def test_one_policy_writes_every_token_at_full_weight_and_has_no_parameters():
    mod = _module("one")
    beta = mod._resolve_beta(_hidden())
    assert torch.all(beta == 1.0)
    # The point of `one` is that it is a SIMPLER MODEL, so the gate's
    # parameters must actually be gone rather than present and ignored --
    # otherwise the comparison is confounded by weight decay acting on dead
    # parameters, and the parameter count still reports them.
    assert not any("beta_proj" in n for n, _ in mod.named_parameters())
    assert mod.beta_proj is None


def test_head_scalar_policy_is_constant_over_tokens_and_varies_over_heads():
    mod = _module("head_scalar")
    with torch.no_grad():
        mod.beta_logit.normal_(0.0, 1.0)
    beta = mod._resolve_beta(_hidden())
    assert beta.shape == (2, 12, mod.H)
    per_token_spread = beta.std(dim=1).max()
    assert per_token_spread == 0, (
        f"head_scalar must not depend on the token; spread {per_token_spread}")
    assert beta.std(dim=-1).min() > 0, "head_scalar must vary across HEADS"
    # H learnable scalars, not H*d_model + H.
    assert mod.beta_logit.numel() == mod.H


def test_linear_policy_shares_learned_betas_but_weights_the_key_by_beta():
    """`linear` differs from `learned` only in the EXPONENT applied to the key,
    so the beta itself must be identical -- otherwise the contrast measures two
    changes at once."""
    learned, linear = _module("learned", seed=3), _module("linear", seed=3)
    with torch.no_grad():
        # Copied, not re-drawn: `normal_` pulls from the global RNG, which the
        # two constructions have already advanced by different amounts, so
        # seeding the constructor is not enough to make the gates agree.
        learned.beta_proj.weight.normal_(0.0, 1.0)
        linear.beta_proj.weight.copy_(learned.beta_proj.weight)
        learned.beta_proj.bias.zero_()
        linear.beta_proj.bias.zero_()
    h = _hidden()
    assert torch.allclose(learned._resolve_beta(h), linear._resolve_beta(h))
    # ... and the key weight is beta, not sqrt(beta).
    beta = learned._resolve_beta(h)
    z = causal_normalize(torch.randn(2, 12, learned.H, 16, dtype=DTYPE), None)
    v = torch.randn(2, 12, learned.H, learned.P, dtype=DTYPE)
    x_lin, _ = linear._weight_key_value(z, beta, v)
    x_sqrt, _ = learned._weight_key_value(z, beta, v)
    assert torch.allclose(x_lin, beta.unsqueeze(-1) * z)
    assert torch.allclose(x_sqrt, beta.clamp_min(0).sqrt().unsqueeze(-1) * z)


@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_every_policy_runs_a_full_forward_and_backward(policy):
    """Cheap, and it is the check that catches a policy wired into
    `_resolve_beta` but not into the weighting, or a missing parameter."""
    mod = _module(policy)
    h = _hidden().requires_grad_(True)
    out = mod(h)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(h.grad).all()


def test_an_unknown_policy_is_refused_at_construction():
    with pytest.raises(ValueError, match="ska_beta_policy"):
        _module("sqrt")
    with pytest.raises(ValueError, match="ska_beta_policy"):
        dataclasses.replace(build_config("1m"), ska_beta_policy="sqrt")


# ---------------------------------------------------------------------------
# The safety property every policy has to keep.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_every_policy_keeps_the_operator_contractive(policy):
    """sigma_max(L^-1 M L^-T) <= 1 for all four.

    The reference config runs `ska_inverse_cholesky: true`, which has no
    spectral clamp, so this is not a nicety -- a policy that violated it would
    apply an expansive operator K times and the run would not fail, it would
    just train something else. All four are single-stream weightings, so the
    bound holds by the argument in test_ska_contractivity_contract.py; this
    pins that no future policy is added without it.
    """
    mod = _module(policy, seed=5)
    with torch.no_grad():
        if mod.beta_proj is not None:
            mod.beta_proj.weight.normal_(0.0, 2.0)
            mod.beta_proj.bias.normal_(0.0, 2.0)
        if getattr(mod, "beta_logit", None) is not None:
            mod.beta_logit.normal_(0.0, 2.0)
    h = _hidden(B=2, T=64)
    beta = mod._resolve_beta(h)
    g = torch.Generator().manual_seed(6)
    z = causal_normalize(
        torch.randn(2, 64, mod.H, mod.rank, generator=g, dtype=DTYPE), 4.0)
    zq = causal_normalize(
        torch.randn(2, 64, mod.H, mod.rank, generator=g, dtype=DTYPE), 4.0)
    v = torch.randn(2, 64, mod.H, mod.P, generator=g, dtype=DTYPE)
    x, vbar = mod._weight_key_value(z, beta.detach(), v)
    G, M, _, _, _ = exact_stats(x, x, zq, vbar, 1e-2)
    sigma = float(torch.linalg.matrix_norm(
        whiten_M(torch.linalg.cholesky(G), M), ord=2).max())
    assert sigma <= 1.0 + 1e-9, f"{policy}: sigma_max = {sigma:.9f} > 1"


# ---------------------------------------------------------------------------
# Identity: the field exists without renumbering anything, and is not invisible.
# ---------------------------------------------------------------------------

def test_the_pinned_golden_identity_survives_the_new_field():
    """The explicit statement of the constraint, next to the mechanism.

    `test_identity_baseline.py` already covers every registry config and every
    committed run spec. This asserts the one identity the handoff notes single
    out, so a failure names the constraint instead of naming a JSON file.
    """
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.spec import group_id, run_id

    spec = resolve_run_spec(REPO_ROOT / "configs/runs/4m-golden.yaml")
    assert spec.model.ska_beta_policy == "learned"
    assert run_id(spec) == "2e63f16e"
    assert group_id(spec) == "d812e412"


def test_a_non_default_policy_changes_config_hash_group_id_and_run_id():
    """The requirement without which the mechanism is a hole.

    Omitting the field at its legacy value is faithful. Omitting it at ANY
    value would let two scientifically different runs share a run_id -- and
    therefore a run directory, and therefore one set of checkpoints.
    """
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.spec import group_id, run_id

    spec = resolve_run_spec(REPO_ROOT / "configs/runs/4m-golden.yaml")
    for policy in sorted(BETA_POLICIES - {"learned"}):
        other = dataclasses.replace(
            spec, model=dataclasses.replace(spec.model, ska_beta_policy=policy))
        assert config_hash(other.model) != config_hash(spec.model), policy
        assert group_id(other) != group_id(spec), policy
        assert run_id(other) != run_id(spec), policy


def test_every_transparent_default_matches_the_dataclass_default():
    """The registry says "the value an archived run implicitly had". For a field
    added this way that is, by construction, the dataclass default -- and if the
    two ever disagree, the hash would omit a value that no archived run held,
    which is the one way this mechanism can silently corrupt identity.
    """
    defaults = {f.name: f.default for f in dataclasses.fields(KoopmanLMConfig)}
    for name, legacy in IDENTITY_TRANSPARENT_DEFAULTS.items():
        assert name in defaults, f"{name} is not a KoopmanLMConfig field"
        assert defaults[name] == legacy, (
            f"IDENTITY_TRANSPARENT_DEFAULTS[{name!r}] = {legacy!r} but the "
            f"dataclass default is {defaults[name]!r}")


def test_an_archived_spec_without_the_field_still_loads(tmp_path):
    """A materialized spec.yaml written before this field existed is the only
    record of what that run did. `_check_model_key_set` compares its model keys
    against the live dataclass and raises on any mismatch, so a bare field
    addition makes every finished run unreadable by eval and resume.

    Migrated with a DeprecationWarning, following `_migrate_microbatch`: a
    silent injection leaves the file looking correct while meaning something
    new, and the warning is what prompts the file to be updated.
    """
    from experimentation.run.resolve import (
        load_materialized_spec, resolve_run_spec, to_flat_dict)
    from experimentation.run.spec import run_id
    import yaml

    spec = resolve_run_spec(REPO_ROOT / "configs/runs/4m-golden.yaml")
    flat = to_flat_dict(spec)
    assert flat["model"].pop("ska_beta_policy") == "learned"
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(flat, sort_keys=True))

    with pytest.warns(DeprecationWarning, match="ska_beta_policy"):
        loaded = load_materialized_spec(path)
    assert loaded.model.ska_beta_policy == "learned"
    # And it is still the same run: the migration restores the value the run
    # actually had, so its directory name does not move.
    assert run_id(loaded) == "2e63f16e"


def test_the_committed_identity_baseline_was_not_regenerated():
    """A guard on the guard.

    `test_identity_baseline.py` is a pin only for as long as its JSON is the
    pre-refactor one. The cheapest way for a branch to make an identity
    regression disappear is to re-run `scripts/gen_identity_baseline.py`, which
    that file's docstring forbids and which nothing detects. These two literals
    are transcribed from the committed baseline, so re-generating it would leave
    them behind and this test would fail with "the baseline moved" rather than
    with silence.

    Values checked against the baseline as committed at 4814812. 4m-golden is
    deliberately not among them: it postdates the baseline, which is why its
    identity is pinned directly in
    `test_the_pinned_golden_identity_survives_the_new_field` instead.
    """
    baseline = json.loads(
        (REPO_ROOT / "code-tests/identity_baseline.json").read_text())
    assert config_hash(build_config("1m")) == baseline["config_hash"]["1m"]
    assert config_hash(build_config("50m")) == baseline["config_hash"]["50m"]
    assert baseline["config_hash"]["1m"] == (
        "10f09f4bf96514cc992553fe032e2322388caba10245813f22d2dbbf457a254a")
    assert set(baseline["run_spec"]) >= {
        "configs/runs/50m-first-real.yaml", "configs/runs/50m-mqar-smoke.yaml"}
