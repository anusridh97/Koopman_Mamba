"""experimentation/sweep/search/space.py -- the space, declared exactly once.

`sweep/spec.py`'s rule is that a sweep's grid is declared in one place. This is
the same rule one level up: the searchable space lives here, and nowhere else
knows the bounds.

Two things this module must get right.

**Distributions are plain data, not optuna objects.** `study.py` translates them.
That keeps the whole space declarable and testable with no optuna installed, and
means swapping the sampler later touches one file.

**`params_to_overrides` emits `"<section>.<field>"` keys**, the vocabulary
`sweep/spec.py::build_cell_run_spec` already accepts. That single choice is what
lets a trial reuse every guarantee the run system provides -- content-hashed
run_id, the dirty-tree gate, verify_shard, the Slurm array -- instead of
reimplementing them. Three of the sampled parameters are not 1:1 config fields
(count+placement become ska_layer_indices, norm_clip_multiplier becomes
ska_norm_clip_c, warmup_ratio becomes warmup_steps), and deriving them is this
module's actual job.
"""
import math

import pytest

pytestmark = pytest.mark.correctness

_SECTIONS = ("model", "data", "optim", "runtime")


def _base_model():
    from koopman_lm.config import build_config
    return build_config("50m")


def _baseline_params():
    """The parameters that should reproduce configs/50m.yaml exactly."""
    cfg = _base_model()
    return {
        "ska_rank": cfg.ska_rank,                       # 24
        "n_ska_layers": len(cfg.ska_layer_indices),     # 4
        "placement": "baseline",
        "ska_ridge": cfg.ska_ridge,                     # 0.01
        "ska_layerscale_init": cfg.ska_layerscale_init,  # 0.01
        "norm_clip_multiplier": cfg.ska_norm_clip_c / math.sqrt(cfg.ska_rank),
        "gamma_value": cfg.ska_gamma_value,             # 1.0
        "ska_power_K": cfg.ska_power_K,                  # 1
        "learning_rate": 4.0e-4,
        "weight_decay": 0.1,
        "warmup_ratio": 0.02,
        "grad_clip": 1.0,
    }


# ------------------------------------------------------- override shape ----

def test_overrides_use_only_valid_section_dot_field_keys():
    from experimentation.sweep.search.space import params_to_overrides

    overrides = params_to_overrides(_baseline_params(), _base_model(), max_steps=15000)
    assert overrides, "must emit something"
    for key in overrides:
        section, _, field = key.partition(".")
        assert field, f"{key!r} is not '<section>.<field>'"
        assert section in _SECTIONS, f"{key!r} names an unknown section"


def test_overrides_build_a_real_runspec_through_the_sweep_machinery():
    """The whole point: a trial's parameters go through the same code path a
    hand-written `cells:` entry does, and come out as a validated RunSpec with a
    content-hashed identity."""
    from experimentation.sweep.spec import build_cell_run_spec
    from experimentation.sweep.search.space import params_to_overrides
    from experimentation.run.spec import run_id
    from koopman_lm.config import KoopmanLMConfig
    import dataclasses

    sections = {
        "model": dataclasses.asdict(_base_model()),
        "data": {"kind": "shard", "shard_dir": "/tmp/shard",
                 "tokenizer": "NousResearch/Llama-2-7b-hf",
                 "mix": {"fineweb": 1.0}, "n_tokens": 1000},
        "optim": {"lr": 4.0e-4, "warmup_steps": 300, "max_steps": 15000,
                  "effective_batch": 96},
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }
    overrides = params_to_overrides(_baseline_params(), _base_model(), max_steps=15000)
    spec = build_cell_run_spec("ska-search", sections, overrides)
    assert isinstance(spec.model, KoopmanLMConfig)
    assert len(run_id(spec)) == 8


# ------------------------------------------------- baseline containment ----

def test_baseline_params_reproduce_the_shipped_config_exactly():
    """If the reference trial is not bit-identical to configs/50m.yaml, every
    comparison against it is against something else."""
    from experimentation.sweep.search.space import params_to_overrides

    cfg = _base_model()
    overrides = params_to_overrides(_baseline_params(), cfg, max_steps=15000)
    assert overrides["model.ska_rank"] == cfg.ska_rank
    assert overrides["model.ska_layer_indices"] == list(cfg.ska_layer_indices)
    assert overrides["model.ska_norm_clip_c"] == pytest.approx(cfg.ska_norm_clip_c)
    assert overrides["model.ska_ridge"] == pytest.approx(cfg.ska_ridge)
    assert overrides["model.ska_layerscale_init"] == pytest.approx(cfg.ska_layerscale_init)


def test_baseline_params_preserve_the_shipped_parameter_count():
    from experimentation.sweep.search.space import params_to_overrides
    from koopman_lm.config import KoopmanLMConfig
    import dataclasses

    cfg = _base_model()
    merged = dataclasses.asdict(cfg)
    for key, value in params_to_overrides(_baseline_params(), cfg, max_steps=15000).items():
        section, _, field = key.partition(".")
        if section == "model":
            merged[field] = value
    assert KoopmanLMConfig(**merged).param_count_estimate() == cfg.param_count_estimate()


def test_the_declared_space_contains_every_baseline_value():
    from experimentation.sweep.search.space import search_space

    cfg = _base_model()
    space = search_space(cfg)
    assert cfg.ska_rank in space["ska_rank"]["choices"]
    assert len(cfg.ska_layer_indices) in space["n_ska_layers"]["choices"]
    assert "baseline" in space["placement"]["choices"]
    baseline_multiplier = cfg.ska_norm_clip_c / math.sqrt(cfg.ska_rank)
    assert any(m == pytest.approx(baseline_multiplier)
               for m in space["norm_clip_multiplier"]["choices"]), (
        "the baseline's own norm-clip multiplier must be selectable, or rank 24 "
        "can never reproduce ska_norm_clip_c=4.0")
    low, high = space["ska_ridge"]["low"], space["ska_ridge"]["high"]
    assert low <= cfg.ska_ridge <= high


# --------------------------------------------------------- derivations ----

def test_norm_clip_multiplier_scales_with_sqrt_rank():
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), ska_rank=32, norm_clip_multiplier=1.25)
    overrides = params_to_overrides(params, _base_model(), max_steps=15000)
    assert overrides["model.ska_norm_clip_c"] == pytest.approx(1.25 * math.sqrt(32))


def test_warmup_ratio_becomes_absolute_warmup_steps():
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), warmup_ratio=0.04)
    assert params_to_overrides(params, _base_model(), max_steps=15000)["optim.warmup_steps"] == 600


def test_warmup_steps_is_at_least_one():
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), warmup_ratio=0.0001)
    assert params_to_overrides(params, _base_model(), max_steps=10)["optim.warmup_steps"] == 1


def test_count_and_placement_become_layer_indices():
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), n_ska_layers=6, placement="late")
    indices = params_to_overrides(params, _base_model(), max_steps=15000)["model.ska_layer_indices"]
    assert len(indices) == 6
    assert indices == sorted(set(indices))


def test_optim_parameters_land_in_the_optim_section():
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), learning_rate=3.1e-4, weight_decay=0.15, grad_clip=0.5)
    overrides = params_to_overrides(params, _base_model(), max_steps=15000)
    assert overrides["optim.lr"] == pytest.approx(3.1e-4)
    assert overrides["optim.weight_decay"] == pytest.approx(0.15)
    assert overrides["optim.grad_clip"] == pytest.approx(0.5)


# ------------------------------------------------------ policy and pins ----

def test_the_modern_eta_gamma_policy_is_pinned_regardless_of_the_base_config():
    """Five tier-2 configs inherit the OLD eta/gamma policy by omission (learnable
    eta and gamma, clamped). A search that silently inherited that from whichever
    base it was pointed at would be comparing across two different
    parameterisations. Pin it explicitly."""
    from experimentation.sweep.search.space import params_to_overrides

    overrides = params_to_overrides(_baseline_params(), _base_model(), max_steps=15000)
    assert overrides["model.ska_eta_learnable"] is False
    assert overrides["model.ska_gamma_learnable"] is False
    assert overrides["model.ska_layerscale"] is True
    assert overrides["model.ska_norm_clip"] is True


def test_rank_must_be_a_multiple_of_eight():
    """Mirrors KoopmanLMConfig.__post_init__'s own assert, but fails at the
    search boundary with a message about the search rather than deep inside
    model construction."""
    from experimentation.sweep.search.space import params_to_overrides

    with pytest.raises(ValueError, match="multiple of 8"):
        params_to_overrides(dict(_baseline_params(), ska_rank=20),
                            _base_model(), max_steps=15000)


@pytest.mark.parametrize("policy,expected", [
    ("exact_auto", {"model.ska_prefix_scan": True, "model.ska_backend": "auto"}),
    ("fused_only", {"model.ska_prefix_scan": True, "model.ska_backend": "cuda_prefix"}),
    ("exact_invchol", {"model.ska_prefix_scan": False,
                       "model.ska_inverse_cholesky": True}),
])
def test_backend_policy_sets_the_expected_fields(policy, expected):
    from experimentation.sweep.search.space import params_to_overrides

    params = dict(_baseline_params(), ska_rank=24)
    overrides = params_to_overrides(params, _base_model(), max_steps=15000,
                                    backend_policy=policy)
    for key, value in expected.items():
        assert overrides[key] == value


def test_fused_only_rejects_any_rank_but_the_kernel_shape():
    """The fused SM100 kernel is specialised to rank 24; asking for anything else
    under that policy is a configuration error, not a slow path."""
    from experimentation.sweep.search.space import params_to_overrides

    with pytest.raises(ValueError, match="fused_only"):
        params_to_overrides(dict(_baseline_params(), ska_rank=32), _base_model(),
                            max_steps=15000, backend_policy="fused_only")


def test_unknown_backend_policy_is_rejected():
    from experimentation.sweep.search.space import params_to_overrides

    with pytest.raises(ValueError, match="backend policy"):
        params_to_overrides(_baseline_params(), _base_model(), max_steps=15000,
                            backend_policy="magic")


def test_gradient_checkpointing_hazard_is_documented_next_to_prefix_scan():
    """train_argv.py:91-106 records that ska_prefix_scan=True with
    torch.compile/gradient-checkpointing dies with
    cudaErrorStreamCaptureInvalidated on H100 (job 415208). Every backend policy
    here except exact_invchol sets prefix_scan, so the warning has to be
    reachable from this module rather than only from the run layer."""
    import experimentation.sweep.search.space as space

    assert "415208" in space.__doc__ or "cudagraph" in space.__doc__.lower()


# --------------------------------------------------------- declarations ----

def test_distributions_are_plain_data_not_optuna_objects():
    from experimentation.sweep.search.space import search_space

    space = search_space(_base_model())
    for name, decl in space.items():
        assert isinstance(decl, dict), f"{name} must be a plain dict"
        assert decl["kind"] in ("categorical", "float"), f"{name}: {decl['kind']}"
        if decl["kind"] == "categorical":
            assert isinstance(decl["choices"], list) and decl["choices"]
        else:
            assert decl["low"] < decl["high"]


def test_every_declared_parameter_is_consumed_by_params_to_overrides():
    """A declared parameter nothing reads is a silent no-op trial axis: the
    sampler would spend trials varying it and the config would never change."""
    from experimentation.sweep.search.space import params_to_overrides, search_space

    cfg = _base_model()
    declared = set(search_space(cfg))
    baseline = _baseline_params()
    assert declared == set(baseline), (
        f"declared but unused: {sorted(declared - set(baseline))}; "
        f"used but undeclared: {sorted(set(baseline) - declared)}")
    # and each one actually moves the output
    reference = params_to_overrides(baseline, cfg, max_steps=15000)
    for name in declared:
        if name == "placement":
            probe = dict(baseline, placement="late")
        elif isinstance(baseline[name], str):
            continue
        elif isinstance(baseline[name], bool):
            probe = dict(baseline, **{name: not baseline[name]})
        else:
            probe = dict(baseline, **{name: baseline[name] * 1.5 + 1})
        if name == "ska_rank":
            probe[name] = 32
        if name == "n_ska_layers":
            probe[name] = 6
        assert params_to_overrides(probe, cfg, max_steps=15000) != reference, (
            f"changing {name} changed no override -- it is a dead search axis")


def test_space_imports_without_optuna():
    import experimentation.sweep.search.space as space

    assert "optuna" not in space.__dict__

# ------------------------------------------- what the measurement decided ----
#
# Jobs 440122 / 440135 measured all four SKA routes on an H100 against
# prefix_scan.dense_exact_oracle in fp64. The result changed which policy can be
# a default, so it is pinned here rather than left in a report:
#
#   route            fp64 error vs oracle   full-model micro-step vs chunked
#   chunked           0.92 - 1.52 (!)        1.00x   (the baseline)
#   prefix_scan ref   <= 4.7e-13             160x at 4m; fused 1.01x at rank 24
#   inverse_cholesky  <= 5.1e-13             0.92x at 4m, 1.41x at 50m
#   exact_intrachunk  <= 5.0e-13             57x at 4m, 87x at 50m
#
# Two consequences, and each has a test below.

def test_inverse_cholesky_is_the_default_policy_everywhere():
    """The measurement removed the reason for any other default: route 3 is exact
    to 1e-13 in fp64 and costs 0.92x the approximation at the 4m geometry. A
    default of `exact_auto` bought 160x for the same answer; a default of
    `proxy_chunked` bought a 92%-152% wrong operator for nothing.

    Checked at every signature that carries the default rather than only at
    space.py's, because a study's policy can enter from five places and four
    agreeing with one disagreeing is how the old default survived.
    """
    import dataclasses
    import inspect
    import pathlib
    import re

    from experimentation.sweep.search import anchors, space, studyspec

    assert space.BACKEND_POLICIES[0] == "exact_invchol", \
        "the first policy is what a reader takes as canonical"
    for fn in (space.params_to_overrides, anchors.designs_to_cells):
        got = inspect.signature(fn).parameters["backend_policy"].default
        assert got == "exact_invchol", f"{fn.__qualname__} still defaults to {got!r}"
    # Read the field default rather than constructing: StudySpec has required
    # fields on purpose (a study with no name or base config is not a study).
    fields = {f.name: f for f in dataclasses.fields(studyspec.StudySpec)}
    assert fields["backend_policy"].default == "exact_invchol"

    # driver.py and report.py import optuna, which the CPU env deliberately does
    # not have (pyproject keeps it in [lab] so a space can be authored without
    # it). Read their defaults out of the source rather than skipping them --
    # they are two of the five sites, and a skipped assertion is not one.
    root = pathlib.Path(__file__).resolve().parents[1] / "experimentation/sweep/search"
    for name in ("driver.py", "report.py"):
        src = (root / name).read_text()
        # Matches both spellings that occur: `backend_policy: str = "..."` (a
        # signature default) and `backend_policy="..."` (a forced call site).
        found = re.findall(r'backend_policy(?:\s*:\s*str)?\s*=\s*"([a-z_]+)"', src)
        assert found, f"{name} names no backend policy at all any more"
        assert set(found) == {"exact_invchol"}, f"{name} still names {sorted(set(found))}"


@pytest.mark.parametrize("policy", ["exact_invchol", "fused_only", "exact_auto"])
def test_every_policy_selects_exactly_one_route(policy):
    """ska.py dispatches on three independent booleans in an if/elif chain, so
    two set at once silently runs whichever comes first. Every policy must pin
    all three -- including the two it turns off -- or the base spec's value
    leaks in and decides which of two exact routes ran.
    """
    from experimentation.sweep.search.space import params_to_overrides

    overrides = params_to_overrides(dict(_baseline_params(), ska_rank=24),
                                    _base_model(), max_steps=15000,
                                    backend_policy=policy)
    flags = ("model.ska_prefix_scan", "model.ska_inverse_cholesky",
             "model.ska_exact_intrachunk")
    for flag in flags:
        assert flag in overrides, f"{policy} leaves {flag} to the base spec"
    assert sum(bool(overrides[f]) for f in flags) == 1, \
        f"{policy} sets {[f for f in flags if overrides[f]]} -- not exactly one route"


def test_the_chunked_policy_is_retired_by_name():
    """`proxy_chunked` was not a cheap screen. It is 92%-152% wrong in the
    forward and 93%-101% wrong in the gradients, at both geometries, on random
    and structured inputs, with no dependence on T / ridge / K -- and
    `ska_rank`, `ska_ridge` and `ska_norm_clip_c` enter the model ONLY through
    the operator it gets wrong, so it cannot screen the things the search
    samples.

    Rejected with its own message rather than falling through to
    'unknown policy', because the caller most likely to hit this is someone
    re-running an archived config and they need to know it was measured, not
    renamed.
    """
    from experimentation.sweep.search.space import params_to_overrides

    with pytest.raises(ValueError, match="exact_invchol") as exc:
        params_to_overrides(_baseline_params(), _base_model(), max_steps=15000,
                            backend_policy="proxy_chunked")
    msg = str(exc.value)
    assert "440122" in msg, "cite the job that measured it"
    assert "exact_invchol" in msg, "name the replacement"


def test_exact_auto_warns_about_the_rank_cliff_not_a_constant_slowdown():
    """The '137x' figure reads as a fixed tax. It is not: the fused kernel needs
    rank EXACTLY 24, and the search samples {8, 16, 24, 32}, so within one study
    exact_auto is 0.0043 s at rank 24 and 0.72-3.19 s at the other three -- a
    167x-738x discontinuity correlated with a searched variable. A sampler
    reading wall-clock across that cliff is being told rank 24 is free.
    """
    import experimentation.sweep.search.space as space

    doc = space.__doc__
    assert "cliff" in doc.lower(), "the discontinuity has to be named as one"
    assert "440122" in doc or "440135" in doc, "cite the measurement"


# ------------------------------------------------------------- power_K ----

def test_power_K_is_searched_rather_than_pinned():
    """It used to be pinned to 1 while `configs/runs/4m-golden.yaml` pins 2, so a
    study on that base compared every trial at K=1 against a baseline at K=2 --
    trials internally consistent, but "we beat the baseline" confounded on one
    axis. Searching it removes the confound instead of picking a side, and costs
    nothing: job 440135 measured route 3 at 0.00542 s (K=1) vs 0.00559 s (K=2).
    """
    from experimentation.sweep.search.space import params_to_overrides, search_space

    decl = search_space(_base_model())["ska_power_K"]
    assert decl["kind"] == "categorical"
    assert set(decl["choices"]) >= {1, 2}, \
        "both values any config in this repo uses must be reachable"
    assert all(isinstance(c, int) for c in decl["choices"]), \
        "power_K indexes a matrix power; a float would silently truncate"

    for k in (1, 2):
        overrides = params_to_overrides(dict(_baseline_params(), ska_power_K=k),
                                        _base_model(), max_steps=15000)
        assert overrides["model.ska_power_K"] == k


def test_a_base_config_pinning_power_K_stays_reachable():
    """Baseline containment, the same rule the rest of the space follows: a space
    that cannot express the config you already run cannot tell you whether you
    improved on it. 4m-golden pins 2, configs/50m.yaml pins 1."""
    import dataclasses

    from experimentation.sweep.search.space import search_space

    for value in (1, 2, 3):
        cfg = dataclasses.replace(_base_model(), ska_power_K=value)
        assert value in search_space(cfg)["ska_power_K"]["choices"]


def test_an_archived_trial_without_power_K_promotes_at_one():
    """report.py replays `trial.params` from a journal through
    params_to_overrides. Trials recorded BEFORE power_K was searchable have no
    such key and ran at the pinned value 1 -- so the fallback has to be 1, not
    the base config's value, or promoting an old study would confirm something
    the trial never ran.
    """
    import dataclasses

    from experimentation.sweep.search.space import params_to_overrides

    # The base MUST pin something other than 1 here, or the wrong implementation
    # (falling back to base_model.ska_power_K) coincides with the right one and
    # this test is a green guard. configs/50m.yaml pins 1; 4m-golden pins 2, and
    # 4m-golden is the base the first real study runs on.
    cfg = dataclasses.replace(_base_model(), ska_power_K=2)
    archived = _baseline_params()
    del archived["ska_power_K"]
    overrides = params_to_overrides(archived, cfg, max_steps=15000)
    assert overrides["model.ska_power_K"] == 1, (
        "an archived trial ran at the pinned K=1; promoting it at the base "
        "config's K would confirm something it never ran")
