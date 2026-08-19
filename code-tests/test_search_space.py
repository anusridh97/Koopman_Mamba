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
    ("proxy_chunked", {"model.ska_prefix_scan": False, "model.ska_backend": "auto"}),
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
    here except proxy_chunked sets prefix_scan, so the warning has to be
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
