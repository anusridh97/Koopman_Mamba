"""The three precision fields on KoopmanLMConfig, and what they refuse.

This is the identity-breaking step. Three new fields change `config_hash` for all
11 registry configs, and therefore every `group_id` and `run_id` derived from
them, so `identity_baseline.json` is regenerated in the same commit with the
old->new mapping recorded in
`docs/superpowers/specs/2026-08-19-precision-identity-mapping.md`.

The defaults are chosen to describe what the code already does -- bf16 autocast,
an fp32 SKA core, an untouched MLP path -- so the fields state existing behaviour
rather than changing it. Nothing consumes them yet; the wiring is the design's
steps 3 onward.

Validation lives in `__post_init__` so a bad combination fails at config-load
time, on a login node, before a GPU is allocated.
"""
import dataclasses

import pytest

pytestmark = pytest.mark.correctness


def _cfg(**overrides):
    from koopman_lm.config import KoopmanLMConfig
    return KoopmanLMConfig(**overrides)


# ------------------------------------------------------------- defaults ----

def test_the_three_fields_exist_with_the_documented_defaults():
    cfg = _cfg()
    assert cfg.compute_precision == "bf16"
    assert cfg.ska_precision == "fp32"
    assert cfg.mlp_precision is None


def test_the_defaults_describe_what_the_code_already_did():
    """bf16 autocast is what train.py does; the SKA core already casts to fp32;
    the MLP path is untouched. If a default here changed behaviour, the fields
    would be a numerics change wearing a config change's clothes."""
    from koopman_lm.precision import dtype_of
    import torch

    cfg = _cfg()
    assert dtype_of(cfg.compute_precision) is torch.bfloat16
    assert dtype_of(cfg.ska_precision) is torch.float32
    assert cfg.mlp_precision is None, "None is a strict no-op, not fp32"


def test_the_shipped_configs_all_load(tmp_path):
    from koopman_lm.config import CONFIG_REGISTRY, build_config

    for name in CONFIG_REGISTRY:
        cfg = build_config(name)
        assert cfg.compute_precision in ("fp32", "bf16", "fp16")


# ----------------------------------------------------------- ska domain ----

@pytest.mark.parametrize("bad", ["bf16", "fp16"])
def test_ska_precision_refuses_the_low_precisions(bad):
    """The core takes a Cholesky of a Gram matrix; bf16's 8 mantissa bits make
    that unreliable, which is part of why ska_ridge exists. The error has to say
    so, or the next person just widens the domain."""
    with pytest.raises(ValueError, match="cholesky|Cholesky"):
        _cfg(ska_precision=bad)


@pytest.mark.parametrize("good", ["fp32", "fp64"])
def test_ska_precision_accepts_fp32_and_fp64(good):
    """fp64 is not hypothetical: the exact prefix-scan path already accepts
    float64, and fp64 is used for numerical validation across kernels/."""
    assert _cfg(ska_precision=good, ska_backend="pytorch").ska_precision == good


def test_fp64_ska_is_rejected_with_the_fused_cuda_kernel():
    """The fused kernel is fp32-only, enforced in the .cu and again in the python
    wrapper. Failing at config load beats a CUDA error mid-run."""
    with pytest.raises(ValueError, match="cuda_prefix"):
        _cfg(ska_precision="fp64", ska_backend="cuda_prefix")


def test_the_fused_kernel_error_names_both_enforcement_sites():
    """So the constraint is discoverable from the message rather than by reading
    two files."""
    with pytest.raises(ValueError) as excinfo:
        _cfg(ska_precision="fp64", ska_backend="cuda_prefix")
    message = str(excinfo.value)
    assert "prefix_scan_ext.cu" in message
    assert "cuda_prefix_scan.py" in message


def test_fp32_ska_is_fine_with_the_fused_kernel():
    """The default combination, and the one every production config uses."""
    assert _cfg(ska_precision="fp32", ska_backend="cuda_prefix").ska_precision == "fp32"


def test_fp64_ska_is_allowed_on_the_pytorch_path():
    assert _cfg(ska_precision="fp64", ska_backend="auto").ska_precision == "fp64"


# ----------------------------------------------------------- mlp domain ----

def test_mlp_precision_accepts_none_and_the_raised_precisions():
    for value in (None, "fp32", "fp64"):
        assert _cfg(mlp_precision=value).mlp_precision == value


@pytest.mark.parametrize("bad", ["bf16", "fp16"])
def test_mlp_precision_refuses_the_low_precisions(bad):
    with pytest.raises(ValueError, match="mlp_precision"):
        _cfg(mlp_precision=bad)


# ------------------------------------------------------- compute domain ----

@pytest.mark.parametrize("good", ["fp32", "bf16", "fp16"])
def test_compute_precision_accepts_the_autocast_dtypes(good):
    assert _cfg(compute_precision=good).compute_precision == good


def test_compute_precision_refuses_fp64():
    """Excluding fp64 here is what makes "components may only raise" impossible
    to violate without an ordering check."""
    with pytest.raises(ValueError, match="compute_precision"):
        _cfg(compute_precision="fp64")


def test_an_unknown_precision_name_is_rejected():
    with pytest.raises(ValueError):
        _cfg(compute_precision="bfloat16")


# ----------------------------------------------------------- sweepable ----

def test_the_fields_are_reachable_as_sweep_overrides():
    """The design's reason for three flat fields rather than a nested policy
    object: sweep/spec.py splits an axis key on the FIRST dot only, so
    model.precision.default would arrive as a field literally named
    "precision.default" and fail as an unexpected keyword."""
    from koopman_lm.config import build_config
    from experimentation.sweep.spec import build_cell_run_spec

    sections = {
        "model": dataclasses.asdict(build_config("50m")),
        "data": {"kind": "shard", "shard_dir": "/tmp/s",
                 "tokenizer": "t", "mix": {"f": 1.0}, "n_tokens": 10},
        "optim": {"lr": 4e-4, "warmup_steps": 10, "max_steps": 100,
                  "effective_batch": 16},
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }
    spec = build_cell_run_spec("precision-sweep", sections, {
        "model.compute_precision": "fp32",
        "model.ska_precision": "fp64",
        "model.ska_backend": "pytorch",
    })
    assert spec.model.compute_precision == "fp32"
    assert spec.model.ska_precision == "fp64"


def test_precision_participates_in_run_identity():
    """The point of putting these on the model config rather than RuntimeSpec:
    two runs differing only in compute precision are different experiments and
    must not collide on run_id."""
    from koopman_lm.config import build_config, config_hash

    import dataclasses as dc
    base = dc.asdict(build_config("50m"))
    from koopman_lm.config import KoopmanLMConfig
    bf16 = KoopmanLMConfig(**dict(base, compute_precision="bf16"))
    fp32 = KoopmanLMConfig(**dict(base, compute_precision="fp32"))
    assert config_hash(bf16) != config_hash(fp32)


# --------------------------------------------------------- round tripping ----

def test_the_new_fields_survive_a_yaml_round_trip():
    import yaml

    from koopman_lm.config import KoopmanLMConfig, build_config, config_hash

    cfg = build_config("50m")
    rebuilt = KoopmanLMConfig(
        **yaml.safe_load(yaml.safe_dump(dataclasses.asdict(cfg))))
    assert rebuilt == cfg
    assert config_hash(rebuilt) == config_hash(cfg)
    assert rebuilt.mlp_precision is None, "None must not round-trip to a string"
