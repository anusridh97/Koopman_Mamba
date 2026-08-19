"""runtime.precision vs model.compute_precision (precision design step 4).

Two fields now describe the same thing, and that is deliberate for one release:
`RuntimeSpec(**d)` raises TypeError on unknown keys, all three
`configs/runs/*.yaml` set `runtime.precision`, and so does every materialized
`spec.yaml` on scratch -- which `load_materialized_spec` reads when a run
*resumes*. Deleting the field would break resume for runs already on disk. So it
stays, is validated, and is removed in a later cleanup once nothing carries it.

Note the asymmetry the design points out: the model section has a migration story
(`_check_model_key_set` raises listing exactly which keys drifted); the runtime
section has none. That is why deletion has to wait.

The disagreement check must live on `RunSpec`, not `RuntimeSpec`: RuntimeSpec is a
standalone frozen dataclass with no access to `spec.model`, and only RunSpec holds
both sections. It raises rather than silently preferring one, because "which of
these two fields wins" is not a question a reader should have to answer from
source.

Step 4 is a no-op at the defaults -- both spell bf16 -- and the gate the design
names is that existing spec.yaml files still load.
"""
import pytest

pytestmark = pytest.mark.correctness


def _spec(compute="bf16", runtime_precision="bf16", **runtime_kwargs):
    import dataclasses

    from koopman_lm.config import KoopmanLMConfig, build_config
    from experimentation.run.spec import (OptimSpec, RunSpec, RuntimeSpec,
                                          ShardDataSpec)

    model = KoopmanLMConfig(
        **dict(dataclasses.asdict(build_config("50m")), compute_precision=compute))
    return RunSpec(
        name="x", model=model,
        data=ShardDataSpec(kind="shard", shard_dir="/tmp/s", tokenizer="t",
                           mix={"f": 1.0}, n_tokens=10),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100,
                        effective_batch=64),
        runtime=RuntimeSpec(per_device_batch_size=16, precision=runtime_precision,
                            **runtime_kwargs),
    )


# ------------------------------------------------------- the agreement ----

def test_runtime_precision_defaults_to_following_the_model():
    """None, not bf16. A concrete default would make model.compute_precision
    unsweepable: overriding it in a cell would disagree with an untouched
    runtime.precision and the cell would fail to construct. Same "None means
    follow" pattern the design uses for mlp_precision."""
    from experimentation.run.spec import RuntimeSpec

    assert RuntimeSpec().precision is None
    assert _spec(runtime_precision=None).model.compute_precision == "bf16"


@pytest.mark.parametrize("value", ["bf16", "fp32"])
def test_matching_values_are_accepted(value):
    assert _spec(compute=value, runtime_precision=value).model.compute_precision == value


def test_a_disagreement_is_rejected_rather_than_resolved():
    """Silently preferring one would make "what precision did this run use?"
    answerable only by reading source -- the exact question this design exists to
    make answerable from the config."""
    with pytest.raises(ValueError, match="precision"):
        _spec(compute="bf16", runtime_precision="fp32")


def test_the_disagreement_message_names_both_fields_and_both_values():
    with pytest.raises(ValueError) as excinfo:
        _spec(compute="fp32", runtime_precision="bf16")
    message = str(excinfo.value)
    assert "runtime.precision" in message
    assert "model.compute_precision" in message
    assert "bf16" in message and "fp32" in message


def test_an_unset_runtime_precision_cannot_disagree():
    """The property that keeps compute_precision sweepable."""
    for compute in ("fp32", "bf16"):
        assert _spec(compute=compute,
                     runtime_precision=None).model.compute_precision == compute


def test_compute_precision_survives_a_sweep_override():
    """The regression this sentinel exists to prevent: a cell overriding only the
    model section must still construct."""
    import dataclasses

    from koopman_lm.config import build_config
    from experimentation.sweep.spec import build_cell_run_spec

    sections = {
        "model": dataclasses.asdict(build_config("50m")),
        "data": {"kind": "shard", "shard_dir": "/tmp/s", "tokenizer": "t",
                 "mix": {"f": 1.0}, "n_tokens": 10},
        "optim": {"lr": 4e-4, "warmup_steps": 10, "max_steps": 100,
                  "effective_batch": 16},
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }
    spec = build_cell_run_spec("precision-sweep", sections,
                              {"model.compute_precision": "fp32"})
    assert spec.model.compute_precision == "fp32"


def test_the_check_lives_on_runspec_not_runtimespec():
    """RuntimeSpec cannot see spec.model, so a check there could only ever be a
    guess. Constructing a RuntimeSpec alone must therefore stay legal."""
    from experimentation.run.spec import RuntimeSpec

    assert RuntimeSpec(precision="fp32").precision == "fp32"


# --------------------------------------------------- the derived flag ----

def test_the_bf16_flag_is_derived_from_the_model_config():
    """The whole point of step 4: the trainer's precision comes from the field
    that is hashed into run identity, not from the one that is not."""
    from experimentation.run.train_argv import build_train_argv

    argv = build_train_argv(_spec(compute="bf16", runtime_precision="bf16"), "/tmp/run")
    assert "--bf16" in argv and "--no_bf16" not in argv


def test_fp32_derives_the_negative_flag():
    from experimentation.run.train_argv import build_train_argv

    argv = build_train_argv(_spec(compute="fp32", runtime_precision="fp32"), "/tmp/run")
    assert "--no_bf16" in argv and "--bf16" not in argv


def test_runtime_precision_can_mirror_every_compute_precision():
    """The deprecated field has to be able to agree with the new one, or an fp16
    config could never satisfy the cross-check. Widening its domain is what makes
    the fp16 test below discriminating rather than accidentally green."""
    from experimentation.run.spec import RuntimeSpec

    for value in ("fp32", "bf16", "fp16"):
        assert RuntimeSpec(precision=value).precision == value


def test_fp16_is_refused_by_the_argv_builder_not_silently_trained_in_fp32():
    """train.py's CLI has only --bf16/--no_bf16, so an fp16 config would map onto
    --no_bf16 and train in fp32 while the config claimed otherwise -- precisely
    the class of bug this design exists to remove. It raises until the training
    side lands (design step 5).

    The RunSpec is constructed first and asserted valid, so the raise provably
    comes from build_train_argv rather than from RuntimeSpec rejecting fp16
    upstream. An earlier version of this test passed for that wrong reason."""
    from experimentation.run.train_argv import build_train_argv

    spec = _spec(compute="fp16", runtime_precision="fp16")
    assert spec.model.compute_precision == "fp16"      # the spec itself is fine

    with pytest.raises(ValueError, match="fp16"):
        build_train_argv(spec, "/tmp/run")


def test_the_argv_builder_reads_the_model_field_not_the_runtime_one():
    """The bf16/fp32 cases above agree in both fields, so they cannot tell which
    one was read. This can: the two fields are required to agree, so the only
    discriminating evidence available is which one the code names."""
    import inspect

    from experimentation.run import train_argv

    source = inspect.getsource(train_argv.build_train_argv)
    assert "spec.model.compute_precision" in source, (
        "build_train_argv must derive precision from model.compute_precision")
    # The attribute ACCESS, not the words: the function's comments name the
    # deprecated field while explaining why they no longer read it, and an
    # earlier version of this assertion failed on that prose.
    assert "spec.runtime.precision" not in source, (
        "the deprecated field must no longer drive the flag")


# ------------------------------------------------------ compatibility ----

def test_an_existing_materialized_spec_still_loads(tmp_path):
    """The gate the design names for step 4. A spec.yaml on scratch carries
    runtime.precision, and load_materialized_spec reads it on resume."""
    import yaml

    from experimentation.run.resolve import load_materialized_spec, to_flat_dict

    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(to_flat_dict(_spec()), sort_keys=False))
    reloaded = load_materialized_spec(path)
    assert reloaded.model.compute_precision == "bf16"


def test_the_shipped_run_specs_all_resolve():
    """All three set runtime.precision, and none sets compute_precision, so they
    only load if the default agrees -- which is the no-op claim, checked."""
    from pathlib import Path

    from experimentation.run.resolve import resolve_run_spec

    root = Path(__file__).resolve().parent.parent
    for rel in sorted((root / "configs" / "runs").glob("*.yaml")):
        spec = resolve_run_spec(rel)
        # Each sets runtime.precision explicitly, so each must agree.
        assert spec.runtime.precision in (None, spec.model.compute_precision), rel.name
