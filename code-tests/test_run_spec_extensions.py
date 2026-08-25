"""§6/§7 on the spec layer: OptimSpec.groups and RunSpec.schedules.

Both are *partial* maps -- absence is the norm -- so they are optional, and
both must be hashed into run_id/group_id and survive materialize -> load.

Pure-Python, no torch model instantiation. See
docs/superpowers/specs/2026-08-08-extension-mechanisms-design.md §6.1, §6.4, §7.
"""
import dataclasses

import pytest

pytestmark = pytest.mark.correctness


def _optim(**kw):
    from experimentation.run.spec import OptimSpec
    base = dict(lr=4.0e-4, warmup_steps=10, max_steps=100)
    base.update(kw)
    return OptimSpec(**base)


def _spec(**kw):
    from koopman_lm.config import build_config
    from experimentation.run.spec import RunSpec, RuntimeSpec, ShardDataSpec
    base = dict(
        name="t",
        model=build_config("1m"),
        data=ShardDataSpec(shard_dir="/d", tokenizer="tok",
                           mix={"fineweb": 1.0}, n_tokens=10),
        optim=_optim(),
        runtime=RuntimeSpec(),
    )
    base.update(kw)
    return RunSpec(**base)


# --------------------------------------------------------------------------
# §7 OptimSpec.groups
# --------------------------------------------------------------------------

def test_optim_spec_defaults_to_no_groups():
    assert _optim().groups == ()


def test_optim_spec_parses_yaml_dicts_into_group_specs():
    from experimentation.training.optim import ParamGroupSpec

    o = _optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}])
    assert o.groups == (ParamGroupSpec(match="*.ska.*", lr_mult=10.0),)


def test_optim_spec_rejects_a_malformed_group():
    from experimentation.run.spec import OptimSpec

    with pytest.raises((ValueError, TypeError)):
        _optim(groups=[{"lr_mult": 10.0}])          # no 'match'
    with pytest.raises((ValueError, TypeError)):
        _optim(groups=[{"match": "a", "lr": 1.0}])  # absolute lr is not a knob


def test_optim_spec_groups_asdict_round_trips_through_yaml():
    import yaml
    from experimentation.run.spec import OptimSpec

    o = _optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0},
                       {"match": "embed.weight", "weight_decay": 0.0}])
    blob = yaml.safe_dump(dataclasses.asdict(o))
    back = OptimSpec(**yaml.safe_load(blob))
    assert back == o


def test_optim_groups_change_run_id_and_group_id():
    """Two runs whose only difference is a per-group LR are different science
    and must not collide on one identity."""
    from experimentation.run.spec import group_id, run_id

    plain = _spec()
    grouped = _spec(optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}]))
    assert run_id(plain) != run_id(grouped)
    assert group_id(plain) != group_id(grouped)


def test_different_lr_mult_values_are_different_runs():
    from experimentation.run.spec import run_id

    a = _spec(optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}]))
    b = _spec(optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 2.0}]))
    assert run_id(a) != run_id(b)


# --------------------------------------------------------------------------
# Identity continuity: adding these two optional sections must not renumber
# runs that do not use them.
# --------------------------------------------------------------------------

def test_absent_sections_do_not_perturb_existing_run_ids():
    """`schedules` and `optim.groups` are partial maps where absence is the norm,
    so an absent one must contribute NOTHING to the hash -- otherwise every run
    recorded before this feature re-identifies under a new run_id and stops
    aggregating with its own earlier attempts.

    Asserted as an INVARIANT rather than against a captured constant. The
    original form pinned 6a561ca9/40ff2dc8, taken from the commit before §6/§7
    landed (2026-08-08), and that broke for a reason having nothing to do with
    §6/§7: `per_device_batch_size` moved from OptimSpec to RuntimeSpec on
    2026-08-19, which legitimately changed the hashed payload. A constant that
    any unrelated schema change invalidates cannot distinguish "this feature
    renumbered every run" from "the schema moved", which is the one thing it
    exists to tell you.

    Verified when §6/§7 was re-integrated: the same helper hashes to 27ede7eb
    both with these fields present-but-empty and at a commit where the fields do
    not exist at all.
    """
    import dataclasses

    from koopman_lm.config import identity_payload
    from experimentation.run.spec import (_json_stable, _scientific_payload,
                                          group_id, run_id)

    spec = _spec()
    payload = _scientific_payload(spec, include_seed=True)

    # What the payload would have been before either field existed: the same
    # dict, with any trace of them removed. Byte-identical means an old run's
    # identity is untouched.
    #
    # `identity_payload` for the model section, not `dataclasses.asdict`: the
    # rule for what a model contributes to identity now has one definition
    # (koopman_lm/config.py::IDENTITY_TRANSPARENT_DEFAULTS), and re-deriving it
    # here with asdict would make this test assert that the rule does not exist.
    # The invariant under test is unaffected -- it is about `schedules` and
    # `optim.groups` -- and the pinned run_id below is the real check that the
    # model rule did not renumber anything.
    legacy = {"model": identity_payload(spec.model),
              "data": dataclasses.asdict(spec.data),
              "optim": {k: v for k, v in dataclasses.asdict(spec.optim).items()
                        if k != "groups"},
              "seed": spec.runtime.seed}
    assert _json_stable(payload) == _json_stable(legacy), (
        "an absent schedules/groups changed the hashed payload, so every run "
        "recorded before §6/§7 has been renumbered")

    # And the ids are stable within a schema, so an accidental payload change is
    # still caught here rather than in a run directory. Re-capture these ONLY
    # alongside a deliberate schema change, and say which in the message.
    assert run_id(spec) == "27ede7eb"
    assert group_id(spec) == "10c5697d"


def test_empty_sections_are_omitted_from_the_hashed_payload():
    from experimentation.run.spec import _scientific_payload

    payload = _scientific_payload(_spec(), include_seed=True)
    assert "schedules" not in payload
    assert "groups" not in payload["optim"]


def test_present_sections_appear_in_the_hashed_payload():
    from experimentation.run.spec import _scientific_payload

    payload = _scientific_payload(
        _spec(optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}]),
              schedules={"model.ska_ridge": {"kind": "constant", "value": 1e-3}}),
        include_seed=True)
    assert payload["schedules"] == {
        "model.ska_ridge": {"kind": "constant", "value": 1e-3}}
    assert payload["optim"]["groups"][0]["match"] == "*.ska.*"


# --------------------------------------------------------------------------
# §6 RunSpec.schedules
# --------------------------------------------------------------------------

def test_run_spec_defaults_to_no_schedules():
    assert _spec().schedules == {}


def test_schedules_are_a_top_level_sibling_not_a_model_field():
    """§6.1: KoopmanLMConfig is a *total* description validated field-for-field
    against the dataclass; schedules is a partial map. They cannot share a
    validation regime, so schedules must not leak into model."""
    from koopman_lm.config import KoopmanLMConfig

    fields = {f.name for f in dataclasses.fields(KoopmanLMConfig)}
    assert "schedules" not in fields
    s = _spec(schedules={"model.ska_ridge": {"kind": "constant", "value": 1e-3}})
    assert "schedules" not in dataclasses.asdict(s.model)


def test_schedules_are_hashed_into_run_id():
    """§6.4: with a schedule, model.ska_ridge in spec.yaml is only the initial
    value -- the schedule is the real story."""
    from experimentation.run.spec import group_id, run_id

    plain = _spec()
    annealed = _spec(schedules={
        "model.ska_ridge": {"kind": "linear", "from": 1e-2, "to": 1e-3,
                            "over": [0, 5000]}})
    assert run_id(plain) != run_id(annealed)
    assert group_id(plain) != group_id(annealed)


def test_two_different_annealings_are_different_runs():
    from experimentation.run.spec import run_id

    a = _spec(schedules={"model.ska_ridge": {"kind": "linear", "from": 1e-2,
                                             "to": 1e-3, "over": [0, 5000]}})
    b = _spec(schedules={"model.ska_ridge": {"kind": "linear", "from": 1e-2,
                                             "to": 1e-4, "over": [0, 5000]}})
    assert run_id(a) != run_id(b)


def test_run_spec_rejects_an_unknown_schedule_target_at_construction():
    """A whitelist, not reflection (§6.4): scheduling something the forward
    never re-reads must be a startup error, not a beautiful no-op curve."""
    with pytest.raises(ValueError, match="model.nonexistent"):
        _spec(schedules={"model.nonexistent": {"kind": "constant", "value": 1.0}})


def test_run_spec_rejects_a_malformed_schedule_at_construction():
    with pytest.raises((ValueError, TypeError, KeyError)):
        _spec(schedules={"model.ska_ridge": {"kind": "linear", "from": 1e-2}})


def test_run_spec_rejects_scheduling_the_learning_rate():
    """§6.6: LambdaLR owns param_group['lr']; two mechanisms fighting over it
    is a bug generator."""
    with pytest.raises(ValueError):
        _spec(schedules={"optim.lr": {"kind": "linear", "from": 1e-3,
                                      "to": 1e-4, "over": [0, 100]}})


# --------------------------------------------------------------------------
# Materialization round trip
# --------------------------------------------------------------------------

def test_materialized_spec_carries_groups_and_schedules(tmp_path):
    from experimentation.run.resolve import load_materialized_spec, materialize

    spec = _spec(
        optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}]),
        schedules={"model.ska_ridge": {"kind": "linear", "from": 1e-2,
                                       "to": 1e-3, "over": [0, 5000]}},
    )
    path = materialize(spec, tmp_path)
    back = load_materialized_spec(path)
    assert back.optim.groups == spec.optim.groups
    assert back.schedules == spec.schedules


def test_materialized_run_id_is_stable_across_the_round_trip(tmp_path):
    from experimentation.run.resolve import load_materialized_spec, materialize
    from experimentation.run.spec import run_id

    spec = _spec(
        optim=_optim(groups=[{"match": "*.ska.*", "lr_mult": 10.0}]),
        schedules={"model.ska_ridge": {"kind": "cosine", "from": 1e-2,
                                       "to": 1e-3, "over": [0, 5000]}},
    )
    path = materialize(spec, tmp_path)
    assert run_id(load_materialized_spec(path)) == run_id(spec)


def test_a_spec_without_schedules_omits_nothing_on_reload(tmp_path):
    """Absence must survive the round trip as absence, not as a stray empty
    key that perturbs identity."""
    from experimentation.run.resolve import load_materialized_spec, materialize
    from experimentation.run.spec import run_id

    spec = _spec()
    path = materialize(spec, tmp_path)
    back = load_materialized_spec(path)
    assert back.schedules == {}
    assert back.optim.groups == ()
    assert run_id(back) == run_id(spec)


def test_resolve_run_spec_reads_groups_and_schedules_from_yaml(tmp_path):
    import yaml
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.training.optim import ParamGroupSpec

    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump({
        "name": "t",
        "model": "1m",
        "data": {"kind": "shard", "shard_dir": "/d", "tokenizer": "tok",
                 "mix": {"fineweb": 1.0}, "n_tokens": 10},
        "optim": {"lr": 4.0e-4, "warmup_steps": 10, "max_steps": 100,
                  "groups": [{"match": "*.ska.*", "lr_mult": 10.0}]},
        "schedules": {"model.ska_ridge": {"kind": "linear", "from": 1.0e-2,
                                          "to": 1.0e-3, "over": [0, 50]}},
    }))
    spec = resolve_run_spec(p)
    assert spec.optim.groups == (ParamGroupSpec(match="*.ska.*", lr_mult=10.0),)
    assert spec.schedules["model.ska_ridge"]["kind"] == "linear"
