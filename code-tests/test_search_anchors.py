"""experimentation/sweep/search/anchors.py -- curated designs -> sampled points.

An anchor is a hand-reasoned point in the space, written relative to the base
config ("three-quarters the SKA depth, placed late, ridge x3") rather than in
absolute field values. Relative is the right vocabulary for a design set that
should stay meaningful when pointed at a different base model, and it is why a
design file is not just a `cells:` sweep spelled differently.

`resolve_design` turns one into the same params dict shape the sampler produces,
snapped onto the declared space, so an anchor and a sampled trial are
interchangeable downstream -- which is what lets anchors seed a study via
`enqueue_trial` and also ship as a plain static sweep before any sampler exists.

The load-bearing property: an all-defaults design must resolve to the baseline
exactly. Otherwise the reference anchor is a near-miss and every comparison
against it is against something nobody chose.
"""
import textwrap

import pytest

pytestmark = pytest.mark.correctness


def _base_model():
    from koopman_lm.config import build_config
    return build_config("50m")


def _space():
    from experimentation.sweep.search.space import search_space
    return search_space(_base_model(), base_name="50m")


def _write_designs(tmp_path, body, name="designs.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


# ------------------------------------------------------------- loading ----

def test_load_designs_reads_a_named_list(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, """
        designs:
          - name: baseline
          - name: deeper-late
            layer_factor: 1.5
            placement: late
    """)
    designs = load_designs(path)
    assert [d.name for d in designs] == ["baseline", "deeper-late"]
    assert designs[1].layer_factor == 1.5
    assert designs[1].placement == "late"


def test_load_designs_defaults_every_factor_to_the_baseline(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, """
        designs:
          - name: baseline
    """)
    design = load_designs(path)[0]
    assert design.layer_factor == 1.0
    assert design.ridge_factor == 1.0
    assert design.layerscale_factor == 1.0
    assert design.lr_factor == 1.0
    assert design.placement == "baseline"
    assert design.rank == "baseline"


def test_load_designs_rejects_a_missing_designs_key(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, "trials:\n  - name: x\n")
    with pytest.raises(ValueError, match="designs"):
        load_designs(path)


def test_load_designs_rejects_an_unnamed_design(tmp_path):
    """Names are how a trial is traced back to the reasoning that proposed it;
    an anonymous anchor is indistinguishable from a sampled point."""
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, "designs:\n  - layer_factor: 2.0\n")
    with pytest.raises(ValueError, match="name"):
        load_designs(path)


def test_load_designs_rejects_an_unknown_field(tmp_path):
    """A typo in a design file would otherwise be silently ignored, and the
    trial would run as the baseline while claiming to test something."""
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, """
        designs:
          - name: typo
            ridge_facter: 3.0
    """)
    with pytest.raises(ValueError, match="ridge_facter"):
        load_designs(path)


def test_load_designs_enforces_a_caller_supplied_minimum(tmp_path):
    """The minimum is the caller's policy, not the loader's. The original
    harness hardcoded `< 15`, which coupled a general validator to one study."""
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, "designs:\n  - name: only-one\n")
    assert len(load_designs(path)) == 1
    with pytest.raises(ValueError, match="at least 15"):
        load_designs(path, minimum=15)


def test_load_designs_rejects_duplicate_names(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = _write_designs(tmp_path, """
        designs:
          - name: same
          - name: same
    """)
    with pytest.raises(ValueError, match="duplicate"):
        load_designs(path)


# ----------------------------------------------------------- resolving ----

def test_an_all_defaults_design_resolves_to_the_baseline():
    """The property everything else rests on."""
    from experimentation.sweep.search.anchors import Design, resolve_design
    from experimentation.sweep.search.space import params_to_overrides

    cfg = _base_model()
    params = resolve_design(Design(name="baseline"), cfg, _space(), base_lr=4e-4)
    overrides = params_to_overrides(params, cfg, max_steps=15000)

    assert overrides["model.ska_rank"] == cfg.ska_rank
    assert overrides["model.ska_layer_indices"] == list(cfg.ska_layer_indices)
    assert overrides["model.ska_norm_clip_c"] == pytest.approx(cfg.ska_norm_clip_c)
    assert overrides["model.ska_ridge"] == pytest.approx(cfg.ska_ridge)
    assert overrides["model.ska_layerscale_init"] == pytest.approx(cfg.ska_layerscale_init)
    assert overrides["optim.lr"] == pytest.approx(4e-4)


def test_resolved_params_match_the_declared_space_keys():
    """An anchor and a sampled trial must be interchangeable downstream."""
    from experimentation.sweep.search.anchors import Design, resolve_design

    space = _space()
    params = resolve_design(Design(name="x"), _base_model(), space, base_lr=4e-4)
    assert set(params) == set(space)


def test_categorical_values_are_snapped_onto_the_declared_choices():
    """A design may ask for anything; the study's space is what exists. An
    unsnapped value would make optuna reject the enqueued trial."""
    from experimentation.sweep.search.anchors import Design, resolve_design

    space = _space()
    params = resolve_design(Design(name="odd", rank=20), _base_model(), space,
                            base_lr=4e-4)
    assert params["ska_rank"] in space["ska_rank"]["choices"]
    assert params["n_ska_layers"] in space["n_ska_layers"]["choices"]
    assert params["weight_decay"] in space["weight_decay"]["choices"]


def test_float_values_are_clamped_into_the_declared_bounds():
    from experimentation.sweep.search.anchors import Design, resolve_design

    space = _space()
    params = resolve_design(Design(name="huge", ridge_factor=1000.0,
                                   layerscale_factor=1000.0, lr_factor=1000.0),
                            _base_model(), space, base_lr=4e-4)
    assert params["ska_ridge"] <= space["ska_ridge"]["high"]
    assert params["ska_layerscale_init"] <= space["ska_layerscale_init"]["high"]
    assert params["learning_rate"] <= space["learning_rate"]["high"]


def test_layer_factor_scales_the_baseline_depth():
    from experimentation.sweep.search.anchors import Design, resolve_design

    space = _space()
    cfg = _base_model()
    shallow = resolve_design(Design(name="s", layer_factor=0.5), cfg, space, base_lr=4e-4)
    deep = resolve_design(Design(name="d", layer_factor=2.0), cfg, space, base_lr=4e-4)
    assert shallow["n_ska_layers"] < len(cfg.ska_layer_indices) < deep["n_ska_layers"]


def test_lr_factor_scales_the_base_learning_rate():
    from experimentation.sweep.search.anchors import Design, resolve_design

    params = resolve_design(Design(name="hot", lr_factor=1.25), _base_model(),
                            _space(), base_lr=4e-4)
    assert params["learning_rate"] == pytest.approx(4e-4 * 1.25)


def test_an_unknown_placement_is_rejected_at_resolve_time():
    from experimentation.sweep.search.anchors import Design, resolve_design

    with pytest.raises(ValueError, match="placement"):
        resolve_design(Design(name="bad", placement="sideways"), _base_model(),
                       _space(), base_lr=4e-4)


def test_anchors_imports_without_optuna():
    import experimentation.sweep.search.anchors as anchors

    assert "optuna" not in anchors.__dict__


# ------------------------------------------- the space and the anchor agree ----
#
# The bidirectional keys-match guard is already
# `test_resolved_params_match_the_declared_space_keys` above, and it is what
# caught ska_power_K missing here when the space grew that dimension on
# 2026-08-21. Worth knowing WHY it matters: study.enqueue_trial accepts a
# PARTIAL params dict and samples whatever is absent, so an unresolved dimension
# does not error -- it silently makes that axis random for every anchor, which
# is the one thing an anchor exists not to be.

def test_an_anchor_runs_at_the_base_configs_power_K():
    """An anchor is defined relative to the base config, so K comes from there
    rather than from a factor -- and `_with_value_int` guarantees the base's own
    value is a declared choice. 4m-golden pins 2, configs/50m.yaml pins 1, and
    both must round-trip."""
    import dataclasses

    from experimentation.sweep.search.anchors import Design, resolve_design

    for value in (1, 2):
        cfg = dataclasses.replace(_base_model(), ska_power_K=value)
        from experimentation.sweep.search.space import search_space
        space = search_space(cfg, base_name="probe")
        params = resolve_design(Design(name="baseline"), cfg, space, base_lr=4e-4)
        assert params["ska_power_K"] == value
        assert value in space["ska_power_K"]["choices"], "snapping would move it"

