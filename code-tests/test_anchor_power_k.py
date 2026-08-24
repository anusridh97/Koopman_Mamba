"""`Design.power_K`: round trip, resolution, and the malformed inputs it refuses.

`Design` could vary rank, depth, placement, ridge, layerscale, LR, norm-clip and
gamma. K was the one searched axis it could only INHERIT, so a design set could
not contain a labelled K=2 cell and a study sampling {1, 2} had a reference at
only one of them.

The asymmetry this file exists to pin: **an inherited K snaps, an explicit K does
not.**

Every other factor here clamps or snaps into the declared space on purpose --
the space is what bounds a trial, and a design written against a base config
should get the closest available point rather than an error. K is different
because it is a matrix power over a set of two or three small integers, so
"closest available" moves a requested 3 to 2, and a design named `reference-k3`
that runs at K=2 is a trial whose params deny its name. That is the one failure
a set of NAMED designs exists to prevent, so an undeclared explicit K raises.
`placement` is refused the same way and for the same reason.

Written after the implementation and verified by mutation rather than by
red-green: each guard below was disabled, the covering test watched to fail, and
the guard restored. The task's instruction is "test-first on failure-prone seams,
mutation-check important guards"; this is the second half of that, and it is
weaker than red-green because these tests could in principle have been written to
agree with a bug. The mutation runs are what rule that out.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

from koopman_lm.config import KoopmanLMConfig                      # noqa: E402
from experimentation.sweep.search.anchors import (                 # noqa: E402
    Design, load_designs, resolve_design)
from experimentation.sweep.search.space import (                   # noqa: E402
    restrict_space, search_space)


def _base(power_k=1) -> KoopmanLMConfig:
    return KoopmanLMConfig(
        d_model=256, n_layers=17, vocab_size=32000, d_state=48, d_conv=4,
        mamba_expand=2, mamba_headdim=32, ska_n_heads=4,
        ska_rank=24, ska_ridge=0.01, ska_power_K=power_k,
        ska_layer_indices=[3, 7, 11, 15], ska_mode="parallel",
        ska_prefix_scan=False, ska_inverse_cholesky=True,
        ska_norm_clip=True, ska_norm_clip_c=4.0,
        ska_layerscale=True, ska_layerscale_init=0.01,
        max_seq_len=1024, mlp_type="swiglu", mlp_expand=2.667)


def _space(base, **kw):
    return restrict_space(search_space(base), base, **kw)


def _resolve(design, base=None, space=None):
    base = base if base is not None else _base()
    space = space if space is not None else _space(base)
    return resolve_design(design, base, space, base_lr=4.0e-4)


# ------------------------------------------------------------ the default ----

def test_the_default_is_inherit_and_not_a_number():
    """A concrete default would silently pin every existing design's K to it."""
    assert Design.power_K == "baseline"
    assert Design("x").power_K == "baseline"


def test_an_unset_power_k_inherits_the_base_configs_own_value():
    assert _resolve(Design("ref"), base=_base(power_k=1))["ska_power_K"] == 1
    assert _resolve(Design("ref"), base=_base(power_k=2))["ska_power_K"] == 2


def test_inheriting_is_exactly_what_happened_before_the_field_existed():
    """The regression this guards: `resolve_design` used to read
    base_model.ska_power_K unconditionally. Every committed design file must
    still resolve to the same K it did then."""
    for base_k in (1, 2):
        base = _base(power_k=base_k)
        resolved = _resolve(Design("legacy-shaped", rank=8, ridge_factor=3.0),
                            base=base)
        assert resolved["ska_power_K"] == base_k


def test_an_inherited_k_never_actually_snaps_away():
    """`space._with_value_int` folds the base config's own K into the choices
    (baseline containment), so the snap is a no-op by construction. If it were
    not, an unset design would silently run at a K its base does not use."""
    base = _base(power_k=2)
    assert 2 in _space(base)["ska_power_K"]["choices"]
    assert _resolve(Design("ref"), base=base)["ska_power_K"] == 2


# ----------------------------------------------------------- explicit values ----

@pytest.mark.parametrize("requested", [1, 2])
def test_an_explicit_declared_k_resolves_to_itself(requested):
    assert _resolve(Design("k", power_K=requested))["ska_power_K"] == requested


def test_an_explicit_k_overrides_the_base_rather_than_being_ignored():
    """K=1 explicitly, on a base whose own K is 2. If explicit did not win, a
    `reference-k1` anchor on a K=2 base would run at K=2."""
    resolved = _resolve(Design("k1", power_K=1), base=_base(power_k=2))
    assert resolved["ska_power_K"] == 1


def test_an_explicit_k_is_an_int_not_whatever_yaml_produced():
    """optuna records the sampled value verbatim, so 2.0 is a DIFFERENT category
    from 2 and an anchor recording the float could never match a sampled int."""
    resolved = _resolve(Design("k2", power_K=2))
    assert resolved["ska_power_K"] == 2
    assert isinstance(resolved["ska_power_K"], int)
    assert not isinstance(resolved["ska_power_K"], bool)


def test_an_explicit_k_inside_a_widened_axis_is_accepted():
    """The escape route the refusal points at: widen the axis in the study's
    search_axes, where a reader can see it."""
    base = _base()
    wide = _space(base, axes={"ska_power_K": {"kind": "categorical",
                                              "choices": [1, 2, 3]}})
    assert _resolve(Design("k3", power_K=3), base=base,
                    space=wide)["ska_power_K"] == 3


# --------------------------------------------------------------- refusals ----

def test_an_undeclared_k_raises_rather_than_snapping():
    """THE asymmetry. Snapping 3 -> 2 would leave a design named for a K it does
    not run at."""
    with pytest.raises(ValueError) as exc:
        _resolve(Design("reference-k3", power_K=3))
    message = str(exc.value)
    assert "reference-k3" in message, "the message must name the design"
    assert "[1, 2]" in message, "the message must name what IS declared"


def test_the_refusal_says_how_to_get_what_you_asked_for():
    """An error that only forbids teaches nothing. This one names the axis to
    widen."""
    with pytest.raises(ValueError, match="search_axes"):
        _resolve(Design("k4", power_K=4))


def test_zero_is_refused_even_though_it_is_a_plausible_integer():
    """K=0 is the identity operator -- no SKA at all -- and is not a declared
    choice anywhere. Worth its own case because 0 is the value a template or an
    unset YAML field is most likely to produce."""
    with pytest.raises(ValueError, match="declares"):
        _resolve(Design("k0", power_K=0))


def test_a_negative_k_is_refused():
    with pytest.raises(ValueError, match="declares"):
        _resolve(Design("k-neg", power_K=-1))


@pytest.mark.parametrize("bad", [1.5, 2.5, 0.5])
def test_a_non_integral_k_is_refused_rather_than_truncated(bad):
    """`int(1.5)` is 1, silently. A journal recording 1 for a design that asked
    for 1.5 is worse than a crash."""
    with pytest.raises(ValueError, match="whole number"):
        _resolve(Design("k-frac", power_K=bad))


@pytest.mark.parametrize("bad", ["two", "", "1.0.0", None, [2], {"K": 2}])
def test_a_non_numeric_k_is_refused_with_a_usable_message(bad):
    """`"baseline"` is the one string that means something. Any other is a typo,
    and the message has to say what to write instead."""
    with pytest.raises(ValueError) as exc:
        _resolve(Design("k-junk", power_K=bad))
    assert "baseline" in str(exc.value)


def test_the_sentinel_is_matched_exactly_and_not_by_prefix():
    """`"base"` is a plausible abbreviation and must not silently mean
    `"baseline"` -- it would look like it inherited and would raise instead,
    which is the correct outcome and worth pinning."""
    with pytest.raises(ValueError):
        _resolve(Design("k-abbrev", power_K="base"))


# ------------------------------------------------------ file-level round trip ----

def test_power_k_round_trips_through_a_design_file(tmp_path):
    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: inherit\n"
        "  - name: k1\n"
        "    power_K: 1\n"
        "  - name: k2\n"
        "    power_K: 2\n"
        "  - name: k2-and-ridge\n"
        "    power_K: 2\n"
        "    ridge_factor: 3.0\n")
    designs = load_designs(path, minimum=4)
    assert [d.power_K for d in designs] == ["baseline", 1, 2, 2]
    resolved = [_resolve(d) for d in designs]
    assert [r["ska_power_K"] for r in resolved] == [1, 1, 2, 2]
    # And the two-factor cell moved its second factor too.
    assert resolved[3]["ska_ridge"] == 0.03


def test_a_misspelled_power_k_key_is_refused_by_the_loader(tmp_path):
    """`load_designs` rejects unknown fields, because silently ignoring a typo
    would run the trial as the baseline while its name claimed otherwise. The
    field is `power_K`, and `power_k` is the spelling a reader will try."""
    path = tmp_path / "designs.yaml"
    path.write_text("designs:\n  - name: k2\n    power_k: 2\n")
    with pytest.raises(ValueError) as exc:
        load_designs(path, minimum=1)
    assert "power_k" in str(exc.value)
    assert "power_K" in str(exc.value), (
        "the error lists the known fields, so the correct spelling is visible")


def test_the_committed_design_file_uses_the_field(tmp_path):
    """Guards the guard: if the real anchor file stopped using `power_K`, every
    assertion above would still pass while the feature went unused in the one
    place it was added for."""
    designs = load_designs(
        REPO / "configs/search/proxy-256x17-anchors.yaml", minimum=24)
    explicit = {d.name: d.power_K for d in designs if d.power_K != "baseline"}
    assert explicit == {"reference-k2": 2, "k2-ridge-low": 2,
                        "k2-ridge-high": 2}


def test_the_existing_example_anchor_file_is_still_valid():
    """No committed design file may change meaning. example_anchors.yaml sets no
    power_K, so every one of its designs must still inherit."""
    designs = load_designs(REPO / "configs/search/example_anchors.yaml",
                          minimum=4)
    assert all(d.power_K == "baseline" for d in designs)
    for base_k in (1, 2):
        base = _base(power_k=base_k)
        for design in designs:
            assert _resolve(design, base=base)["ska_power_K"] == base_k


def test_power_k_survives_the_designs_to_cells_path():
    """The other consumer: a design set materialized as a static `cells:` sweep,
    which is how anchors ship with no optuna anywhere in the path."""
    from experimentation.sweep.search.anchors import designs_to_cells

    base = _base()
    cells = designs_to_cells(
        [Design("k1", power_K=1), Design("k2", power_K=2)],
        base, _space(base), base_lr=4.0e-4, max_steps=600)
    assert [c["model.ska_power_K"] for c in cells] == [1, 2]
