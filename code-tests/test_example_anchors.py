"""The shipped example anchor file must load, and must keep saying it is an example.

`configs/search/curated_15.yaml` -- the real curated set -- does not exist,
because choosing fifteen points a human believes in is a research judgement, and
generating them would imply a judgement nobody made. So a four-point EXAMPLE file
ships instead, purely to make the anchor code path reachable end to end:
`enqueue_anchors`, `is_anchor`, `designs_to_cells`, and the driver's exemption of
anchors from pruning.

The risk with an example file is that it stops reading as one. Six months on,
`configs/search/example_anchors.yaml` sitting next to real configs is exactly the
sort of thing that gets pointed at by a launch script and quietly becomes the
curated set by default. So the labelling is tested, not just written.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from koopman_lm.config import build_config  # noqa: E402
from experimentation.sweep.search.anchors import (  # noqa: E402
    load_designs, resolve_design)
from experimentation.sweep.search.space import search_space  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
PATH = REPO / "configs/search/example_anchors.yaml"


def _designs():
    return load_designs(PATH, minimum=4)


def test_the_file_exists_and_loads():
    designs = _designs()
    assert [d.name for d in designs] == [
        "baseline", "example-lean", "example-wide", "example-conditioned"]


def test_every_design_resolves_against_the_real_space():
    """Loading is not enough -- a design can parse and then fail to snap onto the
    space, which is where an unknown field or an out-of-range factor surfaces."""
    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    for design in _designs():
        params = resolve_design(design, cfg, space, base_lr=4e-4)
        assert params, f"{design.name} resolved to nothing"
        assert "ska_rank" in params and "n_ska_layers" in params


def test_the_baseline_design_reproduces_the_base_config():
    """A design that sets nothing IS the base config. That property is what makes
    every other anchor's score readable as a delta, and it is easy to break by
    giving Design a non-1.0 default."""
    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    baseline = next(d for d in _designs() if d.name == "baseline")
    params = resolve_design(baseline, cfg, space, base_lr=4e-4)
    assert params["ska_rank"] == cfg.ska_rank
    assert params["ska_ridge"] == pytest.approx(cfg.ska_ridge)
    assert params["ska_layerscale_init"] == pytest.approx(cfg.ska_layerscale_init)
    assert params["learning_rate"] == pytest.approx(4e-4)


def test_the_designs_actually_differ_from_each_other():
    """Four identical points would exercise the code path and teach the sampler
    nothing. Cheap to assert, and it catches a clamp that flattened them."""
    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    seen = set()
    for design in _designs():
        params = resolve_design(design, cfg, space, base_lr=4e-4)
        seen.add((params["ska_rank"], params["n_ska_layers"],
                  round(params["ska_ridge"], 6),
                  round(params["learning_rate"], 8)))
    assert len(seen) == 4, f"designs collapsed onto {len(seen)} distinct points"


def test_the_filename_still_says_example():
    assert PATH.name.startswith("example"), (
        "an example anchor set must be named as one -- the failure mode is that "
        "it gets pointed at by a launch script and becomes the curated set")


def test_the_header_still_disclaims_being_curated():
    """Belt and braces with the filename. If someone renames or repurposes this
    file, at least one of the two guards fires."""
    text = PATH.read_text()
    head = text[:2000].lower()
    assert "not a curated" in head, "the disclaimer at the top is gone"
    assert "curated_15.yaml" in head, (
        "the header should still point at the real file it stands in for")
    assert "mechanically" in head, (
        "the header should still say these points were not chosen on merit")


def test_the_real_curated_file_is_still_absent():
    """A control. When curated_15.yaml lands, this test fails -- which is the
    prompt to point the default at it and reconsider whether the example is still
    worth shipping."""
    real = REPO / "configs/search/curated_15.yaml"
    assert not real.exists(), (
        "curated_15.yaml now exists: switch --design-file's default to it and "
        "decide whether example_anchors.yaml still earns its place")
