"""Unit tests for the corpus source registry + mix resolution (mix.py).

Pure Python (no numpy/torch/datasets), so these run anywhere. They pin the
arithmetic the continued-pretraining corpus depends on: fractions renormalize
to 1, the code+math bucket lands at the requested share, overrides never clobber
registry defaults, and malformed --sources entries raise.
"""
import pytest

from experimentation.training.data.mix import (
    SOURCE_SPECS, parse_sources, normalize_mix, resolve_specs, interleave_quota,
)

pytestmark = pytest.mark.correctness

CPT_SOURCES = "fineweb=0.40 code=0.125 math=0.125 cosmopedia=0.20 scrolls=0.15"


def test_parse_and_normalize_cpt_mix():
    m = parse_sources(CPT_SOURCES.split())
    assert set(m) == {"fineweb", "code", "math", "cosmopedia", "scrolls"}
    n = normalize_mix(m)
    assert abs(sum(n.values()) - 1.0) < 1e-9
    # code + math == the 25% code/math bucket
    assert abs((n["code"] + n["math"]) - 0.25) < 1e-9
    assert abs(n["fineweb"] - 0.40) < 1e-9


def test_quota_sums_and_shares():
    n = normalize_mix(parse_sources(CPT_SOURCES.split()))
    q = interleave_quota(n, 50_000_000)
    assert sum(q.values()) == 50_000_000
    assert abs((q["code"] + q["math"]) / sum(q.values()) - 0.25) < 1e-6


def test_normalize_is_scale_invariant():
    assert normalize_mix({"fineweb": 2, "code": 1, "math": 1}) == \
        normalize_mix({"fineweb": 1.0, "code": 0.5, "math": 0.5})


def test_zero_weight_source_gets_zero_quota():
    # a pure single-source mix must not pull a floored token from zero sources
    q = interleave_quota({"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}, 1000)
    assert q == {"fineweb": 1000, "pg19": 0, "scrolls": 0}


def test_overrides_do_not_clobber_defaults():
    n = normalize_mix(parse_sources("fineweb=0.5 code=0.5".split()))
    specs = resolve_specs(n, overrides={"code": {"path": None, "data_dir": "python"}})
    assert specs["code"]["path"] == SOURCE_SPECS["code"]["path"]  # None kept default
    assert specs["code"]["data_dir"] == "python"                  # real override applied


def test_resolve_specs_deep_copies():
    n = normalize_mix(parse_sources("scrolls=1.0".split()))
    specs = resolve_specs(n, overrides={"scrolls": {"subsets": ["qasper"]}})
    assert specs["scrolls"]["subsets"] == ["qasper"]
    # registry default is untouched by the override
    assert SOURCE_SPECS["scrolls"]["subsets"] != ["qasper"]


@pytest.mark.parametrize("bad", [["foo=0.5"], ["fineweb"], ["fineweb=-0.1"], []])
def test_malformed_sources_raise(bad):
    with pytest.raises(ValueError):
        parse_sources(bad)


def test_empty_mix_normalize_raises():
    with pytest.raises(ValueError):
        normalize_mix({"fineweb": 0.0})


def test_retrieval_sources_registered():
    for name in ("wikipedia", "hotpotqa", "musique", "nq"):
        assert name in SOURCE_SPECS
    # the QA-evidence sources are qa_context; wikipedia is plain LM text
    assert SOURCE_SPECS["hotpotqa"]["kind"] == "qa_context"
    assert SOURCE_SPECS["musique"]["kind"] == "qa_context"
    assert SOURCE_SPECS["wikipedia"]["kind"] == "plain"


def test_phase1_retrieval_bucket_is_15pct():
    # the continued_pretrain.sh default retrieval bucket
    n = normalize_mix(parse_sources(
        "fineweb=0.40 code=0.125 math=0.125 cosmopedia=0.20 "
        "wikipedia=0.09 hotpotqa=0.03 musique=0.03".split()))
    assert abs((n["wikipedia"] + n["hotpotqa"] + n["musique"]) - 0.15) < 1e-9
