"""Config correctness: frozen immutability, YAML loading, hashing, param counts.

Pure-Python / CPU — no torch model instantiation, so this runs anywhere
(including CI without a GPU). Marked `correctness` so it runs in the gate suite.
"""
import dataclasses

import pytest
import yaml

from koopman_lm.config import (
    KoopmanLMConfig,
    config_hash,
    build_config,
    CONFIG_REGISTRY,
    CONFIG_FACTORIES,
)

pytestmark = pytest.mark.correctness

# nominal param-count band per scale (millions): (low, high)
PARAM_BANDS = {
    "50m": (40, 65),
    "180m": (160, 210),
    "370m": (330, 410),
    "440m": (410, 480),
    "880m": (800, 960),
    "1p5b": (1400, 1700),
    "3b": (2800, 3200),
}


def test_config_is_frozen():
    cfg = build_config("180m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 1234
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.vocab_size = 50000


def test_all_registry_entries_build_frozen_configs():
    for name in CONFIG_REGISTRY:
        cfg = build_config(name)
        assert isinstance(cfg, KoopmanLMConfig), name
        assert isinstance(cfg.ska_layer_indices, tuple), name
        assert cfg.ska_gamma_clamp is None or isinstance(cfg.ska_gamma_clamp, tuple), name


def test_required_phase0_scales_exist():
    for size in ["50m", "180m", "440m", "880m", "1p5b", "3b"]:
        assert size in CONFIG_REGISTRY
        assert isinstance(build_config(size), KoopmanLMConfig)


def test_build_config_rejects_unknown():
    with pytest.raises(ValueError):
        build_config("999b")


def test_replace_is_the_mutation_path():
    cfg = build_config("180m")
    cfg2 = dataclasses.replace(cfg, vocab_size=50000, max_seq_len=4096)
    assert cfg.vocab_size == 32000
    assert cfg2.vocab_size == 50000 and cfg2.max_seq_len == 4096
    assert isinstance(cfg2, KoopmanLMConfig)


def test_config_hash_is_deterministic_and_stable():
    h1 = config_hash(build_config("440m"))
    h2 = config_hash(build_config("440m"))
    assert h1 == h2
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)


def test_config_hash_distinguishes_configs():
    hashes = {name: config_hash(build_config(name)) for name in CONFIG_REGISTRY}
    assert len(set(hashes.values())) == len(hashes), hashes


def test_config_hash_changes_on_replace():
    base = build_config("180m")
    h_base = config_hash(base)
    h_mut = config_hash(dataclasses.replace(base, vocab_size=50000))
    assert h_base != h_mut


def test_gated_variant_differs_only_in_mlp_gated():
    # 180m_gated is a variant of the 768x24 backbone (180m_dense), not of
    # "180m" -- the current 180m.yaml is an unrelated 640x25 prefix-scan
    # production config that happened to reuse the "180m" name. See
    # configs/180m_dense.yaml for the history.
    base = build_config("180m_dense")
    gated = build_config("180m_gated")
    assert base.mlp_gated is False
    assert gated.mlp_gated is True
    assert config_hash(dataclasses.replace(base, mlp_gated=True)) == config_hash(gated)


@pytest.mark.parametrize("name", sorted(CONFIG_REGISTRY))
def test_config_survives_spec_yaml_round_trip(name):
    """asdict -> spec.yaml -> KoopmanLMConfig(**raw) must yield an EQUAL config.

    This is the invariant experimentation/run does on every launch: resolve.py
    materializes a spec.yaml via dataclasses.asdict (run/spec.py:181) and
    rebuilds the model config from it (run/resolve.py:219). YAML has no tuple
    type, so every collection field comes back as a list; without __post_init__'s
    coercion the rebuilt config compares UNEQUAL and is unhashable, and the
    frozen guarantee is a fiction because a list field can be mutated in place.

    Asserts the property, not a field list, so a newly added collection field
    that forgets metadata={"coerce": ...} fails here instead of silently.
    """
    cfg = build_config(name)
    rebuilt = KoopmanLMConfig(**yaml.safe_load(yaml.safe_dump(dataclasses.asdict(cfg))))
    assert rebuilt == cfg
    assert hash(rebuilt) == hash(cfg)
    assert config_hash(rebuilt) == config_hash(cfg)


def test_every_collection_field_declares_coercion():
    """Any tuple/list-typed field must opt into coercion at its declaration."""
    missing = [
        f.name for f in dataclasses.fields(KoopmanLMConfig)
        if ("uple" in str(f.type) or "ist" in str(f.type))
        and "coerce" not in f.metadata
    ]
    assert not missing, f"collection fields without metadata={{'coerce': ...}}: {missing}"


def test_lists_are_coerced_to_tuples():
    cfg = KoopmanLMConfig(
        n_layers=24,
        ska_layer_indices=[4, 8, 12],
        ska_gamma_clamp=[1.0, 1.5],
        ska_eta_bounds=[1.4, 1.7],
        ska_gamma_bounds=[0.5, 1.5],
    )
    for name in ("ska_layer_indices", "ska_gamma_clamp",
                 "ska_eta_bounds", "ska_gamma_bounds"):
        assert isinstance(getattr(cfg, name), tuple), name


@pytest.mark.parametrize("size,band", PARAM_BANDS.items())
def test_param_counts_in_band(size, band):
    lo, hi = band
    pm = build_config(size).param_count_estimate() / 1e6
    assert lo <= pm <= hi, f"{size}: {pm:.1f}M not in [{lo}, {hi}]M"


def test_50m_param_count_matches_real_gpu_instantiation():
    """param_count_estimate() must match reality, not just fall in a band.

    50,034,044 is what a real GPU instantiation reported (build 415208), not
    a derived constant -- the estimator's Mamba-2 term previously mismatched
    mamba_ssm.Mamba2's actual parameter layout (missing ngroups*d_state/nheads
    in in_proj's width and the internal RMSNormGated entirely), a 1.4%
    (731,172-parameter) overcount. This pins the fix so a future edit to
    param_count_estimate can't silently regress it. mamba_ssm isn't installed
    here (CPU venv) -- the 50,034,044 figure is the real, measured total from
    that GPU run, cross-checked term-by-term against mamba_ssm.Mamba2's
    public source (see koopman_lm/config.py's param_count_estimate comments).
    """
    assert build_config("50m").param_count_estimate() == 50_034_044


def test_param_counts_monotonic():
    order = ["50m", "180m", "370m", "440m", "880m", "1p5b", "3b"]
    counts = [build_config(s).param_count_estimate() for s in order]
    assert counts == sorted(counts), dict(zip(order, counts))


def test_head_dim_consistent():
    for name in CONFIG_REGISTRY:
        cfg = build_config(name)
        assert cfg.d_model % cfg.ska_n_heads == 0, name
        assert cfg.head_dim == cfg.d_model // cfg.ska_n_heads, name


def test_config_factories_compat():
    # CONFIG_FACTORIES is kept for backward compat; verify it still works
    for name, factory in CONFIG_FACTORIES.items():
        cfg = factory()
        assert isinstance(cfg, KoopmanLMConfig), name
