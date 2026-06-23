"""Config correctness: frozen immutability, YAML loading, hashing, param counts.

Pure-Python / CPU — no torch model instantiation, so this runs anywhere
(including CI without a GPU). Marked `correctness` so it runs in the gate suite.
"""
import dataclasses

import pytest

from koopman_lm.globals.config import (
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
    base = build_config("180m")
    gated = build_config("180m_gated")
    assert base.mlp_gated is False
    assert gated.mlp_gated is True
    assert config_hash(dataclasses.replace(base, mlp_gated=True)) == config_hash(gated)


@pytest.mark.parametrize("size,band", PARAM_BANDS.items())
def test_param_counts_in_band(size, band):
    lo, hi = band
    pm = build_config(size).param_count_estimate() / 1e6
    assert lo <= pm <= hi, f"{size}: {pm:.1f}M not in [{lo}, {hi}]M"


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
