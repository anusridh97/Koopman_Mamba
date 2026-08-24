"""configs/runs/proxy-256x17.yaml: the interaction study's base, pinned.

A base run spec is the one input a study cannot vary, so every property the
study's DESIGN depends on has to be asserted somewhere the CPU suite can see it.
Four of them are load-bearing enough to name:

**17 layers.** The whole reason this config exists rather than reusing
`4m-golden.yaml`. Two of the study's nine axes act on layer indices, and at four
layers both are dead (`test_search_axis_restriction.py` proves the dead half).

**Value width exactly 64.** `d_model / ska_n_heads`. It is a property of the base
spec that no sampled rank can repair, which is why `space._backend_overrides`
carries an error message about it: `fused_only` used to check rank alone, pass,
and then die on the first forward on a GPU after materializing a run directory.
Any later confirmation run that wants the fused SM100 kernel needs this number,
and a base that got it wrong would foreclose that for every follow-up.

**Baseline SKA indices [3, 7, 11, 15].** `placement="baseline"` reproduces the
base indices EXACTLY when the count matches, which is what makes the study's
`reference-k1` anchor a real reference rather than an approximation of one.

**~25.35M parameters.** Not a round number anybody chose -- it is what the
geometry costs, and it is asserted because the study records a per-trial
parameter count and a Pareto front against it. If the estimator or the geometry
moves, the recorded baseline moves with it, and every archived comparison
against it becomes wrong rather than merely stale.

Also pinned: the config validates, and the spec survives a materialize/load
round trip. The round trip is what a resumed run depends on -- `spec.yaml` on
scratch is read back by `load_materialized_spec`, and a field that does not
survive it breaks resume for a run already in flight.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

SPEC_PATH = REPO / "configs/runs/proxy-256x17.yaml"

#: The estimate at this exact geometry. Asserted to the parameter, not to a
#: tolerance: this is arithmetic over the config, so any change at all is a
#: change to the geometry or to the estimator, and both are worth stopping for.
EXPECTED_PARAMS = 25_352_736


@pytest.fixture(scope="module")
def spec():
    from experimentation.run.resolve import resolve_run_spec
    return resolve_run_spec(SPEC_PATH)


# --------------------------------------------------------------- geometry ----

def test_it_is_seventeen_layers(spec):
    assert spec.model.n_layers == 17, (
        "the placement and layer-count axes act on depth; at 4 layers both are "
        "dead (see test_search_axis_restriction.py)")


def test_the_head_value_width_is_exactly_sixty_four(spec):
    """d_model / ska_n_heads. No sampled rank can change it."""
    model = spec.model
    assert model.d_model == 256 and model.ska_n_heads == 4
    assert model.head_dim == 64
    assert model.d_model % model.ska_n_heads == 0


def test_the_value_width_matches_the_fused_kernels_own_constant():
    """Kept in step by test rather than by hope: the fused SM100 kernel is
    compiled for exactly one value width and is silently unreachable at any
    other."""
    from experimentation.sweep.search.space import _FUSED_VALUE
    from experimentation.run.resolve import resolve_run_spec

    assert resolve_run_spec(SPEC_PATH).model.head_dim == _FUSED_VALUE


def test_the_baseline_ska_indices_are_the_fifty_million_placement(spec):
    assert tuple(spec.model.ska_layer_indices) == (3, 7, 11, 15)


def test_baseline_placement_reproduces_those_indices_exactly(spec):
    """An approximate reproduction of the reference is worse than none: it looks
    like the baseline and is not."""
    from experimentation.sweep.search.geometry import make_layer_indices

    resolved = make_layer_indices(spec.model.n_layers, 4, "baseline",
                                 list(spec.model.ska_layer_indices))
    assert resolved == [3, 7, 11, 15]


def test_the_estimated_parameter_count(spec):
    assert spec.model.param_count_estimate() == EXPECTED_PARAMS
    assert 25.0e6 < EXPECTED_PARAMS < 25.7e6


# ------------------------------------------------------------- the SKA route ----

def test_it_takes_the_exact_inverse_cholesky_route(spec):
    """`exact_invchol` is the negation of a disjunction of three booleans, so it
    has to be asserted as all three."""
    model = spec.model
    assert model.ska_prefix_scan is False
    assert model.ska_inverse_cholesky is True
    assert model.ska_exact_intrachunk is False


def test_the_route_matches_what_the_studys_backend_policy_would_set(spec):
    """The base spec and `backend_policy: exact_invchol` must not disagree: the
    policy's overrides are applied on top of this spec, and a spec that already
    said something different would make the override the only thing that matters
    -- which is fine until someone launches this base WITHOUT a study."""
    from experimentation.sweep.search.space import _backend_overrides

    overrides = _backend_overrides("exact_invchol", spec.model.ska_rank,
                                   head_dim=spec.model.head_dim)
    for key, value in overrides.items():
        field = key.split(".", 1)[1]
        assert getattr(spec.model, field) == value, (
            f"{field}: spec says {getattr(spec.model, field)!r}, "
            f"exact_invchol sets {value!r}")


def test_it_is_parallel_ska_mode(spec):
    assert spec.model.ska_mode == "parallel"


# ------------------------------------------------- clipping, scale, precision ----

def test_norm_clipping_is_on_at_the_production_equivalent_value(spec):
    assert spec.model.ska_norm_clip is True
    assert spec.model.ska_norm_clip_c == 4.0


def test_the_clip_value_is_a_declared_choice_of_the_studys_multiplier_axis(spec):
    """4/sqrt(24) in the study's vocabulary. If it were not a declared choice the
    reference point would be outside the space, and `reference-k1` could not
    reproduce this config."""
    multiplier = spec.model.ska_norm_clip_c / math.sqrt(spec.model.ska_rank)
    assert multiplier == pytest.approx(0.8164965809277261, rel=0, abs=0)
    # And it round-trips back to 4.0 through params_to_overrides' own rounding.
    assert round(math.sqrt(spec.model.ska_rank) * multiplier, 8) == 4.0


def test_layerscale_is_on_at_one_percent(spec):
    assert spec.model.ska_layerscale is True
    assert spec.model.ska_layerscale_init == 0.01


def test_compute_is_bf16_and_the_ska_core_is_fp32(spec):
    """The whitened core Choleskys a Gram matrix; bf16's 8 mantissa bits make
    that unreliable, and the ridge that compensates is what this study samples.
    A bf16 core would confound the ridge axis with numerical noise."""
    assert spec.model.compute_precision == "bf16"
    assert spec.model.ska_precision == "fp32"


def test_embeddings_are_tied(spec):
    assert spec.model.tie_embeddings is True


def test_the_sequence_length_is_one_thousand_and_twenty_four(spec):
    assert spec.model.max_seq_len == 1024


# ------------------------------------------------------------ data and optim ----

def test_the_data_matches_the_existing_shards_metadata(spec):
    """`verify_shard` checks these against the shard's own meta.json at launch,
    so a mismatch is a startup failure -- but only on a machine that has the
    shard. This is the CPU-side half."""
    data = spec.data
    assert data.kind == "shard"
    assert data.shard_dir == "/scratch/m000151-pm06/jkli/fineweb_small_train"
    assert data.tokenizer == "NousResearch/Llama-2-7b-hf"
    assert data.n_tokens == 100000536
    assert data.mix == {"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}


def test_it_declares_the_same_shard_metadata_as_the_other_specs_on_it():
    """Three committed specs read this shard. They must agree, or two of them
    fail `verify_shard` and it looks like a data problem."""
    from experimentation.run.resolve import resolve_run_spec

    others = [resolve_run_spec(REPO / "configs/runs/50m-first-real.yaml"),
              resolve_run_spec(REPO / "configs/runs/4m-golden.yaml")]
    mine = resolve_run_spec(SPEC_PATH)
    for other in others:
        assert other.data.shard_dir == mine.data.shard_dir
        assert other.data.tokenizer == mine.data.tokenizer
        assert other.data.n_tokens == mine.data.n_tokens


def test_the_optimizer_is_the_studys_fixed_recipe(spec):
    """These three are FIXED axes of the interaction study. The base has to agree
    with them or the study's reference point is not the base config."""
    assert spec.optim.lr == 4.0e-4
    assert spec.optim.weight_decay == 0.1
    assert spec.optim.grad_clip == 1.0
    assert spec.optim.effective_batch == 64


def test_gradient_accumulation_divides_evenly(spec):
    assert spec.optim.effective_batch % spec.runtime.per_device_batch_size == 0


# ----------------------------------------------------------------- runtime ----

def test_it_asks_for_one_gpu_per_trial(spec):
    """Eight trials run concurrently on one node, each pinned to one device. A
    trial asking for two would be asking for a device another trial owns."""
    assert spec.runtime.gpus == 1
    assert spec.runtime.nodes == 1
    assert spec.runtime.ddp is False


def test_it_is_deterministic(spec):
    assert spec.runtime.deterministic is True


def test_the_account_partition_and_qos_match_the_other_pm06_specs(spec):
    from experimentation.run.resolve import resolve_run_spec

    reference = resolve_run_spec(REPO / "configs/runs/50m-first-real.yaml")
    for field in ("partition", "account", "qos", "gpu_arch"):
        assert getattr(spec.runtime, field) == getattr(reference.runtime, field)


# ------------------------------------------------------- validation and I/O ----

def test_the_model_config_validates(spec):
    """`KoopmanLMConfig.__post_init__` ran at resolve time; re-running it from
    the dict form catches a field that only survives because resolve() dropped
    it."""
    from koopman_lm.config import KoopmanLMConfig
    import dataclasses

    rebuilt = KoopmanLMConfig(**dataclasses.asdict(spec.model))
    assert rebuilt == spec.model


def test_the_committed_yaml_declares_no_unknown_model_fields():
    """`KoopmanLMConfig(**value)` would raise, but only when something resolves
    the spec -- and nothing did until this file."""
    raw = yaml.safe_load(SPEC_PATH.read_text())
    from koopman_lm.config import KoopmanLMConfig

    known = set(KoopmanLMConfig.__dataclass_fields__)
    assert set(raw["model"]) <= known, sorted(set(raw["model"]) - known)


def test_it_survives_a_materialize_load_round_trip(tmp_path, spec):
    """`spec.yaml` on scratch is what a resumed run reads back. A field that does
    not survive this breaks resume for a run already in flight."""
    from experimentation.run.resolve import load_materialized_spec, materialize

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reloaded = load_materialized_spec(materialize(spec, run_dir))
    assert reloaded.model == spec.model
    assert reloaded.data == spec.data
    assert reloaded.optim == spec.optim
    assert reloaded.runtime == spec.runtime


def test_the_round_trip_preserves_run_identity(tmp_path, spec):
    """The stronger form: identity is a hash of model+data+optim+seed, so a
    round trip that changed any of them would renumber the run."""
    from experimentation.run.resolve import load_materialized_spec, materialize
    from experimentation.run.spec import group_id, run_id

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reloaded = load_materialized_spec(materialize(spec, run_dir))
    assert (run_id(reloaded), group_id(reloaded)) == (run_id(spec), group_id(spec))
