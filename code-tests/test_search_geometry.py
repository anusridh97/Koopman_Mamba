"""experimentation/sweep/search/geometry.py -- where SKA layers go, and how many.

Pure arithmetic over layer indices. No optuna, no torch, no config: this is the
bottom of the search package and the most heavily reused part of it, so it is
also the part worth pinning hardest.

Two properties matter more than any individual number.

**The space must contain the baseline.** A search whose space cannot express the
config you already run cannot tell you whether you improved on it. So
`layer_count_choices` includes the base count, and `placement="baseline"`
reproduces the base indices exactly rather than approximately.

**Layer 0 and the final layer stay free.** An SKA adapter at depth 0 reads
embeddings that no local mixer has touched yet, and one at the final depth
leaves no Mamba-only cleanup layer after it. Reserving both is a modelling
decision, and it is the reason `n_layers - 2` bounds the capacity.
"""
import pytest

pytestmark = pytest.mark.correctness

# configs/50m.yaml: 17 layers, SKA at [3, 7, 11, 15].
N_LAYERS_50M = 17
BASE_INDICES_50M = (3, 7, 11, 15)


# ---------------------------------------------------------------- counts ----

def test_layer_count_choices_contains_the_base_count():
    from experimentation.sweep.search.geometry import layer_count_choices

    choices = layer_count_choices(4, N_LAYERS_50M)
    assert 4 in choices, "the baseline depth must be reachable by the search"
    assert choices == sorted(set(choices)), "sorted and deduplicated"


def test_layer_count_choices_spans_halving_to_doubling():
    from experimentation.sweep.search.geometry import layer_count_choices

    assert layer_count_choices(4, N_LAYERS_50M) == [2, 3, 4, 6, 8]


def test_layer_count_choices_never_exceeds_the_usable_capacity():
    """With layer 0 and the last layer reserved, a 6-layer backbone has 4 usable
    slots, so doubling a base of 4 must clamp rather than ask for 8."""
    from experimentation.sweep.search.geometry import layer_count_choices

    for base in (1, 2, 3, 4, 8):
        for n_layers in (4, 6, 8, 17, 24):
            choices = layer_count_choices(base, n_layers)
            assert choices, "must offer at least one count"
            assert all(1 <= c <= max(1, n_layers - 2) for c in choices), (
                f"base={base} n_layers={n_layers} -> {choices}")


# ---------------------------------------------------------------- indices ----

def test_baseline_placement_reproduces_the_base_indices_exactly():
    from experimentation.sweep.search.geometry import make_layer_indices

    indices = make_layer_indices(N_LAYERS_50M, 4, "baseline", BASE_INDICES_50M)
    assert indices == [3, 7, 11, 15], (
        "at the baseline count, baseline placement must reproduce configs/50m.yaml "
        "exactly -- an approximation would make the reference trial unusable as a "
        "reference")


@pytest.mark.parametrize("placement", ["baseline", "even", "midlate", "late"])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 6, 8])
def test_indices_are_sorted_unique_and_inside_the_free_window(placement, count):
    from experimentation.sweep.search.geometry import make_layer_indices

    indices = make_layer_indices(N_LAYERS_50M, count, placement, BASE_INDICES_50M)
    assert len(indices) == count
    assert len(set(indices)) == count, "no duplicate depths"
    assert indices == sorted(indices)
    assert min(indices) >= 1, "layer 0 stays free (embedding-adjacent local mixer)"
    assert max(indices) <= N_LAYERS_50M - 2, "the final layer stays Mamba-only"


def test_count_is_clamped_to_the_available_window():
    from experimentation.sweep.search.geometry import make_layer_indices

    indices = make_layer_indices(N_LAYERS_50M, 99, "even", BASE_INDICES_50M)
    assert len(indices) == N_LAYERS_50M - 2 == 15


def test_late_placement_sits_deeper_than_even_placement():
    """The point of having both: `late` biases toward depth, so its mean index
    must exceed `even`'s. Without this the four placements are decoration."""
    from experimentation.sweep.search.geometry import make_layer_indices

    even = make_layer_indices(N_LAYERS_50M, 4, "even", BASE_INDICES_50M)
    late = make_layer_indices(N_LAYERS_50M, 4, "late", BASE_INDICES_50M)
    assert sum(late) / len(late) > sum(even) / len(even)


def test_midlate_placement_starts_no_earlier_than_thirty_percent_depth():
    from experimentation.sweep.search.geometry import make_layer_indices

    midlate = make_layer_indices(N_LAYERS_50M, 4, "midlate", BASE_INDICES_50M)
    assert min(midlate) >= round(0.30 * (N_LAYERS_50M - 1))


def test_even_placement_spans_the_whole_free_window():
    from experimentation.sweep.search.geometry import make_layer_indices

    even = make_layer_indices(N_LAYERS_50M, 4, "even", BASE_INDICES_50M)
    assert even[0] == 1 and even[-1] == N_LAYERS_50M - 2


def test_unknown_placement_is_rejected():
    from experimentation.sweep.search.geometry import make_layer_indices

    with pytest.raises(ValueError, match="placement"):
        make_layer_indices(N_LAYERS_50M, 4, "sideways", BASE_INDICES_50M)


def test_a_tiny_backbone_still_yields_a_usable_window():
    """n_layers=2 leaves no interior at all; the fallback must return something
    valid rather than an empty list or an IndexError."""
    from experimentation.sweep.search.geometry import make_layer_indices

    indices = make_layer_indices(2, 1, "even", (0,))
    assert len(indices) == 1
    assert 0 <= indices[0] < 2


# ------------------------------------------------------------- snapping ----

def test_nearest_snaps_to_the_closest_choice():
    from experimentation.sweep.search.geometry import nearest

    assert nearest(23, [8, 16, 24, 32]) == 24
    assert nearest(0.9, [0.75, 1.0, 1.25]) == 1.0


def test_nearest_breaks_ties_toward_the_smaller_choice():
    """Deterministic tie-breaking matters: a sampler that proposes an exact
    midpoint must always resolve it the same way, or two identical trials get
    different configs and different run_ids."""
    from experimentation.sweep.search.geometry import nearest

    assert nearest(20, [16, 24]) == 16
    assert nearest(1.0, [0.5, 1.5]) == 0.5


def test_clamp_bounds_both_ends():
    from experimentation.sweep.search.geometry import clamp

    assert clamp(5.0, 1.0, 3.0) == 3.0
    assert clamp(0.5, 1.0, 3.0) == 1.0
    assert clamp(2.0, 1.0, 3.0) == 2.0


def test_geometry_imports_without_optuna():
    """geometry/space/anchors must stay importable with no optuna installed --
    that is what lets the anchors sweep ship before the searcher does."""
    import experimentation.sweep.search.geometry as g

    assert "optuna" not in getattr(g, "__dict__", {})
