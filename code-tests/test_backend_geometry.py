"""The fused kernel's geometry, checked where it is cheap instead of where it hurts.

Two defects, both found by a 21-minute study that looked like a hang.

**`backend='auto'` degraded in silence.** The fused CUDA kernel requires value
width 64. `configs/runs/4m-golden.yaml` has `d_model 128 / 4 heads = 32`, so no
trial on that base spec could ever reach it -- and `auto` fell through to the pure
Python reference scan with no diagnostic at all, at a measured **137x**
(4.49s vs 0.033s per fwd+bwd micro-step on an H100). `cuda` and `cuda_prefix` got
a precise RuntimeError; `auto`, the default, got nothing. The expensive
substitution was the silent one.

**`fused_only` checked rank and not value width.** Rank is SAMPLED, value width is
a property of the base spec (`d_model / ska_n_heads`), so no sampled rank can fix
it. A `fused_only` study on a 4m base therefore passed spec validation, materialized
a run directory, queued a job, and died at the first forward on a GPU.

Both are now caught before anything is spent.
"""

import pathlib
import sys
import warnings

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.run.resolve import resolve_run_spec  # noqa: E402
from experimentation.sweep.search.space import (  # noqa: E402
    _FUSED_RANK, _FUSED_VALUE, params_to_overrides, search_space)


def _params(cfg, rank=24):
    space = search_space(cfg, base_name="test")
    p = {k: (v["choices"][0] if v["kind"] == "categorical" else v["low"])
         for k, v in space.items()}
    p["ska_rank"] = rank
    return p


# ------------------------------------------------- the constants stay in step ----

def test_the_fused_constants_match_the_kernels_own():
    """space.py duplicates the kernel's requirements. Duplicated constants drift,
    and this one drifting means the search validates against a geometry the
    kernel no longer wants."""
    from koopman_lm.kernels.cuda_prefix_scan import _RANK, _VALUE
    assert _FUSED_RANK == _RANK, "rank constant drifted from the kernel"
    assert _FUSED_VALUE == _VALUE, "value-width constant drifted from the kernel"


# ------------------------------------------------------- fused_only rejects ----

def test_fused_only_rejects_a_base_spec_the_kernel_cannot_serve():
    cfg = resolve_run_spec(REPO / "configs/runs/4m-golden.yaml").model
    assert cfg.d_model // cfg.ska_n_heads != _FUSED_VALUE, "fixture no longer differs"
    with pytest.raises(ValueError, match="value width"):
        params_to_overrides(_params(cfg), cfg, max_steps=200,
                            backend_policy="fused_only")


def test_fused_only_accepts_the_geometry_it_was_built_for():
    cfg = resolve_run_spec(REPO / "configs/runs/50m-first-real.yaml").model
    assert cfg.d_model // cfg.ska_n_heads == _FUSED_VALUE
    ov = params_to_overrides(_params(cfg), cfg, max_steps=200,
                             backend_policy="fused_only")
    assert ov["model.ska_backend"] == "cuda_prefix"


def test_the_rejection_says_a_sampled_rank_cannot_fix_it():
    """Without that, the obvious reading is 'try another rank', which cannot
    work -- value width is d_model/ska_n_heads."""
    cfg = resolve_run_spec(REPO / "configs/runs/4m-golden.yaml").model
    with pytest.raises(ValueError, match="no sampled rank"):
        params_to_overrides(_params(cfg), cfg, max_steps=200,
                            backend_policy="fused_only")


def test_a_bad_rank_is_still_rejected():
    cfg = resolve_run_spec(REPO / "configs/runs/50m-first-real.yaml").model
    with pytest.raises(ValueError, match="ska_rank"):
        params_to_overrides(_params(cfg, rank=8), cfg, max_steps=200,
                            backend_policy="fused_only")


# ------------------------------------------------------ auto warns, once ----

def _unsupported_shapes(value_width=32):
    B, T, H, r = 1, 32, 1, 24
    return (torch.randn(B, T, H, r), torch.randn(B, T, H, r),
            torch.randn(B, T, H, value_width))


def test_a_cpu_run_does_not_warn():
    """On CPU the reference scan is not a fallback, it is the ONLY
    implementation -- the fused kernel could never have run. An earlier version
    warned here and fired 19 times across the unit suite on 4x3 toy tensors,
    which is precisely how a warning gets filtered and then ignored when it
    finally matters."""
    from koopman_lm.kernels import prefix_scan as PS
    PS._WARNED_GEOMETRIES.clear()
    x, q, v = _unsupported_shapes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PS.ska_prefix_scan(x, q, v, 0.01, 1, 32, 0.0, backend="auto")
    assert not caught, f"CPU should be silent, got {[str(c.message)[:60] for c in caught]}"


@pytest.mark.gpu
def test_auto_warns_on_cuda_when_it_falls_back(  # pragma: no cover - needs a GPU
):
    """The case that cost two hours: a real GPU, a geometry the kernel cannot
    serve, and no diagnostic."""
    from koopman_lm.kernels import prefix_scan as PS
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    PS._WARNED_GEOMETRIES.clear()
    x, q, v = (t.cuda() for t in _unsupported_shapes())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PS.ska_prefix_scan(x, q, v, 0.01, 1, 32, 0.0, backend="auto")
        PS.ska_prefix_scan(x, q, v, 0.01, 1, 32, 0.0, backend="auto")
    assert len(caught) == 1, f"expected one warning for two calls, got {len(caught)}"
    msg = str(caught[0].message)
    assert "value_width=32" in msg and "137x" in msg


def test_the_dedup_key_distinguishes_geometries():
    """Warned once per geometry, not once per call: this is the forward path, and
    warnings' own once-per-location dedup is not enough because one call site
    serves every geometry."""
    from koopman_lm.kernels import prefix_scan as PS
    PS._WARNED_GEOMETRIES.clear()
    assert PS._WARNED_GEOMETRIES == set()
    src = (REPO / "koopman_lm/kernels/prefix_scan.py").read_text()
    assert "_WARNED_GEOMETRIES.add(key)" in src
    assert "int(vbar.shape[-1])" in src, "value width must be part of the key"


def test_the_cuda_error_also_names_the_geometry_it_got():
    """It named the requirements and not the actual values, so the reader had to
    go compute d_model/ska_n_heads themselves."""
    from koopman_lm.kernels import prefix_scan as PS
    x, q, v = _unsupported_shapes()
    with pytest.raises(RuntimeError, match="value_width=32"):
        PS.ska_prefix_scan(x, q, v, 0.01, 1, 32, 0.0, backend="cuda_prefix")


# ------------------------------------- the chunked path must announce itself ----

def test_the_chunked_path_warns_at_construction():
    """It is not a slower-but-fine alternative -- it computes something else.

    With no exact flag set, chunk_stats uses exclusive-chunk-prefix boundaries,
    dropping every within-chunk lag-1..lag-(S-1) cross-covariance term.
    chunk_stats_exact.py's own header measures that at "~100% relative error vs a
    true per-token-causal reference on short-range recall" -- which is the
    capability SKA exists for. A number from this configuration is an upper bound
    on SPEED, not a result.

    At CONSTRUCTION rather than per forward: once per model, before any time is
    spent."""
    from koopman_lm.modules.seq.ska import SKAModule
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SKAModule(d_model=128, n_heads=4, rank=24)
    chunked = [str(c.message) for c in caught if "CHUNKED" in str(c.message)]
    assert len(chunked) == 1, f"expected one chunked warning, got {len(chunked)}"
    msg = chunked[0]
    assert "100% RELATIVE ERROR" in msg, "must quantify, not just caution"
    assert "upper bound on SPEED" in msg, "must say what the number IS good for"
    for route in ("ska_inverse_cholesky", "ska_exact_intrachunk", "ska_prefix_scan"):
        assert route in msg, f"must name the exact alternative {route}"
    # A quantity with no artefact behind it is a rumour. "~100%" was one until
    # 2026-08-24: its only ancestor was commit 40f6653, a bulk import of an
    # external tree whose header asserted it with no harness, and this very test
    # pinned the STRING while the comment above it claimed the header "measures"
    # it. Both citations below are real and reachable from here.
    assert "440122" in msg, "cite the job that measured the forward error"
    assert "test_chunked_route_staleness" in msg, (
        "cite the in-suite characterisation, which is the artefact a reader can "
        "actually run")
    # and the mechanism, not just the scalar: chunk_size is the variable, and
    # max_seq_len is the one a reader would otherwise assume dilutes it
    assert "ska_chunk_size" in msg
    assert "max_seq_len" in msg, (
        "must say the error does NOT shrink with sequence length -- the ladder "
        "is all max_seq_len 8192 and that is the natural wrong assumption")


def test_the_chunked_warning_does_not_recommend_a_route_the_config_cannot_run():
    """`inverse_cholesky` is offered first because it is cheapest. It also
    asserts rank <= 64, and `440m`/`880m` (96) and `1p5b`/`3b` (128) are past
    that -- so on exactly the configs where the warning matters most, its first
    suggestion would have crashed at construction. Naming an unavailable remedy
    is worse than naming none: it reads as "one flag away" when the real
    question is the config's rank."""
    from koopman_lm.modules.seq.ska import SKAModule
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SKAModule(d_model=1024, n_heads=16, rank=96, chunk_size=96)
    msg = [str(c.message) for c in caught if "CHUNKED" in str(c.message)][0]
    assert "UNAVAILABLE" in msg, "must say invchol cannot serve this rank"
    assert "rank 96" in msg, "must name the rank it got"
    assert "ska_inverse_cholesky=True" not in msg, (
        "must not offer the flag it just said is unavailable")
    # the two routes that DO work at any geometry are still offered
    assert "ska_exact_intrachunk=True" in msg
    assert "ska_prefix_scan=True" in msg


@pytest.mark.parametrize("flag", ["prefix_scan", "inverse_cholesky",
                                  "exact_intrachunk"])
def test_any_exact_route_silences_the_chunked_warning(flag):
    """Otherwise it becomes noise and gets filtered, which is how the silent
    prefix-scan fallback stayed invisible for as long as it did."""
    from koopman_lm.modules.seq.ska import SKAModule
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SKAModule(d_model=128, n_heads=4, rank=24, **{flag: True})
    assert not [c for c in caught if "CHUNKED" in str(c.message)], (
        f"{flag}=True is an exact route and must not be warned about")
