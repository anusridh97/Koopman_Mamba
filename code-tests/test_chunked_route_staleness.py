"""What the chunked route's error actually IS -- measured here, not quoted.

`SKAModule.__init__` warns that the chunked route carries "~100% RELATIVE ERROR
against a per-token-causal reference on short-range recall". That figure had no
artefact in this repo when this file was written. Its textual ancestor is commit
`40f6653` ("Add files via upload", 2026-05-21), a bulk import of an external
`echo-ska-440m` tree, where `chunk_stats_exact_torch.py`'s header already read
"Measured cost: ~100% relative error vs a true per-token-causal reference on
short-range recall". No harness, no geometry, no job id came with it, and
`test_backend_geometry.py::test_the_chunked_path_warns_at_construction` pins only
the STRING. `space.py`'s route table later put a real measurement behind the same
claim (jobs 440122 / 440135, "0.92 - 1.52" forward error) but recorded no
geometry either, and neither job's harness is committed.

So this file measures it, in fp64, in the CPU suite, at the geometry each
assertion names. The point is not to re-quote "100%" -- it is that "100%" is a
SUMMARY of a structure, and the structure is what tells you which configs are
affected and how:

  * The chunked route is EXACT for the first token of every chunk, and its error
    grows with the token's offset INTO its chunk. It is not uniformly wrong.
  * For a token at offset j in its chunk, lag-1 associative recall is destroyed
    for query-to-key distances 2 <= d <= j+1 and INFLATED (~+20%) for d > j+1.
    The window is a step function of d, not a decay -- so "concentrated at short
    lags" is right, and the cut-off is `j`, not some correlation length.
  * The controlling variable is `ska_chunk_size` ALONE. Holding chunk size fixed
    and growing the sequence 128 -> 1024 does not change the relative error, so a
    long-context run is no less affected than a short one. (This independently
    reproduces space.py's "no dependence on sequence length".)

Everything here is deterministic, CPU-only, fp64, and seconds to run.
"""

import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from koopman_lm.kernels.chunk_stats import (  # noqa: E402
    chunk_stats as causal_chunk_stats, symmetric_key_value)
from koopman_lm.kernels.chunk_stats_exact import exact_stats  # noqa: E402
from koopman_lm.kernels.inverse_cholesky import (  # noqa: E402
    ska_exact_inverse_cholesky)
from koopman_lm.kernels.ska_operator import ska_core  # noqa: E402

DT = torch.float64
RIDGE = 0.01
K = 1
R = 24          # proxy-256x17's ska_rank
P = 64          # proxy-256x17's value width (d_model 256 / 4 heads)


# --------------------------------------------------------------- fixtures ----

def _stream(B, T, H=1, seed=0, beta_one=True):
    """A key/value stream in the v1.1 symmetric convention."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T, H, R, dtype=DT, generator=g)
    v = torch.randn(B, T, H, P, dtype=DT, generator=g)
    z_n = z / z.norm(dim=-1, keepdim=True)
    beta = (torch.ones(B, T, H, dtype=DT) if beta_one
            else torch.sigmoid(torch.randn(B, T, H, dtype=DT, generator=g)))
    x_n, v_w = symmetric_key_value(z_n, beta, v)
    return z_n, v, x_n, v_w


def _y_chunked(x_n, zq_n, v_w, CS):
    Gf, Mf, Cf, qf, shp = causal_chunk_stats(x_n, x_n, zq_n, v_w, RIDGE, CS)
    Y = ska_core(Gf, Mf, Cf, qf, K)
    Bc, nc, Hc, Pc, S, Tt, _pad = shp
    Y = Y.reshape(Bc, nc, Hc, Pc, S).permute(0, 1, 4, 2, 3)
    return Y.reshape(Bc, nc * S, Hc, Pc)[:, :Tt]


def _whitened(G, M, Cv, q):
    """y = Cv L^-T (L^-1 M L^-T)^K L^-1 q -- the gauge of _ska_whitened_forward.

    Used only to apply MANY queries to ONE set of statistics, which is what makes
    the distance sweep below cheap. Checked against the kernels themselves by
    `test_the_direct_readout_matches_the_kernels`.
    """
    N, r = G.shape[0], G.shape[-1]
    eye = torch.eye(r, dtype=G.dtype).expand(N, r, r)
    Li = torch.linalg.solve_triangular(torch.linalg.cholesky(G), eye, upper=False)
    W = Li @ M @ Li.transpose(-1, -2)
    u = Li @ q
    for _ in range(K):
        u = W @ u
    return (Cv @ Li.transpose(-1, -2)) @ u


def _stats_serving(x_n, v_w, t, CS):
    """((G,M,Cv) the EXACT route gives token t, (G,M,Cv) the CHUNKED route does)."""
    B, T = x_n.shape[0], x_n.shape[1]
    zero_q = torch.zeros_like(x_n)
    Ge, Me, Ce, _, (_B, Te, He, _P) = exact_stats(x_n, x_n, zero_q, v_w, RIDGE)
    ie = torch.arange(B) * Te * He + t * He                     # H == 1
    Gc, Mc, Cc, _, shp = causal_chunk_stats(x_n, x_n, zero_q, v_w, RIDGE, CS)
    nc = shp[1]
    ic = torch.arange(B) * nc * He + (t // CS) * He
    return (Ge[ie], Me[ie], Ce[ie]), (Gc[ic], Mc[ic], Cc[ic])


# ------------------------------------------------- the reference is a reference ----

def test_the_two_exact_routes_agree_in_fp64():
    """invchol is the ground truth used below, so it has to BE one. Both build the
    same per-token exclusive prefix; they must not merely be close."""
    _z, _v, x_n, v_w = _stream(2, 256)
    from koopman_lm.kernels.chunk_stats_exact import exact_stats as _es
    from koopman_lm.kernels.factor_scan import all_prefix_chol, ska_core_given_L
    g = torch.Generator().manual_seed(11)
    zq = torch.randn(2, 256, 1, R, dtype=DT, generator=g)
    zq_n = zq / zq.norm(dim=-1, keepdim=True)

    ic = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)
    Gf, Mf, Cf, qf, (Be, Te, He, Pe) = _es(x_n, x_n, zq_n, v_w, RIDGE)
    w = x_n.permute(0, 2, 1, 3).reshape(Be * He, Te, R)
    Lf = all_prefix_chol(w, RIDGE + 1e-4, downsweep='qr')
    Lf = Lf.reshape(Be, He, Te, R, R).permute(0, 2, 1, 3, 4).reshape(-1, R, R)
    ei = ska_core_given_L(Gf, Mf, Cf, qf, Lf, K).reshape(Be, Te, He, Pe)

    rel = ((ic - ei).norm() / ei.norm()).item()
    assert rel < 1e-11, f"the two exact routes disagree at {rel:.2e}"


def test_the_direct_readout_matches_the_kernels():
    """`_whitened` is a shortcut, not a second implementation. If it drifts from
    ska_core the distance sweep measures the shortcut instead of the route."""
    _z, _v, x_n, v_w = _stream(4, 128)
    g = torch.Generator().manual_seed(7)
    zq = torch.randn(4, 128, 1, R, dtype=DT, generator=g)
    zq_n = zq / zq.norm(dim=-1, keepdim=True)
    CS, t = 16, 5 * 16 + 9
    ex, ch = _stats_serving(x_n, v_w, t, CS)
    q = zq_n[:, t, 0].unsqueeze(-1)
    ke = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)[:, t, 0]
    kc = _y_chunked(x_n, zq_n, v_w, CS)[:, t, 0]
    assert ((_whitened(*ex, q)[:, :, 0] - ke).norm() / ke.norm()).item() < 1e-11
    assert ((_whitened(*ch, q)[:, :, 0] - kc).norm() / kc.norm()).item() < 1e-11


# ------------------------------------------- the error is NOT uniform in t ----

@pytest.mark.parametrize("CS", [16, 64])
def test_the_first_token_of_every_chunk_is_served_exactly(CS):
    """The chunked route is not uniformly approximate. A token at offset 0 sees
    the exclusive prefix over chunks < c, which is every token up to t-1 -- i.e.
    exactly what the per-token reference gives it. Any claim of the form "the
    chunked route is X% wrong" is therefore an average over offsets, and the
    average is the wrong summary for deciding whether a config is affected."""
    T = 256
    _z, _v, x_n, v_w = _stream(2, T)
    g = torch.Generator().manual_seed(3)
    zq = torch.randn(2, T, 1, R, dtype=DT, generator=g)
    zq_n = zq / zq.norm(dim=-1, keepdim=True)
    ex = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)
    ch = _y_chunked(x_n, zq_n, v_w, CS)
    offs = torch.arange(T) % CS
    at0 = offs == 0
    rel0 = ((ch[:, at0] - ex[:, at0]).norm() / ex[:, at0].norm()).item()
    rel_last = ((ch[:, offs == CS - 1] - ex[:, offs == CS - 1]).norm()
                / ex[:, offs == CS - 1].norm()).item()
    assert rel0 < 1e-11, f"offset-0 tokens should be exact, got {rel0:.2e}"
    assert rel_last > 0.5, (
        f"offset-{CS-1} tokens should be badly wrong, got {rel_last:.3f}")


def test_error_grows_monotonically_with_offset_into_the_chunk():
    """The mechanism is staleness: a token at offset j is missing j tokens of its
    own history. So the error is a function of j, and this is what makes
    `ska_chunk_size` -- not sequence length -- the variable that matters."""
    CS, T = 32, 512
    _z, _v, x_n, v_w = _stream(4, T)
    g = torch.Generator().manual_seed(5)
    zq = torch.randn(4, T, 1, R, dtype=DT, generator=g)
    zq_n = zq / zq.norm(dim=-1, keepdim=True)
    ex = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)
    ch = _y_chunked(x_n, zq_n, v_w, CS)
    offs = torch.arange(T) % CS
    curve = []
    for j in range(CS):
        m = offs == j
        curve.append(((ch[:, m] - ex[:, m]).norm() / ex[:, m].norm()).item())
    assert curve[0] < 1e-11
    # rises fast and then saturates near 1: check the shape, not each point
    assert curve[1] > curve[0]
    assert curve[4] > curve[1]
    assert curve[-1] > 0.8
    assert max(curve[:8]) <= max(curve), "not monotone in the large"


# ----------------------------- the lag structure: a STEP, at the chunk offset ----

def test_lag1_recall_is_destroyed_exactly_inside_the_chunk_and_inflated_outside():
    """The sharpest statement in this file.

    SKA's memory is M = sum_i x_i x_{i-1}^T, so querying with the key that
    occurred at position p retrieves the VALUE at p+1: M x_p -> x_{p+1}, then
    Cv x_{p+1} -> v_{p+1}. That is lag-1 associative recall, and it is what the
    warning means by "short-range recall".

    For a query at offset j in its chunk, the chunked route holds pairs (i-1, i)
    only for i <= t-j-1, so the pair it needs (i = p+1 = t-d+1) is present iff
    d >= j+2. The exact route needs only d >= 2. Hence a DEAD WINDOW
    2 <= d <= j+1 -- j values of d, independent of sequence length -- and outside
    it the chunked answer is INFLATED, because its G and Cv were accumulated over
    a shorter prefix and are therefore less ridge-shrunk.
    """
    CS, T, B, j = 64, 512, 512, 32
    t = 5 * CS + j
    z_n, v, x_n, v_w = _stream(B, T, seed=0)
    ex, ch = _stats_serving(x_n, v_w, t, CS)
    ds = list(range(2, 40))
    q = torch.stack([z_n[:, t - d, 0] for d in ds], dim=-1)          # (B,R,D)
    tgt = torch.stack([v[:, t - d + 1, 0] for d in ds], dim=-1)
    tgt = tgt / tgt.norm(dim=-2, keepdim=True)
    se = (_whitened(*ex, q) * tgt).sum(-2).mean(0)                   # (D,)
    sc = (_whitened(*ch, q) * tgt).sum(-2).mean(0)
    ratio = {d: (sc[i] / se[i]).item() for i, d in enumerate(ds)}

    # the exact route recalls, at every distance
    assert min(se).item() > 0.02, f"reference recall too weak: {min(se).item():.4f}"

    inside = [ratio[d] for d in ds if d <= j + 1]
    outside = [ratio[d] for d in ds if d > j + 1]
    assert inside and outside, "the sweep must straddle the step"

    # INSIDE the window: recall is gone. Retained magnitude is small and the
    # sign is not even reliable -- residual crosstalk, not a weakened signal.
    assert max(abs(x) for x in inside) < 0.35, (
        f"inside the dead window recall should be destroyed; max |ratio| "
        f"= {max(abs(x) for x in inside):.3f}")
    assert min(inside) < 0, "with the signal gone, some crosstalk should be negative"

    # OUTSIDE: recovered, and biased HIGH (less ridge shrinkage).
    assert min(outside) > 1.05, f"outside should be inflated, got {min(outside):.3f}"
    assert max(outside) < 1.6, f"inflation is ~20%, not unbounded: {max(outside):.3f}"

    # and the step lands where the algebra says, not one either side
    assert abs(ratio[j + 1]) < 0.35 < ratio[j + 2], (
        f"step misplaced: ratio[{j+1}]={ratio[j+1]:.3f} "
        f"ratio[{j+2}]={ratio[j+2]:.3f}")


def test_the_dead_window_scales_with_chunk_size_not_sequence_length():
    """`ska_chunk_size` is the knob. 1m runs 16, 440m runs 96, 3b runs 128, so the
    same code is between 15/16 and 127/128 wrong at lag 1 depending only on this
    field -- and `max_seq_len` (4096 on 1m, 8192 on 3b) does not enter."""
    for CS in (16, 64):
        j = CS - 1                      # worst-case offset
        T, B = 4 * CS, 256
        t = 2 * CS + j
        z_n, v, x_n, v_w = _stream(B, T, seed=1)
        ex, ch = _stats_serving(x_n, v_w, t, CS)
        for d, expect_dead in ((2, True), (j + 1, True), (j + 2, False)):
            q = z_n[:, t - d, 0].unsqueeze(-1)
            tg = v[:, t - d + 1, 0]
            tg = tg / tg.norm(dim=-1, keepdim=True)
            se = (_whitened(*ex, q)[:, :, 0] * tg).sum(-1).mean().item()
            sc = (_whitened(*ch, q)[:, :, 0] * tg).sum(-1).mean().item()
            r = sc / se
            if expect_dead:
                assert abs(r) < 0.35, f"CS={CS} d={d}: expected dead, ratio={r:.3f}"
            else:
                assert r > 1.05, f"CS={CS} d={d}: expected recovered, ratio={r:.3f}"


def test_relative_error_is_independent_of_sequence_length():
    """Reproduces space.py's "no dependence on sequence length" claim, which is
    the one that decides whether the ladder's 8192-token configs are less
    affected than a toy. They are not."""
    CS = 32
    errs = {}
    for T in (128, 256, 512, 1024):
        _z, _v, x_n, v_w = _stream(2, T, seed=2)
        g = torch.Generator().manual_seed(9)
        zq = torch.randn(2, T, 1, R, dtype=DT, generator=g)
        zq_n = zq / zq.norm(dim=-1, keepdim=True)
        ex = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)
        ch = _y_chunked(x_n, zq_n, v_w, CS)
        errs[T] = ((ch - ex).norm() / ex.norm()).item()
    lo, hi = min(errs.values()), max(errs.values())
    assert lo > 0.9, f"error should be ~100% at every length, got {errs}"
    assert hi - lo < 0.12, f"error must not depend on T; spread {hi-lo:.3f}: {errs}"


def test_the_quoted_hundred_percent_is_reproduced_at_the_ladders_chunk_sizes():
    """The warning's headline number, with a geometry attached at last.

    Global relative L2 error of the chunked output against the per-token exact
    reference, at each `ska_chunk_size` the committed configs actually use.
    This is the artefact the "~100%" claim never had.
    """
    T = 512
    _z, _v, x_n, v_w = _stream(2, T, seed=4)
    g = torch.Generator().manual_seed(13)
    zq = torch.randn(2, T, 1, R, dtype=DT, generator=g)
    zq_n = zq / zq.norm(dim=-1, keepdim=True)
    ex = ska_exact_inverse_cholesky(x_n, zq_n, v_w, RIDGE, K)
    for CS in (16, 64, 96, 128):        # 1m / 180m*+370m / 440m / 880m+1p5b+3b
        ch = _y_chunked(x_n, zq_n, v_w, CS)
        rel = ((ch - ex).norm() / ex.norm()).item()
        nrm = (ch.norm() / ex.norm()).item()
        assert 0.80 < rel < 1.55, f"CS={CS}: rel err {rel:.3f} outside the band"
        # and it is not a small perturbation of the right answer: at the larger
        # chunk sizes most of the operator's OUTPUT MAGNITUDE is missing too.
        if CS >= 64:
            assert nrm < 0.55, f"CS={CS}: expected heavy attenuation, ||ch||/||ex||={nrm:.3f}"
