"""Cholesky-update correctness in fp64 (scaling plan Phase 0, test 1).

The O(r^2) carried-factor updates must equal a full O(r^3) re-factorization.
All pure-torch linear algebra -> runs on CPU. fp64 target: <= 1e-10 (the
docstrings claim ~1e-16; we assert a safe band and report the actual error).
"""
import math

import pytest
import torch

from koopman_lm.kernels.cholesky_update import (
    update_reference,
    update_L_only,
    cholesky_rank1_update_,
)
from koopman_lm.kernels.factor_scan import all_prefix_chol, rank1_chol_update_

pytestmark = pytest.mark.correctness

FP64_TOL = 1e-10


def _spd(r, ridge=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, r, dtype=torch.float64, generator=g)
    return A @ A.T + ridge * r * torch.eye(r, dtype=torch.float64)


def test_rank1_update_matches_fresh_cholesky():
    r = 24
    G = _spd(r, seed=1)
    L = torch.linalg.cholesky(G)
    z = torch.randn(r, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    cholesky_rank1_update_(L, z)            # L -> chol(G + zz^T), in place
    L_ref = torch.linalg.cholesky(G + torch.outer(z, z))
    err = (L - L_ref).abs().max().item()
    assert err < FP64_TOL, f"rank1 cholupdate err {err:.2e}"


def test_update_reference_double_sided():
    r = 20
    G = _spd(r, seed=3)
    M = 0.3 * torch.randn(r, r, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(4))
    z = torch.randn(r, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    w = torch.randn(r, dtype=torch.float64, generator=torch.Generator().manual_seed(6))

    L = torch.linalg.cholesky(G)
    Aw = torch.linalg.solve_triangular(
        L, torch.linalg.solve_triangular(L, M, upper=False).T, upper=False).T
    Aw_plus, vz = update_reference(L, Aw, z, w)   # modifies L -> L+

    # ground truth: full recompute on the updated stats
    G_plus = G + torch.outer(z, z)
    M_plus = M + torch.outer(z, w)
    Lp = torch.linalg.cholesky(G_plus)
    Aw_ref = torch.linalg.solve_triangular(
        Lp, torch.linalg.solve_triangular(Lp, M_plus, upper=False).T, upper=False).T
    vz_ref = torch.linalg.solve_triangular(Lp, z.unsqueeze(1), upper=False).squeeze(1)

    assert (L - Lp).abs().max().item() < FP64_TOL
    assert (Aw_plus - Aw_ref).abs().max().item() < FP64_TOL
    assert (vz - vz_ref).abs().max().item() < FP64_TOL


def test_update_L_only_matches_fresh():
    r = 16
    G = _spd(r, seed=7)
    z = torch.randn(r, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    L = torch.linalg.cholesky(G)
    vz = update_L_only(L, z)
    Lp = torch.linalg.cholesky(G + torch.outer(z, z))
    vz_ref = torch.linalg.solve_triangular(Lp, z.unsqueeze(1), upper=False).squeeze(1)
    assert (L - Lp).abs().max().item() < FP64_TOL
    assert (vz - vz_ref).abs().max().item() < FP64_TOL


def test_all_prefix_chol_matches_fresh_over_all_prefixes():
    # factor scan: L[:,t] == chol(ridge*I + sum_{i<t} w_i w_i^T) for every t
    torch.manual_seed(0)
    N, T, r, ridge = 3, 48, 16, 1e-3
    z = torch.randn(N, T, r, dtype=torch.float64)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    beta = torch.rand(N, T, dtype=torch.float64)
    w = beta.sqrt().unsqueeze(-1) * z
    eye = torch.eye(r, dtype=torch.float64)
    for mode in ("givens", "qr"):
        Ls = all_prefix_chol(w, ridge, downsweep=mode)
        err = 0.0
        for t in range(T):
            G_t = ridge * eye + (w[:, :t].transpose(-1, -2) @ w[:, :t]
                                 if t else torch.zeros(N, r, r, dtype=torch.float64))
            err = max(err, (Ls[:, t] - torch.linalg.cholesky(G_t)).abs().max().item())
        assert err < 1e-9, f"all_prefix_chol({mode}) err {err:.2e}"


def test_unbatched_and_batched_core_are_bit_identical_at_batch_one():
    """cholesky_rank1_update_ (unbatched) and rank1_chol_update_ (batched) are
    two entry points onto the SAME Givens-rotation math (see
    docs/superpowers/specs/2026-08-07-structural-review.md, issue 2). This
    locks in bit-for-bit agreement at batch=1 so a future unification of the
    two cores cannot silently change either one's numerics."""
    for trial in range(200):
        r = 1 + (trial * 7) % 32
        G = _spd(r, ridge=1.0, seed=1000 + trial)
        L = torch.linalg.cholesky(G)
        z = torch.randn(r, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(2000 + trial))

        L_un = L.clone()
        cs, ss = cholesky_rank1_update_(L_un, z)

        L_b = L.clone().unsqueeze(0)
        rank1_chol_update_(L_b, z.clone().unsqueeze(0))

        err = (L_un - L_b.squeeze(0)).abs().max().item()
        assert err == 0.0, f"trial {trial} (r={r}): unbatched vs batched err {err:.3e}"

        # the rotation params must be valid Givens coefficients: c^2+s^2==1
        # (exactly, since c=a/rho, s=b/rho, rho=sqrt(a^2+b^2)), except where
        # rho==0 and the guard sets (c,s)=(1,0), also satisfying c^2+s^2==1.
        one = (cs * cs + ss * ss)
        assert torch.allclose(one, torch.ones_like(one), atol=1e-12), \
            f"trial {trial}: cs^2+ss^2 != 1, max dev {(one - 1).abs().max():.3e}"


def test_batched_core_matches_looped_unbatched_calls():
    """rank1_chol_update_ at B=8 must equal 8 independent unbatched
    cholesky_rank1_update_ calls -- the property the batched kernel is
    supposed to subsume."""
    torch.manual_seed(0)
    B, r = 8, 12
    for trial in range(50):
        Gs = [_spd(r, ridge=1.0, seed=trial * 100 + b) for b in range(B)]
        Ls = [torch.linalg.cholesky(G) for G in Gs]
        zs = [torch.randn(r, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(trial * 100 + 50 + b))
              for b in range(B)]

        L_loop = torch.stack([L.clone() for L in Ls])
        for b in range(B):
            cholesky_rank1_update_(L_loop[b], zs[b])

        L_batch = torch.stack([L.clone() for L in Ls])
        z_batch = torch.stack(zs)
        rank1_chol_update_(L_batch, z_batch.clone())

        err = (L_loop - L_batch).abs().max().item()
        assert err == 0.0, f"trial {trial}: batched vs looped err {err:.3e}"


def test_cholesky_rank1_update_cs_ss_reproduce_the_column_update():
    """Characterizes the (cs, ss) return path: replaying the returned
    rotation coefficients against the ORIGINAL L must reproduce the mutated
    L exactly. This is the API surface update_reference/update_L_only rely on
    (they use cs, ss to also propagate Aw and recover vz) and that the batched
    core (which discards cs, ss) does not provide."""
    r = 10
    G = _spd(r, ridge=1.0, seed=42)
    L0 = torch.linalg.cholesky(G)
    z = torch.randn(r, dtype=torch.float64, generator=torch.Generator().manual_seed(43))

    L = L0.clone()
    cs, ss = cholesky_rank1_update_(L, z)
    assert cs.shape == (r,) and ss.shape == (r,)
    assert cs.dtype == L0.dtype and ss.dtype == L0.dtype

    # replay the rotations against a fresh copy of the pre-update L and z
    L_replay = L0.clone()
    zc = z.clone()
    for k in range(r):
        c, s = cs[k], ss[k]
        col_k = L_replay[:, k].clone()
        L_replay[:, k] = c * col_k + s * zc
        zc = -s * col_k + c * zc
    assert (L_replay - L).abs().max().item() == 0.0


def test_streaming_rank1_matches_prefix_factors():
    # carried-L decode: read with carried L, then rank-1 update; must equal the
    # exclusive-prefix factors at every step (decode == prefill at factor level)
    torch.manual_seed(0)
    N, T, r, ridge = 2, 40, 16, 1e-3
    z = torch.randn(N, T, r, dtype=torch.float64)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    w = torch.rand(N, T, dtype=torch.float64).sqrt().unsqueeze(-1) * z
    Ls = all_prefix_chol(w, ridge, downsweep="givens")
    Lc = math.sqrt(ridge) * torch.eye(r, dtype=torch.float64).expand(N, r, r).contiguous()
    err = 0.0
    for t in range(T):
        err = max(err, (Lc - Ls[:, t]).abs().max().item())   # read-before-write
        rank1_chol_update_(Lc, w[:, t])                       # write after read
    assert err < FP64_TOL, f"streaming rank-1 vs prefix err {err:.2e}"

