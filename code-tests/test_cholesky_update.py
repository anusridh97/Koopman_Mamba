"""Cholesky-update correctness in fp64 (scaling plan Phase 0, test 1).

The O(r^2) carried-factor updates must equal a full O(r^3) re-factorization.
All pure-torch linear algebra -> runs on CPU. fp64 target: <= 1e-10 (the
docstrings claim ~1e-16; we assert a safe band and report the actual error).
"""
import math

import pytest
import torch

from koopman_lm.modules.kernels.cholesky_update import (
    update_reference,
    update_L_only,
    cholesky_rank1_update_,
)
from koopman_lm.modules.kernels.factor_scan import all_prefix_chol, rank1_chol_update_

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

