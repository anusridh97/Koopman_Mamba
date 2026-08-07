"""Torch parity test for the incremental (L,A,R) transport kernel.

The acceptance harness for koopman_lm/globals/modules/ska/incremental_transport.py:
streams rank-1 symmetric-key updates through the TORCH primitives and asserts,
per step, the same Frobenius residuals the NumPy oracle
(test_incremental_lar_parity.py, 38b04a7) validated -- now against the actual
torch code that will drive decode. float64 for machine-precision parity.

This is the gate that promotes the transport kernel into the live decode path
(recurrent.py._ska_step behind a default-off flag): it must be green here AND
the decode-vs-recompute parity must hold before the flag is enabled.

Requires torch (the shared conftest imports it); it is a CI test, not a
torch-less one.
"""
import pytest
import torch

from koopman_lm.kernels.incremental_transport import (
    transport_write, read, residuals_vs_raw)
from koopman_lm.kernels.lin_alg import tri_solve_lower as _tri_solve_lower

pytestmark = pytest.mark.correctness


def _stream(N, r, P, T, K, eps, seed):
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    L = (eps ** 0.5) * torch.eye(r, dtype=dt).expand(N, r, r).clone()
    A = torch.zeros(N, r, r, dtype=dt)
    R = torch.zeros(N, P, r, dtype=dt)
    x_last = None
    G = eps * torch.eye(r, dtype=dt).expand(N, r, r).clone()
    M = torch.zeros(N, r, r, dtype=dt)
    C = torch.zeros(N, P, r, dtype=dt)
    worst = {"rG": 0.0, "rA": 0.0, "rR": 0.0, "sigma_max_A": 0.0}
    for _ in range(T):
        k = torch.randn(N, r, generator=g, dtype=dt)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        beta = torch.rand(N, 1, generator=g, dtype=dt) * 0.95 + 0.05
        x = beta.sqrt() * k
        vbar = beta.sqrt() * torch.randn(N, P, generator=g, dtype=dt)

        L, A, R, _, _ = transport_write(L, A, R, x_last, x, vbar)
        G = G + torch.einsum('nr,ns->nrs', x, x)
        if x_last is not None:
            M = M + torch.einsum('nr,ns->nrs', x, x_last)
        C = C + torch.einsum('np,nr->npr', vbar, x)
        x_last = x

        res = residuals_vs_raw(L, A, R, G, M, C)
        for kk in worst:
            worst[kk] = max(worst[kk], res[kk])

    # read parity: y_inc vs y_ref = eta R_ref (A_ref^K L_ref^-1 q)
    q = torch.randn(N, r, generator=g, dtype=dt)
    y_inc = read(L, A, R, q, K, eta=1.5)
    Lr = torch.linalg.cholesky(G)
    Aref = _tri_solve_lower(Lr, _tri_solve_lower(Lr, M.transpose(-1, -2)).transpose(-1, -2))
    Rref = _tri_solve_lower(Lr, C.transpose(-1, -2)).transpose(-1, -2)
    qw = _tri_solve_lower(Lr, q.unsqueeze(-1))
    h = qw
    for _ in range(K):
        h = Aref @ h
    y_ref = 1.5 * (Rref @ h).squeeze(-1)
    rY = ((y_inc - y_ref).norm(dim=-1) / (y_ref.norm(dim=-1) + 1e-30)).max().item()
    return worst, rY


@pytest.mark.parametrize("r", [16, 48, 64])
def test_transport_matches_fresh_factorization(r):
    worst, rY = _stream(N=6, r=r, P=8, T=150, K=2, eps=1e-3, seed=r)
    assert worst["rG"] < 1e-9, worst
    assert worst["rA"] < 1e-9, worst          # two-sided A transport
    assert worst["rR"] < 1e-9, worst          # one-sided R drag-along
    assert rY < 1e-9, rY                      # read parity
    assert worst["sigma_max_A"] <= 1.0 + 1e-9, worst   # sqrt-beta contractivity
