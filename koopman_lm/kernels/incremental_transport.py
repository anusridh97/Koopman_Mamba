"""Incremental (L, A, R) Givens rotation-replay transport for O(r^2)/token decode.

Torch port of the machine-precision-validated NumPy oracle
(code-tests/test_incremental_lar_parity.py, commit 38b04a7). Maintains the
carried whitened operator A = L^-1 M L^-T and value map R = C L^-T incrementally
via the SAME Givens rotations that update the Cholesky factor L, so a decode step
costs O(r^2) instead of the O(r^3) per-step re-whiten + solves the current
recurrent path pays.

STAGE: reference build (explicit, batched loops over the r rotations; the
18r^2+15r fused kernel is a SEPARATE later step, never the same commit). This
module is standalone and NOT yet wired into the live decode path -- promotion
into recurrent.py._ska_step is gated on the torch parity test
(code-tests/test_incremental_transport_torch.py) being green AND a
decode-vs-recompute parity check, behind a default-off flag. It changes nothing
until then.

The math (Phase 1-4, per the transport derivation):
  Phase 1  augmented rank-1 Cholesky update [L_{t-1} | x_t] -> [L_t | 0];
           recovers u_t = L_t^-1 x_t for free from the Givens sweep.
  Phase 2  A_core = TL_r(Q^T [A_{t-1};0] Q) by replay (left rows then right
           cols, k ascending); A_t = A_core + u_t v_w^T  (M += x_t x_{t-1}^T).
  Phase 3  R_minus = C_{t-1} L_t^-T = first_r([R_{t-1}|0] Q) (right sweep only);
           R_t = R_minus + vbar_t u_t^T  (C += vbar_t x_t^T).
  Phase 4  v_w = L_t^-1 x_{t-1} (the one triangular solve/step) + the read.

The single kernel rot(a,b;c,s) = (c*a+s*b, -s*a+c*b) is used four ways and is
NEVER transposed for the left sweep. Inputs are the SYMMETRIC key x = sqrt(beta)z
and vbar = sqrt(beta)v (v1.1 convention); transport is exact for any rank-1
stream regardless of weighting, so this module is invariant to the Gate-1
outcome. FP32 throughout.

All tensors carry a leading batch dim N (= B*H, flattened heads).
"""
import torch


def _tri_solve_lower(L, B):
    return torch.linalg.solve_triangular(L, B, upper=False)


def phase1_chol_update(L_prev, x):
    """[L_{t-1} | x] -> [L_t | 0]. Returns (L_t, C, S, u) with u = L_t^-1 x.
    L_prev: (N,r,r) lower-tri; x: (N,r). C,S: (N,r) stored rotations."""
    N, r, _ = L_prev.shape
    L = L_prev.clone()
    z = x.clone()
    C = torch.empty(N, r, device=L.device, dtype=L.dtype)
    S = torch.empty(N, r, device=L.device, dtype=L.dtype)
    u = torch.zeros(N, r, device=L.device, dtype=L.dtype)
    e = torch.ones(N, device=L.device, dtype=L.dtype)
    for k in range(r):
        lk = L[:, k, k]
        zk = z[:, k]
        rho = torch.sqrt(lk * lk + zk * zk).clamp_min(1e-30)
        c = lk / rho
        s = zk / rho
        C[:, k] = c
        S[:, k] = s
        col = L[:, k:, k].clone()          # (N, r-k)  rows i=k..r-1 of column k
        zi = z[:, k:].clone()              # (N, r-k)
        cc = c.unsqueeze(-1)
        ss = s.unsqueeze(-1)
        L[:, k:, k] = cc * col + ss * zi
        z[:, k:] = -ss * col + cc * zi
        u[:, k] = s * e
        e = c * e
    return L, C, S, u


def phase2_transport_A(A_prev, C, S):
    """A_core = TL_r(Q^T [A_{t-1};0,0] Q): LEFT sweep (rows) then RIGHT sweep
    (cols), k ascending, on the zero-padded (r+1)x(r+1) matrix. (N,r,r)->(N,r,r)."""
    N, r, _ = A_prev.shape
    At = torch.zeros(N, r + 1, r + 1, device=A_prev.device, dtype=A_prev.dtype)
    At[:, :r, :r] = A_prev
    last = r
    for k in range(r):                     # LEFT: rows k vs row `last`
        cc = C[:, k].unsqueeze(-1)
        ss = S[:, k].unsqueeze(-1)
        rk = At[:, k, :].clone()
        rl = At[:, last, :].clone()
        At[:, k, :] = cc * rk + ss * rl
        At[:, last, :] = -ss * rk + cc * rl
    for k in range(r):                     # RIGHT: cols k vs col `last`
        cc = C[:, k].unsqueeze(-1)
        ss = S[:, k].unsqueeze(-1)
        ck = At[:, :, k].clone()
        cl = At[:, :, last].clone()
        At[:, :, k] = cc * ck + ss * cl
        At[:, :, last] = -ss * ck + cc * cl
    return At[:, :r, :r].contiguous()


def phase3_transport_R(R_prev, C, S):
    """R_minus = C_{t-1} L_t^-T = first_r([R_{t-1} | 0] Q): RIGHT sweep only.
    (N,P,r) -> (N,P,r)."""
    N, P, r = R_prev.shape
    Rt = torch.zeros(N, P, r + 1, device=R_prev.device, dtype=R_prev.dtype)
    Rt[:, :, :r] = R_prev
    last = r
    for k in range(r):
        cc = C[:, k].unsqueeze(-1)
        ss = S[:, k].unsqueeze(-1)
        ck = Rt[:, :, k].clone()
        cl = Rt[:, :, last].clone()
        Rt[:, :, k] = cc * ck + ss * cl
        Rt[:, :, last] = -ss * ck + cc * cl
    return Rt[:, :, :r].contiguous()


def transport_write(L, A, R, x_last, x_t, vbar_t):
    """One WRITE step: fold token t into (L,A,R). Returns updated (L,A,R,u,v_w).
    x_last: previous symmetric key (N,r) or None at t=0; x_t,vbar_t: (N,r),(N,P)."""
    N, r, _ = L.shape
    L_new, C, S, u = phase1_chol_update(L, x_t)
    if x_last is None:
        v_w = torch.zeros(N, r, device=L.device, dtype=L.dtype)
    else:
        v_w = _tri_solve_lower(L_new, x_last.unsqueeze(-1)).squeeze(-1)   # L_new^-1 x_{t-1}
    A_new = phase2_transport_A(A, C, S) + torch.einsum('nr,ns->nrs', u, v_w)
    R_new = phase3_transport_R(R, C, S) + torch.einsum('np,nr->npr', vbar_t, u)
    return L_new, A_new, R_new, u, v_w


def read(L, A, R, q, K, eta=1.0):
    """y = eta * R (A^K L^-1 q). q:(N,r) -> y:(N,P). Contractive A (||A||<=1) so
    no spectral normalization -- the sqrt-beta guarantee (do NOT clamp here)."""
    qw = _tri_solve_lower(L, q.unsqueeze(-1))          # (N,r,1)
    h = qw
    for _ in range(K):
        h = A @ h
    return eta * (R @ h).squeeze(-1)


@torch.no_grad()
def residuals_vs_raw(L, A, R, G, M, C):
    """Debug parity: Frobenius residuals of the carried (L,A,R) against a fresh
    factorization of the raw (G,M,C). Same checks as the NumPy oracle; use in a
    decode debug mode to gate promotion into the live path."""
    Lr = torch.linalg.cholesky(G)
    Aref = _tri_solve_lower(Lr, _tri_solve_lower(Lr, M.transpose(-1, -2)).transpose(-1, -2))
    Rref = _tri_solve_lower(Lr, C.transpose(-1, -2)).transpose(-1, -2)
    def rel(X, Y):
        return (torch.linalg.matrix_norm(X - Y) / (torch.linalg.matrix_norm(Y) + 1e-30)).max().item()
    return {
        "rG": (torch.linalg.matrix_norm(L @ L.transpose(-1, -2) - G)
               / (torch.linalg.matrix_norm(G) + 1e-30)).max().item(),
        "rA": rel(A, Aref),
        "rR": rel(R, Rref),
        "sigma_max_A": torch.linalg.matrix_norm(A, ord=2).max().item(),
    }
