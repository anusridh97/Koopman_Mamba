"""
echo_jax.py -- Echo (Mamba-2 + SKA + Koopman MLP) in JAX/Flax.

Faithful to "Echo: KV-Cache-Free Associative Recall with Spectral Koopman
Operators" (CAIS '26) and the O(r^2) Cholesky-update backward (NeurIPS submission).
Supersedes ska_jax.py / mamba2_ssd_jax.py / recurrent_inference_jax.py.

Contents:
  * SKA forward  y = eta * C_v (G^{-1} M)^K G^{-1} q          (Echo Eq.13)
  * SKA backward = reverse Krylov recurrence; Cholesky never differentiated,
    O(K r^2)                                                  (NeurIPS App.C.8)
  * Chunk-causal stats, exclusive boundary sum i=1..c-1       (Echo Eq.7-10)
  * Spectral norm: detached power iteration, gamma in [1,1.5] (Echo Eq.12)
  * Causal normalization: per-token L2 on key/query + learned write gate beta
    (replaces the non-causal sequence-max)
  * Additive injection h += eta*h_ska, zero-init out_proj
  * Koopman MLP: block rotation, |lambda|<=1                  (Echo Eq.14)
  * Mamba-2 SSD backbone, exact O(1) decode

Precision: GEMMs in bf16 (MXU), linear algebra (covariance/Cholesky/solve/decay)
in fp32. Operator-core matmuls are pinned to fp32 so this holds on TPU too.

---

Archival status (restored 2026-08-07):

This is a REFERENCE implementation, not production code. It is the JAX/Flax
oracle that the torch port (koopman_lm/) is checked against in
code-tests/test_jax_reference.py (test_backward, test_mamba_recurrence,
test_ska_causal, test_ska_decode, test_ska_newton_schulz); it is not imported
by, or a dependency of, any production path.

Why it was set aside: it was collateral damage from a bulk directory
consolidation, not a deliberate call on this file. The "folder collapsing"
refactor merged in PR #10 (commit 50a7e0d, PR merge b2d965e) fired a long
sequence of "Delete <dir>" commits removing genuinely-superseded trees
(koopman-lm-fast/, echo-ska-440m/koopman_lm, setup.py, ...); the bare
`echo-ska/` directory containing this file was swept up in the same pass by
commit 7262785 ("Delete echo-ska directory"), whose message gives no
file-specific rationale. A later commit (ebcf17d, "Restore folders dropped
by the PR #10 folder collapse") confirms this was accidental: it lists
echo-ska/ among four trees removed that "nothing on main replaces" and
restores the surrounding echo-ska/ directory verbatim, but a mechanical
test-recovery pass (42d9224) brought back test_jax_reference.py without this
file, which is why it was still missing at the archive/reference/ path the
test expects. No commit records a technical or correctness objection to this
implementation itself -- the repo history does not give a substantive reason
this file was set aside beyond "caught in a bulk deletion."
"""

import math
import functools
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import flax
import flax.linen as nn

GEMM = jnp.bfloat16   # matmul inputs (MXU)
LIN = jnp.float32     # linear algebra / accumulation


# 1. SKA operator core  (fp32; custom_vjp; vmapped over N = B*C*H instances)

def _st_l(L, A):  return solve_triangular(L, A, lower=True)        # L^{-1} A
def _st_lt(L, A): return solve_triangular(L.T, A, lower=False)     # L^{-T} A
def _solveG(L, B): return _st_lt(L, _st_l(L, B))                   # G^{-1} B


def _whiten_M(L, M):                      # W = L^{-1} M L^{-T}  (one O(r^3), then reused)
    return _st_l(L, _st_l(L, M).T).T


def _specW(W, iters=20):                  # sigma_max(W) via detached power iteration (Eq.12)
    # iters=20: converges even on ill-conditioned chunks
    r = W.shape[0]
    v = jnp.ones((r, 1), W.dtype) / jnp.sqrt(r)
    for _ in range(iters):
        u = W @ v;   u = u / (jnp.linalg.norm(u) + 1e-8)
        v = W.T @ u; v = v / (jnp.linalg.norm(v) + 1e-8)
    return jax.lax.stop_gradient(1.0 / jnp.maximum(jnp.linalg.norm(W @ v), 1.0))


# Whitened forward (Prop.I.8): (alpha*W)^K and the spectral norm are r x r matmuls;
# only the query-whiten and output-unwhiten are triangular solves.
@partial(jax.custom_vjp, nondiff_argnums=(4,))
def ska_core(G, M, Cv, q, K):            # G,M:(r,r) Cv:(P,r) q:(r,p) -> (P,p)
    y, _ = _ska_fwd(G, M, Cv, q, K)
    return y


@jax.default_matmul_precision('highest')   # operator core = LIN(fp32); matmuls on MXU via 3/6-pass
def _ska_fwd(G, M, Cv, q, K):
    L = jnp.linalg.cholesky(G)
    W = _whiten_M(L, M)                              # reused by spectral + filter + backward
    a0 = _specW(W)
    Us = [_st_l(L, q)]                               # U_0 = L^{-1} q
    for _ in range(K):
        Us.append(a0 * (W @ Us[-1]))                 # matmul (MXU), no solve
    XK = _st_lt(L, Us[K])                            # un-whiten once
    return Cv @ XK, (L, W, jnp.stack(Us, 0), Cv, a0)


@jax.default_matmul_precision('highest')   # operator core = LIN(fp32); matmuls on MXU via 3/6-pass
def _ska_bwd(K, res, dY):                # whitened adjoint (NeurIPS App.C.8); L never differentiated
    L, W, Us, Cv, a0 = res
    dCv = dY @ _st_lt(L, Us[K]).T
    P = _st_l(L, Cv.T @ dY)                          # P_K = L^{-1} Cv^T dY
    dMw = jnp.zeros_like(W); dGw = jnp.zeros_like(W)
    for i in range(K, 0, -1):
        dMw = dMw + a0 * (P @ Us[i - 1].T)           # matmul
        dGw = dGw - (P @ Us[i].T)
        P = a0 * (W.T @ P)                           # matmul (MXU), no solve
    dGw = dGw - (P @ Us[0].T)                         # i = 0 term
    dq = _st_lt(L, P)
    unwhiten = lambda A: _st_lt(L, _st_lt(L, A.T).T) # L^{-T} A L^{-1}
    dM = unwhiten(dMw)
    dG = unwhiten(dGw); dG = 0.5 * (dG + dG.T)
    return (dG, dM, dCv, dq)


ska_core.defvjp(_ska_fwd, _ska_bwd)
ska_core_b = jax.vmap(ska_core, in_axes=(0, 0, 0, 0, None))      # batch N


@jax.default_matmul_precision('highest')   # operator core = LIN(fp32); matmuls on MXU via 3/6-pass
def _ska_apply_L(L, M, Cv, q, K):        # decode forward GIVEN L (no grad, no re-cholesky)
    W = _whiten_M(L, M); a0 = _specW(W)
    U = _st_l(L, q)
    for _ in range(K):
        U = a0 * (W @ U)
    return Cv @ _st_lt(L, U)


def cholupdate(L, x):                     # rank-1 update: L' L'^T = L L^T + x x^T  (O(r^2))
    r = x.shape[0]
    def body(i, st):
        L, x = st
        Lii = L[i, i]; xi = x[i]
        rr = jnp.sqrt(Lii * Lii + xi * xi); c = rr / Lii; s = xi / Lii
        below = (jnp.arange(r) > i).astype(L.dtype)
        col = L[:, i]
        ncol = jnp.where(below > 0, (col + s * x) / c, col).at[i].set(rr)
        x = jnp.where(below > 0, c * x - s * ncol, x)
        return (L.at[:, i].set(ncol), x)
    L, _ = jax.lax.fori_loop(0, r, body, (L, x))
    return L


ska_apply_L_b = jax.vmap(_ska_apply_L, in_axes=(0, 0, 0, 0, None))
_chol_b = jax.vmap(jnp.linalg.cholesky)
_cholup_b = jax.vmap(cholupdate)


# Newton-Schulz operator core (training only, behind a flag): replaces the
# Cholesky + solves with an all-matmul path. S = G^{-1/2} via a Gelfand-scaled
# order-2 residual iteration (R = I - X G X^T, X <- (I + R/2 + 3R^2/8) X); then
# W = S M S and G^{-1} = S S are matmuls. Exact vs the Cholesky core (orthogonal
# similarity L = G^{1/2} Q); inference stays on Cholesky. Ridge eps*I + fp32 keep
# it clear of the half-precision Gram-NS instability.

def _gelfand(G):                          # upper bound on lambda_max(G) (SPD), k=1; ~free
    return jnp.linalg.norm(G @ G) ** 0.5  # = ||G^2||_F^{1/2} >= lambda_max(G)


def _ns_isqrt(G, iters):                  # G^{-1/2}: re-anchored order-2 residual NS (all matmul)
    r = G.shape[0]; I = jnp.eye(r, dtype=G.dtype)
    c = _gelfand(G); B = G / c            # scale eigenvalues into (0, 1]
    X = I
    for _ in range(iters):
        M = X @ B @ X.T; M = 0.5 * (M + M.T)         # re-anchor to B each step (built-in restart)
        R = I - M                                    # residual; (I-R)^{-1/2} ~ I + R/2 + 3R^2/8
        X = (I + 0.5 * R + 0.375 * (R @ R)) @ X       # cubic correction near identity
        X = 0.5 * (X + X.T)                          # keep symmetric (kills accumulation drift)
    return X / jnp.sqrt(c)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def ska_core_ns(G, M, Cv, q, K, ns_iters):
    y, _ = _ska_fwd_ns(G, M, Cv, q, K, ns_iters)
    return y


@jax.default_matmul_precision('highest')   # operator core = LIN(fp32); matmuls on MXU via 3/6-pass
def _ska_fwd_ns(G, M, Cv, q, K, ns_iters):
    S = _ns_isqrt(G, ns_iters)            # symmetric whitener G^{-1/2} (all matmul)
    W = S @ M @ S                         # whitened operator (sigma_max == Cholesky form)
    a0 = _specW(W)
    Us = [S @ q]                          # U_0 = G^{-1/2} q
    for _ in range(K):
        Us.append(a0 * (W @ Us[-1]))      # matmul (MXU)
    return Cv @ (S @ Us[K]), (S, W, jnp.stack(Us, 0), Cv, a0)


@jax.default_matmul_precision('highest')   # operator core = LIN(fp32); matmuls on MXU via 3/6-pass
def _ska_bwd_ns(K, ns_iters, res, dY):    # symmetric whitened adjoint; G^{-1/2} never differentiated
    S, W, Us, Cv, a0 = res
    dCv = dY @ (S @ Us[K]).T
    P = S @ (Cv.T @ dY)                    # P_K = G^{-1/2} Cv^T dY
    dMw = jnp.zeros_like(W); dGw = jnp.zeros_like(W)
    for i in range(K, 0, -1):
        dMw = dMw + a0 * (P @ Us[i - 1].T)
        dGw = dGw - (P @ Us[i].T)
        P = a0 * (W.T @ P)                 # matmul (MXU)
    dGw = dGw - (P @ Us[0].T)              # i = 0 term
    dq = S @ P
    dM = S @ dMw @ S                       # un-whiten (matmul)
    dG = S @ dGw @ S; dG = 0.5 * (dG + dG.T)
    return (dG, dM, dCv, dq)


ska_core_ns.defvjp(_ska_fwd_ns, _ska_bwd_ns)
ska_core_ns_b = jax.vmap(ska_core_ns, in_axes=(0, 0, 0, 0, None, None))


# 2. Chunk-causal sufficient statistics  (Echo Eq.7-10, exclusive boundary)
#    z,zq: (B,T,H,r) ALREADY normalized; v: (B,T,H,P) raw.  All fp32.

def chunk_stats(z, zb, zq, v, ridge, CS):
    # z  : L2-normalized key   (B,T,H,r)   -- lagged / right factor
    # zb : beta-weighted key   (B,T,H,r) = beta * z  -- current / write factor
    # G = sum beta_t z_t z_t^T,  M = sum beta_t z_t z_{t-1}^T,  C_v = sum beta_t v_t z_t^T
    B, T, H, r = z.shape; P = v.shape[-1]
    nc = (T + CS - 1) // CS; pad = nc * CS - T
    if pad:
        z = jnp.pad(z, ((0, 0), (0, pad), (0, 0), (0, 0)))
        zb = jnp.pad(zb, ((0, 0), (0, pad), (0, 0), (0, 0)))
        zq = jnp.pad(zq, ((0, 0), (0, pad), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
    zc = z.reshape(B, nc, CS, H, r); zbc = zb.reshape(B, nc, CS, H, r)
    zqc = zq.reshape(B, nc, CS, H, r); vc = v.reshape(B, nc, CS, H, P)
    Gc = jnp.einsum("bcthr,bcths->bchrs", zbc, zc)              # Eq.7  (beta z z^T)
    Mc = jnp.einsum("bcthr,bcths->bchrs", zbc[:, :, 1:], zc[:, :, :-1])
    Cc = jnp.einsum("bcthp,bcthr->bchpr", vc, zbc)
    bnd = jnp.einsum("bchr,bchs->bchrs", zbc[:, 1:, 0], zc[:, :-1, -1])   # (B,nc-1,H,r,r)
    bnd = jnp.concatenate([jnp.zeros((B, 1, H, r, r), z.dtype), bnd], 1)  # (B,nc,H,r,r)

    def excl(x):                                                # exclusive prefix sum
        c = jnp.cumsum(x, 1)
        return jnp.concatenate([jnp.zeros_like(x[:, :1]), c[:, :-1]], 1)

    eye = jnp.eye(r, dtype=z.dtype)
    G = excl(Gc) + ridge * eye                                  # Eq.8
    M = excl(Mc) + excl(bnd)                                    # Eq.9  (i=1..c-1)
    Cv = excl(Cc)                                               # Eq.10
    N = B * nc * H
    Gf = (0.5 * (G + jnp.swapaxes(G, -1, -2))).reshape(N, r, r) + 1e-4 * eye  # jitter
    qf = jnp.transpose(zqc, (0, 1, 3, 4, 2)).reshape(N, r, CS)
    return Gf, M.reshape(N, r, r), Cv.reshape(N, P, r), qf, (B, nc, H, P, CS, T, pad)


# 3. SKA layer  (Echo Sec.3.2)

def _l2(z):                               # per-token causal L2 normalization (GDN-style)
    return z * jax.lax.rsqrt(jnp.sum(z * z, axis=-1, keepdims=True) + 1e-12)


def _squash(raw, lo, hi):                 # smooth bound to (lo, hi)
    return lo + (hi - lo) * jax.nn.sigmoid(raw)


def _raw_init(val, lo, hi):               # raw value s.t. _squash(raw, lo, hi) == val
    s = (val - lo) / (hi - lo)
    return math.log(s / (1.0 - s))


class SKA(nn.Module):
    """SKA with causal normalization: per-token L2 on key/query + a learned
    per-token write gate beta (Gated DeltaNet style), replacing the non-causal
    sequence-max. Every statistic depends only on tokens <= t, so chunk-causal
    training, prefix recurrence, and per-token decode coincide."""
    d_model: int; n_heads: int; rank: int; head_dim: int
    ridge: float = 1e-3; power_K: int = 2; chunk_size: int = 64
    eta: float = 1.5            # injection scale: learnable, squashed to [eta_min, eta_max], init here
    eta_min: float = 1.4; eta_max: float = 1.7
    gamma: float = 0.7          # operator gain: learnable, squashed to [gamma_min, gamma_max], init here
    gamma_min: float = 0.5; gamma_max: float = 1.5
    use_newton_schulz: bool = False; ns_iters: int = 10   # training-only; inference stays on Cholesky

    def setup(self):
        H, r, P = self.n_heads, self.rank, self.head_dim
        self.k = nn.Dense(H * r, use_bias=False, kernel_init=nn.initializers.orthogonal())
        self.q = nn.Dense(H * r, use_bias=False, kernel_init=nn.initializers.orthogonal())
        self.v = nn.Dense(H * P, use_bias=False, kernel_init=nn.initializers.xavier_uniform())
        self.beta = nn.Dense(H, kernel_init=nn.initializers.zeros)   # write gate; bias 0 -> beta=0.5
        self.o = nn.Dense(self.d_model, use_bias=False, kernel_init=nn.initializers.zeros)
        self.eta_raw = self.param("eta_raw",
                                  nn.initializers.constant(_raw_init(self.eta, self.eta_min, self.eta_max)), ())
        self.gamma_raw = self.param("gamma_raw",
                                    nn.initializers.constant(_raw_init(self.gamma, self.gamma_min, self.gamma_max)), ())

    def _project(self, h):
        B, T, _ = h.shape; H, r, P = self.n_heads, self.rank, self.head_dim
        hg = h.astype(GEMM)
        z = _l2(self.k(hg).reshape(B, T, H, r).astype(LIN))          # L2, causal
        zq = _l2(self.q(hg).reshape(B, T, H, r).astype(LIN))
        v = self.v(hg).reshape(B, T, H, P).astype(LIN)               # value: no L2 (GDN)
        beta = jax.nn.sigmoid(self.beta(hg).astype(LIN))             # (B,T,H) in (0,1)
        zb = beta[..., None] * z                                     # write-gated key
        return z, zb, zq, v, beta

    def _eta(self):    return _squash(self.eta_raw, self.eta_min, self.eta_max)
    def _gamma(self):  return _squash(self.gamma_raw, self.gamma_min, self.gamma_max)

    def _scale(self):
        return self._eta() * self._gamma() ** self.power_K   # eta * gamma^K (Eq.12+Eq.13)

    def __call__(self, x):                                          # chunk-causal training
        z, zb, zq, v, _ = self._project(x)
        Gf, Mf, Cf, qf, shp = chunk_stats(z, zb, zq, v, self.ridge, self.chunk_size)
        if self.use_newton_schulz:                                  # all-MXU operator (exact vs Cholesky)
            Y = ska_core_ns_b(Gf, Mf, Cf, qf, self.power_K, self.ns_iters)
        else:
            Y = ska_core_b(Gf, Mf, Cf, qf, self.power_K)
        B, nc, H, P, CS, T, pad = shp
        Y = (Y * self._scale()).reshape(B, nc, H, P, CS)
        Y = jnp.transpose(Y, (0, 1, 4, 2, 3)).reshape(B, nc * CS, H, P)[:, :T]
        return x + self.o(Y.reshape(B, T, H * P).astype(GEMM)).astype(x.dtype)

    # O(1) recurrent inference (Echo Sec.3.5, prefix mode Eq.4-6)
    # Carries the Cholesky factor L; decode updates it by a rank-1 cholupdate
    # (O(r^2)) instead of refactoring (O(r^3)) -- NeurIPS "It Cancels".
    def prefill(self, x):
        B, T, _ = x.shape; H, r, P = self.n_heads, self.rank, self.head_dim
        z, zb, zq, v, _ = self._project(x)
        zp = jnp.transpose(z, (0, 2, 1, 3)); zbp = jnp.transpose(zb, (0, 2, 1, 3))
        vp = jnp.transpose(v, (0, 2, 1, 3))
        G = jnp.einsum("bhtr,bhts->bhrs", zbp, zp) + self.ridge * jnp.eye(r, dtype=LIN)
        M = jnp.einsum("bhtr,bhts->bhrs", zbp[:, :, 1:], zp[:, :, :-1])
        Cv = jnp.einsum("bhtp,bhtr->bhpr", vp, zbp)
        N = B * H
        L = _chol_b((0.5 * (G + jnp.swapaxes(G, -1, -2))).reshape(N, r, r))
        q = jnp.transpose(zq, (0, 2, 3, 1)).reshape(N, r, T)
        Y = ska_apply_L_b(L, M.reshape(N, r, r), Cv.reshape(N, P, r), q, self.power_K)
        Y = Y.reshape(B, H, P, T).transpose(0, 3, 1, 2)             # (B,T,H,P)
        out = x + self.o((Y * self._scale()).reshape(B, T, H * P).astype(GEMM)).astype(x.dtype)
        return out, dict(L=L.reshape(B, H, r, r), M=M, Cv=Cv, zlast=zp[:, :, -1])

    def step(self, x_t, carry):                                     # one token, all O(r^2): no refactor
        B = x_t.shape[0]; H, r, P = self.n_heads, self.rank, self.head_dim
        z, zb, zq, v, beta = self._project(x_t)                     # (B,1,H,*)
        z1 = z[:, 0]; zb1 = zb[:, 0]; zq1 = zq[:, 0]; v1 = v[:, 0]; b1 = beta[:, 0]
        w = jnp.sqrt(jnp.clip(b1, 1e-8, None))[..., None] * z1      # rank-1 vector: beta z z^T = w w^T
        N = B * H
        L = _cholup_b(carry["L"].reshape(N, r, r), w.reshape(N, r)).reshape(B, H, r, r)
        M = carry["M"] + jnp.einsum("bhr,bhs->bhrs", zb1, carry["zlast"])
        Cv = carry["Cv"] + jnp.einsum("bhp,bhr->bhpr", v1, zb1)
        Y = ska_apply_L_b(L.reshape(N, r, r), M.reshape(N, r, r),
                          Cv.reshape(N, P, r), zq1.reshape(N, r, 1), self.power_K)
        y = Y.reshape(B, H, P)
        out = x_t[:, 0] + self.o((y * self._scale()).reshape(B, H * P).astype(GEMM)).astype(x_t.dtype)
        return out[:, None], dict(L=L, M=M, Cv=Cv, zlast=z1)


def L_to_G(L):
    return L @ jnp.swapaxes(L, -1, -2)


# 4. Mamba-2 SSD backbone  (matmul SSD; bf16 GEMM, fp32 decay)

def _segsum(a):
    T = a.shape[-1]; ac = jnp.cumsum(a, -1)
    return jnp.where(jnp.tril(jnp.ones((T, T), bool)), ac[..., :, None] - ac[..., None, :], -jnp.inf)


def ssd(x, a, B, C, bl, ret_state=False):
    b, L, h, p = x.shape; n = B.shape[-1]; nc = L // bl
    xr = x.reshape(b, nc, bl, h, p).astype(GEMM)
    Br = B.reshape(b, nc, bl, h, n).astype(GEMM); Cr = C.reshape(b, nc, bl, h, n).astype(GEMM)
    ar = jnp.transpose(a.reshape(b, nc, bl, h), (0, 3, 1, 2)).astype(LIN)
    acs = jnp.cumsum(ar, -1)
    Lm = jnp.exp(_segsum(ar)).astype(GEMM)
    Yd = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cr, Br, Lm, xr)
    dec = jnp.exp(acs[..., -1:] - acs).astype(GEMM)
    st = jnp.einsum("bclhn,bhcl,bclhp->bchpn", Br, dec, xr)
    st = jnp.concatenate([jnp.zeros_like(st[:, :1]), st], 1)
    dch = jnp.exp(_segsum(jnp.pad(acs[..., -1], ((0, 0), (0, 0), (1, 0))))).astype(GEMM)
    ns = jnp.einsum("bhzc,bchpn->bzhpn", dch, st)
    Yo = jnp.einsum("bclhn,bchpn,bhcl->bclhp", Cr, ns[:, :-1], jnp.exp(acs).astype(GEMM))
    Y = (Yd + Yo).reshape(b, L, h, p).astype(LIN)
    return (Y, ns[:, -1]) if ret_state else Y


def _dwconv(x, w, b):
    B, L, C = x.shape; K = w.shape[1]
    xt = jnp.transpose(jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0))), (0, 2, 1))
    o = jax.lax.conv_general_dilated(xt, w[:, None, :], (1,), [(0, 0)],
                                     dimension_numbers=("NCH", "OIH", "NCH"),
                                     feature_group_count=C)
    return jnp.transpose(o, (0, 2, 1)) + b


class Mamba2(nn.Module):
    d_model: int; d_state: int = 64; d_conv: int = 4; expand: int = 2
    headdim: int = 64; chunk_size: int = 64

    def setup(self):
        self.di = self.expand * self.d_model; self.H = self.di // self.headdim
        self.N = self.d_state; self.dxbc = self.di + 2 * self.N
        self.norm = nn.LayerNorm()
        self.in_proj = nn.Dense(self.di + self.dxbc + self.H, use_bias=False)
        self.cw = self.param("cw", nn.initializers.normal(0.02), (self.dxbc, self.d_conv))
        self.cb = self.param("cb", nn.initializers.zeros, (self.dxbc,))
        self.A_log = self.param("A_log", lambda k, s: jnp.log(jnp.arange(1, s[0] + 1.0)), (self.H,))
        self.dtb = self.param("dtb", nn.initializers.constant(math.log(math.expm1(0.05))), (self.H,))
        self.D = self.param("D", nn.initializers.ones, (self.H,))
        self.ng = self.param("ng", nn.initializers.ones, (self.di,))
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

    def _gate(self, y, z, res):
        y = y * jax.nn.silu(z)
        y = y * jax.lax.rsqrt(jnp.mean(y**2, -1, keepdims=True) + 1e-5) * self.ng
        return res + self.out_proj(y.astype(GEMM)).astype(res.dtype)

    def _pre(self, x):
        z, xbc, dt = jnp.split(self.in_proj(self.norm(x).astype(GEMM)),
                               [self.di, self.di + self.dxbc], -1)
        return z, xbc, dt

    def __call__(self, x):
        B, T, _ = x.shape
        z, xbc, dtr = self._pre(x)
        xbc = jax.nn.silu(_dwconv(xbc, self.cw, self.cb))
        xs, Bs, Cs = jnp.split(xbc, [self.di, self.di + self.N], -1)
        xs = xs.reshape(B, T, self.H, self.headdim)
        Bs = jnp.broadcast_to(Bs[:, :, None], (B, T, self.H, self.N))
        Cs = jnp.broadcast_to(Cs[:, :, None], (B, T, self.H, self.N))
        A = -jnp.exp(self.A_log); dt = jax.nn.softplus(dtr + self.dtb); a = dt * A
        pad = (-T) % self.chunk_size
        xin = xs * dt[..., None]
        if pad:
            xin = jnp.pad(xin, ((0, 0), (0, pad), (0, 0), (0, 0)))
            Bs = jnp.pad(Bs, ((0, 0), (0, pad), (0, 0), (0, 0)))
            Cs = jnp.pad(Cs, ((0, 0), (0, pad), (0, 0), (0, 0)))
            a = jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
        y = ssd(xin, a, Bs, Cs, self.chunk_size)[:, :T] + xs * self.D[None, None, :, None]
        return self._gate(y.reshape(B, T, self.di), z, x)

    def prefill(self, x):
        B, T, _ = x.shape
        z, xbc, dtr = self._pre(x)
        cbuf = jnp.pad(xbc, ((0, 0), (self.d_conv, 0), (0, 0)))[:, -self.d_conv:]
        xbcc = jax.nn.silu(_dwconv(xbc, self.cw, self.cb))
        xs, Bs, Cs = jnp.split(xbcc, [self.di, self.di + self.N], -1)
        xs = xs.reshape(B, T, self.H, self.headdim)
        Bs = jnp.broadcast_to(Bs[:, :, None], (B, T, self.H, self.N))
        Cs = jnp.broadcast_to(Cs[:, :, None], (B, T, self.H, self.N))
        A = -jnp.exp(self.A_log); dt = jax.nn.softplus(dtr + self.dtb); a = dt * A
        pad = (-T) % self.chunk_size; xin = xs * dt[..., None]
        Bp, Cp, ap, xp = Bs, Cs, a, xin
        if pad:
            xp = jnp.pad(xin, ((0, 0), (0, pad), (0, 0), (0, 0)))
            Bp = jnp.pad(Bs, ((0, 0), (0, pad), (0, 0), (0, 0)))
            Cp = jnp.pad(Cs, ((0, 0), (0, pad), (0, 0), (0, 0)))
            ap = jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
        y, hstate = ssd(xp, ap, Bp, Cp, self.chunk_size, ret_state=True)
        y = (y[:, :T] + xs * self.D[None, None, :, None]).reshape(B, T, self.di)
        return self._gate(y, z, x), (hstate, cbuf)

    def step(self, x_t, carry):
        B = x_t.shape[0]; h, cbuf = carry
        z, xbc, dtr = self._pre(x_t)
        buf = jnp.concatenate([cbuf[:, 1:], xbc], 1)
        conv = jax.nn.silu(jnp.einsum("bkc,ck->bc", buf, self.cw) + self.cb)
        xs, Bs, Cs = jnp.split(conv, [self.di, self.di + self.N], -1)
        xs = xs.reshape(B, self.H, self.headdim)
        Bs = jnp.broadcast_to(Bs[:, None], (B, self.H, self.N))
        Cs = jnp.broadcast_to(Cs[:, None], (B, self.H, self.N))
        A = -jnp.exp(self.A_log); dt = jax.nn.softplus(dtr[:, 0] + self.dtb)
        dA = jnp.exp(dt * A)
        hn = dA[:, :, None, None] * h + (dt[:, :, None] * xs)[..., None] * Bs[:, :, None, :]
        y = jnp.einsum("bhpn,bhn->bhp", hn, Cs) + self.D[None, :, None] * xs
        out = self._gate(y.reshape(B, self.di), z[:, 0], x_t[:, 0])
        return out[:, None], (hn, buf)


# 5. Koopman MLP  (Echo Eq.14; |lambda|<=1; norm-preserving rotation)

class KoopmanMLP(nn.Module):
    d_model: int; expand: float = 8.0 / 3.0

    @nn.compact
    def __call__(self, x):
        dk = ((int(self.d_model * self.expand) + 63) // 64) * 64
        h = nn.LayerNorm()(x).astype(GEMM)
        g = jax.nn.silu(nn.Dense(dk, use_bias=False)(h)).astype(LIN)
        g = g.reshape(*g.shape[:-1], dk // 2, 2)
        g1, g2 = g[..., 0], g[..., 1]
        gamma = self.param("gamma", nn.initializers.ones, (dk // 2,))
        omega = self.param("omega", lambda k, s: 0.1 * jax.random.normal(k, s), (dk // 2,))
        rad = jnp.clip(jnp.sqrt(gamma**2 + omega**2), 1e-8, None)
        s = jnp.clip(rad, None, 1.0) / rad                      # |lambda|<=1
        gamma, omega = gamma * s, omega * s
        z = jnp.stack([gamma * g1 + omega * g2, -omega * g1 + gamma * g2], -1).reshape(*g.shape[:-2], dk)
        return x + nn.Dense(self.d_model, use_bias=False)(z.astype(GEMM)).astype(x.dtype)


# 6. Echo model  (Nemotron-H layout; first/last Mamba; SKA at given indices)

def echo_config(size="180m"):
    if size == "180m":
        return dict(d=768, L=24, ska=(8, 16), r=48, H=12, P=64, vocab=32000)
    if size == "50m":
        return dict(d=448, L=16, ska=(3, 7, 11, 15), r=56, H=7, P=64, vocab=32000)
    raise ValueError(size)


class Echo(nn.Module):
    cfg: dict

    def setup(self):
        c = self.cfg
        self.embed = nn.Embed(c["vocab"], c["d"])
        seq, mlp = [], []
        for i in range(c["L"]):
            if i in c["ska"] and i not in (0, c["L"] - 1):
                seq.append(SKA(c["d"], c["H"], c["r"], c["P"],
                               use_newton_schulz=c.get("use_newton_schulz", False),
                               ns_iters=c.get("ns_iters", 15)))
            else:
                seq.append(Mamba2(c["d"], headdim=c["P"]))
            mlp.append(KoopmanMLP(c["d"]))
        self.seq = seq; self.mlp = mlp
        self.norm_f = nn.LayerNorm()
        self.head = nn.Dense(c["vocab"], use_bias=False)

    def __call__(self, ids):
        h = self.embed(ids)
        for s, m in zip(self.seq, self.mlp):
            h = m(s(h))
        return self.head(self.norm_f(h).astype(GEMM)).astype(LIN)

    def prefill(self, ids):
        h = self.embed(ids); carries = []
        for s, m in zip(self.seq, self.mlp):
            h, cr = s.prefill(h); carries.append(cr); h = m(h)
        return self.head(self.norm_f(h).astype(GEMM)).astype(LIN), carries

    def step(self, tok, carries):
        h = self.embed(tok); new = []
        for s, m, cr in zip(self.seq, self.mlp, carries):
            h, ncr = s.step(h, cr); new.append(ncr); h = m(h)
        return self.head(self.norm_f(h).astype(GEMM)).astype(LIN), new


# 7. TPU: mesh + sharded jitted train step (template)

def make_train_step(model, optimizer, mesh=None):
    def loss_fn(params, ids, labels, mask):
        logits = model.apply(params, ids)
        ll = -(jax.nn.log_softmax(logits.astype(LIN), -1)
               * jax.nn.one_hot(labels, logits.shape[-1], dtype=LIN)).sum(-1)
        return (ll * mask).sum() / jnp.clip(mask.sum(), 1, None)

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, batch):
        l, g = jax.value_and_grad(loss_fn)(params, batch["ids"], batch["labels"], batch["mask"])
        upd, opt_state = optimizer.update(g, opt_state, params)
        return optax_apply(params, upd), opt_state, l
    return step


def optax_apply(params, updates):
    import optax
    return optax.apply_updates(params, updates)


# 8. Correctness tests

@jax.default_matmul_precision('highest')  # match the core's fp32 so the test is fp32-vs-fp32 on TPU
def _ref_core(G, M, Cv, q, K):            # autodiff THROUGH cholesky, solve-based (reference)
    L = jnp.linalg.cholesky(G); a0 = _specW(_whiten_M(L, M))
    X = _solveG(L, q)
    for _ in range(K):
        X = a0 * _solveG(L, M @ X)
    return Cv @ X


def test_backward():
    k = jax.random.PRNGKey(0); r, P, p, K = 16, 8, 4, 2
    a, b, c, d = jax.random.split(k, 4)
    A = jax.random.normal(a, (r, r)); G = A @ A.T + r * jnp.eye(r)
    M = 0.3 * jax.random.normal(b, (r, r)); Cv = jax.random.normal(c, (P, r))
    q = jax.random.normal(d, (r, p))
    gc = jax.grad(lambda *t: ska_core(*t, K).sum())(G, M, Cv, q)
    gr = jax.grad(lambda *t: _ref_core(*t, K).sum())(G, M, Cv, q)
    for nm, x, y in zip("GMCq", gc, gr):
        print(f"  d/{nm}: {float(jnp.linalg.norm(x - y) / (jnp.linalg.norm(y) + 1e-9)):.2e}")


def test_mamba_recurrence():
    k = jax.random.PRNGKey(1); B, T, d, T0 = 2, 192, 128, 128
    m = Mamba2(d_model=d); x = jax.random.normal(k, (B, T, d))
    p = m.init(k, x)
    yf = m.apply(p, x)
    yp, cr = m.apply(p, x[:, :T0], method=Mamba2.prefill)
    ys = []
    for t in range(T0, T):
        yt, cr = m.apply(p, x[:, t:t + 1], cr, method=Mamba2.step); ys.append(yt)
    yd = jnp.concatenate(ys, 1)
    print(f"  prefill: {float(jnp.abs(yp - yf[:, :T0]).max()):.2e}  "
          f"decode: {float(jnp.abs(yd - yf[:, T0:]).max()):.2e}")


def test_ska_causal():
    # perturbing a future token must not change earlier outputs
    k = jax.random.PRNGKey(2); B, T, d, CS = 2, 192, 128, 64
    s = SKA(d_model=d, n_heads=4, rank=16, head_dim=32, chunk_size=CS)
    x = jax.random.normal(k, (B, T, d)); p = s.init(k, x)
    y1 = s.apply(p, x)
    x2 = x.at[:, -1].set(jax.random.normal(jax.random.PRNGKey(99), (B, d)))
    y2 = s.apply(p, x2)
    safe = (T // CS - 1) * CS                  # positions strictly before the last chunk
    drift = float(jnp.abs(y1[:, :safe] - y2[:, :safe]).max())
    print(f"  causal leak (future token -> past outputs): {drift:.2e}  (was nonzero with seq-max)")


def test_ska_decode():
    # rank-1 cholupdate decode must match a fresh Cholesky at each position
    B, T, d, T0 = 1, 40, 64, 32
    s = SKA(d_model=d, n_heads=4, rank=16, head_dim=32)
    x = jax.random.normal(jax.random.PRNGKey(3), (B, T, d)); p = s.init(jax.random.PRNGKey(3), x)
    _, cr = s.apply(p, x[:, :T0], method=SKA.prefill)
    stepo = []
    for t in range(T0, T):
        yt, cr = s.apply(p, x[:, t:t + 1], cr, method=SKA.step); stepo.append(yt)
    stepo = jnp.concatenate(stepo, 1)
    gt = jnp.concatenate([s.apply(p, x[:, :t + 1], method=SKA.prefill)[0][:, t:t + 1]
                          for t in range(T0, T)], 1)
    print(f"  decode (rank-1 L) vs fresh-cholesky prefix: {float(jnp.abs(stepo - gt).max()):.2e}")


def test_ska_newton_schulz():
    # NS core must match the Cholesky core (forward+grad), and the module output
    # must match with use_newton_schulz on/off
    k = jax.random.PRNGKey(7); r, P, p_, K, it = 32, 8, 24, 2, 10
    a, b, c, d = jax.random.split(k, 4)
    A = jax.random.normal(a, (r, r)); G = A @ A.T + (r + 1e-3) * jnp.eye(r)
    M = 0.3 * jax.random.normal(b, (r, r)); Cv = jax.random.normal(c, (P, r)); q = jax.random.normal(d, (r, p_))
    f = float(jnp.abs(ska_core_ns(G, M, Cv, q, K, it) - ska_core(G, M, Cv, q, K)).max())
    gn = jax.grad(lambda *t: ska_core_ns(*t, K, it).sum())(G, M, Cv, q)
    gc = jax.grad(lambda *t: ska_core(*t, K).sum())(G, M, Cv, q)
    g = max(float(jnp.linalg.norm(u - v) / (jnp.linalg.norm(v) + 1e-9)) for u, v in zip(gn, gc))
    B, T, dm = 2, 192, 128
    x = jax.random.normal(jax.random.PRNGKey(8), (B, T, dm))
    s_ch = SKA(d_model=dm, n_heads=4, rank=32, head_dim=32)
    s_ns = SKA(d_model=dm, n_heads=4, rank=32, head_dim=32, use_newton_schulz=True, ns_iters=it)
    par = flax.core.unfreeze(s_ch.init(jax.random.PRNGKey(8), x))
    ok = par["params"]["o"]["kernel"]                              # zero-init -> jitter so SKA path shows
    par["params"]["o"]["kernel"] = ok + 0.1 * jax.random.normal(jax.random.PRNGKey(9), ok.shape)
    mod = float(jnp.abs(s_ch.apply(par, x) - s_ns.apply(par, x)).max())
    print(f"  core fwd vs cholesky: {f:.2e}  max grad rel: {g:.2e}  module(on vs off): {mod:.2e}")


if __name__ == "__main__":
    print("backward (whitened custom_vjp vs autodiff-through-cholesky):"); test_backward()
    print("mamba-2 recurrence (decode vs parallel):"); test_mamba_recurrence()
    print("ska causal normalization:"); test_ska_causal()
    print("ska decode (rank-1 cholupdate):"); test_ska_decode()
    print("ska newton-schulz (all-MXU core vs cholesky):"); test_ska_newton_schulz()
