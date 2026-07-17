"""Incremental (L, A, R) Givens-transport parity -- NumPy only (no torch).

Reference oracle + safety net for the O(r^2)/token SKA decode recurrence
(sequencing step 2). Streams rank-1 SKA updates and checks, at every step, the
three Frobenius residuals against a fresh factorization -- the paper's own
Table-1 validation -- plus localizers:

  rG     = ||L L^T - G|| / ||G||                         Cholesky update
  rA     = ||A - L^{-1} M L^{-T}|| / ||...||             two-sided A transport (novel)
  rR     = ||R - C L^{-T}|| / ||...||                    one-sided R drag-along
  rAsand = ||A_core - T A_{t-1} T^T||, T = L_t^{-1}L_{t-1}   replay-vs-identity localizer
  rY     = read parity y = eta R A^K L^{-1} q

A sign/order slip in the two A sweeps yields a wrong-but-still-contractive A_t
that no ||A||<=1 assert catches; rA / rAsand are the catch. Also checks the
Eq.9 chunk-boundary combine (test_chunk_boundary_combine) -- the latent
boundary-bug detector.

Runs WITHOUT the torch wheel: `python3 code-tests/test_incremental_lar_parity.py`
(the shared conftest imports torch, so direct execution bypasses pytest here).
float64 -> residuals ~1e-13. The single kernel `rot`, called four ways, is the
whole algorithm; do NOT transpose it for the left sweep.
"""
import numpy as np

try:
    import pytest
    pytestmark = pytest.mark.correctness
except ImportError:                                    # torch-less direct run
    class _Mark:
        def __getattr__(self, _name):
            def deco(*a, **k):
                def wrap(f):
                    return f
                return wrap
            return deco
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()
    pytestmark = None


def rot(a, b, c, s):
    """The one kernel: (c*a + s*b, -s*a + c*b). Used four ways, never transposed."""
    return c * a + s * b, -s * a + c * b


def phase1_chol_update(L_prev, x):
    """Augmented rank-1 Cholesky update [L_{t-1} | x] -> [L_t | 0]; returns
    L_t, stored (C, S) rotations, and u = L_t^{-1} x recovered for free O(r)."""
    r = L_prev.shape[0]
    L = L_prev.copy()
    z = x.astype(np.float64).copy()
    C = np.zeros(r); S = np.zeros(r); u = np.zeros(r)
    e = 1.0
    for k in range(r):
        l = L[k, k]; zk = z[k]
        rho = np.sqrt(l * l + zk * zk)
        c = l / rho; s = zk / rho
        C[k] = c; S[k] = s
        for i in range(k, r):
            L[i, k], z[i] = rot(L[i, k], z[i], c, s)
        u[k] = s * e
        e = c * e
    return L, C, S, u


def phase2_transport_A(A_prev, C, S):
    """A_core = TL_r(Q^T [A_{t-1};0] Q): left sweep (rows) then right (cols), k up."""
    r = A_prev.shape[0]
    At = np.zeros((r + 1, r + 1)); At[:r, :r] = A_prev
    last = r
    for k in range(r):
        for j in range(r + 1):
            At[k, j], At[last, j] = rot(At[k, j], At[last, j], C[k], S[k])
    for k in range(r):
        for i in range(r + 1):
            At[i, k], At[i, last] = rot(At[i, k], At[i, last], C[k], S[k])
    return At[:r, :r].copy()


def phase3_transport_R(R_prev, C, S):
    """R_minus = C_{t-1} L_t^{-T} = first_r([R_{t-1} | 0] Q): right sweep only."""
    P, r = R_prev.shape
    Rt = np.zeros((P, r + 1)); Rt[:, :r] = R_prev
    last = r
    for k in range(r):
        for i in range(P):
            Rt[i, k], Rt[i, last] = rot(Rt[i, k], Rt[i, last], C[k], S[k])
    return Rt[:, :r].copy()


def _stream_and_check(r=64, P=8, T=200, K=2, eps=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    L = np.sqrt(eps) * np.eye(r); A = np.zeros((r, r)); R = np.zeros((P, r))
    x_prev = np.zeros(r)
    G = eps * np.eye(r); M = np.zeros((r, r)); Cm = np.zeros((P, r))
    w = {k: 0.0 for k in ("rG", "rA", "rR", "rAsand", "sA")}
    for t in range(1, T + 1):
        k_t = rng.standard_normal(r); k_t /= np.linalg.norm(k_t) + 1e-12
        a_t = rng.uniform(0.0, 1.0)
        x_t = np.sqrt(a_t) * k_t
        vbar_t = np.sqrt(a_t) * rng.standard_normal(P)

        L_prev, A_prev = L.copy(), A.copy()
        L, C, S, u = phase1_chol_update(L_prev, x_t)
        v_w = np.linalg.solve(L, x_prev) if t > 1 else np.zeros(r)
        A_core = phase2_transport_A(A_prev, C, S)
        A = A_core + np.outer(u, v_w)                     # M += x_t x_{t-1}^T
        R = phase3_transport_R(R, C, S) + np.outer(vbar_t, u)  # C += vbar_t x_t^T

        G = G + np.outer(x_t, x_t)
        if t > 1:
            M = M + np.outer(x_t, x_prev)
        Cm = Cm + np.outer(vbar_t, x_t)

        Lr = np.linalg.cholesky(G)
        Aref = np.linalg.solve(Lr, np.linalg.solve(Lr, M.T).T)
        Rref = np.linalg.solve(Lr, Cm.T).T
        Tt = np.linalg.solve(L, L_prev)
        w["rG"] = max(w["rG"], np.linalg.norm(L @ L.T - G) / np.linalg.norm(G))
        w["rA"] = max(w["rA"], np.linalg.norm(A - Aref) / (np.linalg.norm(Aref) + 1e-30))
        w["rR"] = max(w["rR"], np.linalg.norm(R - Rref) / (np.linalg.norm(Rref) + 1e-30))
        w["rAsand"] = max(w["rAsand"],
                          np.linalg.norm(A_core - Tt @ A_prev @ Tt.T) /
                          (np.linalg.norm(Tt @ A_prev @ Tt.T) + 1e-30))
        w["sA"] = max(w["sA"], np.linalg.norm(A, 2))
        x_prev = x_t

    q = rng.standard_normal(r)
    h = np.linalg.solve(L, q)
    for _ in range(K):
        h = A @ h
    y_inc = 2.5 * (R @ h)
    Lr = np.linalg.cholesky(G)
    hr = np.linalg.solve(Lr, q)
    Aref = np.linalg.solve(Lr, np.linalg.solve(Lr, M.T).T)
    Rref = np.linalg.solve(Lr, Cm.T).T
    for _ in range(K):
        hr = Aref @ hr
    y_ref = 2.5 * (Rref @ hr)
    w["rY"] = np.linalg.norm(y_inc - y_ref) / (np.linalg.norm(y_ref) + 1e-30)
    return w


@pytest.mark.parametrize("r", [16, 64, 128])
def test_incremental_lar_matches_fresh_factorization(r):
    w = _stream_and_check(r=r, P=8, T=200, K=2, seed=r)
    assert w["rG"] < 1e-9, w
    assert w["rA"] < 1e-9, w          # two-sided A transport oracle
    assert w["rAsand"] < 1e-9, w      # replay == explicit T-sandwich
    assert w["rR"] < 1e-9, w
    assert w["rY"] < 1e-9, w
    assert w["sA"] <= 1.0 + 1e-6, w   # sqrt(a) symmetrization => contractive


def test_chunk_boundary_combine():
    """Eq.9 boundary term: splitting a sequence into two chunks and combining
    dM_{A.B} = dM_A + dM_B + f_B l_A^T must reconstruct the true cumulative
    lag-one M. Guards the latent boundary bug the sigma<=1 assert also trips on."""
    rng = np.random.default_rng(7)
    r, T, c = 24, 40, 17                                # split at token c
    X = rng.standard_normal((T, r))
    true_M = sum(np.outer(X[t], X[t - 1]) for t in range(1, T))
    dM_A = sum(np.outer(X[t], X[t - 1]) for t in range(1, c))
    dM_B = sum(np.outer(X[t], X[t - 1]) for t in range(c + 1, T))
    boundary = np.outer(X[c], X[c - 1])                 # f_B (l_A)^T
    combined = dM_A + dM_B + boundary
    assert np.linalg.norm(combined - true_M) < 1e-10


def test_chunk_boundary_combine_sqrt_beta():
    """Acceptance-spec (B): the weighted boundary term under sqrt(beta) carries
    PER-ENDPOINT weights -- sqrt(beta_{i,0} * beta_{i-1,-1}) * z_{i,0} z_{i-1,-1}^T
    -- NOT a single per-token beta. Built symmetrically as x_t = sqrt(beta_t) z_t,
    the boundary is just x_c x_{c-1}^T.

    beta MUST vary per token, else the geometric mean sqrt(b_c b_{c-1}) coincides
    with the arithmetic/asymmetric forms and the test is toothless. We assert the
    correct combine at machine precision AND that each wrong weighting deviates,
    so a symmetrization regression at the boundary is caught (not just a smaller
    residual that a loose tolerance would swallow)."""
    rng = np.random.default_rng(7)
    r, T, c = 24, 40, 17
    Z = rng.standard_normal((T, r))
    beta = rng.uniform(0.1, 1.0, size=T)               # per-token varying
    x = np.sqrt(beta)[:, None] * Z                      # symmetric weighted key
    true_M = sum(np.outer(x[t], x[t - 1]) for t in range(1, T))
    dM_A = sum(np.outer(x[t], x[t - 1]) for t in range(1, c))
    dM_B = sum(np.outer(x[t], x[t - 1]) for t in range(c + 1, T))

    def rel(boundary):
        return np.linalg.norm(dM_A + dM_B + boundary - true_M) / np.linalg.norm(true_M)

    # correct: both endpoints carry their own sqrt(beta)
    assert rel(np.outer(x[c], x[c - 1])) < 1e-12
    # teeth: each wrong weighting must deviate well above tolerance
    assert rel(np.outer(x[c], Z[c - 1])) > 1e-4         # forgot prev-endpoint weight
    assert rel(np.outer(Z[c], x[c - 1])) > 1e-4         # forgot curr-endpoint weight
    assert rel(beta[c] * np.outer(Z[c], Z[c - 1])) > 1e-4               # asymmetric beta_c
    assert rel(0.5 * (beta[c] + beta[c - 1]) * np.outer(Z[c], Z[c - 1])) > 1e-4  # arithmetic mean


def test_decode_w1_trap_L_divergence():
    """Cross-path L-parity that would have caught the decode `w1` trap.

    Decode carries L via rank-1 Cholesky updates. The update vector MUST be the
    symmetric key x = sqrt(beta) z, so w wᵀ = beta zzᵀ matches G = eps I + sum
    beta zzᵀ. The trap: re-deriving beta from a norm -- beta_hat = ||x|| = sqrt(beta)
    (z unit), then w = sqrt(beta_hat) z = beta^{1/4} z -- adds sqrt(beta) zzᵀ, so
    L stops being the factor of the G the rest of the pipeline uses. Decode stays
    internally self-consistent, so only THIS cross-path check (decode L vs fresh
    chol(G)) catches it -- not a single-recurrence residual.

    Non-constant beta so the geometric/arithmetic-mean forms cannot coincide."""
    rng = np.random.default_rng(3)
    r, T, eps = 32, 60, 1e-3
    L_ok = np.sqrt(eps) * np.eye(r)
    L_bug = np.sqrt(eps) * np.eye(r)
    G = eps * np.eye(r)
    for _ in range(T):
        z = rng.standard_normal(r); z /= np.linalg.norm(z)      # unit key
        beta = rng.uniform(0.1, 1.0)
        x = np.sqrt(beta) * z                                    # symmetric key
        G = G + np.outer(x, x)
        L_ok, *_ = phase1_chol_update(L_ok, x)                   # correct: w = x
        beta_hat = np.linalg.norm(x)                             # = sqrt(beta)  (trap)
        w_bug = np.sqrt(beta_hat) * z                            # = beta^{1/4} z
        L_bug, *_ = phase1_chol_update(L_bug, w_bug)
    # correct decode L is the factor of the pipeline's G, at machine precision
    assert np.linalg.norm(L_ok @ L_ok.T - G) / np.linalg.norm(G) < 1e-10
    # the trap desyncs L from G -- the check has teeth
    assert np.linalg.norm(L_bug @ L_bug.T - G) / np.linalg.norm(G) > 1e-2


def test_alpha_identity_under_sqrt_beta():
    """Acceptance-spec (D): under the √β-symmetric cumulative stats, σ(A) ≤ 1,
    so the spectral-norm scale α = 1/max(σ,1) is EXACTLY 1 -- the 20-iter power
    iteration applies a scale of 1, i.e. it is a no-op. This is the evidence that
    removing it (step 2) is exactly equivalent, established ON the symmetrized
    state with the clamp still notionally live.

    Teeth: under the OLD asymmetric-β stats (M weighted by βₜ alone), σ(A) can
    exceed 1, so α < 1 occurs -- the clamp WAS load-bearing there. Same streamed
    data, same own-weight G; only M's cross-term differs. If the asymmetric α
    never dropped below 1, this test would be vacuous."""
    rng = np.random.default_rng(11)
    r, T, eps = 48, 150, 1e-3
    G = eps * np.eye(r)
    M_sym = np.zeros((r, r)); M_asym = np.zeros((r, r))
    x_prev = z_prev = None
    alpha_sym_min = 1.0; alpha_asym_min = 1.0

    def alpha(L, M):
        A = np.linalg.solve(L, np.linalg.solve(L, M.T).T)      # L^{-1} M L^{-T}
        return 1.0 / max(np.linalg.norm(A, 2), 1.0)

    for _ in range(T):
        z = rng.standard_normal(r); z /= np.linalg.norm(z)
        beta = rng.uniform(0.1, 1.0)
        x = np.sqrt(beta) * z
        G = G + np.outer(x, x)                                  # own-weight β (both)
        if x_prev is not None:
            M_sym = M_sym + np.outer(x, x_prev)                 # √(βₜβₜ₋₁)  (symmetric)
            M_asym = M_asym + beta * np.outer(z, z_prev)        # βₜ         (asymmetric)
        L = np.linalg.cholesky(G)
        alpha_sym_min = min(alpha_sym_min, alpha(L, M_sym))
        alpha_asym_min = min(alpha_asym_min, alpha(L, M_asym))
        x_prev = x; z_prev = z

    assert alpha_sym_min == 1.0, alpha_sym_min      # clamp is EXACTLY a no-op under √β
    assert alpha_asym_min < 1.0, alpha_asym_min     # clamp was load-bearing under asymmetric β


if __name__ == "__main__":
    all_ok = True
    for r in (16, 64, 128):
        w = _stream_and_check(r=r, P=8, T=200, K=2, seed=r)
        ok = (w["rG"] < 1e-9 and w["rA"] < 1e-9 and w["rAsand"] < 1e-9
              and w["rR"] < 1e-9 and w["rY"] < 1e-9 and w["sA"] <= 1 + 1e-6)
        all_ok = all_ok and ok
        print(f"r={r:>3}: rG={w['rG']:.2e} rA={w['rA']:.2e} rR={w['rR']:.2e} "
              f"rAsand={w['rAsand']:.2e} rY={w['rY']:.2e} maxσ(A)={w['sA']:.4f} "
              f"-> {'PASS' if ok else 'FAIL'}")
    # boundary combine
    test_chunk_boundary_combine()
    print("chunk-boundary combine (unweighted): PASS")
    test_chunk_boundary_combine_sqrt_beta()
    print("chunk-boundary combine (sqrt-beta, per-endpoint): PASS")
    test_decode_w1_trap_L_divergence()
    print("decode w1-trap L-divergence (cross-path): PASS")
    test_alpha_identity_under_sqrt_beta()
    print("alpha identity under sqrt-beta (§D, power-iter is a no-op): PASS")
    print("ALL PASS" if all_ok else "SOME FAILED")
