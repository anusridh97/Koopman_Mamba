"""Oracle for the exact lag-1 induction statistic C+ / R+  (post-Gate-2 stage).

Induction (context has ...A B..., A recurs, predict B) is verbatim lag-1 ridge
regression: B+* = argmin Σ‖v_t − B x_{t-1}‖² + ε‖B‖² = C+ G-^-1, with
C+ = Σ v̄_t x_{t-1}ᵀ. It's the paper's Prop I.2 lookup head relabeled one lag
over. Unlike the K=1 power route (which predicts keys and so needs predictable
keys, oracle C9 in test_gated_blend_oracle.py), C+ works regardless of key
predictability because it never predicts keys -- it associates shifted pairs.

This module pins C+/R+ to the same standard before product code, torch-free:

  I1  shifted-ridge optimality: C+ G-^-1 == an INDEPENDENT augmented-QR ridge
      solve over the shifted pairs. Plus the shared-G̃ slack (ship G̃, don't build
      G-; the 1/T shrinkage is measured and documented, not engineered away).
  I2  value-endpoint boundary combine: the cross-chunk C+ term carries
      √(β_{i,0} β_{i-1,-1}) v_{i,0} z_{i-1,-1}ᵀ (VALUE first-of-chunk vs KEY
      last-of-prev). Teeth: non-constant β + wrong weightings deviate.
  I3  R+ = C+ L^-T transported by the ONE-SIDED Seeger drag-along (the low-risk
      right sweep), rank-1 correction v̄_t w_tᵀ reusing w_t = L^-1 x_{t-1} -- the
      exact solve the A-transport already performs. Matches fresh C+ L^-T.

C+ adds Pr state/head and touches the write path -> its own post-Gate-2 stage,
not a gate ride-along. Runs without torch:
  python3 code-tests/test_induction_cplus_oracle.py
"""
import numpy as np

try:
    import pytest
    pytestmark = pytest.mark.correctness
except ImportError:
    class _Mark:
        def __getattr__(self, _n):
            def deco(*a, **k):
                def wrap(f): return f
                return wrap
            return deco
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()
    pytestmark = None


def rot(a, b, c, s):
    return c * a + s * b, -s * a + c * b


def phase1(L_prev, x):
    """Augmented rank-1 Cholesky update [L | x] -> [L' | 0]; C,S rotations."""
    r = L_prev.shape[0]; L = L_prev.copy(); z = x.copy().astype(float)
    C = np.zeros(r); S = np.zeros(r)
    for k in range(r):
        l = L[k, k]; zk = z[k]; rho = np.sqrt(l*l + zk*zk)
        c = l/rho; s = zk/rho; C[k] = c; S[k] = s
        for i in range(k, r):
            L[i, k], z[i] = rot(L[i, k], z[i], c, s)
    return L, C, S


def phase3(R_prev, C, S):
    """One-sided Seeger drag-along: C_{t-1} L_t^-T = first_r([R_{t-1}|0] Q)."""
    P, r = R_prev.shape
    Rt = np.zeros((P, r + 1)); Rt[:, :r] = R_prev; last = r
    for k in range(r):
        for i in range(P):
            Rt[i, k], Rt[i, last] = rot(Rt[i, k], Rt[i, last], C[k], S[k])
    return Rt[:, :r]


def _sym(T, r, P, eps, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, r)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    v = rng.standard_normal((T, P))
    beta = rng.uniform(0.05, 1.0, T)
    x = np.sqrt(beta)[:, None] * z
    vbar = np.sqrt(beta)[:, None] * v
    return x, vbar, beta, z, v


def test_i1_shifted_ridge_optimality():
    x, vbar, *_ = _sym(600, 24, 16, 1e-3, 1); eps = 1e-3
    Xp = x[:-1]; Y = vbar[1:]                              # shifted pairs (x_{t-1}, v̄_t)
    Cp = Y.T @ Xp                                          # Σ v̄_t x_{t-1}^T
    Gm = eps * np.eye(24) + Xp.T @ Xp                      # G- : Gram of PREVIOUS keys
    B_closed = Cp @ np.linalg.inv(Gm)
    # independent augmented-QR ridge solve (per output dim), not the normal eqn
    Xaug = np.vstack([Xp, np.sqrt(eps) * np.eye(24)])
    B_brute = np.empty_like(B_closed)
    for p in range(16):
        yaug = np.concatenate([Y[:, p], np.zeros(24)])
        B_brute[p], *_ = np.linalg.lstsq(Xaug, yaug, rcond=None)
    assert np.linalg.norm(B_closed - B_brute) / np.linalg.norm(B_brute) < 1e-8
    # shared-G̃ read (ship G̃, not G-): slack shrinks ~1/T, documented not removed.
    # Averaged over seeds -- single-seed slack is noisy; the 1/T trend is in the mean.
    def mean_slack(T):
        vals = []
        for sd in range(6):
            xx, vv, *_ = _sym(T, 24, 16, eps, 100 + sd)
            Xp2, Y2 = xx[:-1], vv[1:]; Cp2 = Y2.T @ Xp2
            Gt = eps*np.eye(24) + xx.T@xx                  # G̃ (shared with the A-path)
            Gm2 = eps*np.eye(24) + Xp2.T@Xp2
            vals.append(np.linalg.norm(Cp2@np.linalg.inv(Gt) - Cp2@np.linalg.inv(Gm2))
                        / np.linalg.norm(Cp2@np.linalg.inv(Gm2)))
        return np.mean(vals)
    s = [mean_slack(T) for T in (200, 800, 3200)]
    assert s[0] > s[1] > s[2]                             # monotone ~1/T shrinkage (in the mean)
    assert s[-1] < 5e-3                                   # negligible at T >> 1


def test_i2_value_endpoint_boundary_combine():
    """C+ chunk combine: ΔC+_{A∘B} = ΔC+_A + ΔC+_B + v̄_{B,0} x_{A,-1}ᵀ, the value
    first-of-B against the key last-of-A, each with its own √β. Non-constant β so
    the geometric mean can't be confused with the wrong weightings."""
    rng = np.random.default_rng(7)
    r, P, T, c = 20, 12, 40, 17
    Z = rng.standard_normal((T, r)); V = rng.standard_normal((T, P))
    beta = rng.uniform(0.1, 1.0, T)
    x = np.sqrt(beta)[:, None] * Z; vbar = np.sqrt(beta)[:, None] * V
    trueCp = sum(np.outer(vbar[t], x[t - 1]) for t in range(1, T))
    dA = sum(np.outer(vbar[t], x[t - 1]) for t in range(1, c))
    dB = sum(np.outer(vbar[t], x[t - 1]) for t in range(c + 1, T))

    def rel(bnd): return np.linalg.norm(dA + dB + bnd - trueCp) / np.linalg.norm(trueCp)
    assert rel(np.outer(vbar[c], x[c - 1])) < 1e-12                    # correct
    # teeth (non-const β so these differ):
    assert rel(np.outer(vbar[c], Z[c - 1])) > 1e-4                     # forgot key √β
    assert rel(np.outer(V[c], x[c - 1])) > 1e-4                        # forgot value √β
    assert rel(beta[c] * np.outer(V[c], Z[c - 1])) > 1e-4              # asymmetric β_c
    assert rel(0.5*(beta[c]+beta[c-1]) * np.outer(V[c], Z[c-1])) > 1e-4  # arithmetic mean


def test_i3_Rplus_dragalong_reuses_w():
    """Stream R+ = C+ L^-T via one-sided drag-along + v̄_t w_tᵀ, w_t=L^-1 x_{t-1}
    (the SAME solve the A-transport does). Match fresh C+ L^-T at machine
    precision; confirm the correction vector is literally that solve."""
    x, vbar, *_ = _sym(300, 24, 16, 1e-3, 3); eps = 1e-3
    r, P = 24, 16
    L = np.sqrt(eps) * np.eye(r)
    Rp = np.zeros((P, r))
    Cp = np.zeros((P, r))
    x_prev = None
    worst = 0.0
    for t in range(x.shape[0]):
        L, C, S = phase1(L, x[t])
        if x_prev is not None:
            w_t = np.linalg.solve(L, x_prev)              # = L^-1 x_{t-1}  (== A-path v_w)
            Rp = phase3(Rp, C, S) + np.outer(vbar[t], w_t)
            Cp = Cp + np.outer(vbar[t], x_prev)
            R_ref = Cp @ np.linalg.inv(L).T               # fresh C+ L^-T
            worst = max(worst, np.linalg.norm(Rp - R_ref) / (np.linalg.norm(R_ref) + 1e-30))
            # the correction reuses the exact A-transport solve:
            assert np.allclose(w_t, np.linalg.solve(L, x_prev))
        x_prev = x[t]
    assert worst < 1e-9, worst


if __name__ == "__main__":
    test_i1_shifted_ridge_optimality(); print("I1 shifted-ridge optimality + shared-G̃ slack: PASS")
    test_i2_value_endpoint_boundary_combine(); print("I2 value-endpoint boundary combine + teeth: PASS")
    test_i3_Rplus_dragalong_reuses_w(); print("I3 R+ drag-along reusing w_t: PASS")
    print("ALL PASS")
