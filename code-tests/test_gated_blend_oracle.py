"""Oracle for the input-dependent (query-gated) spectral blend read -- NumPy only.

    lam_t = sigmoid((w_lam . q_t)/sqrt(r) + b_lam)   # gate on the pre-whitening,
                                                     # L2-normalized query
    q_w   = L^{-1} q_t
    h     = (1-lam_t) q_w + lam_t A^2 q_w
    y_t   = eta * R h

Reference + safety net for the FUTURE step-5 read (memo §5 blend made
input-dependent). No product code implements this yet; this is the spec the
torch read will be tested against, validated torch-free with teeth on every claim.

  C1  Pointwise contractivity ||(1-lam)I + lam A^2||_2 <= 1 for lam in (0,1),
      GIVEN symmetric sqrt(beta) stats. TEETH: a STRUCTURED alternating-beta
      sequence (what a trained write gate produces) breaks the OLD asymmetric
      convention hard (sigma >> 1) while sym stays <=1 -> the gate inherits the
      sqrt(beta) dependency and cannot ship without it. (Random beta rarely
      violates the asym bound; structured beta does -- that is the point.)
  C2  Analytic gate gradient dL/dlam = <g_y, eta R (A^2 - I) q_w>, chained to
      (w_lam, b_lam), vs central finite differences. Plus the aux-loss gradient
      (ships OFF by default).
  C3  w_lam=0 => BITWISE equivalence with the fixed blend at lam=sigmoid(b_lam)
      -> Arm 0 (fixed) vs Arm 1 (gated) differ only in requires_grad.
  C4  Cholesky-free form y = eta[(1-lam) C G^{-1} q + lam C G^{-1} M G^{-1} M
      G^{-1} q] == the L-form -> the backward needs no Cholesky derivative.
  C5  Mechanism-level learnability: with dense per-token supervision where the
      spectral and ridge readouts genuinely disagree by token class, plain GD on
      (w_lam, b_lam) ALONE separates lambda by class. TEETH: identical query
      distributions (non-separable) -> no separation -> the effect is
      signal-driven, not optimizer drift.

C5 answers only the MECHANISM half: given signal, the gate separates. Whether
mixed LM+retrieval training PROVIDES that signal at scale is the 50M Arm-0/Arm-1
question and is NOT answerable here (same necessary-not-sufficient boundary as
the §C retrain gate).

Runs without torch: `python3 code-tests/test_gated_blend_oracle.py`.
"""
import numpy as np

try:
    import pytest
    pytestmark = pytest.mark.correctness
except ImportError:
    class _Mark:
        def __getattr__(self, _n):
            def deco(*a, **k):
                def wrap(f):
                    return f
                return wrap
            return deco
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()
    pytestmark = None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stats_from(Z, V, beta, eps=1e-3, mode="sym"):
    """Cumulative SKA stats. mode='sym' = 1f1301b convention (x=sqrt(beta)z in
    both key slots, vbar=sqrt(beta)v); mode='asym' = pre-1f1301b (M by beta_t)."""
    T, r = Z.shape
    P = V.shape[1]
    G = eps * np.eye(r); M = np.zeros((r, r)); C = np.zeros((P, r))
    if mode == "sym":
        X = np.sqrt(beta)[:, None] * Z
        Vb = np.sqrt(beta)[:, None] * V
        for t in range(T):
            G += np.outer(X[t], X[t]); C += np.outer(Vb[t], X[t])
            if t > 0:
                M += np.outer(X[t], X[t - 1])
    else:
        for t in range(T):
            G += beta[t] * np.outer(Z[t], Z[t]); C += beta[t] * np.outer(V[t], Z[t])
            if t > 0:
                M += beta[t] * np.outer(Z[t], Z[t - 1])
    L = np.linalg.cholesky(G); Li = np.linalg.inv(L)
    return dict(G=G, M=M, C=C, L=L, Li=Li, A=Li @ M @ Li.T, R=C @ Li.T, r=r, P=P)


def build_stats(T, r, P, eps=1e-3, mode="sym", rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    Z = rng.standard_normal((T, r)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    V = rng.standard_normal((T, P))
    beta = rng.uniform(0.05, 1.0, size=T)
    return stats_from(Z, V, beta, eps, mode)


def alt_sequence(T, r, delta, rng):
    """Structured alternating low-beta/high-beta tokens along two noisy
    directions -- what a trained write gate produces (beta~0 distractors,
    ~1 facts). Random beta almost never violates the asym bound; this does."""
    e1 = np.zeros(r); e1[0] = 1.0
    e2 = np.zeros(r); e2[1] = 1.0
    Z = np.empty((T, r)); beta = np.empty(T)
    for t in range(T):
        z = (e1 if t % 2 == 0 else e2) + 0.05 * rng.standard_normal(r)
        Z[t] = z / np.linalg.norm(z)
        beta[t] = delta if t % 2 == 0 else 1.0
    return Z, beta


def read_lam(st, q, lam, eta=1.5):
    qw = st["Li"] @ q
    a2 = st["A"] @ (st["A"] @ qw)
    return eta * (st["R"] @ ((1.0 - lam) * qw + lam * a2))


def gate(st, q, w, b):
    return sigmoid((w @ q) / np.sqrt(st["r"]) + b)


def gated_read(st, q, w, b, eta=1.5):
    lam = gate(st, q, w, b)
    return read_lam(st, q, lam, eta), lam


def test_c1_contractivity_and_sqrt_beta_dependency():
    for s in range(10):
        st = build_stats(400, 24, 16, mode="sym", rng=np.random.default_rng(s))
        assert np.linalg.norm(st["A"], 2) <= 1 + 1e-12
        A2 = st["A"] @ st["A"]
        for lam in np.random.default_rng(100 + s).uniform(0, 1, 8):
            assert np.linalg.norm((1 - lam) * np.eye(24) + lam * A2, 2) <= 1 + 1e-12
    # teeth: SAME structured alt-beta sequence, both conventions
    Z, beta = alt_sequence(400, 24, delta=0.01, rng=np.random.default_rng(99))
    V = np.random.default_rng(100).standard_normal((400, 16))
    s_sym = np.linalg.norm(stats_from(Z, V, beta, mode="sym")["A"], 2)
    st_a = stats_from(Z, V, beta, mode="asym")
    s_asym = np.linalg.norm(st_a["A"], 2)
    assert s_sym <= 1 + 1e-12, f"sym broken on adversarial seq: {s_sym}"
    assert s_asym > 1.0, f"teeth failed: asym sigma {s_asym}"       # ~8.9 in practice
    # the blend itself becomes expansive under asym -> gate inherits sqrt(beta)
    assert np.linalg.norm(0.05 * np.eye(24) + 0.95 * (st_a["A"] @ st_a["A"]), 2) > 1.0


def test_c2_gate_gradient_matches_fd():
    rng = np.random.default_rng(11)
    st = build_stats(300, 24, 16, mode="sym", rng=rng)
    eta = 1.5
    q = rng.standard_normal(24); q /= np.linalg.norm(q)
    w = 0.3 * rng.standard_normal(24); b = 2.0
    ystar = rng.standard_normal(16)
    lam = gate(st, q, w, b)
    g = read_lam(st, q, lam, eta) - ystar
    qw = st["Li"] @ q
    D = eta * (st["R"] @ (st["A"] @ (st["A"] @ qw) - qw))
    ana_lam = g @ D
    h = 1e-6
    fd_lam = (0.5 * np.sum((read_lam(st, q, lam + h, eta) - ystar) ** 2)
              - 0.5 * np.sum((read_lam(st, q, lam - h, eta) - ystar) ** 2)) / (2 * h)
    assert abs(ana_lam - fd_lam) / max(1e-12, abs(fd_lam)) < 1e-6

    def loss(w_, b_):
        y_, _ = gated_read(st, q, w_, b_, eta)
        return 0.5 * np.sum((y_ - ystar) ** 2)
    s = lam * (1 - lam)
    ana_w = ana_lam * s * q / np.sqrt(24); ana_b = ana_lam * s
    for k in [0, 7, 23]:
        wp, wm = w.copy(), w.copy(); wp[k] += h; wm[k] -= h
        fd = (loss(wp, b) - loss(wm, b)) / (2 * h)
        assert abs(ana_w[k] - fd) / max(1e-12, abs(fd)) < 1e-6
    fd_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
    assert abs(ana_b - fd_b) / max(1e-12, abs(fd_b)) < 1e-6
    # aux loss (ships OFF): L_aux = -log lam ; dL/db = -(1-lam)
    fd_aux_b = (-np.log(gate(st, q, w, b + h)) + np.log(gate(st, q, w, b - h))) / (2 * h)
    assert abs(-(1 - lam) - fd_aux_b) / max(1e-12, abs(fd_aux_b)) < 1e-6


def test_c3_init_equivalence_to_fixed_blend():
    rng = np.random.default_rng(21)
    st = build_stats(300, 24, 16, mode="sym", rng=rng)
    b = np.log(0.9 / 0.1)
    for _ in range(16):
        q = rng.standard_normal(24); q /= np.linalg.norm(q)
        y_gate, lam = gated_read(st, q, np.zeros(24), b)
        assert abs(lam - sigmoid(b)) == 0.0
        assert np.max(np.abs(y_gate - read_lam(st, q, sigmoid(b)))) == 0.0


def test_c4_cholesky_free_form():
    rng = np.random.default_rng(31)
    st = build_stats(300, 24, 16, mode="sym", rng=rng)
    eta = 1.5
    Ginv = np.linalg.inv(st["G"])
    for _ in range(8):
        q = rng.standard_normal(24); q /= np.linalg.norm(q)
        lam = rng.uniform(0, 1)
        yL = read_lam(st, q, lam, eta)
        t = Ginv @ q
        yG = eta * ((1 - lam) * (st["C"] @ t)
                    + lam * (st["C"] @ (Ginv @ (st["M"] @ (Ginv @ (st["M"] @ t))))))
        assert np.linalg.norm(yL - yG) / np.linalg.norm(yG) < 1e-10


def _c5(separable, seed=3, steps=6000):
    rng = np.random.default_rng(seed)
    r, P, T, n = 16, 16, 256, 512
    st = build_stats(T, r, P, mode="sym", rng=rng)
    eta = 1.5
    A2 = st["A"] @ st["A"]
    mu = rng.standard_normal(r); mu /= np.linalg.norm(mu)

    def mk(sign):
        base = sign * mu[None, :] if separable else 0.0
        Q = base + 0.5 * rng.standard_normal((n, r))
        return Q / np.linalg.norm(Q, axis=1, keepdims=True)
    qs = np.vstack([mk(+1), mk(-1)])
    cls = np.concatenate([np.ones(n), np.zeros(n)])
    QW = qs @ st["Li"].T
    Yr = eta * (QW @ st["R"].T)
    Ys = eta * ((QW @ A2.T) @ st["R"].T)
    D = Ys - Yr
    Ystar = np.where(cls[:, None] == 1, Ys, Yr)
    d2 = np.mean(np.sum(D * D, axis=1))
    lr_b, lr_w = 2.0 / d2, 2.0 / d2 * r
    w = np.zeros(r); b = np.log(0.9 / 0.1); loss0 = None
    for _ in range(steps):
        lam = sigmoid((qs @ w) / np.sqrt(r) + b)
        g = (Yr + lam[:, None] * D) - Ystar
        loss = 0.5 * np.mean(np.sum(g * g, axis=1))
        loss0 = loss if loss0 is None else loss0
        sg = np.sum(g * D, axis=1) * lam * (1 - lam)
        w -= lr_w * (qs.T @ sg) / (len(sg) * np.sqrt(r))
        b -= lr_b * np.mean(sg)
    lam = sigmoid((qs @ w) / np.sqrt(r) + b)
    gap = lam[cls == 1].mean() - lam[cls == 0].mean()
    return gap, loss / loss0


def test_c5_learnability_given_separable_signal():
    gap, lr_ratio = _c5(separable=True)
    assert gap > 0.3 and lr_ratio < 0.2       # gate separates lambda by class


def test_c5_teeth_no_separation_without_signal():
    gap, _ = _c5(separable=False)
    assert abs(gap) < 0.1                      # signal-driven, not optimizer drift


# ===========================================================================
# 3-way basis {I, A, A^2} -- folding in K=1 (the INDUCTION read). The 2-way blend
# above skips K=1; A(A q_w) is already the A^2 intermediate, so the induction
# branch is compute-free. Pinned convention (ECHO_V1_1_PLAN.md §2):
#   3 logits l_k = (w_k . z_q)/sqrt(r) + b_k, k in {I, A, A^2}, softmax over them;
#   all gate rows + all 3 biases off the weight-decay list.
#   Init w_k=0, b = ln(0.05, 0.05, 0.9): A^2 dominant (proven-retrieval regime),
#   A-involving mass 0.95 preserves the bootstrap, both minority branches get an
#   equal trainable foothold (softmax damping ~0.0475 each, vs the scalar gate's
#   0.09 at 0.9). Do NOT init p_A ~ 0.01 -- the logit gradient scales with p_k.
# Convexity (softmax weights nonneg, sum 1) preserves contractivity pointwise.
# ===========================================================================

INIT_LOGITS = np.log(np.array([0.05, 0.05, 0.9]))       # pinned 3-way init


def gate3(st, q, W, b):
    """3-way softmax gate. W:(3,r), b:(3,) -> convex weights (3,) over {I,A,A^2}."""
    logits = (W @ q) / np.sqrt(st["r"]) + b
    ex = np.exp(logits - logits.max())
    return ex / ex.sum()


def read3(st, q, w3, eta=1.5):
    qw = st["Li"] @ q
    a1 = st["A"] @ qw
    a2 = st["A"] @ a1
    h = w3[0] * qw + w3[1] * a1 + w3[2] * a2
    return eta * (st["R"] @ h)


def test_c6_threeway_contractivity():
    st = build_stats(400, 24, 16, mode="sym", rng=np.random.default_rng(6))
    I = np.eye(24); A = st["A"]; A2 = A @ A
    rng = np.random.default_rng(60)
    for _ in range(2000):                       # softmax weights => convex => <=1
        lg = rng.standard_normal(3); w = np.exp(lg - lg.max()); w /= w.sum()
        assert np.linalg.norm(w[0]*I + w[1]*A + w[2]*A2, 2) <= 1 + 1e-12
    worst = 0.0                                 # teeth: leave the convex hull
    for _ in range(4000):
        w = rng.standard_normal(3); w /= w.sum()   # sums to 1 but not nonneg
        worst = max(worst, np.linalg.norm(w[0]*I + w[1]*A + w[2]*A2, 2))
    assert worst > 1.0                          # convexity is load-bearing


def test_c7_threeway_softmax_gradient_matches_fd():
    """Correct softmax gradient vs FD on EVERY logit jointly, plus the teeth:
    the classic bug treats the 3 gates as independent sigmoids (diagonal
    p_k(1-p_k) only, dropping the -p_k Σ_j p_j d_j coupling) -- it passes
    single-logit spot checks but must DIVERGE on the joint FD."""
    rng = np.random.default_rng(70)
    st = build_stats(300, 24, 16, mode="sym", rng=rng); eta = 1.5
    q = rng.standard_normal(24); q /= np.linalg.norm(q)
    W = 0.3 * rng.standard_normal((3, 24)); b = rng.standard_normal(3)
    ystar = rng.standard_normal(16)
    w3 = gate3(st, q, W, b)
    g = read3(st, q, w3, eta) - ystar
    qw = st["Li"] @ q; a1 = st["A"] @ qw; a2 = st["A"] @ a1
    dLdw = np.array([g @ (eta * (st["R"] @ bk)) for bk in (qw, a1, a2)])
    ana_b = (np.diag(w3) - np.outer(w3, w3)) @ dLdw       # softmax Jacobian
    ana_b_bug = w3 * (1.0 - w3) * dLdw                    # independent-sigmoid (WRONG)
    ana_W = np.outer(ana_b, q / np.sqrt(24))
    h = 1e-6

    def loss(W_, b_):
        return 0.5 * np.sum((read3(st, q, gate3(st, q, W_, b_), eta) - ystar) ** 2)
    fd_b = np.empty(3)
    for k in range(3):
        bp, bm = b.copy(), b.copy(); bp[k] += h; bm[k] -= h
        fd_b[k] = (loss(W, bp) - loss(W, bm)) / (2 * h)
        assert abs(ana_b[k] - fd_b[k]) / max(1e-12, abs(fd_b[k])) < 1e-5
    for (i, j) in [(0, 0), (1, 7), (2, 23)]:
        Wp, Wm = W.copy(), W.copy(); Wp[i, j] += h; Wm[i, j] -= h
        fd = (loss(Wp, b) - loss(Wm, b)) / (2 * h)
        assert abs(ana_W[i, j] - fd) / max(1e-12, abs(fd)) < 1e-5
    # teeth: the independent-sigmoid formula diverges on the joint check
    assert np.max(np.abs(ana_b_bug - fd_b) / (np.abs(fd_b) + 1e-12)) > 1e-2


def test_c8_init_collapse_and_k1_expressivity():
    """(i) C3' invariant: w=0 => the gate is bitwise the fixed init mixture
    (0.05,0.05,0.9) for ANY query -> Arm 0 (biases trainable, w frozen) ≡ Arm 1
    (both trainable) at step 0. (ii) p_A -> 0 recovers the validated 2-way blend
    EXACTLY (ties the 3-way oracle to the 2-way one). (iii) with the A-branch on,
    the read is NOT in any {I,A^2} blend -> K=1 adds a genuinely new direction."""
    rng = np.random.default_rng(80)
    st = build_stats(300, 24, 16, mode="sym", rng=rng)
    # (i) w=0 -> exactly softmax(INIT_LOGITS), independent of the query
    for _ in range(8):
        q = rng.standard_normal(24); q /= np.linalg.norm(q)
        w3 = gate3(st, q, np.zeros((3, 24)), INIT_LOGITS)
        assert np.allclose(w3, np.array([0.05, 0.05, 0.9]), atol=1e-12)
    # (ii) p_A -> 0 collapses to the 2-way blend at lam
    for _ in range(8):
        q = rng.standard_normal(24); q /= np.linalg.norm(q)
        lam = rng.uniform(0.05, 0.95)
        b = np.array([0.0, -1e9, np.log(lam / (1 - lam))])   # w -> [1-lam, ~0, lam]
        y3 = read3(st, q, gate3(st, q, np.zeros((3, 24)), b))
        assert np.max(np.abs(y3 - read_lam(st, q, lam))) < 1e-9
    # (iii) K=1 expressivity: A q_w generically outside span{q_w, A^2 q_w}
    q = rng.standard_normal(24); q /= np.linalg.norm(q)
    qw = st["Li"] @ q; a1 = st["A"] @ qw; a2 = st["A"] @ a1
    B = np.stack([qw, a2], axis=1)
    coef, *_ = np.linalg.lstsq(B, a1, rcond=None)
    assert np.linalg.norm(a1 - B @ coef) / np.linalg.norm(a1) > 1e-3


def _orth_stream(iid, sigma, seed):
    """Keys z_{t+1}=normalize(Q z_t + sigma n), Q orthogonal (QR of Gaussian) --
    orthogonal dynamics commute with L2-normalization, so keys stay linearly
    predictable AFTER normalization (rotary-style; paper Appendix B is SO(2) on
    keys). values v_t = W_true z_{t-1} (induction target). Returns
    (err_K1, err_Cp, err_MGiQ) where err_MGiQ = ||M G^-1 - Q|| tests that A LEARNS
    THE ADVANCE ROTATION -- the mechanism, not just the endpoint."""
    rng = np.random.default_rng(seed); r, P, T, eps = 24, 24, 4000, 1e-3
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    Wt = 0.7 * rng.standard_normal((P, r))
    Z = np.empty((T, r)); Z[0] = rng.standard_normal(r); Z[0] /= np.linalg.norm(Z[0])
    for t in range(1, T):
        z = rng.standard_normal(r) if iid else Q @ Z[t - 1] + sigma * rng.standard_normal(r)
        Z[t] = z / np.linalg.norm(z)
    V = np.vstack([np.zeros((1, P)), Z[:-1] @ Wt.T])
    st = stats_from(Z, V, np.ones(T), eps)
    Gi = np.linalg.inv(st["G"])
    Cp = V[1:].T @ (np.sqrt(np.ones(T))[1:, None] * Z[:-1])   # (beta=1) exact lag-1
    e = lambda B: np.linalg.norm(B - Wt) / np.linalg.norm(Wt)
    err_K1 = e(st["C"] @ Gi @ st["M"] @ Gi)
    err_Cp = e(Cp @ Gi)
    err_MGiQ = np.linalg.norm(st["M"] @ Gi - Q) / np.linalg.norm(Q)
    return err_K1, err_Cp, err_MGiQ


def test_c9_orthogonal_dynamics_advance():
    """Closes the caveat: under orthogonal (predictable) key dynamics K=1 IS a
    position-advance -- it recovers the induction map AND M G^-1 matches Q itself.
    Under iid keys K=1 fails (~1.0) while the exact lag-1 statistic C+ works in
    every regime. The M G^-1 ≈ Q check is teeth-grade: it tests the mechanism."""
    k0, c0, mq0 = _orth_stream(iid=False, sigma=0.0, seed=1)
    k2, c2, mq2 = _orth_stream(iid=False, sigma=0.02, seed=1)
    ki, ci, mqi = _orth_stream(iid=True, sigma=0.0, seed=1)
    assert k0 < 0.05 and k2 < 0.05 and ki > 0.5           # K=1: orth works, iid fails
    assert mq0 < 0.05 and mq2 < 0.05 and mqi > 0.5        # A learns the advance rotation
    assert c0 < 0.05 and c2 < 0.05 and ci < 0.05          # C+ exact in every regime


def _route3(separable, seed=3, steps=4000):
    rng = np.random.default_rng(seed); r, P, T, n = 16, 16, 256, 300
    Z = rng.standard_normal((T, r)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    V = rng.standard_normal((T, P)); beta = rng.uniform(0.05, 1.0, T)
    st = stats_from(Z, V, beta, mode="sym"); A = st["A"]; A2 = A @ A
    mus = np.linalg.qr(rng.standard_normal((r, 3)))[0].T
    def mk(c):
        base = mus[c][None, :] if separable else 0.0
        Qy = base + 0.5 * rng.standard_normal((n, r))
        return Qy / np.linalg.norm(Qy, axis=1, keepdims=True)
    qs = np.vstack([mk(0), mk(1), mk(2)]); cls = np.repeat([0, 1, 2], n)
    QW = qs @ st["Li"].T
    reads = np.stack([1.5*(QW@st["R"].T), 1.5*((QW@A.T)@st["R"].T),
                      1.5*((QW@A2.T)@st["R"].T)], axis=1)      # (3n,3,P) per-branch reads
    Ystar = reads[np.arange(3*n), cls]                         # target = the class's branch
    W = np.zeros((3, r)); b = INIT_LOGITS.copy(); lr = 0.5
    for _ in range(steps):
        lg = (qs @ W.T) / np.sqrt(r) + b
        ex = np.exp(lg - lg.max(1, keepdims=True)); wts = ex / ex.sum(1, keepdims=True)
        g = np.einsum('nk,nkp->np', wts, reads) - Ystar
        d = np.einsum('np,nkp->nk', g, reads)
        dlog = wts * (d - (wts * d).sum(1, keepdims=True))     # softmax Jacobian (coupled)
        b -= lr * dlog.mean(0); W -= lr * (dlog.T @ qs) / (3 * n * np.sqrt(r))
    lg = (qs @ W.T) / np.sqrt(r) + b
    ex = np.exp(lg - lg.max(1, keepdims=True)); wts = ex / ex.sum(1, keepdims=True)
    masses = np.stack([wts[cls == c].mean(0) for c in range(3)])   # (class, branch)
    return masses, (masses.max(0) - masses.min(0)).max()


def test_c10_threeclass_routing():
    """C5': three token classes with targets = ridge/induction/spectral reads.
    Tests the gate can route to K=1 SPECIFICALLY (the point of adding it): each
    class's mass must peak on its own branch, incl. the induction class pulling
    mass onto A off the 0.9 A^2 prior. TEETH: identical query distributions ->
    no routing (recalibrated separation floor; the scalar 0.02 does NOT carry)."""
    masses, spread = _route3(separable=True)
    assert spread > 0.3
    for c in range(3):
        assert int(np.argmax(masses[c])) == c, (c, masses[c])
    _, spread0 = _route3(separable=False)
    assert spread0 < 0.06            # recalibrated 3-way non-separable floor (~0.02)


if __name__ == "__main__":
    test_c1_contractivity_and_sqrt_beta_dependency(); print("C1 contractivity + sqrt-beta dependency: PASS")
    test_c2_gate_gradient_matches_fd(); print("C2 gate gradient vs FD: PASS")
    test_c3_init_equivalence_to_fixed_blend(); print("C3 w=0 bitwise == fixed blend: PASS")
    test_c4_cholesky_free_form(); print("C4 Cholesky-free form: PASS")
    test_c5_learnability_given_separable_signal(); print("C5 learnability (separable): PASS")
    test_c5_teeth_no_separation_without_signal(); print("C5 teeth (non-separable): PASS")
    test_c6_threeway_contractivity(); print("C6 3-way {I,A,A^2} contractivity + teeth: PASS")
    test_c7_threeway_softmax_gradient_matches_fd(); print("C7 3-way softmax gradient + indep-sigmoid teeth: PASS")
    test_c8_init_collapse_and_k1_expressivity(); print("C8 init mixture + 2-way collapse + K=1 expressivity: PASS")
    test_c9_orthogonal_dynamics_advance(); print("C9 orthogonal-dynamics advance (A learns Q; C+ always): PASS")
    test_c10_threeclass_routing(); print("C10 3-class routing to K=1 + recalibrated floor: PASS")
    print("ALL PASS")
