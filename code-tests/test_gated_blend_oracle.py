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


if __name__ == "__main__":
    test_c1_contractivity_and_sqrt_beta_dependency(); print("C1 contractivity + sqrt-beta dependency: PASS")
    test_c2_gate_gradient_matches_fd(); print("C2 gate gradient vs FD: PASS")
    test_c3_init_equivalence_to_fixed_blend(); print("C3 w=0 bitwise == fixed blend: PASS")
    test_c4_cholesky_free_form(); print("C4 Cholesky-free form: PASS")
    test_c5_learnability_given_separable_signal(); print("C5 learnability (separable): PASS")
    test_c5_teeth_no_separation_without_signal(); print("C5 teeth (non-separable): PASS")
    print("ALL PASS")
