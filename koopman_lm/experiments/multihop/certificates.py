"""
Training-free test of the multi-hop wall via the papers' certificates.
Model: orthogonal-residual retrieval (NeurIPS paper's minimal frozen multi-hop model).
  base score   A[i,k] = <q_i, d_k>
  first-hop j  shifts scores along ONE direction g_j[k] = <d_j, d_k> (rank-one, Lemma 2)
  residual rule selects theta* = -A[i,j]   (q' = q_i - <q_i,d_j> d_j)
Target: hard sign-matrix relation r_B(i,j) = pi(j) if B[i,j]=+1 else j  (derangement pi).

Computed (no training):
  GEO  = % states where gold is reachable by SOME theta (1 direction)   [Thm 1, interval]
  SEL  = % states where the residual rule's theta* actually ranks gold first
  -> SEL<GEO = selection failures (recoverable); 1-GEO = geometric failures (need re-encode)
  ENV  = row-adaptive envelope (always 1.0)  [Lemma 1]
  d*   = min #directions to make gold reachable (multi-direction LP, Thm 7)
"""
import numpy as np
from scipy.optimize import linprog
rng = np.random.default_rng(0)

def setup(d0, m, n):
    D = rng.standard_normal((m, d0)); D /= np.linalg.norm(D,axis=1,keepdims=True)
    Q = rng.standard_normal((n, d0)); Q /= np.linalg.norm(Q,axis=1,keepdims=True)
    A = Q @ D.T                       # [n,m] base scores
    Gram = D @ D.T                    # [m,m] residual directions
    return A, Gram

def hard_relation(n, m):
    pi = np.roll(np.arange(m), 1)     # derangement
    B = rng.choice([-1, 1], size=(n, m))
    R = np.where(B == 1, pi[None, :], np.arange(m)[None, :])
    return R

def one_dir(A, Gram, R):
    n, m = A.shape; geo = sel = 0; tot = 0
    for i in range(n):
        for j in range(m):
            ks = R[i, j]
            if ks == j:   # degenerate (gold == first hop): skip
                continue
            base = A[i]; gj = Gram[j]
            M = base[ks] - base; Dd = gj[ks] - gj
            lo, hi = -np.inf, np.inf; feas = True
            for k in range(m):
                if k == ks: continue
                Mk, Dk = M[k], Dd[k]
                if abs(Dk) < 1e-12:
                    if Mk <= 0: feas = False; break
                elif Dk > 0: lo = max(lo, -Mk/Dk)
                else:        hi = min(hi, -Mk/Dk)
            tot += 1
            if feas and lo < hi:
                geo += 1
                theta = -A[i, j]
                if lo < theta < hi: sel += 1
    return geo/tot, sel/tot, tot

def min_directions(A, Gram, R, maxJ=8, n_states=120):
    """min #directions (columns of Gram) to rank gold w/ margin>=1, via L0 search using LP feasibility."""
    n, m = A.shape
    states = [(i, j) for i in range(n) for j in range(m) if R[i, j] != j]
    rng.shuffle(states); states = states[:n_states]
    need = []
    for (i, j) in states:
        ks = R[i, j]; base = A[i]
        # contrasts for each candidate k!=ks across ALL possible directions j' (columns)
        # feasibility with a set J: exists theta s.t. (base[ks]-base[k]) + sum_{j' in J} theta_j'(Gram[j',ks]-Gram[j',k]) >= 1
        Mk = np.array([base[ks]-base[k] for k in range(m) if k != ks])     # [m-1]
        found = maxJ + 1
        for J in range(1, maxJ+1):
            cols = list(range(J))   # greedy: first J docs as directions (incl. j-th neighborhood)
            Bmat = np.array([[Gram[c, ks]-Gram[c, k] for c in cols]
                             for k in range(m) if k != ks])                # [m-1, J]
            # feasibility LP: find theta with Bmat theta >= 1 - Mk  ->  -Bmat theta <= Mk - 1
            res = linprog(c=np.zeros(J), A_ub=-Bmat, b_ub=Mk - 1.0,
                          bounds=[(-50, 50)]*J, method='highs')
            if res.success:
                found = J; break
        need.append(found)
    need = np.array(need)
    return need

print("HARD sign-matrix relation, m=n=40.  Rank-one residual (frozen) vs envelope.\n")
print(f"{'d0':>4} | {'GEO(reachable)':>15} | {'SEL(residual)':>14} | {'ENV':>5}")
for d0 in [8, 16, 32, 64, 128]:
    A, Gram = setup(d0, 40, 40)
    R = hard_relation(40, 40)
    geo, sel, tot = one_dir(A, Gram, R)
    print(f"{d0:>4} | {100*geo:>13.1f}% | {100*sel:>12.1f}% | {'100%':>5}")

print("\nMin #directions to reach gold (d0=32, hard relation):")
A, Gram = setup(32, 40, 40); R = hard_relation(40, 40)
need = min_directions(A, Gram, R, maxJ=8)
import numpy as np
for J in range(1, 9):
    print(f"  <= {J} directions: {100*np.mean(need<=J):>5.1f}% of states reachable")
print(f"  unreachable within 8: {100*np.mean(need>8):.1f}%")
