"""
chunk_stats_exact_torch.py -- EXACT per-token-causal SKA statistics.

The default chunk_stats uses EXCLUSIVE-CHUNK-PREFIX boundaries: a token in
chunk c sees only completed chunks < c, so the within-chunk lag-1..lag-(S-1)
cross-covariance terms are absent. Measured cost: ~100% relative error vs a
true per-token-causal reference on short-range recall (the conv bypasses this;
this module FIXES it inside SKA).

This builds the EXACT per-token statistics:
    G_t = ridge*I + sum_{i<t} beta_i z_i z_i^T
    M_t =           sum_{i<t} beta_i z_i z_{i-1}^T       (lag-1, strict prefix)
    Cv_t=           sum_{i<t} v_i (beta_i z_i)^T
i.e. every token t gets the EXCLUSIVE prefix over ALL earlier tokens, including
those in its own chunk. Then it flattens to (B*T*H) and calls the SAME verified
ska_core (whitened L^{-1}M L^{-T}, custom O(K r^2) backward) -- no new backward.

This is "path B": the simple, expensive, exact reference. Cost is O(T) per-token
stats via cumulative sums (cheap) but T whitened-core solves instead of T/S, so
it is slower than chunked. Use it (a) to verify the chunked path's staleness and
(b) as a correctness target for an optimized intra-chunk scan / Woodbury kernel.

Inputs (caller already projected + per-token L2-normalized, beta-gated):
  z  (B,T,H,r)  zb=beta*z (B,T,H,r)  zq (B,T,H,r)  v (B,T,H,P)
Returns flattened (B*T*H, ...) ready for ska_core, plus shape tuple.
"""

import torch

from koopman_lm.kernels.lin_alg import exclusive_cumsum




def exact_stats(z, zb, zq, v, ridge):
    """Per-token exclusive-prefix stats over the WHOLE history (no chunk gap)."""
    B, T, H, r = z.shape
    P = v.shape[-1]
    # per-token rank-1 contributions
    G_contrib = torch.einsum('bthr,bths->bthrs', zb, z)                 # (B,T,H,r,r)
    # lag-1 cross-cov contribution at token t is zb_t z_{t-1}^T (t>=1)
    M_contrib = torch.zeros(B, T, H, r, r, dtype=z.dtype, device=z.device)
    if T > 1:
        M_contrib[:, 1:] = torch.einsum('bthr,bths->bthrs', zb[:, 1:], z[:, :-1])
    C_contrib = torch.einsum('bthp,bthr->bthpr', v, zb)                 # (B,T,H,P,r)

    eye = torch.eye(r, dtype=z.dtype, device=z.device)
    G = exclusive_cumsum(G_contrib, dim=1) + ridge * eye                    # (B,T,H,r,r)
    M = exclusive_cumsum(M_contrib, dim=1)
    Cv = exclusive_cumsum(C_contrib, dim=1)                                 # (B,T,H,P,r)

    N = B * T * H
    Gf = 0.5 * (G + G.transpose(-1, -2)).reshape(N, r, r) + 1e-4 * eye
    Mf = M.reshape(N, r, r)
    Cf = Cv.reshape(N, P, r)
    qf = zq.reshape(N, r, 1)
    return Gf, Mf, Cf, qf, (B, T, H, P)


if __name__ == "__main__":
    # ska_core moved out of the deleted koopman_lm/ska_core_torch.py into
    # kernels/ska_operator.py; loading it by file path from the old location
    # made this self-check unrunnable. Same function, same signature.
    torch.set_default_dtype(torch.float64); torch.manual_seed(0)
    from koopman_lm.kernels.ska_operator import ska_core

    # same structured short-range recall task that showed ~100% chunk error
    B, H, r, P, K, ridge, T = 1, 2, 16, 8, 2, 1e-3, 256
    keys = torch.randn(B, T, H, r); keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-12)
    lag = 3
    zq = torch.zeros(B, T, H, r); zq[:, lag:] = keys[:, :-lag]
    zq = zq / (zq.norm(dim=-1, keepdim=True) + 1e-12)
    beta = torch.rand(B, T, H, 1); zb = beta * keys
    v = torch.randn(B, T, H, P)

    # exact-stats path
    Gf, Mf, Cf, qf, (Bn, Tn, Hn, Pn) = exact_stats(keys, zb, zq, v, ridge)
    Yexact = ska_core(Gf, Mf, Cf, qf, K).reshape(Bn, Tn, Hn, Pn)

    # per-token reference (the ground truth from the staleness measurement)
    eye = torch.eye(r); ref = torch.zeros(B, T, H, P)
    for t in range(T):
        if t == 0:
            G = ridge*eye.reshape(1,1,r,r).expand(B,H,r,r).clone(); M=torch.zeros(B,H,r,r); Cv=torch.zeros(B,H,P,r)
        else:
            zp=keys[:, :t].permute(0,2,1,3); zbp=zb[:, :t].permute(0,2,1,3); vp=v[:, :t].permute(0,2,1,3)
            G=torch.einsum('bhir,bhis->bhrs',zbp,zp)+ridge*eye
            M=torch.einsum('bhir,bhis->bhrs',zbp[:,:,1:],zp[:,:,:-1]) if t>1 else torch.zeros(B,H,r,r)
            Cv=torch.einsum('bhip,bhir->bhpr',vp,zbp)
        Nn=B*H
        ref[:, t]=ska_core(G.reshape(Nn,r,r),M.reshape(Nn,r,r),Cv.reshape(Nn,P,r),
                           zq[:, t].reshape(Nn,r,1),K).reshape(B,H,P)
    rel = (Yexact - ref).norm().item() / (ref.norm().item() + 1e-12)
    print(f"exact-stats vs per-token reference: rel={rel:.3e}  (expect ~0)")
    print("  -> if ~0, the exact intra-chunk path FIXES the ~100% short-range "
          "staleness using the existing verified ska_core (no new backward).")
