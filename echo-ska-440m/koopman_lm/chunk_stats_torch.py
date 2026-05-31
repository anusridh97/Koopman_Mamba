"""
chunk_stats_torch.py -- PyTorch port of echo_jax.py::chunk_stats.

Beta-gated, strictly-causal chunk statistics (Echo Eq.7-10, exclusive boundary).
Replaces the leaky non-causal sequence-max normalization in the original
ska.py. Every statistic at chunk c depends only on tokens in chunks < c
(exclusive prefix) plus the cross-chunk boundary term, so chunk-causal
training, prefix recurrence, and per-token decode all coincide.

Inputs (already projected + per-token L2-normalized by the caller):
  z  : (B,T,H,r)   L2-normalized key   (lagged / right factor)
  zb : (B,T,H,r)   beta-weighted key = beta * z   (current / write factor)
  zq : (B,T,H,r)   L2-normalized query
  v  : (B,T,H,P)   value (no L2)
Returns flattened (BCH=B*nchunks*H) tensors ready for the whitened core:
  Gf (BCH,r,r), Mf (BCH,r,r), Cf (BCH,P,r), qf (BCH,r,CS), shapes
"""

import torch
import torch.nn.functional as F


def _excl_prefix(x):
    """Exclusive prefix sum along dim=1 (chunk axis)."""
    c = torch.cumsum(x, dim=1)
    z0 = torch.zeros_like(x[:, :1])
    return torch.cat([z0, c[:, :-1]], dim=1)


def chunk_stats(z, zb, zq, v, ridge, CS):
    B, T, H, r = z.shape
    P = v.shape[-1]
    nc = (T + CS - 1) // CS
    pad = nc * CS - T
    if pad:
        z  = F.pad(z,  (0, 0, 0, 0, 0, pad))
        zb = F.pad(zb, (0, 0, 0, 0, 0, pad))
        zq = F.pad(zq, (0, 0, 0, 0, 0, pad))
        v  = F.pad(v,  (0, 0, 0, 0, 0, pad))

    zc  = z.reshape(B, nc, CS, H, r)
    zbc = zb.reshape(B, nc, CS, H, r)
    zqc = zq.reshape(B, nc, CS, H, r)
    vc  = v.reshape(B, nc, CS, H, P)

    # within-chunk stats (Eq.7): G uses beta*z (zb) against z; M is lag-1; C_v = v zb^T
    Gc = torch.einsum("bcthr,bcths->bchrs", zbc, zc)
    Mc = torch.einsum("bcthr,bcths->bchrs", zbc[:, :, 1:], zc[:, :, :-1])
    Cc = torch.einsum("bcthp,bcthr->bchpr", vc, zbc)

    # cross-chunk boundary: first token of chunk c against last token of chunk c-1
    bnd = torch.einsum("bchr,bchs->bchrs", zbc[:, 1:, 0], zc[:, :-1, -1])  # (B,nc-1,H,r,r)
    bnd = torch.cat([torch.zeros(B, 1, H, r, r, dtype=z.dtype, device=z.device), bnd], dim=1)

    eye = torch.eye(r, dtype=z.dtype, device=z.device)
    G = _excl_prefix(Gc) + ridge * eye                # Eq.8
    M = _excl_prefix(Mc) + _excl_prefix(bnd)          # Eq.9 (i=1..c-1)
    Cv = _excl_prefix(Cc)                             # Eq.10

    N = B * nc * H
    Gf = (0.5 * (G + G.transpose(-1, -2))).reshape(N, r, r) + 1e-4 * eye   # symmetrize + jitter
    Mf = M.reshape(N, r, r)
    Cf = Cv.reshape(N, P, r)
    qf = zqc.permute(0, 1, 3, 4, 2).reshape(N, r, CS)
    return Gf, Mf, Cf, qf, (B, nc, H, P, CS, T, pad)


if __name__ == "__main__":
    # Causality test (mirror of echo_jax.py::test_ska_causal):
    # perturbing a FUTURE token must not change stats feeding earlier chunks.
    torch.manual_seed(0)
    B, T, H, r, P, CS = 2, 192, 4, 16, 8, 64
    z  = torch.randn(B, T, H, r)
    z  = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    beta = torch.rand(B, T, H, 1)
    zb = beta * z
    zq = torch.randn(B, T, H, r); zq = zq / (zq.norm(dim=-1, keepdim=True) + 1e-12)
    v  = torch.randn(B, T, H, P)

    Gf1, Mf1, Cf1, qf1, shp = chunk_stats(z, zb, zq, v, 1e-3, CS)

    # perturb the LAST token
    z2 = z.clone(); zb2 = zb.clone(); v2 = v.clone()
    z2[:, -1] = torch.randn(B, H, r); z2[:, -1] /= (z2[:, -1].norm(dim=-1, keepdim=True)+1e-12)
    zb2[:, -1] = beta[:, -1] * z2[:, -1]
    v2[:, -1] = torch.randn(B, H, P)
    Gf2, Mf2, Cf2, qf2, _ = chunk_stats(z2, zb2, zq, v2, 1e-3, CS)

    nc = shp[1]
    # stats for chunks strictly before the last chunk must be identical
    Gf1 = Gf1.reshape(B, nc, H, r, r); Gf2 = Gf2.reshape(B, nc, H, r, r)
    Mf1 = Mf1.reshape(B, nc, H, r, r); Mf2 = Mf2.reshape(B, nc, H, r, r)
    Cf1 = Cf1.reshape(B, nc, H, P, r); Cf2 = Cf2.reshape(B, nc, H, P, r)
    safe = nc - 1
    dG = (Gf1[:, :safe] - Gf2[:, :safe]).abs().max().item()
    dM = (Mf1[:, :safe] - Mf2[:, :safe]).abs().max().item()
    dC = (Cf1[:, :safe] - Cf2[:, :safe]).abs().max().item()
    print(f"  causal leak (future token -> earlier-chunk stats): "
          f"G={dG:.2e} M={dM:.2e} C={dC:.2e}  (expect 0)")
