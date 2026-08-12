"""SKA operator health metrics.

A free function rather than a method on `SKAModule`, so the model file holds
only the model. `ska_health` reads the module's projections and scale-resolution
helpers; it never mutates it, and nothing on the training path imports this.

Why it lives in `koopman_lm/` and not `experimentation/`: these are intrinsic
properties of a trained operator, so anyone who installs the package should be
able to point them at a checkpoint. Wiring them into a training loop -- hooks,
`capture()`, wandb, the SKA-vs-Mamba residual ratio -- is a training concern and
stays in `experimentation/training/diagnostics.py`.
"""
import torch

from koopman_lm.kernels.chunk_stats import (
    chunk_stats as _causal_chunk_stats, symmetric_key_value, causal_normalize)
from koopman_lm.kernels.lin_alg import whiten_M, spec_w

__all__ = ["ska_health", "spectral_radius"]


def spectral_radius(A, n_iters=12):
    """Spectral radius (max |eigenvalue|) of a batch of square matrices.

    Diagnostics-only, and SPEED-CRITICAL: torch.linalg.eigvals (the general
    non-symmetric eig) is extremely slow when called on many small matrices and
    dominated the diagnostic step cost. We instead use batched power iteration
    (just matmuls -> fast on GPU): iterate v into the dominant invariant
    subspace, then return ||A v|| for unit v. This is exact for a dominant real
    eigenvalue, and for a complex-conjugate 2x2 block (which acts as rho *
    rotation, norm-preserving) it also yields rho -- accurate enough for a
    health metric. A is (..., r, r); returns (...,).
    """
    v = torch.randn(*A.shape[:-1], 1, device=A.device, dtype=A.dtype)
    v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-12)
    for _ in range(n_iters):
        v = A @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-12)
    return (A @ v).norm(dim=-2).squeeze(-1)


@torch.no_grad()
def ska_health(ska, hidden_states, max_batch=2):
    """Per-(batch, chunk, head) SKA health metrics, computed off the hot
    path but FAITHFUL to the operators the training forward applies.

    Fidelity: this reuses the SAME beta-gated, strictly-causal chunk
    statistics the forward uses (`chunk_stats`), the SAME v1.1 symmetric
    sqrt(beta) key/value convention (`causal_normalize` +
    `symmetric_key_value` -- x = sqrt(beta) * key fed into BOTH key slots,
    vbar = sqrt(beta) * value), and forms each chunk's operator exactly as
    `ska_core` does -- A_eff = gamma * alpha * (L^-1 M L^-T), G/M being the
    EXCLUSIVE-prefix per-chunk sufficient statistics (Eqs. 7-10). gamma is
    included because the forward scales Y by gamma^K after ska_core
    applies (alpha W)^K -- so the operator each filter step actually
    applies is gamma * alpha * W (a no-op in the fixed gamma=1.0 regime;
    material in the squash/clamp learnable regimes, resolved the same way
    forward() resolves it via `_resolve_gamma`). So every operator
    measured here is one a real query sees, not a non-causal whole-sequence
    summary.

    NOTE: nothing is reduced here. The three operator metrics are returned
    as full (B, n_chunks, H) tensors so downstream (plotting / wandb) can
    decide how to aggregate -- per head, per chunk, or the whole pool.
    Chunk 0 has an empty exclusive prefix (G=ridge I, M=0 -> zero operator);
    it is kept (index 0) and excluded by the monitor's summaries by default.

    NOTE: this always reports the CHUNKED-path operators (the same
    beta-gated statistics family `forward()`'s non-exact branch consumes)
    as the bounded-cost health view, regardless of which of the four
    forward strategies (chunked / exact_intrachunk / inverse_cholesky /
    prefix_scan) is actually selected. prefix_scan (the recommended
    quality path) and inverse_cholesky compute per-token, not per-chunk,
    operators via different kernels entirely -- this function does not
    reach into those; it reports the chunk-level approximation as a
    cheap, bounded-cost proxy for the operator family they all share.

    Detached, fp32, no value readout. `max_batch` caps the batch used for
    the per-chunk eigendecompositions so cost is bounded regardless of the
    training batch size. Returns GPU tensors; caller does the CPU sync
    (see experimentation.training.diagnostics.SKAHealthMonitor).

    Returns dict:
      spectral_radius : (B, nc, H) max|eig(A_eff)| per instance.
                        Healthy ~[0.3, 0.95]; ~0 = unlearned; >1 = unstable.
      lambda_min      : (B, nc, H) smallest eig(G_tilde) per instance.
                        Pinned at the ridge floor => rank-deficient keys.
      gap             : (B, nc, H) ||A_eff^K - A_eff|| / ||A_eff||.
                        >0.5 => the power filter dominates the operator.
      n_chunks        : int, number of chunks (chunk 0 = no history).
      gate_mag/beta_mean/outproj_norm/eta/gamma/ridge_eps : scalars.
    """
    r, H, P = ska.rank, ska.H, ska.P
    ridge = float(ska.ridge_eps)
    K = int(ska.power_K)
    CS = ska.chunk_size

    x = hidden_states
    if x.shape[0] > max_batch:
        x = x[:max_batch]
    B, T, _ = x.shape

    z = ska.key_proj(x).reshape(B, T, H, r).float()
    zq = ska.query_proj(x).reshape(B, T, H, r).float()
    v = ska.value_proj(x).reshape(B, T, H, P).float()
    beta = torch.sigmoid(ska.beta_proj(x)).float()                # (B,T,H)

    # SAME normalization + symmetric sqrt(beta) key/value convention the
    # forward uses (v1.1): G/C are own-weight and beta-invariant; M
    # becomes the contractive cross-weight sqrt(beta_t beta_{t-1}).
    z_n = causal_normalize(z, ska.norm_clip_c)
    zq_n = causal_normalize(zq, ska.norm_clip_c)
    x_n, v_w = symmetric_key_value(z_n, beta, v)

    # SAME strictly-causal, beta-gated, exclusive-prefix chunk statistics
    # the training forward consumes. Gf already carries ridge + jitter.
    Gf, Mf, Cf, qf, shp = _causal_chunk_stats(x_n, x_n, zq_n, v_w, ridge, CS)
    Bc, nc, Hc = shp[0], shp[1], shp[2]

    # Per-instance operator (N = B*nc*H), formed exactly like ska_core.
    L, info = torch.linalg.cholesky_ex(Gf)
    if (info > 0).any():
        eye = torch.eye(r, device=Gf.device, dtype=Gf.dtype)
        L_j, _ = torch.linalg.cholesky_ex(Gf + 1e-4 * eye)
        L = torch.where((info > 0).view(-1, 1, 1), L_j, L)

    W = whiten_M(L, Mf)                        # L^-1 M L^-T  (N,r,r)
    alpha = spec_w(W)                          # (N,1) detached spectral-norm scale
    gamma = ska._resolve_gamma()               # float (fixed) or tensor (learnable)
    gamma_val = gamma if isinstance(gamma, float) else float(gamma.detach())
    A_eff = alpha.unsqueeze(-1) * W            # operator applied per filter step
    if gamma_val != 1.0:
        A_eff = A_eff * gamma_val

    radius = spectral_radius(A_eff)                                # (N,)
    lam = torch.linalg.eigvalsh(Gf).amin(dim=-1)                   # (N,)
    A_K = torch.linalg.matrix_power(A_eff, K)
    gap = (A_K - A_eff).norm(dim=(-2, -1)) / (A_eff.norm(dim=(-2, -1)) + 1e-12)

    # (N,) -> (B, nc, H). No reduction: hand back the whole distribution.
    radius = radius.view(Bc, nc, Hc)
    lam = lam.view(Bc, nc, Hc)
    gap = gap.view(Bc, nc, Hc)

    eta = ska._resolve_eta().detach().float().reshape(())
    gate = ska.layerscale_gate
    gate_mag = gate.abs().mean() if gate is not None else eta.abs()

    return {
        'spectral_radius': radius.detach(),
        'lambda_min': lam.detach(),
        'gap': gap.detach(),
        'n_chunks': int(nc),
        'gate_mag': gate_mag.detach(),
        'beta_mean': beta.mean().detach(),
        'outproj_norm': ska.out_proj.weight.detach().float().norm(),
        'eta': eta,
        'gamma': torch.tensor(gamma_val, device=x.device),
        'ridge_eps': torch.tensor(ridge, device=x.device),
    }
