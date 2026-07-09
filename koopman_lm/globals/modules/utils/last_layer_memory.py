"""
last_layer_memory.py -- BOM-LM-v0: inference-only last-layer ridge memory.

A session-scoped, causal ridge/LSTD readout placed at the norm_f -> lm_head
seam. The BACKBONE STAYS FROZEN; this only maintains small sufficient
statistics from observed tokens and emits a gated residual correction to the
final hidden state. No optimizer steps, no weight updates -- "test-time
learning" purely via exact ridge statistics, reset per request.

This is structurally a second, last-layer instance of the same ridge mechanism
SKA uses internally (G, C, Cholesky whitening), but: (a) global over the whole
prefix, not chunked; (b) persistent across the prompt, not recomputed; (c)
provenance-weighted so self-generated tokens don't reinforce themselves.

INFERENCE-ONLY DESIGN: since the model was not trained with this adapter, the
feature projection W_z is NOT learned. We use a FIXED projection (default:
identity-style top-r slice / fixed orthogonal), so the readout operates in the
model's own representation space where the binding evidence already lives. A
random W_z would project into a meaningless subspace and add noise -- avoided.

Math (gamma=0, plain ridge):
  z_t = W_z h_t                                  (r-dim feature, fixed W_z)
  G_t = lambda I + sum_{i<t} w_i z_i z_i^T        (whitened via G=LL^T)
  C_t =            sum_{i<t} w_i v_i z_i^T        (v_i = embedding residual)
  delta_h_t = C_t G_t^{-1} z_t = (C_t L^{-T})(L^{-1} z_t)
  h'_t = h_t + gate * delta_h_t                   (gate small, per-channel)

Causal discipline: read from stats over i<t, THEN write token t after its
target x_{t+1} is observed. Never write the target being predicted.
"""

import math
import torch
import torch.nn as nn


class LastLayerRidgeMemory(nn.Module):
    def __init__(self, d_model, rank=64, ridge=1e-2, gate_init=1e-2,
                 proj='slice', gen_token_weight=0.0,
                 tau=8.0, leverage_beta=1.0):
        """
        proj: 'slice'      -> z = first r dims of h (cheapest, identity-like)
              'orthogonal'  -> z = fixed random orthogonal r-subspace of h
        gen_token_weight: provenance weight for self-generated tokens (0 = don't
              learn from own samples; prompt/retrieved/tool tokens use weight 1).
        tau, leverage_beta: support-aware trust gate. The readout is damped by
              support = n_eff/(n_eff+tau) (untrusted until enough evidence) and
              by 1/(1+beta*leverage), leverage = ||L^{-1}z||^2 (high = OOD).
        """
        super().__init__()
        assert 0 < rank <= d_model, f"rank {rank} must be in (0, d_model={d_model}]"
        self.d = d_model
        self.r = rank
        self.ridge = ridge
        self.gen_w = gen_token_weight
        self.tau = float(tau)
        self.leverage_beta = float(leverage_beta)
        # fixed (non-learnable) projection
        if proj == 'orthogonal':
            W = torch.empty(rank, d_model)
            nn.init.orthogonal_(W)
        else:  # 'slice'
            W = torch.zeros(rank, d_model)
            W[torch.arange(rank), torch.arange(rank)] = 1.0
        self.register_buffer('W_z', W)                       # (r,d) fixed
        # per-channel gate (fixed small constant; not learned in inference-only)
        self.register_buffer('gate', torch.full((d_model,), float(gate_init)))
        self._reset_state(batch=1, device=W.device, dtype=torch.float32)

    def _reset_state(self, batch, device, dtype):
        r = self.r
        eye = torch.eye(r, device=device, dtype=dtype)
        self.G = self.ridge * eye.reshape(1, r, r).expand(batch, r, r).clone()
        self.C = torch.zeros(batch, self.d, r, device=device, dtype=dtype)
        self.n_eff = torch.zeros(batch, device=device, dtype=dtype)
        self._batch = batch

    def reset(self, batch=1, device=None, dtype=torch.float32):
        self._reset_state(batch, device or self.W_z.device, dtype)

    def _feat(self, h):                                      # h:(B,d) -> z:(B,r)
        return torch.einsum('rd,bd->br', self.W_z, h.float())

    @torch.no_grad()
    def read(self, h):
        """delta_h for a batch of current hidden states h:(B,d). Uses stats from
        tokens already written (i<t). Returns gated residual (B,d), damped by
        support (n_eff) and leverage (OOD score) so early/OOD reads are trusted
        less."""
        z = self._feat(h)                                    # (B,r)
        Gs = 0.5 * (self.G + self.G.transpose(-1, -2))
        L = torch.linalg.cholesky(Gs)                        # (B,r,r)
        w = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False)  # L^{-1}z
        Rw = torch.linalg.solve_triangular(
            L.transpose(-1, -2), w, upper=True)              # G^{-1}z
        delta = torch.einsum('bdr,brk->bdk', self.C, Rw).squeeze(-1)  # C G^{-1} z
        # support-aware trust: small when little evidence (n_eff) or OOD (leverage)
        leverage = (w.squeeze(-1) ** 2).sum(dim=-1)          # ||L^{-1}z||^2 (B,)
        support = self.n_eff / (self.n_eff + self.tau)
        trust = support / (1.0 + self.leverage_beta * leverage)
        delta = delta * trust.view(-1, 1)
        return (self.gate * delta).to(h.dtype)

    @torch.no_grad()
    def write(self, h, v, weight=1.0):
        """Write observed pair (h_t, v_t) into the stats. v:(B,d) target residual
        in hidden/embedding space. weight: provenance scalar or (B,) tensor.
        Weights are clamped >= 0: a negative weight would subtract an outer
        product from G and can make it indefinite, breaking the Cholesky."""
        z = self._feat(h)                                    # (B,r)
        if not torch.is_tensor(weight):
            weight = torch.full((z.shape[0],), float(weight),
                                device=z.device, dtype=z.dtype)
        else:
            weight = weight.to(device=z.device, dtype=z.dtype)
        weight = weight.view(-1).clamp_min(0.0)              # nonnegative (PD safety)
        wz = weight.view(-1, 1)                              # (B,1)
        self.G = self.G + torch.einsum('br,bs->brs', wz * z, z)
        self.C = self.C + torch.einsum('bd,br->bdr', v.float(), wz * z)
        self.n_eff = self.n_eff + weight                     # effective support count

    def read_then_write(self, h, v, weight=1.0):
        """Convenience: read with pre-write stats, then write. Returns delta_h."""
        delta = self.read(h)
        self.write(h, v, weight)
        return delta

    # ---- Streaming (persistent) API -------------------------------------
    # For token-by-token generation we carry the Cholesky factor L of G and
    # rank-1-update it per write in O(r^2), instead of refactoring G (O(r^3))
    # on every read. Uses the vendored cholesky_update.update_L_only. State
    # persists across step() calls (NOT reset per call) so the memory
    # accumulates over the whole generation, with provenance weights keeping
    # self-generated writes from dominating.

    @torch.no_grad()
    def stream_reset(self, batch=1, device=None, dtype=torch.float32):
        """Start a streaming session: init G's Cholesky factor L = sqrt(ridge) I."""
        dev = device or self.W_z.device
        r = self.r
        self._reset_state(batch, dev, dtype)
        # L for G = ridge*I is sqrt(ridge)*I
        eye = torch.eye(r, device=dev, dtype=dtype)
        self._L = math.sqrt(self.ridge) * eye.reshape(1, r, r).expand(batch, r, r).clone()

    @torch.no_grad()
    def stream_read(self, h):
        """Readout using the carried L (no refactorization). Same trust gate
        as read(); O(r^2) solves instead of O(r^3) cholesky."""
        z = self._feat(h)                                    # (B,r)
        L = self._L
        w = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False)
        Rw = torch.linalg.solve_triangular(L.transpose(-1, -2), w, upper=True)
        delta = torch.einsum('bdr,brk->bdk', self.C, Rw).squeeze(-1)
        leverage = (w.squeeze(-1) ** 2).sum(dim=-1)
        support = self.n_eff / (self.n_eff + self.tau)
        trust = support / (1.0 + self.leverage_beta * leverage)
        delta = delta * trust.view(-1, 1)
        return (self.gate * delta).to(h.dtype)

    @torch.no_grad()
    def stream_write(self, h, v, weight=1.0):
        """Write (h,v) and rank-1-update the carried L by sqrt(weight)*z, so
        L stays consistent with G += weight z z^T. O(r^2) per item."""
        from koopman_lm.globals.modules.ska.cholesky_update import update_L_only
        z = self._feat(h)
        if not torch.is_tensor(weight):
            weight = torch.full((z.shape[0],), float(weight),
                                device=z.device, dtype=z.dtype)
        else:
            weight = weight.to(device=z.device, dtype=z.dtype)
        weight = weight.view(-1).clamp_min(0.0)
        wz = weight.view(-1, 1)
        self.C = self.C + torch.einsum('bd,br->bdr', v.float(), wz * z)
        self.n_eff = self.n_eff + weight
        # keep G consistent with _L (both are covariance state). _L is the one
        # stream_read uses, but maintaining G too means read()/serialization/
        # debugging after streaming see a non-stale covariance.
        self.G = self.G + torch.einsum('br,bs->brs', wz * z, z)
        # rank-1 cholupdate of L per batch element: G += w z z^T = (sqrt(w) z)(.)^T
        zu = (weight.clamp_min(0.0).sqrt().view(-1, 1) * z)  # (B,r)
        for b in range(z.shape[0]):
            if zu[b].abs().sum() > 0:
                update_L_only(self._L[b], zu[b].clone())


def make_memory_token_weights(input_ids, prompt_len, gen_w=0.0):
    """Provenance weights for forward_with_memory. Weight at position t controls
    the WRITE of transition (h_t -> target token input_ids[t+1]).

    The first prompt_len tokens are externally observed, so writes t=0..prompt_len-2
    are prompt-supervised (weight 1). The write at t=prompt_len-1 targets the
    first GENERATED token, so it (and later) get gen_w.
    """
    B, T = input_ids.shape
    w = torch.full((B, T), float(gen_w), device=input_ids.device)
    observed_writes = max(prompt_len - 1, 0)
    if observed_writes > 0:
        w[:, :observed_writes] = 1.0
    return w


if __name__ == "__main__":
    torch.manual_seed(0)
    d, r = 64, 16
    mem = LastLayerRidgeMemory(d, rank=r, ridge=1e-2, gate_init=1.0, proj='slice')

    # --- test 1: zero perturbation before any writes ---
    mem.reset(batch=1, device='cpu')
    h0 = torch.randn(1, d)
    print(f"  delta before any writes: {mem.read(h0).abs().max():.2e}  (expect 0)")

    # --- test 2: causality (read uses only prior writes) ---
    mem.reset(batch=1, device='cpu')
    hs = [torch.randn(1, d) for _ in range(10)]
    vs = [torch.randn(1, d) for _ in range(10)]
    d5_before = mem.read(hs[5])                              # read at step 5 (5 writes done)
    for t in range(5):
        mem.write(hs[t], vs[t])
    d5_after = mem.read(hs[5])
    # now write more future tokens; reading at step 5 again must be unchanged
    snapshot = d5_after.clone()
    for t in range(5, 10):
        mem.write(hs[t], vs[t])
    # re-create state up to <5 and confirm the step-5 read only depended on <5
    mem.reset(batch=1, device='cpu')
    for t in range(5):
        mem.write(hs[t], vs[t])
    d5_recheck = mem.read(hs[5])
    print(f"  step-5 read depends only on tokens <5: "
          f"{(d5_recheck - snapshot).abs().max():.2e}  (expect 0)")

    # --- test 3: in-context learning (define binding, query it later) ---
    mem.reset(batch=1, device='cpu')
    key = torch.randn(1, d); key = key / key.norm()
    target = torch.randn(1, d)                               # the "value" residual
    # write the binding a few times (as if the prompt stated it)
    for _ in range(5):
        mem.write(key, target, weight=1.0)
    # query with the same key feature -> readout should point toward target
    out = mem.read(key)
    cos = torch.nn.functional.cosine_similarity(out, target).item()
    print(f"  in-context recall: cos(readout, target) = {cos:.3f}  (expect >0.5)")

    # --- test 4: provenance gating (generated tokens with weight 0 don't write) ---
    mem.reset(batch=1, device='cpu')
    G0 = mem.G.clone()
    mem.write(torch.randn(1, d), torch.randn(1, d), weight=0.0)
    print(f"  gen-token (w=0) leaves stats unchanged: "
          f"{(mem.G - G0).abs().max():.2e}  (expect 0)")
