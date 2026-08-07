"""
ska_fast.py -- Optional performance patch for SKAModule (440M rewrite).

RECONCILED with the JAX-parity core. The original ska_fast existed to avoid the
expensive autograd-through-cholesky and to cache the spectral-norm vector. The
new SKAModule already uses a custom autograd Function (ska_core_torch) whose
backward is O(K r^2) and never differentiates the Cholesky -- so the eig /
cached-spectral-norm machinery is GONE. What remains worth keeping:

  * Fused key/query/value projection (3 Linears -> 1 GEMM).
  * BF16 projection (the matmul), fp32 only inside the core (unchanged there).

The patch keeps checkpoint compatibility: it saves/loads with the original
key_proj/query_proj/value_proj names. beta_proj / out_proj / layerscale_gate
are untouched by the fuse.

If you don't need the fused proj, just don't call patch_ska_module -- the plain
SKAModule is already fast and correct.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FusedProjSlice(nn.Module):
    """Read-only nn.Linear-like view into a slice of fused_proj (for recurrent
    compatibility: ska.key_proj(x) etc. still work after patching)."""
    def __init__(self, fused_proj, start, end):
        super().__init__()
        # Stash the fused proj WITHOUT registering it as a submodule: a plain
        # `self._fused_proj = fused_proj` would make nn.Module register it as a
        # child, so every slice would re-export `<name>._fused_proj.weight` into
        # state_dict (3 duplicate copies of the fused weight) and break loading.
        object.__setattr__(self, "_fused_proj", fused_proj)
        self._start = start
        self._end = end

    @property
    def weight(self):
        return self._fused_proj.weight[self._start:self._end]

    @property
    def bias(self):
        return None

    def forward(self, x):
        return F.linear(x, self._fused_proj.weight[self._start:self._end])


def _fused_to_original_state_dict(ska, state_dict, prefix, local_metadata):
    fused_key = prefix + 'fused_proj.weight'
    if fused_key in state_dict:
        w = state_dict.pop(fused_key)
        H, r, P = ska.H, ska.rank, ska.P
        state_dict[prefix + 'key_proj.weight'] = w[:H * r]
        state_dict[prefix + 'query_proj.weight'] = w[H * r:2 * H * r]
        state_dict[prefix + 'value_proj.weight'] = w[2 * H * r:]


def _original_to_fused_state_dict(ska, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):
    kp, qp, vp = (prefix + 'key_proj.weight', prefix + 'query_proj.weight',
                  prefix + 'value_proj.weight')
    if kp in state_dict and qp in state_dict and vp in state_dict:
        fused = torch.cat([state_dict.pop(kp), state_dict.pop(qp),
                           state_dict.pop(vp)], dim=0)
        state_dict[prefix + 'fused_proj.weight'] = fused


def patch_ska_module(ska):
    """Fuse k/q/v projections into one GEMM. Forward/backward math unchanged
    (still routes through the SKAModule.forward custom core)."""
    import types
    d_model, H, r, P = ska.d_model, ska.H, ska.rank, ska.P

    assert ska.key_proj.bias is None and ska.query_proj.bias is None \
        and ska.value_proj.bias is None, "ska_fast assumes no bias on k/q/v"

    fused_dim = 2 * H * r + H * P
    fused = nn.Linear(d_model, fused_dim, bias=False,
                      device=ska.key_proj.weight.device,
                      dtype=ska.key_proj.weight.dtype)
    with torch.no_grad():
        fused.weight[:H * r] = ska.key_proj.weight
        fused.weight[H * r:2 * H * r] = ska.query_proj.weight
        fused.weight[2 * H * r:] = ska.value_proj.weight
    ska.fused_proj = fused

    ska._modules.pop('key_proj'); ska._modules.pop('query_proj'); ska._modules.pop('value_proj')
    ska.key_proj = _FusedProjSlice(fused, 0, H * r)
    ska.query_proj = _FusedProjSlice(fused, H * r, 2 * H * r)
    ska.value_proj = _FusedProjSlice(fused, 2 * H * r, fused_dim)

    try:
        ska.register_state_dict_post_hook(_fused_to_original_state_dict)
        ska.register_load_state_dict_pre_hook(_original_to_fused_state_dict)
    except AttributeError:
        ska._register_state_dict_hook(_fused_to_original_state_dict)
        ska._register_load_state_dict_pre_hook(_original_to_fused_state_dict)

    # Single-GEMM forward: compute the fused projection once, split, then hand
    # off to the SAME causal beta-gated stats + custom whitened core as the
    # plain SKAModule.forward (identical math, one matmul instead of three).
    # The _FusedProjSlice views above keep ska.key_proj(x)/query_proj(x)/
    # value_proj(x) working for the recurrent decode path and checkpoint I/O.
    def forward_fused_single_gemm(self, hidden_states):
        B, T, _ = hidden_states.shape
        Hh, rr, Pp = self.H, self.rank, self.P
        from contextlib import nullcontext
        combined = self.fused_proj(hidden_states)
        z = combined[:, :, :Hh * rr].reshape(B, T, Hh, rr)
        zq = combined[:, :, Hh * rr:2 * Hh * rr].reshape(B, T, Hh, rr)
        v = combined[:, :, 2 * Hh * rr:].reshape(B, T, Hh, Pp)
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda else nullcontext()
        from koopman_lm.kernels.chunk_stats import (
            chunk_stats as _cs, symmetric_key_value, causal_normalize)
        from koopman_lm.kernels.ska_operator import ska_core
        with ctx:
            z_f = z.float(); zq_f = zq.float(); v_f = v.float(); beta_f = beta.float()
            z_n = causal_normalize(z_f, self.norm_clip_c)
            zq_n = causal_normalize(zq_f, self.norm_clip_c)
            # v1.1 symmetric sqrt(beta): x into both key slots, vbar as value.
            x_n, v_w = symmetric_key_value(z_n, beta_f, v_f)
            Gf, Mf, Cf, qf, shp = _cs(x_n, x_n, zq_n, v_w, self.ridge_eps, self.chunk_size)
            Y = ska_core(Gf, Mf, Cf, qf, self.power_K)
            Bc, nc, Hc, Pc, CS, Tt, pad = shp
            g = self._resolve_gamma()
            if isinstance(g, float):
                if g != 1.0: Y = Y * (g ** self.power_K)
            else:
                Y = Y * (g ** self.power_K)
            Y = Y.reshape(Bc, nc, Hc, Pc, CS).permute(0, 1, 4, 2, 3).reshape(Bc, nc * CS, Hc, Pc)[:, :Tt]
        y_hat = self.eta * Y.to(hidden_states.dtype)
        out = self.out_proj(y_hat.reshape(B, T, Hh * Pp))
        if self.layerscale_gate is not None:
            out = out * self.layerscale_gate
        return out

    ska.forward = types.MethodType(forward_fused_single_gemm, ska)
