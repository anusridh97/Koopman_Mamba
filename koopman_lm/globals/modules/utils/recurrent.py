"""
recurrent.py -- O(1) state recurrent inference wrapper for KoopmanLM.

REWRITTEN for the 440M / JAX-parity rewrite. The SKA decode path now matches
the new training math exactly:
  * beta-gated causal statistics (G = sum beta z z^T, M = sum beta z z_{t-1}^T,
    C_v = sum beta v z^T) -- NO non-causal sequence-max.
  * whitened operator core y = C_v L^{-T}(alpha W)^K L^{-1} q, W=L^{-1}M L^{-T}
    (same as ska_core_torch / training forward).
  * eta/gamma applied via the SKAModule's own resolved values (fixed 1.0 for
    440M), and the LayerScale gate applied on the output.

State per SKA layer is O(r^2): {G, M, C_v, x_last, L}. This version carries the
Cholesky factor L of G and updates it with an O(r^2) rank-1 cholupdate per token
(NeurIPS "It Cancels"), so there is no per-token re-Cholesky. The factor is
seeded once from the prompt stats during prefill (the only O(r^3), paid once).

Mamba-2 decode is unchanged (uses mamba_ssm's built-in step()).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from koopman_lm.models.koopman_lm import KoopmanLM, Mamba2Block, SKABlock
from koopman_lm.globals.modules.ska.core import _whiten_M, _spec_w, _tri_solve_lower, _tri_solve_lowerT
from koopman_lm.globals.modules.ska.chunk_stats import symmetric_key_value


def _ska_apply_whitened(L, M, Cv, q, K, gamma_value):
    """y = C_v L^{-T}(alpha W)^K L^{-1} q, given Cholesky L of G. Matches the
    training core's forward (no grad needed at decode).
    L:(N,r,r) M:(N,r,r) Cv:(N,P,r) q:(N,r,1) -> (N,P,1)."""
    W = _whiten_M(L, M)
    alpha = _spec_w(W).unsqueeze(-1)                  # (N,1,1)
    U = _tri_solve_lower(L, q)
    for _ in range(K):
        U = alpha * (W @ U)
    XK = _tri_solve_lowerT(L, U)
    y = Cv @ XK
    if isinstance(gamma_value, float):
        if gamma_value != 1.0:
            y = y * (gamma_value ** K)
    else:
        y = y * (gamma_value ** K)
    return y


class SKAState:
    """Fixed-size recurrent state for one SKA layer (sqrt-beta symmetric)."""
    __slots__ = ['G', 'M', 'C_v', 'x_last', 'L']

    def __init__(self, B, H, r, P, device, dtype=torch.float32, ridge_eps=1e-3):
        eye = torch.eye(r, device=device, dtype=dtype)
        self.G = ridge_eps * eye.reshape(1, 1, r, r).expand(B, H, r, r).clone()
        self.L = math.sqrt(ridge_eps) * eye.reshape(1, 1, r, r) \
                     .expand(B, H, r, r).clone()      # carried Cholesky of G
        self.M = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        self.C_v = torch.zeros(B, H, P, r, device=device, dtype=dtype)
        # (B,H,r) previous SYMMETRIC key x = sqrt(beta)*z (for the boundary M
        # cross-term sqrt(beta_t beta_{t-1})). NOT the raw key -- carrying raw z
        # here is the train/decode divergence bug.
        self.x_last = None


class RecurrentKoopmanLM(nn.Module):
    """O(1) state wrapper for autoregressive generation with KoopmanLM."""

    def __init__(self, model: KoopmanLM):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self._ska_indices = set(self.cfg.ska_layer_indices)
        self._ska_states = {}
        self._mamba_states = {}
        self._conv_caches = {}     # idx -> (B, K-1, d) cached normed h for short conv
        self._initialized = False

    def reset(self):
        self._ska_states.clear()
        self._mamba_states.clear()
        self._conv_caches.clear()
        self._initialized = False

    # ---- helpers to project + causally normalize one or many tokens ----
    @staticmethod
    def _proj_norm(ska, h):
        """h:(B,t,d) -> x,(B,t,H,r) zq_n,(B,t,H,r) vbar,(B,t,H,P), the v1.1
        SYMMETRIC sqrt(beta) key/value (x=sqrt(beta)*z, vbar=sqrt(beta)*v) via
        the shared symmetric_key_value helper -- identical convention to the
        training/chunk path. Query zq is NOT beta-weighted. Note: beta is used
        ONLY inside the helper; no downstream site re-derives it from a norm."""
        B, t, _ = h.shape
        H, r, P = ska.H, ska.rank, ska.P
        z = ska.key_proj(h).reshape(B, t, H, r).float()
        zq = ska.query_proj(h).reshape(B, t, H, r).float()
        v = ska.value_proj(h).reshape(B, t, H, P).float()
        beta = torch.sigmoid(ska.beta_proj(h)).float()           # (B,t,H)
        z_n = z * torch.rsqrt((z * z).sum(-1, keepdim=True) + 1e-12)
        zq_n = zq * torch.rsqrt((zq * zq).sum(-1, keepdim=True) + 1e-12)
        x, vbar = symmetric_key_value(z_n, beta, v)
        return x, zq_n, vbar

    @torch.no_grad()
    def prefill(self, input_ids):
        self.reset()
        model = self.model
        h = model.embed(input_ids)
        for idx, (seq, mlp) in enumerate(zip(model.seq_layers, model.mlp_layers)):
            if idx in self._ska_indices:
                h, st = self._ska_prefill(seq, h, idx)
                self._ska_states[idx] = st
            else:
                h, st = self._mamba_prefill(seq, h, idx)
                self._mamba_states[idx] = st
            h = mlp(h)
        h = model.norm_f(h)
        logits = model.lm_head(h)
        self._initialized = True
        return logits

    def _ska_prefill(self, ska_block, x, idx):
        """Run the (training-identical) parallel SKA forward for the output,
        and accumulate the full beta-gated state for continuation."""
        B, T, _ = x.shape
        ska = ska_block.ska
        H, r, P = ska.H, ska.rank, ska.P

        # correct output via the standard module forward (causal, whitened core)
        output = ska_block(x)

        # accumulate state from the full prefix (no future leak: these are sums
        # over the whole prefill window, which is exactly tokens <= last)
        h = ska_block.norm(x)
        x_n, zq_n, vbar = self._proj_norm(ska, h)
        xp = x_n.permute(0, 2, 1, 3)      # (B,H,T,r) symmetric key sqrt(beta)*z
        vbp = vbar.permute(0, 2, 1, 3)    # (B,H,T,P) sqrt(beta)*v
        st = SKAState(B, H, r, P, x.device, ridge_eps=ska.ridge_eps)
        eye = torch.eye(r, device=x.device, dtype=torch.float32)
        # symmetric key in BOTH slots: G,C own-weight beta (invariant), M gets the
        # contractive cross-weight sqrt(beta_t beta_{t-1}).
        st.G = torch.einsum('bhtr,bhts->bhrs', xp, xp) + ska.ridge_eps * eye
        if T > 1:
            st.M = torch.einsum('bhtr,bhts->bhrs', xp[:, :, 1:], xp[:, :, :-1])
        st.C_v = torch.einsum('bhtp,bhtr->bhpr', vbp, xp)
        st.x_last = xp[:, :, -1]          # (B,H,r) symmetric key
        # seed the carried factor from the prompt stats (one O(r^3), total)
        Gs = 0.5 * (st.G + st.G.transpose(-1, -2)).reshape(B * H, r, r)
        Ls, info = torch.linalg.cholesky_ex(Gs)
        Lj, _ = torch.linalg.cholesky_ex(Gs + 1e-4 * eye)
        st.L = torch.where((info > 0).reshape(-1, 1, 1), Lj, Ls).reshape(B, H, r, r)
        # seed the short-conv cache with the last (K-1) normed inputs so the
        # first decode step's conv sees correct local history.
        if getattr(ska_block, 'short_conv', None) is not None:
            k = ska_block.short_conv.kernel_size[0]
            pad = k - 1
            if T >= pad:
                self._conv_caches[idx] = h[:, T - pad:].detach()
            else:
                left = torch.zeros(B, pad - T, h.shape[-1],
                                   device=x.device, dtype=h.dtype)
                self._conv_caches[idx] = torch.cat([left, h], dim=1).detach()
        return output, st

    @torch.no_grad()
    def _mamba_prefill(self, mamba_block, x, idx):
        B = x.shape[0]
        mamba = mamba_block.mamba
        if not hasattr(mamba, 'layer_idx') or mamba.layer_idx is None:
            mamba.layer_idx = idx
        try:
            from mamba_ssm.utils.generation import InferenceParams
            ip = InferenceParams(max_seqlen=self.cfg.max_seq_len, max_batch_size=B)
            conv_state, ssm_state = mamba.allocate_inference_cache(
                B, self.cfg.max_seq_len, dtype=x.dtype)
            ip.key_value_memory_dict[mamba.layer_idx] = (conv_state, ssm_state)
            h_normed = mamba_block.norm(x)
            out = mamba(h_normed, inference_params=ip)
            output = x + out
            ip.seqlen_offset = x.shape[1]
            state = (conv_state.clone(), ssm_state.clone())
            return output, state
        except (ImportError, AttributeError) as e:
            if not getattr(self, 'allow_slow_mamba_prefill', False):
                # NEVER silently fall back to an empty cache: prefill output
                # would be correct but the decode cache would start from ZERO
                # Mamba state, so generation would proceed as if the prompt
                # never happened -- silently wrong output, no error. Hard-fail.
                raise RuntimeError(
                    "Mamba recurrent prefill needs the mamba_ssm InferenceParams "
                    "path, which is unavailable here. A full-forward + empty-cache "
                    "fallback is NOT decode-correct (decode would start from zero "
                    "Mamba state). Set recurrent.allow_slow_mamba_prefill=True to "
                    "use the slow but correct per-token cache-filling fallback."
                ) from e
            # Slow but CORRECT fallback: step through the prompt one token at a
            # time so conv_state/ssm_state are actually filled from the prompt.
            conv_state, ssm_state = mamba.allocate_inference_cache(
                B, self.cfg.max_seq_len, dtype=x.dtype)
            outs = []
            for t in range(x.shape[1]):
                xt = x[:, t:t+1]
                ht = mamba_block.norm(xt)
                out_t, conv_state, ssm_state = mamba.step(ht, conv_state, ssm_state)
                outs.append(xt + out_t)
            output = torch.cat(outs, dim=1)
            state = (conv_state, ssm_state)
            return output, state

    @torch.no_grad()
    def step(self, token_id):
        assert self._initialized, "Call prefill() first"
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        model = self.model
        h = model.embed(token_id)
        for idx, (seq, mlp) in enumerate(zip(model.seq_layers, model.mlp_layers)):
            if idx in self._ska_indices:
                h = self._ska_step(seq, h, idx)
            else:
                h = self._mamba_step(seq, h, idx)
            h = mlp(h)
        h = model.norm_f(h)
        return model.lm_head(h)

    # ---- Streaming with last-layer memory (CORRECT: carries backbone state) --
    # Unlike model.step_memory_only_debug (which re-encodes length-1 with no
    # backbone state), this drives the full recurrent Mamba+SKA state AND the
    # persistent last-layer memory, so generated-token hidden states match a
    # full-prefix forward. This is the real streaming-generation path.

    @torch.no_grad()
    def prefill_with_memory(self, input_ids):
        """Recurrent prefill that also prefills the last-layer memory from the
        true (post-norm_f) prompt hidden states. Requires
        model.attach_last_layer_memory() to have been called."""
        mem = self.model.last_layer_memory
        assert mem is not None, "call model.attach_last_layer_memory() first"
        self.reset()
        model = self.model
        B, T = input_ids.shape
        assert T > 0, "prefill_with_memory requires at least one prompt token"
        h = model.embed(input_ids)
        for idx, (seq, mlp) in enumerate(zip(model.seq_layers, model.mlp_layers)):
            if idx in self._ska_indices:
                h, st = self._ska_prefill(seq, h, idx); self._ska_states[idx] = st
            else:
                h, st = self._mamba_prefill(seq, h, idx); self._mamba_states[idx] = st
            h = mlp(h)
        h = model.norm_f(h)                                   # (B,T,d) true states
        self._initialized = True
        E = model.lm_head.weight
        mem.stream_reset(batch=B, device=input_ids.device, dtype=torch.float32)
        base = model.lm_head(h); p = torch.softmax(base.float(), dim=-1)
        exp_emb = p @ E.float()
        logits = torch.empty(B, T, E.shape[0], device=input_ids.device, dtype=h.dtype)
        for t in range(T):
            ht = h[:, t]
            logits[:, t] = model.lm_head(ht + mem.stream_read(ht))
            if t + 1 < T:
                mem.stream_write(ht, E[input_ids[:, t + 1]] - exp_emb[:, t], weight=1.0)
        self._mem_last_h = h[:, -1]
        self._mem_last_exp = exp_emb[:, -1]
        return logits

    @torch.no_grad()
    def step_with_memory(self, token_id, write_weight=None):
        """One CORRECT streaming step: advances recurrent backbone state, writes
        the pending memory transition (prev hidden -> this token, provenance-
        weighted), reads memory at the true hidden state, returns next logits."""
        assert self._initialized, "call prefill_with_memory() first"
        mem = self.model.last_layer_memory
        assert mem is not None and hasattr(mem, '_L'), "call prefill_with_memory() first"
        model = self.model
        E = model.lm_head.weight
        ww = mem.gen_w if write_weight is None else write_weight
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        # write pending transition from the previous step's true hidden state
        if getattr(self, '_mem_last_h', None) is not None:
            v = E[token_id[:, 0]] - self._mem_last_exp
            mem.stream_write(self._mem_last_h, v, weight=ww)
        # advance backbone with carried recurrent state
        h = model.embed(token_id)
        for idx, (seq, mlp) in enumerate(zip(model.seq_layers, model.mlp_layers)):
            if idx in self._ska_indices:
                h = self._ska_step(seq, h, idx)
            else:
                h = self._mamba_step(seq, h, idx)
            h = mlp(h)
        h = model.norm_f(h)[:, 0]                             # true hidden state
        base = model.lm_head(h); p = torch.softmax(base.float(), dim=-1)
        self._mem_last_h = h
        self._mem_last_exp = p @ E.float()
        # return (B,1,V) to match step() shape (avoids generation-code mistakes)
        return model.lm_head(h + mem.stream_read(h)).unsqueeze(1)

    def _ska_step(self, ska_block, x, idx):
        """One-token SKA step. Reads SKA state from tokens < t (strict prefix),
        produces the output, THEN writes token t into the state for future
        positions -- matching the un-chunked prefix training/prefill semantics
        (verified decode-vs-prefix = 4.8e-5). Writing before reading would let
        the query at t see its own key/value (a one-token causal leak).

        Also applies the parallel short-conv path from a cached local history of
        normed inputs, so generation matches the trained SKABlock.forward.

        Reads the carried factor L (no per-token re-Cholesky), then writes token
        t into the state and applies an O(r^2) rank-1 cholupdate to L by
        w = sqrt(beta) * z_n ("It Cancels").
        """
        B = x.shape[0]
        ska = ska_block.ska
        H, r, P = ska.H, ska.rank, ska.P
        st = self._ska_states[idx]

        h = ska_block.norm(x)
        x, zq_n, vbar = self._proj_norm(ska, h)
        x1 = x[:, 0]; zq1 = zq_n[:, 0]; vbar1 = vbar[:, 0]   # (B,H,*) symmetric key/value

        # --- READ FIRST: carried factor over tokens < t (no re-Cholesky) ---
        N = B * H
        L = st.L.reshape(N, r, r)
        y = _ska_apply_whitened(
            L, st.M.reshape(N, r, r), st.C_v.reshape(N, P, r),
            zq1.reshape(N, r, 1), ska.power_K, ska._resolve_gamma())
        y = y.reshape(B, H, P)

        y_hat = (ska.eta * y).reshape(B, 1, H * P).to(x.dtype)
        output = ska.out_proj(y_hat)
        if ska.layerscale_gate is not None:
            output = output * ska.layerscale_gate
        out = x + output

        # --- parallel short-conv path (matches training block) ---
        if getattr(ska_block, 'short_conv', None) is not None:
            cache = self._conv_caches.get(idx)          # (B, K-1, d) prior normed h
            k = ska_block.short_conv.kernel_size[0]
            if cache is None:
                cache = torch.zeros(B, k - 1, h.shape[-1],
                                    device=x.device, dtype=h.dtype)
            win = torch.cat([cache, h], dim=1)           # (B, K, d)
            c = win.transpose(1, 2)                      # (B, d, K)
            c = ska_block.short_conv(c)[..., -1:]        # last (current) output
            c = c.transpose(1, 2)                        # (B,1,d)
            out = out + c * ska_block.short_conv_gate
            self._conv_caches[idx] = win[:, 1:].detach() # slide cache forward

        # --- WRITE LAST: fold token t into the state for future positions ---
        # Symmetric key x1 = sqrt(beta)*z in BOTH slots: G,C own-weight beta
        # (invariant), M cross-weight sqrt(beta_t beta_{t-1}) (contractive).
        st.G = st.G + torch.einsum('bhr,bhs->bhrs', x1, x1)
        if st.x_last is not None:
            st.M = st.M + torch.einsum('bhr,bhs->bhrs', x1, st.x_last)
        st.C_v = st.C_v + torch.einsum('bhp,bhr->bhpr', vbar1, x1)
        st.x_last = x1
        # rank-1 cholupdate by the SAME symmetric key: w = x1 = sqrt(beta)*z, so
        # w w^T = beta z z^T = the G increment. Carried EXPLICITLY -- do NOT
        # re-derive beta from ||x1|| (that yields beta^{1/4}, desyncing L from G:
        # the train/decode divergence trap).
        from koopman_lm.globals.modules.ska.factor_scan import rank1_chol_update_
        rank1_chol_update_(st.L.reshape(N, r, r), x1.reshape(N, r))
        return out

    def _mamba_step(self, mamba_block, x, idx):
        conv_state, ssm_state = self._mamba_states[idx]
        h_normed = mamba_block.norm(x)
        out, conv_state, ssm_state = mamba_block.mamba.step(
            h_normed, conv_state, ssm_state)
        self._mamba_states[idx] = (conv_state, ssm_state)
        return x + out

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None):
        B = input_ids.shape[0]
        device = input_ids.device
        logits = self.prefill(input_ids)
        generated = input_ids.clone()
        active = torch.ones(B, dtype=torch.bool, device=device)
        next_logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                active &= (next_token.squeeze(-1) != eos_token_id)
                if not active.any():
                    break
            next_logits = self.step(next_token)[:, 0, :]
        return generated

    @torch.no_grad()
    def generate_with_memory(self, input_ids, max_new_tokens=100,
                             temperature=1.0, top_k=None, top_p=None,
                             eos_token_id=None, gen_w=None):
        """Streaming generation WITH the persistent last-layer memory. Prefills
        backbone + memory from the prompt (prompt transitions weight 1), then
        decodes, writing each generated token with provenance weight gen_w
        (default mem.gen_w, typically 0 so self-generated tokens don't
        reinforce). Requires model.attach_last_layer_memory()."""
        mem = self.model.last_layer_memory
        assert mem is not None, "call model.attach_last_layer_memory() first"
        B = input_ids.shape[0]
        device = input_ids.device
        ww = mem.gen_w if gen_w is None else gen_w
        logits = self.prefill_with_memory(input_ids)
        generated = input_ids.clone()
        active = torch.ones(B, dtype=torch.bool, device=device)
        next_logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                active &= (next_token.squeeze(-1) != eos_token_id)
                if not active.any():
                    break
            next_logits = self.step_with_memory(next_token, write_weight=ww)[:, 0, :]
        return generated

    def _sample(self, logits, temperature=1.0, top_k=None, top_p=None):
        if temperature == 0 or (top_k == 1):
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        if top_p is not None:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cum > top_p
            rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False
            idx_rm = rm.scatter(1, si, rm)
            logits[idx_rm] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def state_size_bytes(self, batch_size=1):
        cfg = self.cfg
        total = 0
        d_inner = cfg.d_model * cfg.mamba_expand
        n_mamba = cfg.n_layers - len(cfg.ska_layer_indices)
        total += n_mamba * batch_size * (d_inner * cfg.d_conv + d_inner * cfg.d_state) * 4
        H, r, P = cfg.ska_n_heads, cfg.ska_rank, cfg.head_dim
        n_ska = len(cfg.ska_layer_indices)
        total += n_ska * batch_size * H * (r * r + r * r + P * r + r + r * r) * 4
        return total
