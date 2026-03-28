"""
recurrent.py -- O(1) state recurrent inference wrapper for KoopmanLM.

Wraps a trained KoopmanLM for autoregressive generation with constant
memory in sequence length. Each component maintains fixed-size state:

  Mamba-2:     conv_state [B, d_inner, d_conv] + ssm_state [B, nheads, headdim, d_state]
               (via mamba_ssm's built-in step() method)

  SKA:         G [B, H, r, r]     Gram accumulator
               M [B, H, r, r]     Cross-covariance accumulator
               C_v [B, H, P, r]   Value readout accumulator
               z_last [B, H, r]   Previous key (for boundary M update)
               max_norm [B, H, 1] Running max key norm

  Koopman MLP: (none -- pointwise)

Usage:
  from koopman_lm.recurrent import RecurrentKoopmanLM

  model = KoopmanLM(cfg)
  model.load_state_dict(...)
  wrapper = RecurrentKoopmanLM(model)

  # Prefill: process prompt in parallel (returns logits + initializes state)
  logits = wrapper.prefill(input_ids)  # [B, T, V]

  # Decode: generate one token at a time with O(1) state
  next_logits = wrapper.step(next_token_id)  # [B, 1, V]

  # Or use the convenience method:
  generated_ids = wrapper.generate(input_ids, max_new_tokens=100)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from koopman_lm.model import KoopmanLM, Mamba2Block, SKABlock
from koopman_lm.ska import _spectral_normalize_power_iter, _power_spectral_filter


class SKAState:
    """Fixed-size recurrent state for one SKA layer."""
    __slots__ = ['G', 'M', 'C_v', 'z_last', 'max_norm', 'token_count']

    def __init__(self, B, H, r, P, device, dtype=torch.float32, ridge_eps=1e-3):
        self.G = ridge_eps * torch.eye(r, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, H, r, r).clone()
        self.M = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        self.C_v = torch.zeros(B, H, P, r, device=device, dtype=dtype)
        self.z_last = None  # [B, H, r] set after first token
        self.max_norm = torch.ones(B, 1, H, 1, device=device, dtype=dtype) * 1e-6
        self.token_count = 0

    def size_bytes(self, B, H, r, P):
        """Total state size in bytes (float32)."""
        # G: B*H*r*r, M: B*H*r*r, C_v: B*H*P*r, z_last: B*H*r, max_norm: B*H
        floats = B * H * (r * r + r * r + P * r + r + 1)
        return floats * 4


class RecurrentKoopmanLM(nn.Module):
    """
    O(1) state wrapper for autoregressive generation with KoopmanLM.

    Does NOT modify the underlying model. Wraps it for efficient inference.
    """

    def __init__(self, model: KoopmanLM):
        super().__init__()
        self.model = model
        self.cfg = model.cfg

        # Identify which layers are SKA vs Mamba
        self._ska_indices = set(self.cfg.ska_layer_indices)

        # State containers (initialized on prefill)
        self._ska_states = {}       # layer_idx -> SKAState
        self._mamba_states = {}     # layer_idx -> (conv_state, ssm_state)
        self._initialized = False

    def reset(self):
        """Clear all recurrent state."""
        self._ska_states.clear()
        self._mamba_states.clear()
        self._initialized = False

    @torch.no_grad()
    def prefill(self, input_ids):
        """
        Process a prompt in parallel using the standard forward pass,
        then extract recurrent state from the intermediate computations.

        Args:
            input_ids: [B, T] prompt token ids

        Returns:
            logits: [B, T, V] full logits over the prompt
        """
        self.reset()
        B, T = input_ids.shape
        device = input_ids.device
        model = self.model

        h = model.embed(input_ids)

        for layer_idx, (seq_layer, mlp_layer) in enumerate(
                zip(model.seq_layers, model.mlp_layers)):

            if layer_idx in self._ska_indices:
                # Run parallel forward, but also extract state
                h, ska_state = self._ska_prefill(seq_layer, h, layer_idx)
                self._ska_states[layer_idx] = ska_state
            else:
                # Mamba-2: run parallel forward, extract final state
                h, mamba_state = self._mamba_prefill(seq_layer, h, layer_idx)
                self._mamba_states[layer_idx] = mamba_state

            h = mlp_layer(h)

        h = model.norm_f(h)
        logits = model.lm_head(h)

        self._initialized = True
        return logits

    def _ska_prefill(self, ska_block, x, layer_idx):
        """
        Run SKA parallel forward and extract the cumulative state
        (G, M, C_v, z_last, max_norm) at the end of the sequence.
        """
        B, T, _ = x.shape
        ska = ska_block.ska
        r = ska.rank
        H, P = ska.H, ska.P

        # Compute projections
        h_normed = ska_block.norm(x)
        z = ska.key_proj(h_normed).reshape(B, T, H, r)
        zq = ska.query_proj(h_normed).reshape(B, T, H, r)
        v = ska.value_proj(h_normed).reshape(B, T, H, P)

        ctx = torch.amp.autocast('cuda', enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()

            # Max norm over full sequence
            max_norm = z_f.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_f = z_f / max_norm
            zq_f = zq_f / max_norm

        # Run the standard parallel forward for correct output.
        # NOTE: This recomputes the projections (z, zq, v) that we already
        # computed above. We accept the redundancy because prefill runs once
        # and the alternative (extracting state mid-forward) would require
        # modifying SKAModule's forward path or duplicating its chunked logic.
        output = ska_block(x)

        # Extract cumulative state for recurrent continuation
        state = SKAState(B, H, r, P, x.device, ridge_eps=ska.ridge_eps)
        with ctx:
            z_normed = z.float() / max_norm.clamp(min=1e-6)
            v_f = v.float()

            # Accumulate G = sum_t z_t z_t^T + ridge * I
            # [B, T, H, r] -> [B, H, r, r]
            z_perm = z_normed.permute(0, 2, 1, 3)  # [B, H, T, r]
            state.G = torch.einsum('bhtr,bhts->bhrs', z_perm, z_perm)
            state.G = state.G + ska.ridge_eps * torch.eye(r, device=x.device, dtype=torch.float32)

            # Accumulate M = sum_t z_t z_{t-1}^T
            if T > 1:
                state.M = torch.einsum('bhtr,bhts->bhrs',
                                       z_perm[:, :, 1:], z_perm[:, :, :-1])

            # Accumulate C_v = sum_t v_t z_t^T
            v_perm = v_f.permute(0, 2, 1, 3)  # [B, H, T, P]
            state.C_v = torch.einsum('bhtp,bhtr->bhpr', v_perm, z_perm)

            # Last key and max norm
            state.z_last = z_normed[:, -1]  # [B, H, r]
            state.max_norm = max_norm
            state.token_count = T

        return output, state

    @torch.no_grad()
    def _mamba_prefill(self, mamba_block, x, layer_idx):
        """
        Run Mamba-2 parallel forward and extract the final (conv_state, ssm_state).

        Uses mamba_ssm's InferenceParams mechanism to capture the state.
        The Mamba2 forward checks inference_params and populates state
        when seqlen_offset == 0 (prefill mode).
        """
        B = x.shape[0]
        mamba = mamba_block.mamba

        # Ensure mamba has layer_idx set for InferenceParams keying
        if not hasattr(mamba, 'layer_idx') or mamba.layer_idx is None:
            mamba.layer_idx = layer_idx

        try:
            from mamba_ssm.utils.generation import InferenceParams
            inference_params = InferenceParams(
                max_seqlen=self.cfg.max_seq_len, max_batch_size=B)

            # Allocate cache for this layer
            conv_state, ssm_state = mamba.allocate_inference_cache(
                B, self.cfg.max_seq_len, dtype=x.dtype)
            inference_params.key_value_memory_dict[mamba.layer_idx] = (
                conv_state, ssm_state)

            # Run forward with inference_params -- this populates the state
            # AND returns the correct output in one pass
            h_normed = mamba_block.norm(x)
            mamba_out = mamba(h_normed, inference_params=inference_params)
            output = x + mamba_out

            # Update seqlen_offset so subsequent step() calls route correctly
            inference_params.seqlen_offset = x.shape[1]

            mamba_state = (conv_state.clone(), ssm_state.clone())
        except (ImportError, AttributeError):
            # Fallback: run normal forward, allocate empty state
            output = mamba_block(x)
            conv_state, ssm_state = mamba.allocate_inference_cache(
                B, self.cfg.max_seq_len, dtype=x.dtype)
            mamba_state = (conv_state, ssm_state)

        return output, mamba_state

    @torch.no_grad()
    def step(self, token_id):
        """
        Process a single new token using O(1) recurrent state.

        Args:
            token_id: [B, 1] or [B] next token id

        Returns:
            logits: [B, 1, V] logits for next position
        """
        assert self._initialized, "Call prefill() first"

        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)

        model = self.model
        h = model.embed(token_id)  # [B, 1, d_model]

        for layer_idx, (seq_layer, mlp_layer) in enumerate(
                zip(model.seq_layers, model.mlp_layers)):

            if layer_idx in self._ska_indices:
                h = self._ska_step(seq_layer, h, layer_idx)
            else:
                h = self._mamba_step(seq_layer, h, layer_idx)

            h = mlp_layer(h)

        h = model.norm_f(h)
        logits = model.lm_head(h)
        return logits

    def _ska_step(self, ska_block, x, layer_idx):
        """
        Single-token SKA step with O(1) state update.

        Math:
          1. Project to z_t, zq_t, v_t
          2. Normalize by running max_norm (update if needed)
          3. Update running stats: G += z_t z_t^T, M += z_t z_{t-1}^T, C_v += v_t z_t^T
          4. Solve: A_w = M G^{-1}, B_v = C_v G^{-1}
          5. Spectral normalize A_w
          6. Filter: w = G^{-1} zq, w_f = A_w^K w
          7. Readout: y = B_v G w_f = C_v G^{-1} G A_w^K G^{-1} zq
        """
        B = x.shape[0]
        ska = ska_block.ska
        r = ska.rank
        H, P = ska.H, ska.P
        state = self._ska_states[layer_idx]

        # Project
        h_normed = ska_block.norm(x)  # [B, 1, d_model]
        z = ska.key_proj(h_normed).reshape(B, 1, H, r)
        zq = ska.query_proj(h_normed).reshape(B, 1, H, r)
        v = ska.value_proj(h_normed).reshape(B, 1, H, P)

        ctx = torch.amp.autocast('cuda', enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            z_f = z.float().squeeze(1)      # [B, H, r]
            zq_f = zq.float().squeeze(1)    # [B, H, r]
            v_f = v.float().squeeze(1)      # [B, H, P]

            # Normalize by the max_norm established during prefill.
            # We freeze max_norm rather than updating it: if we grew it here,
            # the new z_n would be on a different scale than the z's already
            # accumulated in G/M/C_v, mixing normalization scales and causing
            # drift. In practice generation tokens rarely exceed the prompt's
            # max key norm, and the ridge regularization absorbs small
            # discrepancies. If this becomes an issue, the correct fix is to
            # rescale all accumulated stats by (old_max/new_max)^2 when the
            # max changes, but that adds complexity for a near-zero gain.
            mn = state.max_norm.squeeze(1)  # [B, H, 1]

            # Normalize
            z_n = z_f / mn
            zq_n = zq_f / mn

            # Update running statistics
            # G += z_t z_t^T
            state.G = state.G + torch.einsum('bhr,bhs->bhrs', z_n, z_n)

            # M += z_t z_{t-1}^T  (boundary transition from previous token)
            if state.z_last is not None:
                state.M = state.M + torch.einsum('bhr,bhs->bhrs', z_n, state.z_last)

            # C_v += v_t z_t^T
            state.C_v = state.C_v + torch.einsum('bhp,bhr->bhpr', v_f, z_n)

            # Store current key for next step's M update
            state.z_last = z_n
            state.token_count += 1

            # Solve for A_w and B_v using accumulated stats
            # G is [B, H, r, r], need to solve per (b, h)
            BH = B * H
            G_flat = state.G.reshape(BH, r, r)
            M_flat = state.M.reshape(BH, r, r)
            Cv_flat = state.C_v.reshape(BH, P, r)
            zq_flat = zq_n.reshape(BH, r, 1)

            # Symmetrize G for Cholesky
            G_sym = 0.5 * (G_flat + G_flat.transpose(-1, -2))

            # Batched Cholesky
            L, info = torch.linalg.cholesky_ex(G_sym)
            # Jitter fallback (same as training path)
            eye_r = torch.eye(r, device=x.device, dtype=torch.float32)
            G_jittered = G_sym + 1e-4 * eye_r
            L_jit, _ = torch.linalg.cholesky_ex(G_jittered)
            needs_fix = (info > 0).unsqueeze(-1).unsqueeze(-1)
            L = torch.where(needs_fix, L_jit, L)

            # A_w = M @ G^{-1}
            Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L)
            A_w = Aw_T.transpose(-1, -2)
            A_w, _ = _spectral_normalize_power_iter(A_w)
            gamma_safe = torch.clamp(ska.ssn_gamma, min=1.0, max=1.5)
            A_w = A_w * gamma_safe

            # B_v = C_v @ G^{-1}
            Bv_T = torch.cholesky_solve(Cv_flat.transpose(-1, -2), L)
            B_v = Bv_T.transpose(-1, -2)

            # w_q = L^{-1} @ zq
            w_q = torch.linalg.solve_triangular(L, zq_flat, upper=False)

            # Spectral filter + project back + readout
            w_f = _power_spectral_filter(A_w, w_q, ska.power_K)
            z_out = L @ w_f
            y_flat = B_v @ z_out  # [BH, P, 1]

            y_hat = y_flat.reshape(B, H, P).unsqueeze(1)  # [B, 1, H, P]

        y_hat = ska.eta * y_hat.to(x.dtype)
        output = ska.out_proj(y_hat.reshape(B, 1, H * P))

        return x + output

    def _mamba_step(self, mamba_block, x, layer_idx):
        """
        Single-token Mamba-2 step using its built-in recurrent path.
        """
        conv_state, ssm_state = self._mamba_states[layer_idx]

        h_normed = mamba_block.norm(x)  # [B, 1, d_model] -- keep seq dim

        out, conv_state, ssm_state = mamba_block.mamba.step(
            h_normed, conv_state, ssm_state)

        self._mamba_states[layer_idx] = (conv_state, ssm_state)

        return x + out

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None):
        """
        Autoregressive generation with O(1) state per step.

        Args:
            input_ids:      [B, T] prompt token ids
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = greedy with top_k=1)
            top_k:          top-k sampling (None = disabled)
            top_p:          nucleus sampling threshold (None = disabled)
            eos_token_id:   stop on this token (None = generate all max_new_tokens)

        Returns:
            [B, T + max_new_tokens] generated token ids (prompt + continuation)
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Prefill: process prompt in parallel
        logits = self.prefill(input_ids)

        generated = input_ids.clone()
        active = torch.ones(B, dtype=torch.bool, device=device)

        # Get first next-token from prefill logits
        next_logits = logits[:, -1, :]  # [B, V]

        for _ in range(max_new_tokens):
            # Sample or greedy
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            if eos_token_id is not None:
                active &= (next_token.squeeze(-1) != eos_token_id)
                if not active.any():
                    break

            # Step: O(1) state update
            step_logits = self.step(next_token)
            next_logits = step_logits[:, 0, :]

        return generated

    def _sample(self, logits, temperature=1.0, top_k=None, top_p=None):
        """Sample next token from logits."""
        if temperature == 0 or (top_k == 1):
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def state_size_bytes(self, batch_size=1):
        """
        Compute total recurrent state size in bytes.
        This is O(1) in sequence length -- constant regardless of T.
        """
        cfg = self.cfg
        total = 0

        # Mamba-2 layers
        d_inner = cfg.d_model * cfg.mamba_expand
        n_mamba = cfg.n_layers - len(cfg.ska_layer_indices)
        # conv_state: [B, d_inner, d_conv], ssm_state: [B, nheads, headdim, d_state]
        # nheads and headdim depend on mamba_ssm internals; approximate:
        mamba_per_layer = d_inner * cfg.d_conv + d_inner * cfg.d_state
        total += n_mamba * batch_size * mamba_per_layer * 4  # float32

        # SKA layers
        H = cfg.ska_n_heads
        r = cfg.ska_rank
        P = cfg.head_dim
        n_ska = len(cfg.ska_layer_indices)
        ska_per_layer = H * (r * r + r * r + P * r + r + 1)
        total += n_ska * batch_size * ska_per_layer * 4

        return total

    def print_state_summary(self, batch_size=1):
        """Print a summary of the O(1) recurrent state."""
        cfg = self.cfg
        H = cfg.ska_n_heads
        r = cfg.ska_rank
        P = cfg.head_dim
        d_inner = cfg.d_model * cfg.mamba_expand
        n_mamba = cfg.n_layers - len(cfg.ska_layer_indices)
        n_ska = len(cfg.ska_layer_indices)

        mamba_per = d_inner * cfg.d_conv + d_inner * cfg.d_state
        ska_per = H * (r * r + r * r + P * r + r + 1)

        total = self.state_size_bytes(batch_size)
        print(f"\nRecurrent state summary (B={batch_size}):")
        print(f"  Mamba-2: {n_mamba} layers x {mamba_per:,} floats = "
              f"{n_mamba * mamba_per * 4 / 1024:.1f} KB")
        print(f"  SKA:     {n_ska} layers x {ska_per:,} floats = "
              f"{n_ska * ska_per * 4 / 1024:.1f} KB")
        print(f"  Total:   {total / 1024:.1f} KB = {total / (1024*1024):.3f} MB")
        print(f"  (constant regardless of sequence length)")
