import contextlib

import torch
import torch.nn as nn
from koopman_lm.globals.config import KoopmanLMConfig
from koopman_lm.globals.modules.ska import SKAModule
from koopman_lm.globals.modules.koopman_mlp import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.globals.modules.mamba import Mamba2Block  # noqa: F401 (re-exported)


class SKABlock(nn.Module):
    """SKA layer with pre-norm, matching Nemotron-H attention block interface.

    Optional parallel short-range path: a depthwise CAUSAL conv on the normed
    input, summed into the residual alongside SKA. Covers the short-range band
    that chunked SKA stats discard (within-chunk cross-covariance), so SKA's
    gradient isn't poisoned by short-range failures. See config.ska_short_conv.
    """
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ska = SKAModule(
            d_model=cfg.d_model,
            n_heads=cfg.ska_n_heads,
            rank=cfg.ska_rank,
            head_dim=cfg.head_dim,
            ridge_eps=cfg.ska_ridge,
            scale=cfg.ska_scale,
            power_K=cfg.ska_power_K,
            chunk_size=cfg.ska_chunk_size,
            backend=cfg.ska_backend,
            chunk_strategy=cfg.ska_chunk_strategy,
            overlap_fraction=cfg.ska_overlap_fraction,
            decay_alpha=cfg.ska_decay_alpha,
            # --- new scale-parameter + residual policy ---
            eta_learnable=cfg.ska_eta_learnable,
            eta_value=cfg.ska_eta_value,
            eta_bounds=getattr(cfg, 'ska_eta_bounds', None),
            gamma_learnable=cfg.ska_gamma_learnable,
            gamma_value=cfg.ska_gamma_value,
            gamma_clamp=cfg.ska_gamma_clamp,
            gamma_bounds=getattr(cfg, 'ska_gamma_bounds', None),
            layerscale=cfg.ska_layerscale,
            layerscale_init=cfg.ska_layerscale_init,
            out_proj_std=cfg.ska_out_proj_std,
            exact_intrachunk=getattr(cfg, 'ska_exact_intrachunk', False),
            # causal norm-clip (memo §6): None -> L2 (legacy). Resolve c=sqrt(rank)
            # when the flag is on and no explicit threshold is given.
            norm_clip_c=((cfg.ska_norm_clip_c or (cfg.ska_rank ** 0.5))
                         if getattr(cfg, 'ska_norm_clip', False) else None),
        )
        # Parallel short-range causal depthwise conv (covers within-chunk band).
        self.short_conv = None
        if getattr(cfg, 'ska_short_conv', False):
            k = cfg.ska_short_conv_kernel
            self.short_conv_pad = k - 1                     # left-pad => causal
            self.short_conv = nn.Conv1d(
                cfg.d_model, cfg.d_model, kernel_size=k,
                groups=cfg.d_model, bias=True)              # depthwise
            # Lag-biased init (current + lag-1 + lag-2), gated small. Exposes
            # local HISTORY from step 0 -- the band chunked SKA discards --
            # rather than mostly the current token. Weights sum ~1 so the gated
            # path is a gentle local average at init.
            nn.init.zeros_(self.short_conv.weight)
            with torch.no_grad():
                if k >= 3:
                    self.short_conv.weight[:, 0, -1] = 0.50   # current token
                    self.short_conv.weight[:, 0, -2] = 0.35   # lag-1
                    self.short_conv.weight[:, 0, -3] = 0.15   # lag-2
                else:
                    self.short_conv.weight[:, 0, -1] = 1.0
            nn.init.zeros_(self.short_conv.bias)
            # ...BUT gate the whole path by a small learnable per-channel scale,
            # so at init the conv contributes ~gate_init * LayerNorm(x), NOT a
            # full-strength extra residual. Otherwise the block would start as
            # x + h + tiny_ska (a free normalized residual injected at every SKA
            # layer), which confounds the "did restoring local evidence help?"
            # ablation. Gate is learnable so the local path grows as needed.
            self.short_conv_gate = nn.Parameter(
                torch.full((cfg.d_model,),
                           float(getattr(cfg, 'ska_short_conv_gate_init', 1e-2))))
        self._ablate = False   # see KoopmanLM.ablate(): zero this layer's contribution

    def forward(self, x):
        if self._ablate:
            return x            # SKA-zeroed: pure residual passthrough (no SKA, no conv)
        h = self.norm(x)
        out = x + self.ska(h)
        if self.short_conv is not None:
            # (B,T,d) -> (B,d,T), left-pad for causality, conv, trim, back, gate
            c = h.transpose(1, 2)
            c = torch.nn.functional.pad(c, (self.short_conv_pad, 0))
            c = self.short_conv(c)[..., :h.shape[1]]
            out = out + c.transpose(1, 2) * self.short_conv_gate
        return out


class KoopmanLM(nn.Module):
    """180M/370M/440M Nemotron-H style hybrid language model."""
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        ska_set = set(cfg.ska_layer_indices)
        self.seq_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        MLPClass = SpectralKoopmanMLPGated if cfg.mlp_gated else SpectralKoopmanMLP

        for i in range(cfg.n_layers):
            if i in ska_set:
                self.seq_layers.append(SKABlock(cfg))
            else:
                self.seq_layers.append(Mamba2Block(cfg))
            self.mlp_layers.append(MLPClass(
                d=cfg.d_model,
                expand=cfg.mlp_expand,
                spectral_norm_gamma=cfg.mlp_spectral_norm,
                norm_preserving=getattr(cfg, 'mlp_norm_preserving', False),
                # --- Koopman MLP utilization / structure options (v2) ---
                rotation_param=getattr(cfg, 'mlp_rotation_param', None),
                row_norm_lift=getattr(cfg, 'mlp_row_norm_lift', False),
                pair_mixer=getattr(cfg, 'mlp_pair_mixer', None),
                mixer_block=getattr(cfg, 'mlp_mixer_block', 64),
                depth_grade=getattr(cfg, 'mlp_decay_depth_grade', False),
                layer_idx=i,
                n_layers=cfg.n_layers,
            ))

        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Optional INFERENCE-ONLY last-layer ridge memory (BOM-LM-v0). Off by
        # default and NOT part of training/forward -- attach explicitly for an
        # ablation via attach_last_layer_memory(). The frozen model with/without
        # it is the A/B (same checkpoint, no retraining).
        self.last_layer_memory = None

        self.apply(self._init_weights)

        # Re-apply custom inits for SKA and Koopman MLP layers.
        # NOTE (440M rewrite): we deliberately do NOT zero-init ska.out_proj
        # anymore. The exact-zero residual init created a gradient stall (SKA
        # internals get no gradient until out_proj grows). LayerScale now
        # provides the near-zero start (out_proj is full-rank small-std, gated
        # by ska.layerscale_gate ~ 1e-4), so internals receive gradient from
        # step 1. The SKAModule constructor already sets these inits; we only
        # re-assert the projection inits that the generic _init_weights above
        # tramples, and leave out_proj + layerscale_gate as the module set them.
        for layer in self.seq_layers:
            if isinstance(layer, SKABlock):
                ska = layer.ska
                nn.init.orthogonal_(ska.key_proj.weight)
                nn.init.orthogonal_(ska.query_proj.weight)
                nn.init.xavier_uniform_(ska.value_proj.weight)
                nn.init.zeros_(ska.beta_proj.weight)      # beta=sigmoid(0)=0.5 at init
                nn.init.zeros_(ska.beta_proj.bias)
                # out_proj init depends on residual policy:
                #   layerscale (440M): full-rank small-std, gated near-zero by
                #     layerscale_gate -> SKA internals get gradient from step 1.
                #   no layerscale (baselines): exact-zero -- the PUBLISHED
                #     behavior. Doing normal_ unconditionally silently broke the
                #     baselines (full-strength random SKA residual).
                if cfg.ska_layerscale:
                    nn.init.normal_(ska.out_proj.weight, mean=0.0,
                                    std=cfg.ska_out_proj_std)
                else:
                    nn.init.zeros_(ska.out_proj.weight)
                # layerscale_gate is left exactly as constructed (small const).
        for layer in self.mlp_layers:
            # The MLP owns its projection init (row-norm lift, rotation, and mixer
            # params are not nn.Linear, so _init_weights leaves them alone; only
            # the plain Linears -- readout, and lift/gate on the legacy path --
            # need xavier restored after the generic normal(0, 0.02) pass).
            if hasattr(layer, 'reset_projection_params'):
                layer.reset_projection_params()
            elif hasattr(layer, 'lift'):
                nn.init.xavier_uniform_(layer.lift.weight)
                nn.init.xavier_uniform_(layer.readout.weight)
                if hasattr(layer, 'gate'):
                    nn.init.xavier_uniform_(layer.gate.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, loss_weights=None):
        """loss_weights: optional [B, T] per-token weight (recall weighting).
        When provided, the CE loss is a weighted mean over non-ignored tokens.
        """
        h = self.embed(input_ids)
        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h)
            h = mlp_layer(h)
        h = self.norm_f(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            if loss_weights is None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                ce = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='none',
                ).view_as(labels)
                w = loss_weights.to(ce.dtype)
                valid = (labels != -100).to(ce.dtype)
                w = w * valid
                loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        return {"loss": loss, "logits": logits}

    @contextlib.contextmanager
    def ablate(self, zero_ska=False, zero_mamba=False):
        """Temporarily zero the contribution of SKA and/or Mamba sequence layers.

        Enables the eval harness's SKA-zeroed PPL delta and the Phase-1 four-mode
        load-bearing decomposition (full / SKA-zeroed / Mamba-zeroed / both):

            with model.ablate(zero_ska=True):
                ppl_no_ska = eval_held_out_ppl(model, ...)

        A zeroed block becomes a pure residual passthrough (output == input).
        Flags are always restored on exit, even if the body raises.
        """
        ska = [l for l in self.seq_layers if isinstance(l, SKABlock)]
        mamba = [l for l in self.seq_layers if isinstance(l, Mamba2Block)]
        try:
            for b in ska:
                b._ablate = zero_ska
            for b in mamba:
                b._ablate = zero_mamba
            yield self
        finally:
            for b in ska + mamba:
                b._ablate = False

    def attach_last_layer_memory(self, rank=64, ridge=1e-2, gate_init=1e-2,
                                 proj='slice', gen_token_weight=0.0):
        """Attach the inference-only last-layer ridge memory (BOM-LM-v0).
        Backbone stays frozen; this only adds a session-scoped causal readout.
        """
        from koopman_lm.globals.modules.utils.last_layer_memory import LastLayerRidgeMemory
        self.last_layer_memory = LastLayerRidgeMemory(
            self.cfg.d_model, rank=rank, ridge=ridge, gate_init=gate_init,
            proj=proj, gen_token_weight=gen_token_weight).to(self.embed.weight.device)
        # reset onto the model's device so a direct mem.read(h) before any
        # forward_with_memory() call doesn't hit a CPU/GPU mismatch.
        self.last_layer_memory.reset(batch=1, device=self.embed.weight.device,
                                     dtype=torch.float32)
        return self.last_layer_memory

    @torch.no_grad()
    def forward_with_memory(self, input_ids, token_weights=None,
                            prompt_len=None, gen_w=None):
        """Inference forward applying the last-layer ridge memory with strict
        causal write-after-read. Teacher-forced over the whole sequence (memory
        is reset at the start; for persistent decode use prefill/step_with_memory).

        token_weights: (B,T) provenance weights. If None and prompt_len is given,
        built via make_memory_token_weights (prompt writes=1, generated=gen_w).
        If None and prompt_len is None, defaults to all-ones (controlled-eval
        only -- NOT safe for prompt+generation, where generated tokens would get
        weight 1 and self-reinforce).
        """
        assert self.last_layer_memory is not None, "call attach_last_layer_memory() first"
        mem = self.last_layer_memory
        B, T = input_ids.shape
        dev = input_ids.device
        mem.reset(batch=B, device=dev, dtype=torch.float32)

        h = self.embed(input_ids)
        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h); h = mlp_layer(h)
        h = self.norm_f(h)                                   # (B,T,d)

        E = self.lm_head.weight                              # output-space basis
        # Vectorized base logits / expected embeddings (computed once, not per
        # step in the Python loop -- avoids T full-vocab softmaxes).
        base_logits = self.lm_head(h)                        # (B,T,V)
        p = torch.softmax(base_logits.float(), dim=-1)       # (B,T,V)
        exp_emb_all = p @ E.float()                          # (B,T,d)

        logits_out = torch.empty(B, T, E.shape[0], device=dev, dtype=h.dtype)
        if token_weights is None:
            if prompt_len is not None:
                from koopman_lm.globals.modules.utils.last_layer_memory import make_memory_token_weights
                token_weights = make_memory_token_weights(
                    input_ids, prompt_len=prompt_len,
                    gen_w=mem.gen_w if gen_w is None else gen_w)
            else:
                token_weights = torch.ones(B, T, device=dev)

        for t in range(T):
            ht = h[:, t]                                     # (B,d)
            delta = mem.read(ht)                             # uses stats over <t
            logits_out[:, t] = self.lm_head(ht + delta)
            if t + 1 < T:                                    # write AFTER read
                v_t = E[input_ids[:, t + 1]] - exp_emb_all[:, t]
                mem.write(ht, v_t, weight=token_weights[:, t])
        return {"logits": logits_out}

    @torch.no_grad()
    def prefill_memory(self, prompt_ids):
        """Streaming-memory prefill (last-layer adapter only).

        WARNING: this prefills the LAST-LAYER MEMORY over the prompt, but does
        NOT capture the backbone's recurrent (Mamba/SKA) decode state. It pairs
        only with step_memory_only_debug(), which re-encodes each new token as a
        length-1 sequence with NO carried backbone state -- so generated-token
        hidden states differ from a full-prefix forward. For CORRECT streaming
        generation, the backbone must be driven by RecurrentKoopmanLM
        (recurrent.py); that wiring is NOT done yet (see CHANGES s17). Until
        then, use forward_with_memory(prefix+token, prompt_len=..) for
        correctness.

        Writes prompt transitions (weight 1) and persists the memory. Returns
        prompt logits.
        """
        assert self.last_layer_memory is not None, "call attach_last_layer_memory() first"
        mem = self.last_layer_memory
        B, T = prompt_ids.shape
        assert T > 0, "prefill_memory requires at least one prompt token"
        dev = prompt_ids.device
        mem.stream_reset(batch=B, device=dev, dtype=torch.float32)
        h = self.embed(prompt_ids)
        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h); h = mlp_layer(h)
        h = self.norm_f(h)
        E = self.lm_head.weight
        base = self.lm_head(h); p = torch.softmax(base.float(), dim=-1)
        exp_emb = p @ E.float()
        logits = torch.empty(B, T, E.shape[0], device=dev, dtype=h.dtype)
        for t in range(T):
            ht = h[:, t]
            logits[:, t] = self.lm_head(ht + mem.stream_read(ht))
            if t + 1 < T:
                mem.stream_write(ht, E[prompt_ids[:, t + 1]] - exp_emb[:, t], weight=1.0)
        # cache last hidden so the first step() can write the prompt->gen transition
        self._stream_last_h = h[:, -1]
        self._stream_last_exp = exp_emb[:, -1] if T > 0 else None
        return {"logits": logits}

    @torch.no_grad()
    def step_memory_only_debug(self, next_input_id, write_weight=None):
        """DEBUG/ADAPTER-ONLY single step -- NOT a valid generation step.

        This re-encodes next_input_id as a length-1 sequence through the full
        Mamba/SKA/MLP stack with NO carried recurrent backbone state. The
        last-layer memory persists correctly, but the BACKBONE does not: the
        hidden state for the generated token is NOT what a full-prefix forward
        would produce, because Mamba/SKA recurrent state is dropped each call.

        It is kept only to exercise the streaming memory adapter in isolation.
        For correct streaming generation, drive the backbone via
        RecurrentKoopmanLM (recurrent.py) and feed its per-token hidden state to
        mem.stream_read/stream_write -- see CHANGES s17. The gating parity test
        is: forward_with_memory(prefix+tok) last-token logits ==
        prefill+step logits.
        """
        assert self.last_layer_memory is not None, "call attach_last_layer_memory() first"
        mem = self.last_layer_memory
        assert hasattr(mem, '_L'), "call prefill_memory() (or mem.stream_reset()) first"
        ww = mem.gen_w if write_weight is None else write_weight
        E = self.lm_head.weight
        # write the pending transition: previous hidden -> embedding-residual of
        # the token we just saw (provenance-weighted).
        if getattr(self, '_stream_last_h', None) is not None:
            v = E[next_input_id] - self._stream_last_exp
            mem.stream_write(self._stream_last_h, v, weight=ww)
        # now encode the new token and read
        h = self.embed(next_input_id.unsqueeze(1))
        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h); h = mlp_layer(h)
        h = self.norm_f(h)[:, 0]
        base = self.lm_head(h); p = torch.softmax(base.float(), dim=-1)
        self._stream_last_h = h
        self._stream_last_exp = p @ E.float()
        return {"logits": self.lm_head(h + mem.stream_read(h))}

    def no_weight_decay_param_names(self):
        """Parameter names (as in named_parameters) that MUST skip weight decay.

        The WeightNorm lift direction ``lift_v`` is scale-invariant -- decaying it
        drives ``||v|| -> 0`` and the 1/||v|| gradient in the normalized weight
        blows up. The per-row gain ``lift_g`` is the utilization knob: decaying it
        toward 0 would actively push neurons dead, the opposite of the intent.
        The rotation decay logit ``s`` and the Cayley mixer params ``A_raw`` are
        geometric parameterizations, not linear weights; decay would pull them to
        arbitrary reference points (rho -> 0.5, mix -> identity). Empty set unless
        the v2 MLP options are enabled, so v1 training is byte-for-byte unchanged.
        """
        skip = set()
        for name, _ in self.named_parameters():
            leaf = name.rsplit('.', 1)[-1]
            if leaf in ('lift_v', 'lift_g', 's', 'A_raw'):
                skip.add(name)
        return skip

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        mamba_p = sum(p.numel() for layer in self.seq_layers
                      if isinstance(layer, Mamba2Block) for p in layer.parameters())
        ska_p = sum(p.numel() for layer in self.seq_layers
                    if isinstance(layer, SKABlock) for p in layer.parameters())
        mlp_p = sum(p.numel() for layer in self.mlp_layers for p in layer.parameters())
        embed_p = self.embed.weight.numel()
        norm_p = sum(p.numel() for p in self.norm_f.parameters())
        print(f"Total parameters: {total:,}")
        print(f"  Embedding:  {embed_p:,} ({100*embed_p/total:.1f}%)")
        print(f"  Mamba-2:    {mamba_p:,} ({100*mamba_p/total:.1f}%)")
        print(f"  SKA:        {ska_p:,} ({100*ska_p/total:.1f}%)")
        print(f"  Koopman MLP:{mlp_p:,} ({100*mlp_p/total:.1f}%)")
        print(f"  Norms:      {norm_p:,}")
        for layer in self.seq_layers:
            if isinstance(layer, SKABlock):
                ska = layer.ska
                ls = None if ska.layerscale_gate is None else float(ska.layerscale_gate.mean())
                with torch.no_grad():
                    eta_val = float(ska._resolve_eta())
                print(f"  SKA: eta={eta_val:.4f}, gamma_learnable={ska.gamma_learnable}, "
                      f"eta_bounds={ska.eta_bounds}, gamma_bounds={ska.gamma_bounds}, "
                      f"layerscale~{ls}, chunk={ska.chunk_size}/{ska.chunk_strategy}")
                break
        return total
