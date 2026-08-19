import contextlib
import math

import torch
import torch.nn as nn
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.seq.ska import SKAModule
from koopman_lm.modules.mlp.koopman import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.modules.mlp.swiglu import SwiGLUMLP
from koopman_lm.modules.norm import make_norm
from koopman_lm.modules.seq.mamba import Mamba2Block
from koopman_lm.modules.seq.ska_block import SKABlock, MambaSKAParallelBlock


class KoopmanLM(nn.Module):
    """Nemotron-H style hybrid: Mamba-2 at every layer, SKA as parallel memory.

    Each layer is one seq mixer (modules/seq/) plus one mlp mixer
    (modules/mlp/). cfg.ska_layer_indices selects which depths get SKA, and
    cfg.ska_mode picks whether SKA is added beside Mamba (parallel) or
    replaces it. Production sizes are 50M and 180M -- see configs/.
    """
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        ska_set = set(cfg.ska_layer_indices)
        self.seq_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        for i in range(cfg.n_layers):
            if i in ska_set:
                if cfg.ska_mode == 'parallel':
                    self.seq_layers.append(MambaSKAParallelBlock(cfg))
                else:
                    self.seq_layers.append(SKABlock(cfg))
            else:
                self.seq_layers.append(Mamba2Block(cfg))
            self.mlp_layers.append(self._build_mlp(cfg, i))

        self.norm_f = make_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Optional INFERENCE-ONLY last-layer ridge memory (BOM-LM-v0). Off by
        # default and NOT part of training/forward -- attach explicitly for an
        # ablation via attach_last_layer_memory(). The frozen model with/without
        # it is the A/B (same checkpoint, no retraining).
        self.last_layer_memory = None

        if cfg.init_policy == 'legacy':
            self._legacy_initialize()
        else:
            self._mamba_safe_initialize()

    def _build_mlp(self, cfg: KoopmanLMConfig, layer_idx: int) -> nn.Module:
        """Build one feed-forward residual block without hidden config coupling."""
        kind = cfg.resolved_mlp_type
        if kind == 'swiglu':
            return SwiGLUMLP(
                d=cfg.d_model,
                expand=cfg.mlp_expand,
                norm_type=cfg.norm_type,
                norm_eps=cfg.norm_eps,
            )
        if kind not in {'koopman', 'koopman_gated'}:
            raise ValueError(f"unsupported mlp_type={kind!r}")
        cls = SpectralKoopmanMLPGated if kind == 'koopman_gated' else SpectralKoopmanMLP
        return cls(
            d=cfg.d_model,
            expand=cfg.mlp_expand,
            spectral_norm_gamma=cfg.mlp_spectral_norm,
            norm_preserving=cfg.mlp_norm_preserving,
            rotation_param=cfg.mlp_rotation_param,
            row_norm_lift=cfg.mlp_row_norm_lift,
            pair_mixer=cfg.mlp_pair_mixer,
            mixer_block=cfg.mlp_mixer_block,
            depth_grade=cfg.mlp_decay_depth_grade,
            layer_idx=layer_idx,
            n_layers=cfg.n_layers,
            norm_type=cfg.norm_type,
            norm_eps=cfg.norm_eps,
            # getattr for the same reason ska_block.py uses it: a checkpoint's
            # embedded config predates this field and must still build. None is a
            # strict no-op.
            precision=getattr(cfg, 'mlp_precision', None),
        )

    def _init_weights(self, module):
        """Legacy blanket initialization retained for exact reproduction only."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def _legacy_initialize(self) -> None:
        """Reproduce the repository's original blanket initialization policy.

        This intentionally descends into Mamba-2 and overwrites its constructor
        initialization.  It is kept so old YAMLs/checkpoints remain a faithful
        control; new quality configs use ``mamba_safe`` instead.
        """
        self.apply(self._init_weights)

        # Restore the custom projection policies that the generic Linear pass
        # overwrote.  Iterate recursively so this also works in parallel mode.
        for block in (m for m in self.modules() if isinstance(m, SKABlock)):
            ska = block.ska
            nn.init.orthogonal_(ska.key_proj.weight)
            nn.init.orthogonal_(ska.query_proj.weight)
            nn.init.xavier_uniform_(ska.value_proj.weight)
            nn.init.zeros_(ska.beta_proj.weight)
            if ska.beta_proj.bias is not None:
                nn.init.zeros_(ska.beta_proj.bias)
            if self.cfg.ska_layerscale:
                nn.init.normal_(ska.out_proj.weight, mean=0.0,
                                std=self.cfg.ska_out_proj_std)
            else:
                nn.init.zeros_(ska.out_proj.weight)

        for layer in self.mlp_layers:
            if hasattr(layer, 'reset_projection_params'):
                layer.reset_projection_params()

    @staticmethod
    def _scaled_residual_init_(module: nn.Module, denominator: float) -> None:
        """Reinitialize a residual-output matrix, then depth-scale it.

        This mirrors the official Mamba/GPT-2 policy.  Reinitializing matters:
        merely dividing the constructor value makes repeated model-building or
        nested initialization order affect the distribution.
        """
        weight = getattr(module, 'weight', None)
        if isinstance(weight, torch.Tensor):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            with torch.no_grad():
                weight.div_(denominator)

    def _mamba_safe_initialize(self) -> None:
        """Preserve branch-specific constructor initialization.

        Mamba-2 initializes several state-space parameters specially.  A global
        ``model.apply`` destroys those choices and also removes the usual
        depth-aware scaling of residual-output projections.  Here only the token
        embeddings (and an untied LM head) are initialized globally; all nested
        Mamba, SKA, and MLP modules keep their own initialization.
        """
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.initializer_range)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0,
                            std=self.cfg.initializer_range)

        if not self.cfg.rescale_prenorm_residual:
            return

        # One sequence residual and one FFN residual are accumulated per depth.
        # This mirrors the GPT/Mamba residual-path rule: scale only each branch's
        # final projection, not every matrix in the branch.
        denominator = math.sqrt(2.0 * self.cfg.n_layers)
        for module in self.modules():
            if isinstance(module, Mamba2Block):
                out_proj = getattr(module.mamba, 'out_proj', None)
                if out_proj is not None:
                    self._scaled_residual_init_(out_proj, denominator)
            elif isinstance(module, SwiGLUMLP):
                self._scaled_residual_init_(module.w3, denominator)
            elif isinstance(module, (SpectralKoopmanMLP, SpectralKoopmanMLPGated)):
                self._scaled_residual_init_(module.readout, denominator)

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

    def encode(self, input_ids, attention_mask=None, pool="mean"):
        """Pool the backbone to one (B, d) embedding for dense retrieval.

        Runs embed -> seq/mlp layers -> norm_f (NO lm_head) and pools the
        per-token hidden states. Independent of forward(), so LM training/eval
        is byte-for-byte unchanged. The Phase-2 retrieval adapter
        (experimentation/retrieval) wraps this with a projection head + L2 norm.

        The backbone is causal, so RIGHT-padding is safe: pad positions sit after
        the real tokens and cannot leak into their hidden states. Pass the
        attention_mask so pooling ignores pad positions.

        pool: 'mean' -> mask-weighted mean over real tokens (robust default).
              'last' -> hidden at the last real token (the causal summary slot).
        """
        from koopman_lm.pooling import pool_sequence
        h = self.embed(input_ids)
        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h)
            h = mlp_layer(h)
        h = self.norm_f(h)                                   # (B, T, d)
        return pool_sequence(h, attention_mask, pool)

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
        candidates = list(self.modules()) if hasattr(self, 'modules') else list(self.seq_layers)
        ska = [m for m in candidates if isinstance(m, SKABlock)]
        mamba = [m for m in candidates if isinstance(m, Mamba2Block)]
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
        from koopman_lm.modules.wip.memory import LastLayerRidgeMemory
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
                from koopman_lm.modules.wip.memory import make_memory_token_weights
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
        """Higher-rank Koopman-v2 parameters that must skip AdamW decay.

        The trainer separately applies generic no-decay rules to biases, norms,
        embeddings, Mamba state parameters, LayerScale, and scalar geometry.
        Keeping this method narrow preserves the old public/repro contract.
        """
        skip = set()
        for name, _ in self.named_parameters():
            leaf = name.rsplit('.', 1)[-1]
            if leaf in ('lift_v', 'lift_g', 's', 'A_raw'):
                skip.add(name)
        return skip

    @staticmethod
    def _unique_parameter_count(modules) -> int:
        seen = set()
        total = 0
        for module in modules:
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
        return total

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        mamba_blocks = [m for m in self.modules() if isinstance(m, Mamba2Block)]
        ska_blocks = [m for m in self.modules() if isinstance(m, SKABlock)]
        mamba_p = self._unique_parameter_count(mamba_blocks)
        ska_p = self._unique_parameter_count(ska_blocks)
        mlp_p = self._unique_parameter_count(self.mlp_layers)
        embed_p = self.embed.weight.numel()
        norm_p = sum(p.numel() for p in self.norm_f.parameters())
        print(f"Total parameters: {total:,}")
        print(f"  Embedding:   {embed_p:,} ({100*embed_p/total:.1f}%)")
        print(f"  Mamba-2:     {mamba_p:,} ({100*mamba_p/total:.1f}%)")
        print(f"  SKA:         {ska_p:,} ({100*ska_p/total:.1f}%)")
        print(f"  Feed-forward:{mlp_p:,} ({100*mlp_p/total:.1f}%) [{self.cfg.resolved_mlp_type}]")
        print(f"  Final norm:  {norm_p:,}")
        if ska_blocks:
            ska = ska_blocks[0].ska
            ls = None if ska.layerscale_gate is None else float(ska.layerscale_gate.mean())
            with torch.no_grad():
                eta_val = float(ska._resolve_eta())
            print(f"  SKA: eta={eta_val:.4f}, gamma_learnable={ska.gamma_learnable}, "
                  f"eta_bounds={ska.eta_bounds}, gamma_bounds={ska.gamma_bounds}, "
                  f"layerscale~{ls}, chunk={ska.chunk_size}/{ska.chunk_strategy}, "
                  f"mode={self.cfg.ska_mode}")
        return total
