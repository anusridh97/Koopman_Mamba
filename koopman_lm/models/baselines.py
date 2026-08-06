
"""
baselines.py -- Baseline model variants for ablation comparison.

  transformer         — Pure causal Transformer, all attention layers + SwiGLU MLP
  mamba_only          — All Mamba-2 + SwiGLU MLP (no global retrieval)
  mamba_attn           — 75% Mamba-2 + 25% Flash Attention + SwiGLU MLP
  mamba_ska_swiglu     — 75% Mamba-2 + 25% SKA + SwiGLU MLP (isolates SKA from the Koopman MLP)
  mamba_ska_koopman    — 75% Mamba-2 + 25% SKA + Spectral Koopman MLP (Sec 3.3's actual "Echo" MLP)
  koopman (full Echo)  — same layout, with the extras (short-conv, last-layer memory,
                          ablation context manager); see models/koopman_lm.py::KoopmanLM

The attention/SKA indices mirror ska_layer_indices from the config so that
mamba_attn, mamba_ska_swiglu, and mamba_ska_koopman all share the same
Mamba/non-Mamba split -- mamba_ska_koopman vs mamba_ska_swiglu is a clean,
single-variable ablation (only the MLP type differs; same SKABlock, same
Mamba2Block). Note: Table 2 (Sec 4.1) itself specifies SwiGLU MLPs for all
three of its sub-million models -- mamba_ska_koopman is not part of that
table's literal protocol, it's for comparing the Koopman MLP's effect on
the same task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from koopman_lm.globals.config import KoopmanLMConfig
from koopman_lm.globals.modules.koopman_mlp import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.globals.modules.mlp import SwiGLUMLP
from koopman_lm.globals.modules.norm import make_norm
from koopman_lm.globals.modules.mamba import Mamba2Block      # noqa: F401 (re-exported)
from koopman_lm.models.koopman_lm import SKABlock, MambaSKAParallelBlock
from koopman_lm.globals.modules.attention import CausalAttentionBlock  # noqa: F401 (re-exported)


# ============================================================================
# Sequence layer blocks
# ============================================================================

# CausalAttentionBlock is defined in globals/modules/attention.py and imported above.


# SKABlock is imported from models/koopman_lm.py so full and baseline
# models use identical statistics, residual initialization, and local-path policy.

# Mamba2Block is defined in globals/modules/mamba.py and imported above.


# ============================================================================
# Generic model builder
# ============================================================================

def _build_model(cfg, seq_layer_fn, mlp_fn):
    """Generic model builder shared by all variants."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            ska_set = set(cfg.ska_layer_indices)
            self.seq_layers = nn.ModuleList()
            self.mlp_layers = nn.ModuleList()
            for i in range(cfg.n_layers):
                self.seq_layers.append(seq_layer_fn(cfg, i, i in ska_set))
                self.mlp_layers.append(mlp_fn(cfg))
            self.norm_f = make_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed.weight
            if cfg.init_policy == 'mamba_safe':
                self._mamba_safe_initialize()

        @staticmethod
        def _scaled_residual_init_(module, denominator):
            weight = getattr(module, 'weight', None)
            if isinstance(weight, torch.Tensor):
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                with torch.no_grad():
                    weight.div_(denominator)

        def _mamba_safe_initialize(self):
            nn.init.normal_(self.embed.weight, mean=0.0, std=cfg.initializer_range)
            if not cfg.tie_embeddings:
                nn.init.normal_(self.lm_head.weight, mean=0.0,
                                std=cfg.initializer_range)
            if not cfg.rescale_prenorm_residual:
                return
            denominator = math.sqrt(2.0 * cfg.n_layers)
            for module in self.modules():
                if isinstance(module, Mamba2Block):
                    out_proj = getattr(module.mamba, 'out_proj', None)
                    if out_proj is not None:
                        self._scaled_residual_init_(out_proj, denominator)
                elif isinstance(module, CausalAttentionBlock):
                    self._scaled_residual_init_(module.proj, denominator)
                elif isinstance(module, SwiGLUMLP):
                    self._scaled_residual_init_(module.w3, denominator)
                elif isinstance(module, (SpectralKoopmanMLP, SpectralKoopmanMLPGated)):
                    self._scaled_residual_init_(module.readout, denominator)

        def no_weight_decay_param_names(self):
            geometric = {'lift_v', 'lift_g', 'A_raw', 's', 'theta', 'gamma',
                         'omega', 'eta', 'eta_raw', 'ssn_gamma',
                         'layerscale_gate', 'short_conv_gate'}
            return {name for name, _ in self.named_parameters()
                    if name.rsplit('.', 1)[-1] in geometric}

        def forward(self, input_ids, labels=None, loss_weights=None):
            """Forward with the same weighted-CE contract as KoopmanLM.

            train.py always passes ``loss_weights`` when the packed dataset has a
            weights stream.  The old baselines did not accept that argument, so a
            supposedly matched Mamba-only run could fail before its first step.
            """
            h = self.embed(input_ids)
            for seq, mlp in zip(self.seq_layers, self.mlp_layers):
                h = seq(h)
                h = mlp(h)
            h = self.norm_f(h)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                if loss_weights is None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                else:
                    ce = F.cross_entropy(
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

        def param_summary(self):
            total = sum(p.numel() for p in self.parameters())
            seq_types = {}
            for layer in self.seq_layers:
                name = type(layer).__name__
                n = sum(p.numel() for p in layer.parameters())
                seq_types[name] = seq_types.get(name, 0) + n
            mlp_total = sum(
                sum(p.numel() for p in m.parameters())
                for m in self.mlp_layers
            )
            print(f"Total parameters: {total:,}")
            for name, count in sorted(seq_types.items()):
                print(f"  {name}: {count:,} "
                      f"({count/total*100:.1f}%)")
            print(f"  MLP layers: {mlp_total:,} "
                  f"({mlp_total/total*100:.1f}%)")
            return total

    return _Model()


# ============================================================================
# Model variants
# ============================================================================

def build_mamba_only(cfg: KoopmanLMConfig):
    """
    Mamba-only baseline: ALL layers are Mamba-2, no global retrieval.
    SwiGLU MLPs throughout. Ignores ska_layer_indices entirely.
    """
    def seq_fn(c, i, is_ska):
        return Mamba2Block(c)  # always Mamba, regardless of index
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand,
                         norm_type=c.norm_type, norm_eps=c.norm_eps)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_mamba_attention(cfg: KoopmanLMConfig):
    """
    Mamba + Flash Attention baseline: 25% attention at the same layer
    indices where Koopman LM places SKA. SwiGLU MLPs throughout.
    """
    def seq_fn(c, i, is_ska):
        return CausalAttentionBlock(c) if is_ska else Mamba2Block(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand,
                         norm_type=c.norm_type, norm_eps=c.norm_eps)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_transformer(cfg: KoopmanLMConfig):
    """
    Pure causal Transformer: all sequence layers are CausalAttentionBlock,
    all feedforward layers are SwiGLU MLP. No SSM, no SKA.
    Parameter count matches the other 50M variants at the same d_model/n_layers.
    """
    def seq_fn(c, i, is_ska):
        return CausalAttentionBlock(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand,
                         norm_type=c.norm_type, norm_eps=c.norm_eps)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_mamba_ska_swiglu(cfg: KoopmanLMConfig):
    """
    Ablation variant: Mamba-2 + SKA + SwiGLU MLP (isolates Koopman MLP
    contribution by using SKA but with standard MLP).
    """
    def seq_fn(c, i, is_ska):
        if not is_ska:
            return Mamba2Block(c)
        return MambaSKAParallelBlock(c) if c.ska_mode == 'parallel' else SKABlock(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand,
                         norm_type=c.norm_type, norm_eps=c.norm_eps)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_mamba_ska_koopman(cfg: KoopmanLMConfig, mlp_expand: float | None = None):
    """
    Mamba-2 + SKA + Spectral Koopman MLP (Sec 3.3) -- same sequence-layer
    layout as build_mamba_ska_swiglu (same SKABlock, same Mamba2Block, same
    ska_layer_indices), only the MLP type differs. Uses SpectralKoopmanMLPGated
    when cfg.mlp_gated, else the plain (ungated) SpectralKoopmanMLP -- matching
    how models/koopman_lm.py::KoopmanLM picks between the two.

    mlp_expand overrides cfg.mlp_expand for this MLP only (the shared backbone
    -- embeddings, Mamba-2, SKA -- is untouched). SpectralKoopmanMLP has 2
    weight matrices (lift, readout) vs SwiGLUMLP's 3 (gate, up, down), so at
    the SAME expand ratio it's structurally ~2/3 the size in the MLP portion
    alone -- at cfg.mlp_expand's default (2.667) that's 0.74M total vs
    build_mamba_ska_swiglu's 933,732, not the matched-budget comparison an
    ablation needs. Pass mlp_expand explicitly (e.g. table2.py's
    --koopman_mlp_expand) to compensate; None keeps cfg.mlp_expand unchanged.
    """
    def seq_fn(c, i, is_ska):
        if not is_ska:
            return Mamba2Block(c)
        return MambaSKAParallelBlock(c) if c.ska_mode == 'parallel' else SKABlock(c)
    def mlp_fn(c):
        MLPClass = SpectralKoopmanMLPGated if c.mlp_gated else SpectralKoopmanMLP
        expand = c.mlp_expand if mlp_expand is None else mlp_expand
        return MLPClass(
            c.d_model, expand=expand,
            spectral_norm_gamma=c.mlp_spectral_norm,
            norm_type=c.norm_type, norm_eps=c.norm_eps,
        )
    return _build_model(cfg, seq_fn, mlp_fn)
