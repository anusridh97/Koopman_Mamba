"""SKA sequence-mixing blocks.

SKABlock wraps SKAModule with a pre-norm residual, matching the Nemotron-H
attention-block interface. MambaSKAParallelBlock runs a Mamba-2 local
recurrence and the SKA memory side by side:

    x + Mamba(norm_m(x)) + SKA(norm_s(x))

An earlier layout REPLACED the Mamba block at each SKA index; the parallel
form keeps local recurrence at every depth. cfg.ska_mode selects between them.

These live here rather than in models/ because they are layer components, the
same as their siblings Mamba2Block and CausalAttentionBlock.
"""
import torch
import torch.nn as nn

from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.norm import make_norm
from koopman_lm.modules.seq.mamba import Mamba2Block
from koopman_lm.modules.seq.ska import SKAModule

__all__ = ["SKABlock", "MambaSKAParallelBlock"]


class SKABlock(nn.Module):
    """SKA layer with pre-norm, matching Nemotron-H attention block interface.

    Optional parallel short-range path: a depthwise CAUSAL conv on the normed
    input, summed into the residual alongside SKA. Covers the short-range band
    that chunked SKA stats discard (within-chunk cross-covariance), so SKA's
    gradient isn't poisoned by short-range failures. See config.ska_short_conv.
    """
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.norm = make_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
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
            inverse_cholesky=getattr(cfg, 'ska_inverse_cholesky', False),
            prefix_scan=getattr(cfg, 'ska_prefix_scan', False),
            prefix_scan_block_size=getattr(cfg, 'ska_prefix_scan_block_size', 32),
            prefix_scan_jitter=getattr(cfg, 'ska_prefix_scan_jitter', 0.0),
            # causal norm-clip (memo §6): None -> L2 (legacy). Resolve c=sqrt(rank)
            # when the flag is on and no explicit threshold is given.
            norm_clip_c=((cfg.ska_norm_clip_c or (cfg.ska_rank ** 0.5))
                         if getattr(cfg, 'ska_norm_clip', False) else None),
            # getattr for the same reason as the fields above it: a checkpoint's
            # embedded config predates this field and must still build.
            precision=getattr(cfg, 'ska_precision', 'fp32'),
            beta_policy=getattr(cfg, 'ska_beta_policy', 'learned'),
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


class MambaSKAParallelBlock(nn.Module):
    """Mamba local recurrence plus a sparse SKA memory residual.

    The original layout *replaced* a Mamba block at every SKA index.  That
    removes local sequence-mixing depth exactly where the global-memory branch
    is inserted.  In parallel mode both branches read the same residual stream:

        x + Mamba(norm_m(x)) + SKA(norm_s(x)).

    Reusing the existing residual wrappers gives ``mamba(x) + ska(x) - x``.
    """

    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.mamba = Mamba2Block(cfg)
        self.ska = SKABlock(cfg)

    def forward(self, x):
        return self.mamba(x) + self.ska(x) - x
