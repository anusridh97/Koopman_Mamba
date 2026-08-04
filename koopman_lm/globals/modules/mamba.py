import torch.nn as nn
from koopman_lm.globals.config import KoopmanLMConfig


class Mamba2Block(nn.Module):
    """Pre-norm Mamba-2 sequence layer. Requires mamba_ssm."""
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm = nn.LayerNorm(cfg.d_model)
        # headdim is passed only when the config pins it, so existing configs
        # keep mamba_ssm's default (64) and byte-identical behaviour.
        extra = {}
        if getattr(cfg, "mamba_headdim", None) is not None:
            extra["headdim"] = cfg.mamba_headdim
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
            **extra,
        )
        self._ablate = False

    def forward(self, x):
        if self._ablate:
            return x
        return x + self.mamba(self.norm(x))


__all__ = ["Mamba2Block"]
