import torch.nn as nn
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.channel_mixer.norm import make_norm


class Mamba2Block(nn.Module):
    """Pre-norm Mamba-2 sequence layer. Requires mamba_ssm."""
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm = make_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        # headdim is passed only when the config pins it, so existing configs
        # keep mamba_ssm's own default (64) and their parameter counts.
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
