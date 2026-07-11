import torch.nn as nn
from koopman_lm.config import KoopmanLMConfig


class Mamba2Block(nn.Module):
    """Pre-norm Mamba-2 sequence layer. Requires mamba_ssm."""
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
        )
        self._ablate = False

    def forward(self, x):
        if self._ablate:
            return x
        return x + self.mamba(self.norm(x))


__all__ = ["Mamba2Block"]
