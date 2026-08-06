"""Channel mixers -- the feed-forward (MLP-slot) layers.

Interchangeable occupants of a transformer block's second sublayer: each mixes
information ACROSS channels, per-token. SwiGLUMLP is the standard baseline;
SpectralKoopmanMLP(+Gated) is the paper's spectral Koopman MLP (Sec 3.3).
"""
from koopman_lm.modules.channel_mixer.swiglu import SwiGLUMLP
from koopman_lm.modules.channel_mixer.koopman import (
    SpectralKoopmanMLP,
    SpectralKoopmanMLPGated,
)

__all__ = ["SwiGLUMLP", "SpectralKoopmanMLP", "SpectralKoopmanMLPGated"]
