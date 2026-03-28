import torch
import torch.nn as nn
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.ska import SKAModule
from koopman_lm.koopman_mlp import SpectralKoopmanMLP, SpectralKoopmanMLPGated


class Mamba2Block(nn.Module):
    """
    Wrapper for Mamba-2 layer. Requires mamba_ssm package
    (get from Tri Dao / Albert Gu: https://github.com/state-spaces/mamba).
    """
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

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class SKABlock(nn.Module):
    """SKA layer with pre-norm, matching Nemotron-H attention block interface."""
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
        )

    def forward(self, x):
        return x + self.ska(self.norm(x))


class KoopmanLM(nn.Module):
    """
    180M Nemotron-H style hybrid language model.

    Architecture:
      - Embedding + RMSNorm
      - N blocks, each consisting of:
          - Sequence layer: Mamba-2 (most layers) or SKA (2-3 layers)
          - Feedforward layer: Spectral Koopman MLP
      - Final LayerNorm + LM head

    Layer layout follows Nemotron-H conventions:
      - First layer is always Mamba-2
      - SKA layers are evenly spaced
      - Every sequence layer is followed by a Koopman MLP
    """
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
            ))

        self.norm_f = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

        # Re-apply custom inits for SKA and Koopman MLP layers
        # (the generic _init_weights above tramples them)
        for layer in self.seq_layers:
            if isinstance(layer, SKABlock):
                ska = layer.ska
                nn.init.orthogonal_(ska.key_proj.weight)
                nn.init.orthogonal_(ska.query_proj.weight)
                nn.init.xavier_uniform_(ska.value_proj.weight)
                nn.init.zeros_(ska.out_proj.weight)
        for layer in self.mlp_layers:
            if hasattr(layer, 'lift'):
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

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)

        for seq_layer, mlp_layer in zip(self.seq_layers, self.mlp_layers):
            h = seq_layer(h)
            h = mlp_layer(h)

        h = self.norm_f(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        mamba_p = sum(
            p.numel() for layer in self.seq_layers
            if isinstance(layer, Mamba2Block)
            for p in layer.parameters()
        )
        ska_p = sum(
            p.numel() for layer in self.seq_layers
            if isinstance(layer, SKABlock)
            for p in layer.parameters()
        )
        mlp_p = sum(p.numel() for layer in self.mlp_layers for p in layer.parameters())
        embed_p = self.embed.weight.numel()
        norm_p = sum(p.numel() for p in self.norm_f.parameters())

        print(f"Total parameters: {total:,}")
        print(f"  Embedding:  {embed_p:,} ({100*embed_p/total:.1f}%)")
        print(f"  Mamba-2:    {mamba_p:,} ({100*mamba_p/total:.1f}%)")
        print(f"  SKA:        {ska_p:,} ({100*ska_p/total:.1f}%)")
        print(f"  Koopman MLP:{mlp_p:,} ({100*mlp_p/total:.1f}%)")
        print(f"  Norms:      {norm_p:,}")

        # Print chunk strategy info
        for layer in self.seq_layers:
            if isinstance(layer, SKABlock):
                ska = layer.ska
                print(f"  SKA chunk:  strategy={ska.chunk_strategy}, "
                      f"size={ska.chunk_size}", end="")
                if ska.chunk_strategy == 'overlap':
                    print(f", overlap={ska.overlap_fraction}", end="")
                elif ska.chunk_strategy == 'decay':
                    print(f", alpha={ska.decay_alpha}", end="")
                print()
                break  # all SKA layers share the same config

        return total
