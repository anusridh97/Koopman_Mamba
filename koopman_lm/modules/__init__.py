"""Layer components. KoopmanLM alternates one of each, per layer.

    seq/    mixes across SEQUENCE positions   -> KoopmanLM.seq_layers
            Mamba2Block, SKABlock, MambaSKAParallelBlock, CausalAttentionBlock

    mlp/    mixes across FEATURES at a single position -> KoopmanLM.mlp_layers
            SwiGLUMLP, SpectralKoopmanMLP, SpectralKoopmanMLPGated

    norm.py make_norm() -- shared by both, and by models/. Not a mixer.

    wip/    NOT on the training path. Reachable only through explicit opt-in
            (e.g. KoopmanLM.forward_with_memory). Treat as experimental.

The rule: if it needs neighbouring tokens, it is a seq mixer. If it works on
one position independently of the others, it is an mlp mixer.

Directory names deliberately match the attribute names on the model, so
`seq_layers` and `seq/` are the same idea under one vocabulary rather than two.

Numerical machinery -- triangular solves, Cholesky updates, prefix scans, the
CUDA kernels -- is NOT here. It lives in koopman_lm/kernels/, a sibling of this
package, because it defines autograd Functions and plain routines rather than
nn.Modules.
"""
