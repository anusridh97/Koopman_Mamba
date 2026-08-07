"""Optional JIT loader for the fused small-rank decode kernel (csrc/small_rank_ext.cu).

The kernel is the B200 inference backend for the inverse-Cholesky
representation: per token it performs, in ONE launch and entirely on-chip,

    u = P z;  Givens coefficients from one block scan of u^2;  in-place
    P <- updated L^{-1};  Aw <- Q^T [Aw 0; 0 0] Q (+ outer-product correction),

i.e. exactly the O(r^2)/token recurrence whose batched training-time analogue
is inverse_cholesky.py (which factors all prefixes at once instead of
recurring). Mapping to SKA decode (v1.1 symmetric sqrt-beta convention):

    z  ->  x_t     = sqrt(beta_t) * z_t   (symmetric key)
    w  ->  x_{t-1}                        (asymmetric path only)
    P  ->  L^{-1}   (lower 64x64 T2 tiles; r <= 64 uses the first tile,
                     row-major: P[row*64 + col])
    Aw ->  L^{-1} M L^{-T} (asymmetric path, dense r x r) or the symmetric
           congruence L^{-1} S L^{-T} (symmetric path, T2 tiles)

The value-map transport R = C_v L^{-T} is NOT part of the kernel; see
incremental_transport.phase3_transport_R for that piece. Requires CUDA and
nvcc; import stays cheap until load_small_rank_ext() is first called.
"""

import os

import torch

_ext = None


def load_small_rank_ext(verbose=False):
    """JIT-compile (once) and return the small_rank_ext module."""
    global _ext
    if _ext is not None:
        return _ext
    if not torch.cuda.is_available():
        raise RuntimeError(
            "small_rank_ext is a CUDA decode backend; no CUDA device available")
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), 'csrc', 'small_rank_ext.cu')
    _ext = load(
        name='ska_small_rank_ext',
        sources=[src],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=verbose,
    )
    return _ext


def init_inverse_state(batch, rank, ridge, device, dtype=torch.float32):
    """(P, Aw) state tensors for r <= 64 in the kernel's single-tile layout.

    P is seeded to L^{-1} = I / sqrt(ridge) (L = sqrt(ridge) I), Aw to zero.
    Each state is one 64x64 tile (4096 floats) regardless of rank; only the
    leading rank x rank block is meaningful.
    """
    assert rank <= 64, "single-tile helper covers r <= 64 only"
    P = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
    idx = torch.arange(rank, device=device)
    P[:, idx, idx] = ridge ** -0.5
    Aw = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
    return P.reshape(batch, 64 * 64), Aw.reshape(batch, 64 * 64)
