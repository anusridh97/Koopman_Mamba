# `koopman_lm/kernels/` — numerics and CUDA

Everything here is numerical machinery: `torch.autograd.Function`s, triangular
solves, Cholesky updates, prefix scans, and the CUDA sources. No file here
defines an `nn.Module` -- that's why this is a sibling of `modules/` rather
than a child of it. (`fast.py`, the one file that used to define an
`nn.Module` here, moved to `modules/seq/fast.py`.)

## What's in here

| File | Role |
|---|---|
| `lin_alg.py` | shared primitives: `tri_solve_lower/​lowerT`, `whiten_M`, `spec_w`, `inv_sqrt_ns`. Imported by `ska_operator`, `factor_scan`, `incremental_transport`, and `models/recurrent`. |
| `ska_operator.py` | the whitened SKA operator, forward + hand-derived backward |
| `prefix_scan.py` | exact blocked causal prefix scan (PyTorch reference path) |
| `cuda_prefix_scan.py` | strict autograd/JIT wrapper around the fused CUDA kernel |
| `csrc/prefix_scan_ext.cu` | the fused rank-24 forward + analytic backward |
| `csrc/small_rank_ext.cu` | fused small-rank decode kernel |
| `cholesky_update.py`, `cholesky_update_triton.py` | rank-1 Cholesky updates |
| `chunk_stats.py`, `chunk_stats_exact.py` | per-chunk / per-token sufficient statistics |
| `factor_scan.py` | all-prefix Cholesky factors via a PSD square-root scan |
| `inverse_cholesky.py`, `incremental_transport.py`, `fused_state_reference.py` | alternative/reference state formulations |
| `adaptive_chunking.py`, `small_rank_backend.py` | overlap/decay stats (exported but unwired — see its docstring), JIT loader |

**The PyTorch prefix scan is the correctness reference.** The CUDA path must
match it; it is never silently substituted for it.

## The fused CUDA prefix scan

`csrc/prefix_scan_ext.cu` is specialized for the geometry both production
configs use:

- rank `r=24`, value/head width `p=64`, operator power `K=1`
- exact scheduling blocks of 32 tokens, backward checkpoints every 8
- FP32 Cholesky/operator state; BF16 model inputs converted inside the SKA module
- native `[batch, time, head, width]` access, so no head-major full-sequence copies

Forward is an exact blocked prefix scan, not stale chunking:

1. build each 32-token block's raw ΔG, ΔM, ΔC summary
2. exclusive-scan those summaries across blocks
3. reconstruct each block's incoming whitened state in parallel
4. process all 32 tokens exactly with read-before-write O(r²+pr) rank-one updates
5. save compact (P, A, R) checkpoints for the analytic backward

Each CTA packs 1, 2, 4, or 8 independent warp-owned scans depending on
available opt-in shared memory and the number of active states. The rank-24
matrices use a padded row stride of 25 to avoid 32-bank row conflicts.

Backward is a reverse prefix scan over raw-statistic adjoints, recomputing
eight-token intervals from checkpoints. It does not run autograd through
per-token Cholesky factors and never materializes a full prefix trajectory.

**The `cuda_prefix` backend is strict.** A shape, dtype, or build mismatch
raises immediately rather than falling back to Python.

## Building it (B200 / SM100)

Install a CUDA-enabled PyTorch first. Native SM100 needs CUDA 12.8+; a CUDA
12.4 PyTorch environment will not work for the B200 build.

```bash
pip install --no-build-isolation -e '.[cuda,dev]'
bash scripts/build_b200_prefix_scan.sh
```

The build script compiles a native `sm_100` extension, runs deterministic
forward/backward checks against the dense exact oracle, reports maximum
numerical error, and prints the selected states-per-CTA and dynamic
shared-memory sizes.

For register/spill reporting:

```bash
SKA_PREFIX_SCAN_PTXAS_VERBOSE=1 bash scripts/build_b200_prefix_scan.sh
```

Then the CUDA correctness suite (requires a GPU):

```bash
PYTHONPATH=. pytest -q code-tests/test_fused_prefix_scan_cuda.py
```

## Benchmarking

Times the complete SKA branch, projections and backward included:

```bash
# 50M geometry
python scripts/benchmark_prefix_scan.py --batch 2 --length 2048 --d-model 384 --heads 6

# 180M geometry
python scripts/benchmark_prefix_scan.py --batch 1 --length 2048 --d-model 640 --heads 10
```

Prints median step time, peak allocated memory, tokens/sec, and the
exact-vs-chunk-64 time ratio. This is the authoritative performance
measurement on the target GPU.
