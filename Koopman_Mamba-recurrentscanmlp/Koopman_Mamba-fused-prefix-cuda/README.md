# Koopman-Mamba with fused exact SKA prefix scan

This is the runnable repository for the recommended 50M and 180M models. It keeps Mamba-2 at every depth and adds exact causal SKA as parallel residual memory.

## Recommended models

| Config | Width | Depth | Parallel SKA layers | SKA rank | Parameters |
|---|---:|---:|---:|---:|---:|
| `configs/50m.yaml` | 384 | 17 | 4 | 24 | 50,765,216 |
| `configs/180m.yaml` | 640 | 25 | 6 | 24 | 180,049,352 |

Both use SwiGLU, RMSNorm, fixed \(\eta=\gamma=1\), causal norm clipping, LayerScale, and the strict `cuda_prefix` backend.

## What the CUDA kernel does

The production path is `koopman_lm/globals/modules/ska/csrc/prefix_scan_ext.cu`. It is specialized for the geometry used by both configs:

- rank `r=24`;
- value/head width `p=64`;
- operator power `K=1`;
- exact scheduling blocks of 32 tokens;
- backward checkpoints every 8 tokens;
- FP32 Cholesky/operator state, with BF16 model inputs converted inside the SKA module;
- native `[batch,time,head,width]` tensor access, so the fused path does not make head-major full-sequence copies.

The forward is an exact blocked prefix scan, not stale chunking:

1. Build each 32-token block's raw \(\Delta G,\Delta M,\Delta C\) summary.
2. Exclusive-scan those summaries across blocks.
3. Reconstruct each block's incoming whitened state in parallel.
4. Process all 32 tokens exactly with read-before-write \(O(r^2+pr)\) rank-one updates.
5. Save compact \((P,A,R)\) checkpoints for the analytic backward.

Each CTA packs 1, 2, 4, or 8 independent warp-owned scans according to the
available opt-in shared memory and the number of active states. The rank-24
matrices use a padded row stride of 25 to avoid 32-bank row conflicts.

The backward performs a reverse prefix scan over raw-statistic adjoints, recomputing eight-token intervals from checkpoints. It does not invoke autograd through per-token Cholesky factors and does not materialize a full prefix matrix trajectory.

The strict backend never silently falls back to the Python implementation. A shape, dtype, or build mismatch raises immediately.

## B200 prerequisites

Install a CUDA-enabled PyTorch build first, followed by this repository. For a native B200/SM100 cubin, use a CUDA toolkit and PyTorch build that support compute capability 10.0; CUDA 12.8 or newer is required for native SM100 compilation.

```bash
# Install the CUDA/PyTorch versions appropriate for the machine first.
pip install --no-build-isolation -e '.[cuda,dev]'
```

Do not use a CUDA 12.4 PyTorch environment for the B200 build.

## Compile and validate the fused kernel

Run this once on a B200 compute node before launching distributed training:

```bash
bash scripts/build_b200_prefix_scan.sh
```

The build script:

- compiles a native `sm_100` extension;
- runs deterministic forward and backward checks against the dense exact oracle;
- reports maximum numerical errors;
- prints the selected states-per-CTA and dynamic shared-memory sizes.

For compiler register/spill reporting:

```bash
SKA_PREFIX_SCAN_PTXAS_VERBOSE=1 bash scripts/build_b200_prefix_scan.sh
```

Run the full CUDA correctness suite after compilation:

```bash
PYTHONPATH=. pytest -q code-tests/test_fused_prefix_scan_cuda.py
```

## Benchmark exact scan against chunk 64

The benchmark times the complete SKA branch, including projections and backward:

```bash
python scripts/benchmark_prefix_scan.py \
  --batch 2 --length 2048 --d-model 384 --heads 6
```

For the 180M geometry:

```bash
python scripts/benchmark_prefix_scan.py \
  --batch 1 --length 2048 --d-model 640 --heads 10
```

The script prints median step time, peak allocated memory, tokens per second, and the exact/chunk-64 time ratio. This is the authoritative performance measurement on the target GPU.

## Train

```bash
bash scripts/train_50m.sh
bash scripts/train_180m.sh
```

Useful overrides:

```bash
STEPS=100 PDBS=2 GA=1 RUN_NAME=smoke-50m bash scripts/train_50m.sh
NPROC=8 RUN_NAME=main-180m bash scripts/train_180m.sh
```

The launcher tokenizes FineWeb-Edu when needed, trains, saves checkpoints, and prints evaluation commands. The production launch disables whole-block gradient checkpointing because the fused SKA backward already stores compact eight-token state checkpoints; wrapping the whole Mamba+SKA block would recompute the fused forward and reduce throughput. Data and runs default to `${SCRATCH}/data` and `${SCRATCH}/runs`; set `DATA_ROOT` and `RUN_ROOT` explicitly where appropriate.

## CPU/reference verification

```bash
PYTHONPATH=. pytest -q code-tests \
  --ignore=code-tests/test_fused_prefix_scan_cuda.py
```

The CPU tests cover the inverse-Cholesky formula, asymmetric operator/readout transport, raw-prefix adjoints, block-size invariance, strict causality, recurrent parity, and production parameter counts.

## Verification status of this archive

The recurrence, host integration, Python packaging, and CPU mathematical tests were executed in the packaging environment: **16 tests passed and 10 CUDA-only tests were skipped**. The CUDA device and host translation units also passed Clang CUDA syntax parsing, and a wheel build confirmed that the `.cu` source is packaged.

That environment did not contain a CUDA toolkit, `nvcc`, or an NVIDIA GPU, so the extension has **not** yet been compiled by NVCC or benchmarked on B200. The B200 build script performs that remaining target-machine gate and fails before training if the device, toolkit, forward result, or backward gradients do not match the required contract.
