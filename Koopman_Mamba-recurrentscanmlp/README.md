# Koopman-Mamba: exact prefix-scan SKA

This is the pared-down runnable repository for the recommended model setup.

## Recommended models

- `configs/50m.yaml`: 17 Mamba-2 layers, four parallel rank-24 SKA adapters.
- `configs/180m.yaml`: 25 Mamba-2 layers, six parallel rank-24 SKA adapters.

Both use SwiGLU, RMSNorm, exact causal prefix-scan SKA, and O(r^2) rank-one Cholesky updates. Mamba remains present at every depth.

## Install

Install a PyTorch build matching the machine CUDA version first, then:

```bash
pip install -e '.[cuda,dev]'
```

For evaluation through lm-evaluation-harness:

```bash
pip install -e '.[cuda,dev,lmharness]'
```

## Train

```bash
bash scripts/train_50m.sh
bash scripts/train_180m.sh
```

Useful environment overrides:

```bash
STEPS=100 PDBS=2 GA=1 RUN_NAME=smoke-50m bash scripts/train_50m.sh
NPROC=8 RUN_NAME=main-180m bash scripts/train_180m.sh
```

The launcher tokenizes FineWeb-Edu if the requested shard does not already exist, trains, saves checkpoints, and prints evaluation commands.

Default data and run roots are `${SCRATCH}/data` and `${SCRATCH}/runs`; set `DATA_ROOT` and `RUN_ROOT` explicitly when needed.

## Verify

```bash
PYTHONPATH=. pytest -q code-tests
```

## CUDA status

The repository includes the current asymmetric SKA CUDA prototype under `koopman_lm/globals/modules/ska/csrc/`. The exact Torch prefix-scan path is the canonical correctness implementation. The final fully fused packed CUDA forward/backward remains a performance port; it is not silently substituted for the verified recurrence.
