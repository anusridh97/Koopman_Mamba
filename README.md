# Koopman-Mamba

A language model that keeps a **Mamba-2 recurrent backbone at every layer** and
adds **SKA** (Spectral Koopman Attention) as parallel residual memory at a few
intermediate depths. The research question is whether that memory buys anything
over the Mamba backbone alone, so the code is built to swap one component and
hold everything else fixed.

| Config | Width | Depth | SKA layers | SKA rank | Parameters |
|---|---:|---:|---:|---:|---:|
| `configs/50m.yaml` | 384 | 17 | 4 | 24 | 50,034,044 |
| `configs/180m.yaml` | 640 | 25 | 6 | 24 | 176,342,680 |

Both use SwiGLU, RMSNorm, fixed η=γ=1, causal norm clipping, LayerScale, and
the strict `cuda_prefix` backend.

## How it fits together

Five things, in dependency order:

```
configs/*.yaml  ->  KoopmanLMConfig      koopman_lm/config.py
                          |
                          v
                    KoopmanLM            koopman_lm/models/
                          |
              +-----------+-----------+
              v                       v
        seq_layers               mlp_layers      koopman_lm/modules/
        (mix across              (mix across
         positions)               features)
              |                       |
              +-----------+-----------+
                          v
                       kernels          koopman_lm/kernels/
                (solves, scans, CUDA)
```

Every layer is one seq mixer plus one mlp mixer. The 50M puts a
`MambaSKAParallelBlock` at layers 3, 7, 11, 15 and a plain `Mamba2Block`
everywhere else — SKA is added *beside* Mamba, not instead of it:

```
x + Mamba(norm_m(x)) + SKA(norm_s(x))
```

## Layout

```
koopman_lm/       the model package -- imports nothing from experimentation/
  config.py       KoopmanLMConfig, the YAML loader, CONFIG_REGISTRY
  models/         KoopmanLM, RecurrentKoopmanLM (decode), ablation baselines
  modules/        layer components -- seq/ and mlp/ mixers, shared norm
  kernels/        numerics: solves, Cholesky, prefix scans, CUDA   [README]
  diagnostics/    measurement of a built model, kept out of the model itself
experimentation/  everything that runs the model
  run/            one run: resolve a spec, guard it, materialize it, launch it
  sweep/          many runs: the grid is declared exactly once
  training/       the training loop and data pipeline
  evaluation/     perplexity, lm-eval-harness, NIAH, RULER, BABILong, MQAR
  experiments/    Table 2, MQAR fine-tuning, curricula
  retrieval/      contrastive retrieval adaptation
  results.py      walk run dirs, emit one table
configs/          the 11 shipped model configs, plus runs/ and sweeps/
scripts/          launchers, CUDA build/benchmark, dev utilities
code-tests/       test suite
docs/             codebase guide, specs, plans
```

`koopman_lm/modules/__init__.py` explains the seq/mlp split.
`koopman_lm/kernels/README.md` covers the CUDA kernel, its build, and benchmarks.

## Install

Install a CUDA-matched PyTorch **first**, then:

```bash
pip install --no-build-isolation -e '.[cuda,dev]'
bash scripts/build_b200_prefix_scan.sh    # B200/SM100; see kernels/README.md
```

## Train

```bash
bash scripts/train_50m.sh
bash scripts/train_180m.sh
```

Overrides go through the environment:

```bash
STEPS=100 PDBS=2 GA=1 RUN_NAME=smoke-50m bash scripts/train_50m.sh
NPROC=8 RUN_NAME=main-180m bash scripts/train_180m.sh
```

The launcher tokenizes FineWeb-Edu if the shard is missing, trains, saves
checkpoints, and prints the evaluation commands. Data and runs default to
`${SCRATCH}/data` and `${SCRATCH}/runs`; set `DATA_ROOT` and `RUN_ROOT` to
override.

Production runs disable whole-block gradient checkpointing on purpose: the
fused SKA backward already stores compact eight-token checkpoints, so wrapping
the whole Mamba+SKA block would recompute the fused forward for no benefit.

## Test

```bash
PYTHONPATH=. pytest code-tests -q
```

GPU-marked tests auto-skip without a CUDA device, so this runs on a CPU box —
it will report skips, not failures. The full suite needs a GPU.

## More

- `docs/CODEBASE_GUIDE.md` — orientation, the pipeline end to end, what to review
- `koopman_lm/kernels/README.md` — the fused CUDA prefix scan
- `koopman_lm/modules/__init__.py` — the seq/mlp mixer split
