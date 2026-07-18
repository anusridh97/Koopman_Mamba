# Training Echo — 50M / 180M / 440M

One entrypoint, three sizes. This file explains **everything you need to
install**, **how to run**, and **every knob** that matters.

```bash
bash scripts/setup_env.sh                 # one-time: uv sync (torch + CUDA kernels + project)

sbatch scripts/slurm_pretrain.sh 50m      # SLURM: tokenize → train → print eval commands
sbatch --time=96:00:00  scripts/slurm_pretrain.sh 180m
sbatch --time=168:00:00 --gres=gpu:h100:4 scripts/slurm_pretrain.sh 440m   # 4-GPU DDP

# no SLURM? run the same pipeline directly (uv picks the env from pyproject):
uv run --extra cuda bash scripts/pretrain.sh 50m
```

> **Convention note (read once).** This branch trains under the v1.1 `√β`
> symmetrized SKA statistics, which are **provisional** — gated by the §C
> retrain (`scripts/sqrt_beta_ablation/`, see `ECHO_V1_1_PLAN.md`). Training
> here produces `√β` models. To train the pre-`√β` baseline, check out the
> parent ref (the ablation harness does this A/B automatically). Nothing
> behavior-changing beyond `√β` is on by default (the MLP norm-preserving flag
> and the incremental decode kernel are both off/inert — see §Optional knobs).

---

## 1. Install

**Recommended (uv, one command).** `pyproject.toml`'s `[tool.uv]` pins torch to
the CUDA wheel index and builds the CUDA kernels with no build isolation, so the
old torch-first dance is not needed:

```bash
bash scripts/setup_env.sh                      # installs uv if missing, then:
#   uv sync --extra cuda                        # torch + mamba-ssm + causal-conv1d + triton + project
#   uv sync --extra cuda --extra lmharness      # + zero-shot eval harness (later)
```
Then run anything under the env with `uv run` (no manual activation), e.g.
`uv run --extra cuda bash scripts/pretrain.sh 50m` or `uv run pytest -m correctness code-tests/`.

**Plain pip (fallback).** Install torch FIRST for your CUDA, then the extras:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[cuda]'         # mamba-ssm>=2.2.2, causal-conv1d>=1.4.0, triton>=2.2
pip install -e '.[lmharness]'    # zero-shot benchmarks only
```

**Environment / auth:**
- `HF_HOME` — set to a large-disk cache; FineWeb-Edu streams a lot.
- Tokenizer: default is `NousResearch/Llama-2-7b-hf` (ungated, 32k vocab, matches
  the paper's Llama-2 tokenizer). The gated `meta-llama/Llama-2-7b-hf` needs
  `huggingface-cli login`; the mirror avoids that.
- A CUDA toolkit matching your torch build must be on `PATH`/`LD_LIBRARY_PATH`
  for `mamba-ssm`/`causal-conv1d` to compile their kernels.

**Sanity check (CPU, no GPU needed):**
```bash
python -c "import koopman_lm, torch; print('import OK', torch.__version__)"
pytest -m correctness code-tests/          # the NumPy/oracle suite (torch-free ones run anywhere)
```

---

## 2. Hardware & the three recipes

Defaults target **one 80 GB GPU** (H100/A100). The knobs are all env-overridable
(§5). Provenance is marked: **[paper]** = from the Echo paper §3.4/§4.3;
**[eng]** = reasonable engineering default where the paper is silent.

| Size | Params | Tokens | Steps | Seq | Eff. batch (PDBS×GA) | LR | Warmup | Provenance |
|---|---|---|---|---|---|---|---|---|
| `50m`  | ~51M  | 3B   [paper] | 15000 | 2048 | 96  (16×6)  | 6e-4 | 300  | [paper] |
| `180m` | ~191M | 10B  [paper] | 51000 | 2048 | 96  (8×12)  | 6e-4 | 1000 | tokens [paper]; batch/lr [eng] |
| `440m` | ~444M | 20B  [eng]   | 76000 | 2048 | 128 (4×32)  | 4e-4 | 1000 | [eng] — not a paper config; 20B ≈ the paper's ~55 tok/param regime, "more" than 180M |

`steps ≈ tokens / (eff_batch × seq_len)`. **Effective batch = PDBS × GA ×
world_size** — keep the product at the target when you change per-device batch
for VRAM. If you OOM, halve `PDBS` and double `GA` (same eff. batch, more
grad-accum). 440M in particular may need `PDBS=2 GA=64` on 80 GB.

**Multi-GPU (DDP):** launch under `torchrun` and add `--ddp`; effective batch
then multiplies by the GPU count, so divide `GA` accordingly.
```bash
EXTRA_TRAIN_ARGS="--ddp" torchrun --nproc_per_node=8 -m koopman_lm.training.train \
  --model_type koopman --model_size 180m --data_dir "$DATA" ...   # (pretrain.sh wraps the single-GPU path)
```

**Smoke test (minutes, one GPU):**
```bash
STEPS=50 TOKENS=5000000 scripts/pretrain.sh 50m
```

---

## 3. Data

`pretrain.sh` tokenizes automatically (and skips if the shards already exist):
- **train**: pure FineWeb-Edu (`--mix 1.0 0.0 0.0`), `TOKENS` tokens → `$DATA_ROOT/fineweb_<size>_train/`
- **val**: a disjoint 20M-token held-out shard (skips past the docs the train
  shard consumed) → `$DATA_ROOT/fineweb_<size>_val/`

Each shard is `train.bin` (uint16 token ids) + `meta.json` (+ optional
`weights.bin` for recall-weighted CE; unused in these pure-LM runs → CE is a
plain mean). To tokenize manually or build a mixed corpus (FineWeb / PG19 /
SCROLLS), call `python -m koopman_lm.training.data.pretokenize --help`.

---

## 4. Config knobs (`configs/<size>.yaml`)

The model architecture is fixed by the YAML; everything not listed falls back to
the dataclass defaults in `koopman_lm/globals/config.py` (the authoritative,
commented list). The knobs that matter:

**Architecture**
| Key | Meaning |
|---|---|
| `d_model`, `n_layers`, `vocab_size` | model width / depth / tokenizer vocab |
| `d_state`, `d_conv`, `mamba_expand` | Mamba-2 backbone dims |
| `ska_layer_indices` | which layers are SKA (rest are Mamba-2). ⚠️ see the layer-placement note below |
| `mlp_gated` | Koopman MLP gated (SwiGLU-parity) vs plain |

**SKA operator**
| Key | Meaning |
|---|---|
| `ska_n_heads`, `ska_rank` (`r`), `ska_chunk_size` (`S`) | heads / Koopman rank / chunk size |
| `ska_ridge` (`ε`, default 1e-3) | ridge regularization of the Gram matrix |
| `ska_power_K` (**now pinned = 2**) | power-filter order. Paper K=2; **pin it — it is not a learned weight, and an unpinned default drifts silently between train and eval** |
| `ska_backend` | `auto` / `triton` / `pytorch` |
| `ska_exact_intrachunk` | exact per-token stats vs chunked (default chunked) |
| `ska_norm_clip` / `ska_norm_clip_c` | **default off** (per-token ℓ2). On → causal norm-clip `k/max(1,‖k‖/c)`, `c=√rank` if unset. Behavior change with its own before/after eval; **must precede the Gate-2 gate arms** (`ECHO_V1_1_PLAN.md §6`). A/B it by training with a config that sets `ska_norm_clip: true` vs the default |

**SKA scale-parameter policy — differs by size (documented, not a bug):**
| Key | 50m | 180m | 440m |
|---|---|---|---|
| `ska_eta_*` | learnable, squashed [1.4,1.7] | learnable, unbounded (default) | fixed 1.0 |
| `ska_gamma_*` | learnable, squashed [0.5,1.5] | learnable, clamped [1.0,1.5] (default) | fixed 1.0 |
| `ska_layerscale` | off | off | on (init 1e-4) |
| `ska_short_conv` | off | off | on (kernel 4) |

`50m` reproduces the paper/`echo_jax.py`-faithful policy; `440m` is the later
"production rewrite" policy (fixed η/γ, LayerScale residual gate, parallel
short-conv); `180m` uses the dataclass defaults. **These are three different
policies** — deliberate, but it means cross-scale comparisons are not
single-variable. If you want a clean scaling ladder, unify them first (a
retrain-affecting decision, out of scope here).

**Koopman MLP**
| Key | Meaning |
|---|---|
| `mlp_expand` (~8/3) | hidden expansion |
| `mlp_spectral_norm` | eigenvalue clamp on the 2×2 rotations (on) |
| `mlp_norm_preserving` | **default off.** On → exact unit-circle rotation (σ=1), retrain-only (see `ECHO_V1_1_PLAN.md`) |

> **Layer-placement note (audit `A1`).** `ska_layer_indices` in the current
> configs places an SKA layer on the *final* layer for every size except 180M
> (e.g. 50M `{3,7,11,15}` with 16 layers → index 15 is last), which contradicts
> the paper's "first and last layers are always Mamba-2." See
> `PAPER_CODE_AUDIT.md#A1`. This is a **known, retrain-affecting** discrepancy
> left as-is here; fixing it (`{2,6,10,14}` or a builder guard) is a deliberate
> change, not part of wiring up training.

---

## 5. Runtime knobs (`pretrain.sh` env vars + `train.py` flags)

Every `pretrain.sh` default is overridable inline:

```bash
STEPS=1000 PDBS=8 GA=12 LR=3e-4 scripts/pretrain.sh 180m
DATA_ROOT=/data RUN_ROOT=/runs TOKENIZER=meta-llama/Llama-2-7b-hf scripts/pretrain.sh 50m
EXTRA_TRAIN_ARGS="--no_compile --wandb_project echo --deterministic" scripts/pretrain.sh 50m
```

| Env (pretrain.sh) | Default | Meaning |
|---|---|---|
| `TOKENS` / `STEPS` | per-size (§2) | train-shard token budget / optimizer steps |
| `PDBS` / `GA` | per-size | per-device batch / grad-accum (product = eff. batch) |
| `SEQ_LEN` | 2048 | sequence length |
| `LR` / `WARMUP` | per-size | peak LR / linear-warmup steps (then cosine decay) |
| `WEIGHT_DECAY` / `GRAD_CLIP` | 0.1 / 1.0 | AdamW decay / grad-norm clip |
| `SEED` | 42 | RNG + dataloader seed |
| `TOKENIZER` | NousResearch/Llama-2-7b-hf | HF tokenizer id |
| `DATA_ROOT` / `RUN_ROOT` | `$SCRATCH/{data,runs}` | shard / checkpoint roots |
| `NUM_WORKERS` | 4 | dataloader workers |
| `EXTRA_TRAIN_ARGS` | "" | passthrough to `train.py` |

Key `train.py` flags (full list: `python -m koopman_lm.training.train --help`):
`--bf16/--no_bf16` (bf16 autocast, on), `--compile/--no_compile` (torch.compile
the SKA/attention modules, on — turn **off** if compile errors), `--ska_fast`
(fused k/q/v projection; optional, math unchanged), `--gradient_checkpointing`
(on; saves memory on the Mamba layers), `--ddp`, `--wandb_project`,
`--save_steps` (checkpoint cadence), `--deterministic` (reproducible curves,
slower).

---

## 6. Outputs & evaluation

Checkpoints land in `$RUN_ROOT/echo-<size>/`: `step_<N>/` every `save_steps` and
`final/`. Each dir has `model.pt` (state dict), `meta.pt` (**the full resolved
`cfg`** + `cfg_hash` + tokenizer — so K, layer indices, everything is recoverable
from the checkpoint), and the tokenizer files. Eval **prefers the checkpoint's
embedded `cfg`**, falling back to `configs/<size>.yaml` only if `meta.pt` is
absent.

`pretrain.sh` prints the eval commands at the end:
```bash
# held-out FineWeb-Edu + WikiText-103 perplexity
python -m koopman_lm.evaluation.evaluate --checkpoint <ckpt> --model_size <size> --mode fineweb_ppl --held_out_data_dir <val> --output ...
python -m koopman_lm.evaluation.evaluate --checkpoint <ckpt> --model_size <size> --mode ppl --output ...
# zero-shot (needs .[lmharness])
python -m koopman_lm.evaluation.lm_harness_eval --model koopman --model_args checkpoint=<ckpt>,model_size=<size>,max_length=2048 \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai --batch_size 8 --device cuda --output_path ...
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `mamba-ssm` / `causal-conv1d` build fails | CUDA toolkit must match torch's CUDA; install torch first, confirm `nvcc` matches `torch.version.cuda`, then `pip install -e '.[cuda]'` |
| Gated-repo 401 on the tokenizer | use the default `NousResearch/Llama-2-7b-hf` mirror, or `huggingface-cli login` |
| OOM | halve `PDBS`, double `GA` (same eff. batch); ensure `--gradient_checkpointing`; 440M may need `PDBS=2 GA=64` |
| `torch.compile` errors on your stack | `EXTRA_TRAIN_ARGS="--no_compile"` |
| CUDA unavailable | training requires a GPU (Mamba-2 kernels); the CPU path is import/oracle-test only |
| No `train.bin` | tokenization didn't finish — rerun `pretrain.sh` (it resumes/ skips completed shards) |

---

## 8. Optional / off-by-default capabilities

These exist on the branch but do **not** affect a default run:
- **`mlp_norm_preserving`** (config, default off) — exact norm-preserving Koopman
  MLP rotation; retrain-only. `ECHO_V1_1_PLAN.md §2`.
- **Incremental `(L,A,R)` decode kernel** (`incremental_transport.py`,
  standalone, inert) — O(r²) decode; not wired into the live path, promotion
  gated on its torch parity test. `ECHO_V1_1_PLAN.md §3`.
- **`√β` power-iteration removal** — specced, lands after Gate 1.

The single reference for the v1.1 architecture, gates, and their promotion rules
is `ECHO_V1_1_PLAN.md`; the paper↔code discrepancies are in `PAPER_CODE_AUDIT.md`.
