# Koopman_Mamba — codebase guide

Written 2026-08-06, alongside the `reorg-module-layout` branch. Everything
below was verified by running it, not inferred from source. Where something is
unverified, it says so.

---

## 1. What this repo is

A language model that keeps a **Mamba-2 recurrent backbone at every layer** and
adds **SKA (Spectral Koopman Attention)** adapters at a few intermediate depths.
The research question is whether the SKA memory buys anything over the Mamba
backbone alone — so the code is built to swap that one component and hold
everything else fixed.

Two production sizes ship: **50M** and **180M**.

---

## 2. Layout after the reorg

```
koopman_lm/
  config.py              KoopmanLMConfig dataclass + YAML loader + CONFIG_REGISTRY
  models/
    koopman_lm.py        KoopmanLM — assembles the layer stack. SKABlock lives here too.
    recurrent.py         RecurrentKoopmanLM — step-by-step decode path
    baselines.py         ablation arms (mamba_only, mamba_attn, transformer, ...)
  modules/
    token_mixer/         things that mix across SEQUENCE: mamba.py, ska.py, attention.py
    channel_mixer/       things that mix across FEATURES: mlp.py (SwiGLUMLP),
                         koopman.py, koopman_diag.py, norm.py
    kernels/             numerical machinery, no nn.Module opinions:
                         prefix_scan.py, cuda_prefix_scan.py, cholesky_update*.py,
                         chunk_stats*.py, ska_operator.py, factor_scan.py,
                         inverse_cholesky.py, csrc/*.cu
    wip/                 not wired into training: memory.py (LastLayerRidgeMemory)
  training/
    train.py             the training loop (entry point)
    repro.py             seeding / determinism
    data/                pretokenize.py, dataset.py, mix.py
  evaluation/            evaluate.py, lm_harness_eval.py, niah_quick.py, ruler.py,
                         babilong.py, harness.py, mqar/
  experiments/           table2.py, mqar_finetune.py, curricula.py
  retrieval/             adapt.py, data.py, encoder.py
configs/                 50m.yaml, 180m.yaml (+ _prefix_scan aliases)
scripts/                 pretrain.sh, train_50m.sh, train_180m.sh, build_*, benchmark_*
code-tests/              the test suite
```

**The organizing idea:** `modules/` is split by *role*, not by technology.
A token mixer mixes across sequence positions; a channel mixer mixes across
features; kernels are numerics with no module opinions. If you're adding a new
attention variant it goes in `token_mixer/`; a new MLP goes in `channel_mixer/`;
a new Cholesky routine goes in `kernels/`.

`wip/` is a real signal, not a dumping ground: nothing in it is on the training
path. `memory.py` is reachable only through `forward_with_memory`.

---

## 3. The pipeline, end to end

### What actually runs

```
scripts/train_50m.sh
  └─ sets TORCH_CUDA_ARCH_LIST=10.0, SKA_REQUIRE_B200=1
  └─ exec scripts/pretrain.sh 50m_prefix_scan
       │
       ├─ 1. defaults for this size
       │     TOKENS=3e9  STEPS=15000  PDBS=16  GA=6  LR=4e-4  WARMUP=300
       │     SEQ_LEN=2048  eff_batch = PDBS × GA = 96
       │
       ├─ 2. build + validate the fused CUDA prefix scan   (BUILD_PREFIX_SCAN=1)
       │     python scripts/build_prefix_scan_cuda.py
       │     ...done once in the parent process, so DDP ranks don't each rebuild it
       │
       ├─ 3. tokenize FineWeb-Edu if the shard is absent
       │     python -m koopman_lm.training.data.pretokenize
       │       --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT
       │       --tokenizer NousResearch/Llama-2-7b-hf   (32k vocab, ungated mirror)
       │     -> $DATA_ROOT/fineweb_50m_quality_train/train.bin
       │     val shard skips past the train docs so there's no overlap
       │
       └─ 4. train
             python -m koopman_lm.training.train \
               --model_type koopman --model_size 50m_prefix_scan \
               --data_dir .../train --max_seq_len 2048 \
               --per_device_train_batch_size 16 --gradient_accumulation_steps 6 \
               --max_steps 15000 --learning_rate 4e-4 --warmup_steps 300 \
               --bf16 --no_compile --no_gradient_checkpointing \
               --output_dir $RUN_ROOT/echo-50m_prefix_scan
             -> $RUN_DIR/final/model.pt

  NPROC>1 swaps `python -m` for `torchrun --nproc_per_node=$NPROC` and adds --ddp,
  dividing GA so effective batch stays constant.
```

### Then evaluation (pretrain.sh prints these at the end)

```bash
python -m koopman_lm.evaluation.evaluate --checkpoint $CKPT --model_size 50m \
    --mode fineweb_ppl --held_out_data_dir $VAL_DIR --output $RUN_DIR/fineweb_ppl.json

python -m koopman_lm.evaluation.lm_harness_eval --model koopman \
    --model_args checkpoint=$CKPT,model_size=50m,max_length=2048 \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --batch_size 8 --device cuda
```

### Config → model

`--model_size 50m` is resolved by `koopman_lm/config.py`:

```
"50m" -> CONFIG_REGISTRY -> configs/50m.yaml -> KoopmanLMConfig -> KoopmanLM(cfg)
```

The registry has exactly 4 entries and `configs/` ships exactly 4 files — they
match. (They did not always: before the reorg the registry advertised 24 names,
20 of which pointed at YAML that was never shipped.)

### What the 50M actually builds

Verified by loading the real config:

```
d_model=384  n_layers=17  vocab=32000
ska_layer_indices=(3, 7, 11, 15)  ska_mode=parallel  ska_rank=24
mlp_type=swiglu  norm_type=rmsnorm  ska_backend=cuda_prefix

layer  0: Mamba2Block            + SwiGLUMLP
layer  1: Mamba2Block            + SwiGLUMLP
layer  2: Mamba2Block            + SwiGLUMLP
layer  3: MambaSKAParallelBlock  + SwiGLUMLP     <-- SKA
layer  4..6: Mamba2Block         + SwiGLUMLP
layer  7: MambaSKAParallelBlock  + SwiGLUMLP     <-- SKA
layer  8..10: Mamba2Block        + SwiGLUMLP
layer 11: MambaSKAParallelBlock  + SwiGLUMLP     <-- SKA
layer 12..14: Mamba2Block        + SwiGLUMLP
layer 15: MambaSKAParallelBlock  + SwiGLUMLP     <-- SKA
layer 16: Mamba2Block            + SwiGLUMLP
```

`ska_mode: parallel` is why it's `MambaSKAParallelBlock` and not `SKABlock`:
the SKA is added *beside* Mamba, not *instead of* it —

```
x + Mamba(norm_m(x)) + SKA(norm_s(x))
```

An earlier layout replaced the Mamba block at each SKA index; the parallel form
keeps local recurrence at every depth. That distinction matters when reading
ablation results.

---

## 4. How to run things

```bash
# tests (CPU box, no GPU needed)
PYTHONPATH=. pytest code-tests -q          # -> 17 passed, 10 skipped

# the 10 skips are GPU-marked and auto-skip when torch.cuda.is_available() is False.
# To actually run them you need a GPU node.

# install (GPU box; install a CUDA-matched torch FIRST)
pip install -e '.[cuda,dev]'
bash scripts/build_b200_prefix_scan.sh

# smoke a tiny training run
STEPS=100 PDBS=2 GA=1 RUN_NAME=smoke-50m bash scripts/train_50m.sh
```

**CPU-only verification env used for this branch** (not committed):
torch 2.13.0+cpu, pytest, numpy, pyyaml. Enough to import the package, load
configs, and run the 17 non-GPU tests. Not enough for `mamba_ssm`, `triton`,
or `transformers` — 7 modules can't import without those, unchanged before and
after the reorg.

---

## 5. What to review, in priority order

The refactor was mostly mechanical and machine-verified. These are the spots
where a human made a judgment call, so they're where review pays.

### Highest value — judgment, not mechanism

**a) `cb7680d` — the doubled-segment import fix.**
The old tree had a package `ska/` containing a module `ska.py`. My rewrite map
had a key for the package path but not the deeper module path, so a prefix match
stranded a trailing segment: `token_mixer.ska.ska`. One file was affected
(`scripts/benchmark_prefix_scan.py`). Worth confirming there's no sibling case I
missed. The map now has both keys, and the static checker was widened to scan
`scripts/` (it previously only walked `koopman_lm/`, which is how this reached a
commit).

**b) `84b1cbc` — the two hand-fixes.** Neither could be automated.

- `modules/kernels/cholesky_update.py:128` — was
  `from koopman_lm import cholesky_update_triton`. It's inside
  `except Exception`, so a wrong path fails **silently** and permanently
  disables the Triton kernel. No test can catch this class of bug. Verify by
  inspection.
- `config.py:306` — `_CONFIGS_ROOT` lost one `.parent` because `config.py` moved
  up a directory. Off-by-one here means every config load fails.

**c) The `wip/` placement.** `last_layer_memory.py` → `modules/wip/memory.py`
follows `pr/module-reorg`'s intent, but it's a naming judgment. If that code is
closer to production than "wip" suggests, say so.

### Medium — verify the claims rather than the code

I ran these; re-run them if you want independent confirmation:

```bash
# every rename byte-identical
git diff --name-status -M b2d965e HEAD | grep "^R" | grep -v "^R100"     # expect: empty

# nothing outside the package touched
git diff --stat b2d965e HEAD -- MQAR echo-ska lowrank_residual_cuda 'sys+toolcall'

# history survived the moves
git log --follow --oneline koopman_lm/modules/token_mixer/ska.py
```

I also diffed every public symbol between `b2d965e` and this branch:
**0 of 236 lost, 0 definition lines moved, 0 modules without a counterpart.**
41 signature diffs were purely the annotation path being renamed
(`globals.config.KoopmanLMConfig` → `config.KoopmanLMConfig`); the 42nd was
`dataclasses.field`'s repr containing a memory address.

### Low — cosmetic, deliberately not fixed

Stale comments still saying `globals/modules/...` at `config.py:117`,
`experiments/curricula.py:196`, `models/baselines.py:40,46`. Non-executable.
Left alone to keep the diff honest; trivial to sweep separately.

---

## 6. What is NOT verified

Be skeptical of anything in this category — no GPU was available.

- **All 10 GPU-marked tests.** They skipped, they did not pass.
- **The Mamba-2 backbone.** `mamba_ssm` never imported, so `KoopmanLM` was never
  actually instantiated — only its config and layer plan were checked.
- **The fused CUDA prefix scan** (`csrc/prefix_scan_ext.cu`) and the Triton
  Cholesky kernel. Import *paths* are correct; the kernels never compiled or ran.
- **Any training or eval run.** No checkpoint was produced or loaded.

Before merging, the honest gate is a GPU node running:

```bash
PYTHONPATH=. pytest code-tests -q          # expect 27 passed, 0 skipped
STEPS=100 PDBS=2 GA=1 RUN_NAME=smoke bash scripts/train_50m.sh
```

---

## 7. Repo history worth knowing

Three reorganizations happened in quick succession; commit archaeology is
confusing without this.

1. **PR #10 (`recurrentscanswiglu`)** consolidated `echo-ska-440m/` and
   `koopman-lm-fast/` into a single root-level `koopman_lm/`. It also deleted 44
   unrelated files as collateral.
2. **`restore-deleted-folders` (`ebcf17d`)** put back the 22 that nothing
   replaced: `MQAR/`, `lowrank_residual_cuda/`, `echo-ska/`,
   `sys+toolcall/SKAv9.py`. The other 22 stayed deleted as genuinely superseded.
3. **This branch** applies `pr/module-reorg`'s layout without merging it — a
   trial merge produced 72 conflicted files, because both branches independently
   reorganized the same ancestor tree.

**`pr/module-reorg` and `pr/eval-consolidation` are superseded but must not be
deleted yet.** `pr/eval-consolidation` = `pr/module-reorg` + 4 eval-dedup commits
that exist nowhere else:

```
0aa3664  Extract the one canonical load_model into evaluation/loader.py
17824a5  Extract shared _greedy_generate into evaluation/generation.py
dd48a0f  Unify the NIAH generator + scorers into evaluation/tasks/niah.py
34fc331  Dedup lm_harness_eval.py's checkpoint-loading via evaluation/loader.py
```

Those touch `koopman_lm/evaluation/`, which this reorg does not move, so they
should replay with far less friction than the module commits would have. Close
the PRs now; delete the branches only after those four are cherry-picked.
