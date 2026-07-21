# Overview — continued pretraining, retrieval adaptation, masking & tokenization

A single entry point that summarizes the design, what is trained on, how tokens
are produced, how masking works (including the prefix-mask question), and the
exact commands to run each stage. Deep dives live in `TRAINING.md` (pretraining
mechanics) and `RETRIEVAL.md` (the two-phase recipe); this file is the map.

- **Model**: Koopman-Mamba ("Echo") hybrid — Mamba-2 backbone with interleaved
  SKA (Structured Kernel Attention) layers and a spectral Koopman MLP. Default
  scale here is `180m_v2` (d_model 768, 24 layers, SKA at layers {8, 16},
  Aurora-style v2 MLP).
- **What we added**: a continued-pretraining pipeline (Phase 1) and a retrieval
  adaptation stage (Phase 2), on top of the existing pretraining/eval code.

---

## 1. The plan in one paragraph

Start from the best checkpoint of the 10B-token FineWeb-Edu run (best by
zero-shot **benchmark**, not lowest perplexity). **Phase 1** continues
pretraining for ~1–2B tokens on a broadened 4-bucket mix (web + code/math +
reasoning + retrieval-evidence), causal-LM loss only, as a weights-only warm
start with a fresh, reduced-LR cosine schedule. **Phase 2** then runs a short
(5k-step) contrastive retrieval adaptation that turns the model into a dense
dual-encoder — short query/passage lengths (no long contexts), InfoNCE, and a
small LM anchor so retrieval gains don't damage general LM quality.

---

## 2. Masking — including the "prefix mask" question

There are three *different* masking concepts in this repo. Keeping them distinct
matters:

### 2a. Prefix mask (prefix-LM) — exists, but only in the archived benchmark
`archive/MQAR/prefix_bench/prefix_masks.py` defines a full prefix-mask taxonomy
used to compare SKA vs. softmax attention vs. Mamba on associative recall:

- **SKA modes**: `none`, `prefix`, `soft`, `causal` — a prefix mask here is a
  per-token weight `w_t` on the streaming sufficient statistics (context tokens
  fit the operator, query tokens only read it).
- **Attention modes**: `causal`, `prefix_lm` — `prefix_lm` makes the context
  block bidirectionally visible to all positions while the query/suffix stays
  causal (`build_prefix_lm_bias`).

**This machinery is confined to the MQAR benchmark.** It is **not** wired into
production LM training, and it is **not** used by the NIAH evaluation
(`niah_quick.py` calls `model(input_ids)` with no mask/segment argument). So:
the codebase *has* a prefix mask, but neither the pretraining we run here nor the
needle-in-a-haystack eval uses one.

### 2b. Chunk-causal masking — what the real LM actually uses
The production SKA layer (`koopman_lm/globals/modules/ska/ska.py`) is
**strictly chunk-causal**: sufficient statistics are accumulated with an
exclusive prefix-sum over fixed 64-token chunks, so a query in chunk *c* reads an
operator fit on chunks `0..c-1`. This is the streaming / language-modeling regime
(the taxonomy's `causal` mode) — every token is both a context contributor for
later chunks and a query of earlier ones. No supervised context/query boundary is
ever handed to the model in Phase 1 or Phase 2 LM training.

### 2c. Padding attention mask — what the Phase-2 encoder uses
The retrieval encoder pools token hidden states into one embedding. Because the
backbone is causal, sequences are **right-padded** and an `attention_mask` marks
real vs. pad tokens; pooling (masked-mean by default, `last` optional) ignores
pad positions. This is a *padding* mask for pooling — **not** a prefix-LM mask.
The dual-encoder encodes query and passage independently, so there is no
cross-sequence attention to mask at all.

**Summary**: prefix-LM masking is available (archived benchmark) but unused by
our training/eval; the LM runs chunk-causal; the retrieval encoder uses only a
padding mask.

---

## 3. Design choices (and why)

| Choice | What | Why |
|---|---|---|
| Weights-only warm start | Phase 1 loads model weights, discards optimizer/scheduler/Adam moments; fresh cosine, reduced peak LR (~½–⅓ of base) | Standard when the data mixture changes; a full resume would keep a schedule/moment state tuned to the old distribution |
| Best checkpoint by benchmark | Pick base (and continued) checkpoint via `eval_sweep.sh`, not lowest PPL | FineWeb-Edu runs often keep improving val PPL while zero-shot regresses |
| 4-bucket Phase-1 mix | web / code+math / reasoning / retrieval-evidence | Re-diversify the distribution after 10B FineWeb-Edu tokens without discarding its strengths |
| Retrieval-LM bucket = QA evidence corpora | Wikipedia + HotpotQA + MuSiQue evidence as plain LM text (`qa_context`) | Preserve/strengthen retrieval-relevant representations under LM loss before the contrastive phase |
| Dual-encoder + InfoNCE (Phase 2) | Short query/passage encoded independently, cosine similarity, temp 0.05 | Directly optimizes the embedding space; short lengths avoid long-context memory cost (GPU-constrained) |
| LM anchor | `L = L_retrieval + 0.1·L_LM`, or 4:1 retrieval:LM batch schedule | Prevents retrieval adaptation from degrading general LM quality |
| Projection head w/ separate LR | Backbone LR 8e-6, projection LR 1e-4 | The backbone is already capable (nudge gently); the fresh head must learn the retrieval space fast |
| Hard negatives from structure | Supporting paragraph = positive; same-article distractors = hard negatives; + in-batch | Genuinely hard, same-topic negatives with no extra mining infra; avoids false negatives |
| Pooling = masked-mean | `POOL=last` available | Robust default for a decoder-as-encoder |
| NQ off by default | Full `natural_questions` is heavy; its evidence is Wikipedia-derived | Wikipedia already covers it; `nq=…` can be added to `SOURCES` explicitly |

---

## 4. What we train on (data)

### Phase 1 — continued pretraining (causal LM)
Mix specified as `--sources name=frac …`, renormalized to sum 1
(`koopman_lm/training/data/mix.py`):

| bucket | share | source(s) (HF) |
|---|---|---|
| FineWeb-Edu | 40% | `HuggingFaceFW/fineweb-edu` (`fineweb`) |
| code / math | 25% | `bigcode/starcoderdata` (`code`, 12.5%) + `open-web-math/open-web-math` (`math`, 12.5%) |
| structured QA / reasoning | 20% | `HuggingFaceTB/cosmopedia` (`cosmopedia`) |
| retrieval-oriented LM | 15% | `wikimedia/wikipedia` 9% + `hotpot_qa` 3% + `dgslibisey/MuSiQue` 3% |

The retrieval bucket's QA records are flattened to "question + evidence/distractor
paragraphs" plain text (`kind=qa_context`). `nq` (`natural_questions`) and
`scrolls` (`tau/scrolls`) are registered sources but off this default mix.

> **Gating**: `bigcode/starcoderdata` and `hotpot_qa` are gated / need
> `trust_remote_code` (accept terms + `huggingface-cli login`); StarCoder is
> language-partitioned (`--starcoder_data_dir python`). All HF coordinates are
> overridable via `--*_path` flags or `SOURCES`.

### Phase 2 — retrieval adaptation (contrastive)
Contrastive `(query, positive, hard-negatives)` pairs (`SOURCES`, renormalized):

| source | share | positive / hard-negative |
|---|---|---|
| HotpotQA (distractor) | 35% | supporting paragraph / same-question distractor paragraphs |
| Wikipedia (self-supervised) | 45% | title→section, section→nearby, first-sentence→paragraph, heading→subsection / other sections of same article |
| MuSiQue | 20% | is_supporting paragraph / non-supporting paragraphs |

Wikipedia's 45% subsumes NQ evidence. In-batch negatives (every other query's
positive) are added by the loss.

---

## 5. Tokenization

- **Tokenizer**: `NousResearch/Llama-2-7b-hf` (ungated Llama-2 mirror, **32000
  vocab**). Must match the base checkpoint's vocab, or `--init_from` aborts with a
  shape-mismatch error. Tokens pack as **uint16** (vocab < 65536).
- **Phase-1 dual-stream format** (`pretokenize.py` → a shard dir):
  - `train.bin` — uint16 flat token ids, documents separated by EOS.
  - `weights.bin` — uint8 per-token **recall weight** (aligned 1:1 with tokens).
  - `meta.json` — `{n_tokens, vocab_size, tokenizer, mix, sources, docs_consumed, …}`.
  - The recall weight up-weights answer spans (`RECALL_W=4`) **only for `scrolls`**;
    every other source (including the current Phase-1 mix) gets weight 1, so
    `weights.bin` is all-ones and the CE reduces to a plain mean. Recall weighting
    activates only if you add `scrolls` to the mix.
  - `qa_context` sources are tokenized as one flattened text per record; `scrolls`
    tokenizes context/query/answer **separately** then concatenates, so the
    answer boundary is exact (no BPE cross-boundary drift).
- **Packing / windows**: `MemmapPackedDataset` draws fixed `max_seq_len` windows
  (default 2048) with a per-epoch random offset; labels are the next-token shift;
  `loss_weights` align to label positions.
- **Phase-2 tokenization** (contrastive): each query/passage tokenized to a
  **fixed length** (query 64, passage/negative 256), **right-padded** with an
  `attention_mask`. No document packing, no long concatenation.

---

## 6. How to run everything

Setup (once, on a CUDA node):
```bash
bash scripts/setup_env.sh                 # uv sync (torch + mamba-ssm + causal-conv1d + triton)
uv sync --extra cuda --extra lmharness    # + zero-shot eval harness (for eval_sweep)
export HF_HOME=/path/to/large/cache       # HotpotQA/StarCoder/Wikipedia stream a lot
```

### Phase 0 — pick the base checkpoint (by benchmark, not PPL)
```bash
scripts/eval_sweep.sh runs/echo-180m_v2 180m_v2
# reads MMLU/ARC/PIQA/HellaSwag/WinoGrande per checkpoint; pick the best step_*.
```

### Phase 1 — continue pretraining (~1.5B tokens)
```bash
INIT_FROM=runs/echo-180m_v2/step_<best>/model.pt scripts/continued_pretrain.sh
```
This tokenizes the 4-bucket mix into `$DATA_ROOT/mix_180m_v2_cpt_train/` (skips if
present), then warm-starts and trains. Checkpoints land in
`$RUN_ROOT/echo-180m_v2-cpt/` every ~100M tokens. Common overrides:
```bash
TOKENS=2000000000 LR=2e-4 scripts/continued_pretrain.sh     # 2B tokens, lower LR
NPROC=4 scripts/continued_pretrain.sh                       # 4-GPU DDP
SIZE=440m INIT_FROM=runs/echo-440m/final/model.pt scripts/continued_pretrain.sh
```
Then rank the continued checkpoints again:
```bash
scripts/eval_sweep.sh runs/echo-180m_v2-cpt 180m_v2
```

### Phase 2 — retrieval adaptation (5000 steps)
```bash
INIT_FROM=runs/echo-180m_v2-cpt/step_<best>/model.pt \
LM_DATA_DIR=$DATA_ROOT/mix_180m_v2_cpt_train \
scripts/retrieval_adapt.sh
```
Prints Recall@{1,5,20} every 500 steps and saves `model.pt` (backbone) +
`retrieval_encoder.pt` (backbone + projection) to `$RUN_ROOT/echo-180m_v2-ret/`.
Confirm LM benchmarks did not regress:
```bash
scripts/eval_sweep.sh runs/echo-180m_v2-ret 180m_v2
```
Anchor variants:
```bash
LM_MODE=combined LM_ANCHOR_WEIGHT=0.1 scripts/retrieval_adapt.sh   # InfoNCE + 0.1*LM every step
POOL=last scripts/retrieval_adapt.sh                              # last-token pooling
```

---

## 7. Schedules & hyperparameters

**Phase 1** (defaults; all env-overridable):

| knob | value |
|---|---|
| tokens / steps | 1.5B / ~7,629 (eff-batch 96 = 8×12, seq 2048 → 196,608 tok/step) |
| peak LR / warmup | 2.5e-4 (~½–⅓ of base 6e-4) / ~1.5% of steps (~114) |
| schedule / save | cosine decay / every ~100M tokens (~508 steps) |
| weight decay / grad clip | 0.1 / 1.0 |

**Phase 2** (defaults; all env-overridable):

| knob | value |
|---|---|
| steps / warmup | 5000 / 250 |
| query / passage len | 64 / 256 |
| batch / temperature | 128 / 0.05 |
| projection dim / pool | 768 / masked-mean |
| backbone LR / projection LR | 8e-6 / 1e-4 |
| hard negatives / grad clip | 2 (+ in-batch) / 1.0 |
| eval (Recall@1/5/20) | every 500 steps |

---

## 8. File map

| file | role |
|---|---|
| `koopman_lm/training/data/mix.py` | source registry + mix resolution (pure Python) |
| `koopman_lm/training/data/pretokenize.py` | dual-stream tokenizer; `--sources`, `qa_context` flattening |
| `koopman_lm/training/train.py` | LM training; `--init_from` weights-only warm start |
| `scripts/continued_pretrain.sh` | Phase-1 driver |
| `scripts/eval_sweep.sh` | rank checkpoints by zero-shot benchmark |
| `koopman_lm/models/koopman_lm.py` | `KoopmanLM.encode()` pooled embeddings |
| `koopman_lm/retrieval/encoder.py` | `RetrievalEncoder` (projection + L2) + InfoNCE + pooling |
| `koopman_lm/retrieval/data.py` | contrastive `(query, positive, hard-neg)` extraction (pure) |
| `koopman_lm/retrieval/_torch_data.py` | contrastive `IterableDataset` (tokenization) |
| `koopman_lm/retrieval/adapt.py` | Phase-2 loop + Recall@k eval |
| `scripts/retrieval_adapt.sh` | Phase-2 driver |
| `archive/MQAR/prefix_bench/prefix_masks.py` | prefix-mask taxonomy (benchmark only) |
| `TRAINING.md` / `RETRIEVAL.md` | pretraining mechanics / two-phase deep dive |

---

## 9. Verification status

Runs on a GPU cluster (Mamba CUDA kernels; multi-billion-token streaming). What
is checked in-repo:

- **Runs anywhere** (`pytest -m correctness code-tests/test_mix.py
  code-tests/test_retrieval_data.py`): mix parsing/normalization/quota,
  contrastive extraction (HotpotQA/MuSiQue/Wikipedia positives + hard negatives),
  weighted source sampling.
- **Cluster (CPU ok, no GPU needed)** (`test_retrieval_encoder.py`,
  `test_init_from.py`): InfoNCE, masked-mean/last pooling, encoder unit-norm,
  backbone/projection param groups, warm-start loader — via a CPU stub backbone.
- **GPU-only**: the full training loops and the Mamba/SKA forward.

Known caveats: StarCoder/HotpotQA gating; MuSiQue/Wikipedia config pinning; the
QA extractors degrade gracefully on schema drift (shorter docs, not crashes).
Phase 2 runs single-process (Phase 1 supports DDP via `NPROC>1`).
