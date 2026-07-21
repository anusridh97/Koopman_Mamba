# Continued pretraining + retrieval adaptation

A two-phase recipe to broaden the model after the 10B-token FineWeb-Edu run and
then repair/strengthen its retrieval representations — without long contexts.

The motivation: after 10B FineWeb-Edu tokens, val PPL improved but zero-shot
regressed, and retrieval quality can lag even when perplexity looks good. Phase 1
re-diversifies the data distribution (code/math/reasoning/retrieval-evidence);
Phase 2 directly optimizes the embedding space with a short contrastive run.

---

## Phase 1 — continue pretraining (1–2B tokens)

Warm-start from the **best base checkpoint by zero-shot benchmark** (not
necessarily the final one — rank first with `scripts/eval_sweep.sh`).

```bash
# 0. pick the base checkpoint by benchmark, not PPL
scripts/eval_sweep.sh runs/echo-180m_v2

# 1. continue on the 4-bucket mix (weights-only warm start, fresh cosine)
INIT_FROM=runs/echo-180m_v2/step_<best>/model.pt scripts/continued_pretrain.sh
```

**Mix** (causal LM loss only — no contrastive loss yet):

| bucket | share | source |
|---|---|---|
| FineWeb-Edu | 40% | `fineweb` |
| code / math | 25% | `code` (StarCoder) + `math` (OpenWebMath) |
| structured QA / reasoning | 20% | `cosmopedia` |
| retrieval-oriented LM | 15% | `wikipedia` 9% + `hotpotqa` 3% + `musique` 3% (evidence text as plain LM) |

The retrieval bucket uses the *same* corpora Phase 2 trains on, but here consumed
as ordinary causal-LM text (`kind=qa_context` flattens each QA record to
"question + evidence/distractor paragraphs"). NQ evidence is Wikipedia-derived, so
`wikipedia` covers it; `nq` is a registered source but off the default mix (full
NQ is heavy to stream — add `nq=…` to `SOURCES` if you want it).

**Schedule** (weights-only warm start — discard optimizer/scheduler/Adam moments):

```
peak LR  = 2.5e-4     (~1/2–1/3 of the base 6e-4)
warmup   = ~1.5% of steps
cosine decay
save     = every ~100M tokens   (so you can rank by benchmark, not PPL)
```

Defaults: `TOKENS=1.5e9` → ~7.6k steps at eff-batch 96×2048; warmup ~114; save
every ~508 steps. All env-overridable (`TOKENS=2e9 LR=2e-4 scripts/continued_pretrain.sh`).

**Evaluate frequently** — every ~100–200M tokens, rank checkpoints on the full
suite and do **not** pick by lowest PPL:

```bash
scripts/eval_sweep.sh runs/echo-180m_v2-cpt        # MMLU/ARC/PIQA/HellaSwag/WinoGrande
```

---

## Phase 2 — retrieval adaptation (5,000 steps)

Adapt the representations into a dense dual-encoder. Short sequences only, so no
long-context memory cost: **query = 64, passage/negative = 256**, each encoded
independently. Backbone nudged gently; a fresh projection head learns the
retrieval space; an LM anchor protects general LM quality.

```bash
INIT_FROM=runs/echo-180m_v2-cpt/final/model.pt \
LM_DATA_DIR=data/mix_180m_v2_cpt_train \
scripts/retrieval_adapt.sh
```

**Objective** — InfoNCE on unit embeddings (cosine), in-batch + per-query hard
negatives:

```
L = -log  exp(q·d⁺/τ) / ( exp(q·d⁺/τ) + Σ_i exp(q·d_i⁻/τ) )
τ = 0.05,  embeddings L2-normalized,  projection dim = 768,  pool = masked-mean
```

**LM anchor** — two realizations of `L = L_retrieval + 0.1·L_LM`, pick with `LM_MODE`:
- `schedule` (default): 4 retrieval batches : 1 LM batch (the plan's batch schedule).
- `combined`: `InfoNCE + lm_anchor_weight·LM_CE` every step (set `LM_ANCHOR_WEIGHT=0.1`).

**Datasets** (`SOURCES`, renormalized): HotpotQA 35%, self-supervised Wikipedia
45% (subsumes NQ evidence), MuSiQue 20%. Positives/hard-negatives come from
structure — HotpotQA/MuSiQue supporting paragraph = positive, the distractor
paragraphs = same-topic hard negatives; Wikipedia self-sup builds
title→section / section→nearby / first-sentence→paragraph / heading→subsection,
with other sections of the same article as hard negatives.

**Negatives per query**: `N_HARD=2` explicit hard negatives (same-article
distractors) + all in-batch negatives (every other query's positive).

**Hyperparameters** (all overridable):

```
steps            5000        warmup           250
query length     64          passage length   256
batch size       128         temperature      0.05
projection dim   768         backbone LR      8e-6
projection LR    1e-4        grad clip        1.0
```

**Evaluation** — Recall@{1,5,20} on a held-out in-domain pool every 500 steps
(printed in-loop). Zero-shot LM benchmarks are **not** run in-loop (too slow);
each eval step saves a checkpoint (`model.pt` = backbone, `retrieval_encoder.pt`
= backbone + projection) so you rank the adapted checkpoints offline — retrieval
should improve **without** regressing LM benchmarks:

```bash
scripts/eval_sweep.sh runs/echo-180m_v2-ret        # confirm LM benchmarks hold
```

If retrieval adaptation hurts zero-shot LM: **lower `BACKBONE_LR` or raise the LM
anchor** (`LM_MODE=combined LM_ANCHOR_WEIGHT=0.1`), rather than training longer or
swapping datasets.

---

## What's where

| file | role |
|---|---|
| `koopman_lm/training/data/mix.py` | source registry (adds wikipedia/hotpotqa/musique/nq) |
| `koopman_lm/training/data/pretokenize.py` | `qa_context` flattening for the retrieval-LM bucket |
| `scripts/continued_pretrain.sh` | Phase-1 driver (4-bucket mix, warm start) |
| `scripts/eval_sweep.sh` | rank checkpoints by zero-shot benchmark, not PPL |
| `koopman_lm/models/koopman_lm.py` | `KoopmanLM.encode()` pooled embeddings |
| `koopman_lm/retrieval/encoder.py` | `RetrievalEncoder` (projection + L2) + InfoNCE |
| `koopman_lm/retrieval/data.py` | contrastive (query,positive,hard-neg) extraction |
| `koopman_lm/retrieval/adapt.py` | Phase-2 loop + Recall@k eval |
| `scripts/retrieval_adapt.sh` | Phase-2 driver |

## Assumptions / knobs you may want to change

- **Pooling** defaults to masked-mean (`POOL=last` for last-token). Swap if your
  eval prefers the causal-summary slot.
- **Dataset coordinates** are HF defaults; StarCoder + HotpotQA are gated /
  `trust_remote_code` and MuSiQue/Wikipedia configs may need pinning. Override
  paths via the `--*_path` flags (pretokenize) or `SOURCES` / source specs.
- **DDP**: Phase 2 runs single-process (the plan is GPU-constrained, 5k steps);
  Phase 1 supports DDP via `NPROC>1`.
