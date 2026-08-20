# Decisions and queued work — 2026-08-20

Written so the reasoning behind two threads survives the conversation that
produced it. Both came out of a manual, commit-by-commit review of
`jack/search-and-provenance`.

Detail lives in the linked specs; this is the decision log and the sequencing.

---

## 1. Precision

### 1.1 The three-way distinction (the thing that was being conflated)

"Eval precision" is not one question. Each of these has a *different* correct
answer, and treating them as one is what produced three inconsistent call sites:

| question | correct precision | why |
|---|---|---|
| Will this config train well? | **training precision** | you are predicting a training run, so use its arithmetic |
| What did this model learn? | **fp32 / highest available** | measures the function, not the arithmetic; stable across hardware; keeps models trained at different precisions comparable |
| What will it do in production? | **serving precision, chosen explicitly** | you are validating a deployment |

### 1.2 Divergence is fine; *unmeasured, unrecorded* divergence is not

Serving at a precision other than training's is not wrong in principle —
quantized serving of bf16-trained models is standard practice with a measured
tradeoff. What made the old `weight_dtype="bf16"` default wrong was that nobody
chose it, nothing recorded that a number came from it, and the divergence was
never measured.

Divergence also has a **direction**, which matters more than its existence.
`compute_precision: bf16` means fp32 master weights with a bf16 autocast, so:

- **Upward** (serving in fp32, or fp32 weights + autocast) gives the function
  those fp32 master weights represent. Arguably *more* faithful — training
  precision is partly an artifact of gradient accumulation, and there are no
  gradients at inference.
- **Downward** (truncating parameters) is the risky direction. Nothing in
  training ever saw truncated weights.

The old code diverged downward. Current code does not diverge downward anywhere.

### 1.3 Status of each site

| site | question it answers | state |
|---|---|---|
| `evaluation/lm_harness_eval.py` | production / benchmark | **done** — fp32 weights + autocast at `cfg.compute_precision` |
| `evaluation/quick_eval.py` | will this config train well | **queued**, see 1.5 |
| `evaluation/evaluate.py` | what did this model learn | **stays fp32**, see 1.4 |

### 1.4 `evaluate.py` stays fp32 — but for a better reason than before

The original defence ("keeps trials comparable across swept precisions") was
wrong: `compute_precision` is not in the search space, so there is no variation
to confound. That argument is dead.

The surviving reason: a reported benchmark number should describe the **learned
function**, not the arithmetic that happened to be cheap on the training cluster.
Evaluating every model in fp32 keeps models trained at different precisions
comparable on the same benchmark; evaluating each at its own training precision
does not. So fp32 is right here — as a **stated policy with the precision
recorded**, not as an accident of nobody writing an autocast.

### 1.5 `quick_eval` — thread training precision through (small)

```python
def evaluate_loss(model, device, loader, *, max_batches=None, precision='fp32'):
    with torch.no_grad(), autocast(str(device).split(':')[0], precision):
```

`precision` flows from `run_quick_eval(..., precision=cfg.compute_precision)`.
`koopman_lm.precision.autocast` already returns a reusable context manager and
`nullcontext()` for fp32, so the default keeps current behaviour until a caller
opts in.

**Priority: low, and deliberately recorded as such.** The measurement in 1.7 puts
the fp32-vs-bf16 gap around 0.005 in loss, far below the differences the search
distinguishes between trials, so the effect on ranking is probably nil today. It
is cheap and principled, not a live bug. An earlier version of this note claimed
the objective was "blind to the failure mode a spectral model is most exposed to";
that overstated what the data supports.

### 1.6 Where serving precision belongs: the result envelope, NOT a RunSpec section

`RunSpec`'s four sections encode exactly two categories — `model`/`data`/`optim`
**define the experiment** (hashed into `group_id`), and `runtime` **provably does
not change results** (deliberately unhashed). Serving precision is neither: it
changes the number but is not part of what the experiment *is*. It is a fifth
category `RunSpec` cannot express, and forcing it into any existing section would
corrupt that section's meaning.

It is a property of a **result**. So it belongs in the common result envelope the
run-system design already specifies (`2026-08-07-run-system-design.md` §4.2:
`run_id`, `task`, `checkpoint`, `git_commit`, `created_at`, metrics nested).

**Queued:** add `weight_dtype` and `autocast_precision` to that envelope, and
extend the envelope from `write_quick_eval` (which already writes
`eval/<ckpt>/quick_eval.json` and reads `run_id` back from `spec.yaml`) to
`evaluate.py` and `lm_harness_eval.py`. This also makes precision-vs-throughput a
plottable axis instead of a guess, which is the real reason to want it.

A `serve:` config section becomes correct only once eval is a run-system task
(§4.2's "eval reads `spec.yaml`, not `--model_size`"), because only then does a
consumer exist. Until then it would be write-only config.

### 1.7 Measurement (job 438232, H100, the 30-step 50m checkpoint)

Four regimes, identical batches and order, weights restored from a pristine fp32
copy between each:

```
A  fp32 weights, no autocast                 loss 7.693878   ppl 2194.87
B  fp32 weights + bf16 autocast (as trained) loss 7.692783   ppl 2192.47
C  bf16 weights, no autocast (OLD default)   loss 7.687500   ppl 2180.92
D  bf16 weights + bf16 autocast              loss 7.688641   ppl 2183.41
```

Spread ~0.005 loss / ~14 ppl. **Inconclusive, and the reason is instructive:**
the lower-precision regimes score *better*, so precision is not behaving
monotonically. That is a small perturbation landing randomly, not systematic
degradation — expected on a 30-step model at ppl ~2200 whose weights are barely
trained. This cannot distinguish "precision does not matter here" from "precision
matters but this model is too undertrained to show it."

**To settle it properly:** re-run against a converged checkpoint. Also add a
warmup pass — the throughput column from this job (265 → 10,728 → 4,421 →
202,622 tok/s, non-monotonic over three orders of magnitude) is pure CUDA warmup
and kernel caching, because regime A paid all of it. Those numbers are invalid and
must not be cited.

Reproduce: `sbatch scripts/measure_precision_spread.sbatch`.

---

## 2. Trainer refactor (`TrainTask` / `SyntheticTask`)

Specified in `2026-08-07-run-system-design.md` §6.2. Not started, and correctly
not part of this PR.

### 2.1 What it is

Three near-duplicate training loops exist:

```
train.py            600 lines    MemmapPackedDataset, weighted CE through the model
mqar_finetune.py    534 lines    MQARDataset generator, external masked CE
table2.py           390 lines    inline per-step generators, external masked CE
curricula.py        304 lines    the generators themselves -- these stay
```

Everything that genuinely differs is the **data source**, the **loss**, and an
optional **in-loop eval hook**. The loop, resume, DDP, gradient accumulation,
parameter grouping, logging and checkpointing are one algorithm written three
times. `TrainTask` supplies exactly those three things and owns nothing else;
`RunSpec.data.kind` selects the task. So this **removes** ~1500 lines of
duplication rather than adding a second system.

### 2.2 Why it matters here: `SyntheticDataSpec` is the config half of this change

`SyntheticDataSpec` already exists, validates, and can be built into a `RunSpec`.
What is missing is the execution half — and both launch paths refuse **loudly**
rather than silently doing the wrong thing:

- `run/train_argv.py:101` — "only supports `data.kind='shard'` … synthetic needs
  TrainTask/SyntheticTask"
- `run/data_verify.py:53` — "`data.kind='synthetic'` cannot be launched"

`data_verify`'s own comment records that it *used* to skip silently. So this is
the acceptable form of half-built: the config language runs ahead of the
launcher, and the gap is a named error rather than a trap. Finishing it is what
turns MQAR and Table 2 into ordinary runs — gaining run directories, result
envelopes, sweeps declared once, and exact resume.

### 2.3 Sequencing: its own PR, after this one merges

1. This PR is 36 commits and GPU-verified on an H100. A trainer refactor would
   invalidate that verification.
2. `TrainTask` restructures the very code this PR just changed (`amp_for`,
   precision threading, `checkpoint_meta`). Doing it afterwards means refactoring
   from a known-good, hardware-verified state.
3. It needs its own GPU budget: every task × resume × DDP path, plus regenerating
   Table 2's numbers.

The trainer is the highest-risk code in the repo, and this branch supplied the
precedent: the `autocast` reuse bug (`3be37b7`) passed 843 CPU tests and killed
training at step 2 on real hardware. Nothing about a trainer refactor is
verifiable on a login node.

### 2.4 One correction to the design's own argument

§6.3 presents the weight-decay discrepancy as the strongest case for `TrainTask`
— `table2.py` decaying norms, biases, embeddings and the Mamba state parameters
(`A_log`, `D`, `dt_bias`) that `train.py` excludes, making published Table 2
numbers a different optimizer regime from every other result in the repo.

**That is already fixed.** Both `table2.py:244` and `mqar_finetune.py` now route
through the shared `experimentation.training.optim.param_groups`, with comments
recording the old numbers as superseded. So the urgency is lower than §6.3
implies. The duplication argument stands on its own.

---

## 3. Queue, in order

| # | work | size | gate |
|---|---|---|---|
| 1 | `quick_eval` precision (1.5) | ~4 lines | none; low priority |
| 2 | `precision` in the result envelope (1.6) | small — extends `write_quick_eval` | none |
| 3 | Re-run 1.7 on a converged checkpoint, with warmup | one GPU job | needs a converged ckpt |
| 4 | Install `lm_eval` into a `--target` dir so tests can import `lm_harness_eval` | small | — |
| 5 | `asdict(cfg)` + `_check_model_key_set` in checkpoint meta | `2026-08-20-checkpoint-meta-resolved-config.md` | changes checkpoint format |
| 6 | `TrainTask` / `SyntheticTask` (§2) | ~1500 lines restructured | after this PR merges |

Also open, from the same review:

- **Coverage is unmeasured.** `coverage` is not installed. The new import gate
  proves every module *loads*; nothing measures what fraction of lines any test
  executes. Eight modules currently do nothing but load. This is a larger gap
  than the one the import gate closed.
- `configs/search/curated_15.yaml` still absent — the one input blocking a
  launchable 15-cell anchor sweep.
- `--gradient-checkpointing` + `ska_prefix_scan` trap: still a comment, not a
  `__post_init__` assert.
