# TrainTask: collapsing three training loops into one

**Status:** design, for review before implementation.
**Supersedes in detail:** `2026-08-07-run-system-design.md` §6.2, which sketched
this. Two things it did not account for are in §3 and §6.3 below.

**Goal:** one training loop. `train.py` (600 lines), `mqar_finetune.py` (534) and
`table2.py` (390) implement the same algorithm three times; the parts that
genuinely differ are the data source, the loss, and an optional in-loop eval.

**Non-goal:** changing what any of the three currently computes. Every number
they produce today should be reproducible afterwards. That is a testable claim,
and §7 says how it is tested.

---

## 1. Why this is not just deduplication

The obvious framing — "three loops, one algorithm, factor it out" — leads directly
into a silent correctness bug. The rest of this document exists because of §2.

Secondary motivations, both real but not the point:

- `train.py`'s loop is at **29% line coverage**. Three copies of an
  under-tested loop is three times the surface for a change to go unnoticed.
- `SyntheticDataSpec` already exists, validates, and can be built into a
  `RunSpec` — but `run/train_argv.py:101` and `run/data_verify.py:53` both refuse
  to launch it, by name. Finishing this is what turns MQAR and Table 2 into
  ordinary runs with run directories, result envelopes, sweeps declared once, and
  exact resume.

## 2. The invariant nobody wrote down

Causal LM training offsets inputs from targets by one. **This repo does that
offset in two different places**, and which place is a property of the data:

| | where the offset happens |
|---|---|
| shard path | the **dataset**. `training/data/dataset.py:57` emits `input_ids=chunk[:-1]`, `labels=chunk[1:]`, so `KoopmanLM.forward` scores `logits[i]` against `labels[i]` — positional, no shift |
| synthetic path | the **loss**. `make_mqar` emits *aligned* pairs, and `mqar_finetune.py:299` / `table2.py:287` score `logits[:, :-1]` against `labels[:, 1:]` |

Both are correct next-token prediction. From the outside they are
indistinguishable: same function, same `ignore_index=-100`, both produce a
descending loss. Which is what makes "these two loss computations are duplicates"
look like obvious cleanup.

**Measured consequence of getting it wrong.** In aligned MQAR data, at every
supervised position the label *is* the input at that position
(`inputs[0,25] == labels[0,25] == 60`). So feeding aligned data to the positional
loss scores the model on **copying a token it was just handed**. A one-hot
input-copier scores `<1e-4` under the positional loss and `>1.0` under the shifted
one. A model could reach that by learning identity, report excellent MQAR
accuracy, and never learn recall. Pinned by
`code-tests/test_loss_alignment.py`, mutation-checked.

**And the convention is not confined to the loss.** It appears at seven places in
two mutually inconsistent styles:

```
data prep          dataset.py:57 (shard, offsets) / make_mqar (aligned)
weight alignment   dataset.py:61  w[1:]  "align to label positions"
training loss      model-internal positional / mqar:299, table2:287 shifted
in-loop accuracy   curricula.py:99  eval_mqar, shifted
in-loop split loss table2.py:310,312  shifted
held-out ppl       evaluate.py:188 offsets in data
retrieval eval     evaluate_retrieval.py:421 offsets in data
```

So **the task's three responsibilities are one responsibility with three
surfaces.** That is the real justification for §6.2's boundary, and it means the
loss must NOT be unified — only the loop around it.

## 3. What §6.2 missed: resume is entangled with data iteration

`train.py` does not merely consume a `DataLoader`. It **constructs** one per
epoch, with an explicitly computed index list, because exact resume needs to skip
forward over already-consumed samples without re-reading them
(`train.py:376-393`, `resume.py:epoch_permutation` / `resume_indices`). It also
passes `generator=torch.Generator()` — with a comment explaining this is
*required*, not an optimisation, because `DataLoader.__iter__` draws from the
global RNG on every fresh iteration, so a resumed run would desync its dropout
masks from an uninterrupted twin.

Therefore "the task supplies batches" is too loose. If a task hands over an
opaque iterable, exact resume becomes the task's problem and stops being uniform
— which was one of the main prizes.

**Resolution: the task supplies a `Dataset`, not an iterator.** The loop keeps
ownership of the sampler, the index arithmetic, the generator, and resume.

## 4. The interface

```python
class TrainTask(Protocol):
    def build_model(self, cfg) -> nn.Module: ...
    def dataset(self, cfg, args) -> torch.utils.data.Dataset: ...
    def step_loss(self, model, batch) -> torch.Tensor: ...
    def in_loop_eval(self, model, step) -> Optional[dict]: ...     # default None
    def on_final(self, model, run_dir, cfg) -> None: ...           # default no-op
```

Two deliberate shapes:

**`step_loss` performs the forward call itself**, rather than receiving logits.
That is what lets `ShardTask` call
`model(input_ids, labels, loss_weights)["loss"]` while `SyntheticTask` calls
`model(input_ids)` and then computes shifted CE — each preserving its convention
exactly, with no flag and no branch in the loop. The loop enters autocast around
this call; the task does not manage precision.

**`build_model` is a fourth responsibility**, mildly against §6.2's "it owns
nothing else". It exists so the extension-mechanisms design's §4 (construction-time
variation) and §5 (post-construction mutation) have somewhere to attach later.
Extensions are explicitly out of scope, but this seam is nearly free now and
expensive to retrofit once call sites exist.

## 5. Ownership

| concern | owner | note |
|---|---|---|
| dataset | task | indexable, so the loop can do resume arithmetic |
| **loss + forward call** | **task** | §2. The thing that must not be unified |
| in-loop eval | task | `eval_mqar`, table2's split losses |
| model construction | task | the extensions seam |
| the loop, grad accumulation | loop | |
| DDP | loop | |
| resume | loop | `train.py`'s implementation; the other two are deleted |
| amp / GradScaler | loop | via `training/amp.py::amp_for` |
| param groups | loop | via `training/optim.py::param_groups`, now pinned at 100% |
| checkpointing + `checkpoint_meta` | loop | one `code_id`/`dirty` writer instead of four |
| logging | loop | must keep `loss %.4f` — the golden comparator parses it |

**The conservative property that makes this safe:** with the two scripts'
current configs, everything the loop newly owns is a **no-op** for them.
Gradient accumulation at `accum=1` is `loss/1` plus a step every iteration. DDP at
`world_size=1` does nothing. So their numbers should be preserved bit-for-bit —
testable, not hoped for.

## 6. The three tasks, and what each migration actually costs

### 6.1 `ShardTask` — the easy one
`MemmapPackedDataset`; `step_loss` calls the model with `labels` and
`loss_weights` and returns `out["loss"]`. No in-loop eval. This is `train.py`
unchanged in substance.

### 6.2 `SyntheticTask` (MQAR) — straightforward
`MQARDataset` is already an indexable `Dataset` keyed on `base_seed + idx`, so it
drops into the loop's sampler directly. `step_loss` calls `model(input_ids)` and
computes shifted CE. `in_loop_eval` delegates to `eval_mqar`.

### 6.3 `table2` — the real migration cost, and §6.2 understated it
`table2.py` has **no dataset**. It calls `make_train_batch(step, args)`
(`table2.py:114`), a function of the *step number* returning a whole batch. Two
consequences:

- It must be wrapped as a `Dataset` whose index is the step. Deterministic in
  `step`, so this is arguably *more* resumable than the shard path.
- But `Dataset.__getitem__` conventionally returns **one sample** for `DataLoader`
  to collate, while `make_train_batch` returns a **full batch**. So the wrapper
  needs `batch_size=None` or a custom `collate_fn`. Not free, and worth doing
  explicitly rather than discovering at runtime.

Also `table2` trains in **fp16 with a `GradScaler`**, unlike the other two. That
is already expressible: it declares `compute_precision='fp16'` and `amp_for`
returns a scaler. No special case needed — but the loop must actually honour the
scaler path, which only `table2` exercises today.

## 7. Verification

In order. Each gate must pass before the next migration step.

1. **The golden curve.** `configs/runs/4m-golden.yaml`, 5.2M params, 400 steps,
   gradient accumulation 4 (chosen so the accumulation path is exercised rather
   than a no-op). Two identical runs agree to **max 0.0002**, which is the log
   format's precision, so the tolerance is 0.001 —
   `scripts/compare_golden_curve.py`, exit 1 on regression.
2. **An MQAR golden**, captured the same way before touching anything. The 4m
   golden only covers the shard path, and MQAR is the path whose convention
   differs. *This does not exist yet and is the first thing to do.*
3. **`test_loss_alignment.py`** must stay green, and each new task's `step_loss`
   gets the same positional-sensitivity treatment.
4. **`test_param_groups.py`** — 19 tests, mutation-checked, pinning the
   decay/no-decay policy so a rewritten loop cannot silently change the optimizer.
5. **The bit-for-bit claim** of §5: run `mqar_finetune` and `table2` before and
   after at a fixed seed and diff the curves.
6. **`--resume`** on the unified loop. Note this is currently unverified even for
   `train.py`, behind two pre-existing nvcc-related GPU failures, so this gate
   may need the nvcc issue addressed first.

## 8. The objective hook

Optuna reads a trial's score from `run_dir/eval/<ckpt>/quick_eval.json`. **Nothing
writes it** — the only caller of `write_quick_eval` on the branch is a
verification sbatch, by hand. So a study today trains N models and records N
FAILs.

`on_final` is where that closes: after the final checkpoint, call `run_quick_eval`
and `write_quick_eval`. Behind a flag (`--eval_on_final`), set by
`build_train_argv` for search-launched runs only, so ordinary runs and static
sweeps stay byte-identical by default.

This is why the hook belongs here rather than being patched into `train.py`: the
file is being restructured anyway, so it stops being an edit to a frozen file.

Known and deliberately deferred: `quick_eval` measures in fp32 while trials train
in bf16. Measured spread ~0.005 loss and inconclusive (job 438232), so it is filed
low-priority in `DECISIONS-2026-08-20.md` §1.5.

## 9. Out of scope

- The extension mechanisms (§4 builders, §5 probes). `build_model` leaves the
  seam; nothing more.
- Replacing `TRAIN_RE` log-scraping with a structured metric stream. Cleaner, and
  a separate decision — and note the golden comparator and the search pruner both
  parse that format, so they would move together.
- Regenerating Table 2's numbers. Already superseded by the `param_groups` fix;
  whether to re-run is a call for a human.
- Fleet-launching tooling for parallel studies.

## 10. Order, and when to stop

1. MQAR golden captured (gate: two replicates agree)
2. `TrainTask` protocol + the loop, with `ShardTask` only. Gate: 4m golden matches
3. `SyntheticTask` (MQAR). Gate: MQAR golden matches, alignment suite green
4. `table2` task, including the collate wrinkle. Gate: its curve matches
5. `SyntheticDataSpec` wired through `train_argv` / `data_verify`
6. `on_final` objective hook. Gate: a study records a real score, not FAIL

**Stop and write up rather than push through** if: a golden diverges beyond 0.001,
the alignment suite fails, the CPU suite drops below 1051, or a step needs a
decision this document did not anticipate. A blocked task with a clear question is
worth more than 1500 lines built on a guess.
