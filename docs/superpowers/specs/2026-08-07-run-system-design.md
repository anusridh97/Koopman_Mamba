# Run system design — configs, launch, and run identity

Status: **ready for review.** Owner: jkli. Written 2026-08-07 during a design
session on branch `reorg-module-layout`.

---

## 1. Why

The repo has a strong spine for *architecture* — a frozen `KoopmanLMConfig`,
YAML instances, a content hash stamped into every checkpoint — and no spine at
all for everything that varies around it. Optimization settings live in a bash
`case` statement, data composition lives in a `.bin`'s sidecar, results land
wherever the caller pointed, and four naming schemes are maintained by hand.

There is no artifact anywhere that represents *a run*. A run is the tuple
(YAML file, bash case arm, env overrides at launch, CLI flags), and only the
first element is reconstructable afterwards.

### The evidence

Each of these is checkable against the tree as of `a7ced87`.

**Identity is half-captured.** `config_hash` (`config.py:281`) hashes only
`KoopmanLMConfig`. It goes into every checkpoint (`train.py:346`) and into the
wandb group name (`train.py:287`). Two runs at LR 4e-4 and 3e-4 over different
token budgets are therefore indistinguishable by hash and land in the same
wandb group. The half of a run's identity that is actually swept is the half
that is not captured.

**Optimization settings are shell.** `pretrain.sh:15-41` is a per-size
hyperparameter table (`TOKENS`, `STEPS`, `PDBS`, `GA`, `LR`, `WARMUP`) written
as a bash `case`. It is configuration in a costume, and nothing can import,
type-check, or test it.

**DDP arithmetic is untested logic in shell.** `pretrain.sh:96` divides
gradient accumulation by `NPROC` to hold effective batch constant. No test
covers it.

**Evaluation is emitted as strings.** `pretrain.sh:119-121` `echo`s the eval
commands for a human to copy. An orchestrator that cannot orchestrate.

**Data identity is unverified.** `pretrain.sh:48` names shards
`fineweb_${DATA_TAG}_train`, encoding only a size tag — not the tokenizer, token
count, or mix. `pretrain.sh:70` skips tokenization whenever `train.bin` exists.
`pretokenize.py:344` does record the tokenizer into `meta.json`, but
`MemmapPackedDataset.__init__` (`dataset.py:28-35`) reads that file only for
`n_tokens` and never compares the tokenizer against the one `train.py` loaded.
Overriding `TOKENIZER` on a second launch therefore trains silently on tokens
produced by a different tokenizer than the model's vocab was built from.

**Configs are duplicated.** `configs/50m.yaml` and
`configs/50m_prefix_scan.yaml` are byte-identical (`md5 1badfc8d`), likewise the
180m pair (`md5 8c5f86aa`). `CONFIG_REGISTRY` advertises four configs; two
exist. The `_prefix_scan` variants set no backend field, so
`pretrain.sh 50m_prefix_scan` trains a plain 50m.

**Names are maintained by hand.** The size token appears in four schemes with
three spellings — `50m`, `50m_prefix_scan`, `50m_quality`. `phase_tag` defaults
to the literal `"run"` (`train.py:413`) and `pretrain.sh` never sets it, so every
production wandb group is `run-50m_prefix_scan-<hash>`.

---

## 2. Branch lineage audit (prerequisite finding)

Before designing on top of `config.py`, we established which of two diverged
lineages is authoritative. Merge base is `1829e6f` (2026-06-10); 37 commits on
`reorg-module-layout`, 43 on `origin/cody-phase2a-1m-prep`.

Comparing at `b2d965e` (our pre-reorg tip, so paths align and rename noise
drops out):

| File | Ahead | Evidence |
|---|---|---|
| `config.py` | **ours** | 10 fields exist only here: all `ska_prefix_scan*`, `ska_mode`, `mlp_type`, `norm_type`/`norm_eps`, `init_policy`, `initializer_range`, `rescale_prenorm_residual` — plus every `ValueError` validation in `__post_init__`. Cody has one: `mamba_headdim`. |
| `koopman_lm.py` | **ours** | `MambaSKAParallelBlock`, `_build_mlp`, both init policies. Cody: nothing unique. |
| `recurrent.py` | **ours** | `PrefixSKAState`, `_prefill_sequence_layer`, `_step_sequence_layer`. Cody: nothing unique. |
| `baselines.py` | **ours** | Cody's apparent extras (`SKABlock`, `SwiGLUMLP`) merely relocated — now at `models/koopman_lm.py:14` and `modules/channel_mixer/mlp.py:11`. |
| `ska.py` | ours (net) | +8/−32 toward Cody; no API divergence. |
| `train.py` | ours (net) | +47/−79 toward Cody; no API divergence. |
| `attention.py` | **Cody** | RoPE dtype fix. The one confirmed regression. |

**Verdict: `reorg-module-layout` is the authoritative source lineage.** The
raw diffstat (429 insertions / 672 deletions toward Cody) is misleading — our
side is larger because it carries the prefix-scan architecture Cody's never
received.

Three targeted cherry-picks close the gap, and nothing else:

1. `mamba_headdim` into `config.py`. Needed at small `d_model`: `mamba_ssm`
   packs z/x/B/C/dt into one `in_proj` whose width the channel-last
   `causal_conv1d` kernel requires to be a multiple of 8. At `d_model=64` the
   default `headdim=64` yields width 290 and the kernel rejects it at runtime.
2. The RoPE dtype fix in `attention.py:18`. `cos`/`sin` are built in fp32, so
   `q` and `k` are silently promoted while `v` stays in the activation dtype and
   SDPA raises. Autocast masks it; pure bf16 (e.g. DeepSpeed) does not.
3. `mqar_cell_fits` into `evaluation/mqar/mqar.py`. Its absence aborts
   collection of the entire test suite.

Additionally, the 8 lost config YAMLs (`1m`, `180m_gated`, `180m_v2`, `370m`,
`440m`, `880m`, `1p5b`, `3b`) were verified to use **zero** fields unknown to
our dataclass. Restoring them is safe and turns 25 of 29 current test failures
green.

---

## 3. Design

### 3.1 `RunSpec` is the unit of configuration and of identity

Four frozen dataclasses:

- `model: KoopmanLMConfig` — unchanged, exactly as it exists today
- `data: DataSpec` — **polymorphic**, tagged by `kind`:
  - `kind: shard` — shard reference plus the tokenizer/mix/token-count it must match
  - `kind: synthetic` — generator name and its parameters (`mqar`, `toolcall`,
    `sysprompt`, `niah`), for the experiments of §6
- `optim: OptimSpec` — lr, warmup, schedule, effective batch, weight decay, grad clip
- `runtime: RuntimeSpec` — seed, precision, ddp, partition, account, QoS, workers

`data.kind` is the *only* place the pretraining and synthetic-experiment paths
differ. See §6.

### 3.2 Two-stage lifecycle

Author `configs/runs/<name>.yaml` using `extends:` plus a short override block.
At launch a resolver flattens inheritance, stamps provenance (git commit, shard
sha256, tokenizer revision, torch/CUDA versions), and writes a fully-materialized
`spec.yaml` — no `extends:`, no defaults, every value spelled out — into the run
directory.

**Every downstream consumer reads only the materialized file.** Eval, resume,
analysis. Base configs are never re-read after launch, so drift in a base cannot
retroactively change the meaning of a finished run.

This is the resolution of a real tension: composition is right for *authoring*
(a shared default should be edited once, and reviewers need readable diffs),
while a flat self-contained record is right for *defending a number*. They are
not competing options; they belong at different lifecycle stages.

### 3.3 Run identity

```
group_id = sha256(model + data + optim)[:8]          # the experiment
run_id   = sha256(model + data + optim + seed)[:8]   # the datapoint
```

Two hashes, because seed occupies an awkward middle position: a different seed
*is* a different datapoint, but three seeds are also obviously one experiment.
Hashing with and without it gives both readings, and makes replicates visible on
the filesystem rather than only recoverable through the aggregation layer:

```
$RUN_ROOT/50m-fineweb-3b.<group_id>/seed42.<run_id>/
```

The hashed/not-hashed line is **not** the struct boundary. It is:

- **Scientific inputs** — a different value means a different experiment.
  Architecture, data, optimizer, and seed. All hashed. Seed is included because
  three seeds are three datapoints.
- **Execution details** — a different value means the same experiment run
  differently. GPU count, partition, account, dataloader workers, DDP. Not
  hashed.

A Slurm job ID is deliberately **not** the run id. It is not reproducible (a
rerun yields a new number, so "have I already run this?" is unanswerable), it
carries no information about what the run was, and Anu's Nebius jobs have none.
Job IDs are recorded as provenance instead:

- **`run_id`** — content hash of scientific inputs. Answers *what experiment is
  this*. Identical on Marlowe and Nebius.
- **attempt record** — one entry per execution: timestamp, host, `SLURM_JOB_ID`
  or Nebius instance id, git commit, resolved paths. Answers *which execution
  produced these bytes*.

One `run_id`, many attempts — which is required anyway once a `preempt`-partition
job is killed and requeued.

### 3.4 Python owns orchestration; shell shrinks to the scheduler boundary

```
python -m koopman_lm.run configs/runs/50m-fineweb-3b.yaml --launcher slurm
  ├── resolve spec (extends -> flat) + validate
  ├── verify data shard matches spec (tokenizer, mix, n_tokens)
  ├── materialize spec.yaml + attempt record into the run dir
  └── hand off to a Launcher
        ├── LocalLauncher   -> subprocess / torchrun, in place
        ├── SlurmLauncher   -> generates launch.sbatch into the run dir, submits
        └── (later) NebiusLauncher
```

The spec says *what* to run; the launcher says *where*. Anu changes one flag and
the `run_id` is unchanged, so her results and ours are comparable by
construction.

Slurm genuinely requires a file containing `#SBATCH` directives, so bash does
not disappear — but it stops being *source* and becomes an *artifact*: generated
per run, written beside the checkpoints it produced, never hand-edited. It
doubles as a reproduction record.

**Marlowe-specific**: submitting is not merely `--partition=batch`. Accountless
jobs are rejected, and the `batch` partition requires QoS `medium`, which lives
on account `marlowe-m000151-pm06` — not the default `marlowe-m000151` (QoS
`normal`). Account and QoS are therefore real `RuntimeSpec` fields with correct
defaults, not an afterthought. This is currently tribal knowledge in one
person's shell history.

### 3.5 Data verification (scoped deliberately small)

`RunSpec.data` names a shard directory. The launcher reads that shard's
`meta.json` and asserts tokenizer, mix, and `n_tokens` match the spec, failing
before any GPU time is spent. Roughly twenty lines; no workflow change; no
backfill of existing shards on scratch.

Full content-addressing (hashing shard bytes, naming directories by digest) is a
strictly larger change protecting against a shard being *modified* after the
fact — a real but rarer risk. Deferred; it can be added later without redesign
because the spec already names the shard.

### 3.6 Tree changes

New:

```
koopman_lm/run/
  spec.py        RunSpec + sub-specs; resolve(), materialize(), run_id()
  launch.py      Launcher ABC; LocalLauncher, SlurmLauncher
  __main__.py    python -m koopman_lm.run <spec.yaml> [--launcher slurm|local]
configs/
  base/          50m.yaml, 180m.yaml   (model fields only, deduplicated)
  runs/          50m-fineweb-3b.yaml, 180m-fineweb-10b.yaml
```

Run directory:

```
$RUN_ROOT/50m-fineweb-3b.a3f91c2e/
  spec.yaml            fully resolved
  launch.sbatch        generated
  attempts.jsonl       one line per execution
  resume.pt            rolling; optimizer + scheduler + RNG + data position
  step_1000/ step_2000/ final/      weights-only, archival
  eval/
    step_30000/zeroshot.json
    final/fineweb_ppl.json
```

Deleted: `scripts/pretrain.sh`, `scripts/train_50m.sh`, `scripts/train_180m.sh`.
Their per-size tables become `configs/runs/*.yaml`; their DDP arithmetic moves
into `SlurmLauncher` where it is unit-testable.

Retained as shell, correctly: `scripts/setup_env.sh`,
`scripts/build_b200_prefix_scan.sh`, `scripts/slurm_tests.sh`.

Deduplicated: `configs/50m_prefix_scan.yaml` and `configs/180m_prefix_scan.yaml`
are deleted; the registry drops to two entries.
`test_config_hash_distinguishes_configs` already fails on exactly this, so the
fix has a test waiting for it.

### 3.7 Artifact write policy

Content-addressed directories create an overwrite hazard that the rest of this
design would otherwise walk straight into: because `run_id` is derived from the
spec, relaunching an identical spec resolves to an identical directory. Without
a policy, a re-launch silently clobbers a completed run's `final/` — and since
the spec is identical, nothing in the config would reveal that anything was
lost. The current code has the same exposure: `_save_checkpoint`
(`train.py:362`) does `os.makedirs(..., exist_ok=True)` and then writes over
whatever is present.

The governing distinction is between **earned** bytes and **derived** bytes.
Checkpoints and result JSONs are earned — hours to days of GPU time. Specs,
sbatch files, and aggregation tables are derived and reproducible. The policy is
strict for the former and relaxed for the latter.

- **Run-directory creation refuses to clobber.** If the target exists and
  contains `final/`, abort with the path and the conflicting `run_id`.
  Proceeding requires either `--resume` (continue from `resume.pt`) or
  `--force` (explicit, and recorded in `attempts.jsonl`).
- **`attempts.jsonl` is append-only**, never rewritten. It is the audit trail of
  every execution against this `run_id`, including forced ones.
- **All state writes are atomic** — temp file plus `os.replace` — so a
  preemption mid-write cannot leave a truncated `resume.pt` or `spec.yaml`.
- **Re-scoring a checkpoint overwrites its result file.** This is a deliberate
  exception: eval output is cheap and re-derivable, and the alternative
  accumulates unversioned near-duplicates. Stated here so it is a decision
  rather than an accident.

---

## 4. Results and evaluation layer

### 4.1 The shape of the problem

A run is **one training job**. Evaluations are *attached results*, not part of
the run: training is expensive and singular, scoring is cheap, repeatable, and
plural. One run yields many checkpoints, and each checkpoint is scored many
times — because benchmarks get added, eval code gets fixed, and the best
checkpoint by loss is frequently not the best by benchmark. (Cody's
`eval_sweep.sh` existed precisely to rank checkpoints by zero-shot benchmark
rather than PPL.) Binding scoring into the run would mean retraining to
re-score.

"Evaluation" in this repo covers at least four distinct operations, and the
design must not assume they are alike:

| Kind | Entry point | Nature |
|---|---|---|
| Perplexity | `evaluate.py --mode fineweb_ppl` / `ppl` | teacher-forced forward pass; no generation |
| Zero-shot benchmarks | `lm_harness_eval.py` | mostly answer-likelihood scoring |
| Long-context retrieval | `niah_quick.py`, `ruler.py`, `babilong.py` | genuine generation |
| Retrieval adaptation | `evaluate_retrieval.py --ft_steps 500` | *trains* 500 steps, then scores |

`experiments/table2.py` and `mqar_finetune.py` are a different species again —
they train a small model from scratch on generated data and score in-loop, with
no shard and no tokenizer. They are brought onto the shared loop in §6.

### 4.2 What changes

**Scoring is an explicit step.** Training produces checkpoints and stops. A
separate command scores a checkpoint. This matches how the work is actually
done, and it keeps a crashed eval from being indistinguishable from a failed
training run.

**Eval writes into the run directory, keyed by checkpoint.**

```
<run_dir>/eval/<checkpoint>/<task>.json
```

The checkpoint segment is required, not cosmetic: scoring three checkpoints on
one task would otherwise collide on a single filename. `--output` survives as an
override but stops being load-bearing. Today it defaults to `None`
(`evaluate.py:608`, `evaluate_retrieval.py:810`), so forgetting it prints the
numbers to stdout and loses them; `harness.py:207` spells the same flag `--out`.

**Eval reads `spec.yaml`, not `--model_size`.** This removes the current failure
mode where `evaluate.py --model_size 180m` can silently disagree with the
architecture of the checkpoint it was handed.

**Every result file carries a common envelope.** Today
`results/echo50m_table4/fineweb_ppl.json` is `{"model_type": "koopman",
"fineweb_ppl": {...}}` — no run id, no step, no commit, no timestamp. The
envelope adds `run_id`, `task`, `checkpoint`, `git_commit`, `created_at`, with
task-specific numbers nested under `metrics`. This is what makes aggregation
possible at all.

**The filesystem is the store; aggregation is a walk.**
`python -m koopman_lm.results $RUN_ROOT` reads each run's `spec.yaml` and
`eval/**/*.json` and emits one table — a row per (run, checkpoint, task), with
columns for the swept axes. No database, no service.

**The sweep grid is declared exactly once.** `scripts/slurm_array.sh:29-31`
currently hardcodes `MODEL_TYPES`/`KV_PAIRS`/`GAPS` in bash beneath a comment
reading *"must match PAPER_MODEL_TYPES, PAPER_KV_PAIRS, PAPER_GAPS in
mqar_finetune.py."* Two copies of a grid kept in sync by a comment is a drift
bug with a countdown on it. Instead `configs/sweeps/<name>.yaml` declares the
axes, a generator materializes one `RunSpec` per cell, and the array job indexes
into that materialized list. Bash never knows the grid.

### 4.3 Two storage tiers

Run directories live on scratch (`$RUN_ROOT`) and are never committed — they
hold checkpoints. A small curated `results/` **is** committed, for numbers
backing a paper table; this is the convention already established by
`results/echo50m_table4/`. The difference is that a committed result now carries
its `run_id` and so can be traced to the run that produced it.

---

## 5. Exact resume

### 5.1 Why this is in scope

`train.py` cannot resume. `--init_from` (`train.py:107`) is explicitly a
weights-only warm start — optimizer, LR schedule, and step counter are all
fresh. `_save_checkpoint` (`train.py:362`) writes `model.pt`, `meta.pt`, and the
tokenizer; no optimizer state.

**But the capability already exists twice in-tree.** `mqar_finetune.py:116`
saves optimizer and scheduler state and `:145` restores both plus the step;
`table2.py:168` and `:183` are a near byte-for-byte copy of the same pair. So
this section is a *consolidation* of three implementations — two working, one
absent — not new capability.

That duplication is itself the finding. `CONTRIBUTING.draft.md` records the same
failure mode from the branch audit: *"`ati-180m-exact-resume` and
`table4-reproduction-fix` independently wrote the same resume implementation and
the same test file, same day, same author."* It has since happened again, inside
a single branch. §6 removes the conditions that cause it.

The Marlowe partitions cap at 30 days (`hero`), 2 days (`batch`), and 12 hours
(`preempt`). The 180m recipe is 51,000 steps. On anything but `hero` that run
cannot finish, and a preemption restarts it from zero. Sweeps at scale live on
`batch` and `preempt` by necessity.

Resume belongs here rather than in a separate spec because the artifacts it
requires — the run directory, `spec.yaml`, and the attempt record — are being
defined by this design. Deferring it would mean revising that layout later.

### 5.2 What exact resume requires

| Component | Today | Needed |
|---|---|---|
| Model weights | saved | — |
| Step counter | in `meta.pt` | — |
| Optimizer state | **absent** | AdamW moments |
| LR scheduler | **absent** | derivable from step, but saved explicitly |
| RNG state | **absent** | torch, CUDA, numpy, python |
| Dataloader position | **absent** | epoch + samples consumed within it |

The dataloader position is the only subtle one, and it is tractable here.
`MemmapPackedDataset.__getitem__` is a deterministic slice at
`idx * max_seq_len + epoch_offset`, and `set_epoch` derives the offset from
`seed + epoch` (`dataset.py:46`). Sampler shuffling is seeded from
`seed_everything`. So the permutation is reproducible given `(seed, epoch)`, and
resume needs only `(epoch, samples_consumed)` plus a forward skip over indices —
index arithmetic, not data reads. Under DDP, `DistributedSampler.set_epoch` is
deterministic per rank, so the same reconstruction holds.

### 5.3 Rolling resume state, separate from archival checkpoints

Optimizer state is large. For the 180m at fp32, weights are ~720 MB and AdamW
moments ~1.4 GB. Writing that at every `save_steps=1000` over 51,000 steps would
be ~107 GB per run.

So the two concerns are separated:

- **`resume.pt`** — a single rolling file at the run root, overwritten each time.
  Optimizer, scheduler, RNG, epoch, samples consumed. Written atomically (temp
  file plus rename) so a kill mid-write cannot corrupt it.
- **`step_<N>/`** — periodic, weights-only, archival. What eval consumes.

### 5.4 Preemption handling

`SlurmLauncher` generates `--signal=B:USR1@300` and `--requeue`. The trainer
installs a `SIGUSR1` handler that writes `resume.pt` and exits cleanly; Slurm
requeues the job, and startup finds `resume.pt` and continues. Each requeue
appends to `attempts.jsonl`, so one `run_id` accumulates many attempts — which
is exactly the model §3.3 already specifies.

### 5.5 The invariant worth testing

Resume is correct when training *N* steps, killing, resuming, and training to
*2N* produces the same weights as training *2N* uninterrupted, under
`--deterministic`. That is a test, not a claim, and it is the acceptance
criterion for this section.

---

## 6. Trainer unification

### 6.1 Three loops, one algorithm

`train.py`, `mqar_finetune.py`, and `table2.py` are near-duplicates:

| | `train.py` | `mqar_finetune.py` | `table2.py` |
|---|---|---|---|
| Batches from | `MemmapPackedDataset` | `MQARDataset` (generator) | inline per-step generators, no DataLoader |
| Loss | model-internal, `loss_weights` | external CE, `ignore_index=-100` | external CE, `ignore_index=-100` |
| Resume | **none** | full | full |
| DDP / grad-accum | yes | no | no |
| Param groups | `_param_groups` no-decay policy | `model.parameters()` | `model.parameters()`, wd=0.01 |

Everything that genuinely differs is the **data source**, the **loss**, and an
optional **in-loop eval hook**. The loop, resume, DDP, gradient accumulation,
parameter grouping, logging, and checkpointing are one algorithm written three
times.

### 6.2 `TrainTask`

```
TrainTask                one loop, one resume, one param-group policy
├── ShardTask            MemmapPackedDataset; weighted CE through the model
└── SyntheticTask        curricula.py generators; masked CE; in-loop accuracy
      ├── mqar                        (mqar_finetune)
      ├── toolcall / sysprompt        (table2 training curriculum)
      └── niah                        (table2 held-out eval)
```

A task supplies batches, computes loss, and optionally exposes an in-loop eval.
It owns nothing else. `RunSpec.data.kind` selects the task (§3.1), so the
synthetic experiments become ordinary runs: they gain the run directory, the
result envelope of §4.2, sweep grids declared once instead of duplicated into
`slurm_array.sh`, and exact resume — while `mqar_finetune.py` and `table2.py`
shrink to their generators and eval logic.

This removes code rather than adding a second system.

### 6.3 The weight-decay discrepancy

`train.py:61` deliberately excludes norms, biases, embeddings, and the Mamba
state parameters (`A_log`, `D`, `dt_bias`) from weight decay, documenting that
at long schedules `weight_decay=0.1` shrinks them several-fold before gradients
are considered. `table2.py:206` instead passes `model.parameters()` flat at
`weight_decay=0.01`.

**Published Table 2 numbers were therefore produced under a different and
almost certainly unintended optimizer regime than every other result in the
repo.** This is a live inconsistency in a paper result, not a style question.

Resolution: apply the `_param_groups` policy uniformly and **regenerate Table 2**.
The existing numbers are treated as superseded, not annotated-and-kept — a
result that cannot be reproduced by the current code is a liability regardless
of how it is labelled. Because Table 2 runs at 1m scale, the re-run is cheap
relative to the ambiguity it removes.

This is also the strongest available argument for §6.2: the discrepancy exists
*because* there are three loops. One loop makes it unrepresentable.

---

## 7. Migration

`pretrain.sh` is **not** deleted in the commit that adds the Python path.

1. Land the three cherry-picks from §2 (`mamba_headdim`, the RoPE fix,
   `mqar_cell_fits`) and the 8 config YAMLs. This is a prerequisite: it turns 25
   of the 29 currently-failing tests green, so later steps land against a
   working gate rather than a red one.
2. Add `koopman_lm/run/` alongside the existing scripts. Nothing deleted,
   nothing breaks. Lands inert per `CONTRIBUTING.draft.md` rule 2.
3. Add exact resume (§5) with the kill-and-continue equivalence test of §5.5.
   Independently valuable — it is what makes the `batch` and `preempt`
   partitions usable at all — and testable before any launcher change.
4. Launch one real 50m run through the new path on the `batch` partition.
   Confirm the checkpoint loads, resume works after a real preemption, and the
   loss curve is as expected.
5. Unify the trainers behind `TrainTask` (§6.2), porting `mqar_finetune.py` and
   `table2.py` onto the shared loop.
6. Regenerate Table 2 under the corrected parameter groups (§6.3) and supersede
   the old numbers in `results/`.
7. Delete the three shell scripts, in their own commit.

Both paths work until step 7.

Steps 1, 3, and 5 each stand alone: the cherry-picks fix a red suite, resume
makes the `batch` and `preempt` partitions usable, and trainer unification
removes the duplication behind §6.3 — none of them require the launcher work to
have landed.

---

## 8. Out of scope

- `koopman_lm/experiments/phase2/` (~12.5k lines on Cody's branch). Its schemas
  are being mined — the run manifest fields, config+data hashing, runtime
  fingerprint, and results contract all informed this design — but the machinery
  itself is never-executed (`"status": "premerge_infrastructure_only"`, every
  capability flag `false`) and purpose-built for one Optuna-style study with
  governance approvals. Not adopted.
- The 9 phase2-coupled tests.
- Full content-addressed data shards (see §3.5).
- Serving or deployment inference. `models/recurrent.py` provides step-by-step
  decode for generation-based evals and parity tests; nothing here concerns
  productionising it.

---

## 9. Related work in flight

- Branch `jack/test-recovery`: 23 recovered test files, CI cpu job,
  `scripts/check_imports.py`, `scripts/slurm_tests.sh`. 208 passed / 29 failed /
  28 skipped / 1 collection error. 25 of the 29 failures are the 8 missing
  config YAMLs.
- `CONTRIBUTING.draft.md` in `/users/jkli/koopman-consolidation-plan/` proposes
  branch naming `<owner>/<topic>`, a docs policy, and rule 6 — "a finding
  becomes a test, not a document" — which this design follows by pairing each
  invariant with an assertion rather than prose.
