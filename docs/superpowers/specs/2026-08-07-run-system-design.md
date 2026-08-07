# Run system design — configs, launch, and run identity

Status: **in progress.** Sections 1–3 are settled with the design owner. The
results/eval layer (§4) is still being designed and will be appended before this
spec goes to review.

Owner: jkli. Written 2026-08-07 during a design session on branch
`reorg-module-layout`.

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
- `data: DataSpec` — shard reference, plus the tokenizer/mix/token-count it must match
- `optim: OptimSpec` — lr, warmup, schedule, effective batch, weight decay, grad clip
- `runtime: RuntimeSpec` — seed, precision, ddp, partition, account, QoS, workers

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
run_id = sha256(model + data + optim + seed)[:8]
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
  spec.yaml        fully resolved
  launch.sbatch    generated
  attempts.jsonl   one line per execution
  step_1000/ step_2000/ final/
  eval/
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

---

## 4. Results and evaluation layer

Not yet designed. To be appended.

---

## 5. Migration

`pretrain.sh` is **not** deleted in the commit that adds the Python path.

1. Add `koopman_lm/run/` alongside the existing scripts. Nothing deleted,
   nothing breaks. Lands inert per `CONTRIBUTING.draft.md` rule 2.
2. Launch one real 50m run through the new path on the `batch` partition.
   Confirm the checkpoint loads and the loss curve is as expected.
3. Delete the three shell scripts, in their own commit.

Both paths work until step 3.

---

## 6. Out of scope

- `koopman_lm/experiments/phase2/` (~12.5k lines on Cody's branch). Its schemas
  are being mined — the run manifest fields, config+data hashing, runtime
  fingerprint, and results contract all informed this design — but the machinery
  itself is never-executed (`"status": "premerge_infrastructure_only"`, every
  capability flag `false`) and purpose-built for one Optuna-style study with
  governance approvals. Not adopted.
- The 9 phase2-coupled tests.
- Full content-addressed data shards (see §3.5).
- Sweep expression and the aggregation layer over many run directories — depends
  on §4.

---

## 7. Related work in flight

- Branch `jack/test-recovery`: 23 recovered test files, CI cpu job,
  `scripts/check_imports.py`, `scripts/slurm_tests.sh`. 208 passed / 29 failed /
  28 skipped / 1 collection error. 25 of the 29 failures are the 8 missing
  config YAMLs.
- `CONTRIBUTING.draft.md` in `/users/jkli/koopman-consolidation-plan/` proposes
  branch naming `<owner>/<topic>`, a docs policy, and rule 6 — "a finding
  becomes a test, not a document" — which this design follows by pairing each
  invariant with an assertion rather than prose.
