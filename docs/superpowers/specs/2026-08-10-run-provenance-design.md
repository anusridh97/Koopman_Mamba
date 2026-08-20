# Run provenance: making "what did this run actually do?" mechanically answerable

Written 2026-08-10. Companion to `2026-08-10-precision-policy-design.md`, but
independently implementable. Every claim about current behavior was verified by
reading or running the code, with file:line given.

---

## 1. The question

Given a finished run — a directory, or just a checkpoint — can you determine
exactly what produced it? Today: partly, and the part that's missing is the part
that matters.

Behavior is `(config values) × (code that interprets them)`. The config side is
content-addressed and self-contained. The code side is only a **pointer**.

### The four cases

| | `config_hash` | behavior | recorded by |
|---|---|---|---|
A. field added, default preserves old behavior | changed | same | — |
B. field added, default changes behavior | changed | **changed** | the materialized spec (§2) |
C. **code logic changed, no config surface** | **same** | **changed** | `code_id` only |
D. a value changed in a config | changed | changed | the config itself |

Case C is the exposed one, and for it `config_hash` is useless *by design* —
`run_id` deliberately excludes code, which is why
`run/artifacts.py:86-95`'s collision message exists to explain a code change
colliding on `run_id`.

The concrete shape of Case C: someone changes a ReLU to a SiLU in code, with no
config field for it. `config_hash` is byte-identical. Nothing in the config, the
hash, or the load path records that anything moved.

---

## 2. What is already solved — do not lose it

A materialized `spec.yaml` records **every resolved value**, not just the
overrides. Verified:

```
model keys written to spec.yaml: 56
KoopmanLMConfig fields         : 56
every field explicit?          : True
```

`run/resolve.py:to_flat_dict` does `dataclasses.asdict(spec.model)` for all four
sections, and its docstring states the intent: *"a later drift in
`configs/*.yaml` cannot retroactively change a finished run's meaning."*
`run/resolve.py:load_materialized_spec` then enforces completeness on read via
`_check_model_key_set`, so an old spec cannot be silently reinterpreted under a
newer schema — it raises, listing exactly which keys drifted.

**Consequence: Cases A, B and D need no new mechanism** for any run launched
through `experimentation.run`. A default change cannot reach backwards.

This also means `configs/*.yaml` does **not** need archiving (§4.2) — after
materialization it is irrelevant to a finished run.

---

## 3. The three remaining gaps

**3.1 Case C has only a pointer.** `run/resolve.py:materialize` stamps
`provenance.git_commit` and `code_id`, and `check_git_clean` (`resolve.py:88`)
refuses to launch from a dirty tree unless `--allow-dirty` (recorded as
`dirty: true`). So `code_id` is *trustworthy* — but it is a pointer into mutable
history, and this repo rewrites history. On 2026-08-10 commit `7d7aea3` was
force-pushed away; any run recording `code_id: 7d7aea3` would now have dangling
provenance. A pointer is strictly weaker than a copy.

Answering a question also costs a checkout, and a differing `code_id` tells you
*that* code changed, never *what* — a docs-only commit and a numerics change are
indistinguishable.

**3.2 Checkpoints carry no code identity.** `training/train.py:498-512`'s
`checkpoint_meta` records `{step, cfg, cfg_hash, model_type, model_size,
torch_version}`. No commit, no dirty flag. So a checkpoint detached from its run
directory is **unattributable** — and detachment is normal:
`evaluation/evaluate.py` takes `--checkpoint <path>`, and
`evaluation/harness.py:46-60` builds all its provenance from `meta` alone.

**3.3 Two trainers produce runs with no `spec.yaml` at all.**
`experiments/table2.py:174` and `experiments/mqar_finetune.py:129` write their own
meta dicts and never go through `experimentation.run`. Their outputs have no
materialized spec, so §2's protection does not apply to them either. This is a
second, independent reason the Table 2 and MQAR numbers need regenerating, beyond
the weight-decay bug in `HANDOFF-2026-08-08.md` §7: their provenance is not
recoverable even if the numbers were right.

**3.4 A recorded hash mismatch is discarded rather than reported.**
`evaluation/harness.py:56-60`:

```python
info = {..., "cfg_hash": meta.get("cfg_hash"), ...}
if cfg is not None:
    h = config_hash(cfg)
    info["cfg_hash"] = h        # OVERWRITES the recorded value
```

The docstring says it *"hashes it to confirm the recorded cfg_hash."* It does not
confirm; it replaces. So a checkpoint whose config schema drifted since it was
written is silently accepted and the drift is invisible.

---

## 4. The design

Four changes. Two are trivial, one is a policy, one is selective.

### 4.1 `code_id` and `dirty` into every checkpoint (closes 3.2, 3.3)

```python
# training/train.py:checkpoint_meta
from experimentation.run.provenance import git_commit, git_dirty_paths
return {..., "code_id": git_commit(), "dirty": bool(git_dirty_paths())}
```

(This doc originally said `run/resolve.py`; `3100e9c` extracted both helpers
into `run/provenance.py`, which is where they live.)

Applied to all **four** places that write a checkpoint meta dict:

```
training/train.py:507              checkpoint_meta
retrieval/adapt.py:253             inline torch.save
experiments/table2.py:174          inline dict
experiments/mqar_finetune.py:129   inline dict
```

(The last three build their own dicts rather than calling `checkpoint_meta` —
itself a duplication worth collapsing, but out of scope here.)

`meta.pt` is not hashed, so this changes **no identity** — it is purely additive.
Every checkpoint becomes self-attributing.

The import is `from experimentation.run.provenance import git_commit,
git_dirty_paths`. `training/` already imports across the package this way
(`training/resume.py` pulls `atomic_torch_save` from
`experimentation/atomic_io.py`), so this introduces no new layering edge, and it
does not touch the `koopman_lm` ← `experimentation` boundary at all.

Also fix 3.4: `harness.py` must stop **overwriting** the recorded `cfg_hash`
with its own recomputation. The recorded value describes the code that wrote the
checkpoint; substituting a hash computed by the code reading it relabels the run.

> **Revised 2026-08-20.** This section originally also called for surfacing a
> `cfg_hash_mismatch` field. That was implemented and then removed, because the
> comparison is a weak detector: adding a *defaulted* config field moves the hash
> without changing the model, so a mismatch means "named under an older schema",
> not "wrong". Schema drift belongs at the load boundary, where `resolve.py`'s
> `_check_model_key_set` already rejects a materialized spec whose key set does
> not match `dataclasses.fields(KoopmanLMConfig)` exactly. Extending that guard
> to `meta.pt` is tracked in
> `2026-08-20-checkpoint-meta-resolved-config.md`. The non-overwrite fix above
> stands on its own and is unaffected.

### 4.2 Archive the source into the run directory (closes 3.1)

A byte-reproducible tarball of the code that determines behavior, written beside
`spec.yaml`:

```
$RUN_DIR/source.tar.gz        koopman_lm/**/*.py  +  experimentation/**/*.py
```

Measured sizes: `koopman_lm/` alone is **216K**; with `experimentation/` it is
**436K** — negligible beside a multi-gigabyte checkpoint. Both are included
because the trainer affects results as much as the model does.
`configs/` is excluded for the reason in §2.

Two requirements:

- **Excluded:** `__pycache__`, `*.pyc`.
- **Byte-reproducible:** sorted member order and a fixed mtime, so
  `sha256(source.tar.gz)` is a stable content hash of the behavior-determining
  code. That hash *is* a `behavior_id` — two runs with equal archive hashes are
  provably running the same code, which is usually the question being asked
  ("were these comparable?"). This is why a separate source-hash field is not
  needed (§7).

Needs one new helper: `atomic_write_bytes` in `run/artifacts.py`, mirroring
`atomic_write_text` (temp + `os.replace`, PID in the temp name). The module today
has text, json, and torch writers but no binary one.

Recovery procedure, to be documented in the run-system design:

```bash
tar xzf $RUN_DIR/source.tar.gz -C /tmp/recovered
PYTHONPATH=/tmp/recovered python -m experimentation.run $RUN_DIR/spec.yaml --dry_run
```

No git required, no surviving-commit assumption, immune to force-pushes.

### 4.3 Policy: behavior-affecting code must have config surface

> **A change that can alter numerics must be expressible as a config field, so
> the materialized spec records it. Hidden behavior is a Case-C hazard by
> construction.**

This converts Case C into Case A/B, where §2's protection already applies. It is
the only measure that *shrinks* the problem rather than instrumenting it.

The precision policy design is precisely this rule applied once:
`modules/seq/ska.py:535`'s hardcoded fp32 is behavior with no config
representation — a Case-C hazard today. Turning it into `ska_precision` makes it
a value the materialized spec records. See
`2026-08-10-precision-policy-design.md`.

Being a rule applied by humans, its reliability is bounded by discipline. It is
listed here so the reasoning is on record, not because it is self-enforcing.

### 4.4 Golden behavior tests, selectively (catches Case C at introduction)

Everything above helps you *reconstruct* after the fact. Golden tests make Case C
**fail at the moment it is introduced**, which is far cheaper.

Fixed seed plus fixed tiny config, asserting exact output, on the three paths
where numerics matter most:

1. the SKA whitened core forward — the oracle pattern already exists
   (`kernels/prefix_scan.py:dense_exact_oracle`, `archive/reference/echo_jax.py`,
   `code-tests/test_jax_reference.py`);
2. `modules/mlp/koopman.py:_rotation_coeffs` — `_disk_clamp` and
   `exp(-softplus(s))`, which run every forward in 9 of 11 configs;
3. a full `KoopmanLM` forward at the `1m` scale, which is CPU-instantiable
   (`ska_mode='replace'` needs no `mamba_ssm`).

Following the `code-tests/test_identity_baseline.py` doctrine exactly: golden
values in a committed JSON, a generator script, and a docstring stating
*regenerate only when an intentional change has been decided and the old→new
mapping recorded — never to make a red test green.* That discipline already works
in this repo; this reuses it rather than inventing a second convention.

---

## 5. What a run directory looks like afterwards

```
$RUN_ROOT/<name>.<group_id>/seed<seed>.<run_id>/
  spec.yaml         every resolved value + provenance + code_id [+ dirty]
  source.tar.gz     the code that interpreted it            <-- NEW
  attempts.jsonl    one line per execution
  resume.pt         rolling optimizer/scheduler/RNG state
  step_<N>/         weights-only archival checkpoints
    model.pt
    meta.pt         + code_id, + dirty                     <-- NEW
  final/
  eval/**/*.json
```

Self-contained: config, code, and execution history all present, with no
dependence on git history or on `configs/*.yaml` still saying what it said.

---

## 6. Testing

All CPU-runnable, all in the `correctness` gate suite.

1. `checkpoint_meta` includes `code_id` and `dirty`; `dirty` is `True` when the
   tree has uncommitted paths.
2. `harness.py` leaves a `meta.pt`'s recorded `cfg_hash` untouched even when the
   stored cfg now hashes to something else, and only falls back to the
   recomputed hash when nothing was recorded. (Revised: no `cfg_hash_mismatch`
   field -- see the note in 4.1.)
3. `source.tar.gz` is **byte-reproducible**: archiving the same tree twice gives
   identical bytes (this is what makes the hash a usable `behavior_id`).
4. The archive excludes `__pycache__` and `*.pyc`, and contains every `.py` under
   `koopman_lm/` and `experimentation/`.
5. `atomic_write_bytes` leaves no temp file on success and does not corrupt an
   existing file on failure — mirroring
   `test_run_artifacts.py:127,140`'s coverage of `atomic_torch_save`.
6. Round-trip: extract a `source.tar.gz` to a temp dir and resolve the run's own
   `spec.yaml` against it.

---

## 7. Rejected alternatives

**A separate `behavior_id` field hashing `koopman_lm/**/*.py`.** Strictly better
than `code_id` for detection, and about fifteen lines. Rejected as redundant once
§4.2 lands: a byte-reproducible archive yields the same hash *and* the diff
besides. Worth reconsidering if archiving is ever dropped.

**A hand-maintained changelog of material default/behavior changes.** It has one
property nothing else does — it records *what* changed and *why* in human terms,
where an archive hash only says "different." But it is the lowest-reliability
option, because it depends on discipline this repo has not sustained for prose:
as of 2026-08-10, `README.md`'s layout section, `CODEBASE_GUIDE.md`'s file counts
and test numbers, and `HANDOFF-2026-08-08.md`'s references to two specs that do
not exist had all drifted. Recommended as a **supplement** to §4.2 if it will
actually be maintained; rejected as a primary mechanism.

**Hashing code into `run_id`.** Rejected: it would mean a docs-only commit
creates a new experiment identity, and `results.py` would stop grouping genuine
replicates. Code identity belongs in provenance, not identity — which is what
`run/artifacts.py:86-95` already says.

**Archiving `configs/`.** Unnecessary; see §2.

---

## 8. Non-goals

**Environment capture beyond versions.** `provenance()` records torch, CUDA and
Python versions. Full environment capture (a lockfile, a container digest) is a
larger change and is not attempted here. Noted because the fused CUDA kernel's
behavior depends on the toolchain that built it, which the versions only
partially pin.

**Reconstructing provenance for existing runs.** The one real run
(`50m-first-real.81033b58/seed42.5467934f`, job 415896) predates this design; its
`spec.yaml` has `code_id` but no archive. Nothing here retroactively fixes that,
and the `table2.py`/`mqar_finetune.py` outputs have no `spec.yaml` at all.

**Detecting behavior changes in third-party code.** `mamba_ssm` version changes
can alter numerics; only the version string is recorded, and archiving
site-packages is out of scope.

---

## 9. Implementation order

Each step is independently landable and independently verifiable.

1. **`atomic_write_bytes`** in `experimentation/atomic_io.py` + its tests. Nothing
   consumes it. (This doc said `run/artifacts.py`; `f0e3dbd` split that file into
   `run/write_policy.py` and `experimentation/atomic_io.py`, and the atomic writers
   went to the latter.)
2. **`code_id`/`dirty` in the three `checkpoint_meta` writers** + the
   `harness.py` comparison fix. No identity change; smallest useful increment.
3. **`source.tar.gz`** in `materialize()`, with the reproducibility test. Gate:
   the round-trip in §6.6.
4. **Golden behavior tests** (§4.4), one path at a time, easiest first (the MLP
   rotation coefficients need no model instantiation).

Step 2 alone closes the gap that motivated this document; step 3 is what makes
the answer survive a force-push.
