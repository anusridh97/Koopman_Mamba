# Measurement provenance: an index for facts that are already recorded

**Status:** design, revision 3. Not implemented.

**Revision history, because it is the argument.** Revision 1 was rejected: its
motivating premise was false and two of its four proposals duplicated machinery
that already existed. Revision 2 was rejected again: its replacement guard
rejected all three of its own examples, its replacement flagship example
committed the same miscitation revision 1 was rejected for, and three of its
claims about the run store were contradicted by the store.

**Every factual claim in revision 3 was produced by a command whose output was
read, not from memory.** That is the specific fix, because the specific defect in
revisions 1 and 2 was the same both times: numbers derived by hand, expensive to
re-derive, therefore never re-derived. A spec arguing that unverified numbers are
the problem cannot contain unverified numbers.

## 1. The failure

Measured facts in this repo *are* recorded. Every job id in the table below is
cited in tracked files that were committed **before** revision 1 was written
(counted at `bb91498`, excluding the spec itself):

| measurement | job | tracked files citing it |
|---|---|---|
| sigma = 9.50e-3 @ 600 steps | 445832 | 5 |
| sigma = 7.54e-3 @ 1500 steps | 445994 | 2 |
| layerscale -> loss, 4 points | 445689 | 12 |
| the loop closes, 4 trials | 445657 | 4 |

They are real measurement statements, not bare numbers -- e.g.
`configs/search/beta-policy-1500.yaml:35` reads "Job 445832 measured sigma =
9.50e-3 across five seeds at 600 steps on this base".

So the problem is **not** that facts go unrecorded. It is that they are recorded
in whichever artefact their author happened to be editing -- a YAML header, a
module docstring, a test docstring, a commit message -- with **no index**. A
reader who does not already know which file to open cannot find them. That is
how revision 1's author came to re-derive facts sitting in tracked files two
directories away.

Exactly one number from that table is genuinely uncited anywhere in the tree: the
SKA ablation delta, 0.025139 [job 445994].

**The corrected diagnosis points at an index and away from new capture
machinery.** Revisions 1 and 2 proposed both; only the index and its supporting
tool survive.

## 2. What already exists

Verified against the store (117 `spec.yaml` files under
`/scratch/m000151-pm06/jkli`) and the code:

**Runs record their conditions.** `spec.yaml` carries `run_id`, `group_id`,
`code_id`, `study_name`, `trial_number`, `anchor_name`, a `provenance` block, and
the full resolved `model` / `data` / `optim` / `runtime`.

**`results.py` walks the store** -- "the filesystem is the store, and this is a
walk over it" -- one row per (run, checkpoint, task), with 11 hardcoded axis
columns.

**`group_id` groups seeds.** `group_id = sha256(identity_payload(model) + data +
optim[minus empty groups] [+ schedules when non-empty])`, seed excluded. Verified:
group `c7b818ff` -> 5 runs, seeds 42-46. Note `identity_payload`, not `asdict`:
identity-transparent defaults are dropped (`koopman_lm/config.py:51`), which
matters for §3.2 item 3.

**Aggregation exists on a richer source.** `analysis.noise_floor` computes sigma
from the optuna journal, which carries FAILED and PRUNED trials that a filesystem
walk cannot see, and returns `{"available": False, "reason": ...}` for a
degenerate group rather than raising. `analysis.py::anchor_contrasts` emits
`sigma*sqrt(1+1/n)`; `scripts/analyze_beta_policy.py` tabulates all three
denominators in its docstring and emits two (`sigma*sqrt(2)` and
`sigma*sqrt(2/n)`).

**`attempts.jsonl` records that an attempt STARTED, and nothing about how it
ended.** All 104 records on the store carry exactly `{timestamp, host, job_id,
git_commit, code_id, forced}` (102 records; 2 pre-date `code_id`). There is no
exit status, no failure reason, and no microbatch. 19 of 117 run directories have
no `attempts.jsonl` at all -- and all 19 *do* have `eval/`.

**`test_docs_are_not_stale.py` shows what a working guard looks like.** It does
not check syntax; it asserts `claimed_count == len(CONFIG_REGISTRY)` -- the claim
compared against the thing it claims about. Its docstring records why: the config
count "has now been wrong twice".

## 3. Scope

Three units. **Build strictly in this order**, because each makes the next
cheaper and safer.

### 3.1 Generated maps -- build first, independent of everything else

Three tables were built by hand this week and would otherwise be rebuilt. Each
becomes a generator plus a byte-exact regenerate-and-diff test, following
`code-tests/test_proxy_anchor_design.py:390-398`, which diffs
`docs/proxy-256x17-anchors-resolved.md` and puts the regenerate command in the
failure message.

| map | answers |
|---|---|
| config -> SKA route | which of the 11 registry configs run chunked vs an exact route |
| route -> core -> computes alpha | which backend calls `spec_w`, and via which core |
| study axis -> RunSpec field(s) | which dotted keys each sampled axis writes |

The third is **many-to-many and the generator must show that**:
`norm_clip_multiplier` -> `model.ska_norm_clip_c` depends on `ska_rank`, and
`n_ska_layers` + `placement` jointly write `model.ska_layer_indices` via
`geometry.make_layer_indices`. A one-to-one table would be a new false claim.

Self-contained, mechanical, no open questions. This is the piece that would have
surfaced the nine-of-eleven-configs-run-chunked finding as a table nobody had to
think to look for.

### 3.2 `results.py`: derive axis columns, and add `--group-by group_id`

**Why:** `max_steps` is not among the 11 hardcoded axis columns, so a 600-step
and a 1500-step run are indistinguishable in the table. That is not hypothetical.
Writing revision 3, I grouped the chunked-route ablation cells and got
`chunked-cs64` mean loss 5.05 against 4.39 for the others -- because job 446346
is a **20-step smoke run** (loss 9.04) that my grouping pooled with the 1500-step
trials. Nothing in the table would have told me. Filtering to `max_steps == 1500`
changed the cell's ablation delta from 0.011738 to 0.013693 and its t from 5.38
to 6.81.

**The rule:** emit an axis column only when that field varies within the queried
root, after excluding bookkeeping. Measured over the store:

| | keys |
|---|---|
| flattened keys, whole store | 103 |
| strict-varying (JSON-canonical), whole store | 41 |
| of those, `provenance.*` | 2 (`git_commit`, `materialized_at`) |

`provenance.materialized_at` varies in **every** multi-run root, including a pure
seed-replicate group, so without an exclusion the rule emits a timestamp column
everywhere -- and for the six `golden-*` / `resume-golden-*` roots it is the
*only* varying key. With the exclusion set below, measured per-root widths:

```
root                          runs   raw  clean
resume-golden-440211             3     1      0
study-smoke-445832               5     5      1     <- seed only
study-smoke-445994               5     5      1     <- seed only
study-smoke-445689               4     6      1
study-smoke-446074              24    12      5
study-smoke-446055               7    16     11
```

Distribution of clean width over the 19 multi-run roots: `{0: 6, 1: 3, 5: 6,
9: 1, 10: 2, 11: 1}`. **Max 11.** So width does track the question -- but only
with the exclusion; revision 2 claimed the naive rule already gave "1 column for
a replicate group", and it gives 5.

**Exclusion set** (identical to the always-emitted identity set, pinned by a test
so the two cannot drift): `{run, name, run_id, group_id, code_id, sweep_id,
sweep_name, study_name, trial_number, anchor_name}` plus `provenance.*`.
Note `code_id` and `sweep_id` are **not** currently emitted by `aggregate()`;
adding them is part of this change, not an existing property. `sweep_id` is
absent from every `spec.yaml` on the store, so it will be a column of blanks
until something writes it.

**Four details the implementation must settle**, each a measured defect:

1. **Canonically scalarize list/dict values.** Exactly two keys on the store hold
   them: `model.ska_layer_indices` and `optim.groups`. `len(set(values)) > 1`
   raises `TypeError: unhashable type: 'list'`. Use
   `json.dumps(v, sort_keys=True, default=str)` for the varies-test and any group
   key. `data.mix` is a *dict* that the flattener descends into, so it never
   reaches `set()` -- item 4 governs it, not this one.
2. **Run the migrations before comparing.** `results.py` reads raw
   `yaml.safe_load` and never calls `_migrate_microbatch`, so
   `optim.per_device_batch_size` (legacy) and `runtime.per_device_batch_size`
   both appear -- two mostly-empty columns for one concept.
3. **Normalize, then count.** Absent, or equal to an identity-transparent
   default, both count as the default value; emit a column iff >= 2 distinct
   values survive. Revision 2 stated this as "a field added later is constant",
   and named `model.ska_beta_policy` as the example. That is false: it has **four
   distinct values on the store** (`learned` 28, `head_scalar` 8, `one` 7,
   `linear` 6, absent 68), and implemented as revision 2 wrote it the rule would
   have deleted the beta-policy study's own axis. Real single-value examples are
   `runtime.deterministic` and `optim.groups` -- and both sit outside `model`, so
   "identity-transparent" (a `KoopmanLMConfig` concept) does not even reach them.
   The normalization must handle absent-vs-default generally, not via that hook.
4. **Collapse nested dicts to one column per concept.** `data.mix` flattens to
   three keys; one mix change should be one column.

`--all-axes` prints the excluded and constant columns, for confirming what the
constant was.

**`--group-by group_id`**, ~40 lines on top of the above: emit `n`, `mean`, `sd`,
`SEM` per (group_id, checkpoint, task, metric), with `sd`/`SEM` as `None` at
n = 1, deduplicating on `run_id` so the same group appearing in two roots is one
group. Named metric, or an argument -- a sigma over `peak_memory_gib` is
meaningless.

**No resolvable-effect verdict.** There are already two conventions in the tree
(`_RESOLVE_SIGMAS = 2.0` in `analysis.py`; `2.39` family-wise corrected in
`analyze_beta_policy.py`). A third would manufacture the disagreement the index
exists to prevent. The tool reports spread; interpretation stays in §3.3 and in
the existing analysis modules.

**Why this is in scope at all:** without it, the path from a number in the index
to its raw data is *job id -> root directory by an undocumented naming
convention -> N JSON files -> write your own script*. That is the re-derivation
cost §1 exists to eliminate, and it is what produced revision 2's miscitation and
this revision's 20-step contamination.

**Explicitly out of scope:** a state column for failed and pruned runs. 26 of 117
run directories have no `eval/`, but **16 of those have `final/`** -- they trained
to completion and were simply never evaluated by this harness. Only **10** are
genuine early terminations. Revision 2 called all 26 "failed and pruned" and hung
a policy conclusion on the inflated figure. The honest structural point stands --
rows exist only for evaluated runs -- but the state itself is not available:
`attempts.jsonl` has no exit status, and adding one is a capture change of
unknown size. The optuna journal has it, which is where `analysis.noise_floor`
already reads it.

### 3.3 The index: `RETRACTIONS.md` and generated numbers -- build last

Revision 2 proposed one hand-written `MEASUREMENTS.md` whose numbers were guarded
by a citation pattern. Two reviews established that this cannot work: a syntax
check can verify a marker is *present* but never that the number *came from* it,
which is exactly the failure it was written to catch, and which revision 2's own
flagship example committed.

So split the file by what is mechanically checkable.

**`docs/RETRACTIONS.md`** -- hand-written, and the genuinely ungenerable part.
Every RETRACTS line in revisions 1 and 2 contained **no numbers at all**: prose
plus a date. So the guard is *"this file contains no numeric literals except a
leading ISO date per entry"* -- trivially enforceable, ungameable, and it makes
miscitation impossible by construction.

```
2026-08-24  "longer trials will fix the noise floor."
  Refuted by measurement at the same horizon it proposed. See the sigma-vs-horizon
  row in the generated index.

2026-08-24  "the chunked route's SKA contributes nothing measurable."
  Named as the cleanest possible statement of the problem before measuring it.
  The route retains most of the exact route's contribution; see the route-ablation
  rows.

2026-08-24  "SKA is not established as load-bearing; the delta is below the floor."
  Judged a five-run mean against a between-trial floor -- the wrong statistic.

2026-08-25  "measured facts live nowhere a later reader would find them."
  This spec's own first draft. The facts were cited in tracked files at the time.
```

I ran the guard against that block while writing it. It failed: "revision 1"
contributed a numeric literal. Fixed by rewording to "first draft" rather than by
loosening the guard -- which is the choice revisions 1 and 2 got wrong twice, each
having written a guard they never ran against their own examples. Referring to
drafts by name rather than number is a small constraint the rule imposes, and it
is worth paying for a check that cannot be gamed.

**The numbers: generated, not written.** `docs/measurements/index.generated.md`,
produced by `results.py --group-by group_id` over the roots named in a small
committed input file, with the same byte-diff test as §3.1's maps. A number in
the index is derived from the store on every regeneration, so it cannot be
miscited -- there is no hand step to get wrong.

**Retention.** `/scratch` is cleanable, so the regenerate test must *skip* a
missing root, never fail on it, or it becomes a test of filesystem retention. The
committed snapshot is the durable artefact; the run directory is perishable.

**What is deliberately lost:** a hand-written interpretation sitting beside each
number. That was revision 2's main appeal and it is what made it unfixable.
Interpretation lives in `RETRACTIONS.md` (what we stopped believing) and in the
docstrings that already carry these facts next to the code they constrain --
which, per §1, is where they are already being written.

## 4. Withdrawn, with reasons

**`slurm_job_id` in the `provenance` block.** The field already exists in
`attempts.jsonl`, and revision 1's location would have been *worse*:
`materialize()` and `append_attempt` both run inside `claim_run_dir`, before
`launcher.submit`, so on the Slurm path they execute where `SLURM_JOB_ID` is
unset -- null exactly for the batch runs the field exists to trace. The OOM
ladder also re-materializes `spec.yaml` per rung on one directory, overwriting it,
where `attempts.jsonl` appends. Revision 1's *safety* claim was true and is worth
the record: `provenance` is outside `_scientific_payload` and is stamped after
identity is computed, so nothing in it can move `run_id` or `group_id`.

**One real gap, promoted into §3.2's scope:** `SlurmLauncher.submit` receives
sbatch's job id (`launchers.py:328-330` returns `result.stdout.strip()`) and the
caller discards it. Appending it to the attempt record is ~10 lines, and job ids
are the index's entire citation currency. Note the corollary: every job id in the
104 existing attempt records is the *supervisor's* job, not the training run's --
they coincide today only because the search driver uses `LocalLauncher` inside one
allocation.

**A new group aggregator with verdicts.** Withdrawn at Jack's call, and three
independent objections support it. (a) Revision 1's "a group with two members at
one seed must raise" fires on correct data: group `c7b818ff` appears in
`study-smoke-445994` and `study-smoke-446074` with identical five `run_id`s, a
deliberate bit-for-bit reproduction documented in commit `348c8f3`. (b) It
misdescribed its precedent -- `analysis.noise_floor` returns `available: False`
rather than raising, which is better, because raising aborts a whole index over
one group. (c) It would have emitted the two statistics that produced a judgement
`RETRACTIONS.md` now records, while omitting the mean-vs-mean case that replaced
it. §3.2's `--group-by` is the surviving remnant: spread without verdicts.

**A general stale-comment sweep.** Withdrawn. Revision 1 offered arithmetic for
this and the arithmetic was wrong twice over -- 18% of the short stratum is ~3.9%
of blocks, not the "6%" claimed, and the <=45-character stratum it sampled is
7.1% of comment *lines* but 22-32% of comment *blocks* depending on the
tokenizer, so the "7.1% coverage" rebuttal swapped denominators mid-argument.
Neither revision defined "comment block" or how "short" was measured, so none of
those figures is reproducible.

**The conclusion stands on an argument that needs no arithmetic:** the deletion
criterion is undecidable, this repo's comments frequently record a bug that
recurred, and deleting one of those is invisible until the bug returns. Anyone
wanting the empirical case must draw a stratified sample covering the long
blocks, which nobody has done.

Two mechanical deletions survive, if anyone wants them: banner-only blocks
(define the pattern -- 1 strictly, 39 loosely) and the five `# frozen: use
replace` occurrences.

Renaming was considered and dropped: `Gf`, `Mf`, `Lf`, `qf` match the paper's
notation, which is right in code whose correctness argument is a derivation.

## 5. Testing

- **maps** (§3.1): byte-exact regenerate-and-diff per map; the axis map must
  express many-to-many rather than flattening it.
- **derived columns** (§3.2): a root where one field varies emits one column for
  it; a root where it is constant emits none; a list-valued field does not raise;
  legacy and current microbatch names collapse to one column; an
  absent-in-old-specs field at its default produces no column while
  `ska_beta_policy` (four values) does; `provenance.*` never appears;
  `--all-axes` emits both sets; the exclusion set and the identity set are the
  same object. **Mutation:** restore the hardcoded 11-key list and the
  varying-field test must fail.
- **`--group-by`** (§3.2): a five-seed group yields n=5 and a finite SEM; an n=1
  group yields `None` rather than 0; the same group in two roots is deduplicated
  to one; runs at different `max_steps` are not pooled. **Mutation:** drop the
  `max_steps` column and the not-pooled test must fail -- this is the defect that
  contaminated this revision's own recomputation.
- **`RETRACTIONS.md`** (§3.3): contains no numeric literal except a leading ISO
  date per entry.
- **index** (§3.3): regenerate and diff; a missing root skips rather than fails.
- No golden and no `identity_baseline.json` may be touched. Nothing here goes
  near `_scientific_payload`.

## 6. Risks

**The index can cite runs `/scratch` no longer holds.** Accepted; the skip-on-
missing rule is the defence, and it is a deliberate weakening -- a cleaned store
means the index is unverifiable, not wrong.

**`RETRACTIONS.md` is still hand-written.** Its guard makes miscitation
impossible but cannot make an entry *true*. The mitigation is that it records
what we stopped believing, which is self-attesting in a way a measurement is not.

**A wide table on a mixed root.** Correct behaviour, not a defect.
`RUN_ROOT=$SCRATCH/<name>-${SLURM_JOB_ID}` is one root per **job**, not per
study, so cross-horizon questions require walking a parent of several roots. The
mixed-root case is normal and the docs should say so rather than pretending
per-study is the only query.

## 7. Appendix: the recomputation that produced §3.2's example

Chunked-route ablation, filtered to `max_steps == 1500`, grouped by route and
chunk size, every contributing job named:

```
cell             n   mean loss   abl mean      SEM      t   jobs
chunked-cs16     5     4.39385   0.020148 0.001351  14.91   446369,446370,446371,446418,446419
chunked-cs64     6     4.39014   0.013693 0.001151  11.90   446366,446367,446368,446415,446416,446417
exact-invchol    6     4.39030   0.024418 0.001075  22.71   446363,446364,446365,446412,446413,446414

dropped (wrong horizon): job 446346 at max_steps=20
loss spread across cells: 0.00371
floor 2*sigma*sqrt(2/6), sigma=7.5373e-3 [job 445994]: 0.00870
chunked-cs16   82.5% of exact   diff +0.004270  t=2.47
chunked-cs64   56.1% of exact   diff +0.010725  t=6.81
```

Revision 2's entry 3 claimed 84%, t = 1.25, and a 0.00295 spread, citing
`[job 446363-446371]` -- a range that omits jobs 446412-446419 entirely, and
which is the wrong source for the floor (that is job 445994's sigma). The
corrected figures above are what a generated index would have produced without a
hand step. The qualitative finding is unchanged and now better supported: the
loss spread (0.00371) is below the n=6 floor (0.00870) while the ablation deltas
separate strongly.
