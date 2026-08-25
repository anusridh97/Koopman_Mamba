# Measurement provenance: an index for facts that are already recorded

**Status:** design, revision 2. Not implemented.
**Revision 2 exists because revision 1's motivating premise was false.** An
adversarial review checked it against the repo and the live run store; §1 below
is rewritten against what it found, and two of the four proposed changes are
withdrawn. The diff between the two revisions is itself the argument for the
surviving parts.

## 1. The failure -- corrected

Revision 1 claimed measured facts "live nowhere a later reader would find". That
is false, and checkable in one command. Every job id from revision 1's own
motivating table is cited in tracked files, committed before the spec was
written:

| measurement | job | cited in |
|---|---|---|
| sigma = 9.50e-3 @ 600 steps | 445832 | 6 tracked files |
| sigma = 7.54e-3 @ 1500 steps | 445994 | 3 |
| layerscale -> loss, 4 points | 445689 | 13 |
| the loop closes, 4 trials | 445657 | 5 |

`configs/search/beta-policy-1500.yaml`, `code-tests/test_noise_floor_analysis.py`,
`scripts/analyze_beta_policy.py`, and others. Only one number from that table --
the SKA ablation delta, 0.025139 -- is genuinely uncited anywhere in the tree.

**So the problem is not that facts go unrecorded. It is that they are recorded
ad hoc and scattered, with no index.** A fact lands in whichever artefact its
author happened to be editing -- a YAML header, a module docstring, a test
docstring, a commit message -- and a reader who does not already know which file
to open cannot find it. That is why revision 1's author re-derived facts that
were, at that moment, sitting in tracked files two directories away.

Revision 1's supporting evidence was itself the failure it described. Its count
-- "30 of 41 measurement-shaped comment blocks cite nothing" -- covered `#`
comments in `koopman_lm/` and `experimentation/` **only**, excluding `configs/`,
`scripts/`, `code-tests/` and all 581 docstrings in the same two packages: the
exact places the citations turned out to live. The number was asserted from a
biased sample and presented as a measurement. That is the same shape as the
defect this repo has been correcting all week -- `ska.py`'s "~100% RELATIVE ERROR
... measured", which traced to commit `40f6653`, "Add files via upload", with no
harness and no job id behind it.

The corrected diagnosis points at **an index** and away from new capture
machinery. Revision 1 proposed both; only the index survives.

## 2. What already exists

Verified, not assumed:

**Runs record their own conditions.** A materialized `spec.yaml` carries
`run_id`, `group_id`, `code_id`, `study_name`, `trial_number`, `anchor_name`, a
`provenance` block, and the full resolved `model` / `data` / `optim` / `runtime`.

**`attempts.jsonl` records the launch.** One append-only record per attempt:

```json
{"timestamp": "2026-08-25T02:11:58Z", "host": "n17", "job_id": "445994",
 "git_commit": "48148123...", "code_id": "48148123...", "forced": false}
```

`write_policy.make_attempt_record` already takes `job_id`, and every launch path
already passes `SLURM_JOB_ID` into it.

**`experimentation/results.py` walks the store** -- "the filesystem is the store,
and this is a walk over it" -- emitting one row per (run, checkpoint, task).

**`group_id` groups seeds.** `group_id = sha256(model + data + optim [+
schedules when non-empty])` excludes the seed, so seed-replicates share one.
Verified: group `c7b818ff` -> 5 runs, seeds 42-46.

**Aggregation already exists, on a richer source.** `analysis.noise_floor`
computes sigma and resolvable effects from the optuna journal -- which carries
FAILED and PRUNED trials, `model_seed` and `reference_group`, none of which a
filesystem walk can see. `scripts/analyze_beta_policy.py` already lays out all
three comparison denominators (`sigma*sqrt(2)`, `sigma*sqrt(1+1/n)`,
`sigma*sqrt(2/n)`) with a family-wise correction.

**`test_docs_are_not_stale.py` encodes the philosophy**: pin only mechanically
checkable claims, never wording, because "a test that pinned wording would fail
on every edit and get deleted, which is worse than no test."

## 3. The three changes, in build order

### 3.1 Generated maps -- build first

Three tables were built by hand this week and would otherwise be rebuilt. Each
becomes a generator plus a regenerate-and-diff test, following
`docs/proxy-256x17-anchors-resolved.md`, which already works this way and whose
inputs are committed YAML.

| map | answers |
|---|---|
| config -> SKA route | which of 11 registry configs run chunked vs an exact route |
| route -> core -> computes alpha | which backend calls `spec_w`, and via which core |
| study axis -> RunSpec field(s) | which dotted keys each sampled axis writes |

The third is **many-to-many and the generator must show that**, not flatten it:
`norm_clip_multiplier` -> `model.ska_norm_clip_c` depends on `ska_rank`, and
`n_ska_layers` + `placement` *jointly* write `model.ska_layer_indices`. A
one-to-one table here would be a new false claim.

This piece is self-contained, mechanical, needs nothing else in this spec, and is
the one that would have surfaced the nine-configs-run-chunked finding months
earlier -- as a table nobody had to think to look for.

### 3.2 `docs/MEASUREMENTS.md` -- the index, and the highest-value piece

Hand-written, small, and deliberately **not** a store of numbers. It is a map
from a claim to the artefact that supports it, plus the interpretation and
retraction that generation cannot produce.

```
sigma does not shrink materially with horizon.
  9.50e-3 [job 445832] @600 steps -> 7.54e-3 [job 445994] @1500: a 21% reduction,
  not the order of magnitude predicted. sigma is a near-constant 0.17-0.18% of
  the loss at both horizons, so more steps buys resolution only as fast as it
  buys loss.
  RETRACTS: "longer trials will fix the noise floor" (2026-08-24).

SKA is load-bearing on the 256x17 proxy.
  ablation delta 0.025139 +- 0.002976 SEM over 5 seeds, t = 8.4
  [group c7b818ff, job 445994].
  RETRACTS: "not established, the delta is below the floor" -- that judged a
  five-run mean against a between-trial floor, the wrong statistic.

The chunked route is ~100% wrong on the mechanism and invisible in LM loss.
  ablation delta retains 84% of exact at CS=16, t = 1.25 [job 446363-446371];
  cells span 0.00295 against a 0.0213 floor.
  RETRACTS: "chunked SKA contributes nothing measurable" (2026-08-24).
```

**The retraction lines are the point.** Four claims died in one week; a reader
seeing only the surviving prose re-derives the dead ones at GPU cost. Nothing
else in the repo records that a claim was killed, or by what.

**The guard, and why revision 1's was useless.** Revision 1 proposed "each entry
cites a `job \d{6}`". Its own flagship example passed that guard while being
wrong: it cited job 445657 for a figure that came from job 445689, and a date
like `2026-08-24` contains four consecutive digits that a loose pattern matches.

So: **bind the citation to the number, not to the entry.** Every numeric literal
must be followed within the same sentence by a `[job NNNNNN]`, `[run_id ...]` or
`[group_id ...]` marker. The test extracts (number, citation) pairs and fails on
any number lacking one. Prose, headings and dates live outside entries and are
exempt.

A second check, cheap and worth having: every cited job id must appear somewhere
else in the tree, or in `sacct`. A citation to a job that never existed is the
failure this file is for.

### 3.3 Widen `results.py`'s axis columns -- derived, not hardcoded

`aggregate()` hardcodes 11 axis keys. `max_steps` is not among them, so **a
600-step and a 1500-step run are indistinguishable in the table today** -- the
exact comparison the sigma work turned on.

Adding a column per field does not scale (60 + 8 + 13 + data + top-level ~ 88,
nearly all constant in any query). So: **emit an axis column only when that field
varies within the queried root.** Empirically validated by the review: a
whole-store walk yields 37 varying fields of 96; a single study root yields 3-8.

Five details the implementation must settle, each a real defect found in review:

1. **List-valued fields must be canonically scalarized.**
   `model.ska_layer_indices` varies across the store and round-trips as a Python
   `list`; the natural `len(set(values)) > 1` raises `TypeError: unhashable type`.
   `optim.groups` and `data.mix` have the same shape. Use
   `json.dumps(v, sort_keys=True)` for both the varies-test and any group key.
2. **Run the migrations before comparing.** `results.py` reads raw
   `yaml.safe_load` and never calls `_migrate_microbatch`, so
   `optim.per_device_batch_size` (legacy) and `runtime.per_device_batch_size`
   both appear and both register as varying -- two mostly-empty columns for one
   concept.
3. **Absent is not varying.** A field added later and identity-transparent
   (`model.ska_beta_policy`) is absent in older specs and at its default in newer
   ones. That is constant, and must not produce a column.
4. **Nested dicts must not explode.** `data.mix` flattens to three columns; one
   mix change should be one column.
5. **`code_id` is in the always-emitted identity set**; do not also emit it as a
   varying column.

**This is a tool, not a committed artefact.** Revision 1 proposed committing a
generated index with a byte-diff test; that is incoherent, because the column set
would be a function of `/scratch` contents, which are not in the repo and change
on every launch. The `anchors-resolved.md` precedent works precisely because its
inputs are committed. A `--all-axes` flag prints constant columns for the case
where someone needs to confirm what the constant was.

**Query scope, stated honestly:** `RUN_ROOT=$SCRATCH/<name>-${SLURM_JOB_ID}` is
one root **per job**, not per study, so cross-horizon questions (600 vs 1500
steps) require walking a parent of several roots. The mixed-root case is normal,
and a wide table there is correct behaviour rather than a defect.

**Failed and pruned runs.** 26 of 117 run directories on the store today have a
`spec.yaml` and no `eval/` -- 22%. `aggregate()` yields no rows for them, so a
study that pruned 40% of its trials shows a survivorship-biased picture with no
sign of it. Emit a state column sourced from `attempts.jsonl`, and count
attempted-versus-completed. `attempts.jsonl` is also the right source for the
per-attempt microbatch: the OOM ladder re-materializes `spec.yaml` per rung on
one directory, so the surviving file shows only the final rung.

## 4. Withdrawn, with reasons

**Recording `slurm_job_id` in the `provenance` block.** Withdrawn: the field
already exists in `attempts.jsonl`, and revision 1's proposed location would have
been *worse*. `materialize()` runs before `launcher.submit()`, so on the Slurm
path it executes on the login node where `SLURM_JOB_ID` is unset -- null exactly
for the batch runs it exists to trace -- while `SlurmLauncher.submit` captures
sbatch's stdout containing the real job id and discards it. And the OOM ladder
overwrites `spec.yaml` per rung, where `attempts.jsonl` appends.

Revision 1's *safety* claim was true and is worth keeping on record:
`provenance` is not part of `_scientific_payload` and is stamped after identity
is computed, so nothing in it can move `run_id` or `group_id`.

The one real gap here is small and separable: `SlurmLauncher.submit` should
append the job id it already receives to the attempt record.

**A new group aggregator.** Withdrawn, and this was the user's call. Three
independent objections: (a) revision 1's "a group with two members at one seed
must raise" fires on correct data that exists right now -- group `c7b818ff`
appears in two roots with the same five `run_id`s, a deliberate bit-for-bit
reproduction documented in commit `348c8f3`; (b) it misdescribed its own
precedent, since `analysis.noise_floor` returns `{"available": False, "reason":
...}` rather than raising, which is better behaviour because raising aborts a
whole index over one group; (c) it would emit the two statistics that produced a
judgement `MEASUREMENTS.md` retracts, and omit the mean-vs-mean case
(`sigma*sqrt(2/n)`) that replaced it -- while `scripts/analyze_beta_policy.py`
already emits all three, correctly, from the optuna journal.

There are already two "resolvable effect" conventions in the tree
(`_RESOLVE_SIGMAS = 2.0` in `analysis.py`, `2.39` family-wise corrected in
`analyze_beta_policy.py`). A third would manufacture the disagreement the index
exists to prevent. **Aggregation stays where it is; `MEASUREMENTS.md` cites it.**

**A general stale-comment sweep.** Withdrawn, but revision 1's evidence for
withdrawing it was weaker than it looked and the honest version is stated here.

Revision 1 hand-classified 45 blocks drawn from the 188-197 blocks of <=45
characters, and found ~18% low-value. **That stratum holds 216 of 3025 comment
lines -- 7.1% of the text.** The categories it reported (tensor-shape
annotations, banners, math annotations, terse one-liners) are definitionally
short-block categories; the sample frame guaranteed they would dominate, and it
could not see the long blocks where duplicated rationale would live. The
arithmetic was also wrong: 18% of the short stratum is ~3.9% of blocks, not the
"6%" claimed.

So the conclusion -- don't sweep -- **stands on a different argument**: the
deletion criterion is undecidable, this repo's comments frequently record a bug
that recurred, and the failure mode of deleting one is invisible until the bug
returns. If anyone wants the empirical case, it requires a stratified sample
covering the 672 long blocks, which nobody has done.

Two mechanical deletions survive: banner-only blocks (define the pattern -- 1
strictly, 39 loosely) and the five `# frozen: use replace` occurrences (one
original, four duplicates).

Renaming was also considered and dropped: `Gf`, `Mf`, `Lf`, `qf` match the
paper's notation, which is right in code whose correctness argument is a
derivation.

## 5. Testing

- **maps**: regenerate and diff, per map; the axis map must express many-to-many.
- **`MEASUREMENTS.md`**: every numeric literal has a citation in the same
  sentence; every cited job id resolves somewhere. Mutation: the revision-1
  example (1.17e-2 cited to job 445657) must FAIL the guard.
- **derived columns**: a root where one field varies emits one column for it; a
  root where it is constant emits none; a list-valued field does not raise; the
  legacy and current microbatch names collapse to one column; an absent-in-old
  identity-transparent field produces no column; `--all-axes` emits both sets.
  Mutation: restore the hardcoded 11-key list and the varying-field test fails.
- No golden or `identity_baseline.json` may be touched by any of this. None of
  these changes goes near `_scientific_payload`.

## 6. Risks

**`MEASUREMENTS.md` grows into the prose doc this repo already distrusts.** The
number-bound citation guard is the structural defence; the discipline is that it
records interpretation and retraction only, never a number the cited artefact
does not contain.

**The index can cite runs that `/scratch` no longer holds.** Accepted -- the
index is the durable artefact and the run directory is perishable. The job-id
resolution check must tolerate a cleaned root rather than fail on it, or it
becomes a test of filesystem retention.

**A wide table on a mixed root.** Correct behaviour, not a defect; the docs
should say so rather than pretending per-study is the only query.
