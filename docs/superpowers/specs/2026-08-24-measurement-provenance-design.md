# Measurement provenance: a generated ledger, and maps that cannot go stale

**Status:** design, not yet implemented.
**Motivating failure:** measured facts in this repo are mostly untraceable, so
they get re-derived at GPU cost or, worse, believed without an artefact.

## 1. The failure, with numbers

Of 869 comment blocks in `koopman_lm/` + `experimentation/`, 41 are
measurement-shaped (they quote a percentage, a factor, a scientific-notation
figure, or the word "measured"). **11 cite a job id, a run_id, a commit or a PR.
30 cite nothing.**

The most consequential instance was found on 2026-08-24. `ska.py`'s chunked-route
warning claimed "~100% RELATIVE ERROR ... measured against a per-token-causal
reference", and that figure is why nine of eleven registry configs carry a
disclaimer on their results. Its provenance turned out to be commit `40f6653`,
"Add files via upload" (2026-05-21) -- a bulk import of an external tree, with no
harness, no geometry, and no job id. The number was *asserted* upstream and
*cited as measured* here.

The same session re-derived, at GPU cost, facts that had been measured hours
earlier in the same session:

| fact | job | what it cost to learn |
|---|---|---|
| sigma = 9.50e-3 at 600 steps, 5 seeds | 445832 | 41 min |
| sigma = 7.54e-3 at 1500 steps, 5 seeds | 445994 | 1 h 51 min |
| SKA ablation delta 0.0251 +- 0.0030 SEM | 445994 | (same run) |
| layerscale 0.01/0.1/0.5/1.0 -> loss | 445689 | 33 min |

None of those numbers lives anywhere a later reader would find it. They exist in
this conversation and in `/scratch`, which is cleanable.

## 2. What already exists -- most of it

This is not a new system. Four facts, verified rather than assumed:

**Every run already records its own conditions.** A materialized `spec.yaml`
carries `run_id`, `group_id`, `code_id` (a full commit sha), `study_name`,
`trial_number`, `anchor_name`, a `provenance` block (git commit, torch, cuda,
python, materialized_at), and the complete resolved `model` / `data` / `optim` /
`runtime`. Alongside it: `attempts.jsonl` (failure history), `train.log`, and
`eval/**/quick_eval.json` (loss, ppl, tokens_per_sec, peak_memory_gib,
`ska_ablation.loss_delta`).

**`experimentation/results.py` already walks that store.** Its docstring states
the model outright: "No database, no service: the filesystem is the store, and
this is a walk over it." It emits one row per (run, checkpoint, task).

**`group_id` already groups seeds.** `group_id = sha256(model + data + optim)`
excludes the seed, so runs differing only in `runtime.seed` share one. Verified
on job 445994: `group_id c7b818ff` -> 5 runs, seeds [42, 43, 44, 45, 46].

That last point is the design's hinge. **A noise floor is a derived property of
the run store, not a special experiment.** Any group with n >= 2 yields one for
free. The two dedicated noise-floor studies were necessary to *create* replicate
groups, not to compute anything the store could not.

**`test_docs_are_not_stale.py` already encodes the philosophy.** Its docstring
records a prior finding -- "docstrings are accurate, prose docs have drifted" --
and its rule: pin only mechanically checkable claims, never wording, because "a
test that pinned wording would fail on every edit and get deleted, which is worse
than no test."

## 3. The four changes

### 3.1 Widen `results.py`'s axis columns -- derived, not hardcoded

`aggregate()` currently hardcodes 11 axis keys: run, name, run_id, group_id,
d_model, n_layers, data_kind, lr, seed, sweep_id, sweep_name.

That set is missing every condition that varied in this session's work:
`max_steps` (so a 600-step and a 1500-step run are **indistinguishable in the
table today**), the SKA route, `ska_rank`, `ska_layerscale_init`,
`ska_chunk_size`, `ska_beta_policy`.

The naive fix -- add a column per field anyone might vary -- does not scale:
`KoopmanLMConfig` has 60 fields, `OptimSpec` 8, `RuntimeSpec` 13, plus data.
**Roughly 85 columns, most of them constant in any given query.**

So: **emit a column for a field only when it VARIES within the queried root.**
This is self-limiting by construction rather than by taste.

- a 5-run replicate group -> only `seed` differs -> 1 axis column
- a 256-trial interaction study -> 9 sampled axes differ -> 9 axis columns
- a whole-scratch walk -> wide, and correctly so: those runs really do differ in
  many ways, and a narrow table would be hiding it

Width therefore tracks the actual dimensionality of the question asked, and can
never exceed it. Identity columns (`run`, `run_id`, `group_id`, `code_id`) are
always emitted; the full conditions always remain in each run's `spec.yaml`, so
nothing is lost by omitting a constant column -- it is one file read away.

A `--all-axes` escape hatch prints the constant columns too, for the case where
someone needs to confirm *what* the constant was.

### 3.2 Record the Slurm job id at materialization

The `provenance` block carries git commit, torch/cuda/python versions and
`materialized_at`, but **no job id**. Without it a ledger row cannot reach the
log, the sbatch script, or `sacct`'s record of what was allocated and billed.

Add `slurm_job_id` (from `SLURM_JOB_ID`, null when run interactively) to the
provenance block.

**Identity-safe by construction, and this must be verified rather than assumed:**
`provenance` is not part of `_scientific_payload`, so `run_id` and `group_id`
cannot move. The implementation must assert `4m-golden` stays
`2e63f16e / d812e412` and that `identity_baseline.json`'s three pinned specs are
unmoved.

### 3.3 A group aggregator: `python -m experimentation.ledger <root>`

Groups rows by `group_id` and emits, per group: n, mean, sigma, SEM, and the
derived resolvable effects `2*sigma*sqrt(2)` (trial vs trial) and
`2*sigma*sqrt(1 + 1/n)` (a single trial against the group mean).

Two behaviours are load-bearing:

**n = 1 groups must be marked "no sigma available", not omitted and not given
sigma = 0.** A single run silently trusted is the exact error made in this
session: the SKA ablation delta was judged against a between-trial floor when it
was in fact a five-run mean whose SEM was the right statistic (t = 8.4, not
"below the floor"). The aggregator's job is to make the distinction
unmissable.

**A group with two members at the same seed must raise.** Under
`runtime.deterministic: true` a same-seed repeat is bit-identical, so it
contributes a spurious zero and drives sigma toward zero -- a fabricated floor,
which is the most dangerous wrong answer available. (`analysis.noise_floor`
already refuses this; the aggregator must not be a second, laxer path to the same
number.)

Output is a committed snapshot, `docs/measurements/index.generated.md`, because
`/scratch` is cleanable and the index must outlive it. A test regenerates and
diffs against whatever run roots still exist, tolerating absent ones.

### 3.4 `docs/MEASUREMENTS.md` -- interpretation only, with a guard

Generation cannot produce the part that matters most: what a measurement *means*,
and what it *retracts*. That layer is hand-written and deliberately small.

Each entry is a claim, its citation, and -- where applicable -- what it overturned:

```
sigma does not shrink materially with horizon.
  9.50e-3 @ 600 steps (job 445832) -> 7.54e-3 @ 1500 (job 445994): 21%, not the
  order of magnitude predicted. sigma is a near-constant 0.17-0.18% of the loss
  at both horizons, so more steps buys resolution only as fast as it buys loss.
  RETRACTS: "longer trials will fix the noise floor" (proposed 2026-08-24).

SKA is load-bearing on the 256x17 proxy.
  ablation delta 0.025139 +- 0.002976 SEM over 5 seeds, t = 8.4 (group c7b818ff,
  job 445994).
  RETRACTS: "not established, the delta is below the floor" -- that judged a
  five-run mean against a between-trial floor, the wrong statistic.

The 4m ablation delta (1.17e-4, job 445657) is a SCALE artefact, not evidence
that SKA is inert. The proxy gives 1.17e-2 at the same layerscale -- 100x.
  RETRACTS: "SKA is switched off by LayerScale and never switches on".
```

The retraction lines are the highest-value content, because a reader who sees
only the surviving prose will re-derive the dead claim. Four claims died in one
session.

**The guard:** a test asserting every number in `MEASUREMENTS.md` appears in an
entry that cites a `job \d{6}`, a `run_id`, or a `group_id` -- scoped to the
entry (a block separated by a blank line), not to a line window, so reformatting
an entry cannot break it. Section headings, dates and the file's own prose are
exempt by living outside entries. This keeps the file
interpretation-only -- a paragraph with no citation fails -- and stops it growing
into the prose doc that `test_docs_are_not_stale.py` was written because of.

## 4. Maps

Three tables were built by hand during this session and would otherwise be
rebuilt. Each becomes a generator plus a regenerate-and-diff test, following
`docs/proxy-256x17-anchors-resolved.md`, which already works this way:

| map | answers |
|---|---|
| config -> SKA route | which of 11 registry configs run chunked vs an exact route |
| route -> core -> computes alpha | which backend calls `spec_w`, and via which core |
| study axis -> RunSpec field | which dotted key each sampled axis writes |

A written map goes stale the first time a config flips; a generated one cannot,
because the test fails and names the file to regenerate. The route map in
particular would have surfaced the nine-config chunked finding months earlier, as
a table nobody had to think to look for.

## 5. What this deliberately does NOT do

A general stale-comment / clutter sweep was considered and **dropped on
evidence**. A hand-classified sample of 45 randomly drawn short comment blocks
(seed 20260824, reproducible) found:

| what it actually is | share |
|---|---|
| tensor-shape annotations (`# (B,T,H,r,r)`) | ~20% |
| informative-terse (`# left-pad => causal`) | ~36% |
| navigational (`# ---- Batched Cholesky ----`) | ~16% |
| math annotations (`# U_0 = L^{-1} q`) | ~11% |
| genuinely low-value | **~18%** |

The largest restatement-*looking* category is tensor shapes, which are the type
system Python does not provide here; the math annotations tie code lines to the
whitening derivation. The deletable prize is roughly 6% of blocks, against a
large diff across the most load-bearing comments in the repo -- many of which
record a bug that recurred, where deletion loses the only record that it can.
(n = 45, so call the 18% figure 10-30%; the conclusion is unchanged at either
end.)

Two mechanical slices are kept, needing no judgement: the 18 pure banner blocks
and the four duplicated `# frozen: use replace` comments.

Renaming was also considered and dropped: the terse kernel names (`Gf`, `Mf`,
`Lf`, `qf`) match the paper's notation, which is the right call in code whose
correctness argument is a derivation.

## 6. Testing

- widened `results.py`: a root where one field varies emits exactly one axis
  column for it; a root where it is constant emits none; `--all-axes` emits both.
  Mutation: hardcode the old 11-key list and the varying-field test must fail.
- `slurm_job_id`: present when `SLURM_JOB_ID` is set, null when absent, and
  `4m-golden` / the three `identity_baseline.json` specs unmoved either way.
- aggregator: a 5-seed group yields the sigma this session measured; an n=1 group
  is marked, not silently zeroed; a duplicated seed raises.
- `MEASUREMENTS.md`: every number cites a job/run_id/group_id.
- each generated map: regenerate and diff.

## 7. Risks

**The index can outlive its data.** `/scratch` is cleanable, so a committed index
may cite run roots that no longer exist. Accepted: the index is the durable
artefact and the run dir is the perishable one. The regenerate test must tolerate
absent roots rather than fail on them, or it becomes a test of filesystem
retention.

**`MEASUREMENTS.md` could grow into the prose doc this repo already learned to
distrust.** The citation guard is the structural defence; the discipline is that
it records interpretation and retraction only, never a number the index already
carries.

**Sparse axis columns across a mixed root.** Walking all of `/scratch` at once
produces a wide table by construction. That is correct behaviour -- the runs
really do differ -- but the useful query is per-study, and the docs should say so.
