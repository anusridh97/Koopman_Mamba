# Generated maps, and grouping runs by the key that already exists

**Status:** design. Small enough to hand to one implementer.

**Supersedes** `2026-08-24-measurement-provenance-design.md`, which was rejected
three times. Each rejection found the same thing: a proposal duplicating
machinery the repo already had. Four proposals became three became two. The two
below are what survived, and this document contains **no measured values** --
because in the rejected spec none of the measurements drove a design decision,
while the section added purely to demonstrate rigour produced two of the three
blocking defects.

Every claim here is structural and stated as a command you can run.

---

## Part 1 -- Three generated maps

### Why

Three tables were needed repeatedly this week and rebuilt by hand each time.
Written down they go stale; generated they cannot, because the test regenerates
and diffs.

The most consequential of them, config -> SKA route, would have surfaced as a
routine table that most registry configs run the *approximate* chunked SKA route
-- a fact that took a dedicated GPU investigation to establish and that had been
recorded in scattered prose since 2026-08-10.

### The pattern to follow, exactly

`scripts/resolve_anchor_design.py` + `docs/proxy-256x17-anchors-resolved.md` +
`code-tests/test_proxy_anchor_design.py:385-399`. That test asserts

```python
assert ARTIFACT.read_text() == expected, (
    "the committed artifact is stale. Regenerate with:\n  python scripts/... ")
```

Byte-exact, with the regenerate command in the failure message. Copy this shape:
a `render_markdown()` that returns a string, a `main()` that prints it, a
committed artefact, and a test that imports the renderer and compares.

### The three maps

| # | artefact | question |
|---|---|---|
| 1 | `docs/maps/ska-routes.generated.md` | which registry config runs which SKA route |
| 2 | `docs/maps/ska-backends.generated.md` | which route reaches which core, and whether it computes the spectral scale |
| 3 | `docs/maps/search-axes.generated.md` | which RunSpec field(s) each search axis writes |

**Map 1.** Iterate `koopman_lm.config.CONFIG_REGISTRY`. For each config emit
name, `ska_rank`, `ska_chunk_size`, and the resolved route. The route is decided
by the same precedence `SKAModule.forward` uses -- read it there rather than
reimplementing the order, and if the source order changes the map must follow.
Include `ska_prefix_scan` / `ska_inverse_cholesky` / `ska_exact_intrachunk` as
columns so the resolution is auditable, and mark the no-exact-route case with the
word the constructor's own warning uses.

**Map 2.** For each route: which function in `koopman_lm/kernels/` it calls, and
whether `spec_w` is reached. Derive the `spec_w` answer by inspecting the call
graph (e.g. `ast` over the kernel modules), **not** by hardcoding a table -- a
hardcoded one is the thing this map exists to replace. Two prior sessions
disagreed about which routes compute the spectral scale; the map settles it
mechanically and must keep settling it.

**Map 3.** `experimentation/sweep/search/space.py::search_space` declares the
axes; `params_to_overrides` writes the dotted keys. Emit axis -> the key(s) it
writes.

**Map 3 is many-to-many and the generator must show that, not flatten it.**
`norm_clip_multiplier` writes `model.ska_norm_clip_c` but its value depends on
`ska_rank`; `n_ska_layers` and `placement` *jointly* write
`model.ska_layer_indices` via `geometry.make_layer_indices`. A one-to-one table
would be a new false claim of exactly the kind this work exists to stop. Confirm
the relationships by reading `params_to_overrides` before writing the renderer.

### Constraints

- `space.py` and `geometry.py` must stay importable **without optuna** -- there
  are tests pinning this. A generator that imports them is fine; one that imports
  `study.py` or `driver.py` is not.
- Do not import `mamba_ssm`. Map 1 and 2 need config values and source
  structure, not model construction. `build_config` is safe; `KoopmanLM(cfg)` is
  not (it is unavailable on CPU).
- The artefacts go under `docs/maps/`; create it.

### Tests

For each map: a byte-diff test as above, plus a `test_the_generator_runs_as_a_command`
mirroring the precedent. Plus one guard per map that the *input* is non-empty --
a generator that iterates an empty registry produces a valid-looking empty table
and a passing diff, which is the vacuous-green failure this repo has hit before.

---

## Part 2 -- `results.py --group-by group_id`

### Why, and why it is this small

`experimentation/results.py` walks the run store and emits one row per
(run, checkpoint, task). It has no grouping, so any question of the form "what
did these N seeds measure, and how much do they disagree?" is answered by writing
a throwaway script. That is the re-derivation cost this work exists to remove,
and hand-derivation is where every error in the rejected spec came from.

`group_id` is already the right key and is already emitted. Verify before
building:

```bash
grep -n 'def group_id' -A3 experimentation/run/spec.py     # what it hashes
grep -n '"group_id"' experimentation/results.py            # already a column
```

`group_id = sha256(model + data + optim [+ schedules])` with the seed excluded,
so seed-replicates share one and runs differing in any scientific field --
including `optim.max_steps` -- do not. **The rejected spec proposed a whole
derived-columns feature to solve a pooling problem this key already solves.**
Check it yourself on two runs of the same study at different `max_steps`; their
`group_id`s differ.

### What to build

Add `--group-by group_id` to the existing CLI (`parse_args` at
`experimentation/results.py:84`). When given, emit one row per
`(group_id, checkpoint, task)` with:

- `n` -- number of contributing runs
- `mean`, `sd`, `SEM` of a named metric
- the contributing `run_id`s, so any row can be traced back
- the values of any spec field that is *constant within the group* and useful for
  reading the table -- at minimum `name` and `optim.max_steps`

`--metric` selects the metric path (default `full.loss`). A sigma over
`peak_memory_gib` is meaningless, so the metric must be explicit rather than
"every numeric column".

### Rules that are not negotiable

1. **`sd` and `SEM` are `None` at n = 1, never 0.** A single run silently
   reported with zero spread is how a measurement gets trusted that has no
   support. Emit the row, mark it.
2. **Deduplicate on `run_id`.** The same `group_id` legitimately appears in more
   than one run root -- a deliberate bit-for-bit reproduction is recorded in
   commit `348c8f3`. Two roots holding the same `run_id` are one datapoint, not
   two, and must not be refused as an error.
3. **No resolvable-effect verdict, and no threshold.** Two conventions already
   exist in the tree (`analysis.py`'s `_RESOLVE_SIGMAS`, and a family-wise
   corrected one in `scripts/analyze_beta_policy.py`). A third would manufacture
   a disagreement. This tool reports spread; interpretation stays where it is.
4. **Name the source in the output.** The walk reads JSON under each run's
   `eval/`, and other measurement artefacts exist outside it -- e.g.
   `route_ablation.json`, written by
   `scripts/measure_chunked_route_ablation.py` at a different eval-batch count.
   A number taken from one of those will legitimately differ from this table, and
   a reader must be able to see why rather than concluding one is wrong. Check
   what `experimentation/evaluation/result.py::iter_results` actually globs before
   describing it; it is broader than `quick_eval.json`.
5. **Do not add a state column for failed or pruned runs.** Rows exist only for
   evaluated runs, and the exit state is not in `attempts.jsonl` -- confirm with
   `grep -n 'def make_attempt_record' -A20 experimentation/run/write_policy.py`.
   The optuna journal has it, which is where `analysis.noise_floor` already reads
   it. Out of scope here.

### Tests

- a group of several seeds yields one row with `n` equal to the seed count and a
  finite `SEM`
- an n = 1 group yields `None` for `sd`/`SEM`, not `0`
- the same `run_id` present under two roots is counted once
- runs of the same study at different `max_steps` land in **different** groups
  (this is the property that makes the feature unnecessary elsewhere -- pin it)
- `--metric` selects; a bad path errors with the available paths listed
- the default (no `--group-by`) output is byte-identical to today's

Mutation-check at least the n=1 rule and the dedup rule: break each, watch the
named test fail, restore.

---

## Out of scope, deliberately

**Derived axis columns.** The rejected spec's largest unit. It solved a pooling
problem `group_id` already solves, and its own detail list had an unresolved
counting rule for absent-versus-default fields. If a real need appears later it
can be specified then.

**Any hand-written index of measurements.** Three attempts, three rejections,
every failure in the hand-written layer beside the numbers. Build the tool that
makes numbers cheap to recompute first; if a place to record "we stopped
believing X" is still wanted afterwards, that is a separate and much smaller
question.

**Escalating `SKAModule`'s chunked-route warning to an error.** It would make
several registry configs unconstructible. A project decision, not this work's.

---

## Verification before reporting

```bash
PYTHONPATH=$PWD:/users/jkli/.venvs/koopman-optuna/site \
  /users/jkli/.venvs/koopman-cpu/bin/python -m pytest code-tests -q -p no:randomly
PYTHONPATH=$PWD /users/jkli/.venvs/koopman-cpu/bin/python -m pytest code-tests -q -p no:randomly
git diff --check
```

Both suites must be green, and the second matters: it proves nothing new imports
optuna. Report both counts against the baseline you measured at the start.

`configs/runs/4m-golden.yaml` must still resolve to its pinned `run_id` /
`group_id`; `code-tests/identity_baseline.json` must not be touched. Nothing in
this work goes near `_scientific_payload`, so if identity moves, stop and report
rather than regenerating a baseline.
