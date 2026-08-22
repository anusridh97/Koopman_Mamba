# Reviewing `jack/search-and-provenance`

72 commits, 125 files, +15,349 / −387. Written to be reviewed commit-by-commit
rather than as one diff, because most of the insertions are new files where a diff
tells you nothing you wouldn't get from reading the file.

**Test counts.** 1142 passed / 45 skipped standard; ~1160 with optuna on the path.
`main` was 499 / 28.

This file is the entry point. It says what changed, in what order to look, how to
run each new thing, and — most importantly — **what is not verified**.

---

## 0. If you only have twenty minutes

```bash
cd /users/jkli/Koopman_Mamba/.claude/worktrees/jack+search-and-provenance
git log --oneline --reverse main..HEAD          # the shape of the work
git log main..HEAD                              # the reasoning; messages are long on purpose
/users/jkli/.venvs/koopman-cpu/bin/python -m pytest code-tests/ -q
```

Then read §3's table and §6. Everything else is depth.

## 1. What this branch does, by workstream

| | what | verified how |
|---|---|---|
| **A** | Precision policy: `compute_precision` / `ska_precision` / `mlp_precision`, threaded into the SKA core, the Koopman rotation, and all five trainers via one `amp_for` helper | GPU job 436063: real 30-step H100 training, loss 10.08 → 8.26 |
| **B** | Provenance: `code_id` + `dirty` in all four checkpoint writers; `harness.py` stops overwriting a recorded `cfg_hash` | GPU 436063 step 3b: `meta.pt` carries `code_id=3be37b7` |
| **C** | Adaptive search (`experimentation/sweep/search/`, 9 modules + a CLI): the space declared once, TPE + median pruning, ask/tell over the **unmodified** run system | **works end to end.** GPU job 439883: 4 trials COMPLETE with real objectives (7.18–7.64), 4 `quick_eval.json`, 20 reported steps each, all reports written. Caveat in §6 |
| **D** | `per_device_batch_size` moved `OptimSpec` → `RuntimeSpec`, so an OOM-ladder rung no longer renames the experiment | 0 of 11 config hashes moved |
| **E** | Test infrastructure: import gate, static undefined-name check, `param_groups` characterization, loss-alignment conformance, two golden training curves | see §4 |
| **G** | `TrainTask` seam: both trainers now get their loss from a task instead of inlining it, so the three-loop unification lands against loops that already delegate | **both gates passed on GPU** — 4m golden MATCH at 0.0001, MQAR golden MATCH at **0.000000**. §6 says what is left |
| **F** | Cleanup: both documented training commands were broken; `load_model` deduplicated; dangling in-tree pointers | §4 |

## 2. Read in this order

The commit stack is bottom-up, so commit order *is* dependency order.

**The 26 modified non-test files are the whole "did this break something" surface**
— 601 insertions, 73 deletions. That is the part worth diffing:

```bash
git --no-pager diff main..HEAD -- koopman_lm/modules/seq/ska.py        #  33  live forward path
git --no-pager diff main..HEAD -- koopman_lm/modules/mlp/koopman.py    #  50  the other forward path
git --no-pager diff main..HEAD -- experimentation/training/train.py     #  22  where the autocast bug bit
git --no-pager diff main..HEAD -- koopman_lm/config.py                 #  50  the three new fields
git --no-pager diff main..HEAD -- experimentation/run/spec.py           #  62  microbatch moved
git --no-pager diff main..HEAD -- experimentation/run/train_argv.py     #  69  batch_plans -- check this one
git --no-pager diff main..HEAD -- experimentation/run/resolve.py        #  58  migration shim, 3 load paths
```

The new **search** package reads bottom-up and only calls downward:

```
geometry.py  129   pure arithmetic over layer indices, no optuna    <- start here
space.py     218   the space, and params -> RunSpec overrides
anchors.py   186   curated designs -> concrete params
------------------------------------------------- the optuna line
study.py     222   sampler / pruner / storage / enqueue
driver.py    223   ask -> materialize -> submit -> tell
metrics.py   328   log tailing, OOM detection, the objective
report.py    214   trials.csv, top_trials.md, promotions
```

Everything above the line is importable and useful with optuna absent. That split
is what lets a curated anchor set ship as an ordinary `cells:` sweep before the
adaptive machinery is usable.

## 3. How to run each new thing

```bash
export CPU=/users/jkli/.venvs/koopman-cpu/bin/python
export OPT=/users/jkli/.venvs/koopman-optuna/site      # optuna 4.9.0
export TOOLS=/users/jkli/.venvs/koopman-tools/site     # coverage, lm_eval
```

**Tests.**
```bash
$CPU -m pytest code-tests/ -q                                    # 1136 / 45 skipped
PYTHONPATH=.:$OPT $CPU -m pytest code-tests/ -q                   # ~1150 / 31
```

**Coverage** — 58% overall; this branch's own new code is 95%.
```bash
PYTHONPATH=.:$OPT:$TOOLS $CPU -m coverage run --source=experimentation,koopman_lm \
  -m pytest code-tests/ -q
PYTHONPATH=$TOOLS $CPU -m coverage report --sort=cover | head -20
```
Read the *low* numbers only. See §5 on why the high ones lie.

**A real training run** (this is what `main` could not do reproducibly):
```bash
$CPU -m experimentation.run configs/runs/4m-golden.yaml --launcher local \
     --run_root /tmp/try --dry_run          # print the plan, touch nothing
sbatch scripts/capture_golden_curve.sbatch  # 5M params, 400 steps, ~2 min/replicate
```

**Did a change alter training?**
```bash
$CPU scripts/compare_golden_curve.py <a-train.log>
# tolerance 0.001, read from the artifact; exit 1 on regression
```

**GPU verification of the whole branch:**
```bash
sbatch scripts/verify_search_and_provenance.sbatch
```

**An adaptive search:**
```bash
$CPU -m experimentation.sweep.search configs/search/smoke-4m.yaml --dry_run
sbatch scripts/verify_study_e2e.sbatch      # 4 trials x 200 steps, ~minutes
```

**The static anchor sweep** (works today, needs no optuna):
```bash
$CPU scripts/gen_anchor_sweep.py --design-file configs/search/curated_15.yaml \
     --out configs/sweeps/anchors.yaml        # <- design file does not exist yet, §6
```

## 4. What the tests actually pin

Not "there are 1051 of them" — that number is nearly meaningless on its own. What
matters is which of them would catch a real mistake:

| test file | catches |
|---|---|
| `test_loss_alignment.py` | the off-by-one a loss-dedup refactor introduces. Mutation-checked: swapping the two conventions fails 4 of 9 |
| `test_param_groups.py` | a flat `AdamW` decaying norms/biases/embeddings/Mamba state — the bug that made published Table 2 numbers a different optimizer regime. Mutation-checked, both mutants caught |
| `test_module_import_health.py` | a `NameError` in a module no test imports. Found two real ones on first run |
| `test_ska_precision_wiring.py` | the precision refactor is bit-identical at its defaults, against a golden captured **before** the change |
| `test_identity_baseline.py` | no config hash moved that shouldn't have |
| `golden_4m_curve.json` + comparator | the shard loop still trains identically. Noise floor 0.0002 -- log-format precision, not numerical. **Used as a gate:** job 439919 re-ran it after `train.py`'s loss moved into `ShardTask` and MATCHED at worst delta 0.0001 |
| `golden_mqar_curve.json` | the synthetic loop, the path whose shift convention differs. Noise floor **0.0** with `--deterministic`; **0.53 without it**, which is how that missing flag was found. **Used as a gate:** job 439941 re-ran it after `mqar_finetune.py`'s loss moved into `SyntheticTask` and MATCHED at **0.000000** |
| `test_train_task.py` | each task's `step_loss` is bit-identical to the inline code it replaces. Mutation-checked against all three mistakes the refactor can make |
| `test_progress_log_format.py` | a trainer whose log the pruner cannot parse, which made synthetic trials silently unprunable |
| `test_search_trial_budget.py` | a trial training for the base spec's budget instead of the study's -- a 25x overspend against a 15000-step base |
| `test_local_launcher_nonblocking.py` | a locally-launched trial that cannot be cancelled, i.e. pruning that frees no GPU |

**On mutation checks.** Several of these were written *after* the code, so each
was verified by breaking the code and confirming the test fails. A test that has
never failed is not evidence. Where I did this, the commit message says so.

## 5. Coverage is 58% and that number is Goodhart-able

Line coverage means one thing: this line executed. Not that anything checked the
result. Demonstrated on my own work — the import gate added in this branch raises
coverage from 56% to 58% (152 statements) while verifying **nothing** about
behaviour, because importing a module executes its module-level statements.

So: use it to find holes, never as a target, and never report a coverage delta as
a quality delta. The real measure is mutation — break the code, see if the tests
notice.

## 6. NOT VERIFIED, and what is still broken

Read this section before trusting anything above it.

**A study closes its loop; a pruning DECISION has never been observed.** GPU job
439883 passed all five checks: 4 trials COMPLETE with real objectives (7.1819,
7.1934, 7.2203, 7.6398), 4 `quick_eval.json` files, **20 reported intermediate
steps per trial**, `trials.csv` / `top_trials.md` / `best_trial.json` written, and
each trial's `spec.yaml` stamped with study/trial/anchor.

What that proves is the *integration*: progress is parsed from a real log and
reported to a real journal. Job 439891 then ran 6 **non-anchor** trials to ask
whether a prune fires, and **none did** — legitimately. The objectives went 7.337,
7.461, 7.388 then 6.954, 6.965, 6.942, so every trial after the first two beat the
running median. Nothing was ever hopeless enough to cut. That is a fact about the
space, not a gap in the machinery.

The pruning *decision* is pinned at unit level instead, which is where it belongs:
`test_search_pruning.py` covers a hopeless trial being cancelled and marked PRUNED,
the anchor exemption in both directions, the timeout, and never reporting a step
twice. So: decision logic unit-verified, integration GPU-verified, and a prune
firing during a real study still unobserved — because the sampler kept finding
better configs, which is the outcome you want.

It took four attempts, and each failure was different and real: a trial-budget bug
(25x overspend against a 15000-step base), an `exact_auto` stall, two bugs in my own
verification harness, and then two independent reasons pruning could not fire. None
were findable by reading.

**Also missing for a *good* study:** `configs/search/curated_15.yaml` — 15
hand-chosen anchor designs. A study runs without it, but the first four trials are
then random *and* unprunable, so TPE models from noise. This one needs a human;
inventing 15 designs would imply a research judgment nobody made.

**Three pre-existing GPU test failures**, none from this branch:
`test_full_model_decode_prefill_parity` and
`test_e2e_train_checkpoint_reload_decode` are named in
`scripts/gpu_triage_verify.sbatch`'s own header as nvcc-build collateral. Which
also means **`--resume` and decode/prefill parity are unverified.**
`test_importing_koopman_lm_does_not_reach_the_cuda_extras` fails on GPU and passes
*vacuously* on CPU, because `triton` is absent there — checked by running its
probe against untouched `main` in the same venv, where it fails identically.

**`backend_policy: exact_auto` can look like a hang.** Measured (job 439754): a
study whose base spec pins `ska_backend: pytorch` / `ska_prefix_scan: false` has
those overridden to `auto` / `true`, and the resulting path sat at **99% CPU for 21
minutes without logging a single step** on a 5.2M model that trains in ~2 minutes
standalone. One trainer, one attempt, `--no_compile` -- so neither a collision nor
compilation. Silent: no error, no warning, no slow-path notice.

**RESOLVED, and the slowness is now explained.** Jobs 440122 / 440135 measured
all four routes on an H100 against `prefix_scan.dense_exact_oracle` in fp64. The
21 minutes was the Python reference scan: at the 4m geometry (value width 32) the
fused kernel does not apply, and the reference fallback measured **160x** the
chunked cost per full-model micro-step. Worse than a constant tax, `exact_auto` is
a **cliff** -- the fused kernel needs rank *exactly* 24 while the space samples
{8,16,24,32}, so within one study it measured 0.0043 s at rank 24 and 0.72-3.19 s
at the other three, a 167x-738x discontinuity correlated with a searched variable.

The default is now `exact_invchol` (`ska_inverse_cholesky`), which agrees with the
reference scan to 5.1e-13 in fp64 and measured **0.92x** chunked at 4m / 1.41x at
50m. Exactness turned out to be free, so nothing is traded.

Overriding the backend at all remains *correct* (trials must be comparable), and
there is now a second reason it must resolve to ONE route per study: `chunk_stats`
and `exact_stats` add a `1e-4*I` jitter on top of `ska_ridge` and
`ska_prefix_scan` does not, so switching route between trials perturbs a *sampled*
parameter by 0.3%-10%. See `space.py`'s docstring.

**The three-loop unification is NOT done.** What is done is the seam: all three
trainers now get their loss from `TrainTask` instead of inlining it, each verified
bit-identical on GPU. `table2` and `mqar` were re-run at **worst |delta| =
0.000000 over 40 steps** in job **440203** against their committed goldens (both
have a 0.000000 noise floor, so that is exact and not a tolerance). What remains,
from `docs/superpowers/specs/2026-08-21-traintask-design.md` §10:

- **The loop body is now `training/loop.py`**, extracted from `train.py`
  unchanged and verified **bit-identical to the committed golden** (job 440500,
  max |delta| 0.000000 -- identity, not tolerance). It has a `GradScaler` path,
  which table2 needs and which nothing else exercises.
- **DONE: all three trainers now run it.** mqar lost 76 lines of loop, table2 82,
  each verified bit-identical against its own golden (jobs 440574 / 440608, both
  0.000000). The duplication the design is about is gone.
- **The table2 migration failed its golden once, and the failure was real.** Worst
  |delta| 8e-4 over 26 of 40 steps -- small, bidirectional, non-compounding, which
  is a REPORTING signature rather than a training one. Cause: the original
  computes `loss` inside `with autocast:` but the per-task split outside it, and
  autocast promotes `cross_entropy` to fp32, so moving the split into `step_loss`
  reported at a precision that trainer never used. Training was never affected.
  Nothing in the 1379-test CPU suite could see it.
- **What remains is the capability, not the loop:** only `train.py` writes
  `quick_eval.json`, which `sweep/search/metrics.py::read_quick_eval_objective`
  reads, so neither synthetic trainer produces a search objective yet and
  `run/train_argv.py` still refuses `data.kind='synthetic'`. Wiring `on_final` for
  them is now a small change, and it is what would let a study rank on **MQAR
  recall** instead of a short-horizon LM loss that may not distinguish an exact
  SKA from a broken one.
- **The loop is now executed by the CPU suite** (`test_training_loop.py`), for
  the first time. It was at 29% line coverage, which is why this work needed four
  GPU goldens to say anything at all.
- **A decision the doc did not make, now made.** Its ownership table puts
  `dataset` on the task and `resume` on the loop; §3 says the two are entangled,
  and they are. train.py and mqar iterate an epoch permutation and resume by
  skipping a consumed prefix; table2 has no epochs at all, its batches being a
  pure function of the step counter, so a permutation would hand it batches it
  has never trained on. Resolved with a DEFAULTED `iter_batches` (epoch path by
  default, overridden by `Table2Task`) so there is one loop rather than an `if`
  inside it. Verified bit-identical to the committed golden (job 440235,
  max |delta| 0.000000).
- **`table2`'s numbers would need regenerating** if its architecture changed --
  it did not, but that call is a human's either way.
- **THE CENSUS WAS WRONG: there are FIVE training loops, not three.**
  `experimentation/retrieval/adapt.py` and
  `experimentation/evaluation/evaluate_retrieval.py` each build an AdamW and call
  `loss.backward()` / `opt.step()`. The design doc is written about three loops
  and its §9 out-of-scope list never mentions retrieval. Neither has a golden, so
  neither can be refactored safely yet; both are recorded in
  `code-tests/test_golden_coverage.py::NO_CURVE_EXPECTED` with the reason written
  next to them. **Whether they belong in the unification is an open decision.**

**FIXED: a spec-driven run can now be reproducible.** `RuntimeSpec.deterministic`
exists and `train_argv` passes `--deterministic`. It lives on `runtime` because
`run_id` is sha256(model + data + optim + seed), so it renumbers nothing --
`4m-golden.yaml` keeps `run_id 2e63f16e` / `group_id d812e412`, verified against
HEAD and pinned by test. Default False; `4m-golden.yaml` sets it, because that
spec's purpose is to be re-run and compared.

Consequences, all now measured (jobs 440211 / 440216):

- `golden_4m_curve.json`'s old "noise floor" of 2.0e-4 **was** nondeterminism, not
  precision. Re-captured deterministically it is **0.000000**. All four
  instruments are now equally sharp.
- **A now-possible measurement that has NOT been run:** the chunked-vs-exact SKA
  route comparison. It previously produced 2.3e-4 against a 2.0e-4 floor, i.e.
  unresolvable. Deterministic runs resolve to 1e-4 (the log prints `loss %.4f`),
  and 2.3e-4 is above that, so the question "how much training loss does the
  chunked approximation actually cost" is now answerable and worth answering
  before trusting a short-horizon study.
- Search trials are still not reproducible: `StudySpec` has no way to request
  determinism for its trials. Deliberate for now (it would slow every trial), but
  it means trial-to-trial differences below ~2e-4 are noise.

**What the instruments can resolve.** The progress line prints `loss %.4f`, so
1e-4 is the smallest representable difference and `compare_golden_curve.py`'s
`MIN_TOLERANCE` is exactly that. A "0.000000 floor" means *identical at the
printed resolution*, not to arbitrary precision.

**Resume through `train.py`'s own loop is verified exact** (`golden_resume_curve.json`,
job 440216): interrupted after the step-266 checkpoint, resumed to 400,
0.000000 at every shared step. Its first run found a real defect that no existing
test could see -- `test_resume_equivalence.py` exercises its own loop body, not
`train.py`'s -- namely that the running-loss window did not cross the checkpoint,
showing up as 0.0011 at the first post-resume log. Training was already bit-exact;
the reporting was not. Fixed in `train.py::_save_all`.
- **`SyntheticDataSpec` still cannot launch.** `run/train_argv.py:101` and
  `run/data_verify.py:53` still refuse `kind: synthetic` by name, so MQAR is not
  yet an ordinary run with a run directory and a result envelope.

**§6 schedules and §7 per-group optimizer settings are back and WIRED.** The
stranded PR B work (`7bec8ad`, 1997 lines) reached this tree as a `git stash pop`
of a half-resolved cherry-pick and had to be re-applied from the commit itself at
post-split paths. Verified: `4m-golden.yaml` still hashes to `run_id 2e63f16e` /
`group_id d812e412`, because an absent `schedules`/`optim.groups` is omitted from
the hashed payload -- so no existing run is renumbered.

It arrived INERT (read and validated, never applied) with 93 tests passing, the
third instance of that shape on this branch after `ska_backend` and the dead
`chunk_strategy` fields. `code-tests/test_extensions_are_wired.py` now makes
inertness a test failure.

**Never exercised:** multi-GPU / DDP. Which SKA backend a run actually selected
(nothing asserts it). `ska_precision='fp64'` on GPU. A schedule actually
annealing anything end-to-end on GPU -- the mechanism is wired and unit-tested,
but no golden covers a run WITH a schedule active.

**An optuna study CAN now drive real training** -- jobs 440183 (wiring) and
440184 (pruning) both pass on the `exact_invchol` default: 4 trials COMPLETE with
real objectives, `quick_eval.json` written and read back, 20 reported intermediate
steps, and 440184 additionally shows **2 PRUNED**. What has *not* run is a study
large enough to be a search rather than a wiring test; `configs/search/4m-adaptive.yaml`
is the first one and is unrun.

**The objective may not be sensitive to SKA at short horizons.** Comparing job
439883 (chunked, ~100% wrong operator) against 440183 (exact) on the same four
anchors, every objective moved by at most **2.3e-4** while the spread ACROSS
anchors was **0.458**. Both runs were nondeterministic and the 4m instrument's
noise floor is 2.0e-4, so the honest reading is that the route effect is *not
resolvable here* -- not that it equals 2.3e-4. Either way it means a screen scored
on 200-step loss cannot distinguish an exact SKA from a broken one, so it is
unclear how much of that 0.458 is SKA at all. SKA's output is gated by
`ska_layerscale_init` (0.01) into a Mamba residual, which is the mechanism.
**Unresolved, and it bears directly on whether a short-horizon study ranks what
you want ranked.**

**Eval precision.** `quick_eval` and `evaluate.py` measure in fp32 while trials
train in bf16. Measured spread (job 438232) is ~0.005 loss and **inconclusive** —
the lower-precision regimes scored *better*, so it is perturbation, not
degradation, on a 30-step model. That job's throughput column is invalid: no
warmup pass, so regime A paid all the CUDA startup.

**A question, not a defect.** `koopman_lm/config.py:43` defaults
`ska_power_K = 2`, but `configs/50m.yaml:19` and `configs/180m.yaml:18` both pin
`1`. So `370m`, `880m`, `1p5b`, `3b`, `180m_gated` silently use a different K than
both production configs. Behaviour is unchanged by this branch; whether it is
intended is for you.

**One unknown surfaced, not fixed.** `kernels/chunk_stats_exact.py`'s `__main__`
loaded a deleted module by file path, so it could never run. Now runnable, and it
prints `rel=2.9e-3` against a comment saying "expect ~0". No baseline exists, so
the numerics were left alone.

## 7. Where the reasoning lives

Long commit messages are deliberate — they carry the *why*, and the deviations.
`git log main..HEAD` reads as a design review.

| doc | what |
|---|---|
| `docs/superpowers/DECISIONS-2026-08-20.md` | precision policy and the trainer refactor: decisions, and where I was wrong |
| `docs/superpowers/GPU-VERIFICATION-2026-08-19.md` | job 436063, including the autocast bug 843 CPU tests missed |
| `docs/superpowers/BACKLOG-2026-08-18.md` | the original plan, with a status banner |
| `specs/2026-08-20-eval-at-training-precision.md` | three eval sites, three different right answers |
| `specs/2026-08-20-checkpoint-meta-resolved-config.md` | why `meta.pt` should hold `asdict(cfg)` |
| `specs/2026-08-19-*-identity-mapping.md` | the two deliberate identity breaks, with old → new hashes |

## 8. Corrections I made to my own earlier claims

**"The loop unification fixes a correctness bug in two trainers."** I predicted
this from reading the code: `mqar_finetune.py` and `table2.py` restore no RNG on
resume (`grep -c rng` is 0 in both) while `train.py` does, so a resumed run's
stochastic stream should desync. Job 440248 measured it and **table2's resume is
exact, 0.000000 across the interruption**. The prediction was wrong.

Why it holds: `KoopmanLMConfig` declares no dropout field, so the forward has no
stochastic op, and the curriculum generators use a LOCAL
`torch.Generator().manual_seed(args.seed + step)` rather than the global stream.
A table2 step consumes no global RNG, so there is nothing to restore.

The real residue is smaller but genuine: that exactness is a property of the
current architecture and data generator, **not** of the resume mechanism. Add a
dropout, or swap one local Generator for a global call, and table2's resume
silently stops being reproducible with nothing to catch it. Now pinned by
`test_table2_batches_consume_no_global_rng` for the half that runs on CPU.

So the loop unification is **deduplication, not a bug fix** -- which is worth
knowing before spending the riskiest remaining refactor on it.


**"The route effect is 2.3e-4."** I wrote that as though it were a measurement of
how much the SKA route matters. It is an upper bound at the instrument's own noise
floor: both runs compared were nondeterministic and `golden_4m_curve.json`'s floor
is 2.0e-4, so the correct statement is that the effect is *not resolvable* by that
comparison. The conclusion that matters -- a 200-step screen cannot tell an exact
SKA from a ~100%-wrong one -- survives either reading, but the number does not
support the precision I gave it.

**Two golden `args` records could not reproduce their own curves,** and I wrote
both. The step grid is set by `--log_every`, whose defaults are 100
(`mqar_finetune`) and 200 (`table2`), while both goldens log every 10 -- and
neither recorded it. Driving a re-capture from the recorded args would have logged
4 points, compared them against 40, and reported 36 missing steps as a run
failure. Fixed and guarded; replicate values verified byte-equal.

**I swept another agent's unstaged work into a commit of mine.** `git add -A` in a
worktree shared with a concurrent agent put its `token_trace.py` /
`inspect_html.py` fixes into `709c474`, whose subject is about the search space.
Documented in `9f99315` rather than rebased, because the branch was already pushed.

**"`proxy_chunked` is the cheap approximate screen."** I wrote that, switched
`configs/search/smoke-4m.yaml` to it, and quoted `space.py`'s own docstring as the
authority -- without checking what *approximate* meant. Job 440122 measured it:
92%-152% wrong in the forward, 93%-101% wrong in the gradients, at both
geometries, on random inputs and on a lag-3 recall task, with no dependence on
sequence length, ridge or power_K. It is not an approximation of the exact
operator; it is a different operator. And `ska_rank`, `ska_ridge` and
`ska_norm_clip_c` reach the model *only* through it, so it could not have screened
the parameters the search samples. My "fix" made the smoke study fast and its
numbers meaningless. The policy is retired (`90d17cb`); the chunked forward path
stays in `ska.py` only because `collect_diagnostics` deliberately reports it as a
bounded-cost proxy and archived specs must stay loadable.

**"~137x slower" understated it and mischaracterised its shape.** The figure is
real (reproduced at 160x with a slightly faster chunked baseline) but it reads as
a fixed cost. It is a discontinuity at rank 24, not a slope -- which matters far
more for a sampler than the magnitude does.


Recorded because a branch that hides its retractions is harder to trust:

- "One entry point short of usable" (the search) — **wrong**, nothing writes the
  objective either.
- "Extension mechanisms are already written, just port them" — **wrong**, ~4,425
  lines and it needs a core→experimentation import edge the boundary test forbids.
- "The prefill outside `no_grad` caused different arithmetic" — **wrong**,
  `no_grad` does not affect arithmetic. It was wasted gradient tracking.
- "`evaluate.py` in fp32 keeps trials comparable across swept precisions" —
  **wrong**, `compute_precision` is not in the search space. The real reason is
  that a benchmark should measure the function, not the arithmetic.
- "Added-defaulted-field schema drift is undetectable in principle" — **wrong**,
  and `resolve.py`'s `_check_model_key_set` was already the counterexample on
  `main`.
- Reported 58% coverage as if it meant something, in the same session where I
  demonstrated it does not.
- "The search is one entry point short of usable" — **wrong twice.** It also
  needed an objective producer, and then a trial-budget fix, and then a backend
  policy that does not sit at 99% CPU. Each was found by trying to run it, not by
  reading it.
- Wrote `StudySpec.max_steps` documented as "steps per trial, the single most
  important field to pin" while nothing applied it. The spec promised a budget the
  driver never delivered.
- Claimed a run-directory collision from two `train.py` processes sharing an
  `--output_dir`. **Wrong** — `attempts.jsonl` showed one attempt; the second pid
  was a `--num_workers 4` DataLoader worker inheriting argv. Checked before
  reporting, which is the only reason it is here as a retraction and not as a
  finding.
- Committed and pushed `fd05a82` with a test failing, having read "1099 passed"
  and missed "1 failed" one line above it in the same combined command.
