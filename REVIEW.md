# Reviewing `jack/search-and-provenance`

64 commits, 124 files, +15,010 / −373. Written to be reviewed commit-by-commit
rather than as one diff, because most of the insertions are new files where a diff
tells you nothing you wouldn't get from reading the file.

**Test counts.** 1136 passed / 45 skipped standard; ~1150 with optuna on the path.
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
| **E** | Test infrastructure: import gate, static undefined-name check, `param_groups` characterization, loss-alignment conformance, a golden training curve | see §4 |
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
| `golden_mqar_curve.json` | the synthetic loop, the path whose shift convention differs. Noise floor **0.0** with `--deterministic`; **0.53 without it**, which is how that missing flag was found |
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

Overriding the backend is *correct* (trials must be comparable, so a search cannot
inherit whichever kernel the base spec chose), so the fix is not to stop pinning
it. But the default policy for a study is `exact_auto`, and a real study on it
would appear to hang. `configs/search/smoke-4m.yaml` now uses `proxy_chunked` --
space.py's own "cheap approximate screen" -- and the underlying slowness is
**unexplained and untracked beyond this note.**

**Never exercised:** multi-GPU / DDP. Which SKA backend a run actually selected
(nothing asserts it). `ska_precision='fp64'` on GPU. An optuna study driving real
training.

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
