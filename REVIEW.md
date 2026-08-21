# Reviewing `jack/search-and-provenance`

49 commits, 104 files, +11,734 / −361. Written to be reviewed commit-by-commit
rather than as one diff, because 89% of the insertions are new files where a diff
tells you nothing you wouldn't get from reading the file.

**Test counts.** 1051 passed / 43 skipped standard; 1127 / 31 with optuna on the
path. `main` was 499 / 28.

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
| **C** | Adaptive search (`experimentation/sweep/search/`, 8 modules): the space declared once, TPE + median pruning, ask/tell over the **unmodified** run system | unit-tested to 92–100%; **NOT runnable yet — see §6** |
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
$CPU -m pytest code-tests/ -q                                    # 1051 / 43 skipped
PYTHONPATH=.:$OPT $CPU -m pytest code-tests/ -q                   # 1127 / 31
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
| `golden_4m_curve.json` + comparator | the training loop still trains identically. Noise floor measured at 0.0002 by running twice |

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

**The adaptive search cannot be run.** Two independent blockers:
- there is no `search/__main__.py`. `experimentation/sweep/search/__init__.py`
  advertises `python -m experimentation.sweep.search <study.yaml>`; that
  invocation does not exist. My docstring, my error.
- **nothing writes the objective file.** Optuna reads
  `run_dir/eval/<ckpt>/quick_eval.json`, and the only caller of
  `write_quick_eval` anywhere is `scripts/verify_search_and_provenance.sbatch`,
  by hand. A study today would train N models successfully and record N FAILs.

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
