# Resume-equivalence divergence: root-cause diagnosis

Status: **root cause found and fixed.** Diagnoses the failure of
`code-tests/test_resume_equivalence.py::test_interrupted_and_resumed_run_matches_uninterrupted_run`,
the acceptance criterion for §5.5 of `2026-08-07-run-system-design.md`.
Branched from `jack/foundation` at `50cb445`.

---

## 1. Divergence characterisation

Ran the test as-is (before any fix). All 5 tensors in the state dict differ,
not just `embed.weight` (which fails first only because it's first in
iteration order):

| tensor | max\|a-c\| |
|---|---|
| `embed.weight` | 0.02805 |
| `lin1.weight` | 0.02420 |
| `lin1.bias` | 0.01369 |
| `lin2.weight` | 0.02676 |
| `lin2.bias` | 0.00044 |

Magnitude is ~1e-2, not ~1e-8 (rules out float non-associativity/numerical
drift) and not O(1) (rules out a completely different data trajectory — e.g.
wrong indices or a badly desynced sampler). This is the signature of "correct
inputs, one bad random draw partway through" — consistent with an RNG-stream
or step/LR misalignment, not a gross logic error.

Isolating to exactly the first post-resume step (train run A straight to
step 21, train run B to 20/kill/resume/train C to 21) reproduces the same
signature at smaller scale — all 5 tensors already differ after **exactly
one** resumed step (max\|a-c\| ranges 0.00005-0.0053), and run A's loss at
step 20 (4.282775) already differs from run C's loss at step 20 (4.280323)
**before any optimizer update runs**. So the divergence is a forward-pass
difference at the very first resumed step, not something that accumulates
over many steps.

## 2. Hypotheses tested

All tests below use a diagnostic script built on the test's own helpers
(`_build_model_optim_sched`, `_run_loop`, `apply_resume_state`), instrumented
to compare intermediate state rather than only final weights.

**Dataloader position (wrong index / desynced sampler).** Eliminated.
Extracted the raw index list `full_perm = epoch_permutation(len(ds), seed, 0)`
and the resume tail `resume_indices(len(ds), seed, 0, 80)`, and confirmed
`tail == full_perm[80:]` exactly. Then built run A's *continuous* DataLoader
and pulled its 21st batch, and run C's *fresh* DataLoader (over the resume
tail) and pulled its first batch: `torch.equal(ids_a, ids_c)` and
`torch.equal(labels_a, labels_c)` are both `True`. The data run C consumes at
the resume point is bit-identical to what run A consumes at the same step.

**RNG restore ordering (the test's own lines 152-156 warning).** Eliminated,
in the form originally hypothesised. Compared `torch.random.get_rng_state()`
at three points: run A's live state immediately before its own step-20
forward pass; run B's state captured by `capture_rng_state()` immediately
after its step-20 update (what gets serialized into `resume.pt`); and run
C's restored state immediately after `apply_resume_state()` returns (i.e.
after `model_c`'s fresh, RNG-consuming `__init__`, and before the resumed
loop's first forward pass). All three are `torch.equal` to each other. The
order the test's comment demands -- restore after init, before forward --
is what the code actually does, and it correctly reproduces the RNG
snapshot.

**Optimizer state (AdamW `step` / `exp_avg` / `exp_avg_sq`).** Eliminated.
Deep-compared `optimizer.state_dict()["state"]` per parameter between `opt_b`
(pre-save, in memory) and `optimizer_c` (post-restore): `step`, `exp_avg`,
and `exp_avg_sq` are `torch.equal` for all 5 parameters. Also checked the
value that actually drives the next update, `optimizer.param_groups[0]["lr"]`
(not just the scheduler's internal bookkeeping) -- it matches exactly
(`0.006112604669781572` both sides). This is not a coincidence:
`apply_resume_state` calls `optimizer.load_state_dict()` *before*
`scheduler.load_state_dict()` (`resume.py:111-112`), and it's the
optimizer's own saved `param_groups` that carries `lr` forward -- stock
PyTorch's `LambdaLR.load_state_dict()` only restores the scheduler's own
`__dict__` (`last_epoch`, `_last_lr`) and does not push anything back into
`optimizer.param_groups`, so if the order were reversed this would be a live
bug, but it isn't reversed.

**Scheduler `last_epoch` off-by-one.** Eliminated. `sched_b.state_dict()["last_epoch"]`
and `scheduler_c.state_dict()["last_epoch"]` are both `20`; `_last_lr` matches
exactly on both sides.

**`_run_loop`'s resume boundary (wrong step math).** Eliminated as a
step-arithmetic problem -- `(r_step, r_epoch) == (step, epoch)` holds, and
the isolation test above shows correct data/state entering the first
resumed step. But this hunt is what led to the actual root cause, in the
*mechanics* of how `_run_loop` re-enters at that boundary -- see §3.

## 3. Root cause

**`torch.utils.data.DataLoader` draws one `int64` from the ambient global
RNG stream every time a *new* iterator is created over it -- even with
`num_workers=0` and even when `sampler=` is a plain, non-shuffling list --
and this draw is invisible to `capture_rng_state`/`restore_rng_state`.**

Source, torch 2.13.0+cpu,
`.../site-packages/torch/utils/data/dataloader.py:690-694`, in
`_BaseDataLoaderIter.__init__` (runs on every `iter(loader)`, i.e. every time
a `for batch in loader:` begins a fresh pass):

```python
self._base_seed = (
    torch.empty((), dtype=torch.int64)
    .random_(generator=loader.generator)
    .item()
)
```

`loader.generator` is `None` unless explicitly passed, in which case
`.random_(generator=None)` falls back to the *default* (global) generator --
the exact stream `torch.random.get_rng_state()`/`set_rng_state()` manage and
`nn.Dropout` draws from. Confirmed empirically: constructing a `DataLoader`
does not touch the global RNG, but the first `iter(loader)` call always
does, regardless of `sampler=<list>` and `num_workers=0`.

**Why this breaks resume specifically, and only sometimes:** both
`_run_loop` (`code-tests/test_resume_equivalence.py:91`) and the real
training loop (`koopman_lm/training/train.py:385-388`) construct a *new*
`DataLoader` every time the outer `while` loop begins a *new epoch* -- which
is also, unconditionally, every time either function is (re-)entered, epoch
boundary or not:

```python
epoch_loader = DataLoader(
    train_ds, batch_size=args.per_device_train_batch_size,
    sampler=indices, num_workers=args.num_workers,
    pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
```

No `generator=` is passed at either call site. In an **uninterrupted** run,
this DataLoader-construction RNG draw happens once per epoch -- a cost both
runs being compared would pay identically, at the same point, if they always
resumed at epoch boundaries. But `_run_loop`/`train.py` also reconstruct the
DataLoader immediately at a **resume** point, including a **mid-epoch**
resume (`samples_consumed > 0`, `resume_indices(...)` instead of
`epoch_permutation(...)`). The uninterrupted twin (run A in the test) never
reconstructs its DataLoader mid-epoch -- it just keeps iterating the one
built at epoch start. So the resumed run pays this hidden global-RNG draw
**one extra time**, exactly at the boundary where `apply_resume_state` has
just finished restoring the RNG stream to bit-parity with the uninterrupted
run. That extra draw immediately desyncs the two streams before the very
first resumed dropout call, and `_StandInLM.drop` (`p=0.3`) makes that
desync visible in every downstream weight.

**Confirmed as the cause, not merely correlated:** patched only the
`DataLoader(...)` construction inside `_run_loop` to pass
`generator=torch.Generator()` (a private generator, never fed from or into
the global stream), reran the exact single-extra-step isolation scenario,
and got bit-exact equality on all 5 tensors (`max|a-c| == 0.0` everywhere)
and identical per-step losses at every step 0-20 including the resume
boundary itself (both branches: `4.204225540161133`).

## 4. Is the bug in the implementation or the test?

**Implementation.** `_run_loop` is a faithful mirror of `train.py`'s Task-7
loop (per its own docstring), and both independently hit the identical
`DataLoader()`-without-`generator=` construction at the resume boundary.
This is a genuine gap in exact-resume, present in the real trainer that
`--resume`/`SIGUSR1` drives, not an artifact of the test's stand-in loop.

The test's demand for bitwise equality (`torch.equal`, `atol=0`) is correct
and achievable -- see §3's confirmation -- so this is not a case of "the
test demands more precision than the design supports." §5.2's table ("What
exact resume requires") is simply missing a row: the DataLoader's own
internal `_base_seed` draw is state that affects reproducibility and was
not accounted for, because it doesn't appear anywhere in `resume.pt` or in
any of the documented RNG streams (python/numpy/torch/torch_cuda) -- it's a
torch *internals* detail, not an application-level RNG consumer, so it
wasn't a natural candidate for the original inventory.

## 5. The fix

Pass an explicit, private `generator=torch.Generator()` to every
`DataLoader(...)` construction in both call sites, so the internal
`_base_seed` draw is sourced from a generator nobody else reads from or
writes to, and never touches the global stream that `capture_rng_state`/
`restore_rng_state`/dropout share:

- `koopman_lm/training/train.py:385-388` (`epoch_loader = DataLoader(...)`)
- `code-tests/test_resume_equivalence.py:91` (`_run_loop`'s
  `loader = DataLoader(...)`)

This is the minimal fix: it doesn't require adding a new field to
`resume.pt`, and it doesn't change `epoch_permutation`/`resume_indices`
(which were already correct). The alternative fix -- capturing and
replaying the DataLoader's `_base_seed` as part of resume state -- would be
strictly more complex (a new per-construction seed to persist, re-derived
once per epoch *and* once per resume) for no benefit, since nothing in this
codebase depends on `_base_seed`'s value (no worker RNG uses it
meaningfully when `num_workers=0`, and `seed_worker` already reseeds
numpy/random from `torch.initial_seed()` per worker independently).
Decoupling it from the global stream is strictly the right fix, not a
workaround.

Applied and verified: `test_interrupted_and_resumed_run_matches_uninterrupted_run`
passes with `torch.equal`/`atol=0` after this change. Full suite run showed
no new failures beyond the 4 pre-existing parked ones
(`test_gated_variant_differs_only_in_mlp_gated`,
`test_param_count_estimate_reflects_v2_additions`,
`test_baseline_embedding_std_and_tied_weight_identity[build_transformer]`,
`test_transformer_initial_logit_scale_and_loss_near_uniform`).

## 6. Is §5.5's claim, as written, achievable?

**Yes, as written, with this fix.** "Training N steps, killing, resuming,
and training to 2N must produce the same weights as training 2N
uninterrupted, under determinism" holds bitwise once every DataLoader
construction site in the resume-relevant loop is pinned to a private
generator. No relaxation of the tolerance is needed, and the spec's "exact
equality is the correct bar" argument in the test's own docstring (lines
12-18) survives untouched.

**Recommended spec addendum**, not a retraction: §5.2's table should gain a
row --

| Component | Today | Needed |
|---|---|---|
| DataLoader's internal `_base_seed` draw | draws from the global RNG stream on every `iter()`, invisible to `resume.pt` | construct every `DataLoader` in the resume-relevant loop with an explicit private `generator=`, so it never touches the stream `capture_rng_state`/`restore_rng_state` manage |

-- and §5.5 should note explicitly that the invariant was only exercised
end-to-end for a **mid-epoch** resume (`samples_consumed > 0`); an
epoch-boundary resume would not have surfaced this bug, since both the
continuous and resumed runs would reconstruct their DataLoader at the same
relative point and pay the hidden draw identically. Future resume-adjacent
work (§6's trainer unification, which adds `SyntheticTask`/`ShardTask`
dataloaders) should carry this same private-`generator=` discipline
forward, since it's easy for a new call site to reintroduce the same gap
silently.
