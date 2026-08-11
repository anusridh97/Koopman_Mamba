# Codebase reading progress — handoff, 2026-08-10

Jack is reading the codebase end to end in order to refactor it. This records
what is already understood, what was **verified false** in the existing docs, and
what remains. Everything below was checked by reading or running the code — where
a finding contradicts a committed document, the document is wrong.

**If you are an agent picking this up: read §1 first.** It will stop you
repeating work and stop you relaying claims that are already known to be stale.

---

## 1. Read this before trusting any prose doc

The pattern, verified repeatedly: **docstrings are accurate, prose docs have
drifted.** Every module `__init__.py` claim tested held. Nearly every line
number, count, and shell command in the markdown had rotted.

### Verified stale

| Claim | Reality |
|---|---|
`README.md:50-63` and `CODEBASE_GUIDE.md:23-52` layout | `training/`, `evaluation/`, `experiments/`, `retrieval/` are **not** in `koopman_lm/` — all four moved to top-level `experimentation/` in `e4f5fdc` |
`README.md:77-88` "## Train", `CODEBASE_GUIDE` §3–§4 | **documents a dead path** — see below |
`CODEBASE_GUIDE` §3 "registry has exactly 4 entries, configs/ ships 4 files" | 11 and 11 |
`CODEBASE_GUIDE` §4 "17 passed, 10 skipped" · §6 "27 passed, 0 skipped" | **480 passed, 28 skipped** |
`CODEBASE_GUIDE` §5-low "cosmetic, deliberately not fixed" | all three were fixed |
`CODEBASE_GUIDE` §5b `config.py:306` | `_CONFIGS_ROOT` is at line 345 |
`HANDOFF-2026-08-08.md` §4 "391 passing (413 with diagnostics)" | 480 |
`HANDOFF` §5.7 "`load_model` duplicated across three eval files" | two define `def load_model`; `lm_harness_eval.py` open-codes it at `:100`/`:131` |
`HANDOFF` §6 `read/05-how-to-train.md` | not in the repo; it's in `/users/jkli/koopman-consolidation-plan/read/` |
`HANDOFF` §8 branch inventory | missing 7 newer branches (`agent/*`, `jack/foundation`, `jack/overnight-cleanup`, `jack/package-restructure`) |

### The dead path — most misleading thing in the docs

`scripts/train_50m.sh:7` and `train_180m.sh:5` exec `pretrain.sh <size>_prefix_scan`.
`pretrain.sh:14-25` has cases for only `50m)` and `180m)`, then
`*) echo "unknown production config"; exit 1`. **Both launchers fail immediately.**
Independently, `build_config('50m_prefix_scan')` raises — commit `f43a27e` deleted
the byte-identical `_prefix_scan` YAML duplicates and registry aliases but never
updated the scripts.

The live entry point is `python -m experimentation.run <spec.yaml>`.

### Two referenced design docs do not exist

- `specs/2026-08-08-trainer-behavioural-diff.md` — **never committed on any
  branch.** `HANDOFF` §5.3 calls it the written decision doc for the trainer
  merge. It is lost; reading `train.py` against `table2.py` and
  `mqar_finetune.py` is how to reconstruct it.
- `specs/2026-08-07-structural-review.md` — exists only on `jack/struct-review`
  (`a05db51`).

### Verified accurate — safe to trust

`README`'s config table (exact to the parameter: 50m = 50,034,044; 180m =
176,342,680) and its "How it fits together" diagram (lines 17-46) ·
`modules/__init__.py` (all 7 class names exist at their claimed roles; `kernels/`
defines zero `nn.Module`) · `experimentation/__init__.py` (all subpackages
present; **zero** imports of `experimentation` anywhere in `koopman_lm/`) ·
`kernels/README.md` (all 14 `.py` + 2 `.cu` references resolve) ·
`docs/SKA_DIAGNOSTICS.md` (clean; its four "missing" paths are the left column of
an old→new mapping table).

---

## 2. Environment facts — do not re-derive

```bash
V=/users/jkli/.venvs/koopman-cpu/bin
$V/pytest code-tests/ -q                              # 480 passed, 28 skipped, ~35s
$V/pytest code-tests/ -m "correctness and not gpu" -q # what CI gates on
$V/python scripts/check_imports.py                    # AST gate, no torch needed
```

Branch `jack/package-restructure`, on `origin` = `anusridh97/Koopman_Mamba`. Per
Jack, this branch will likely become the new `main`.

**CI does not run on branch pushes** — `.github/workflows/ci.yml` triggers on
`push` to `main` and on `pull_request` only. Opening a PR is what runs the gate.

**Exclude worktrees from every search.** There are two roots — `.worktrees/` and
`.claude/worktrees/` — holding ~23 repo copies. A `getattr` audit returned 220
hits before filtering and **13** after. Scope greps to
`koopman_lm experimentation code-tests scripts` or exclude both directories
explicitly.

---

## 3. What is already understood

### `koopman_lm/` — essentially complete

`config.py` (deeply: `__post_init__`, tuple coercion, `param_count_estimate`,
`build_config`, `CONFIG_REGISTRY`) · `__init__.py` · `pooling.py` ·
`models/koopman_lm.py` (all 17 methods) · `models/recurrent.py` +
`recurrent_state.py` · `modules/__init__.py` (the organizing rules) ·
`modules/seq/{mamba,ska_block,fast}.py` · `modules/mlp/{koopman,koopman_diag}.py` ·
`kernels/lin_alg.py::spec_w` · `kernels/ska_operator.py`

**Gaps in `koopman_lm/`:** `models/baselines.py` (293) — the ablation arms, and
the only place `CausalAttentionBlock` is used. `kernels/prefix_scan.py` (1,302) —
the largest file and **the live path** (both production configs set
`ska_prefix_scan: true`); only skimmed.

### `experimentation/` — `run/` nearly done, rest untouched

Covered: `__init__.py` (the boundary rule) · `run/spec.py` (the four dataclasses,
`group_id`/`run_id`) · `run/__main__.py` · `run/data_verify.py` ·
`run/artifacts.py` · `run/resolve.py` and `run/launch.py` partially.

**Left in `run/`:** `slurm.py` (198), `eval_result.py` (51).
**Untouched:** `training/` (1,944), `evaluation/` (2,618), `experiments/` (1,186),
`retrieval/` (723), `sweep/` (358), `results.py` (101).

---

## 4. Findings — verified, and not written down anywhere else

1. **`ska_mode: parallel`.** `MambaSKAParallelBlock.forward` is
   `mamba(x) + ska(x) - x`; the `- x` corrects double-counting because both
   children are already residual blocks. Ablating SKA collapses it to exactly
   `mamba(x)` — a bit-identical Mamba baseline from the same checkpoint. Costs
   +3,776,656 params on 50m vs `replace`.

2. **`ska_block.py:120-122`'s stated rationale is false.** It claims `replace`
   mode "removes local sequence-mixing depth exactly where the global-memory
   branch is inserted." But both branches read the *same* `x`, so SKA gains
   nothing at its own depth in either mode — at the **first** SKA layer the two
   modes give bit-identical input. The real effect is cumulative (+1/+2/+3 Mamba
   blocks upstream at layers 7/11/15) and about **matched capacity between
   ablation arms**, not about what SKA reads. Docstring fix offered and not yet
   applied.

3. **~170 lines of dead SKA numerics.** `_spectral_normalize_power_iter`
   (`ska.py:51`) is called only from `_post_cholesky_pytorch` (`:184`) and
   `_post_cholesky_triton` (`:266`), which have **zero call sites**. The live
   spectral clamp is `spec_w` in `kernels/lin_alg.py:51` — same algorithm,
   `iters=20` not 6. The source already says so at `ska.py:379-385`, 190 lines
   away from the dead code. Likely covered by `1b27e89` on `pr/module-reorg`,
   awaiting salvage.

4. **`modules/seq/fast.py` is live and has two bugs.** Reached via
   `train.py:168` behind `--ska_fast` (default off, nothing passes it). It
   hardcodes the legacy chunked path with no branch for `prefix_scan` /
   `inverse_cholesky` / `exact_intrachunk` — and **both production configs set
   `ska_prefix_scan: true`**, so `--ska_fast` would silently swap the algorithm.
   It also uses `self.eta` (`:135`) where the canonical forward uses
   `_resolve_eta()`, which would `AttributeError` under the squash regime.
   `test_ska_fast_patch.py` builds a default module, so it is blind to both.

5. **`configs/` holds three design generations**, confirmed on three independent
   axes:

   | tier | configs | ska_mode / mlp | short_conv | eta-gamma policy | init |
   |---|---|---|---|---|---|
   1 | `50m`, `180m` | parallel / swiglu | off | new | mamba_safe |
   2 | `1m`, `180m_dense/gated/v2`, `370m` | replace / koopman | off | **old** | legacy |
   3 | `440m`, `880m`, `1p5b`, `3b` | replace / koopman | **on** | new | legacy |

6. **Five configs depend on a stale default staying stale.** `SKAModule`'s
   constructor defaults match **production** (η=γ=1 fixed, LayerScale on); the
   `KoopmanLMConfig` *field* defaults encode the **old** policy. The five tier-2
   configs get the old policy by *omission*. "Modernizing" the field defaults
   would silently change all five — measured: `config_hash` moves for exactly
   those five, and `1m`/`370m` parameter counts change.

7. **The spectral clamp does not strictly guarantee ‖A‖ ≤ 1**, but it does not
   matter. On random Gaussians `spec_w`'s 20 iterations leave the post-clamp norm
   at up to 1.11 (6 iterations: 1.29). On **real** operators captured from a live
   forward, σ_max ≤ 0.9725 and the clamp never fires at all — `W` is already
   contractive, corroborating `config.py:74-81`.

8. **`_spectral_radius` is biased upward on real operators.** Median absolute
   error 1.6e-2, max 1.2e-1, over-estimating. The assumption that these operators
   are near-normal is **false**: measured σ_max/ρ median **1.686**, max 2.078.
   Diagnostics-only, so low stakes, but the metric is not the guarantee it looks
   like.

9. **`harness.py:56-60` discards a hash mismatch.** Its docstring says it
   "hashes it to confirm the recorded `cfg_hash`"; the code **overwrites** the
   recorded value with the recomputed one. Schema drift is silently accepted.

10. **No checkpoint records the code that produced it.** All four meta writers
    (`train.py:507`, `adapt.py:253`, `table2.py:174`, `mqar_finetune.py:129`)
    omit any git commit — while `eval_result.py`'s envelope *does* carry
    `git_commit`. So results are attributable and checkpoints are not.

11. **`artifacts.py` has no locking.** Zero `flock`/`fcntl`/`O_EXCL` in the
    package. `create_run_dir` checks `final/` then `mkdir(exist_ok=True)` —
    a TOCTOU race — and the guard only keys on **completed** runs, so two
    simultaneous launches of the same spec both proceed into one directory.

12. **Empty orphan directories** left by `ff13957`'s renames: `modules/kernels/`
    and `modules/channel_mixer/` are empty and untracked (git cannot track empty
    dirs). `modules/token_mixer/` was a third; Jack deleted it, and it was
    verified genuinely empty. Note `koopman_lm/kernels/` — the real one, 18 files
    — is a **sibling** of `modules/`, which is what the empty
    `modules/kernels/` husk makes confusing.

13. **13 `getattr(cfg, "...", default)` sites, zero typos** — and since every
    name is a real dataclass field, all the defaults are unreachable. Pure
    downside; converting to `getattr(cfg, f)` without a default would make a
    mistyped name fail loudly.

---

## 5. Changes landed this session

```
900494b  refactor(config): declare tuple coercion per field instead of listing them
9276e7c  refactor(run): move resolve_model_config into spec.py beside data_spec_from_dict
4c06a64  docs(spec): precision policy -- per-component dtypes on the model config
0e2fd2a  docs(spec): run provenance -- make "what did this run do?" answerable
```

`900494b` and `9276e7c` are pushed. The two design docs are **committed but not
pushed**.

One commit was created and then deliberately removed: `7d7aea3`
(hash `runtime.precision` into `group_id`/`run_id`) was pushed, judged premature,
then `reset --hard` + `--force-with-lease`. **It no longer exists.** Identity is
back to `50m-first-real` = `81033b58`/`5467934f`. That episode is also the live
demonstration of why `code_id` alone is weak provenance — see the provenance spec
§3.1.

### Two design docs, both awaiting implementation

- `specs/2026-08-10-precision-policy-design.md` (381 lines) — three flat fields
  on `KoopmanLMConfig` (`compute_precision`, `ska_precision`, `mlp_precision`)
  under the rule *"`compute_precision` sets the floor; the other two raise it for
  components whose math needs more."* §12 has a six-step order whose first four
  steps are behavior no-ops.
- `specs/2026-08-10-run-provenance-design.md` (322 lines) — `code_id` into all
  four checkpoint-meta writers, a byte-reproducible `source.tar.gz` in the run
  dir (216K/436K measured), the policy that numerics changes must have config
  surface, and golden behavior tests. §9 step 2 is two lines and closes the main
  gap.

---

## 6. What to read next, in order

```
1. run/eval_result.py     51    the result envelope; closes the loop to results.py
2. run/slurm.py          198    array jobs, --signal=B:USR1@300 preemption wiring
3. models/baselines.py   293    the ablation arms (gap in koopman_lm/)
4. training/train.py     592    + optim 56, repro 65, resume 115
5. training/data/        603    pretokenize 398, dataset 65, mix 140
6. results.py + evaluation/harness.py   326   the aggregation walk
7. sweep/                358
8. kernels/prefix_scan.py 1302  read 54, 103, 479, 538, 1166; SKIP 899-1165
```

Skip entirely: `archive/` (4,186 lines, proven inert by test) · the `wip/` memory
surface (`memory.py` 251 + four `KoopmanLM` methods — **zero** call sites, and
`prefill_memory`/`step_memory_only_debug` self-document as not working) ·
`modules/seq/fast.py` (already understood; off by default).

**Read `experimentation/` differently from `koopman_lm/`.** The model package
rewarded file-by-file; this is infrastructure and rewards tracing one path:
`configs/runs/50m-first-real.yaml` → the five orchestration steps in
`run/__main__.py` → `launch.py`'s generated command → `train.py`'s loop → `meta.pt`
→ `results.py`. That spec is a good reference artifact — its self-description was
verified accurate (only `max_seq_len` differs from `configs/50m.yaml`).

---

## 7. Open items, none blocking

1. `ska_block.py:120-122` docstring rewrite (finding §4.2). Offered, not applied.
2. `artifacts.py` → `write_policy.py` rename. Agreed the name misleads
   ("artifact write policy" is its own docstring's first line); deferred until
   the seven-PR stack merges, since a rename creates rebase conflict surface.
3. A `.running` sentinel for `create_run_dir` (finding §4.11). `os.mkdir` is
   atomic even on NFS, so ~5 lines.
4. `docs/gen_inventory.py` — untracked in the working tree, predates this session.
5. The two design docs are unpushed.
6. **The seven-PR stack (#12–#18) and the salvage list.** If this branch becomes
   `main` without salvaging `pr/module-reorg` and `pr/eval-consolidation`, the
   commits `HANDOFF` §5.7 audited as genuinely unique are stranded — including
   `1b27e89` (deletes exactly the dead code in finding §4.3), `38d6629`,
   `96ce016`, the four eval-dedup commits, `results/echo50m_table4/`, and
   `2605.06997v1.md`, **the paper itself**. Highest-stakes open item.
