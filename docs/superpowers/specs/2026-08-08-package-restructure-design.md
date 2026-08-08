# Package restructure design — PR #9 on the stack

**Date:** 2026-08-08
**Branch:** cut from `jack/diagnostics` (tip of the eight-PR stack `#12..#19`)
**Status:** design, approved in conversation; not yet implemented

---

## 1. Goal

Turn `koopman_lm` into a narrow, publishable model package by moving everything
cluster- and study-specific out to `experimentation/`, **changing zero
semantics**.

Four things this is meant to buy, all stated by the author:

1. **Internal clarity** — the model core stops being tangled with run/eval/sweep
   machinery.
2. **`pip install`-able by outsiders** — someone gets a model without inheriting
   a Slurm account, a wandb project, or an lm-eval dependency tree.
3. **Easier to add architecture variants** — partially; the config reshape that
   fully delivers this is deliberately deferred (§7).
4. **Easier to experiment with.**

Success criterion, executable:

```bash
# fresh venv, no wandb / datasets / lm-eval installed
pip install koopman-lm
python -c "from koopman_lm import KoopmanLM, build_config; KoopmanLM(build_config('50m'))"
```

## 2. Model: mamba_ssm

`mamba_ssm/` contains only the model: `models/`, `modules/`, `ops/`
(organized by backend: triton, cute, tilelang), `distributed/`, `utils/`.
Everything else lives outside the package at repo root — `evals/`
(`lm_harness_eval.py`, 1,287 bytes), `benchmarks/`, and `tests/` mirroring the
package tree. There is **no training code in the repo at all**.

We adopt the boundary and the test layout. We do not adopt "no trainer" — the
trainer moves out of the package but stays in the repo.

## 3. Current state

The boundary we want is **already almost real**. Of the six packages leaving,
exactly one edge points the wrong way:

```
koopman_lm/models/koopman_lm.py::encode
    -> from koopman_lm.retrieval.encoder import pool_sequence
```

`pool_sequence` is 17 lines of pure tensor math whose own docstring says it is
"shared by KoopmanLM.encode and any encoder." It is misfiled; it belongs in the
core.

Everything else already flows core <- periphery. Import counts into the core
from the departing packages (all of which keep working unchanged, since
`koopman_lm.models` stays where it is):

| package | import lines into core |
|---|---|
| `evaluation/` | 16 |
| `training/` | 8 |
| `experiments/` | 8 |
| `run/` | 4 |
| `sweep/` | 1 |
| `retrieval/` | 1 |

Size split of the current 19,711-line package:

| | lines |
|---|---|
| `models/` + `modules/` + `kernels/` + `config.py` (stays) | 10,144 |
| `training/ evaluation/ run/ sweep/ retrieval/ experiments/ results.py` (leaves) | 9,567 |

## 4. Target layout

```
koopman_lm/                    <- installable, publishable
  __init__.py                  public API: KoopmanLM, KoopmanLMConfig, build_config
  config.py
  models/      koopman_lm.py  baselines.py  recurrent.py  recurrent_state.py
  modules/     seq/  mlp/  wip/
  kernels/     lin_alg  prefix_scan  ...  csrc/*.cu
  pooling.py                   <- pool_sequence lands here

experimentation/               <- repo root, NOT packaged, not installed
  training/     train.py  optim.py  data/
  evaluation/
  run/  sweep/  results.py
  retrieval/
  experiments/

tests/                         <- mirrors koopman_lm/
  models/  modules/  kernels/
  experimentation/             <- tests for the unpackaged half
configs/  scripts/  docs/  archive/
```

## 5. Commits

Reviewable individually, in this order.

| # | Commit | Content |
|---|---|---|
| 1 | `chore(archive): rescue prefix_bench + SKAv9` | cherry-pick `7dad6f0` from `jack/overnight-cleanup`. Additive, `archive/` only. |
| 2 | `refactor(models): pool_sequence into the core` | the single inverted edge (§3) |
| 3 | `refactor: git mv six packages -> experimentation/` | pure move, no content change |
| 4 | `refactor: rewrite imports + module-path strings` | see §8 |
| 5 | `test: pin config_hash/group_id/run_id across the move` | see §6 |
| 6 | `build: pyproject deps + console script` | see §8 |
| 7 | `test: code-tests/ -> tests/ mirroring the package` | update `testpaths` |

Commit 1 is not structural, but it belongs here: it puts `prefix_bench` (1,793
lines, no counterpart anywhere in the tree) and `SKAv9.py` under `archive/` with
provenance, which is the precondition for deleting the four PR #11 folders later
(§10).

## 6. The property that makes a whole-repo move reviewable

`config_hash`, `group_id`, and `run_id` must be **byte-identical** before and
after, for all 11 shipped model configs and all 4 run specs.

Generate the expected values on the pre-move tree and commit them **first**, so
the pin cannot be written to match whatever the post-move code happens to
produce.

```python
# tests/test_restructure_is_semantics_free.py
EXPECTED_CONFIG_HASHES = { "50m": "...", "180m": "...", ... }   # generated pre-move
EXPECTED_RUN_IDS       = { "50m-fineweb-3b.yaml": ("<group_id>", "<run_id>"), ... }
```

If this passes, the move provably changed no science, and review collapses to
*"did anything change besides paths?"*

Also required to land:

- full suite green
- `scripts/check_imports.py` AST gate clean
- clean-venv install test (§1)

## 7. OPEN: the model config schema

**Not in this PR, and not yet decided.** `KoopmanLMConfig` keeps its ~60 flat
fields. This section records the problem and the options; it does not pick one.

An earlier draft of this spec presented "adopt mamba's nested dicts" as merely
deferred. That was wrong, and §7.3 explains why.

### 7.1 The problem being solved

Adding one field to `KoopmanLMConfig` today breaks eval and resume for every
completed run.

```python
def _check_model_key_set(model_dict):                 # run/resolve.py:192
    expected = {f.name for f in dataclasses.fields(KoopmanLMConfig)}
    actual   = set(model_dict)
    if expected - actual or actual - expected:
        raise ValueError(...)
```

Exact set equality. A new field lands in `missing` for every existing
*materialized* `spec.yaml`, so `load_materialized_spec` raises — and that is the
function eval and resume both call. Authoring configs under `configs/runs/` are
unaffected: the check is deliberately scoped to materialized specs, because
`extends:` leaves legitimately carry partial key sets.

### 7.2 Why the check is strict on purpose

From its own docstring:

> A spec.yaml written before a field was added would otherwise **silently
> receive the new field's default** on load, making an old run look like it
> declared a value it never had.

The strictness is a deliberate choice of a **loud** failure over a **silent**
misdescription of a completed experiment. Any replacement has to be judged
against that, not just against convenience.

### 7.3 Nested dicts do not straightforwardly fix this

Mamba's pattern (`ssm_cfg: dict`, `attn_cfg: dict`, opaque, splatted into the
module constructor) gets its extensibility from the dict being **unvalidated**,
not from it being **nested**. Two consequences:

- Old `spec.yaml` carrying `ska_cfg: {rank: 24}` loads cleanly after a new
  `gate_temp` option is added, and the run is now described as having a
  `gate_temp` it never had. That is exactly the hazard §7.2 exists to prevent,
  pushed down one level.
- Typed sub-dataclasses (`ska_cfg: SKAConfig`) buy grouping and readability but
  **nothing** on the extensibility axis — adding a field to `SKAConfig` still
  changes the validated key set.

Mamba can take this trade because mamba has no run-provenance system. This repo
does.

### 7.4 Options

| | Grouped / readable | New field doesn't break old specs | Validation | Old runs stay honest |
|---|---|---|---|---|
| **A.** Flat (today) | no | no | yes | yes (loud) |
| **B.** Nested, typed sub-dataclasses | yes | no | yes | yes (loud) |
| **C.** Nested, untyped dicts (mamba) | yes | yes | no | **no (silent)** |
| **D.** Flat + schema version | no | yes | yes | yes (loud) |

**D** is the only row that gives nothing up. Record a schema version in
`spec.yaml`; let a known set of fields be absent from older versions with their
historical defaults recorded explicitly, so an old run resolves to what it
actually had rather than to today's default. Adding a field becomes a two-line
migration entry instead of a break. More machinery than **C**, and it does not
improve readability — but it is the only option that preserves the §7.2
guarantee while removing the cost.

**B** is worth considering purely for readability of a 60-field dataclass, on
its own merits, independent of the extensibility question.

### 7.5 Decided regardless of which option wins

- The reshape is a **separate PR** from the move. The move is mechanical and
  provable (§6); a config reshape is semantic and touches run identity. One diff
  containing both makes a regression impossible to attribute.
- If the chosen option changes `dataclasses.asdict(spec.model)`, every
  `group_id`/`run_id` changes, because `_scientific_payload` hashes it directly.
  Accept a **clean break** and record the old→new mapping for the runs that
  exist (effectively one: `50m-first-real`).
- A flatten-before-hash compatibility shim was considered and **rejected**: it
  only covers configs expressible in the old flat vocabulary, and it puts a
  permanent translation table inside the one function that guarantees run
  identity.

## 8. Known breakages to fix in commit 4/6

Strings, not imports — easy to miss because no import gate catches them.

- `run/slurm.py:139-140` hardcodes `"-m", "koopman_lm.training.train"`. This
  string is baked into every generated `launch.sbatch` and `launch_line.sh`.
- `pyproject.toml` `[project.scripts] koopman-train = "koopman_lm.training.train:main"`
  breaks outright. Drop it or repoint it.
- Stale `python -m koopman_lm.*` paths: `README.md` (7), `docs/CODEBASE_GUIDE.md`
  (7), `experiments/table2.py` (4), `sweep/` (6), `run/__main__.py` (2),
  `training/train.py` (2).
- `pyproject.toml` core `dependencies` should shed `wandb` and `datasets` — the
  model does not need them; only the trainer does. `[tool.setuptools.packages.find]`
  already scopes to `koopman_lm*`, so `experimentation/` is excluded automatically.
- `[tool.setuptools.package-data] "koopman_lm.kernels" = ["csrc/*.cu"]` stays.

Non-breakage worth noting: the generated sbatch already does `cd {repo_root}`, so
`python -m experimentation.training.train` resolves without `experimentation/`
being installed. Existing run dirs' `launch_line.sh` go stale — acceptable, they
are historical artifacts.

## 9. Explicitly out of scope

| Item | Why |
|---|---|
| **Deduplication** | Every target (4 training loops, duplicated `save_checkpoint`/`load_checkpoint`, 12 CLI entry points, 2× `make_mqar`) lives in a directory that is moving. Combined, every file reads as delete+add, and §6 stops proving anything. Own PR, post-merge. |
| **builders + probes** (`ca1e666`, `1cbc6af`) | Features, not structure. Land into the final shape rather than dragging them through the move. |
| **Any change to the model config schema** | Open question, own PR. §7 |
| **Deleting `csrc/small_rank_ext.cu`** | ~1,750 lines, zero callers (verified statically, including dynamic-import paths). Real dead weight in a package about to be published — but it predates this work. Anu call. |
| **`koopman_lm/modules/wip/`** | A directory named "wip" inside a package about to be published, imported in production at `models/koopman_lm.py:257` (`LastLayerRidgeMemory`). Rename/promote/leave is a judgment call. Same category as the above. Anu call. |
| **Deleting the four PR #11 folders** | Not possible here — they do not exist on `jack/diagnostics`. §10. |
| **HF `from_pretrained`/`save_pretrained`** | Worth having if publishing, but a feature, not a move. |

## 10. Merge-time concerns (not this PR)

Local `main` is **2 commits behind `origin/main`**:

```
c349805 Merge pull request #11 from anusridh97/restore-deleted-folders
ebcf17d Restore folders dropped by the PR #10 folder collapse
```

The entire stack was built on the older local `main`. Merging
`jack/diagnostics` into current `origin/main` is **textually clean** — verified
by a scratch-worktree merge — but resurrects four pre-reorg top-level trees:

```
archive/ code-tests/ configs/ docs/ koopman_lm/
echo-ska/  lowrank_residual_cuda/  MQAR/  sys+toolcall/     <- resurrected
```

Git cannot flag this: from its perspective the diagnostics side never touched
those paths. The result contains duplicates — `echo-ska/` alongside
`archive/reference/echo_jax.py`, `MQAR/` alongside `evaluation/mqar/` +
`models/baselines.py`.

Survival audit of those 22 files against `jack/diagnostics` (blob-hash checked):

| File(s) | Survives? | Where |
|---|---|---|
| `echo-ska/echo_jax.py` | yes | `archive/reference/echo_jax.py` — code identical, doc header added. Live oracle for `test_jax_reference.py`. |
| `echo-ska/prepare_data.py` | probably | `training/data/pretokenize.py` has a strictly larger function surface. Surface checked, behavior not. |
| `echo-ska/train_echo.py` | **no** | JAX/TPU harness; z-loss term and TPU mesh have no counterpart |
| `MQAR/mqar_ska_mamba_benchmark.py` | reorganized | `evaluation/mqar/mqar.py` + `models/baselines.py` |
| `MQAR/MQAR_SKA_Mamba_Benchmark-2.ipynb` | **yes, after commit 1** | `archive/MQAR/MQAR_SKA_Mamba_Benchmark-2.ipynb` |
| `MQAR/prefix_bench/` (13 files, 1,793 lines) | **yes, after commit 1** | `archive/MQAR/prefix_bench/` — no counterpart before it; `curricula.py:196-201` documents the gap its absence causes |
| `lowrank_residual_cuda/` (3 files) | **no** | deliberately: unbuildable, unimported, untested, wrong gauge for SKA |
| `sys+toolcall/SKAv9.py` | **yes, after commit 1** | `archive/toolcall/SKAv9.py` |

Commit 1 of this PR rescues more than first assessed: `prefix_bench`, `SKAv9.py`
**and** the MQAR notebook (874 lines), all under `archive/` with provenance in
`archive/README.md` and inertness enforced by `code-tests/test_archive_is_inert.py`.

That leaves exactly **two** files dropped from the working tree on purpose:
`echo-ska/train_echo.py` (JAX/TPU harness; its z-loss term and TPU multi-host
mesh have no counterpart) and `lowrank_residual_cuda/` (3 files). Both remain
recoverable from `c349805` indefinitely **provided `main` advances by merge and
is never force-pushed**. `echo-ska/prepare_data.py` and
`MQAR/mqar_ska_mamba_benchmark.py` are superseded rather than dropped.

Anu's PR #11 restored these on the grounds that "nothing on main replaces"
them — still true for those three — so the deletion is their call, not a
merge-time slip.

**For the call with Anu:**

1. The four folders — delete via a cleanup commit on `main` after the merge?
2. `echo-ska/train_echo.py` and `lowrank_residual_cuda/` — the only two files
   dropped from the working tree with no counterpart. Confirm.
3. `csrc/small_rank_ext.cu` + `small_rank_backend.py` (~1,750 lines, zero callers).
4. `koopman_lm/modules/wip/`.

## 11. Sequencing

```
#12..#19    existing stack, unchanged
#20 (9th)   this PR
            -> then call Anu, merge the whole chain, handle §10 on main
post-merge  deduplication (own PR, informed by an audit)
post-merge  builders + probes re-landed into the final shape
```

Landing dedup and builders/probes *after* the merge rather than as stack
entries 10 and 11 is deliberate: a nine-deep stack already makes any change to
an early PR expensive, and a restructure at the tip makes it worse. Post-merge
they are ordinary PRs against a clean single `main`.

## 12. Risks

| Risk | Mitigation |
|---|---|
| The move silently changes behavior | §6 hash pin + full suite + import gate |
| A stale module-path string survives into a generated sbatch | §8 checklist; `test_docs_commands_resolve.py` covers README/config pointers once builders+probes land |
| Review on `#12..#19` forces a rebase through a moved tree | Land #9 and merge promptly; do not let the stack sit |
| `experimentation/` not being installed breaks an invocation | Generated sbatch already `cd`s to repo root; verify in the smoke test |
