# Module Layout Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `main`'s `koopman_lm/` package into the role-based layout designed in `pr/module-reorg`, by replaying that PR's rename map mechanically rather than merging it.

**Architecture:** `pr/module-reorg` cannot be merged — a trial merge produces 72 conflicted files, because both it and `main` independently reorganized the same ancestor tree. However, `main`'s current `koopman_lm/` is *substantially the same layout that PR was written to transform*: 42 of its 49 pre-reorg paths are byte-identical to paths on `main` today. So the reorg reduces to a 14-entry rename map (all 14 applicable to `main`) plus a mechanical import rewrite. We apply that with `git mv` to preserve `git log --follow`, then place the 19 files that postdate the PR by the same rules.

**Tech Stack:** Python 3.10, PyTorch (CPU build for verification), pytest, git.

## Global Constraints

- **Never commit to `main`.** All work lands on branch `reorg-module-layout`, branched from `origin/main` (`b2d965e`).
- **Use `git mv` for every move.** Never delete-and-recreate; that breaks `git log --follow` and inflates the diff.
- **Baseline is `16 passed, 10 skipped`.** Every task must end at exactly this, or better. Never fewer passes.
- **The 10 skips are GPU tests** (`torch.cuda.is_available()` is false on `login-01`, which has no GPU). They are expected to skip and must not be "fixed".
- **Verification venv:** `/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/venv` — torch 2.13.0+cpu, pytest 9.1.1. Invoke as `$VENV/bin/pytest`. Do not add it to the repo.
- **Run tests with `PYTHONPATH=.`** from the repo root; the package is not pip-installed.
- **One commit per task.** Renames and content edits go in *separate* commits so reviewers can read the rename diff as a plain file listing.
- **Do not touch** `MQAR/`, `echo-ska/`, `lowrank_residual_cuda/`, `sys+toolcall/` — restored in `ebcf17d`, unrelated to this reorg.
- **Do not delete `pr/module-reorg` or `pr/eval-consolidation`.** They are the reference material, and 4 eval commits exist nowhere else.

---

## File Structure

The target layout, derived from `pr/module-reorg` at `f803539`:

```
koopman_lm/
  config.py                    <- globals/config.py  (stub at config.py is deleted)
  models/
    koopman_lm.py              (unchanged)
    baselines.py               (unchanged)
    recurrent.py               <- globals/modules/utils/recurrent.py
  modules/
    token_mixer/               attention.py, mamba.py, ska.py
    channel_mixer/             koopman.py, swiglu.py, (mlp.py, koopman_mlp_diag.py, norm.py)
    kernels/                   cholesky_update*.py, chunk_stats*.py, ska_operator.py,
                               factor_scan.py, lin_alg.py, + 10 new SKA/CUDA files
    wip/                       memory.py <- globals/modules/utils/last_layer_memory.py
  training/
    train.py, repro.py, data/
  evaluation/                  (unchanged)
  experiments/                 (unchanged)
  retrieval/                   (unchanged — already top-level, correct)
```

`globals/` is dissolved entirely.

---

### Task 1: Branch setup and baseline capture

**Files:**
- Create: none
- Modify: none

**Interfaces:**
- Produces: branch `reorg-module-layout` at `origin/main`; a recorded baseline all later tasks compare against.

- [ ] **Step 1: Fast-forward local main and branch**

```bash
cd /users/jkli/Koopman_Mamba
git switch main && git merge --ff-only origin/main
git switch -c reorg-module-layout
git log --oneline -1     # expect: b2d965e Merge pull request #10 ...
```

- [ ] **Step 2: Record the baseline**

```bash
VENV=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/venv
PYTHONPATH=. $VENV/bin/pytest code-tests -q 2>&1 | tail -3
```

Expected, exactly: `16 passed, 10 skipped`

If this does not match, STOP — the environment differs from what this plan assumes.

- [ ] **Step 3: Commit nothing**

Task 1 produces no commit. It is a gate, not a change.

---

### Task 2: Add a regression test for the stale `cholesky_update` import

This goes FIRST, before any move. The bug is real and currently untested — proving it with a failing test now means Task 4's import rewrite can't silently paper over it.

**Files:**
- Create: `code-tests/test_streaming_memory_imports.py`
- Test: the file itself

**Interfaces:**
- Consumes: `koopman_lm.globals.modules.utils.last_layer_memory.LastLayerRidgeMemory` (note: `LastLayerRidgeMemory`, NOT `LastLayerMemory`).
- Produces: a test that must keep passing after the Task 3 rename, at the module's new path `koopman_lm.modules.wip.memory`.

- [ ] **Step 1: Write the failing test**

```python
"""Regression: stream_write's lazy import must resolve.

`LastLayerRidgeMemory.stream_write` does a function-level
`from koopman_lm.cholesky_update import update_L_only`. That path is a
leftover from the old flat echo-ska-440m layout and does not exist in the
package layout. Because the import is lazy, nothing catches it until
stream_write is actually called -- and it IS called, from
models/koopman_lm.py and models/recurrent.py.

NOTE ON THE CURRENT FAILURE MODE: until the module reorg lands, this test
fails on `No module named 'koopman_lm.modules'` -- the NEW home of the
memory class, which does not exist yet. That is a different error from the
stale-import bug above. Both are resolved by the reorg. The import is done
inside the test body rather than at module scope so that this transitional
failure stays a single failing test instead of aborting collection for the
whole suite.
"""
import pytest
import torch


@pytest.mark.correctness
def test_stream_write_import_resolves():
    # Imported inside the test on purpose -- see the module docstring.
    from koopman_lm.modules.wip.memory import LastLayerRidgeMemory

    # Signatures verified against the real class:
    #   __init__(self, d_model, rank=64, ridge=0.01, ...)
    #   stream_reset(self, batch=1, device=None, dtype=torch.float32)
    #   stream_write(self, h, v, weight=1.0)
    # NOTE: the reset kwarg is `batch`, NOT `batch_size`.
    r, d, B = 4, 6, 2
    mem = LastLayerRidgeMemory(d_model=d, rank=r)
    mem.stream_reset(batch=B, device=torch.device("cpu"), dtype=torch.float32)

    h = torch.randn(B, d)
    v = torch.randn(B, d)

    # Before the fix this raises ModuleNotFoundError, not an assertion failure.
    mem.stream_write(h, v, weight=1.0)
```

- [ ] **Step 2: Run it and confirm it fails for the RIGHT reason**

```bash
VENV=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/venv
PYTHONPATH=. $VENV/bin/pytest code-tests -q; echo "exit=$?"
```

Run the FULL suite, with no `--ignore` and no extra flags. Using `--ignore` here hides exactly the failure mode this step exists to catch.

Expected right now: `16 passed, 1 failed, 10 skipped`, exit code 1. The new test FAILS with `ModuleNotFoundError: No module named 'koopman_lm.modules'` because it targets the post-reorg path, which does not exist yet. It goes green during Task 3/4.

CRITICAL: the import must be inside the test body, not at module scope. At module scope pytest aborts the whole session on the collection error -- exit code 2, ZERO tests run -- which violates the Global Constraint that the suite never drop below 16 passes.

To confirm the underlying bug independently of the rename, run:

```bash
PYTHONPATH=. $VENV/bin/python -c "
from koopman_lm.globals.modules.utils.last_layer_memory import LastLayerRidgeMemory as M
import torch
m = M(d_model=6, rank=4); m.stream_reset(batch=2, device=torch.device('cpu'), dtype=torch.float32)
m.stream_write(torch.randn(2,6), torch.randn(2,6), weight=1.0)"
```

Expected (verified on `b2d965e`): traceback ending in

```
  File ".../last_layer_memory.py", line 165, in stream_write
    from koopman_lm.cholesky_update import update_L_only
ModuleNotFoundError: No module named 'koopman_lm.cholesky_update'
```

- [ ] **Step 3: Commit the failing test**

```bash
git add code-tests/test_streaming_memory_imports.py
git commit -m "test: pin stream_write's lazy cholesky_update import (currently broken)"
```

---

### Task 3: Pure renames — no content changes

**Files:**
- Modify (move only): the 14 mapped files + 13 new-file placements listed below.

**Interfaces:**
- Consumes: branch from Task 1.
- Produces: the target directory tree. Imports are still broken at the end of this task — that is expected and fixed in Task 4.

- [ ] **Step 1: Create package directories**

```bash
mkdir -p koopman_lm/modules/{token_mixer,channel_mixer,kernels,wip}
```

- [ ] **Step 2: Apply the 14-entry rename map from pr/module-reorg**

```bash
git mv koopman_lm/globals/modules/attention.py                    koopman_lm/modules/token_mixer/attention.py
git mv koopman_lm/globals/modules/mamba.py                        koopman_lm/modules/token_mixer/mamba.py
git mv koopman_lm/globals/modules/ska/ska.py                      koopman_lm/modules/token_mixer/ska.py
git mv koopman_lm/globals/modules/koopman_mlp.py                  koopman_lm/modules/channel_mixer/koopman.py
git mv koopman_lm/globals/modules/ska/cholesky_update.py          koopman_lm/modules/kernels/cholesky_update.py
git mv koopman_lm/globals/modules/ska/cholesky_update_triton.py   koopman_lm/modules/kernels/cholesky_update_triton.py
git mv koopman_lm/globals/modules/ska/chunk_stats.py              koopman_lm/modules/kernels/chunk_stats.py
git mv koopman_lm/globals/modules/ska/chunk_stats_exact.py        koopman_lm/modules/kernels/chunk_stats_exact.py
git mv koopman_lm/globals/modules/ska/core.py                     koopman_lm/modules/kernels/ska_operator.py
git mv koopman_lm/globals/modules/ska/factor_scan.py              koopman_lm/modules/kernels/factor_scan.py
git mv koopman_lm/globals/modules/utils/recurrent.py              koopman_lm/models/recurrent.py
git mv koopman_lm/globals/modules/utils/last_layer_memory.py      koopman_lm/modules/wip/memory.py
git mv koopman_lm/globals/modules/utils/repro.py                  koopman_lm/training/repro.py
git mv koopman_lm/globals/__init__.py                             koopman_lm/modules/__init__.py
```

- [ ] **Step 3: Place the 13 files that postdate the PR, by the same rules**

Ten SKA/CUDA files follow the `ska/* -> kernels/` rule already established above:

```bash
git mv koopman_lm/globals/modules/ska/adaptive_chunking.py        koopman_lm/modules/kernels/adaptive_chunking.py
git mv koopman_lm/globals/modules/ska/prefix_scan.py              koopman_lm/modules/kernels/prefix_scan.py
git mv koopman_lm/globals/modules/ska/cuda_prefix_scan.py         koopman_lm/modules/kernels/cuda_prefix_scan.py
git mv koopman_lm/globals/modules/ska/fast.py                     koopman_lm/modules/kernels/fast.py
git mv koopman_lm/globals/modules/ska/fused_state_reference.py    koopman_lm/modules/kernels/fused_state_reference.py
git mv koopman_lm/globals/modules/ska/incremental_transport.py    koopman_lm/modules/kernels/incremental_transport.py
git mv koopman_lm/globals/modules/ska/inverse_cholesky.py         koopman_lm/modules/kernels/inverse_cholesky.py
git mv koopman_lm/globals/modules/ska/small_rank_backend.py       koopman_lm/modules/kernels/small_rank_backend.py
git mv koopman_lm/globals/modules/ska/csrc                        koopman_lm/modules/kernels/csrc
```

Three channel-mixer files:

```bash
git mv koopman_lm/globals/modules/mlp.py                          koopman_lm/modules/channel_mixer/mlp.py
git mv koopman_lm/globals/modules/koopman_mlp_diag.py             koopman_lm/modules/channel_mixer/koopman_diag.py
git mv koopman_lm/globals/modules/norm.py                         koopman_lm/modules/channel_mixer/norm.py
```

`koopman_lm/retrieval/` and `koopman_lm/training/data/mix.py` are already at correct paths — do not move them.

- [ ] **Step 4: Move config and remove the dead stub**

`koopman_lm/config.py` on `main` is a redirect stub that re-exports from `globals.config`. The real config becomes `koopman_lm/config.py`.

```bash
git rm koopman_lm/config.py
git mv koopman_lm/globals/config.py koopman_lm/config.py
git rm koopman_lm/globals/modules/ska/__init__.py \
       koopman_lm/globals/modules/utils/__init__.py \
       koopman_lm/globals/modules/__init__.py
```

- [ ] **Step 5: Add the new package `__init__.py` files**

```bash
touch koopman_lm/modules/token_mixer/__init__.py \
      koopman_lm/modules/channel_mixer/__init__.py \
      koopman_lm/modules/kernels/__init__.py \
      koopman_lm/modules/wip/__init__.py
git add koopman_lm/modules/*/__init__.py
```

- [ ] **Step 6: Verify `globals/` is gone and the diff is renames-only**

```bash
test -d koopman_lm/globals && echo "FAIL: globals/ still exists" || echo "ok: globals/ dissolved"
git diff --cached --name-status -M | awk '{print substr($1,1,1)}' | sort | uniq -c
```

Expected (verified by dry run): `29 R`, `2 A`, `2 D`, **`1 M`**.

The single `M` is `koopman_lm/config.py` and is correct: the 13-line redirect stub is replaced by the 355-line real config. Git reports modify rather than rename because the destination path already existed. Confirm it is that file and nothing else:

```bash
git diff --cached --name-status -M | grep -E "^M" | sed 's/^/  /'
```

Expected exactly: `M	koopman_lm/config.py`

The A/D counts are lower than the 4-new/4-deleted you might expect because all empty `__init__.py` files share one blob hash, so git pairs some as renames. That is cosmetic.

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor: move modules into role-based layout (renames only, no content changes)"
```

---

### Task 4: Mechanical import rewrite

**Files:**
- Modify: every `.py` under `koopman_lm/` and `code-tests/` that references an old path.

**Interfaces:**
- Consumes: the tree from Task 3.
- Produces: a tree where every intra-package import resolves. Task 2's test goes green here.

- [ ] **Step 1: Write the rewrite script**

Create `/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/rewrite_imports.py`:

```python
"""Rewrite koopman_lm import paths after the Task 3 renames.

Longest-prefix-first so that e.g. globals.modules.ska.core is rewritten
before globals.modules.ska, which would otherwise steal the match.
"""
import pathlib, re, sys

MAP = {
    "koopman_lm.globals.modules.ska.cholesky_update_triton": "koopman_lm.modules.kernels.cholesky_update_triton",
    "koopman_lm.globals.modules.ska.cholesky_update":        "koopman_lm.modules.kernels.cholesky_update",
    "koopman_lm.globals.modules.ska.chunk_stats_exact":      "koopman_lm.modules.kernels.chunk_stats_exact",
    "koopman_lm.globals.modules.ska.chunk_stats":            "koopman_lm.modules.kernels.chunk_stats",
    "koopman_lm.globals.modules.ska.adaptive_chunking":      "koopman_lm.modules.kernels.adaptive_chunking",
    "koopman_lm.globals.modules.ska.cuda_prefix_scan":       "koopman_lm.modules.kernels.cuda_prefix_scan",
    "koopman_lm.globals.modules.ska.prefix_scan":            "koopman_lm.modules.kernels.prefix_scan",
    "koopman_lm.globals.modules.ska.fused_state_reference":  "koopman_lm.modules.kernels.fused_state_reference",
    "koopman_lm.globals.modules.ska.incremental_transport":  "koopman_lm.modules.kernels.incremental_transport",
    "koopman_lm.globals.modules.ska.inverse_cholesky":       "koopman_lm.modules.kernels.inverse_cholesky",
    "koopman_lm.globals.modules.ska.small_rank_backend":     "koopman_lm.modules.kernels.small_rank_backend",
    "koopman_lm.globals.modules.ska.factor_scan":            "koopman_lm.modules.kernels.factor_scan",
    "koopman_lm.globals.modules.ska.core":                   "koopman_lm.modules.kernels.ska_operator",
    "koopman_lm.globals.modules.ska.fast":                   "koopman_lm.modules.kernels.fast",
    "koopman_lm.globals.modules.ska":                        "koopman_lm.modules.token_mixer.ska",
    "koopman_lm.globals.modules.utils.last_layer_memory":    "koopman_lm.modules.wip.memory",
    "koopman_lm.globals.modules.utils.recurrent":            "koopman_lm.models.recurrent",
    "koopman_lm.globals.modules.utils.repro":                "koopman_lm.training.repro",
    "koopman_lm.globals.modules.koopman_mlp_diag":           "koopman_lm.modules.channel_mixer.koopman_diag",
    "koopman_lm.globals.modules.koopman_mlp":                "koopman_lm.modules.channel_mixer.koopman",
    "koopman_lm.globals.modules.attention":                  "koopman_lm.modules.token_mixer.attention",
    "koopman_lm.globals.modules.mamba":                      "koopman_lm.modules.token_mixer.mamba",
    "koopman_lm.globals.modules.mlp":                        "koopman_lm.modules.channel_mixer.mlp",
    "koopman_lm.globals.modules.norm":                       "koopman_lm.modules.channel_mixer.norm",
    "koopman_lm.globals.config":                             "koopman_lm.config",
    # Stale flat-layout leftovers from echo-ska-440m (the Task 2 bug):
    "koopman_lm.cholesky_update_triton":                     "koopman_lm.modules.kernels.cholesky_update_triton",
    "koopman_lm.cholesky_update":                            "koopman_lm.modules.kernels.cholesky_update",
}

ORDERED = sorted(MAP.items(), key=lambda kv: -len(kv[0]))
root = pathlib.Path(sys.argv[1])
changed = 0
for p in list(root.rglob("*.py")):
    if ".git" in p.parts:
        continue
    src = p.read_text(encoding="utf-8")
    out = src
    for old, new in ORDERED:
        out = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", new, out)
    if out != src:
        p.write_text(out, encoding="utf-8")
        print(f"  rewrote {p.relative_to(root)}")
        changed += 1
print(f"{changed} files rewritten")
```

- [ ] **Step 2: Run it**

```bash
SP=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad
$SP/venv/bin/python $SP/rewrite_imports.py /users/jkli/Koopman_Mamba
```

- [ ] **Step 3: Handle the one import the script cannot fix**

`koopman_lm/modules/kernels/cholesky_update.py` contains
`from koopman_lm import cholesky_update_triton as t` — a `from X import <submodule>` form, not a dotted path, so the regex does not match it. Fix by hand:

```python
# in _try_triton(), replace:
            from koopman_lm import cholesky_update_triton as t
# with:
            from koopman_lm.modules.kernels import cholesky_update_triton as t
```

This is the second stale import found earlier. It is swallowed by `except Exception`, so it fails silently and permanently disables the Triton path — no test will catch it. It must be fixed by inspection.

- [ ] **Step 3b: Fix `_CONFIGS_ROOT` — REQUIRED, found by dry run**

Moving `config.py` from `koopman_lm/globals/config.py` up to `koopman_lm/config.py` shortens its path to the repo root by one level, so its relative pointer at `configs/` now resolves ABOVE the repo. Symptom if skipped: `test_production_configs.py` fails both tests with `FileNotFoundError: .../configs/180m_prefix_scan.yaml`, looking one directory too high.

In `koopman_lm/config.py`:

```python
# replace:
_CONFIGS_ROOT = Path(__file__).parent.parent.parent / "configs"
# with:
_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"
```

`pr/module-reorg` made this exact change at `f803539:koopman_lm/config.py:180` — confirm against it if unsure.

- [ ] **Step 4: Verify every intra-package import resolves (static)**

This catches broken paths that pytest misses, because lazy function-level imports never execute during a test run — which is exactly how both stale imports survived for months. Create `$SP/imports.py`:

```python
import ast, os, sys
root = sys.argv[1]
pkg = os.path.join(root, "koopman_lm")
mods = set()
for dp, dn, fn in os.walk(pkg):
    for f in fn:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(dp, f), root)[:-3].replace(os.sep, ".")
            mods.add(rel)
            if rel.endswith(".__init__"):
                mods.add(rel[:-9])
bad = []
for dp, dn, fn in os.walk(pkg):
    for f in fn:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dp, f)
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e:
            bad.append((os.path.relpath(p, root), e.lineno, f"SYNTAX ERROR: {e.msg}"))
            continue
        cur = os.path.relpath(p, root)[:-3].replace(os.sep, ".")
        base = cur[:-9] if cur.endswith("__init__") else cur.rsplit(".", 1)[0]
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.level:
                    parts = base.split(".")
                    up = parts[:len(parts) - (n.level - 1)] if n.level > 1 else parts
                    m = ".".join(up + ([n.module] if n.module else []))
                elif n.module and n.module.split(".")[0] == "koopman_lm":
                    m = n.module
                else:
                    continue
                if m not in mods:
                    bad.append((os.path.relpath(p, root), n.lineno, f"unresolved module: {m}"))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] == "koopman_lm" and a.name not in mods:
                        bad.append((os.path.relpath(p, root), n.lineno, f"unresolved module: {a.name}"))
if bad:
    for b in sorted(set(bad)):
        print(f"    {b[0]}:{b[1]}  {b[2]}")
    raise SystemExit(1)
print("    all intra-package imports resolve; no syntax errors")
```

Run it:

```bash
SP=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad
$SP/venv/bin/python $SP/imports.py /users/jkli/Koopman_Mamba
```

Expected: `all intra-package imports resolve; no syntax errors`

Known limitation: it does not catch the `from koopman_lm import <submodule>` form, which is why Step 3 must be done by hand.

- [ ] **Step 5: Verify the Triton path actually loads now**

On a CPU box `import triton` fails regardless, so do NOT assert the import succeeds. Assert instead that the failure is about **triton** and not about **koopman_lm** — that is the difference between "optional dep absent" (fine, resolves on a GPU node) and "wrong path, permanently dead" (the bug).

```bash
PYTHONPATH=. $SP/venv/bin/python -c "
import importlib.util as u
spec = u.find_spec('koopman_lm.modules.kernels.cholesky_update_triton')
assert spec is not None, 'FAIL: module path does not resolve -- the rename is wrong'
print('  path resolves:', spec.origin)
try:
    import koopman_lm.modules.kernels.cholesky_update_triton
    print('  imports cleanly (triton installed)')
except ModuleNotFoundError as e:
    assert e.name == 'triton', f'FAIL: expected missing triton, got missing {e.name}'
    print('  correct: only triton itself is missing (expected on CPU)')
"
```

- [ ] **Step 6: Run the full suite**

```bash
PYTHONPATH=. $SP/venv/bin/pytest code-tests -q 2>&1 | tail -5
```

Expected: `17 passed, 10 skipped` — the baseline 16, plus Task 2's regression test now green.

If it is 16 passed and Task 2's test errors on collection, the rename in Task 3 Step 3 did not land. If it is 16 passed and Task 2's test FAILS with `ModuleNotFoundError: koopman_lm.cholesky_update`, the MAP's last two entries did not apply.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rewrite imports for the new module layout

Also fixes two stale flat-layout imports inherited verbatim from
echo-ska-440m, where koopman_lm.cholesky_update{,_triton} were valid
top-level modules:

  wip/memory.py         stream_write raised ModuleNotFoundError when called
  kernels/cholesky_update.py  silently disabled the Triton kernel via
                              a swallowed ImportError"
```

---

### Task 5: Reconcile SwiGLU

`pr/module-reorg` adds `modules/channel_mixer/swiglu.py`. `main` already has a SwiGLU inside what is now `modules/channel_mixer/mlp.py`. Only one should survive. This is the single place where the PR and the new work genuinely overlap, and it needs judgment, not a script.

**Files:**
- Modify: `koopman_lm/modules/channel_mixer/mlp.py`
- Reference: `git show f803539:koopman_lm/modules/channel_mixer/swiglu.py`

**Interfaces:**
- Consumes: the tree from Task 4.
- Produces: one canonical SwiGLU implementation.

- [ ] **Step 1: Diff the two implementations**

```bash
git show f803539:koopman_lm/modules/channel_mixer/swiglu.py > /tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/swiglu_pr.py
diff /tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad/swiglu_pr.py koopman_lm/modules/channel_mixer/mlp.py
```

- [ ] **Step 2: Decide and record**

If they are functionally equivalent, keep `main`'s (it is the one the trained configs exercise) and do not import the PR's. If the PR's is a strict improvement, port the delta into `mlp.py` — do NOT add a second file. Either way, write one sentence in the commit message saying which won and why.

STOP and ask the user if the two differ in a way that would change model numerics. That is a research decision, not a refactor decision.

- [ ] **Step 3: Verify**

```bash
SP=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad
PYTHONPATH=. $SP/venv/bin/pytest code-tests -q 2>&1 | tail -3
```

Expected: `17 passed, 10 skipped`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: single canonical SwiGLU in channel_mixer/mlp.py"
```

---

### Task 6: Final verification and push

**Files:** none

- [ ] **Step 1: Confirm history is preserved through the renames**

```bash
git log --follow --oneline koopman_lm/modules/token_mixer/ska.py | tail -5
```

Expected: history reaching back past the rename into `echo-ska-440m/koopman_lm/ska.py`. If it stops at this branch's first commit, a `git mv` was done as delete+add.

- [ ] **Step 2: Full suite + static check together**

```bash
SP=/tmp/claude-851721614/-users-jkli/53e14a4c-7ea7-4dad-90f7-cd3441a7ea97/scratchpad
PYTHONPATH=. $SP/venv/bin/pytest code-tests -q 2>&1 | tail -3
$SP/venv/bin/python $SP/imports.py /users/jkli/Koopman_Mamba
test -d koopman_lm/globals && echo "FAIL: globals/ survived" || echo "ok"
```

- [ ] **Step 3: Confirm nothing outside the package moved**

```bash
git diff --stat origin/main HEAD -- MQAR echo-ska lowrank_residual_cuda 'sys+toolcall'
```

Expected: empty.

- [ ] **Step 4: Push and open a PR**

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/koopman_deploy -o IdentitiesOnly=yes" \
  git push git@github.com:anusridh97/Koopman_Mamba.git \
  reorg-module-layout:refs/heads/reorg-module-layout
```

---

### Task 7: Retire the superseded PRs

Only after Task 6's PR is merged.

- [ ] **Step 1: Cherry-pick the 4 eval commits**

These exist ONLY on `pr/eval-consolidation`. They must be replayed onto the new layout before that branch is deleted:

```
0aa3664  Extract the one canonical load_model into evaluation/loader.py
17824a5  Extract shared _greedy_generate into evaluation/generation.py
dd48a0f  Unify the NIAH generator + scorers into evaluation/tasks/niah.py
34fc331  Dedup lm_harness_eval.py's checkpoint-loading via evaluation/loader.py
```

They touch `koopman_lm/evaluation/`, which this reorg does not move, so they should apply with far less friction than the module commits did. Treat as a separate plan if conflicts exceed a handful of files.

- [ ] **Step 2: Close, then delete**

Close `pr/module-reorg` and `pr/eval-consolidation` on GitHub with a comment pointing at the replacement PR. Delete the branches only after Step 1 lands.
