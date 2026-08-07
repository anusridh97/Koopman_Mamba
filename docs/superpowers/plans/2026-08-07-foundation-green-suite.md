# Foundation: Green Test Suite and Frozen Module Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the test suite from 29 failures + 1 collection error to green, fix the three real bugs those tests exposed, and land the final module layout so no further reorganization is needed.

**Architecture:** Every task in this plan has a **failing test that already exists** — the tests were recovered onto `jack/test-recovery` and are currently red. So each task is: run the red test, fix the source, watch it go green, commit. No test is written from scratch, and no test may be modified to make it pass. Tasks 1–5 are independent source fixes; Task 6 lands the layout rename and re-points every import; Task 7 corrects two stale references the rename invalidates.

**Tech Stack:** Python 3.10+, PyTorch 2.13 (CPU), pytest 9.1. No new dependencies.

## Global Constraints

- **Base branch:** `jack/foundation`, created from `jack/test-recovery` (see Task 0).
- **Virtualenv:** `/scratch/m000151/jkli/venvs/koopman-cpu`. Referred to below as `$V`. Export it once: `export V=/scratch/m000151/jkli/venvs/koopman-cpu`.
- **Full-suite command:** `PYTHONPATH=. $V/bin/pytest code-tests -q --continue-on-collection-errors -p no:cacheprovider`
- **No GPU on this node.** 28 tests are `gpu`/`jax`-marked and will skip. That is expected and correct; a skip is not a failure.
- **Never edit a test to make it pass.** Every failure in this plan is a source bug or a missing source file. Exactly two exceptions, both justified in place: Task 4 Step 3 (renaming a config the test loads, where the two files were byte-identical so the assertion is unchanged in meaning) and Task 5 (fixture data provably below the filter threshold, assertion correct).
- **No new third-party dependencies.**
- **Commit at the end of every task**, with the test output that justifies it.
- **Baseline:** 29 failed, 208 passed, 28 skipped, 1 error. Final target: **0 failed, 0 errors, 28 skipped.**
- **On intermediate counts:** each task states which named tests it must turn green. Verify *those tests* pass and that the total failure count strictly decreases with no previously-passing test breaking. Do **not** treat the illustrative totals as exact contracts — the failure categories overlap between files, so intermediate arithmetic is approximate. Only the final state (0 failed, 0 errors) is a hard gate.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `koopman_lm/modules/token_mixer/attention.py` → `modules/seq/attention.py` | RoPE + causal attention. Fix dtype promotion. | 1, 6 |
| `koopman_lm/evaluation/mqar/mqar.py` | MQAR generator. Restore the missing `mqar_cell_fits` predicate. | 2 |
| `koopman_lm/evaluation/mqar/__init__.py` | Re-export surface. | 2 |
| `koopman_lm/config.py` | Add `mamba_headdim`; extend `CONFIG_REGISTRY`; drop duplicate entries. | 3, 4 |
| `koopman_lm/modules/token_mixer/mamba.py` → `modules/seq/mamba.py` | Pass `headdim` to `Mamba2` only when pinned. | 3, 6 |
| `configs/{1m,180m_gated,180m_v2,370m,440m,880m,1p5b,3b}.yaml` | Restored scale ladder. | 3 |
| `configs/{50m,180m}_prefix_scan.yaml` | Deleted — byte-identical duplicates. | 4 |
| `code-tests/test_retrieval_data.py` | Fixture strings lengthened past the stub filter. | 5 |
| Whole `koopman_lm/` tree | Layout rename: `token_mixer`→`seq`, `channel_mixer`→`mlp`, `modules/kernels`→`kernels`. | 6 |
| `pyproject.toml:81` | `package-data` key pointing at a package that no longer exists. | 7 |
| `docs/superpowers/specs/2026-08-07-run-system-design.md` | Path citations invalidated by Task 6. | 7 |

---

## Task 0: Set up the working branch

**Files:**
- No source changes.

**Interfaces:**
- Consumes: branches `jack/test-recovery` (23 recovered tests, CI, `check_imports.py`) and `jack/run-system-design` (the spec).
- Produces: branch `jack/foundation` containing both, and a verified red baseline.

- [ ] **Step 1: Create the branch and bring the spec along**

The spec is a single new documentation file and shares no paths with the test
branch, so this merge is guaranteed conflict-free. Bringing it in now means
Task 7 can edit it in the same tree.

```bash
cd /users/jkli/Koopman_Mamba
export V=/scratch/m000151/jkli/venvs/koopman-cpu
git switch -c jack/foundation jack/test-recovery
git merge --no-edit jack/run-system-design
```

- [ ] **Step 2: Confirm the red baseline**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q --continue-on-collection-errors -p no:cacheprovider 2>&1 | tail -3`

Expected, exactly:
```
29 failed, 208 passed, 28 skipped, 1 error
```

If the numbers differ, **stop** and report. Every later task's success
criterion is measured against this baseline, so a different starting point
means the plan's arithmetic is wrong.

---

## Task 1: Fix the RoPE dtype regression

The `cos`/`sin` angle tables are built in float32. Multiplying a bf16 tensor by
them silently promotes `q` and `k` to float32, while `v` — which never passes
through RoPE — stays bf16. `scaled_dot_product_attention` then rejects the
mixed dtypes. `torch.autocast` hides this because it re-casts every SDPA input,
so it surfaces only in pure bf16 with no autocast (e.g. DeepSpeed bf16).

**Files:**
- Modify: `koopman_lm/modules/token_mixer/attention.py:8-20`
- Test: `code-tests/test_baseline_training_contract.py::test_rope_preserves_dtype_so_attention_runs_without_autocast`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_apply_rope(x: Tensor) -> Tensor` now returns a tensor of the same dtype as its input. Signature unchanged.

- [ ] **Step 1: Run the failing test**

Run:
```bash
PYTHONPATH=. $V/bin/pytest "code-tests/test_baseline_training_contract.py::test_rope_preserves_dtype_so_attention_runs_without_autocast" -q -p no:cacheprovider
```

Expected: FAIL with `RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16`

- [ ] **Step 2: Apply the fix**

Replace the body of `_apply_rope` in `koopman_lm/modules/token_mixer/attention.py`. Two changes: `.float()` on the two input halves, and `.to(x.dtype)` on the result.

```python
def _apply_rope(x):
    """Rotary position embeddings applied to x of shape (B, H, T, D).

    The angle tables are built in float32 for precision, but the result is cast
    back to x's dtype. Without that cast the float32 tables silently PROMOTE q
    and k to float32 while v -- which never goes through RoPE -- stays in the
    activation dtype, and scaled_dot_product_attention then rejects the mixed
    dtypes ("Expected query, key, and value to have the same dtype").

    Under torch.autocast this was invisible, because autocast casts every SDPA
    input to bf16 regardless. It only surfaces when the model runs in pure bf16
    with no autocast, e.g. under DeepSpeed bf16. Casting here is numerically
    identical for the autocast path (same float32 value, same single rounding to
    bf16) and makes the non-autocast path correct.
    """
    B, H, T, D = x.shape
    half = D // 2
    device = x.device
    theta = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(T, device=device).float()
    ang   = pos.unsqueeze(1) * theta.unsqueeze(0)        # (T, D/2)
    cos   = ang.cos()[None, None]                         # (1, 1, T, D/2)
    sin   = ang.sin()[None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1).to(x.dtype)
```

- [ ] **Step 3: Verify the test passes**

Run:
```bash
PYTHONPATH=. $V/bin/pytest "code-tests/test_baseline_training_contract.py::test_rope_preserves_dtype_so_attention_runs_without_autocast" -q -p no:cacheprovider
```

Expected: `1 passed`

- [ ] **Step 4: Verify nothing else regressed**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q --continue-on-collection-errors -p no:cacheprovider 2>&1 | tail -3`

Expected: `28 failed, 209 passed, 28 skipped, 1 error` — one fewer failure, one more pass.

- [ ] **Step 5: Commit**

```bash
git add koopman_lm/modules/token_mixer/attention.py
git commit -m "fix: cast RoPE output back to input dtype

The float32 angle tables promoted q and k while v stayed bf16, so SDPA
rejected the mixed dtypes. Invisible under autocast, which re-casts every
SDPA input; surfaces in pure bf16 (e.g. DeepSpeed). Numerically identical
on the autocast path.

Caught by test_rope_preserves_dtype_so_attention_runs_without_autocast."
```

---

## Task 2: Restore `mqar_cell_fits`

`code-tests/test_mqar_generator.py` imports `mqar_cell_fits` from
`koopman_lm.evaluation.mqar`. The function does not exist anywhere in the tree.
Because this is an import error at module scope, it **aborts collection of the
entire suite** unless `--continue-on-collection-errors` is passed — so in CI it
reports zero tests run rather than three failures.

**Files:**
- Modify: `koopman_lm/evaluation/mqar/mqar.py` (add the function; use it in the existing assert)
- Modify: `koopman_lm/evaluation/mqar/__init__.py` (re-export)
- Test: `code-tests/test_mqar_generator.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `mqar_cell_fits(seq_len: int, num_kv_pairs: int, num_queries: int | None = None) -> bool`, importable as `from koopman_lm.evaluation.mqar import mqar_cell_fits`.

- [ ] **Step 1: Run the failing test**

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_mqar_generator.py -q -p no:cacheprovider`

Expected: collection ERROR — `ImportError: cannot import name 'mqar_cell_fits' from 'koopman_lm.evaluation.mqar'`

- [ ] **Step 2: Add the predicate**

In `koopman_lm/evaluation/mqar/mqar.py`, insert immediately after `import torch` (line 14) and before `def make_mqar`:

```python
def mqar_cell_fits(seq_len, num_kv_pairs, num_queries=None):
    """Whether the repo-local KV and query blocks fit, including exact fits."""
    num_queries = num_kv_pairs if num_queries is None else num_queries
    return 2 * num_kv_pairs + 2 * num_queries <= seq_len
```

- [ ] **Step 3: Route the existing assert through it**

In the same file, `make_mqar` currently open-codes the same arithmetic. Replace
the assert at line 32-33 so there is exactly one definition of "does this cell
fit":

Find:
```python
    assert 2 * num_kv_pairs + 2 * num_queries <= seq_len, \
        f"seq_len={seq_len} too short for {num_kv_pairs} pairs + {num_queries} queries"
```

Replace with:
```python
    assert mqar_cell_fits(seq_len, num_kv_pairs, num_queries), \
        f"seq_len={seq_len} too short for {num_kv_pairs} pairs + {num_queries} queries"
```

- [ ] **Step 4: Re-export it**

Replace the whole of `koopman_lm/evaluation/mqar/__init__.py` with:

```python
from koopman_lm.evaluation.mqar.mqar import (
    make_mqar, eval_mqar, eval_mqar_grid, mqar_cell_fits,
)

__all__ = ["make_mqar", "eval_mqar", "eval_mqar_grid", "mqar_cell_fits"]
```

- [ ] **Step 5: Verify the test passes**

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_mqar_generator.py -q -p no:cacheprovider`

Expected: `4 passed`

- [ ] **Step 6: Verify collection now succeeds without the escape hatch**

This is the point of the task — the suite must collect cleanly with no special flag.

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3`

Expected: `28 failed, 213 passed, 28 skipped` — **no `error` line**.

- [ ] **Step 7: Commit**

```bash
git add koopman_lm/evaluation/mqar/mqar.py koopman_lm/evaluation/mqar/__init__.py
git commit -m "fix: restore mqar_cell_fits, lost in the PR #10 merge

Its absence was an import error at module scope, which aborted collection
of the whole suite rather than failing three tests. make_mqar's assert now
routes through it so the fit rule has one definition."
```

---

## Task 3: Restore the scale ladder and `mamba_headdim`

Twenty-five failures are a single cause: `CONFIG_REGISTRY` lost eight entries
and `configs/` lost the eight matching YAMLs. Tests spanning `test_config.py`,
`test_baseline_training_contract.py`, `test_koopman_mlp_utilization.py`,
`test_eval_harness.py`, `test_ska_ablation.py`, and
`test_table2_repro_contract.py` all die with `ValueError: Unknown model_size`.

The 1m config needs `mamba_headdim`: `mamba_ssm` packs z/x/B/C/dt into one
`in_proj`, and the channel-last `causal_conv1d` kernel requires that width to be
a multiple of 8. At small `d_model` the default `headdim=64` produces a width
the kernel rejects at runtime.

**Files:**
- Create: `configs/1m.yaml`, `configs/180m_gated.yaml`, `configs/180m_v2.yaml`, `configs/370m.yaml`, `configs/440m.yaml`, `configs/880m.yaml`, `configs/1p5b.yaml`, `configs/3b.yaml`
- Modify: `koopman_lm/config.py` (add `mamba_headdim` field; extend `CONFIG_REGISTRY`)
- Modify: `koopman_lm/modules/token_mixer/mamba.py:11-17`
- Test: `code-tests/test_config.py`, and the five other files listed above

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `KoopmanLMConfig.mamba_headdim: int | None = None`. `CONFIG_REGISTRY` gains eight keys: `1m`, `180m_gated`, `180m_v2`, `370m`, `440m`, `880m`, `1p5b`, `3b`.

- [ ] **Step 1: Run the failing tests**

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_config.py -q -p no:cacheprovider 2>&1 | tail -3`

Expected: 10 failed, 9 passed — the failures citing `Unknown model_size`.

- [ ] **Step 2: Restore the eight YAMLs verbatim**

These are pure data and were verified to use **zero** fields unknown to the
current dataclass, so they load as-is.

```bash
for c in 1m 180m_gated 180m_v2 370m 440m 880m 1p5b 3b; do
  git show origin/cody-phase2a-1m-prep:configs/$c.yaml > configs/$c.yaml
done
ls -1 configs/*.yaml
```

- [ ] **Step 3: Add the `mamba_headdim` field**

In `koopman_lm/config.py`, inside `KoopmanLMConfig`, immediately after the
`mamba_expand: int = 2` line:

```python
    # Mamba-2 head width. None keeps mamba_ssm's own default (64).
    # Must be set explicitly at small d_model: mamba_ssm packs z/x/B/C/dt into
    # one in_proj of width 2*d_inner + 2*ngroups*d_state + nheads, and the
    # channel-last causal_conv1d kernel requires that width to be a multiple
    # of 8. At d_model=64 the default headdim=64 gives nheads=2 and width 290,
    # which the kernel rejects at runtime; headdim=16 gives nheads=8 and 296.
    mamba_headdim: int | None = None
```

- [ ] **Step 4: Consume it in the Mamba block**

In `koopman_lm/modules/token_mixer/mamba.py`, replace the `Mamba2(...)`
construction inside `Mamba2Block.__init__` so `headdim` is passed **only** when
pinned — existing configs must keep `mamba_ssm`'s default behaviour exactly:

```python
        from mamba_ssm import Mamba2
        self.norm = make_norm(cfg.d_model, cfg.norm_type, cfg.norm_eps)
        # headdim is passed only when the config pins it, so existing configs
        # keep mamba_ssm's own default (64) and their parameter counts.
        extra = {}
        if getattr(cfg, "mamba_headdim", None) is not None:
            extra["headdim"] = cfg.mamba_headdim
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
            **extra,
        )
```

- [ ] **Step 5: Extend the registry**

In `koopman_lm/config.py`, replace the `CONFIG_REGISTRY` literal with:

```python
CONFIG_REGISTRY = {
    # Synthetic-transfer scale (paper Sec 4.1 / Table 2).
    "1m": "1m.yaml",
    # Canonical fused-prefix production configurations.
    "50m": "50m.yaml",
    "50m_prefix_scan": "50m_prefix_scan.yaml",
    "180m": "180m.yaml",
    "180m_prefix_scan": "180m_prefix_scan.yaml",
    "180m_gated": "180m_gated.yaml",
    "180m_v2": "180m_v2.yaml",
    # Scaling ladder.
    "370m": "370m.yaml",
    "440m": "440m.yaml",
    "880m": "880m.yaml",
    "1p5b": "1p5b.yaml",
    "3b": "3b.yaml",
}
```

The two `_prefix_scan` entries are deliberately still present here; Task 4
removes them together with their files.

- [ ] **Step 6: Verify every config loads**

Run:
```bash
PYTHONPATH=. $V/bin/python -c "
from koopman_lm.config import CONFIG_REGISTRY, build_config
for n in sorted(CONFIG_REGISTRY):
    c = build_config(n)
    print(f'{n:20s} d_model={c.d_model:5d} n_layers={c.n_layers:3d} params={c.param_count_estimate()/1e6:8.1f}M')
"
```

Expected: twelve lines, no exception.

- [ ] **Step 7: Verify the suite**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3`

Expected: a large drop in failures — roughly 25 tests turn green across
`test_config.py`, `test_baseline_training_contract.py`,
`test_koopman_mlp_utilization.py`, `test_eval_harness.py`,
`test_ska_ablation.py`, and `test_table2_repro_contract.py`. **No failure
mentioning `Unknown model_size` may remain:**

```bash
PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | grep -c "Unknown model_size"
```

Expected: `0`.

Record the names of whatever still fails. It should be
`test_config_hash_distinguishes_configs` (Task 4) and
`test_wiki_pairs_have_distinct_neg` (Task 5), plus possibly one further
`test_config.py` case that asserts on registry size, which Task 4 clears. Do
not fix any of them here.

- [ ] **Step 8: Commit**

```bash
git add configs/ koopman_lm/config.py koopman_lm/modules/token_mixer/mamba.py
git commit -m "fix: restore the 8 lost config YAMLs and mamba_headdim

configs/{1m,180m_gated,180m_v2,370m,440m,880m,1p5b,3b}.yaml and their
registry entries were lost in the PR #10 merge, which is the single cause
of 25 test failures. mamba_headdim is required at small d_model, where
causal_conv1d rejects an in_proj width that is not a multiple of 8; it is
passed to Mamba2 only when pinned, so existing configs are unaffected."
```

---

## Task 4: Delete the duplicate `_prefix_scan` configs

`configs/50m_prefix_scan.yaml` is byte-identical to `configs/50m.yaml`
(`md5 1badfc8d`), and `180m_prefix_scan.yaml` to `180m.yaml` (`md5 8c5f86aa`).
The registry advertises four configs where two exist. Because the `_prefix_scan`
variants set no distinguishing field, `scripts/pretrain.sh 50m_prefix_scan`
trains a plain 50m — the suffix promises a behaviour it does not deliver.

**Files:**
- Delete: `configs/50m_prefix_scan.yaml`, `configs/180m_prefix_scan.yaml`
- Modify: `koopman_lm/config.py` (drop two registry entries)
- Modify: `scripts/pretrain.sh:15-25,44-47` (accept the canonical names)
- Modify: `code-tests/test_production_configs.py:5,16`
- Test: `code-tests/test_config.py::test_config_hash_distinguishes_configs`

**Interfaces:**
- Consumes: `CONFIG_REGISTRY` as extended by Task 3.
- Produces: `CONFIG_REGISTRY` with exactly ten keys, each mapping to a distinct file and therefore a distinct `config_hash`.

- [ ] **Step 1: Confirm the duplication and run the failing test**

```bash
md5sum configs/50m.yaml configs/50m_prefix_scan.yaml configs/180m.yaml configs/180m_prefix_scan.yaml
PYTHONPATH=. $V/bin/pytest "code-tests/test_config.py::test_config_hash_distinguishes_configs" -q -p no:cacheprovider
```

Expected: the md5s pair up, and the test FAILS on a count mismatch between
registry entries and distinct hashes.

- [ ] **Step 2: Delete the duplicates and their registry entries**

```bash
git rm configs/50m_prefix_scan.yaml configs/180m_prefix_scan.yaml
```

Then remove these two lines from `CONFIG_REGISTRY` in `koopman_lm/config.py`:

```python
    "50m_prefix_scan": "50m_prefix_scan.yaml",
    "180m_prefix_scan": "180m_prefix_scan.yaml",
```

- [ ] **Step 3: Update the two existing tests that name the deleted configs**

In `code-tests/test_production_configs.py`, change `build_config('50m_prefix_scan')` on line 5 to `build_config('50m')`, and `build_config('180m_prefix_scan')` on line 16 to `build_config('180m')`.

This is not weakening a test: the files were byte-identical, so the assertions
are unchanged in meaning.

- [ ] **Step 4: Update `pretrain.sh` to the canonical names**

`pretrain.sh` is deleted later, in the run-system plan — but it must not be left
referring to configs that no longer exist. In `scripts/pretrain.sh`, replace
every occurrence of `50m_prefix_scan` with `50m` and `180m_prefix_scan` with
`180m`:

```bash
sed -i 's/50m_prefix_scan/50m/g; s/180m_prefix_scan/180m/g' scripts/pretrain.sh
grep -n "50m\|180m" scripts/pretrain.sh
```

Confirm the `case` arms at lines 15-25 and the `DATA_TAG` block now read `50m)` and `180m)`.

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3
PYTHONPATH=. $V/bin/python scripts/check_imports.py
```

Expected: `test_config_hash_distinguishes_configs` now PASSES, and the only
remaining failure is `test_wiki_pairs_have_distinct_neg`. `check_imports.py`
reports no unresolved imports.

- [ ] **Step 6: Commit**

```bash
git add -A configs/ koopman_lm/config.py code-tests/test_production_configs.py scripts/pretrain.sh
git commit -m "fix: delete byte-identical _prefix_scan config duplicates

50m_prefix_scan.yaml == 50m.yaml and 180m_prefix_scan.yaml == 180m.yaml.
The registry advertised four configs where two existed, and the suffix
promised a backend the files never set, so pretrain.sh 50m_prefix_scan
trained a plain 50m.

Caught by test_config_hash_distinguishes_configs."
```

---

## Task 5: Fix the wiki fixture off-by-one

`wiki_to_pairs` drops sections shorter than the stub filter
(`retrieval/data.py:119`, `len(p) > 60`) and returns `[]` when fewer than two
survive. The fixture's three paragraphs are **60, 54, and 52** characters, so
all three are dropped and `assert pairs` fails. The assertion is correct; the
fixture data is wrong. This test was already failing on
`origin/cody-phase2a-1m-prep` — `retrieval/data.py` is byte-identical there — so
it is an inherited bug, not a porting error.

**Files:**
- Modify: `code-tests/test_retrieval_data.py:44-47`
- Test: `code-tests/test_retrieval_data.py::test_wiki_pairs_have_distinct_neg`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Run the failing test and confirm the lengths**

```bash
PYTHONPATH=. $V/bin/pytest "code-tests/test_retrieval_data.py::test_wiki_pairs_have_distinct_neg" -q -p no:cacheprovider
PYTHONPATH=. $V/bin/python -c "
for s in ['Intro paragraph long enough to survive the stub filter here.',
          'History paragraph also long enough to pass the filter.',
          'Process paragraph likewise long enough to be usable.']:
    print(len(s), 'survives' if len(s) > 60 else 'DROPPED')
"
```

Expected: test FAILS on `assert pairs`; the three lengths print 60/54/52, all DROPPED.

- [ ] **Step 2: Lengthen the fixture paragraphs past the filter**

In `code-tests/test_retrieval_data.py`, replace lines 44-47:

```python
    ex = {"title": "Photosynthesis",
          "text": ("Intro paragraph that is comfortably long enough to survive the stub filter here.\n\n"
                   "== History ==\n\nHistory paragraph that is also comfortably long enough to pass the filter.\n\n"
                   "== Process ==\n\nProcess paragraph that is likewise comfortably long enough to be usable.")}
```

- [ ] **Step 3: Confirm all three now clear the filter**

Run:
```bash
PYTHONPATH=. $V/bin/python -c "
for s in ['Intro paragraph that is comfortably long enough to survive the stub filter here.',
          'History paragraph that is also comfortably long enough to pass the filter.',
          'Process paragraph that is likewise comfortably long enough to be usable.']:
    assert len(s) > 60, (len(s), s)
    print(len(s), 'survives')
"
```

Expected: three lines, all > 60, no assertion error.

- [ ] **Step 4: Verify the suite is green**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3`

Expected: **zero failures, zero errors.** Record the exact passed/skipped
counts — Task 6 must reproduce them identically.

- [ ] **Step 5: Commit**

```bash
git add code-tests/test_retrieval_data.py
git commit -m "test: lengthen wiki fixture paragraphs past the stub filter

wiki_to_pairs drops sections of 60 chars or fewer (data.py:119) and
returns [] below two survivors. The fixture's paragraphs were 60/54/52,
so all three were dropped. The assertion was right; the data was wrong.
Inherited from cody-phase2a-1m-prep, where data.py is byte-identical."
```

---

## Task 6: Land the final module layout

`docs-and-structure` renames `modules/token_mixer` → `modules/seq`,
`modules/channel_mixer` → `modules/mlp`, and promotes `modules/kernels` →
`koopman_lm/kernels`. It also extracts `kernels/lin_alg.py` and moves
`SKABlock`/`MambaSKAParallelBlock` into `modules/seq/ska_block.py`.

**`git merge-tree` reports zero conflicts, and that is misleading.** The two
branches touch disjoint files — one adds tests, the other renames source — so
there is no textual conflict, but all 23 recovered tests import
`koopman_lm.modules.kernels.*` and similar, which the merge relocates. A clean
merge produces a suite that cannot import. The import rewrite in Step 3 is the
substance of this task.

**Files:**
- Merge: branch `docs-and-structure` (48 files, +618/−432)
- Modify: every `code-tests/*.py` importing a moved module
- Test: the whole suite, plus `scripts/check_imports.py`

**Interfaces:**
- Consumes: the green suite from Task 5.
- Produces: final module paths — `koopman_lm.modules.seq.*`, `koopman_lm.modules.mlp.*`, `koopman_lm.modules.norm`, `koopman_lm.kernels.*`, `koopman_lm.modules.seq.ska_block.{SKABlock,MambaSKAParallelBlock}`. **This layout is frozen.**

- [ ] **Step 1: Record the pre-merge green state**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -2`

Expected: zero failures. Write the exact passed/skipped counts down — Step 5
must match them exactly.

- [ ] **Step 2: Merge**

```bash
git merge --no-edit docs-and-structure
git status --short | head
```

Expected: merge succeeds with no conflicts.

- [ ] **Step 3: Rewrite the test imports**

Derive the map from git rather than hand-writing it, exactly as the original
port did:

```bash
PYTHONPATH=. $V/bin/python - <<'PY'
import re, subprocess, pathlib
out = subprocess.run(
    ["git", "diff", "--name-status", "-M", "reorg-module-layout", "HEAD"],
    capture_output=True, text=True).stdout
rmap = {}
for line in out.splitlines():
    p = line.split("\t")
    if p[0].startswith("R") and len(p) == 3 and p[1].endswith(".py"):
        rmap[p[1][:-3].replace("/", ".")] = p[2][:-3].replace("/", ".")
changed = []
for f in pathlib.Path("code-tests").glob("*.py"):
    src = new = f.read_text(encoding="utf-8-sig")
    for old in sorted(rmap, key=len, reverse=True):
        new = re.sub(rf"\b{re.escape(old)}\b", rmap[old], new)
    if new != src:
        f.write_text(new, encoding="utf-8")
        changed.append(f.name)
print(f"{len(rmap)} renames; rewrote {len(changed)} test files")
for c in sorted(changed): print("  ", c)
PY
```

- [ ] **Step 4: Resolve `SKABlock` and `MambaSKAParallelBlock` by hand**

These moved *between* files rather than being renamed, so the map above does not
cover them. Find and fix:

```bash
grep -rn "SKABlock\|MambaSKAParallelBlock" code-tests/ | grep import
```

Every such import must read `from koopman_lm.modules.seq.ska_block import ...`.
Apply that to each hit.

- [ ] **Step 5: Verify the suite still matches Step 1 exactly**

```bash
PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3
PYTHONPATH=. $V/bin/python scripts/check_imports.py
```

Expected: **identical passed/skipped counts to Step 1**, zero failures, and
`check_imports.py` reporting no unresolved imports.

A pure rename must not change test outcomes. Any difference means the merge
altered behaviour, not just paths. If the numbers differ, **stop and report**
rather than adjusting tests.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: adopt the final seq/mlp/kernels module layout

Merges docs-and-structure and re-points every test import. git reported no
conflict because the branches touch disjoint files, but the recovered tests
import modules.kernels.* and modules.token_mixer.*, which the merge moves --
a clean merge would have produced a suite that cannot import.

Suite counts unchanged, as a pure rename requires.
This layout is frozen."
```

---

## Task 7: Fix the two references the rename invalidates

Two references pointed at pre-reorg paths and are now doubly stale. The
`pyproject.toml` one is a genuine packaging break: the CUDA sources are silently
excluded from any built wheel or sdist, and editable installs mask it
completely.

**Files:**
- Modify: `pyproject.toml:81`
- Modify: `docs/superpowers/specs/2026-08-07-run-system-design.md`
- Test: a packaging assertion added to `code-tests/test_production_configs.py`

**Interfaces:**
- Consumes: the frozen layout from Task 6.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

The `.cu` exclusion has no test, which is why it survived a reorg. Per
`CONTRIBUTING.draft.md` rule 6, a finding becomes a test. Append to
`code-tests/test_production_configs.py`:

```python
def test_package_data_points_at_the_real_cuda_sources():
    """The csrc/*.cu glob must name a package that exists.

    A stale key here excludes the CUDA sources from any built wheel or sdist.
    Editable installs mask it completely, so only this assertion catches it.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    pkg_data = cfg["tool"]["setuptools"]["package-data"]
    assert pkg_data, "package-data is empty; the .cu sources would not ship"

    for pkg, globs in pkg_data.items():
        pkg_dir = root / Path(*pkg.split("."))
        assert pkg_dir.is_dir(), f"package-data names {pkg!r}, which is not a directory"
        for g in globs:
            assert list(pkg_dir.glob(g)), f"{pkg!r} glob {g!r} matches no files"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `PYTHONPATH=. $V/bin/pytest "code-tests/test_production_configs.py::test_package_data_points_at_the_real_cuda_sources" -q -p no:cacheprovider`

Expected: FAIL — `package-data names 'koopman_lm.globals.modules.ska', which is not a directory`

- [ ] **Step 3: Fix `pyproject.toml`**

Replace the `[tool.setuptools.package-data]` block at the end of the file:

```toml
[tool.setuptools.package-data]
"koopman_lm.kernels" = ["csrc/*.cu"]
```

- [ ] **Step 4: Verify**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider 2>&1 | tail -3`

Expected: zero failures; passed count is Task 5's total plus one.

- [ ] **Step 5: Update the spec's path citations**

Task 6 froze the layout, so the spec's `file:line` references must match. In
`docs/superpowers/specs/2026-08-07-run-system-design.md`, update every citation
of a moved module:

```bash
sed -i \
  -e 's#modules/token_mixer/#modules/seq/#g' \
  -e 's#modules/channel_mixer/mlp\.py#modules/mlp/swiglu.py#g' \
  -e 's#modules/channel_mixer/#modules/mlp/#g' \
  -e 's#modules/kernels/#kernels/#g' \
  docs/superpowers/specs/2026-08-07-run-system-design.md
grep -n "token_mixer\|channel_mixer\|modules/kernels" docs/superpowers/specs/2026-08-07-run-system-design.md
```

Expected: the final `grep` prints nothing.

- [ ] **Step 6: Re-verify the spec's line numbers**

Task 6 moved `SKABlock` out of `models/koopman_lm.py`, so line-numbered
citations into that file have shifted. Check each remaining one resolves:

```bash
grep -oE '`[a-z_0-9/]+\.py:[0-9]+' docs/superpowers/specs/2026-08-07-run-system-design.md \
  | tr -d '`' | sort -u | while IFS=: read -r f l; do
    p=$(find ./koopman_lm ./scripts -name "$(basename $f)" 2>/dev/null | head -1)
    printf '%-46s %s\n' "$f:$l" "$( [ -n "$p" ] && sed -n "${l}p" "$p" | cut -c1-50 || echo '*** MISSING ***')"
  done
```

Correct any citation whose line no longer shows the code it claims.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml code-tests/test_production_configs.py docs/superpowers/specs/
git commit -m "fix: package-data pointed at a package deleted by the reorg

'koopman_lm.globals.modules.ska' has not existed since the module reorg, so
csrc/*.cu was silently excluded from every wheel and sdist. Editable
installs mask it, which is why no one noticed. Now asserted by a test, per
CONTRIBUTING rule 6.

Also re-points the run-system spec at the frozen seq/mlp/kernels layout."
```

---

## Done Criteria

- [ ] `PYTHONPATH=. $V/bin/pytest code-tests -q` reports **0 failed, 0 errors, 28 skipped** — with no `--continue-on-collection-errors`. (Passed count should land near 242; the exact figure is whatever 0-failure state the suite reaches, not a target to engineer toward.)
- [ ] `PYTHONPATH=. $V/bin/python scripts/check_imports.py` reports no unresolved imports.
- [ ] `grep -rn "token_mixer\|channel_mixer\|globals" koopman_lm/ code-tests/ scripts/ pyproject.toml` returns nothing.
- [ ] All twelve — then ten, after Task 4 — registry configs load.
- [ ] The 28 skips are all `gpu`/`jax`-marked. No test was deleted, `xfail`-ed, or weakened.

## Follow-ups Not In This Plan

- `scripts/slurm_tests.sh` has never been submitted. Running the 28 skipped GPU tests on a real node is the honest gate and belongs in the run-system plan.
- Plans 2–4 (run layer, resume + trainer unification, consolidation) follow from the spec's §7 migration.
