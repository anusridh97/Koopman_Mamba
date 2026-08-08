# Extension mechanisms: probes, builders, per-layer dimensions, schedules

Status: **designed, not implemented.** None of this exists in the repo. Worked out
in conversation on 2026-08-08; written down so a fresh session can implement it
without re-deriving the reasoning — including the parts we decided *against*.

Owner: jkli. Prerequisite reading: `2026-08-07-run-system-design.md` (§3 for
`RunSpec`, §3.3 for identity, §6 for trainer unification).

---

## 1. The problem this solves

`KoopmanLMConfig` has 56 fields. **Eleven** carry comments marking them as
back-compat, legacy, or "default reproduces v1 exactly." Ten `ValueError`s in
`__post_init__` reject incompatible combinations, two of them purely for
mutually-exclusive exact-path flags. That is what happens when every one-off
experiment becomes a permanent config field: the config accumulates knobs nobody
sets, `config_hash` covers them all, and every future reader must work out whether
each one matters.

`mlp_type: 'auto'` resolving through the older `mlp_gated` boolean is the scar
from this. Its own comment says: *"New quality runs should set this explicitly so
an architecture ablation cannot silently change when a legacy boolean is copied
between YAMLs."* Someone got burned and left a note.

The alternative — hardcode the experiment on a branch — fails differently. A
hardcoded change makes `spec.yaml` **lie by omission**: the run looks ordinary.
Worse, `run_id = sha256(model + data + optim + seed)` is unchanged by a code
edit, so a probe run and a clean run **collide on one `run_id`** unless something
else distinguishes them.

So neither "add a config field" nor "hardcode it on a branch" is right for a
one-off. This document defines the middle.

---

## 2. The taxonomy

Split by **what kind of thing varies**, not by how general the change is:

| What varies | Mechanism | Section |
|---|---|---|
| **Dimensions** per layer (rank, width, `d_state`) | config field accepting scalar-or-sequence | §3 |
| **Kind** of layer, or the stack's structure | builder (construction-time) | §4 |
| **A built model**, after construction | probe (post-construction mutation) | §5 |
| **The trajectory** — anything time-varying | schedule | §6 |
| **Per-parameter-group optimizer settings** | `optim.groups` | §7 |

The load-bearing distinction is that §3-§5 all describe **the model at step 0**,
while §6 describes **how it moves**. Those are different mathematical objects and
need different vocabulary. The original design had none for §6 at all.

---

## 3. Per-layer dimensions: scalar-or-sequence config fields

For "I want the SKA rank to be `[10,10,10,20,30,40,30,20,10]` across layers."

This is **not** a builder or a probe. It is a config-shape question, and it stays
declarative — so it lands in `spec.yaml`, is queryable in aggregation, and
`param_count_estimate` can account for it. A builder would hide a dimension sweep
inside code, which is the worst place for it.

```yaml
ska_rank: [10, 10, 10, 20, 30, 40, 30, 20, 10]   # per SKA layer
d_state:  128                                     # scalar broadcasts
```

```python
def _broadcast(v, n, name):
    if isinstance(v, (int, float)):
        return (v,) * n
    seq = tuple(v)
    if len(seq) != n:
        raise ValueError(f"{name} has {len(seq)} entries, expected {n}")
    return seq
```

Validated in `__post_init__`; read via an accessor like `cfg.rank_at(i)`.
Precedent already exists: `ska_layer_indices: Optional[Tuple[int, ...]]` is
exactly a per-layer structure.

**Two real costs.**

`param_count_estimate` currently does `per_ska * n_ska`, assuming uniform rank. It
becomes a sum over layers. Mechanical, but it must be updated or the
parameter-band tests will lie. (Note this estimator was found to be wrong in four
places on 2026-08-07 and is now bit-exact against a real instantiation — do not
regress it.)

**The fused CUDA kernel is hard-specialized to rank 24.** `README.md` says so
explicitly ("specialized for the geometry used by both configs: rank `r=24`;
value/head width `p=64`"), and `csrc/prefix_scan_ext.cu` bakes in a padded row
stride of 25 to avoid bank conflicts at that exact rank. So **any non-uniform or
non-24 rank forces `ska_backend: pytorch`.** A rank sweep is inherently on the
slow path unless someone generalizes the kernel. Know this before planning one.

---

## 4. Builders: construction-time variation

For "layer 3 uses a different mixer", "interleave a different block type",
"weight-tie two layers".

```python
@register_builder("l3-mlp-sin")
class _L3Sin(KoopmanLM):
    """Layer 3's MLP activation -> sin. Everything else identical to the base."""
    def build_layer(self, i, cfg):
        blk = super().build_layer(i, cfg)
        if i == 3:
            blk.mlp.act = torch.sin
        return blk
```

```yaml
builder: l3-mlp-sin        # or null for the default
```

**The enabling refactor** is extracting a layer-factory seam. `KoopmanLM.__init__`
currently builds its stack inline; it needs:

```python
def build_layer(self, i, cfg):
    if i in cfg.ska_layer_indices:
        return (MambaSKAParallelBlock(cfg) if cfg.ska_mode == 'parallel'
                else SKABlock(cfg))
    return Mamba2Block(cfg)
```

That seam is why a builder **does not have to own construction** — `super()` owns
it, and the builder owns only its delta. You override the smallest unit that
contains your change; at the extreme you can override the whole stack.

**This formalizes something that already exists informally.** `train.py`'s
`--model_type` dispatch picks among `build_mamba_attention`, `build_mamba_only`,
`build_mamba_ska_swiglu`, `build_mamba_ska_koopman` via a hardcoded `if/elif`
chain nobody can extend. Those five become registered builders, so production
baselines and experimental variants go through one mechanism.

**Convention to enforce, not merely document:** a builder overrides one or two
methods and **never inherits from another builder** — flat, one level off
`KoopmanLM`. Enforce with a test that walks the registry and asserts MRO depth.
Subclassing otherwise invites diamond problems.

**Implementation discipline:** the seam touches real model construction. Write a
characterization test proving the refactored `KoopmanLM` produces **bit-identical**
models for all registry configs *before* changing anything. `mamba_ssm` is
unavailable on CPU dev boxes, but `ska_mode="replace"` configs instantiate fine —
use those, and derive the Mamba term analytically where you cannot.

---

## 5. Probes: post-construction mutation

For "swap an activation", "freeze these parameters", "bolt an adapter onto a
pretrained checkpoint", "insert an extra projection", "install a hook".

```python
@register_probe("l3-extra-proj")
def apply(model, cfg):
    """Insert an extra d_model x d_model projection before layer 3's block.
    Identity-initialised, so the model starts numerically identical to the
    unprobed one and any difference is attributable to training, not init."""
    blk = model.seq_layers[3]
    blk.extra = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    nn.init.eye_(blk.extra.weight)
    orig = blk.forward
    blk.forward = lambda x: orig(blk.extra(x))
```

```yaml
probes: [l3-extra-proj, freeze-embed]     # ordered, composable
```

Probes **can add parameters** — `nn.Module.__setattr__` registers the submodule,
so it appears in `model.parameters()`. The identity-init convention above is worth
following: the model starts numerically equivalent to the unprobed one, so any
difference is attributable to training rather than initialization. (Same idea as
`CONTRIBUTING`'s "land it inert".)

The repo already contains one instance of this pattern: `patch_ska_module()` in
`modules/seq/fast.py` mutates a live `SKAModule`'s projections and state-dict
hooks.

### 5.1 Four constraints, and the second one will bite

**Ordering.** A probe must run **before the optimizer is constructed**, or new
parameters get no updates. And it must run in **every** path that builds a model —
train, eval, resume, decode. `evaluation/evaluate.py` has its own `load_model`; if
that path skips probes, **eval silently scores a different architecture than was
trained.**

**Checkpoint keys.** A probe that adds parameters changes the `state_dict`. That
checkpoint will not load without the probe (unexpected key), and a non-probe
checkpoint will not load with it (missing key). Solvable, because `spec.yaml`
records the probe list and every loader reads `spec.yaml` — but only if each
loader applies probes *before* `load_state_dict`. This is the same shape as the
`--resume` bug found on 2026-08-07: two correct units, wrong composition. **Test
the composed path, not the units.**

**`torch.compile`.** A Python lambda wrapping `forward` forces a graph break.
Mostly irrelevant here — `train.py` already leaves the fused SKA path eager — and
attribute swaps cost nothing at runtime. If a probe becomes hot, that is the signal
to promote it to a real module where it can be compiled.

**What probes cannot do:** change anything a third-party module fixes at
construction (`mamba_ssm.Mamba2`'s internal shapes), anything needing a different
class hierarchy, or anything that must be visible to `param_count_estimate` before
the model exists. Those want §3 or §4.

### 5.2 Never delete a probe file

A probe file is ~10 lines. Deleting one **orphans every checkpoint it produced**,
because that checkpoint's `state_dict` can no longer be reconstructed. Keep them
permanently as a record of what was tried; mark deprecated if you must, but the
file stays.

### 5.3 Why a name plus a docstring is the right amount of prose

The registry name is the concise, queryable label that lands in `spec.yaml` and
appears as a column in `koopman_lm.results`. The docstring carries the hypothesis.
That is the minimum viable prose, and it lives next to the code it describes rather
than in a document that drifts.

Probes accumulating in `experiments/probes/` become a legible log of what has been
tried. When one wins, promote it to a real config field — and you will design a
better field, because by then you know its shape (per-layer or global? two options
or five?) rather than guessing an interface for a result you do not have.

---

## 6. Schedules: anything time-varying

The gap the original design had no vocabulary for. Covers: ridge epsilon annealing,
freeze-then-unfreeze, sequence-length curriculum, data-mix curriculum, loss-weight
ramps, tightening gamma/eta bounds, stochastic-depth rate schedules.

```yaml
schedules:
  model.ska_ridge:
    kind: linear
    from: 1.0e-2
    to:   1.0e-3
    over: [0, 5000]

  data.seq_len:
    kind: piecewise
    at:     [0,   4000, 8000]
    values: [512, 1024, 2048]

  freeze:
    match: "seq_layers.*.ska.*"
    frozen_until: 2000
```

### 6.1 Schedules live on `RunSpec`, NOT in `KoopmanLMConfig`

`KoopmanLMConfig` is a **total** description: every field has a value, absence is
impossible, and the key-set check added on 2026-08-07 depends on exactly that
totality — it validates that a materialized spec's `model` keys match the dataclass
field-for-field. `schedules` is a **partial** map where absence is the norm and
means something specific (constant). Those cannot share a validation regime.

So `schedules` is a top-level sibling of `model`/`data`/`optim`/`runtime`. Same
materialized `spec.yaml`, its own section. Framing: **`model` describes the model at
step 0; `schedules` describes how it moves.**

Useful consequence: `KoopmanLMConfig` stays standalone. `build_config('50m')` still
yields a model with no schedule machinery anywhere near it, keeping eval, decode and
the recurrent path simple.

### 6.2 The load-bearing constraint: pure functions of global step

**A schedule must be a pure function of global step. No internal state.**

That single rule means resume needs *nothing* extra — the step is already in
`resume.pt`, so replaying a schedule is free and exact. A stateful schedule would
be one more thing to checkpoint and one more way for §5.5's bit-exact resume test
to break.

It does rule out anything adaptive ("reduce ridge when loss plateaus"). That is a
deliberate trade: exactness of resume over adaptivity. Revisit only with a plan for
checkpointing schedule state.

### 6.3 Implementation: a whitelist and a `setattr` loop

The codebase makes this nearly trivial. `SKAModule.__init__` stores
`self.ridge_eps = ridge_eps` and `self.power_K = power_K` as **plain Python
floats/ints**, and `forward` reads `self.ridge_eps` at call time, passing it down as
a function argument. So `setattr(module, "ridge_eps", v)` between steps just works —
no module surgery, no hot-path indirection, and it works on the fused CUDA path too
since ridge is a kernel argument rather than compiled in.

```python
SCHEDULABLE = {                       # spec target -> (module class, attribute)
    "model.ska_ridge":   (SKAModule, "ridge_eps"),
    "model.ska_power_K": (SKAModule, "power_K"),
}

class ScheduleApplier:
    def __init__(self, model, schedules):
        self.bindings = []
        for target, sched in schedules.items():
            cls, attr = SCHEDULABLE[target]          # unknown target -> startup error
            sites = [m for m in model.modules() if isinstance(m, cls)]
            if not sites:
                raise ValueError(f"schedule target {target!r} matched no modules")
            self.bindings.append((target, sched, sites, attr))

    def apply(self, step) -> dict[str, float]:
        vals = {}
        for target, sched, sites, attr in self.bindings:
            v = sched(step)
            for m in sites:
                setattr(m, attr, v)
            vals[target] = v
        return vals                                   # returned for logging
```

One line in the training loop, before the forward: `sched_values = applier.apply(step)`.

**Kinds:** `constant`, `linear`, `cosine`, `piecewise`, `step`. Anything exotic goes
in a registry keyed by name, exactly like probes — and promotes to a declarative
kind if it earns it.

### 6.4 Three decisions that matter more than the code

**A whitelist, not reflection.** `SCHEDULABLE` is explicit. The silent-failure mode
is scheduling something the forward never re-reads — you would see a beautiful
annealing curve in wandb and the model would have ignored it entirely. A whitelist
makes that a startup `KeyError`. For the same reason, **an empty match must raise**,
not warn.

**Log every scheduled value.** Without it you cannot distinguish "the schedule ran"
from "the schedule was silently a no-op."

**`schedules` must be hashed into `run_id`.** `ska_ridge` is part of the model
config, so with a schedule the materialized `spec.yaml`'s `model.ska_ridge` is only
the *initial* value — the schedule is the real story. Two runs with different
annealing must not share a `run_id`.

### 6.5 Two other handlers, because they are not `setattr`

**freeze** -> `p.requires_grad_(step >= until)` over `fnmatch`-ed names. **This
changes optimizer construction:** `_param_groups` currently does
`if not p.requires_grad: continue`, so a parameter frozen at step 0 would never
enter the optimizer and could never unfreeze. The optimizer must be built over
**all** parameters, with freezing expressed purely through `requires_grad`.

**`data.seq_len`** -> consumed by the epoch loop when it rebuilds `epoch_loader`,
not settable on a module. This fits: `train.py` already rebuilds the loader inside
the epoch loop. Anything finer would mean rebuilding mid-epoch and breaking the
resume index arithmetic.

### 6.6 What cannot be scheduled

**Dimensions.** Rank, width, `d_state` are fixed at construction. Schedules only
reach values the forward pass reads at runtime.

**`optim.lr`** — deliberately excluded. The existing `LambdaLR` owns the learning
rate; two mechanisms fighting over `param_group['lr']` is a bug generator. A custom
LR shape belongs in `OptimSpec.schedule`, not here.

**eta / gamma** — deferred. They are `nn.Parameter` when learnable and read through
resolver properties (`_resolve_eta`). Scheduling them means writing into `.data`,
which only makes sense when they are *not* learnable — a special case worth
deferring rather than half-supporting.

---

## 7. Per-group optimizer settings

```yaml
optim:
  lr: 4.0e-4
  weight_decay: 0.1
  groups:
    - match: "*.ska.*"
      lr_mult: 10.0
    - match: "embed.weight"
      lr_mult: 0.1
      weight_decay: 0.0
```

**`lr_mult`, not absolute `lr`.** `get_cosine_schedule_with_warmup` is a `LambdaLR`
that multiplies every group's `initial_lr` by the same factor, so a multiplier
composes cleanly with the shared warmup+cosine. Absolute per-group LRs would fight
the schedule.

`param_groups` (now in `koopman_lm/training/optim.py`) becomes: for each parameter,
compute `(should_decay, matched_override)` and bucket by that pair. The existing
no-decay policy stays the default when nothing matches, so **an empty `groups` list
leaves every current config bit-identical**.

**First-match-wins ordering**, and **a validation error if a pattern matches
nothing** — a silently-dead pattern is how you believe you ran an experiment you
did not.

Note this gap was identified months ago: Cody's `phase2a_capabilities.template.json`
lists `"per_group_learning_rates": false`.

`optim.groups` has the same partial-map shape as `schedules` — absence is normal —
so it belongs as an optional list inside `OptimSpec`, not as new required fields.

---

## 8. Decisions made *against*, and why

Recorded because they were argued through and re-deriving them would waste time.

**sha256 of a git diff, as probe provenance.** Rejected. A hash identifies; it does
not explain. Storing the *patch* would explain, but see below.

**Storing a patch for dirty-tree launches.** Superseded by simply **refusing to
launch from a dirty tree** (implemented 2026-08-07). If a run launches from a
commit, `git_commit` alone is complete — check it out and you have the exact tree.
Commit discipline is the reproducibility mechanism; hashing schemes are not. Short-
lived branches and frequent commits (`CONTRIBUTING.draft.md` rules 2 and 3) are what
make `git diff commitA commitB` readable enough to be useful.

**`schema_version` on materialized specs.** Rejected. The key-set diff already
carries the useful information ("missing: [...]", "unknown: [...]"), and a version
integer would only distinguish an intentional schema change from a corrupted file.
Not worth it yet.

**Folding `code_id` into `run_id`.** Rejected. It would make every unrelated commit
spawn a new run identity and destroy the property that three seeds of one config are
one experiment. Two orthogonal identifiers instead: `run_id` for declared science,
`code_id` for what actually executed.

**A generic `overrides: dict[str, Any]` escape hatch.** Rejected. Tempting, and it
defeats the point — `__post_init__` stops being able to reject nonsense and configs
can silently mean anything.

**`schedules` inside `KoopmanLMConfig`.** Rejected on the totality-vs-partiality
argument in §6.1.

---

## 9. Gaps deliberately not designed here

- **Auxiliary losses** (spectral regularizers, orthogonality penalties, distillation
  terms). Loss composition; nothing composes losses today. Additive later.
- **Eval task registry.** Adding an eval still means a hand-written module with its
  own CLI. `2026-08-07-run-system-design.md` §4 designed where results *land* but not
  how tasks are *registered*.
- **Multi-stage run chains** (pretrain -> continued-pretrain -> retrieval adapt).
  Expressible today as *n* `RunSpec`s with `init_from` pointing at the previous
  `run_id`, just not ergonomic.

---

## 10. Suggested implementation order

1. **§7 per-group optimizer.** Smallest, already-identified need, no new concepts.
2. **§6 schedules.** Not because it is needed immediately, but because it is the one
   gap that is *expensive to retrofit* — it lives inside the training loop, while
   every other mechanism sits outside it.
3. **§5 probes.** Self-contained; the risk is §5.1's ordering/loader integration, so
   test the composed path.
4. **§4 builders.** Last, because the `build_layer` seam touches real model
   construction and wants characterization tests first.
5. **§3 per-layer dimensions.** Independent of the rest; do it when a sweep needs it,
   and remember the rank-24 kernel constraint.
