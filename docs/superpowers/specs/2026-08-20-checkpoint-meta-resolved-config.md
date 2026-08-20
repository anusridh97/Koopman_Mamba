# Store a resolved config in `meta.pt`, not a pickled object

**Status:** proposed, not implemented. Deliberately out of the
`jack/search-and-provenance` PR — it changes the checkpoint format.

**Origin:** a manual review of that branch. The reviewer asked why an old
checkpoint doesn't simply *fail* to load against a newer `KoopmanLMConfig`, and
the answer turned out to be that the codebase already has the right pattern in
one place and not the other.

## 1. The asymmetry

Every run directory stores the same config twice, and only one copy is guarded.

| | artifact | form | resolved? | strict on read? |
|---|---|---|:--|:--|
| run system | `spec.yaml` | YAML | yes — 59/59 fields | **yes**, `_check_model_key_set` |
| checkpoint | `meta.pt` | pickled `KoopmanLMConfig` | n/a | **no** |

`resolve.py:157` `_check_model_key_set` compares a materialized spec's model key
set against `dataclasses.fields(KoopmanLMConfig)` and raises on any `missing` or
`unknown`, naming both lists. Its docstring already draws the distinction this
spec depends on:

> Scoped to *materialized* specs only — authoring specs with `extends:`
> legitimately carry partial key sets.

That is the whole idea, and it predates this branch. Two config contracts:

- **authoring** (`configs/runs/50m-first-real.yaml`, 40 keys) — partial by
  design; defaults and `extends:` fill the rest.
- **resolved** (`spec.yaml`, 59 keys) — every field explicit. Nothing is
  defaulted at read time, so a key set that doesn't match exactly is an error,
  not something to paper over.

It is the lockfile pattern. `package.json` carries ranges; `package-lock.json`
pins exact versions and a mismatch is a failure, never a fallback.

`meta.pt` is on the wrong side of that line. It stores `"cfg": cfg` — a live
dataclass instance (`train.py:515`, `adapt.py:256`, `table2.py:177`,
`mqar_finetune.py:132`). It is the only custom class in any checkpoint file.

## 2. Why a pickled object cannot be guarded

Pickle stores **values plus the class's import path**, never the class
definition, never a field list, never a version. On load it imports that path and
uses whatever class is there now, via `cls.__new__(cls)` followed by
`instance.__dict__.update(state)`.

`__init__` never runs. Therefore `__post_init__` never runs, so the precision
validation at `config.py:223+` never fires on a loaded checkpoint, and neither
does the tuple coercion. (Both are harmless today only because a pickle is made
*from* an already-constructed instance, so its values were validated when the
object was first built.)

The consequence for schema drift:

- **field added, with a default** — silent. `asdict()` reads the field list off
  the *class* and `getattr` falls through to the class attribute, so the new
  field materializes out of nowhere with its default.
- **field added, no default** — `AttributeError` from inside `config_hash`,
  which is a confusing place to land.
- **field removed** — silent. The stale key sits in `__dict__` and `asdict()`
  simply stops looking at it.

Measured on the real artifacts: a checkpoint written 2026-08-07 stores **56**
fields and records `cfg_hash 8807a902`. Read by this branch's 59-field class it
hashes to `c267087f` — and `asdict()` of it is *byte-identical* to a freshly
built current 50m config, because the three added fields' defaults describe
exactly what the old code did implicitly. So the identity moved while the model
did not.

## 3. The change

Store resolved data and validate it on read, matching what the run system does.

**Write** — in all four `checkpoint_meta`-style writers:

```python
"cfg": dataclasses.asdict(cfg),      # was: cfg
```

**Read** — wherever `meta["cfg"]` is consumed (`harness.py:load_meta` consumers,
`evaluate.py:load_model`):

```python
raw = meta["cfg"]
if isinstance(raw, dict):                    # new format
    _check_model_key_set(raw)                # reuse resolve.py's guard verbatim
    cfg = KoopmanLMConfig(**raw)             # __post_init__ runs -> validation fires
else:                                        # legacy pickled instance
    cfg = raw
```

The `isinstance` branch is what keeps every existing checkpoint loadable, and it
is the only new concept in the change. Everything else is reuse.

## 4. What this buys

1. **Schema drift becomes loud**, with the missing/unknown field names — the same
   error `spec.yaml` already produces. This is what the reviewer's question was
   actually asking for, and it is the reason `cfg_hash_mismatch` was removed
   rather than kept: a boolean flag noticed later is strictly worse than a
   refusal to load with both lists named.
2. **Validation actually runs.** `__post_init__` fires, so the precision-policy
   checks apply to loaded checkpoints instead of only to freshly built configs.
3. **`weights_only=True` becomes possible for `meta.pt`.** Today `model.pt` — 200
   MB of weights — loads safely, while the 3 KB metadata file requires executing
   an arbitrary pickle program. That inversion goes away.
4. **`meta.pt` becomes readable** by anything that can read a dict, without
   importing `koopman_lm`.

## 5. Costs and non-goals

- **Checkpoint format change.** Old checkpoints keep working through the
  `isinstance` branch, but new ones are not readable by older code. Worth a
  `meta_version` key.
- **Does not fix defaulted-field drift on its own.** `KoopmanLMConfig(**d)` with
  a 56-key dict succeeds silently — verified, not assumed. The strictness comes
  entirely from `_check_model_key_set`, so the guard is the load-bearing half of
  this change and shipping `asdict` without it accomplishes very little.
- **Does not re-detect drift in already-written checkpoints.** Nothing can; the
  old files record no schema. `cfg_hash` remains the only artifact in them
  computed by the writing code, which is why it must stay verbatim.
- **Not `frozen=True`-preserving by itself.** `KoopmanLMConfig(**raw)` rebuilds a
  new frozen instance; callers holding the old object by identity (there should
  be none) would notice.

## 6. Testing

1. `checkpoint_meta` stores a `dict`, and `asdict`→`KoopmanLMConfig(**d)`
   round-trips to an **equal** config (the `coerce` metadata already makes this
   true for `spec.yaml`; pin it here too).
2. A meta dict missing a field raises, naming that field. A meta dict with an
   unknown field raises, naming it.
3. A legacy `meta.pt` holding a pickled instance still loads, and yields the same
   cfg it always did.
4. `torch.load(meta.pt, weights_only=True)` succeeds for a newly written
   checkpoint.
5. `__post_init__` validation fires on load: a meta dict with
   `compute_precision: 'fp8'` raises the config's own `ValueError`, not an
   obscure failure downstream.
