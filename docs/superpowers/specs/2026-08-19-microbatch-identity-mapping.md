# The microbatch migration: the identity break, old -> new

Recorded 2026-08-19. `per_device_batch_size` moved from `OptimSpec` to
`RuntimeSpec`, so `group_id` and `run_id` move for every spec that declares
one. Regenerated with `scripts/gen_identity_baseline.py`, never by hand.

## Why

`OptimSpec` is hashed whole into `run_id`; `RuntimeSpec` is documented as
"NOT hashed ... a different partition/account/worker count/ddp setting is the
same experiment run differently". A microbatch is exactly that: halving it
while doubling gradient accumulation preserves `effective_batch` exactly and
changes no result. While it sat in `OptimSpec`, descending a rung of the OOM
ladder renamed the experiment:

    effective_batch=64, pdbs=16  ->  run_id 58674511
    effective_batch=64, pdbs=8   ->  run_id 02705fcd

`effective_batch` stays in `OptimSpec`, because that one does change results.

## What did NOT move, and why that is the check that this was scoped right

`config_hash` is unchanged for all 11 registry configs
(verified below). The field lives on a run spec, not on the model config, so
moving it between run-spec sections cannot touch the model hash. `sweep_id` is
likewise unchanged: it hashes the sweep declaration, which did not change.

## Run specs

| spec | field | old | new |
|:--|:--|:--|:--|
| `configs/runs/50m-fineweb-3b.yaml` | config_hash | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` |
| `configs/runs/50m-fineweb-3b.yaml` | group_id | `eaf32d51` | `d71e3f5c` |
| `configs/runs/50m-fineweb-3b.yaml` | run_id | `a9b06e63` | `e9b2dec7` |
| `configs/runs/50m-first-real.yaml` | config_hash | `c267087fb1381954f00577fe3e21705a333c153f967ee94d8d2addc042bbe3e1` | `c267087fb1381954f00577fe3e21705a333c153f967ee94d8d2addc042bbe3e1` |
| `configs/runs/50m-first-real.yaml` | group_id | `f9815bf2` | `92c7bb96` |
| `configs/runs/50m-first-real.yaml` | run_id | `e3c1ff03` | `27f0de5d` |
| `configs/runs/50m-mqar-smoke.yaml` | config_hash | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` |
| `configs/runs/50m-mqar-smoke.yaml` | group_id | `77d93ee6` | `0def5819` |
| `configs/runs/50m-mqar-smoke.yaml` | run_id | `3ebf0ae1` | `dd0aa1df` |

## Sweeps

| sweep | field | old | new |
|:--|:--|:--|:--|
| `configs/sweeps/ska-rank-lr.yaml` | sweep_id | `eb729e29` | `eb729e29` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 0 run_id | `f1f75a15` | `6a10f5a9` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 1 run_id | `21babe66` | `d2c23ff3` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 2 run_id | `6e9f4bcc` | `2c899f48` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 3 run_id | `35c43817` | `599a99e4` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 4 run_id | `a9b06e63` | `e9b2dec7` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 5 run_id | `e9ded32a` | `b59b3a30` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 6 run_id | `4f4f4baf` | `7ac8121f` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 7 run_id | `ba5dca39` | `7b828de8` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 8 run_id | `c040b227` | `a25e9ba4` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 9 run_id | `f471be6b` | `723f42fc` |

## Compatibility

Every authored `configs/runs/*.yaml` and every already-materialized
`spec.yaml` puts the field under `optim:`. A bare move would have made
finished run directories unreadable -- and a run directory's `spec.yaml` is
the only record of what that run did. So all three loaders
(`resolve_run_spec`, `load_materialized_spec`, `load_raw_spec`) migrate the
key forward and emit a `DeprecationWarning`; declaring it in both sections
with different values is an error rather than a guess. The tracked configs
were updated in the same commit, so nothing in the repo triggers the warning.

## Verification

- config_hash moved for **0 of 11** registry configs (zero expected: this is not a model-config change).
- sweep_id unchanged: True.
- Every rung of `batch_plans(64, 16)` now resolves to one `run_id`, pinned by
  `test_the_oom_ladder_now_keeps_one_identity_across_rungs`.
