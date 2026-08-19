# Precision policy: the identity break, old -> new

Recorded 2026-08-19, per `test_identity_baseline.py`'s instruction that the
baseline is regenerated **only** when an intentional change to run identity
has been decided and the old->new mapping written down -- never to make a red
test green.

## What changed and why

`compute_precision`, `ska_precision` and `mlp_precision` were added to
`KoopmanLMConfig`. `config_hash` hashes the whole dataclass, so three new
fields move it for all 11 registry configs, and `group_id`/`run_id` move with
it because they hash `model` wholesale.

The defaults were chosen to *describe* existing behaviour -- bf16 autocast, an
fp32 SKA core, an untouched MLP path -- so **no numerics changed**. Only the
names did.

## The completed run that loses lineage

`50m-first-real` (job 415896, 100M tokens, held-out ppl 214.6) is the one real
completed run affected. Per the design doc's recommendation the break is
accepted and noted rather than patched: it is a smoke test, and
`experimentation.results` still finds it -- it globs `spec.yaml` and reads
contents -- but groups it separately from post-change runs.

## config_hash

| config | old | new |
|:--|:--|:--|
| `180m` | `079d2b71aacbe44352685ee27519c5015eb58914d52ab308f948ebd6ea91cb90` | `9db99c86788927fdf5d542a0ff84df044f1926cc5024f5b5f3bc2af387ee9328` |
| `180m_dense` | `89ac30ac475661c75ebd7d5047b21742909339a39e0f7fed87331ddcedb0e569` | `3f89d0f3f83e3ad72abbf8dbc584d379ef36f6dc4c68b6342cc6f93e58e0479f` |
| `180m_gated` | `80cf112a288914dea055939f41ebacfc9c1655b7c745f98fc195e83dc1f96ae5` | `97698aabc4dbdd549144799064350d972e46ebe1afe147c3c0463c3238d7818a` |
| `180m_v2` | `0d6bb6db761f3fe4a9cbb403a33ec89982ece083e91aca919caf10db53438ff4` | `54a1eb52d81440020935f3e7f84a9b849c1c0a2d0fcebd7f9ec70e6ef0cdeeb4` |
| `1m` | `93ea1b686e65159aad261c1e171113a706a8b18f5193c96ca006d1da33276648` | `10f09f4bf96514cc992553fe032e2322388caba10245813f22d2dbbf457a254a` |
| `1p5b` | `c0ebd7df5ec78749488d9ed30b28693bccb02610d3cd460c98d7448f3d338121` | `5da75c03c0fb8e414342fefc0ca6d8b894458e28725157de3c1d45bd57f72208` |
| `370m` | `1b452adaa64c2f14295053102ea23594695531208bb75b7ed798e0958358705f` | `a214f6a75ad6298094779864fea22bf53fb27366461a9c7b6728dafbf5c15a80` |
| `3b` | `ee70676f505192643377bc304e8d16e02df035d6952f1d2abf9b5146e868200d` | `5a52870fbe6ddc1becf77e5360a0bf601564b77210ea39aae8394caaff8e8714` |
| `440m` | `1ced2721490de037ec6d811582eacda64940e7abe7967319734927234859a5fe` | `84fd93bb9bfef895f294796b982c86ef344a9ef8d19f54bff7e1b466eab54e51` |
| `50m` | `c639c3ea6c1233cbb7db44e1ca6a88f8726a344f5a85047d2dd41801f36a0f81` | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` |
| `880m` | `e8df1f543af61f671a854dcc92b0b5222d92a5280b2c49353d461b6d1d37a0a5` | `57da55bb9118a08c89d31e139a220ad38612ed63780bffc03f0785a7a0c9c92f` |

## Run specs

| spec | field | old | new |
|:--|:--|:--|:--|
| `configs/runs/50m-fineweb-3b.yaml` | config_hash | `c639c3ea6c1233cbb7db44e1ca6a88f8726a344f5a85047d2dd41801f36a0f81` | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` |
| `configs/runs/50m-fineweb-3b.yaml` | group_id | `6ed9335d` | `eaf32d51` |
| `configs/runs/50m-fineweb-3b.yaml` | run_id | `3fff7e40` | `a9b06e63` |
| `configs/runs/50m-first-real.yaml` | config_hash | `8807a902e9b765a28a08039ebe7372d073909dbf4000e13ae34c659f9b276302` | `c267087fb1381954f00577fe3e21705a333c153f967ee94d8d2addc042bbe3e1` |
| `configs/runs/50m-first-real.yaml` | group_id | `81033b58` | `f9815bf2` |
| `configs/runs/50m-first-real.yaml` | run_id | `5467934f` | `e3c1ff03` |
| `configs/runs/50m-mqar-smoke.yaml` | config_hash | `c639c3ea6c1233cbb7db44e1ca6a88f8726a344f5a85047d2dd41801f36a0f81` | `c5cec7fb54cb426da6a00991fe4ebef1dbf31529b037ed9ba2efebd134723c6f` |
| `configs/runs/50m-mqar-smoke.yaml` | group_id | `3f0c3941` | `77d93ee6` |
| `configs/runs/50m-mqar-smoke.yaml` | run_id | `cecbcff0` | `3ebf0ae1` |

## Sweeps

`sweep_id` hashes the sweep's own declaration (name, base, axes, exclude) and
not the resulting configs, so it is unchanged -- correctly: the sweep
declaration did not change. Its cells' identities did.

| sweep | field | old | new |
|:--|:--|:--|:--|
| `configs/sweeps/ska-rank-lr.yaml` | sweep_id | `eb729e29` | `eb729e29` |
| `configs/sweeps/ska-rank-lr.yaml` | n_cells | 10 | 10 |
| `configs/sweeps/ska-rank-lr.yaml` | cell 0 run_id | `11f12406` | `f1f75a15` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 1 run_id | `7440f48f` | `21babe66` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 2 run_id | `e5df2a2b` | `6e9f4bcc` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 3 run_id | `465786ef` | `35c43817` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 4 run_id | `3fff7e40` | `a9b06e63` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 5 run_id | `6340174d` | `e9ded32a` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 6 run_id | `f7aa33ea` | `4f4f4baf` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 7 run_id | `a82df1e3` | `ba5dca39` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 8 run_id | `6ab01bd7` | `c040b227` |
| `configs/sweeps/ska-rank-lr.yaml` | cell 9 run_id | `00c65923` | `f471be6b` |

## Verification

- config_hash moved for **11 of 11** registry configs (all of them, as expected -- the field set is shared).
- sweep_id unchanged: True.
- Regenerated with `python scripts/gen_identity_baseline.py`, never by hand.
