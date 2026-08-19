# GPU verification of `jack/search-and-provenance`

Every commit on this branch was written and verified on a CPU login node, where
28–37 tests skip because they are GPU-marked and a real `KoopmanLM` cannot be
built without `mamba_ssm`. The branch touches the live SKA forward path, the
trainer's autocast, the config schema and the run system — so "843 tests pass"
was never sufficient evidence. This records what real hardware said.

Reproduce with `sbatch scripts/verify_search_and_provenance.sbatch`.

## Result

Job **436063**, H100 80GB (driver 580.173.02, sm_90), 8 minutes, at commit
`3be37b7`.

| step | status | what it proves |
|---|:--|---|
| 1 · GPU-marked tests | 2 failed / 18 passed | both failures pre-existing (below) |
| 2 · full suite | 3 failed / **871 passed** | same three, no new ones |
| 3 · real 30-step training | **pass** | the branch trains |
| 3b · checkpoint provenance | **pass** | `meta.pt` carries `code_id` + `dirty` |
| 4 · `quick_eval` | **pass** | the searcher's metric works on a real model |

### Step 3 — it trains

```
Training: 30 steps, eff_batch=64, tok/step=131,072
step 10/30 | loss 10.0827 | ppl 23925.3 |  54.6K tok/s
step 20/30 | loss  9.2069 | ppl  9965.9 |  85.3K tok/s
step 30/30 | loss  8.2602 | ppl  3866.9 | 105.2K tok/s
```

Loss falls monotonically and throughput ramps as the caches warm. Checkpoints at
`step_10/20/30/final`. This is the first time anything on this branch trained.

### Step 3b — the checkpoint attributes itself

```
meta.pt keys: cfg, cfg_hash, code_id, dirty, model_size, model_type, step, torch_version
  code_id = '3be37b7fb111ceb6094a12d0f343a85de8408be0'
  dirty   = False
  cfg.compute_precision = 'bf16'   ska_precision = 'fp32'   mlp_precision = None
```

`code_id` is exactly the commit that produced it. This is the provenance gap
(design §3.2) closed and then demonstrated, rather than asserted by a unit test.

### Step 4 — the searcher's metric, on a real model

```
Parameters: 50,034,044
full: loss 7.7238  ppl 2261.55  n_tokens 32768  peak_memory 1.69 GiB
ska_ablation: supported=True  loss_delta = 1.45e-05
```

Two things worth reading carefully.

`50,034,044` is exactly the parameter count `README` documents for 50m — so the
three new config fields changed no parameter, independently confirmed on the
hardware that builds the model.

**The SKA delta of 1.45e-05 is expected here, and would be alarming elsewhere.**
Zeroing the SKA branch changes the loss by almost nothing because after 30 steps
with `ska_layerscale_init=0.01` the branch has barely learned anything yet. At 30
steps that means "too short to say", not "SKA is useless". A real study reading a
delta this small at 600+ steps would mean the opposite, which is exactly why
`quick_eval` reports it.

## The bug this caught

Job **435898**, the first attempt, died at training step 2:

```
File "experimentation/training/train.py", line 403, in train
    with autocast_ctx:
AttributeError: args
```

`precision.autocast` was a `@contextlib.contextmanager` generator, which is
single-use — `_GeneratorContextManager.__enter__` deletes `self.args`. But all
five trainers build the context **once** before the loop and enter it **per
step**, which `torch.amp.autocast` supports. Fixed in `3be37b7` by returning
`torch.autocast` / `nullcontext`, both reusable by construction.

843 CPU tests passed over this. Every one entered the context exactly once — the
bf16 branch, the fp32 branch, the disabled branch, all single-entry. A context
manager's contract includes how many times it may be entered and nothing tested
that. Three tests now enter each branch three or four times.

This is the class of defect a CPU suite cannot find by accident: it needs a real
training loop, or a test that deliberately reuses the object. Step 3 was the first
thing to ever execute a second training step.

## The three pre-existing failures

None are this branch's, and each was checked rather than assumed.

| test | evidence |
|---|---|
| `test_full_model_decode_prefill_parity` | named in `scripts/gpu_triage_verify.sbatch`'s own header as known collateral from the nvcc build bug |
| `test_e2e_train_checkpoint_reload_decode` | same header |
| `test_importing_koopman_lm_does_not_reach_the_cuda_extras` | ran its probe against untouched `main` (`a4850de`) in the same venv: `import koopman_lm` pulls in `triton` there too, identically |

The last one is worth a note: it is **latent on CPU**, because `triton` is not
installed there, so `import koopman_lm` cannot pull it in and the assertion passes
vacuously. It only has teeth in a CUDA environment, where it has presumably been
failing unnoticed. Fixing it is out of scope here but it is a real finding.

The first two also mean `--resume` and full-prefix decode parity remain unverified
on this branch — they were already broken before it.

## What is still not verified

- **Multi-GPU / DDP.** This ran one H100.
- **The fused `cuda_prefix` kernel specifically.** `50m-first-real.yaml` sets
  `ska_backend: cuda_prefix`, and 30 steps at 105K tok/s is consistent with it
  running, but nothing here asserts which backend was selected.
- **`ska_precision='fp64'`.** Reachable only on the PyTorch path and never
  exercised on GPU.
- **Optuna's adaptive loop against real training.** Steps 3–4 prove the pieces a
  trial needs; no study has actually driven them.
