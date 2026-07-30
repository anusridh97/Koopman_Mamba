# Phase 2 integration matrix

The preparation branch is based on `origin/pr/eval-consolidation`, the
role-based package closest to the Scaling Plan's Phase 0 target. Portable
experiment tooling lives here; scientific launch remains fail-closed until the
architecture and training capabilities below land.

## Required integration order

1. Start from `pr/eval-consolidation`.
2. Port/merge `phase1-finalize`, preserving the consolidated evaluation loader.
3. Port the paper-mode behavior from commit `bec8517` semantically into the
   role-based modules; do not cherry-pick the old-layout commit wholesale.
4. Port exact-resume semantics from `ati-180m-exact-resume`.
5. Implement the Phase 2 optimizer, objective, evaluator callback, and launch
   adapters against the finalized APIs.
6. Run all correctness and preflight gates, then a small pipeline pilot.

The role-based kernels already implement the exclusive boundary and two-sided
whitening formulas. They still need protected paper sequence-max/ungated mode,
recurrent frozen prefill normalization, mode-aware diagnostics, and the literal
Appendix-F paper-mode test.

## Capability status on this branch

| Capability | Status | Evidence/action |
|---|---|---|
| role-based package | READY | `modules/`, `models/`, `training/`, `evaluation/` |
| unified eval entry point | PARTIAL | present; dataset/evaluator revisions are not frozen |
| exclusive cross-chunk boundary | READY | role-based chunk statistics use exclusive prefixes |
| two-sided whitening | READY | role-based operator uses `L^-1 M L^-T` |
| paper sequence-max/no-gate mode | BLOCKED | semantically port `bec8517` and its literal reference test |
| Phase 1 diagnostics | PARTIAL | port `phase1-finalize`; make beta metrics mode-aware |
| beta init probability | BLOCKED | add config and initialize bias with `logit(p)` |
| QKNorm | BLOCKED | define its position relative to existing normalization, then implement |
| effective chunk sweep | PARTIAL | prep config selects strict causal chunking; final adapter must prove each chunk value reaches the active backend and changes execution |
| per-group learning rates | BLOCKED | current trainer has one AdamW group |
| Birdie objective mixture | BLOCKED | generators, mixer, normalization, and equal-token accounting absent |
| step-500 MQAR callback | BLOCKED | training loop has no evaluator/pruner callback |
| pinned Zoology MQAR oracle | BLOCKED | local generator is provisional; exact-fit 256x64 boundary is fixed, but generator parity still needs fixtures |
| exact resume | controller provenance READY; adapter BLOCKED | controller independently hashes and terminally binds the immutable step-500 bundle; port optimizer/scheduler/RNG/data-offset state, load that exact bundle, and prove resumed-vs-uninterrupted parity |
| full trial hash | READY in prep tooling | must include non-model inputs |
| actual parameter/FLOP accounting | BLOCKED | extensions are absent from the provisional estimator; final adapter must report built-model core/total counts, analytic and measured FLOPs, scale/drift outcomes, and actual GPU hours through the strict result contract |
| optimizer group audit | BLOCKED | final adapter must emit exhaustive/exclusive parameter inventories and recomputable assignment digests; central validation is ready |
| literal requested ridge | BLOCKED | current kernel adds a hidden fixed increment; instrument the operator so the requested ridge is the normal value and every recovery increase is explicitly counted |
| environment fingerprint | READY in prep tooling; lock BLOCKED | `koopman-phase2-runtime` captures and strictly verifies Python, OS kernel, package imports/extensions, PyTorch/CUDA/cuDNN, driver, GPU identity, and a CUDA smoke op; generate the approved lock in the final one-GPU worker allocation |
| finalized data hashes | READY in prep tooling; artifacts BLOCKED | preflight streams and verifies distinct, read-only regular bulk train/WikiText/MQAR plus tokenizer, shard-order, MQAR oracle/sample/vocabulary/token-map, and Birdie JSON auxiliary artifacts; immutable paths and checksums remain placeholders |
| distributed study storage | BLOCKED | approve PostgreSQL or validate Optuna JournalStorage on SCG |
| storage snapshot receipt | READY in prep tooling; snapshot BLOCKED | after workers stop, bind the backend-native snapshot to final analysis plus quiescent status/spec/backend/immutable-ledger identity; final reporting re-hashes ledger/status evidence and rechecks the terminal claim hash set, while live Optuna query and dump provenance remain operator-attested; record either target reached or an explicitly approved genuine budget-exhausted early stop, excluding tail target-guard rows from the fresh quota |
| W&B report publication | BLOCKED after offline analysis | every healthy metrics file must cross-link a unique W&B run; after snapshot, analysis emits the payload and a separate verification artifact must bind the exact analysis/study, report URL, and payload run-ID set before the no-network finalizer records publication and reviewed hypothesis decisions |

`BLOCKED` means do not launch a scientific trial. It does not prevent schema,
materialization, analysis, or synthetic orchestration tests.

## Correctness gates

All must be green on the finalized integration commit:

- literal Appendix-F paper-mode parity;
- exclusive boundary `M` reference test;
- two-sided whitening reference test;
- state-rebuild decode/prefill parity at most 1e-4 in every stats mode;
- chunk-boundary parallel/decode parity at most 1e-4 using prompts whose
  lengths are integer multiples of the active chunk size;
- within-chunk drift reported at offsets 1, floor(chunk-size/2), and
  chunk-size-1 (informational while exact intrachunk is disabled);
- rank 96 and rank 128 decode parity;
- causal no-future-leak test;
- Cholesky update at the Scaling Plan's requested relative tolerance, or a
  documented and approved revised tolerance;
- Newton-Schulz versus Cholesky BF16 error at most 1e-3;
- spectral stability on 500 FP32 and BF16 cases;
- beta `sigmoid(bias)` equals requested initialization probability;
- diagnostics work with gated and beta-free SKA;
- diagnostic wall-time overhead below 3%.
- literal requested ridge reaches the operator without a hidden increment and
  every Cholesky recovery increase is explicitly counted.

The existing tests must not be treated as equivalent where they use weaker
tolerances or smaller scopes.

## Training gates

- Each trainable parameter belongs to exactly one optimizer group.
- Groups are mutually exclusive, collectively exhaustive, and nonempty when
  their component is active.
- Every group's parameter names, count, effective LR, and weight decay are
  logged.
- Gamma/eta LR axes are inactive when gamma/eta are fixed.
- Birdie weights sum to one and all objective arms see equal token budgets.
- Resumed and uninterrupted training produce matching subsequent weights and
  losses.
- Two runs from the same trial manifest match through the first 1,000 steps.
- Extreme configurations, especially rank 128 plus ridge 1e-4, complete the
  pilot without OOM or nonfinite state.

## Data/evaluation gates

- Pretokenization is a separate dependency job, never performed by every array
  worker.
- Dataset and tokenizer names, revisions, document ranges, packing procedure,
  sampler revision, shuffle seed, shard order, epoch/repeat policy, worker
  partitioning, seeds, and checksums are frozen.
- Checkpoint-local tokenizer loading is mandatory.
- The MQAR generator is cross-checked against the intended Zoology/Arora
  protocol.
- All 20 MQAR cells and the frozen WikiText subset are identical across trials.
- Unified result JSON includes retrieval metrics, perplexity, diagnostics,
  timing, and four-mode SKA load-bearing results.

## Orchestration gates

- Step 500 is counted in optimizer updates.
- Numerical failures are `FAIL`, below-median healthy trials are `PRUNED`.
- Startup trials and full-budget controls quantify false early-pruning risk.
- Concurrent workers pass a storage/restart/duplicate-trial smoke test.
- The same shared ledger atomically caps mixed Optuna/fixed-run concurrency,
  excludes fixed claims from Optuna recovery, and preserves heartbeat-estimated
  GPU time across a hard worker kill.
- Recovery-only Optuna rows do not consume the 2,000 fresh-configuration quota,
  and directory analysis rejects summaries without matching terminal claims.
- W&B and Optuna identifiers cross-link.
- The final offline payload is published to W&B and its report URL is recorded.
- Atomic result writes prevent partial files being mistaken for completed runs.
- Shared SQLite and concurrent worker-side pretokenization are prohibited.

DeepSpeed is not necessary for the 1M pilot. One trial per GPU through a Slurm
array is the useful parallelism at this stage.
