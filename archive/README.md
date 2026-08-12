# archive/

Code kept for provenance, never imported by `koopman_lm/`.

Nothing here is on an import path. `code-tests/test_archive_is_inert.py` enforces
that: `koopman_lm/` may not import from `archive/`, so restoring a file here can
never change model or training behaviour. Tests may load an archived file by
*path* (that is how the JAX oracle works) -- what is forbidden is a package
import.

Why keep any of it: a deleted file whose only copy was a branch tip is one
`git push --delete` away from gone. Everything below was verified byte-identical
to its source at restore time.

**This is the single copy.** PR #11 (`ebcf17d`) independently restored four of
these trees at their original top-level paths, so the PR #20 merge briefly
carried both. The top-level duplicates were removed afterwards -- 15 blobs
verified byte-identical to their counterpart here, plus `echo-ska/echo_jax.py`,
whose archived version is the same file with a 27-line provenance header added
(diff: 27 insertions, **0 deletions**, one hunk inside the module docstring).
If you are looking for `MQAR/prefix_bench/`, `sys+toolcall/SKAv9.py` or
`echo-ska/echo_jax.py` at the repo root, that is why they are not there.

| Path | Provenance | Why it is kept |
|---|---|---|
| `reference/echo_jax.py` | `echo-ska/echo_jax.py`, deleted by `7262785`, restored 2026-08-07 | **Live dependency.** `code-tests/test_jax_reference.py` loads it by path as the JAX parity oracle. Also the only surviving implementation of the layer-placement rule `i not in (0, L-1)` (`:517`), which the PyTorch side resolves differently. |
| `MQAR/prefix_bench/` | `MQAR/prefix_bench/`, deleted by `c3ad7ea` | 1,793 lines with no counterpart in `koopman_lm/`: the SKA/attention mask taxonomy, Causal Structured Prefixing, multi-turn chat-SFT recall, and a data-driven rho-gate. `experimentation/experiments/curricula.py` documents the gap its absence causes -- SKA has "no mechanism to mark 'this pair matters, that one doesn't'", which is exactly `prefix_masks.sample_weights_from_mask`. Restored from `origin/pr/module-reorg`, where it was the only tracked copy. |
| `MQAR/MQAR_SKA_Mamba_Benchmark-2.ipynb` | same deletion | Superseded by `models/baselines.py` + `evaluation/mqar/mqar.py`; kept only because it arrived with `prefix_bench/`. Its stored outputs are a 478-character truncated run, so it holds no unique results. |
| `toolcall/SKAv9.py` | `sys+toolcall/SKAv9.py`, deleted by `79a069e` | Its SKA core is a superseded fork, but the surrounding work is unique in the entire lineage: the ToolAlpaca pipeline, `evaluate_tool_calling` (function-name / args / full-call accuracy), a BFCL v3 comparison table, a tool-call BPE tokenizer, `Mamba3Block` with fused-RoPE Triton kernels, and `PrincipledCGLift`. Note it uses the *correct* `L^-1 M L^-T` whitening, not the `M G^-1` form some notes attribute to it. |
| `reference/prepare_data.py` | `echo-ska/prepare_data.py`, relocated 2026-08-12 | The JAX side's tokenization path. `experimentation/training/data/pretokenize.py` has a strictly larger function surface, but only the *surface* was compared -- behaviour was never diffed, so this is kept rather than called superseded. |
| `reference/train_echo.py` | `echo-ska/train_echo.py`, relocated 2026-08-12 | **No counterpart anywhere.** JAX/TPU training harness; its z-loss term and TPU multi-host mesh setup have nothing equivalent on the PyTorch side. |
| `MQAR/mqar_ska_mamba_benchmark.py` | `MQAR/mqar_ska_mamba_benchmark.py`, relocated 2026-08-12 | The original MQAR benchmark driver, reorganized into `experimentation/evaluation/mqar/mqar.py` + `koopman_lm/models/baselines.py` rather than ported. Kept as the record of what the reorganization started from. |
| `lowrank_residual_cuda/` | `lowrank_residual_cuda/`, relocated 2026-08-12 | The optimized symmetric low-rank residual CUDA extension. Unbuildable here, unimported, untested, and the **wrong gauge for SKA** -- it maintains a symmetric residual, whereas SKA's transition update `x_t x_{t-1}^T` is lagged and non-symmetric. `koopman_lm/kernels/fused_state_reference.py` is the dense oracle written for a correct SKA port of it. |

## Restoring something from here

Don't import it. Port it: move the code into `koopman_lm/` under the current
layout, with tests, as a deliberate change. These files were written against
older module layouts and older conventions, so a copy-paste revival is how two
divergent versions of the same idea start.
