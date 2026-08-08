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

| Path | Provenance | Why it is kept |
|---|---|---|
| `reference/echo_jax.py` | `echo-ska/echo_jax.py`, deleted by `7262785`, restored 2026-08-07 | **Live dependency.** `code-tests/test_jax_reference.py` loads it by path as the JAX parity oracle. Also the only surviving implementation of the layer-placement rule `i not in (0, L-1)` (`:517`), which the PyTorch side resolves differently. |
| `MQAR/prefix_bench/` | `MQAR/prefix_bench/`, deleted by `c3ad7ea` | 1,793 lines with no counterpart in `koopman_lm/`: the SKA/attention mask taxonomy, Causal Structured Prefixing, multi-turn chat-SFT recall, and a data-driven rho-gate. `koopman_lm/experiments/curricula.py` documents the gap its absence causes -- SKA has "no mechanism to mark 'this pair matters, that one doesn't'", which is exactly `prefix_masks.sample_weights_from_mask`. Restored from `origin/pr/module-reorg`, where it was the only tracked copy. |
| `MQAR/MQAR_SKA_Mamba_Benchmark-2.ipynb` | same deletion | Superseded by `models/baselines.py` + `evaluation/mqar/mqar.py`; kept only because it arrived with `prefix_bench/`. Its stored outputs are a 478-character truncated run, so it holds no unique results. |
| `toolcall/SKAv9.py` | `sys+toolcall/SKAv9.py`, deleted by `79a069e` | Its SKA core is a superseded fork, but the surrounding work is unique in the entire lineage: the ToolAlpaca pipeline, `evaluate_tool_calling` (function-name / args / full-call accuracy), a BFCL v3 comparison table, a tool-call BPE tokenizer, `Mamba3Block` with fused-RoPE Triton kernels, and `PrincipledCGLift`. Note it uses the *correct* `L^-1 M L^-T` whitening, not the `M G^-1` form some notes attribute to it. |

## Restoring something from here

Don't import it. Port it: move the code into `koopman_lm/` under the current
layout, with tests, as a deliberate change. These files were written against
older module layouts and older conventions, so a copy-paste revival is how two
divergent versions of the same idea start.
