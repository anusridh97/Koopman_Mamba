# prefix_bench — fair, prefix-mask-aware associative recall

This package rebuilds the MQAR benchmark around a single idea that the earlier
`mqar_ska_mamba_benchmark.py` got wrong: **a prefix mask is a first-class input to
the Koopman operator, and it must be handed to every architecture the same way to
be a fair comparison.**

## The problem it fixes

SKA (Spectral Koopman Attention) estimates a shared linear operator from streaming
sufficient statistics:

```
G   = Σ_t w_t z_t z_tᵀ + εI          (regularized Gram)
C_v = Σ_t w_t v_t z_tᵀ               (value–key covariance)
M   = Σ_t w_t w_{t-1} z_t z_{t-1}ᵀ   (lag-1 covariance)
B_v = C_v G⁻¹                        (ridge readout / associative memory)
A_w = M   G⁻¹                        (Koopman transition operator)
ŷ   = B_v · f(A_w) · z_q
```

The per-token weight `w_t` is a **choice of measure** for this regression. A prefix
mask specifies `w_t`: which positions are the stored context the operator is fit on,
versus which are queries that only read out.

The original `SKABlock` accumulated `G, M, C_v` over **all** positions with no mask,
so the query tokens polluted the very statistics they read from → near-chance MQAR.
The VADA notebook fixed this with a hard prefix mask, but only SKA received it — an
unfair oracle. This package makes the boundary an in-sequence **`SEP` delimiter
token** (observable to every model) and passes the corresponding structural signal
to attention (prefix-LM masking) and to all models (a shared segment embedding).

## Why the mask "takes" in Koopman but not attention

Attention returns `Σ_s α_qs v_s`. A mask can only zero out disallowed `α_qs`
(support restriction) — it cannot change that the output is a per-query convex
combination of values. There is no shared operator whose estimation the mask
improves. In SKA the mask changes the **fitted operator itself** (`B_v, A_w`, their
conditioning and spectrum). So attention *can* be handed a prefix mask (we do, via
prefix-LM masking) but it consumes it only as support restriction, whereas SKA
consumes it as a change of estimand. Measuring that asymmetry — both models handed
the identical signal — is the experiment.

The boundary is genuinely available from the data in every setting we care about:
MQAR (context precedes queries), tool-calling (trace precedes the answer turn), and
multi-turn chat (turn boundaries are known).

## The mask taxonomy (`prefix_masks.py`)

For SKA (`ska_mode`):

| mode     | `w_t`                                   | meaning |
|----------|-----------------------------------------|---------|
| `none`   | all 1, one global operator              | no separation; queries pollute their own operator (broken baseline) |
| `prefix` | 1 on context, 0 on queries (from `SEP`) | paper "prefix/masked mode"; boundary is a visible token → fair |
| `soft`   | 1 on context, `w_query∈(0,1)` on queries| interpolates `prefix`↔`none`; probes sensitivity to query pollution |
| `causal` | exclusive prefix-sum over chunks        | streaming/LM regime; no boundary handed over; hardest for SKA |

For attention (`attn_mode`):

| mode        | meaning |
|-------------|---------|
| `causal`    | plain lower-triangular mask (ignores the prefix) |
| `prefix_lm` | context visible to all positions (bidirectional within itself), query region causal — this is "passing the prefix mask to attention" |

All models additionally get a shared **segment embedding** `seg_embed(seg_ids)`
(`0=context, 1=query`) so the "where do queries start" signal is available to
everyone. Disable with `--no-seg-embed`.

## Two mathematically-motivated SKA options

Both are on by default in the models; toggle from the runners.

- **ρ-gated dynamics** (`use_rho_gate`, default on). The Koopman path `A_w^K` is
  blended in proportion to `ρ = tr(M G⁻¹ Mᵀ)/tr(G)` — the fraction of key variance
  explained by lag-1 dynamics. On MQAR (no temporal structure) `ρ≈0`, so retrieval
  collapses to the clean associative readout `B_v z_q`; on tool traces `ρ>0`, so the
  persistence filter engages. This fixes the fact that on MQAR the raw `A_w` is noise
  and `A_w^K` actively corrupts the query. Set `--no-rho-gate` for the paper-faithful
  pure power filter.
- **lag-aligned value** (`lag_value`, default off). Pair each value with the
  *previous* key (`C_v = Σ v_t z_{t-1}ᵀ`), so `B_v z_q` retrieves the value that
  *followed* the matching key — the correct estimand for k→v emission. Enable with
  `--lag-value`.

## Experiments

### 1. MQAR mask taxonomy — `run_mqar.py`

Compares `mamba`, `attn/causal`, `attn/prefixLM`, `ska/none`, `ska/prefix`,
`ska/soft`, `ska/causal` at matched size across sequence lengths.

```bash
python -m prefix_bench.run_mqar --device cuda --steps 4000 \
    --seq-lens 64 128 256 512 --vocab-size 256
```

What to look for: `ska/none` ≈ chance (confirms the pollution bug); `ska/prefix`
strong and flat; `ska/soft` degrading as query weight rises; `ska/causal` in between
(the honest streaming number); `attn/prefixLM` ≥ `attn/causal` but still paying
`O(T²)`. If SKA only wins under `prefix`, the paper's number leaned on the boundary;
if it holds under `causal`, it transfers to real LM.

### 2. Long-context tool calls — `run_toolcall.py`

Trains at a short context, evaluates at much longer contexts (length
generalization). Bindings are **overwritten** (recency): the answer is the latest
value, which a pure associative sum gets wrong but recency-aware retrieval does not.
This is the regime with real temporal structure (`ρ>0`).

```bash
python -m prefix_bench.run_toolcall --device cuda --steps 4000 \
    --train-context 1024 --eval-contexts 1024 2048 4096 8192 16384 \
    --num-keys 16 --num-query 8 --overwrite-prob 0.4
```

What to look for: `ska/prefix` flat across context length (constant-memory
sufficient statistics); attention degrading beyond its training length and paying
quadratic compute; pure Mamba collapsing (memory cliff).

## Causal Structured Prefixing (`csp.py`) — generalizing the chunk-causal mask

The fixed chunk-causal SKA mask is a special case of a single rule:

> **A token may use any structured object that was computable before that token.**

Each item `α` (a raw token, or a *record*: a turn, a chunk summary, an entity table,
a tool result, an SKA sufficient-statistic block, a learned memory slot) has a
**release time** `ρ_α`, and may condition target token `x_t` only if `ρ_α < t`.
Equivalently the mask over the combined `[raw ⊕ records]` stream is
`A_{t,j} = 1[ρ_j < t]`. Raw tokens have `ρ(x_i)=i`; a record produced after chunk `c`
(ending at `b_c`) has `ρ(r)=b_c`.

In code this is a per-token **release group** `grp[t]` (non-decreasing in `t`): items
in group `g` are released at the end of `g`; a token in group `g` uses groups strictly
earlier than `g`, plus (for attention) causal context within its own group. The
current fixed chunk-causal mask is `grp = arange(T)//chunk_size`; per-token release
`grp = arange(T)` is fully causal; turn-aligned `grp` gives multi-span prefix-LM.

New modes wired through `train_eval.build_conditioning`:

| kind | mode        | effect |
|------|-------------|--------|
| ska  | `release`   | SKA fits **one operator per release group** from strictly-earlier groups (`SKABlock._segment_causal`) — generalizes fixed chunks to semantic segments/turns. |
| attn | `segment`   | attention gets `segment_causal_bias(grp)` — earlier (closed) groups fully visible, causal within the group. |

The two concrete instantiations:

- **Method 1 — chat / stopped-prefix** (`chat_data.py`, `run_chat.py`): records are
  released at turn boundaries; loss on assistant spans only. SKA accumulates
  `G, M, C_v` over the prefix (earlier turns) and decodes — the paper's prefix mode,
  multi-span. This matches chat SFT / agent inference shape.

  ```bash
  python -m prefix_bench.run_chat --device cuda --steps 4000 \
      --n-rounds 4 --kv-per-round 6 --q-per-round 4 --num-keys 24
  ```

  Key comparison: `ska/release` and `attn/segment` (both handed the turn structure)
  vs `ska/causal` (fixed chunks that cut across turns) and `attn/causal`.

- **Method 2 — timestamped structured-prefix LM**: records released at chunk
  boundaries during an ordinary document, loss on all tokens. The `release`/`segment`
  plumbing is identical — any generator can emit `meta["release_grp"]` (and, for
  heterogeneous records, append record slots to the stream with their own release
  times via `csp.release_attn_bias`). The unified SKA statistics are
  `G_t = εI + Σ_{ρ_α<t} z_α z_αᵀ`, `C_{v,t} = Σ_{ρ_α<t} v_α z_αᵀ`,
  `M_t = Σ_{ρ_β<ρ_α<t} z_α z_βᵀ`.

Efficiency note: `_segment_causal` solves one operator per group and gathers it per
token (`O(T·r²)` memory for the gather). Use few, coarse groups for very long
contexts; the number of groups sets the number of `O(r³)` solves.

## Notes

- Self-contained: no `mamba_ssm`, no Triton. `SimpleMamba2` is a sequential-scan
  reference block; fine for these sizes, slow for very long contexts (expected).
- Data generators (`mqar_data.py`, `toolcall_data.py`) share the return signature
  `(x, y, loss_mask, prefix_mask, seg_ids, meta)` and each has a `_self_test()`.
- Everything trains online (fresh batches per step) and evaluates on a fixed
  held-out batch.
