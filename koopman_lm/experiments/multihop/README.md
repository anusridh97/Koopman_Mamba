# Multi-hop retrieval: optimal-SKA vs Mamba+Attention vs Attention

Self-contained study of where the Koopman/SKA read helps on **multi-hop**
retrieval, and how it compares to attention, using the **real `Mamba3Block`
and `SKAModule`** (extracted verbatim into `koopman_core.py`).

## Files
- `koopman_core.py` — the model core (RoPE, CGFeatureLift, `ModelConfig`, the
  chunked recurrence, `Mamba3Block`, `SKAModule`, `SwiGLUMLP`, `Mamba3CGSKALM`),
  taken **verbatim** from the working notebook so the experiment runs on the real blocks.
- `multihop_task.py` — variable-binding multi-hop task (`v0=lit; v1=v0; …; vk=v(k-1)`
  + distractors; resolve `vk`). Verified: 2000/2000 independent-resolver agreement.
- `run_multihop.py` — param-matched 3-arm comparison: `attention`, `mamba_attn`,
  `mamba_ska`. All share `SwiGLUMLP`; `d_model` is tuned per arm to match params.
- `certificates.py` — **training-free** reachability analysis (the score-family
  margin LP + directional complexity), the orthogonal-residual model.

## Run
From the repo root:
```
python koopman_lm/experiments/multihop/multihop_task.py    # verify the task (numpy only)
python koopman_lm/experiments/multihop/certificates.py     # training-free wall/way-out analysis (numpy+scipy)
python -m koopman_lm.experiments.multihop.run_multihop     # train + eval, hop-by-hop accuracy (needs torch + GPU)
```
Note: `koopman_core.py` is a deliberately self-contained snapshot of the SKA/Mamba
core used for this experiment (frozen at the investigation's state), independent of
`koopman_lm.modules`.

## "Optimal SKA" configuration (and why)
Set in `run_multihop.make_cfg`, from the investigation:
- `ska_gated=False` (ungated additive) — the sigmoid gate can't learn from the
  sparse answer-token supervision (it sits near `sigmoid(-2)≈0.12`); additive lets
  the SKA signal flow.
- `use_cg=False` — the CG lift is a *polynomial reranker over cached features*;
  both experiment and the retrieval-failure theorem say it doesn't escape the
  multi-hop (rank-one) bottleneck.
- `power_K=2`, spectral-norm on, independent projections.

### Optional extension: Chebyshev filter (more expressive, same cost)
Drop-in for the power filter (`A_w` is spectrally normalized to ≈[-1,1]):
```python
def chebyshev_filter(A_w, w_q, coeffs):  # coeffs: nn.Parameter([K+1]), init [0,1,0,...]
    out = coeffs[0] * w_q
    if coeffs.shape[0] > 1:
        tp, tc = w_q, A_w @ w_q; out = out + coeffs[1] * tc
        for k in range(2, coeffs.shape[0]):
            tn = 2.0 * (A_w @ tc) - tp; out = out + coeffs[k] * tn; tp, tc = tc, tn
    return out
```
Init `coeffs=[0,1,0,…]` so it starts identical to `A_w^1` and can only improve.

## What the certificates say (training-free, see `certificates.py`)
On a hard (sign-matrix) multi-hop relation:
- A single rank-one frozen residual reaches **~6–15%** of the relation; a
  row-adaptive (re-encoding) map reaches 100%. Adding embedding **dimension does
  not help and can hurt** (more-orthogonal docs remove the correlations the
  residual relies on).
- Most failures are **selection** failures (feasible θ exists, the residual rule
  misses it), recoverable by a better/learned selection rule **without more capacity**.
- Feasibility climbs to **~99% at ~8 directions** — multi-direction (rank-k /
  multi-head) is the escape; the count is the directional-complexity / set-cover budget.

## Predictions for `run_multihop.py`
- `attention` tops multi-hop, slowest decay with hop depth.
- `mamba_ska` should be **much closer to attention than an isolated-SKA primitive**,
  because it now has depth + MLP (re-encoding) carrying the conditioning.
- Residual gap concentrates at **deep hops** (the geometric / high-sign-rank fraction).

## Status
Task + certificates run here (numpy/scipy). The training loop is **py_compile-verified
but not runtime-tested** (built without a GPU) — produce the numbers on your hardware.
`Mamba3Block`/`SKAModule`/`Mamba3CGSKALM` are your own code, unchanged.
