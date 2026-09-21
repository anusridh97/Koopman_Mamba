# Echo/SKA hyperparameter sweeps — configs and results

Four completed architecture+optimizer searches on one protocol. Everything below
is measured, with the caveats that matter for extrapolation stated explicitly.

## Protocol (identical across all four rungs)

- Data: FineWeb-Edu, **Llama-2 tokenizer, vocab 32,000**, seq len 2,048
- Objective: LM loss with recall weighting (`recall_weight: 4` on ~4% of tokens)
- Precision: bf16 autocast, fp32 master weights, fused AdamW
- `effective_batch: 96` sequences = 196,608 tokens/step
- Cosine-to-zero schedule over `max_steps`, no early stopping
- Budget: **~65 tokens per parameter at every rung** (this is a design choice and
  it is the main limitation — see "What these cannot establish")

## The four rungs

| rung | d_model | n_layers | N (total, tied emb) | tokens | tok/param | untuned ref loss | seed σ | best loss | gain |
|---|---|---|---|---|---|---|---|---|---|
| 3M   | 64  | 13 | 2,991,928   | 0.21 B | 70.2 | 4.34420 | 0.00142 | **4.15179** | 0.1924 (136σ) |
| 10M  | 128 | 23 | 10,217,328  | 0.67 B | 65.4 | 3.55655 | 0.00448 | **3.43874** | 0.1178 (26σ) |
| 50M  | 384 | 18 | 51,698,320  | 3.36 B | 65.0 | 2.78960 | 0.00582 | **2.75870** | 0.0309 (5.3σ) |
| 180M | 640 | 27 | 182,665,776 | 10.00 B | 54.7 | 2.40712 | 0.00144 | **2.40218** | 0.0049 (3.4σ) |

Trials completed: 1,041 / 177 / 30 / 14. σ is a measured seed-replicate noise
floor per rung; "gain" is best-vs-untuned-reference in units of σ.

**180M ran anchors only** — 15 controlled one-factor contrasts, no architecture
search. Its "best" is the reference geometry at a better LR.

## Best configuration per rung

| axis | 3M | 10M | 50M | 180M† |
|---|---|---|---|---|
| `mamba_expand` | 3 | 3 | 3 | 2 |
| `d_state` | 32 | 24 | 32 | 24 |
| `ska_n_heads` | 4 | 2 | 4 | 4 |
| `ska_rank` | 32 | 48 | 48 | 24 |
| `n_ska_layers` | 6 | 6 | 4 | 4 |
| `beta_policy` | linear | learned | learned | learned |
| `ska_power_K` | 1 | 1 | 1 | 1 |
| `depth_tier` | full | full | full | full |
| `weight_decay` | 0.15 | 0.1 | 0.1 | 0.1 |
| `warmup_ratio` | 0.04 | 0.06 | 0.02 | 0.02 |
| **learning_rate** | **0.00541** | **0.00752** | **0.00483** | **0.00256** |

† anchors only, see above.

### Was each optimum actually bracketed? (checked against the trial tables)

| rung | LR range SAMPLED | winner | trials above winner | verdict |
|---|---|---|---|---|
| 3M | 0.00150 – 0.00550 | 0.00541 | **0** | censored; optimum may be higher |
| 10M | 0.00111 – 0.00798 | 0.00752 | 14 (5%) | interior but marginal (94% of max) |
| 50M | 0.00080 – 0.00884 | 0.00483 | 24 (38%) | **properly interior** |
| 180M | 0.00121 – 0.01789 | 0.00256 | 16 (41%) | **properly interior** |

So three of the four are usable as locations, not just bounds.

## The two things to know before extrapolating

**1. The LR optimum falls with width, and the local slope is NOT constant.**
Three interior measurements (10M, 50M, 180M — see the bracketing table above):

```
d_model 128 -> lr 0.00752
d_model 384 -> lr 0.00483      pairwise exponent p = -0.403
d_model 640 -> lr 0.00256      pairwise exponent p = -1.243
least squares through all three: lr = 0.167 * width^-0.627
```

The pairwise exponent changes by 3x between the two intervals, so this is a
crude summary of a curve rather than a law (muP would predict p = -1 throughout).
It predicts 0.00311 / 0.00247 / 0.00188 for d_model 576 / 832 / 1280.

Treat it as a prior for choosing a search RANGE, not as a substitute for
searching. Two things also shift it by an unknown amount: this trend is measured
at `effective_batch 96` and ~65 tok/param, and a larger batch pushes the optimum
UP while a longer horizon pushes it DOWN. The 3M point is excluded because it is
genuinely censored (zero trials above the winner).

**2. Architecture importance decays with scale; the optimizer's does not.**
Variance share of `learning_rate` is 0.752 / 0.589 / 0.751 / 0.584 across the
rungs — always the largest axis by 2–8×. `ska_rank` is the only architecture axis
that resolved at more than one rung and its share halves each step:
`0.340 → 0.292 → 0.187 → 0.075`, and it is flat at 180M (Δ=−0.00102 against a
0.00316 threshold). `depth_tier` was the one resolved architecture axis at 10M
and exactly 0.000 at 50M. Most of the disagreement in the table above is on axes
that never resolved, i.e. those "winners" are noise.

Practical read: **a fixed sensible geometry plus a tuned LR captures nearly all
available gain by 50M+.** `ska_rank 48` is the one defensible non-default (it
resolved better than 24 at both 10M and 50M; rank 8 resolved *worse* at 10M by
+0.0595).

## Loss along the ray

Over 61× in N, with D ≈ 65N throughout:

```
untuned reference   L ~ N^-0.1444    R2 = 0.99613
best found          L ~ N^-0.1334    R2 = 0.99579
```

**This is not Chinchilla's α.** It is the slope *along* D = 65N and mixes the N
and D contributions.

## Fixed-N data scaling (separate 39-run grid, same architecture)

Loss gain per DOUBLING of tokens at fixed model size — this one is directly
useful and needs no fit:

| N | 8→16× tok/param | 16→32× | 32→64× |
|---|---|---|---|
| 17.5 M | −0.513 | −0.251 | −0.129 |
| 28.4 M | −0.301 | −0.174 | −0.130 |
| 62.9 M | −0.177 | −0.138 | — |
| 124.3 M | −0.165 | — | — |

Each doubling buys 0.5–0.75× the previous one. 4 of 39 runs diverged (NaN), all
at lr 0.0100, across four different widths — treat 0.010 as an upper bound.

## Measured throughput (H100 80GB, for cost conversion)

| config | tok/s/GPU | implied MFU |
|---|---|---|
| 180M, vocab 32,000, `ska_rank 48` | 51,107 | 4.4% |
| 360M-class smoke, vocab 32,000 | 26,700 | 5.8% |
| 180M, vocab 128,256, pdbs 8, 4 GPUs | 39,400 | — |

MFU is low partly by choice: `torch.compile` is disabled (a documented conflict
with activation checkpointing under the fused prefix-scan path) and
`ska_inverse_cholesky` at rank 48 keeps per-token (B,T,H,r,r) statistics.
FLOPs/token ≈ `6·d_model·vocab + 6·N_non_embedding` predicted the 180M cost to
within 0.4% of measurement, so it is a usable cost model.

## What these cannot establish

- **Chinchilla α and β.** All four rungs sit on D ≈ 65N by design, so in
  (log N, log D) they are 94% collinear; profiling found 198 distinct (α, β)
  pairs fitting the four points equally well, α anywhere in 0.05–1.00.
- **That SKA beats Mamba-2.** The 3M baselines TIE when both are untuned, and no
  tuned Mamba-2 ladder exists. What is established: `ska_delta` (loss with the
  SKA path ablated at eval, minus loss as trained) is positive in **0 of 1,262**
  completed trials, i.e. SKA is load-bearing *within* this architecture.

## What we are running now (different protocol — do not mix with the above)

**Llama-3.1 tokenizer, vocab 128,256; FineWeb-Edu 100 B tokens; seq 2,048;
`effective_batch: 256` (524,288 tok/step, 190,734 steps = exactly one epoch);
plain LM loss, no recall weighting.**

Size labels follow GPT-3/Pythia/Mamba convention = **total** params including
embeddings. Mamba's 130M/370M/1.4B shapes at vocab 128,256 give
183.4M/433.3M/1.471B, which is where Mamba-3's 180M/440M/1.5B labels come from
(agreement within 0.3%). Our block carries a Mamba-2 mixer **and** a SwiGLU
channel mixer — roughly 2× their per-layer cost — so we solve for our own shapes
at aspect ratio ≈ 25:

| target | our shape | N (total) | N (non-emb) | band | 100B cost (H100) |
|---|---|---|---|---|---|
| 180M | d576 / L23 | 186,046,408 | 111,451,834 | +1.4% | ~705 GPU-h (measured) |
| 440M | d832 / L33 | 439,686,136 | 332,977,144 | +1.5% | ~1,672 GPU-h |
| 1.5B | d1280 / L54 | 1,442,134,256 | 1,277,966,576 | −1.9% | ~5,510 GPU-h |

Frozen from the sweeps: `ska_rank 48`, `d_state 24`, `ska_n_heads 4`,
`mamba_expand 2`, 4 SKA layers (even placement), `beta_policy learned`,
`power_K 1`, exact-inverse-Cholesky backend, `weight_decay 0.1`,
`grad_clip 1.0`, warmup 2% of steps.

**Being re-measured, not extrapolated:** learning rate. 6 points at 180M
(0.0008–0.010) and 4 at 440M (0.0012–0.0096), then one extrapolation step to
1280 verified by a 2-point bracket at the target width.

Measured per-rank memory at rank 48, seq 2048, vocab 128,256 (real fwd+bwd+step,
fp32 params + bf16 autocast + fused AdamW), plus DDP's gradient bucket:

| size | per_device_batch | peak GiB of 79.2 |
|---|---|---|
| 180M | 8 | 48.5 |
| 440M | 4 | 37.0 |
| 1.5B | 2 | 48.2 |

1.5B at pdbs 4 measures 71.9 GiB — fits but only 7.3 GiB spare, so we run 2.
