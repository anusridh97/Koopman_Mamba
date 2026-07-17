# Acceptance Spec: `√β` Symmetrization + Power-Iteration Removal

**Status gate.** This spec must pass before step (1) — `√β` symmetrization — is
considered landed. Residual parity is **necessary but not sufficient**; the
binding acceptance criterion is a retrained task eval (§C). Everything before
that is a prerequisite, not a proof of correctness.

**Why this spec exists.** The incremental-transport oracle (`38b04a7`,
`code-tests/test_incremental_lar_parity.py`) proves the backend equals
recomputation *given the statistics*. `√β` changes *which statistics are
accumulated*, so it is invisible to that oracle — both asymmetric-β and `√β`
are internally self-consistent and each passes a parity-against-itself test.
Only a task eval on **retrained** weights distinguishes "correct" from
"self-consistent but degraded." This is the same "needs a task eval, not a
residual" property the causal-norm-clip has, and it is *not* the
exact-equivalent property the transport has. The checkpoint incompatibility
(`√β` models ≠ asymmetric-β checkpoints) is not merely "expected for v1.1" — it
is the proof that this change cannot be validated on existing weights at all,
which is exactly why §C must be on a freshly-retrained config.

---

## A. Convention consistency — the accumulation sites, diffed side by side

The `√β` change must use a **byte-identical weighting convention** across every
path that accumulates SKA statistics. The most likely silent bug is not a math
error in one site but a *divergence* between the training path and the decode
path that no single-recurrence oracle catches. Present all sites in one diff,
aligned, before committing.

Sites: `chunk_stats.py` (fast chunk-causal training), `chunk_stats_exact.py`
(exact-replay reference), the SKA `forward` callers (`ska.py`, `fast.py`), and
the decode path (`recurrent.py` `_proj_norm` + `_ska_step`).

For each site, at the same point relative to the existing ℓ2 normalization:

| Quantity | Required form | Bug to rule out |
|---|---|---|
| Weighted key | `x = √β · z` | applying `β` not `√β` (reintroduces asymmetry) |
| Weighted value | `v̄ = √β · v` | forgetting the value weight — leaves `C` in the wrong geometry |
| Interior lag term (`M`) | `√(βₜ·βₜ₋₁) · zₜ zₜ₋₁ᵀ` | using `√βₜ` alone (the asymmetric-β bug being fixed) |
| Weight-vs-norm order | `√β` applied *after* ℓ2, identically at every site | one site normalizes-then-weights, another weights-then-normalizes |

**Own-weight vs cross-weight (the asymmetry to preserve):** with `x=√β·z` and
`v̄=√β·v`, `G += x xᵀ` gives own-weight `βₜ`, `C += v̄ xᵀ` gives own-weight
`βₜ` (matching `G`'s diagonal), and only `M`'s cross-term takes the
geometric-mean `√(βₜβₜ₋₁)`. Getting `C` and `M` the *same* is a bug in either
direction. **Cross-check:** `G` and `C` are numerically *invariant* under the
switch (both already produce `β·zzᵀ` / `β·vzᵀ`); only `M` and the boundary
move. A diff that changes `G` or `C` is wrong.

**Acceptance for A:** a reviewer confirms the four rows are identical across all
sites in one screen. Any row differing across sites → **stop**, that is the
divergence bug; it surfaces as a train/decode mismatch that looks like a
retrieval regression but isn't.

---

## B. Weighted boundary oracle — DONE (`test_chunk_boundary_combine_sqrt_beta`)

The cross-chunk term under `√β` carries **per-endpoint** weights:
`ΔM_boundary = √(β_{i,0}·β_{i-1,-1}) · z_{i,0} z_{i-1,-1}ᵀ` — two different
per-token weights, not a shared chunk weight. Built symmetrically as
`x_t=√β_t z_t`, the boundary is just `x_c x_{c-1}ᵀ`.

Implemented in `code-tests/test_incremental_lar_parity.py`:
`test_chunk_boundary_combine_sqrt_beta`, with **per-token-varying β** at the
boundary (a constant-β test cannot separate geometric- from arithmetic-mean).
Asserts the correct combine at machine precision (<1e-12) AND that four wrong
weightings each deviate >1e-4 (forgot-prev-weight, forgot-curr-weight,
asymmetric `βₜ`, arithmetic mean) — so a boundary regression is caught, not
swallowed by a loose tolerance. Validated: correct ~1e-16; bugs 2e-4…3e-2.

---

## C. Retrain task eval — the binding acceptance criterion

Retrain the smallest config that exhibits both behaviors (≈1M synthetic scale,
2 Mamba-2 + 2 SKA, `r=24` is sufficient and cheap). Two runs, identical except
the `√β` change:

- **Baseline:** current asymmetric-β + spectral norm (the paper's regime).
- **v1.1:** `√β` symmetric + power iteration removed.

Evaluate **both** metric regimes on **both** runs:

| Regime | Metrics | A regression here means |
|---|---|---|
| Structured retrieval | MQAR, ToolTrace | `√β` changed the lag-one spectrum enough to hurt the transient-mode suppression tool-calls depend on |
| Diffuse / LM-like | WikiText-103 ppl, or synthetic CoT-retrieval tiers | `√β` hurt the general path (unexpected — flag) |

**Go / no-go:**

```
Retrieval (MQAR/ToolTrace) within noise of baseline?
├─ YES → LM/ppl within noise (or better)?
│        ├─ YES → LAND. Proceed to power-iter-removal commit.
│        └─ NO  → investigate LM path; suspect an §A convention bug the
│                 retrieval tasks tolerated. Re-audit the sites before landing.
└─ NO  → retrieval REGRESSED under √β.
         ├─ Is α (from §D parity) actually ≈1 in these runs?
         │   ├─ NO  → spectral norm was load-bearing here; √β did NOT make it a
         │   │        no-op → contradicts the contractivity proof for this data
         │   │        → symmetrization bug (§A) OR boundary bug (§B). Do NOT land.
         │   └─ YES → α≈1 but retrieval dropped → symmetrization changed the
         │            operator's useful spectrum, not just its norm. The real
         │            risk from the tool-calling analysis. Do NOT land as
         │            unconditional; escalate to a design call (tool-call cost
         │            vs LM gain), not a commit.
```

The critical branch is **α≈1 AND retrieval regressed** — the outcome the
residual oracle cannot see and the whole reason this eval gates the commit.
Shipping `√β` silently there would trade away the ToolTrace result the paper's
filter analysis says depends on exactly the spectrum `√β` perturbs.

**Acceptance for C:** top path (retrieval holds AND LM holds) on the retrained
small config. Any other terminal node blocks the unconditional land.

---

## D. Parity-vs-current-output — retained, correctly scoped

- **Proves:** where α was already ≈1, removing the 20-iter power method is an
  exact no-op → justifies the *power-iteration-removal* commit (step 2).
- **Does NOT prove:** `√β`-retrained task quality (same weights through
  different stats is off-distribution; see §C).

**Runtime assertion (power-iter-removal commit):** `σ(A) ≤ 1 + tol`, `tol`
absorbing the strict-margin float slack (oracle showed max σ = 0.999; use e.g.
`tol=1e-4` in FP32). Double duty — symmetry guard *and* latent-boundary-bug
detector (a wrong §B boundary makes stats non-cumulative, can push σ>1). Keep it
in the exact-replay and decode paths, not just training.

---

## Commit sequencing (gates attached)

1. **`√β` symmetrization** — §A side-by-side diff + §B oracle green — **gated by
   §C retrain eval before land.**
2. **Power-iteration removal** — separate commit, `σ ≤ 1+tol` assert live (§D).
   Separate so a moved residual attributes to exactly one change.
3. **Incremental `(L,A,R)` torch kernel** — tested against the `38b04a7` oracle.

Do NOT fold §A/§B into the power-iter-removal commit: if a residual or metric
moves, it must localize to one change — the discipline that made `rAsand`
useful (separate the mechanism from the identity).

**The one line:** the oracle certifies the backend; only the §C retrain eval
certifies the statistics change, and the `α≈1 yet retrieval drops` node is the
specific failure the oracle is blind to and the tool-calling analysis predicts
is possible.
