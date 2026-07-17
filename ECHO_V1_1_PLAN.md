# Echo v1.1 — Consolidated Plan: State, Readout, and Training

This document ties together everything established across the v1.1 effort: the
symmetrized statistics, the incremental backend, the input-dependent spectral
readout, the training modes, the mixed LM+retrieval protocol, and the
experiment ladder with its gates. It is the single reference the
implementation is built against and the retrain decisions are read against.
Status of every component is marked explicitly — the plan's central discipline
is that *provisional is not landed*, and each gate below says exactly what
promotes one to the other.

> Ledger reconciled to the committed repo state (§5): the gated-read oracle is
> `code-tests/test_gated_blend_oracle.py` (`01b7ce5`), and the §C retrain
> harness (`scripts/sqrt_beta_ablation/`, `1525708`) is listed as the Gate-1
> mechanism. All other commit hashes verified against `git log`.

---

## 0. How it all fits together

Two structural ideas organize the entire design. Everything else is
consequence.

**Spine 1 — one invariant chain.** Weight the accumulated statistics
*symmetrically* (`x_t = √β_t·z_t` in both key slots, cross-term
`√(β_t·β_{t−1})`), and the whitened lag-one operator becomes provably
contractive: `‖A‖₂ ≤ 1`, strictly. That single bound pays out three times at
once. First, the six-step spectral normalization becomes a literal no-op
(`α ≡ 1`, provably — the power-iteration estimator is a *lower* bound on
`σ_max`, so `σ_est ≤ σ_true ≤ 1 ⟹ α = 1` in the live path, not just in a
numpy model), so it can be deleted as an exact equivalence. Second, the
incremental Givens transport of the whitened operator is exactly 1-Lipschitz,
so numerical drift between rebuilds is pure accumulated rounding — linear,
bounded, and reset by periodic refactorization. Third, any per-token *convex*
blend over `{I, A²}` is contractive pointwise, which is what makes the
input-dependent readout safe with zero new normalization machinery. One
weighting convention, three guarantees.

**Spine 2 — state/read separation.** Everything on the write side — the
statistics `G, M, C` and the derived factors `L, A, R` — is fixed,
convention-pinned (commit `1f1301b`), and validated by residual oracles
against fresh recomputation. Everything *adaptive* lives on the read side: the
gate `λ_t` modulates how a query reads the state, and it reads only
state-independent inputs. Every new capability in v1.1 was deliberately added
read-side, because the write side is where exactness lives and the read side
is where per-token flexibility is free. The corollary that shapes the whole
plan: each component is independently verifiable, and the plan's ordering *is*
the order of verification — with hard gates exactly where verification
requires retraining rather than a residual check.

---

## 1. State and write path (per SKA head)

The accumulated statistics, all in FP32:

```
G = εI + Σₜ xₜxₜᵀ          xₜ = √βₜ · zₜ        (zₜ: ℓ2-normalized key)
M = Σₜ xₜxₜ₋₁ᵀ                                   (cross-weight √(βₜβₜ₋₁))
C = Σₜ v̄ₜxₜᵀ               v̄ₜ = √βₜ · vₜ
```

with `ε = 10⁻³` and `βₜ ∈ (0,1)` the *existing* write weight, unchanged from
`1f1301b`. The convention has a structural asymmetry that the code reflects:
`G` and `C` are **own-weight** terms — both slots (or value-times-own-key)
carry the same token's `β`, which multiplies to `β` either way — so they are
*invariant* under the symmetrization. Only `M`, the **cross-weight** term, and
the cross-chunk boundary term change. Consequently the implementation touches
only the five call sites (chunk, exact, fast, prefill, decode) via one shared
`symmetric_key_value` helper; the stats kernels themselves are byte-unchanged,
which is why the transport oracle's validation of their einsum/boundary
structure carries over.

Derived state carried per head:

```
L = chol(G)   (r×r lower-tri) — updated incrementally by Givens in decode
A = L⁻¹ML⁻ᵀ  (r×r)           — the whitened lag-one (Koopman) operator
R = CL⁻ᵀ     (P×r)           — the value map (one-sided whitened)
x_last        (r)             — previous symmetric key (renamed from z_last
                                so a raw-key carry cannot silently return)
```

**The contractivity theorem and why √β is load-bearing.** With symmetric
weighting, `A = Σ yₜyₜ₋₁ᵀ` where `Σ yᵢyᵢᵀ = I − εG⁻¹ ⪯ I`, and
Cauchy–Schwarz (equivalently, the projection-matrix argument of the Cholesky
paper's Lemma 3.1) gives `‖A‖₂ ≤ 1 − ε/λ_max(G) < 1`. Under the old
asymmetric weighting, `A = Σ √(βₜ/βₜ₋₁)·yₜyₜ₋₁ᵀ`, and the bound breaks
whenever a high-β token follows a low-β one. The gated-read oracle's C1 teeth
made this concrete: *random* β almost never violates the bound (max σ = 0.56
over 30 seeds) — the violation requires **structured** write-weights, an
alternating low-β/high-β pattern that produced σ = 8.93 asymmetric vs 0.89
symmetric *on the same sequence*. Two consequences worth remembering: it
explains why the old asymmetric code survived empirically (unstructured β
rarely tripped the clamp), and it identifies exactly who the adversary is — a
*trained* write gate produces β ≈ 0 on distractors and β ≈ 1 on facts, the
precise structure that breaks the asymmetric bound. The old clamp was a
landmine, not dead weight.

**Cross-chunk boundary.** In chunk mode, the term crossing chunks carries
per-endpoint weights: `√(β_{i,0}·β_{i−1,−1})·z_{i,0}z_{i−1,−1}ᵀ` — the first
token of chunk *i* against the last of chunk *i−1*, each with its own β. The
weighted-boundary oracle validates this with *non-constant* β at the boundary
specifically, because a constant-β test cannot distinguish the geometric mean
from the arithmetic-mean near-miss.

**The one hard boundary rule:** the read gate `λ_t` (Section 2) must never
feed the write weight `β_t`. The state/read separation is what makes every
guarantee above independent of the readout; coupling them collapses it. This
is enforced as a code comment on the gate and as a review pin.

---

## 2. Readout: the input-dependent spectral blend

The full read, per token, per head:

```
λₜ  = σ( (w_λ · z_qₜ)/√r + b_λ )        # gate on the PRE-whitening query
q_w = L⁻¹ z_qₜ                           # one triangular solve
h   = (1−λₜ)·q_w + λₜ·A(A q_w)           # convex blend over {I, A²}
yₜ  = η · R h                            # value readout
```

**What the dial means.** `λ = 0` is exact ridge retrieval (the closed-form
associative-recall predictor); `λ = 1` is the original K=2 spectral SKA
(persistent-mode amplification, transient suppression — the mechanism behind
the tool-call and MQAR results). Intermediate values retain diffuse
information while partially suppressing transients. The gate makes this a
*per-token* choice, which is where the LM-vs-retrieval tension actually lives:
a binding query and a diffuse-context query interleave in the same sequence
through the same head, and a fixed per-head constant cannot track that.

**One parameterization, three regimes.** A synthesis worth stating plainly:
the memo-§5 fixed blend and the input-dependent gate are *the same
parameterization* under different freeze settings. `b_λ` **is** the memo's
per-head `λ_h`, in logit space. Freeze `w_λ ≡ 0` with `b_λ` trainable and you
have exactly memo §5 (a learned per-head scalar); unfreeze `w_λ` and you have
input dependence; freeze both and you have original SKA at `λ = σ(b_λ)`.
There is no separate `λ_h` parameter anywhere. This is what makes the
experiment arms a clean single-variable design: Arm 0 vs Arm 1 differ *only*
in `requires_grad` on `w_λ`, and both start (bitwise — oracle check C3) from
identical behavior.

**Pinned conventions** (the `w1`-class decisions, fixed here so the diff
doesn't re-litigate them):

1. **Gate input is the ℓ2-normalized query *before* whitening.** Two
   rationales, and the second is the stronger one. (a) State-independence:
   `q_w = L⁻¹q` depends on the accumulated state, so gating on it would make
   the gate's semantics drift as `L` evolves — a read-gate/state coupling of
   exactly the class the whole design eliminates. (b) Train/decode identity:
   chunk-causal training reads the *chunk-start* operator while decode reads
   the up-to-the-token operator (a pre-existing mismatch). Because the gate's
   input is state-independent, the *gate itself* is immune — same content,
   same λ, train and decode; only the operator it modulates differs, which is
   the pre-existing mismatch, unchanged, and eventually eliminated by exact
   replay. Gating on `q_w` would have made the gate inherit the mismatch.
2. **Init `w_λ = 0`, `b_λ = logit(0.9) ≈ 2.197`.** Starts in the
   proven-retrieval regime but inside the trainable band (σ′ at 0.9 is
   0.09; at 0.99 it would be 10× smaller). Deviation from retrieval behavior
   requires dense evidence — and (Section 4) the same init protects the
   operator's gradient bootstrap.
3. **`w_λ` and `b_λ` are excluded from weight decay.** Decay on `w_λ` pulls
   it to zero — i.e., weight decay is a hidden prior for
   *non*-input-dependence, which would bias the experiment against the very
   mechanism under test. Both join the existing no-decay list (ridge, η, MLP
   angles, norm scales).
4. **λ never feeds the write path** (Section 1's boundary rule).

**Why it is safe.** Contractivity is preserved *pointwise by convexity*:
`λₜ ∈ (0,1)` per token, so `‖(1−λₜ)I + λₜA²‖ ≤ (1−λₜ) + λₜ‖A²‖ ≤ 1` for
every token independently. The runtime `σ(A) ≤ 1+tol` assert is untouched and
remains valid. No normalization, no clamp, no new state.

**The gradient, and where it concentrates.**

```
∂L/∂λₜ = ⟨ g_y , η R (A² − I) q_w ⟩         (validated vs FD to 1e-10)
```

The gate's per-token gradient magnitude is proportional to the *disagreement*
between the spectral and ridge readouts — tokens where the choice is
inconsequential contribute nothing. Signal is automatically spent where the
decision matters. The chain to `(w_λ, b_λ)` and the aux-loss gradient are
likewise FD-validated (oracle C2).

**The backward stays Cholesky-free.** The read equals
`η[(1−λ)·CG⁻¹q + λ·CG⁻¹MG⁻¹MG⁻¹q]` (oracle C4, 8e-16), so the App-C.7
recurrence extends to the gated read and no Cholesky derivative ever enters —
the compatibility certificate for exact-replay training, pre-paid.

**Cost.** One length-r dot product and a sigmoid per token per head
(`r+1` parameters/head, ≈1.2K total at 180M scale), one broadcast axpy in the
read. `q_w` is already materialized as the input to the `A²` chain, so the
ridge branch is free. λ is computed in FP32 alongside the read.

---

## 3. Execution modes

Four modes share the same mathematical object; they differ in how the state is
constructed and what the backward needs.

| Mode | State construction | Read | Backward | Status |
|---|---|---|---|---|
| **Chunk-causal training** | Batched: per-chunk stat GEMMs + exclusive scan + batched Cholesky at chunk boundaries | All tokens in chunk *c* share `A^(c)`; each token has its own λₜ | Plain autograd (the gate adds one differentiable sigmoid + axpy; no custom kernel) | Production path |
| **Prefill** (agentic/tool-calling hot path) | Batch-accumulate over the prompt, factorize once | Per-token λ over the prefix-final operator | n/a | Production path |
| **Decode** | Incremental rotation-replay: stored `(cₖ,sₖ)` Givens sweeps, `18r²+15r` flops/token, `u_t = L⁻¹x_t` recovered free | Per-token exact | n/a | Kernel next; math validated by the transport oracle (`38b04a7`) |
| **Exact replay** (reference/ablation; future training) | Per-token incremental, or rank-*c* block updates (`O(r²c)`, BLAS-3) per chunk | Per-token exact | C.7 recurrence, `O(Kr²p)`, Cholesky-free | Gated on wall-clock benchmark vs batched |

**The crossover economics that fix the split.** At Echo's ranks
(r ≈ 48–56), highly-optimized batched Cholesky beats sequential rank-1
updates for *batch* work — so training and prefill stay batched. For decode,
the flop count flips even at r = 56 (full re-factorization ≈ 175K flops/token
vs ≈ 57K incremental, one small matrix, no batching advantage for the
baseline) — so decode gets the incremental kernel. The rank-*c* block form is
the lever that could someday make exact-replay *training* competitive; it is
explicitly benchmark-gated, not assumed.

**Precision policy.** FP32 for all statistics, factor operations, triangular
solves, and λ; BF16 for projections and the residual stream. The incremental
path's per-step drift is `O(r·u)` with the *FP32* unit roundoff
(u ≈ 6×10⁻⁸ — budget against the Cholesky paper's FP32 anchor of
< 3.1×10⁻⁷ over 50-step windows, **not** its float64 table). Exact-arithmetic
transport is 1-Lipschitz, so drift is linear in steps and reset by periodic
refactorization; start with rebuild cadence `N ≈ r` and tune against the
measured Frobenius residuals. The `v_w` forward-substitution accumulator is
held in higher precision.

---

## 4. Training protocol (mixed LM + retrieval)

**Data.** Sequence-level interleaving: each training sequence is *either* a
FineWeb-Edu document *or* one synthetic retrieval sequence (the table2-style
system-prompt + tool-trace curricula). Retrieval episodes are never packed
inside LM documents — SKA statistics are cumulative over the sequence, so
cross-document bleed inside a packed sequence would teach retrieval across
document boundaries, which is noise. Mix ratio: start at 10% retrieval, sweep
{5, 10, 20}% — retrieval tasks saturate quickly at these scales, and past the
knee you are spending tokens on a solved distribution and paying in
perplexity. Constant ratio, no curriculum; anneal only on evidence of
interference.

**Loss.** Uniform cross-entropy on all tokens in both sequence types.
Answer-token accuracy is a *metric*, not a loss term — uniform CE avoids
per-token masking machinery, and the trace tokens' triviality costs little.

**Optimizer.** AdamW exactly as-is (betas, warmup, cosine, clipping). The
no-decay list gains `w_λ` and `b_λ` (pin 3). No other optimizer change — the
standing rule of one variable at a time applies to the optimizer too.

**The gradient-flow map — what trains what.** The value map `R` and the
whitened query `q_w` receive gradient from *both* branches of the blend, so
`G`, `C`, and the content-addressing role of `W_k`/`W_q` are trained by every
token regardless of λ. But `A` appears only in the λ-branch — so **the
lag-one statistic `M`'s gradient is λ-weighted per token**. What high-λ
tokens train is not the keys wholesale but their *temporal persistence
structure*: as the gate specializes, the operator's learning signal
concentrates on exactly the tokens that use the filter. Plausibly good
(retrieval queries shape the spectrum; LM queries stop diluting it) — but it
creates one specific feedback risk:

**The bootstrap risk.** If λ drifts down *globally* early — before `W_k` has
learned useful lag structure — then `M`'s gradient starves, `A` stays
unstructured, the spectral branch never becomes worth choosing, and λ stays
down. Self-sealing. Three design choices already mitigate it: the 0.9 init
means `M` gets ~full gradient from step 0 (deviation requires evidence — the
init now protects the operator bootstrap, not just retrieval behavior);
oracle C3 guarantees step-0 behavior is bitwise the status quo (the
zero-initialized output projection / additive-injection warmup story is
untouched); and the failure has a clean observable, distinguishable from
healthy specialization. Its pre-committed response lives in the decision tree
below (freeze `b_λ` early; the aux loss is the *wrong* lever for this one).

**The auxiliary loss** (`−log λₜ` on labeled query positions, weight
0.01–0.1) is designed, FD-validated, and **ships OFF**. Supervising λ toward
1 from step 0 would make the learnability question — the one genuinely
uncertain thing the experiment exists to answer — unanswerable
(learned-vs-imposed). It is the pre-committed response to one specific
diagnostic outcome, not a default. When enabled, it applies only where
position labels exist, i.e. only the synthetic fraction of the mixed batch —
automatic, since FineWeb sequences carry no labels.

---

## 5. Experiment ladder and gates

**Status ledger.**

| Component | Status | Evidence / gate |
|---|---|---|
| MLP angle-only norm-preserving | **Landed** (`c363b57`) | Retrain-gated flag, default off; σ=1 ∀θ, no origin singularity |
| Incremental (L,A,R) transport oracle | **Landed** (`38b04a7`) | rG/rA/rR/rAsand/rY at machine precision; chunk-boundary combine |
| Weighted-boundary oracle + acceptance spec | **Landed** (`c2a6ccb`) | Teeth vs 4 boundary bugs incl. arithmetic-mean near-miss |
| √β five-site symmetrization | **Provisional** (`1f1301b`, `Gated-By: SKA-C-retrain-eval`) | Diff + tests green; `z_last→x_last`; §D α≡1 proven live-path. **Blocked on Gate 1** |
| §C retrain-ablation harness | **Landed** (`1525708`, `scripts/sqrt_beta_ablation/`) | Two-ref A/B + α probe + verdict file; decision tree verified on synthetic inputs. The Gate-1 mechanism |
| Power-iteration removal | Specced, next after Gate 1 | Exact no-op given √β (α≡1 by lower-bound direction); ships with `σ≤1+tol` assert |
| Gated-read oracle (C1–C5) | **Landed** (`01b7ce5`, `code-tests/test_gated_blend_oracle.py`) | Contractivity+teeth, gradients, init-equivalence, Cholesky-free form, learnability smoke |
| Incremental decode kernel (torch) | **Reference build landed** (`incremental_transport.py` + `test_incremental_transport_torch.py`); standalone, default-inert | Batched port verified faithful to the `38b04a7` oracle (numpy transcription, machine precision); torch parity test is the CI gate. **Not wired into live decode** — promotion needs the torch test green + decode-vs-recompute parity, behind a default-off flag. Fusion (`18r²+15r`) is a later separate step. Invariant to Gate 1 (transport is exact for any rank-1 stream) |
| Causal norm-clip (memo §6) | Specced | Own before/after eval (behavior change, not exact); **must precede Gate 2 arms** |
| Gate implementation + arms | Specced (Section 2 pins; oracle = acceptance) | **Blocked on Gates 1 and norm-clip** |
| Exact-replay training | Future | Wall-clock benchmark vs batched (rank-c form) |

**Gate 1 — the √β retrain (§C).** The transport oracle certifies the
*backend*; only a retrain certifies the *statistics change*, because both
weighting conventions are internally self-consistent and residuals cannot
distinguish them. Binding stage: the ~1M synthetic config (the paper's own
982K transfer scale) on MQAR/ToolTrace vs the asymmetric-β parent — cheap,
dispositive for the tool-calling risk. Conditional stage (only if 1M passes):
50M FineWeb two-run for a readable WikiText-103 point, carrying at least one
retrieval metric forward so a scale-dependent effect is visible. The A/B is
two git refs (parent vs `1f1301b`), identical config and seed; α is captured
on the v1.1 run as *confirmation of the §D proof under retraining*, not as a
discriminator — the α-branch of the tree is decided by proof, and the run
only has to not contradict it.

```
Gate 1 verdict tree:
retrieval (MQAR/ToolTrace) within noise of baseline?
├─ YES → LM/ppl within noise? ── YES → ✅ land √β; proceed to power-iter removal
│                             └─ NO  → convention bug the retrieval tasks
│                                       tolerated; re-audit the five sites
└─ NO  → α confirmed ≈1 (it will be, by proof) →
         the symmetrization changed the operator's *useful* spectrum, not its
         norm. Do NOT land unconditionally. Escalate: accepting a tool-call
         cost for LM gain is a design decision, not a commit.
```

The bottom node is the outcome the residual oracle is structurally blind to
and the reason the gate exists.

**Gate 2 — the readout arms.** After Gate 1 and the norm-clip land (ordering
constraint below):

| Arm | Config | Isolates |
|---|---|---|
| **ref** | √β, pure FineWeb (reuses the Gate-1 50M v1.1 run — one run, two duties) | Cost of mixing |
| **0** | Mixed data; `b_λ` trainable, `w_λ` frozen at 0 (= memo §5 exactly) | Does the *blend* + mixing preserve both capabilities? The spread of learned `b_λ` across heads is the per-head-specialization diagnostic. |
| **1** | Mixed data; both trainable | Does *input dependence* train, and does it beat Arm 0? |

Staged as before: 1M sanity (does trainable λ match fixed-λ retrieval, does
it specialize under retrieval-heavy signal), then the 50M mixed A/B — the
mixed-rescue hypothesis is only testable where dense LM signal exists.

```
Gate 2 verdict tree (λ-separation read against the 0.02 noise floor):
λ separates by token class AND retrieval ≥ Arm-0 AND ppl ≈ ref?
├─ YES → input-dependence works under mixed signal. Land Arm 1.
├─ λ ≈ init everywhere (gap ≲ 0.02) → gate not training even with LM density
│    → turn ON aux loss, re-run 1M stage. Learned-vs-imposed is now answered
│      ("not unaided"); the aux loss is the documented cost.
├─ λ separates but retrieval < Arm-0 → check direction: λ DOWN on query
│    tokens → LM gradient drowning → aux loss / answer-token upweighting;
│    if still regressed → fall back to Arm 0 (already trained; the fallback
│    is the other arm, free).
├─ λ drifts DOWN globally in EARLY training, no separation
│    → operator-bootstrap starvation (Section 4), not preference
│    → freeze b_λ for the first N steps (hold λ≈0.9 while W_k/A develop),
│      then unfreeze; w_λ trainable throughout. Aux loss stays OFF — this
│      failure is not about query-position signal.
└─ ppl regressed vs ref beyond noise → mix ratio too high; sweep down
     before touching the gate.
```

**One cross-constraint the synthesis surfaced: the causal norm-clip must land
before the Gate-2 arms train.** The gate reads the post-normalization query.
Under the current per-token ℓ2, every `z_q` is unit-norm, so the gate keys on
*direction only*; under the norm-clip, `z_q` retains magnitude below the
threshold, so the gate can also key on query norm (an expressivity gain —
fact queries plausibly run high-norm). A gate trained under ℓ2 does not
transfer to norm-clip: swapping the normalization mid-stream would hand the
gate experiment a moving input distribution and un-attribute any result. The
original sequencing already put norm-clip before the blend; this is the
reason the ordering must not be casually swapped.

---

## 6. Diagnostics and invariants (consolidated)

**Runtime asserts.** `σ(A) ≤ 1 + tol` in the read path (tol absorbing the
strict-margin FP32 slack, e.g. 1e-4). This one assert does double duty: it
guards the √β symmetry convention *and* trips on a latent chunk-boundary bug,
since a wrong boundary term produces statistics that are no longer
true-cumulative. It survives the gate unchanged (convexity). λ ∈ (0,1) holds
by construction.

**Test oracles, and the specific near-miss each has teeth against.**
Every test in the suite discriminates correct-from-plausibly-wrong, not
merely self-consistent-from-broken:

- Transport residuals `rG/rA/rR` vs independently-maintained raw `G,M,C` —
  the `rA` residual is the two-sided-transport correctness oracle; `rAsand`
  (the explicit `TAT⊤` sandwich, test-only O(r³)) localizes replay-mechanics
  bugs vs identity bugs.
- Weighted-boundary check with *per-token-varying* β at the boundary — teeth
  against the arithmetic-mean near-miss, which constant-β tests cannot see.
- `w1`-trap cross-path check — correct decode `L` matches `chol(G)` at
  1e-10; the norm-reinference bug (`w = β^{1/4}z`) desyncs past 1e-2. Guards
  the class of bug where a path is self-consistent but wrong.
- Gated-read oracle C1–C5 — C1's teeth use the *structured-β adversary*
  (same sequence, both conventions); C3 pins arm-equivalence at init
  bitwise; C5's non-separable teeth calibrate the separation noise floor.

**λ-logging spec.** On synthetic evals (position labels free from the
curricula): per-head λ statistics by token class {query, fact, distractor};
separation metric = mean λ_query − mean λ_other, read against the
**0.02 noise floor** calibrated by C5's teeth (the gate manufactures ~0.02
spurious gaps from finite-sample noise; that is what "no separation" looks
like). Early-training global-mean-λ trajectory, for the bootstrap-starvation
signature. Written to the same metrics stream the verdict logic reads.

**Review pins** (the decisions that do not get re-litigated in diffs): gate
input pre-whitening; λ never feeds β; no norm-based β re-inference anywhere
(the `.norm(` grep is the structural check — the helper pins construction,
the grep pins usage); `w`/`w1` G-increments keep the raw key; `w_λ, b_λ` off
the decay list; one new learnable at a time; provisional commits carry the
`Gated-By:` trailer and the verdict file is the merge check's input.

**Document drift-guards** (the same discipline, applied to this doc — else the
authoritative reference is the one artifact exempt from the drift-guards it
describes):

1. **Ledger updates couple mechanically to gate promotions.** The commit that
   promotes a provisional component (resolving its `Gated-By` state per the
   verdict file) MUST update this plan's §5 ledger row *in the same commit*, so
   verdict, trailer resolution, and ledger travel together and cannot diverge.
   Applies to every future promotion, not just `√β`.
2. **The pins in §1, §2, and §6 are the contract the diffs are reviewed
   against.** Edits to them are convention changes and get diff-review
   treatment (same bar as a code convention change), not drive-by edits.
   Reconciling §5 *status/hash* facts to the committed tree is maintenance, not
   a convention change — but changing a *pin* is.

---

## 7. Open questions, stated honestly

**Mixed-signal sufficiency.** C5 closed the *mechanism* half of gate
learnability: given dense supervision where the readouts disagree, plain GD
on the gate alone separates λ (gap ~0.91, gate recovers the true class
direction at |cos| = 0.98). Whether mixed LM+retrieval training *provides*
that signal at scale is the one thing only the 50M Arm-0/Arm-1 comparison
answers. The paper's own gate failure (§6.1, sparse supervision) is the
prior for skepticism; the dense-LM-rescue hypothesis is the reason to run it.

**Lag-one vs structured bindings.** The spectral filter keys on the lag-one
operator — adjacent transitions. Tool-call bindings (`Agent:Tool→Result`)
span non-adjacent structured positions. Whether persistent eigenmodes of the
lag-one operator capture multi-token bindings is architectural (inherited
from base SKA, orthogonal to everything in v1.1), untested by the paper's
clean synthetics, and the reason RULER/BABILong-class evaluation eventually
matters.

**`W_q`'s dual role.** The query projection now serves content addressing
(through `q_w`) and gate routing (through λ). Capacity exists (`w_λ` can
claim a subspace); no preemptive fix. The λ-logging detects a conflict (clean
separation *and* retrieval drop when the gate trains → routing cannibalizing
addressing); the fallback — gating on the residual stream instead of `z_q` —
is costlier and taken only on evidence.

**Exact-replay training economics.** The rank-c block form is the lever; the
decision is a wall-clock benchmark, not an assumption.

**Scale.** Everything above is validated at ≤50M-scale designs and
machine-precision oracles. 180M+ behavior of the full v1.1 stack is future
work, inheriting the paper's own scale caveat.

---

## Appendix: the one-paragraph version

Symmetrize the write weights and the whitened lag operator is provably
contractive; that one bound lets you delete spectral normalization exactly,
run an exact O(r²) incremental decode backend whose drift is pure rounding,
and blend ridge and spectral readouts per token with a convex gate that needs
no new safety machinery. The state is fixed and oracle-verified; everything
adaptive is read-side and reads only state-independent inputs, so train and
decode see the same gate. Training changes nothing structural — one sigmoid
and an axpy in the batched read, autograd backward — but λ-weights the lag
statistic's gradient, so the operator learns from the tokens that use it;
the 0.9 init protects the bootstrap, and the one uncertain question — does
mixed training supply the signal the mechanism provably suffices on — is
answered by exactly one comparison (Arm 0 vs Arm 1 at 50M), behind exactly
two gates (the √β retrain, then the arms), with every failure path
terminating in something already trained or a single pre-specified knob.
