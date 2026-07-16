# Echo / SKA — Paper ↔ Code Audit (master issue list)

Branch: `claude/whitened-operator-spectral-clamp-eig01p` (== `ati-phase0-refactor`).
Paper: *Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators* (ACM_low_memory11.pdf).
Reference impl: `archive/reference/echo_jax.py` (the JAX code that produced the paper's numbers).

Legend — **Sev**: 🔴 behavior/parity bug · 🟠 paper-accuracy (code self-consistent, manuscript wrong) · 🟡 reproducibility/hygiene · ⚪ cosmetic.
**Retrain** = fixing it changes a trained model's outputs.

---

## A. Correctness / cross-implementation parity (change model behavior)

### A1 🔴 PyTorch model is missing the first/last-Mamba guard → SKA lands on the final layer  · Retrain
- JAX enforces it: `echo_jax.py` `Echo.setup` → `if i in c["ska"] and i not in (0, c["L"]-1)`.
- PyTorch drops it: `koopman_lm/models/koopman_lm.py:108-112` → `if i in ska_set:` (no exclusion), builder is 0-indexed.
- **50M** (`configs/50m.yaml`, L=16, `{3,7,11,15}`): PyTorch puts SKA on layer **15 = the last sequence layer** (violates paper §3.4 "first and last layers are always Mamba-2"). JAX drops 15 → SKA at `{3,7,11}`, last layer Mamba.
- Systematic: the `≡3 (mod 4)` spacing hits `L-1` for **every config except 180M** (50m/370m/440m/880m/1m/1p5b/3b). 180M `{8,16}`/24 is the only clean one. First layer (idx 0) is never in any list, so only the *last* is violated.
- Side effect: with the guard the 50M has **3** effective SKA layers, so the paper's "4 SKA layers at {3,7,11,15}" over-counts by one.
- Repo's PyTorch 50M results (`results/echo50m_table4/`) were trained with this extra last-layer SKA — a different architecture than the JAX/paper 50M.
- **Fix:** add the `and i not in (0, cfg.n_layers-1)` guard to `KoopmanLM` (parity with JAX), and/or a hard assertion in `config.py.__post_init__`. Decide 50M intent: `{3,7,11}` (guarded, 3 layers) vs `{2,6,10,14}` (4 layers, shifted). Do **not** apply the guard to the sub-million/Appendix-G "final two layers" path.

### A2 🔴 Koopman MLP "spectral clamp" is not norm-preserving  · Retrain
- Code clamps to the unit **disk**: `koopman_mlp.py:32-35` (and Gated `:71-74`), JAX `echo_jax.py:465-467` → `scale = clamp(radius, max=1)/radius`, i.e. `|λ| ≤ 1`.
- Paper §3.3 "Gradient preservation" claims `σ_min(R)=σ_max(R)=1`, *"exactly norm-preserving and rank-preserving,"* and uses this for the "zero rank compression beyond the LM head" argument.
- The 2×2 block is `radius·R(θ)` with `radius=√(γ²+ω²)`; σ_min=σ_max=**radius**. That equals 1 **only when radius ≥ 1** (true at init because γ inits to 1, so radius≥1→clamps to 1). Once training pulls the pair below radius 1, the block is strictly contractive (σ<1) → norm preservation lost. Verified numerically (radius 0.63 → σ=0.63; radius 0.30 → σ=0.30).
- Present in **both** JAX and PyTorch, so the paper's numbers were produced with the non-norm-preserving version.
- **Fix (changes all results):** project onto the unit **circle** — `scale = 1.0/radius` (force radius≡1) — to actually get σ_min=σ_max=1. Gate behind a config flag, default off; or soften the paper's claim to "non-expansive."

### A3 🟠 Key/query normalization is per-token ℓ2 + a learned β write-gate, not the paper's sequence-max
- Paper §3.2.1 Eq. 3 (+ Remark 5, + **Appendix E** "Sequence-Max Normalization" with Prop E.1) specifies `m = max_{s≤T} ‖z_s‖₂` and explicitly argues **against** per-token ℓ2 ("inflates low-norm noise tokens to unit norm").
- Code does exactly per-token ℓ2 **plus a learned β write-gate** (Gated-DeltaNet style): PyTorch `ska.py:486-488`, `fast.py:121-123`, `recurrent.py _proj_norm`, `chunk_stats.py`; JAX `echo_jax.py:228-274`. `chunk_stats.py` docstring: *"Replaces the leaky non-causal sequence-max normalization."*
- β appears **nowhere** in the paper (0 hits). It re-weights the sufficient statistics: `G=Σβ z zᵀ`, `M=Σβ z z_prevᵀ`, `Cv=Σβ v zᵀ` — the paper's Eqs. 4–11 have no β.
- Code is self-consistent (JAX ↔ PyTorch), so this is primarily a **manuscript** problem: §3.2.1 and Appendix E describe a scheme the trained model doesn't use.
- **Fix:** update the paper to document per-token ℓ2 + β (and drop/repurpose the sequence-max analysis), or (not recommended, breaks causality in training) implement sequence-max. No code change needed for parity.

---

## B. Whitened Koopman operator (Q1)

### B0 ✅ The live paths are CORRECT (record, not a bug)
- `core.py` (`ska_core`/`_whiten_M`), `factor_scan.py` (`ska_core_given_L`), `fast.py`, `recurrent.py` (`_ska_apply_whitened`) all compute `W = L⁻¹ M L⁻ᵀ` — matches Appendix A.3 / Remark 2. Training and decode both use it.

### B1 🔴/🟡 Dead `_post_cholesky_*` computes the **unwhitened** operator `M·G⁻¹`
- `ska.py:173-175` and `:252-254`: `A_w = (cholesky_solve(Mᵀ,L))ᵀ = M G⁻¹`. Same eigenvalues as `W` but a **different matrix** — verified ~90% different head output. This is exactly the "reversed order" form Remark 2 warns is wrong; it matches only the paper's loose main-text Eq. (11).
- Currently unreachable (`forward()` routes through `ska_core`; `chunk_strategy != 'standard'` is warned as ignored at `ska.py:340`). Landmine if the overlap/decay chunking or legacy backend is ever rewired.
- **Fix:** delete `_post_cholesky_*` + the `_get_chunk_stats` overlap/decay route, or hard-`assert` them unreachable.

### B2 🟡 Newton–Schulz path uses the symmetric gauge `G^{-1/2} M G^{-1/2}`
- `core.py:166` `ska_core_ns`. Same eigenvalues, different matrix; only *exactly* equal to the Cholesky-whitened operator when the spectral clamp is inactive (its own docstring says so). Forward-only. Flag so nobody profiles it as an exact substitute for the clamped operator.

### B3 🟠 Paper self-inconsistency: Eq. (11) `A_w = M G⁻¹` vs Appendix A.3 `A_w = L⁻¹ M L⁻ᵀ`
- Fix Eq. (11) (or annotate it as the *similar-but-not-implemented* form) so it doesn't invite the B1 bug.

---

## C. Power-filter order K for the 180M LM test (Q2)

### C1 🟡 K is not pinned for 180M and the 180M training script isn't committed
- `ska_power_K` defaults to **2** (`globals/config.py:36`); **not** set in `configs/180m.yaml` (only `1m.yaml` pins it). `train.py` has **no** `--ska_power_K` flag. No `slurm_pretrain_180m*.sh` in `scripts/`.
- Eval **does** prefer the checkpoint's embedded `cfg` (`evaluate.py:71-74`, `lm_harness_eval.py:80-84`), falling back to `build_config('180m')`→2 only if `meta.pt` is missing. So the run was K=2 unless launched with a custom YAML.
- **Fix:** add `ska_power_K: 2` to `180m.yaml`; commit the actual 180M launch script; confirm the shipped checkpoint via `torch.load('.../final/meta.pt')['cfg'].ska_power_K`.

---

## D. Scale-parameter (γ / η) range discrepancies

### D1 🟠 γ range disagrees three ways
- Paper §3.2.3: **γ ∈ [1.0, 1.5]** ("restoring variance", ≥1).
- JAX default: `gamma_min=0.5, gamma_max=1.5, init=0.7` (`echo_jax.py:250-251`) — γ can **damp below 1**. (Its own header comment line 13 says "[1,1.5]", contradicting the default.)
- PyTorch **50M** (`50m.yaml`): `gamma_bounds [0.5,1.5]`, init 0.7 (matches JAX, not the paper).
- PyTorch **180M** (config.py defaults): `gamma_clamp (1.0,1.5)`, learnable (matches paper, **not** JAX). → 50M and 180M use **different** γ policies.
- **Fix:** pick one policy; reconcile the paper text and both configs.

### D2 ⚪ η range differs between scales
- JAX / 50M: η squashed to `[1.4,1.7]`, init 1.5. 180M (config defaults): η **unconstrained** learnable, init 1.5. Paper Appendix G: η=1.5 (standard) / 2.5 (strong). Minor, but another 50M-vs-180M inconsistency.

---

## E. Reproducibility / hygiene

### E1 ⚪ Spectral-norm iteration count doesn't match the paper
- Active `core.py:23 _spec_w` uses **20** power iterations; paper says **6**; the dead `ska.py:_spectral_normalize_power_iter` uses 6 (on the wrong operator). Converges either way — cosmetic — but pick one and match the text. Two divergent spec-norm impls exist.

### E2 ⚪ Ridge ε in the exact-intrachunk path is 1.1e-3, not 1e-3
- `ska.py:504` / `factor_scan.all_prefix_chol` seed with `ridge_eps + 1e-4`. Paper ε=1e-3. Tiny numerical deviation; only the `exact_intrachunk` path.

### E3 🟡 Checkpoints embed full `cfg` (good) but it's easy to miss
- `train.py:240` stores `cfg`; consider also dumping the resolved config (esp. `ska_power_K`, `ska_layer_indices`) to a human-readable JSON beside `model.pt`, so K / layer layout are verifiable without unpickling.

---

## Suggested fix batches
- **Safe now (no retrain):** B1 (guard/remove dead op), C1 (pin K + commit script), B3/A3/D1 (paper text), E1/E2/E3.
- **Retrain-gated (changes the model), decide explicitly:** A1 (first/last guard + 50M indices), A2 (circle-projection MLP), D1 (unify γ policy).
