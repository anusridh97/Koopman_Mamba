# Koopman MLP v2 — utilization-by-construction (Aurora-inspired)

Aurora's outcome was *denser* models — every neuron contributing, so the model
is more data-efficient — achieved on the optimizer side (uniform row norms on the
**update**). The architectural analog implemented here makes the Koopman MLP's
**structure** enforce uniform utilization, and fixes the frozen-decay pathology,
by construction. Four changes, each targeting a measured problem. All are opt-in;
the defaults reproduce v1 exactly (same parameter names and shapes, so existing
checkpoints still load).

Config: [`configs/180m_v2.yaml`](configs/180m_v2.yaml) (registered as `180m_v2`).
Code: [`koopman_lm/globals/modules/koopman_mlp.py`](koopman_lm/globals/modules/koopman_mlp.py).

## 1. Row-normalized lift with explicit gains — the direct Aurora analog

Aurora enforces uniform row norms on the *update*; we enforce uniform row norms
on the *weights*. The lift is parameterized as weight-normalized rows times a
learned per-row scalar gain:

```
W_lift[i, :] = g_i · v_i / ‖v_i‖
```

Every lifted dimension then gets equal geometric leverage from the input, and
"how much does this neuron matter" collapses into one interpretable scalar `g_i`
per row — which you can read, histogram, and regularize. Dead neurons become
visible as `g_i → 0` rather than hiding in row-norm drift. This is WeightNorm,
applied here as the architectural counterpart of Aurora's row-uniformity
constraint. Gains initialize to a single constant (the expected xavier row norm),
so utilization starts perfectly uniform.

`mlp_row_norm_lift: true`. The scale-invariant direction `v` (and the gains `g`)
**skip weight decay** — see `KoopmanLM.no_weight_decay_param_names`, wired into
the optimizer in `train.py::_param_groups`. Without that, decay drives `‖v‖ → 0`
and the `1/‖v‖` gradient blows up.

## 2. (log ρ, θ) reparameterization — fixes the frozen-ω pathology

Replace the learned `(γ, ω)` with a decoupled decay/angle pair:

```
γ = e^{−softplus(s)} · cos θ,   ω = e^{−softplus(s)} · sin θ,   ρ = e^{−softplus(s)} ∈ (0, 1]
```

Decay rate and rotation angle are now separate parameters with comparable
gradient scales; the `ρ ≤ 1` (non-expansive) constraint is built in smoothly (no
clamp, no kink). `∂L/∂θ` gets the full chain rule at whatever ρ is, instead of
being the small antisymmetric-wedge component it is in raw `(γ, ω)` coordinates.

`mlp_rotation_param: logrho_theta`. With `mlp_decay_depth_grade: true` the `s`
init is graded by depth — aggressive decay early (ρ ≈ 0.5), gentle late
(ρ ≈ 0.97) — the profile the model tends to learn anyway, handed to it at init.

## 3. Break the arbitrary pairing

The 2×2 blocks otherwise yoke dimensions `(2i, 2i+1)` by reshape order forever.
An orthogonal mixer inserted between the lift and the rotation lets the network
choose which coordinates get paired. It is block-diagonal (block size 64) to keep
the cost bounded — a dense `d_k × d_k` mix would add ~100M params/buffer across
24 layers; block-diagonal adds ~3M (learnable) or 0 (fixed). Every mode is exactly
orthogonal, so it preserves the lifted L2 norm and composes with the
norm-preserving rotation.

`mlp_pair_mixer: learned` (block-diagonal Cayley, adaptive) — or `ortho` (fixed
random orthogonal, zero trained params) / `perm` (fixed permutation, cheapest).

## 4. The diagnostic that makes it publishable

Aurora's credibility came from *measuring* the pathology (25% dead neurons by
step 500), not asserting it. The analog:
[`koopman_mlp_diag.py`](koopman_lm/globals/modules/koopman_mlp_diag.py) measures

* the **`g_i` histogram** and its coefficient of variation (lower CV = tighter,
  more uniform utilization), and
* the **dead-pair fraction** — per-pair activation variance in the lifted space.

Both are defined identically for v1 and v2, so the comparison is apples-to-apples.

```bash
# side-by-side v1 vs v2 (fresh init, or add --ckpt DIR for a trained model)
python scripts/koopman_utilization_report.py \
    --config configs/180m.yaml --compare configs/180m_v2.yaml
```

Even at init this shows the mechanism: the gain CV drops from the xavier row-norm
spread (v1) to ~0 (v2, uniform by construction). Tracked over training, v1 vs v2,
it turns "we redesigned the MLP" into "we identified an under-utilization
pathology, fixed it structurally, and show the utilization distribution
tightening."

## Training

```bash
python -m koopman_lm.training.train --model_size 180m_v2   # + your usual flags
```

The optimizer automatically excludes the WeightNorm directions and geometric
rotation/mixer parameters from weight decay. Nothing else changes; v1
(`--model_size 180m`) trains byte-for-byte as before.
