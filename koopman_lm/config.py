from dataclasses import dataclass, field, fields, replace, asdict
from typing import Optional, Tuple
import hashlib
import json
from pathlib import Path

import yaml


@dataclass(frozen=True)
class KoopmanLMConfig:
    """Immutable model config.

    Frozen so a config can't be mutated after creation (configs are content-
    addressed via ``config_hash`` and stored in every checkpoint). To derive a
    variant, use ``dataclasses.replace(cfg, field=value)`` — NOT attribute
    assignment, which raises ``FrozenInstanceError``. Runtime adjustments
    (e.g. ``vocab_size`` from the tokenizer, ``max_seq_len`` from CLI args) must
    therefore go through ``replace`` at the call site.
    """
    # 180M model, Nemotron-H layout ratios
    d_model: int = 768          # multiple of 64 (tensor core aligned)
    n_layers: int = 24
    vocab_size: int = 32000     # Llama-2 / Mistral tokenizers are both 32000

    # Mamba-2 config
    d_state: int = 128          # multiple of 8
    d_conv: int = 4
    mamba_expand: int = 2
    # Mamba-2 head width. None keeps mamba_ssm's own default (64).
    # Must be set explicitly at small d_model: mamba_ssm packs z/x/B/C/dt into
    # one in_proj of width 2*d_inner + 2*ngroups*d_state + nheads, and the
    # channel-last causal_conv1d kernel requires that width to be a multiple
    # of 8. At d_model=64 the default headdim=64 gives nheads=2 and width 290,
    # which the kernel rejects at runtime; headdim=16 gives nheads=8 and 296.
    mamba_headdim: int | None = None

    # SKA config
    ska_n_heads: int = 12
    ska_rank: int = 48          # multiple of 16
    ska_ridge: float = 1e-3
    # DEAD FIELD. Kept only so config_hash stays stable for the 11 registry
    # configs (see identity_baseline.json). It was the legacy eta-init, and it
    # is now unreachable twice over: no YAML in configs/ sets it, and its only
    # reader -- ska_block.py's `scale=cfg.ska_scale` -> SKAModule(scale=...) --
    # threads it into a parameter that SKAModule's body never touches (eta is
    # built from eta_value). Do not use it; do not delete it without a
    # deliberate identity-baseline regeneration.
    ska_scale: float = 1.5
    ska_power_K: int = 2
    ska_chunk_size: int = 64    # multiple of 8
    ska_backend: str = 'auto'

    # --- SKA scale-parameter policy ---
    #
    # These defaults ARE the modern policy: fixed-scalar eta/gamma, no clamp.
    # They used to encode the superseded regime (learnable eta init 1.5,
    # learnable gamma clamped to [1.0, 1.5]), which every production YAML then
    # overrode in full -- so "fall back to the defaults" was a trap rather than
    # a sane baseline, and the sweep search compensated by re-pinning all eight
    # fields into every trial. The five configs that predate this policy now
    # state the old values explicitly (1m, 370m, 180m_dense, 180m_gated,
    # 180m_v2), so inheriting by omission is finally safe.
    ska_eta_learnable: bool = False
    ska_eta_value: float = 1.0
    # Smooth sigmoid-squash bounds (echo_jax.py parity), distinct from the
    # hard torch.clamp ska_gamma_clamp below: squash keeps a nonzero gradient
    # at the boundary, clamp kills it. None preserves prior (clamp/fixed)
    # behavior; set both to reproduce the original paper-faithful regime
    # (eta in [1.4, 1.7] init 1.5, gamma in [0.5, 1.5] init 0.7).
    ska_eta_bounds: Optional[tuple] = field(default=None, metadata={"coerce": tuple})
    ska_gamma_learnable: bool = False
    ska_gamma_clamp: Optional[tuple] = field(default=None, metadata={"coerce": tuple})
    ska_gamma_bounds: Optional[tuple] = field(default=None, metadata={"coerce": tuple})
    ska_gamma_value: float = 1.0

    # Residual injection. On by default (part of the modern policy above).
    ska_layerscale: bool = True
    ska_layerscale_init: float = 1e-4
    ska_out_proj_std: float = 0.02

    # --- SKA short-range conv path ---
    ska_short_conv: bool = False
    ska_short_conv_kernel: int = 4
    ska_short_conv_gate_init: float = 1e-2

    # --- Exact intra-chunk causal stats ---
    ska_exact_intrachunk: bool = False

    # --- Small-rank exact per-token path (inverse-Cholesky representation) ---
    # Replaces chunk-64 stats + cross-chunk boundary AND the factor-scan exact
    # path: exact per-token exclusive-prefix stats, one batched Cholesky over
    # all prefixes, whitened core as pure matmul against P = L^{-1}, no
    # spectral power iteration (sqrt-beta symmetric keys => contractive A_w).
    # Per-token stats are (B,T,H,r,r): use with a SMALLER ska_rank (<= 32
    # recommended, hard cap 64). Takes precedence over ska_exact_intrachunk.
    ska_inverse_cholesky: bool = False

    # --- Exact two-level prefix scan ---
    # The associative scan runs over raw segment sufficient statistics at block
    # boundaries; each block then executes the exact per-token recurrence with
    # O(r^2) rank-1 Cholesky updates.  This is the recommended mathematical
    # path for quality runs and supersedes the materialized per-prefix
    # inverse-Cholesky implementation when enabled.
    ska_prefix_scan: bool = False
    ska_prefix_scan_block_size: int = 32
    ska_prefix_scan_jitter: float = 0.0

    # --- Causal key/query normalization (memo §6) ---
    # False: per-token L2 (unit norm; legacy). True: causal norm-CLIP
    #   k <- k / max(1, ||k||/c) -- bounds leverage, causal, does NOT inflate
    #   low-norm distractors to unit norm (Appendix E Remark 5 warns L2 does).
    #   Behavior change (not exact) -> its own before/after eval; MUST precede
    #   the Gate-2 gate arms (the gate reads the post-normalization query, so a
    #   gate trained under L2 does not transfer to clip). Default ON, matching
    #   what every production YAML already sets; the five configs that predate
    #   the policy pin `ska_norm_clip: false` explicitly.
    ska_norm_clip: bool = True
    ska_norm_clip_c: Optional[float] = None   # threshold c; None -> sqrt(rank)

    # SKA adaptive chunking
    ska_chunk_strategy: str = 'standard'
    ska_overlap_fraction: float = 0.5
    ska_decay_alpha: float = 0.95

    # Koopman MLP config
    # ``auto`` preserves the legacy ``mlp_gated`` switch.  New quality runs
    # should set this explicitly so an architecture ablation cannot silently
    # change when a legacy boolean is copied between YAMLs.
    mlp_type: str = 'auto'      # auto | koopman | koopman_gated | swiglu
    mlp_expand: float = 2.667
    mlp_spectral_norm: bool = True
    # False: eigenvalue modulus clamped to the unit DISK (|lambda|<=1, non-expansive,
    #   legacy/back-compat). True: projected onto the unit CIRCLE (sigma_min=sigma_max=1),
    #   an exact norm-preserving rotation (paper S3.3 "Gradient preservation").
    #   Changes the forward pass -> only meaningful for a fresh training run.
    mlp_norm_preserving: bool = False
    mlp_gated: bool = False

    # --- Koopman MLP utilization / structure options (Aurora-inspired v2) ---
    # All default to reproducing v1 exactly (param names/shapes unchanged).
    # See modules/mlp/koopman.py for the full rationale.
    #
    # (1) Row-normalized lift with explicit per-row gains (WeightNorm over rows):
    #     W[i,:] = g_i * v_i / ||v_i||. "How much a neuron matters" collapses to
    #     one scalar g_i; dead neurons surface as g_i -> 0. Requires the WeightNorm
    #     direction params to skip weight decay (KoopmanLM.no_weight_decay_param_names).
    mlp_row_norm_lift: bool = False
    # (2) Rotation parameterization. None -> back-compat (resolve from
    #     mlp_norm_preserving: True->'angle', False->'legacy'). Explicit:
    #     'legacy' (learned gamma,omega disk-clamped), 'angle' (cos/sin, rho=1),
    #     'logrho_theta' (gamma=e^-softplus(s) cos t, omega=e^-softplus(s) sin t:
    #     decay rate and angle decoupled, |lambda|<=1 built in smoothly).
    mlp_rotation_param: Optional[str] = None
    # Depth-grade the logrho_theta decay init (aggressive decay early, gentle late).
    mlp_decay_depth_grade: bool = False
    # (3) Orthogonal pair mixer between lift and rotation. None/'none' -> off.
    #     'perm' (fixed random permutation), 'ortho' (fixed block-diag random
    #     orthogonal), 'learned' (learnable block-diag Cayley orthogonal).
    mlp_pair_mixer: Optional[str] = None
    mlp_mixer_block: int = 64

    # Shared residual-block normalization.  The original implementation
    # hard-coded LayerNorm separately in Mamba, SKA, the MLP, and norm_f.
    norm_type: str = 'layernorm'   # layernorm | rmsnorm
    norm_eps: float = 1e-5

    # SKA can replace a Mamba sequence mixer (paper layout) or be added as a
    # sparse parallel memory adapter while retaining Mamba's local recurrence.
    # ``parallel`` is the recommended quality-first mode.
    ska_mode: str = 'replace'      # replace | parallel

    # Initialization policy.  ``mamba_safe`` preserves constructor-specific
    # Mamba/SKA initialization and only initializes embeddings globally, then
    # depth-scales residual-output projections.  ``legacy`` reproduces the old
    # blanket normal_(0, .02) pass over every nested Linear.
    init_policy: str = 'legacy'      # mamba_safe | legacy
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True

    # Layer layout
    ska_layer_indices: Optional[Tuple[int, ...]] = field(
        default=None, metadata={"coerce": tuple})

    # Precision policy (specs/2026-08-10-precision-policy-design.md).
    # compute_precision sets the floor for the whole network; the two component
    # fields RAISE it where the math needs more. Their domains make that
    # invariant unsatisfiable to violate rather than something to validate:
    # compute excludes fp64, and the component fields exclude bf16/fp16, so the
    # floor can never exceed the component minimum. See koopman_lm/precision.py.
    #
    # The defaults state what the code already did -- bf16 autocast, an fp32 SKA
    # core, an untouched MLP path -- so adding them changed no numerics. They do
    # change config_hash for all 11 registry configs, which is why
    # identity_baseline.json was regenerated alongside them.
    compute_precision: str = 'bf16'            # fp32 | bf16 | fp16
    ska_precision: str = 'fp32'                # fp32 | fp64
    mlp_precision: Optional[str] = None        # None (no-op) | fp32 | fp64

    # Training
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.ska_layer_indices is None:
            object.__setattr__(self, "ska_layer_indices", (4, 8, 12, 16, 20, 23))

        # Normalize every field declaring metadata={"coerce": ...}. YAML and JSON
        # have no tuple type, so a deserialized config always arrives with lists
        # here -- including the spec.yaml the run system writes and reads back on
        # each launch (run/spec.py:181 -> run/resolve.py:219). Coercing is what
        # makes frozen=True a real guarantee (a list field is still mutable in
        # place), keeps __eq__/__hash__ working, and lets
        # asdict -> yaml -> KoopmanLMConfig(**raw) round-trip to an EQUAL config.
        # Declared per field rather than listed here so a new collection field
        # opts in at its own declaration and cannot be silently forgotten.
        for f in fields(self):
            conv = f.metadata.get("coerce")
            if conv is None:
                continue
            value = getattr(self, f.name)
            if value is not None:
                object.__setattr__(self, f.name, conv(value))

        valid_mlp = {'auto', 'koopman', 'koopman_gated', 'swiglu'}
        if self.mlp_type not in valid_mlp:
            raise ValueError(f"mlp_type={self.mlp_type!r}; expected one of {sorted(valid_mlp)}")
        if self.ska_mode not in {'replace', 'parallel'}:
            raise ValueError("ska_mode must be 'replace' or 'parallel'")
        if self.norm_type.lower().replace('_', '') not in {'layernorm', 'ln', 'rmsnorm', 'rms'}:
            raise ValueError("norm_type must be layernorm or rmsnorm")
        if self.init_policy not in {'mamba_safe', 'legacy'}:
            raise ValueError("init_policy must be mamba_safe or legacy")

        # --- precision policy -------------------------------------------
        # Imported here rather than at module scope to keep config.py free of a
        # torch import: it is loaded by tooling (gen_inventory, the sweep
        # expander) that has no reason to pay for torch.
        from koopman_lm.precision import COMPONENT_PRECISIONS, COMPUTE_PRECISIONS

        if self.compute_precision not in COMPUTE_PRECISIONS:
            raise ValueError(
                f"compute_precision={self.compute_precision!r}; expected one of "
                f"{sorted(COMPUTE_PRECISIONS)}. fp64 is deliberately excluded: it "
                f"would let the global floor exceed what ska_precision/"
                f"mlp_precision can raise to, which is the invariant those "
                f"fields rely on (see koopman_lm/precision.py).")
        if self.ska_precision not in COMPONENT_PRECISIONS:
            raise ValueError(
                f"ska_precision={self.ska_precision!r}; expected one of "
                f"{sorted(COMPONENT_PRECISIONS)}. The whitened core takes a "
                f"cholesky of a Gram matrix (kernels/ska_operator.py), and "
                f"bf16's 8 mantissa bits make that unreliable -- part of why "
                f"ska_ridge exists. This field protects that computation, so it "
                f"cannot be used to lower its precision.")
        if self.ska_precision == 'fp64' and self.ska_backend == 'cuda_prefix':
            raise ValueError(
                "ska_precision='fp64' is incompatible with "
                "ska_backend='cuda_prefix': the fused kernel is fp32-only, "
                "enforced in kernels/csrc/prefix_scan_ext.cu (CHECK_F32) and "
                "again in kernels/cuda_prefix_scan.py. Use ska_backend='auto' or "
                "'pytorch' for an fp64 core -- the exact prefix-scan path accepts "
                "float64.")
        if self.mlp_precision is not None and self.mlp_precision not in COMPONENT_PRECISIONS:
            raise ValueError(
                f"mlp_precision={self.mlp_precision!r}; expected None or one of "
                f"{sorted(COMPONENT_PRECISIONS)}. None is the default and a "
                f"strict no-op: the MLP path is left untouched.")
        if self.ska_prefix_scan and self.ska_inverse_cholesky:
            raise ValueError(
                "ska_prefix_scan and ska_inverse_cholesky are alternative exact paths")
        if self.ska_prefix_scan and self.ska_exact_intrachunk:
            raise ValueError(
                "ska_prefix_scan and ska_exact_intrachunk are alternative exact paths")
        if len(set(self.ska_layer_indices)) != len(self.ska_layer_indices):
            raise ValueError("ska_layer_indices must not contain duplicates")
        if any(i < 0 or i >= self.n_layers for i in self.ska_layer_indices):
            raise ValueError(
                f"ska_layer_indices={self.ska_layer_indices} must lie in [0, {self.n_layers})")

        # % 16 (not 64) allows the paper's d=96 while still ensuring reasonable alignment.
        # Production runs (d≥448) are always multiples of 64 via their YAML configs.
        assert self.d_model % 16 == 0, \
            f"d_model={self.d_model} must be a multiple of 16"
        assert self.d_state % 8 == 0, \
            f"d_state={self.d_state} must be a multiple of 8"
        assert self.ska_chunk_size % 8 == 0, \
            f"ska_chunk_size={self.ska_chunk_size} must be a multiple of 8"
        assert self.ska_prefix_scan_block_size > 0, \
            "ska_prefix_scan_block_size must be positive"
        assert self.ska_prefix_scan_block_size % 8 == 0, \
            "ska_prefix_scan_block_size must be a multiple of 8"
        # % 8 (not 16) allows the paper's rank=24 (24 % 8 == 0).
        assert self.ska_rank % 8 == 0, \
            f"ska_rank={self.ska_rank} must be a multiple of 8"
        assert self.d_model % self.ska_n_heads == 0, \
            f"d_model={self.d_model} must be divisible by ska_n_heads={self.ska_n_heads}"

    @property
    def head_dim(self):
        return self.d_model // self.ska_n_heads

    @property
    def resolved_mlp_type(self):
        if self.mlp_type != 'auto':
            return self.mlp_type
        return 'koopman_gated' if self.mlp_gated else 'koopman'

    def param_count_estimate(self):
        d = self.d_model
        V = self.vocab_size
        n = self.n_layers
        n_ska = len(self.ska_layer_indices)
        n_mamba = n if self.ska_mode == 'parallel' else n - n_ska

        embed = V * d * (1 if self.tie_embeddings else 2)

        # mamba_ssm.Mamba2's REAL parameter layout (koopman_lm never overrides
        # ngroups, so it is always mamba_ssm's own default of 1; headdim
        # likewise falls back to mamba_ssm's default of 64 unless
        # mamba_headdim pins it). The previous formula here
        # (d*d_inner*2 + d_inner*d_state*2 + d_inner*d_conv + d_inner + d_inner*d)
        # was never checked against mamba_ssm.Mamba2's actual __init__ and
        # both mis-sized in_proj (its true width is 2*d_inner +
        # 2*ngroups*d_state + nheads, not a plain 2*d_inner) and omitted
        # Mamba2's own internal gated RMSNorm (RMSNormGated(d_inner), weight
        # only) entirely -- a 1.4% (731k-parameter) overcount on 50m,
        # confirmed against a real GPU instantiation (build 415208:
        # 50,034,044 measured vs 50,765,216 estimated). Mamba2Block's own
        # separate wrapping pre-norm is counted in `norms` below, not here.
        d_inner = d * self.mamba_expand
        headdim = self.mamba_headdim if self.mamba_headdim is not None else 64
        nheads = d_inner // headdim
        ngroups = 1
        d_in_proj = 2 * d_inner + 2 * ngroups * self.d_state + nheads
        conv_dim = d_inner + 2 * ngroups * self.d_state
        per_mamba = (
            d * d_in_proj +                        # in_proj (bias=False)
            conv_dim * self.d_conv + conv_dim +    # conv1d: depthwise weight + bias
            3 * nheads +                           # dt_bias + A_log + D
            d_inner +                              # internal RMSNormGated(d_inner)
            d_inner * d                            # out_proj (bias=False)
        )
        mamba_total = per_mamba * n_mamba

        per_ska = (
            d * self.ska_n_heads * self.ska_rank * 2 +
            d * self.ska_n_heads * self.head_dim +
            self.ska_n_heads * self.head_dim * d +
            d * self.ska_n_heads + self.ska_n_heads +
            (self.d_model if self.ska_layerscale else 0) +
            (d * (self.ska_short_conv_kernel + 1) + d
             if self.ska_short_conv else 0) +
            # eta/gamma are each a single learnable scalar ONLY when their
            # *_learnable flag is set (config.py's own scale-parameter
            # policy) -- the unconditional "+ 2" here counted them even when
            # both are fixed (as every current production YAML does).
            (1 if self.ska_eta_learnable else 0) +
            (1 if self.ska_gamma_learnable else 0)
        )
        ska_total = per_ska * n_ska

        d_k = ((int(d * self.mlp_expand) + 63) // 64) * 64
        mlp_kind = self.resolved_mlp_type
        n_mlp_proj = 3 if mlp_kind in ('swiglu', 'koopman_gated') else 2
        per_mlp = d * d_k * n_mlp_proj                    # lift + readout (+ gate)
        if mlp_kind != 'swiglu':
            # (2) rotation params: legacy/logrho_theta store d_k coeffs
            #     (gamma+omega, or s+theta); angle stores d_k/2 (theta only).
            rp = self.mlp_rotation_param or ('angle' if self.mlp_norm_preserving else 'legacy')
            per_mlp += (d_k // 2) if rp == 'angle' else d_k
            if self.mlp_row_norm_lift:
                per_mlp += d_k
            if self.mlp_pair_mixer == 'learned':
                b = min(self.mlp_mixer_block, d_k)
                per_mlp += d_k * b
        mlp_total = per_mlp * n

        # Every layer's sequence mixer and its MLP each own a pre-norm (d
        # params): 2 per layer, plus the final norm. In 'parallel' mode the
        # n_ska layers get a THIRD: MambaSKAParallelBlock = Mamba2Block(cfg)
        # + SKABlock(cfg) run side by side, each with its own separate
        # wrapping norm, on top of that layer's usual MLP norm. 'replace'
        # mode has no such layer (SKA swaps in for Mamba 1-for-1, so every
        # layer keeps exactly 2 norms) -- this term was previously flat
        # regardless of ska_mode.
        extra_parallel_norms = n_ska * d if self.ska_mode == 'parallel' else 0
        norms = n * d * 2 + extra_parallel_norms + d

        total = embed + mamba_total + ska_total + mlp_total + norms
        return total


# ============================================================================
# Config hashing
# ============================================================================

def config_hash(cfg: KoopmanLMConfig) -> str:
    """Stable SHA-256 hex digest of a config's contents."""
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Layer index helpers
# ============================================================================

def _evenly_spaced_indices(n_layers, n_special):
    if n_special == 0:
        return []
    if n_special >= n_layers:
        return list(range(n_layers))
    step = n_layers / n_special
    indices = [int(round((i + 1) * step)) - 1 for i in range(n_special)]
    indices = sorted(set(min(idx, n_layers - 1) for idx in indices))
    return indices


# ============================================================================
# YAML / JSON loader
# ============================================================================

_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"

CONFIG_REGISTRY = {
    # Synthetic-transfer scale (paper Sec 4.1 / Table 2).
    "1m": "1m.yaml",
    # Canonical fused-prefix production configurations.
    "50m": "50m.yaml",
    "180m": "180m.yaml",
    # 768x24 backbone that "180m" used to mean before it was overwritten by
    # the 640x25 prefix-scan design above; 180m_gated and 180m_v2 are variants
    # of THIS config, not of the current "180m". Kept under its own name so
    # neither lineage overwrites the other -- see configs/180m_dense.yaml.
    "180m_dense": "180m_dense.yaml",
    "180m_gated": "180m_gated.yaml",
    "180m_v2": "180m_v2.yaml",
    # Scaling ladder.
    "370m": "370m.yaml",
    "440m": "440m.yaml",
    "880m": "880m.yaml",
    "1p5b": "1p5b.yaml",
    "3b": "3b.yaml",
}


def load_config(path) -> KoopmanLMConfig:
    """Load a KoopmanLMConfig from a YAML or JSON file.

    YAML lists are normalised to tuples by __post_init__, so ska_layer_indices
    and ska_gamma_clamp round-trip cleanly.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    # Strip YAML anchor helper keys not present in the dataclass
    data.pop("production", None)
    return KoopmanLMConfig(**data)


def build_config(model_size: str) -> KoopmanLMConfig:
    """Resolve a model size name or file path to a KoopmanLMConfig.

    Accepts:
      "440m"                    → looks up CONFIG_REGISTRY → loads YAML
      "path/to/custom.yaml"     → loads directly (one-off sweep configs)
    """
    p = Path(model_size)
    if p.suffix in (".yaml", ".yml", ".json"):
        return load_config(p)
    if model_size not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model_size: {model_size!r}. "
            f"Known: {sorted(CONFIG_REGISTRY)}"
        )
    return load_config(_CONFIGS_ROOT / CONFIG_REGISTRY[model_size])


# Backward-compat shim: callers that do `factory()` still work.
# Values are zero-arg callables (lambdas) returning KoopmanLMConfig, not strings.
CONFIG_FACTORIES = {name: (lambda n=name: build_config(n)) for name in CONFIG_REGISTRY}
