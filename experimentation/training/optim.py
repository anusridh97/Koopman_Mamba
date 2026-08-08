"""
optim.py -- Shared AdamW parameter-group policy.

Extracted from training/train.py so experiment scripts (table2.py,
mqar_finetune.py) can apply the same decay/no-decay split instead of each
re-deriving (or, historically, skipping) it. Kept dependency-light on
purpose: importing this module must not require transformers, DDP, or any
GPU-only extras, so CPU-only experiment scripts and tests can use it freely.
"""


def param_groups(raw_model, weight_decay):
    """Return standard AdamW decay/no-decay groups.

    A flat ``model.parameters()`` call decays every parameter unless a rare
    Koopman-v2 option is enabled. That includes norm scales, biases,
    embeddings, Mamba state-space parameters, LayerScale, and
    spectral/geometry variables. At long schedules ``weight_decay=0.1`` (or
    even table2.py's 0.01) can shrink those parameters by several-fold even
    before gradients are considered. Matrix weights are decayed; state,
    scale, bias, embedding, and explicitly geometric parameters are not.
    """
    fn = getattr(raw_model, 'no_weight_decay_param_names', None)
    explicit_skip = set(fn() if fn is not None else ())
    special_leaves = {
        # Mamba state-space / discretization parameters.
        'A_log', 'D', 'dt_bias',
        # Koopman/SKA geometry and residual controls (some are >1D).
        'lift_v', 'lift_g', 'A_raw', 's', 'theta', 'gamma', 'omega',
        'eta', 'eta_raw', 'ssn_gamma', 'layerscale_gate', 'short_conv_gate',
    }

    decay, no_decay = [], []
    seen = set()
    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        leaf = name.rsplit('.', 1)[-1]
        is_embedding = name == 'embed.weight' or '.embed.' in name or 'embedding' in name
        skip = (
            name in explicit_skip
            or p.ndim < 2
            or leaf == 'bias'
            or leaf in special_leaves
            or is_embedding
            or bool(getattr(p, '_no_weight_decay', False))
        )
        (no_decay if skip else decay).append(p)

    groups = []
    if decay:
        groups.append({'params': decay, 'weight_decay': weight_decay})
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups
