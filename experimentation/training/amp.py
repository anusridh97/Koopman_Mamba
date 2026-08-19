"""One place that turns a model config into (autocast context, GradScaler).

Four trainers each hardcoded their own answer before this:

    training/train.py:304          bfloat16, enabled=args.bf16
    retrieval/adapt.py:179         bfloat16, enabled=args.bf16
    experiments/mqar_finetune.py   bfloat16
    experiments/table2.py:233      float16 + GradScaler, both hardcoded

Four copies of a policy is four places for it to drift, and one had already
drifted: table2.py trains in fp16 while the other three use bf16, a fact recorded
nowhere except that line. After this, the precision a trainer runs at is a
property of the config it was handed, which is also the thing hashed into the run
identity -- so "what precision did this run use?" becomes answerable from the run
directory instead of from the trainer's source.

**The GradScaler is derived, never configured.** It exists if and only if the
compute precision is fp16. "fp16 without a scaler" silently produces NaNs once
gradients underflow, and "bf16 with a scaler" is pointless overhead; neither
should be reachable, so neither is expressible -- `amp_for` takes no scaler
argument.

**`enabled` stays separate** so `--bf16/--no_bf16` keeps working. The dtype comes
from the config; the flag only switches autocast off. That is what makes this a
no-op at the defaults: bf16 config plus flag-on reproduces the bfloat16 autocast
every trainer already had.
"""
from __future__ import annotations

import contextlib
from typing import Any, Optional, Tuple

import torch

from koopman_lm.precision import autocast, needs_grad_scaler

__all__ = ["amp_for"]


def amp_for(cfg, device_type: str = "cuda",
            enabled: bool = True) -> Tuple[Any, Optional[Any]]:
    """`(autocast_context, grad_scaler_or_None)` for `cfg`.

    `cfg` is a KoopmanLMConfig; only `compute_precision` is read. The context is
    entered per step by the caller, exactly as the hardcoded one was.

    A scaler is returned only when it is both needed (fp16) and meaningful
    (autocast actually running) -- scaling gradients for an autocast that is
    disabled would inflate them for no reason.
    """
    compute = getattr(cfg, "compute_precision", "bf16")
    context = autocast(device_type, compute, enabled=enabled)
    scaler = None
    if enabled and needs_grad_scaler(compute):
        scaler = torch.amp.GradScaler(device_type, enabled=True)
    return context, scaler
