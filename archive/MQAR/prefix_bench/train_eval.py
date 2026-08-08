"""
train_eval.py -- shared online-training + evaluation harness for the prefix-mask
benchmarks. Generates fresh random batches each step (synthetic tasks) and evaluates
on a fixed held-out batch.

The conditioning handed to each model depends on its kind:
  * "ska"   receives prefix_weights  = sample_weights_from_mask(prefix_mask, ska_mode)
  * "attn"  receives attn_bias       = attn_bias_for_mode(seg_ids, attn_mode)
  * all     receive seg_ids          (for the shared segment embedding)
"""

import math
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefix_bench.prefix_masks import sample_weights_from_mask, attn_bias_for_mode
from prefix_bench.csp import segment_causal_bias, SKA_RELEASE_MODE, ATTN_SEGMENT_MODE

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]


def build_conditioning(kind: str, ska_mode: str, attn_mode: str,
                       prefix_mask: torch.Tensor, seg_ids: torch.Tensor,
                       soft_weight: float, release_grp: Optional[torch.Tensor] = None):
    """Returns (prefix_weights, attn_bias, release_grp) for the given model kind."""
    prefix_weights, attn_bias, grp_out = None, None, None
    if kind == "ska":
        if ska_mode == SKA_RELEASE_MODE:
            grp_out = release_grp  # SKA uses per-token release groups (csp)
        else:
            prefix_weights = sample_weights_from_mask(prefix_mask, ska_mode, soft_weight)
    elif kind == "attn":
        if attn_mode == ATTN_SEGMENT_MODE and release_grp is not None:
            attn_bias = segment_causal_bias(release_grp)
        else:
            attn_bias = attn_bias_for_mode(seg_ids, attn_mode)
    return prefix_weights, attn_bias, grp_out


def masked_ce(logits: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor):
    loss_all = F.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view_as(y)
    return (loss_all * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def _release_grp(meta, device):
    grp = meta.get("release_grp") if isinstance(meta, dict) else None
    return grp.to(device) if torch.is_tensor(grp) else None


@torch.no_grad()
def evaluate(model, kind, eval_batch: Batch, ska_mode, attn_mode, soft_weight, device):
    model.eval()
    x, y, lm, pm, seg, meta = eval_batch
    x, y, lm, pm, seg = [t.to(device) for t in (x, y, lm, pm, seg)]
    grp = _release_grp(meta, device)
    pw, ab, rg = build_conditioning(kind, ska_mode, attn_mode, pm, seg, soft_weight, grp)
    logits = model(x, prefix_weights=pw, attn_bias=ab, seg_ids=seg, release_grp=rg)
    preds = logits.argmax(dim=-1)
    hit = ((preds == y) & (lm > 0)).sum().item()
    tot = (lm > 0).sum().item()
    return hit / max(tot, 1)


def train_and_eval(
    model: nn.Module,
    kind: str,
    sample_fn: Callable[[int], Batch],
    eval_batch: Batch,
    *,
    steps: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    ska_mode: str = "prefix",
    attn_mode: str = "prefix_lm",
    soft_weight: float = 0.1,
    device: str = "cpu",
    grad_clip: float = 1.0,
    log_every: int = 0,
    label: str = "model",
    use_amp: bool = True,
) -> Dict:
    """Train `model` online with `sample_fn(batch_size)` and return best eval acc."""
    dev = torch.device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = max(1, steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    amp = use_amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None

    best, t0 = 0.0, time.time()
    for step in range(steps):
        model.train()
        x, y, lm, pm, seg, meta = sample_fn(batch_size)
        x, y, lm, pm, seg = [t.to(dev) for t in (x, y, lm, pm, seg)]
        grp = _release_grp(meta, dev)
        pw, ab, rg = build_conditioning(kind, ska_mode, attn_mode, pm, seg, soft_weight, grp)

        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, prefix_weights=pw, attn_bias=ab, seg_ids=seg, release_grp=rg)
                loss = masked_ce(logits, y, lm)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x, prefix_weights=pw, attn_bias=ab, seg_ids=seg, release_grp=rg)
            loss = masked_ce(logits, y, lm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()

        if log_every and (step + 1) % log_every == 0:
            acc = evaluate(model, kind, eval_batch, ska_mode, attn_mode, soft_weight, device)
            best = max(best, acc)
            print(f"  [{label}] step {step+1:5d}/{steps} | loss {loss.item():.4f} "
                  f"| acc {acc:.4f} | best {best:.4f} | {time.time()-t0:.1f}s")

    acc = evaluate(model, kind, eval_batch, ska_mode, attn_mode, soft_weight, device)
    best = max(best, acc)
    return {"best_acc": best, "final_acc": acc}
