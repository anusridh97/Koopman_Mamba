"""The training loop, once.

Extracted from `train.py` unchanged. Its three callers -- `train.py`,
`mqar_finetune.py` and `table2.py` -- previously each carried their own copy, and
only train.py's had gradient accumulation, an `on_final` objective, DDP,
preemption handling or an RNG-restoring resume. That is why this is not tidying:
the other two produce no `quick_eval.json`, which is what
`sweep/search/metrics.py::read_quick_eval_objective` reads, so **neither can be
driven by the adaptive search at all**. `run/train_argv.py` says so outright when
it refuses `data.kind='synthetic'`.

What is a task's and what is the loop's is `training/task.py`'s question; see the
design doc's ownership table. Briefly: the task owns the loss, the data and its
iteration order; the loop owns accumulation, DDP, resume, amp, checkpointing and
the log line.

**The log format is load-bearing.** `scripts/compare_golden_curve.py` and
`sweep/search/metrics.py::parse_progress` both parse `loss %.4f` out of it -- the
golden comparator and the search pruner share one regex. Changing the line breaks
both, silently, because each just stops matching.

This extraction keeps every identifier byte-identical to the block it came from,
so it is auditable as a move rather than a rewrite. Verified against all four
goldens rather than by inspection.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from experimentation.training.task import IterContext

__all__ = ["LoopResult", "run_training_loop"]


@dataclass
class LoopResult:
    """What the caller needs after the loop, and nothing more."""

    step: int
    epoch: int
    tokens_seen: int
    #: True if SIGUSR1 arrived and resume.pt was written. The caller owns DDP
    #: teardown, because the caller owns DDP setup.
    preempted: bool


def run_training_loop(
    *,
    task,
    model,
    raw_model,
    optimizer,
    scheduler,
    args,
    device,
    train_ds,
    autocast_ctx,
    _save_all,
    start_step: int = 0,
    start_epoch: int = 0,
    start_samples_consumed: int = 0,
    log_window: Optional[dict] = None,
    is_main: bool = True,
    is_ddp: bool = False,
    world_size: int = 1,
    local_rank: int = 0,
    sampler: Any = None,
    monitor: Any = None,
    grad_monitor: Any = None,
    preempt_flag: Any = None,
    t_start: Optional[float] = None,
) -> LoopResult:
    """Run to `args.max_steps`, or until preempted.

    Keyword-only on purpose: twenty-one positional arguments is how a caller
    silently swaps `start_epoch` for `start_step`.

    `_save_all` keeps its underscore because it is the caller's closure, moved
    here unrenamed -- it writes both the archival `step_<N>/` checkpoint and
    `resume.pt`, which must be written together or a resume finds a step number
    with no weights.
    """
    # The counters the loop owns. `log_window` is restored rather than zeroed so
    # the first progress line after a resume averages over the SAME window an
    # uninterrupted run would -- without it the two disagree at exactly one step
    # (measured 0.0011 in job 440211) while training is bit-identical.
    step = start_step
    micro_step = 0
    tokens_seen = 0
    _win = log_window or {}
    running_loss = torch.tensor(float(_win.get("running_loss", 0.0)), device=device)
    loss_count = int(_win.get("loss_count", 0))
    if t_start is None:
        t_start = time.time()

    if is_main:
        eff = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        # Flushed because this is the run's first sign of life. In job 439754 it
        # was the only line that survived 21 minutes -- and only by accident, a
        # DataLoader fork calling _flush_std_streams just after it.
        print(f"\nTraining: {args.max_steps} steps, eff_batch={eff}, "
              f"tok/step={eff*args.max_seq_len:,}", flush=True)
    optimizer.zero_grad(set_to_none=True)

    epoch = start_epoch
    preempted = False
    while step < args.max_steps:
        if hasattr(train_ds, 'set_epoch'): train_ds.set_epoch(epoch)
        # §5.2's explicit index arithmetic (no implicit shuffle=True) and the
        # required `generator=` now live in TrainTask.iter_batches, unchanged --
        # see its docstring for why each is load-bearing. Moved rather than
        # rewritten so the one loop can serve table2, whose batches are a
        # function of the step counter and have no epoch permutation at all.
        samples_consumed = start_samples_consumed if epoch == start_epoch else 0
        epoch_loader = task.iter_batches(train_ds, args, IterContext(
            epoch=epoch, start_step=step, skip_samples=samples_consumed,
            sampler=sampler if is_ddp else None,
            is_ddp=is_ddp, local_rank=local_rank))
        for batch in epoch_loader:
            if step >= args.max_steps: break
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lw = batch.get("loss_weights")
            if lw is not None: lw = lw.to(device, non_blocking=True)
            with autocast_ctx:
                # §6.2 step one: the loss now comes from the TASK rather than
                # being inlined here. Nothing else changes yet -- ShardTask's
                # step_loss is `model(input_ids, labels, loss_weights)["loss"]`,
                # pinned bit-identical to the two lines it replaces by
                # code-tests/test_train_task.py.
                #
                # This is the seam the unification needs, and putting it in first
                # while the loop is otherwise untouched means the risky part --
                # collapsing three loops -- lands against a loop that already
                # delegates, verified by the 4m golden curve rather than by
                # inspection.
                #
                # The loop still enters autocast; a task never manages precision.
                raw_loss = task.step_loss(model, {
                    "input_ids": ids, "labels": labels, "loss_weights": lw})
                scaled_loss = raw_loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += raw_loss.detach(); loss_count += 1
            tokens_seen += ids.numel(); micro_step += 1
            samples_consumed += ids.size(0)
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); step += 1
                if is_main and step % args.logging_steps == 0:
                    avg = running_loss.item() / max(loss_count, 1)
                    lr = optimizer.param_groups[0]["lr"]; el = time.time() - t_start
                    tps = tokens_seen / el; ppl = math.exp(min(avg, 20))
                    # flush=True is load-bearing, not tidiness. When stdout is
                    # a FILE rather than a tty -- which it is for every
                    # non-blocking local launch, and for sbatch -- CPython
                    # block-buffers at 8 KB. A 200-step trial at logging_steps 10
                    # emits ~1.4 KB, so without this the log stays EMPTY until the
                    # process exits and then flushes everything at once.
                    #
                    # Two things break on that. A cancelled run loses its output
                    # entirely (job 439754 showed 21 minutes of nothing, and was
                    # not hung -- ~70 steps had run). And pruning becomes
                    # decorative: wait_for_objective tails this log, so every
                    # report lands after training already finished. Job 439883
                    # recorded "20 reported steps" per trial and every one of them
                    # arrived too late to prune anything.
                    print(f"step {step:>6d}/{args.max_steps} | loss {avg:.4f} | "
                          f"ppl {ppl:.1f} | lr {lr:.2e} | {tps/1e3:.1f}K tok/s",
                          flush=True)
                    if args.wandb_project:
                        import wandb
                        wandb.log({"loss": avg, "ppl": ppl, "lr": lr,
                                   "tokens_per_sec": tps}, step=step)
                    running_loss = torch.tensor(0.0, device=device); loss_count = 0

                # ---- SKA health diagnostics (separate cheap fwd, amortized) ----
                if monitor is not None and step % args.diag_every == 0:
                    was_training = raw_model.training
                    with torch.no_grad(), monitor.capture():
                        raw_model(input_ids=ids)        # labels=None -> no loss
                    if was_training:
                        raw_model.train()
                    health = monitor.collect()
                    scal = {k: v for k, v in health.items()
                            if isinstance(v, (int, float))}
                    radii = [v for k, v in scal.items()
                             if k.endswith("/spectral_radius_mean")]
                    gates = [v for k, v in scal.items() if k.endswith("/gate_mag")]
                    rad_avg = sum(radii) / len(radii) if radii else float("nan")
                    gate_avg = sum(gates) / len(gates) if gates else float("nan")
                    rr = scal.get("ska/residual_ratio", float("nan"))
                    lmr = min((v for k, v in scal.items()
                               if k.endswith("/lambda_min_over_ridge")), default=float("nan"))
                    print(f"  [ska-health] step {step}: radius~{rad_avg:.3f} "
                          f"gate~{gate_avg:.2e} resid_ratio~{rr:.2e} "
                          f"lmin/ridge~{lmr:.2f}")
                    if args.wandb_project:
                        import wandb
                        wandb.log(health, step=step)

                # ---- SKA gradient-flow diagnostics (extra fwd+bwd, amortized) ----
                if grad_monitor is not None and step % args.diag_every == 0:
                    # dedicated fwd+bwd on the current micro-batch; grads are
                    # cleared afterwards so the next accumulation window starts
                    # clean (we are right after optimizer.zero_grad anyway).
                    with grad_monitor.capture():
                        gout = raw_model(input_ids=ids, labels=labels, loss_weights=lw)
                        gout["loss"].backward()
                    raw_model.zero_grad(set_to_none=True)
                    gflow = grad_monitor.collect()
                    if gflow:
                        gr = gflow.get("ska/grad_norm_ratio", float("nan"))
                        print(f"  [ska-grad] step {step}: grad_norm_ratio~{gr:.2e}")
                        if args.wandb_project:
                            import wandb
                            wandb.log(gflow, step=step)

                _window = {"running_loss": float(running_loss.item()),
                           "loss_count": int(loss_count)}
                if is_main and step > 0 and step % args.save_steps == 0:
                    _save_all(step, epoch, samples_consumed, _window)
                if is_main and preempt_flag is not None and preempt_flag.is_set():
                    _save_all(step, epoch, samples_consumed, _window)
                    print(f"  SIGUSR1 received -- wrote resume.pt at step {step}, exiting cleanly")
                    preempted = True
                    break
        if preempted:
            break
        epoch += 1

    return LoopResult(step=step, epoch=epoch, tokens_seen=tokens_seen,
                      preempted=preempted)
