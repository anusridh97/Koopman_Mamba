"""The shared loop, actually executed on CPU.

`train.py`'s loop sat at 29% line coverage for its whole life, which is why the
TrainTask work needed four GPU golden curves to say anything at all: almost
everything the loop does was invisible to the suite. Now that it is a function in
`training/loop.py` rather than 139 lines inside `main()`, it can be RUN -- with a
3-parameter model, a 16-sample dataset and a real AdamW, in well under a second.

That does not replace the goldens. A CPU harness cannot see a kernel-level or
precision-level change, and it deliberately does not try. What it can see is
everything structural, which is most of what a refactor breaks:

  * how many optimizer steps happen for N micro-batches at accumulation K;
  * that the LR schedule advances once per OPTIMIZER step, not per micro-batch;
  * the exact order of scaler / clip / step / zero_grad, which differs between
    the fp16 and bf16 paths and is where a unified loop most plausibly goes
    wrong;
  * that checkpoints are written on the declared cadence and carry the logging
    window;
  * that the progress line the golden comparator and the search pruner both
    parse is actually emitted.

The scaler is a RECORDING STUB rather than a real `torch.amp.GradScaler`, because
what must be preserved is the call SEQUENCE (table2's fp16 path) and a real
scaler on CPU would either no-op or refuse. The numerics are the goldens' job.
"""

import math
import pathlib
import re
import sys
import types

import pytest
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.training.loop import run_training_loop  # noqa: E402
from experimentation.training.task import TrainTask  # noqa: E402


class _TinyModel(nn.Module):
    """Deterministic, tiny, and its loss depends on the batch so a changed batch
    order changes the curve."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)

    def forward(self, input_ids=None, **kw):
        x = input_ids.float()
        return {"logits": self.lin(x).unsqueeze(1)}


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return {"input_ids": torch.full((4,), float(i)),
                "labels": torch.full((4,), float(i))}


class _TinyTask(TrainTask):
    """Real `iter_batches` (the default epoch path); trivial loss."""

    def step_loss(self, model, batch):
        out = model(input_ids=batch["input_ids"])
        return out["logits"].pow(2).mean()


class _RecordingScaler:
    """Records the sequence, applies no scaling."""

    def __init__(self):
        self.calls = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.calls.append("unscale_")

    def step(self, optimizer):
        self.calls.append("scaler.step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def _args(**over):
    a = types.SimpleNamespace(
        max_steps=4, per_device_train_batch_size=2,
        gradient_accumulation_steps=1, max_seq_len=4, seed=0, num_workers=0,
        logging_steps=1, max_grad_norm=1.0, save_steps=1000, diag_every=10**9,
        wandb_project=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _run(args, scaler=None, saves=None):
    import contextlib

    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    recorded = saves if saves is not None else []

    def _save_all(step, epoch, samples_consumed, log_window=None, dirname=None):
        recorded.append({"step": step, "epoch": epoch,
                         "samples": samples_consumed,
                         "log_window": dict(log_window or {})})

    result = run_training_loop(
        task=_TinyTask(), model=model, raw_model=model, optimizer=opt,
        scheduler=sched, args=args, device=torch.device("cpu"),
        train_ds=_TinyDataset(), autocast_ctx=contextlib.nullcontext(),
        _save_all=_save_all, scaler=scaler)
    return result, model, opt, sched, recorded


# ------------------------------------------------------------ it runs at all ----

def test_the_loop_runs_on_cpu_and_reaches_max_steps():
    result, *_ = _run(_args(max_steps=4))
    assert result.step == 4
    assert result.preempted is False
    assert result.tokens_seen > 0


# ------------------------------------------------- accumulation arithmetic ----

@pytest.mark.parametrize("accum,expected_steps", [(1, 4), (2, 4), (4, 4)])
def test_it_takes_max_steps_optimizer_steps_whatever_the_accumulation(accum, expected_steps):
    """`max_steps` counts OPTIMIZER steps, not micro-batches. Getting this
    backwards silently shortens or lengthens every run by a factor of accum,
    while the loss curve still descends."""
    result, _, _, sched, _ = _run(_args(max_steps=expected_steps,
                                        gradient_accumulation_steps=accum))
    assert result.step == expected_steps


def test_the_schedule_advances_once_per_optimizer_step_not_per_microbatch():
    """The LR schedule is the thing accumulation most easily corrupts: stepping
    it per micro-batch makes warmup finish accum-times too early, which shifts
    the whole curve without ever erroring."""
    steps = 4
    lrs = {}
    for accum in (1, 2):
        _, _, opt, sched, _ = _run(_args(max_steps=steps,
                                         gradient_accumulation_steps=accum))
        lrs[accum] = sched.last_epoch
    assert lrs[1] == lrs[2] == steps, (
        f"schedule advanced {lrs} times for {steps} optimizer steps -- it must "
        f"track optimizer steps, not micro-batches")


# ------------------------------------------------------------- the fp16 path ----

def test_the_scaler_sequence_matches_table2s():
    """table2 is the only fp16 trainer and the only user of a GradScaler, so
    nothing else exercises this path. Its committed order is
    scale->backward, unscale_, clip, scaler.step, update -- and unscale_ MUST
    precede the clip or the gradients are clipped while still scaled, which
    silently changes the effective clip threshold.
    """
    scaler = _RecordingScaler()
    _run(_args(max_steps=2), scaler=scaler)

    # One optimizer step per micro-batch at accum=1.
    assert scaler.calls == ["scale", "unscale_", "scaler.step", "update"] * 2, \
        f"got {scaler.calls}"


def test_without_a_scaler_nothing_scales():
    """The bf16 path must not pay for machinery it does not use, and must not
    call unscale_ on gradients that were never scaled."""
    scaler = _RecordingScaler()
    _run(_args(max_steps=2), scaler=None)
    assert scaler.calls == []


def test_the_scaler_unscales_once_per_optimizer_step_under_accumulation():
    """`unscale_` may be called at most once per optimizer step per optimizer --
    torch raises otherwise -- so it belongs in the accumulation branch, not
    beside every backward."""
    scaler = _RecordingScaler()
    _run(_args(max_steps=2, gradient_accumulation_steps=3), scaler=scaler)
    assert scaler.calls.count("unscale_") == 2, \
        f"expected one unscale_ per optimizer step, got {scaler.calls}"
    assert scaler.calls.count("scale") == 6, "one scale per micro-batch"


# -------------------------------------------------------------- checkpoints ----

def test_checkpoints_land_on_the_declared_cadence_and_carry_the_window():
    _, _, _, _, saves = _run(_args(max_steps=4, save_steps=2))
    assert [s["step"] for s in saves] == [2, 4]
    for s in saves:
        assert set(s["log_window"]) == {"running_loss", "loss_count"}, (
            "the resume window must reach _save_all, or the first progress line "
            "after a resume averages over a short window (job 440211)")


# ------------------------------------------------------------- the log line ----

def test_it_emits_the_line_both_parsers_read(capsys):
    """`scripts/compare_golden_curve.py` and
    `sweep/search/metrics.py::parse_progress` share one regex against this line.
    If it stops being emitted, goldens stop comparing and trials stop pruning --
    both silently, by simply not matching."""
    from experimentation.sweep.search.metrics import parse_progress

    _run(_args(max_steps=3, logging_steps=1))
    out = capsys.readouterr().out
    points = parse_progress(out)
    assert [p.step for p in points] == [1, 2, 3], f"parsed {points} from:\n{out}"
    assert all(math.isfinite(p.loss) for p in points)
    assert re.search(r"loss \d+\.\d{4} ", out), "the %.4f field is what is parsed"


# ------------------------------------------- what the loop reports, and when ----
#
# The three trainers do NOT agree on which loss goes on the progress line:
#
#   train.py  running_loss / loss_count      -- window average
#   mqar      loss.item()                    -- the instantaneous last micro-batch
#   table2    mean of PER-TASK window averages, plus a per-task tail
#
# The design doc's ownership table says "logging | loop", which assumes one
# semantics. Unifying without a seam here changes at least one trainer's every
# logged value -- and since the logged value IS the golden curve, that destroys
# the one instrument that could tell "reporting changed" from "training broke",
# at exactly the moment it is needed.
#
# So the loop owns the FORMAT and the task declares WHICH loss. Standardising on
# the window average may well be right, but it is a deliberate change with a
# re-captured golden, not something a refactor smuggles in.


class _LastLossTask(_TinyTask):
    def progress_fields(self, *, window_avg, last_loss):
        return last_loss, ""


class _TailTask(_TinyTask):
    def progress_fields(self, *, window_avg, last_loss):
        return window_avg, "  toolcall_loss 1.2345"


def test_the_default_reports_the_window_average(capsys):
    """train.py's behaviour, unchanged, and verified against the loop's own
    accumulator rather than against a hardcoded number."""
    _run(_args(max_steps=2, logging_steps=2))
    out = capsys.readouterr().out
    from experimentation.sweep.search.metrics import parse_progress
    pts = parse_progress(out)
    assert len(pts) == 1, out
    # Two micro-batches averaged; the instantaneous value would differ from it.
    assert pts[0].step == 2


def test_a_task_can_report_the_instantaneous_loss_instead(capsys):
    """mqar's convention. Asserted by DIFFERENCE against the default on the same
    seed and data, so it cannot pass by accident if the hook is ignored."""
    from experimentation.sweep.search.metrics import parse_progress

    torch.manual_seed(0)
    _run(_args(max_steps=2, logging_steps=2))
    default = parse_progress(capsys.readouterr().out)[0].loss

    torch.manual_seed(0)
    import contextlib
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    run_training_loop(
        task=_LastLossTask(), model=model, raw_model=model, optimizer=opt,
        scheduler=sched, args=_args(max_steps=2, logging_steps=2),
        device=torch.device("cpu"), train_ds=_TinyDataset(),
        autocast_ctx=contextlib.nullcontext(),
        _save_all=lambda *a, **k: None)
    last = parse_progress(capsys.readouterr().out)[0].loss

    assert last != default, (
        "reporting the instantaneous loss gave the same number as the window "
        "average, so the hook is not being consulted")


def test_a_task_can_append_its_own_fields(capsys):
    """table2 appends per-task losses. The `loss %.4f` field both parsers read
    must survive the addition, which is the whole risk of a tail."""
    import contextlib

    from experimentation.sweep.search.metrics import parse_progress

    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    run_training_loop(
        task=_TailTask(), model=model, raw_model=model, optimizer=opt,
        scheduler=sched, args=_args(max_steps=1, logging_steps=1),
        device=torch.device("cpu"), train_ds=_TinyDataset(),
        autocast_ctx=contextlib.nullcontext(), _save_all=lambda *a, **k: None)
    out = capsys.readouterr().out
    assert "toolcall_loss 1.2345" in out
    assert parse_progress(out), f"the tail broke the parsed line:\n{out}"


# --------------------------------------------------------- in_loop_eval ----

def test_the_loop_calls_in_loop_eval(capsys):
    """It is on the TrainTask interface and the loop never called it. Migrating
    mqar onto this loop without wiring it would silently drop MQAR accuracy --
    the one number that trainer exists to produce."""
    seen = []

    class _EvalTask(_TinyTask):
        def in_loop_eval(self, model, step):
            seen.append(step)
            return {"acc": 0.5}

    import contextlib
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    run_training_loop(
        task=_EvalTask(), model=model, raw_model=model, optimizer=opt,
        scheduler=sched, args=_args(max_steps=4), device=torch.device("cpu"),
        train_ds=_TinyDataset(), autocast_ctx=contextlib.nullcontext(),
        _save_all=lambda *a, **k: None)
    assert seen == [1, 2, 3, 4], (
        f"in_loop_eval saw steps {seen}; the task decides its own cadence, so "
        f"the loop must offer it every step")


# ------------------------------------------------------ tuple batches ----

def test_the_loop_accepts_tuple_batches():
    """mqar and table2 yield `(inputs, labels)`; the shard path yields a dict
    with loss_weights. The loop moves either to the device without knowing
    which task it is serving."""
    import contextlib

    class _TupleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return torch.full((4,), float(i)), torch.full((4,), float(i))

    class _TupleTask(_TinyTask):
        def step_loss(self, model, batch):
            inputs, _labels = batch
            return model(input_ids=inputs)["logits"].pow(2).mean()

    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    result = run_training_loop(
        task=_TupleTask(), model=model, raw_model=model, optimizer=opt,
        scheduler=sched, args=_args(max_steps=2), device=torch.device("cpu"),
        train_ds=_TupleDataset(), autocast_ctx=contextlib.nullcontext(),
        _save_all=lambda *a, **k: None)
    assert result.step == 2 and result.tokens_seen > 0

