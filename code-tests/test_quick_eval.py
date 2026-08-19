"""experimentation/evaluation/quick_eval.py -- the cheap screening metric.

What a search needs per trial: held-out loss, and how much of it the SKA branch
is actually responsible for. The second is the interesting one -- a config can
improve loss while making SKA irrelevant, which is a worse result than it looks
for a paper about SKA.

Reconstructed from pieces that already existed rather than written fresh:
`evaluate.py::eval_fineweb_ppl` for the held-out loss and
`koopman_lm/models/koopman_lm.py::ablate(zero_ska=True)` for the ablation, which
`harness.py::_ska_delta` already drives the slow way (recomputing full PPL).

Results are written through `evaluation/result.py::write_result`, so a trial's
numbers land at `run_dir/eval/<checkpoint>/quick_eval.json` and
`python -m experimentation.results` sees them with no extra wiring. That is the
main reason this is a repo eval task and not a standalone script emitting its own
JSON shape.

A real KoopmanLM needs mamba_ssm and a GPU, so what is exercised here is the
accumulation loop (against a stub whose contract is `model(input_ids=, labels=)
-> {"loss": tensor}`), the metric assembly, and the run-directory wiring. The
numbers themselves need a GPU node.
"""
import json
import math

import pytest
import torch
import yaml

pytestmark = pytest.mark.correctness


class _StubModel:
    """Honours the one contract evaluate_loss depends on. Not a mock of a mock:
    the losses it returns are real tensors and the arithmetic under test is the
    token-weighted mean over them."""

    def __init__(self, losses):
        self._losses = list(losses)
        self.calls = 0
        self.eval_called = 0

    def eval(self):
        self.eval_called += 1
        return self

    def __call__(self, input_ids=None, labels=None):
        loss = self._losses[min(self.calls, len(self._losses) - 1)]
        self.calls += 1
        return {"loss": torch.tensor(float(loss))}


def _batches(n, *, batch=2, seq=4):
    return [{"input_ids": torch.zeros(batch, seq, dtype=torch.long),
             "labels": torch.zeros(batch, seq, dtype=torch.long)} for _ in range(n)]


# ------------------------------------------------------- accumulation ----

def test_evaluate_loss_is_token_weighted():
    from experimentation.evaluation.quick_eval import evaluate_loss

    model = _StubModel([1.0, 3.0])
    result = evaluate_loss(model, "cpu", _batches(2))
    assert result["loss"] == pytest.approx(2.0)
    assert result["n_tokens"] == 2 * 4 * 2


def test_evaluate_loss_reports_ppl_as_exp_of_loss():
    from experimentation.evaluation.quick_eval import evaluate_loss

    result = evaluate_loss(_StubModel([1.5]), "cpu", _batches(1))
    assert result["ppl"] == pytest.approx(math.exp(1.5))


def test_evaluate_loss_clamps_ppl_on_a_diverged_model():
    """exp(large) overflows to inf and poisons every downstream comparison;
    eval_fineweb_ppl already clamps at 20 and this must match it."""
    from experimentation.evaluation.quick_eval import evaluate_loss

    result = evaluate_loss(_StubModel([50.0]), "cpu", _batches(1))
    assert math.isfinite(result["ppl"])
    assert result["ppl"] == pytest.approx(math.exp(20))


def test_max_batches_caps_the_work():
    """A screening eval must be cheap and, more importantly, must look at the
    SAME number of tokens for every trial -- otherwise trial-to-trial loss
    differences partly measure how much data each one happened to see."""
    from experimentation.evaluation.quick_eval import evaluate_loss

    model = _StubModel([1.0])
    result = evaluate_loss(model, "cpu", _batches(100), max_batches=5)
    assert model.calls == 5
    assert result["n_batches"] == 5


def test_evaluate_loss_puts_the_model_in_eval_mode():
    from experimentation.evaluation.quick_eval import evaluate_loss

    model = _StubModel([1.0])
    evaluate_loss(model, "cpu", _batches(1))
    assert model.eval_called == 1


def test_evaluate_loss_reports_throughput_and_no_peak_memory_on_cpu():
    from experimentation.evaluation.quick_eval import evaluate_loss

    result = evaluate_loss(_StubModel([1.0]), "cpu", _batches(3))
    assert result["tokens_per_sec"] > 0
    assert result["peak_memory_gib"] is None, "CUDA-only measurement"


def test_evaluate_loss_on_an_empty_loader_does_not_divide_by_zero():
    from experimentation.evaluation.quick_eval import evaluate_loss

    result = evaluate_loss(_StubModel([1.0]), "cpu", [])
    assert result["n_tokens"] == 0
    assert result["n_batches"] == 0
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------- assembly ----

def test_assemble_reports_the_ska_delta_both_ways():
    """loss_delta is what the search rewards: how much worse the model gets when
    the SKA branch is zeroed. Positive and large means SKA is load-bearing."""
    from experimentation.evaluation.quick_eval import assemble_quick_eval

    full = {"loss": 2.0, "ppl": math.exp(2.0), "n_tokens": 100,
            "n_batches": 4, "tokens_per_sec": 10.0, "peak_memory_gib": None}
    zeroed = dict(full, loss=2.5, ppl=math.exp(2.5))
    metrics = assemble_quick_eval(full, zeroed)

    assert metrics["full"]["loss"] == pytest.approx(2.0)
    assert metrics["ska_ablation"]["supported"] is True
    assert metrics["ska_ablation"]["loss"] == pytest.approx(2.5)
    assert metrics["ska_ablation"]["loss_delta"] == pytest.approx(0.5)
    assert metrics["ska_ablation"]["ppl_delta"] == pytest.approx(math.exp(2.5) - math.exp(2.0))


def test_assemble_marks_the_ablation_unsupported_when_it_was_not_run():
    """A mamba-only baseline has no SKA to zero. Recording that as unsupported is
    different from recording a delta of 0.0, which would read as "SKA present
    and useless"."""
    from experimentation.evaluation.quick_eval import assemble_quick_eval

    full = {"loss": 2.0, "ppl": 7.39, "n_tokens": 100, "n_batches": 4,
            "tokens_per_sec": 10.0, "peak_memory_gib": None}
    metrics = assemble_quick_eval(full, None)
    assert metrics["ska_ablation"] == {"supported": False}
    assert "loss_delta" not in metrics["ska_ablation"]


def test_assemble_keeps_the_full_block_shape_the_search_reads():
    from experimentation.evaluation.quick_eval import assemble_quick_eval

    full = {"loss": 1.0, "ppl": 2.7, "n_tokens": 10, "n_batches": 1,
            "tokens_per_sec": 5.0, "peak_memory_gib": 1.5}
    metrics = assemble_quick_eval(full, None)
    assert set(metrics) == {"full", "ska_ablation"}
    assert set(metrics["full"]) == {"loss", "ppl", "n_tokens", "n_batches",
                                    "tokens_per_sec", "peak_memory_gib"}


# ------------------------------------------------------ run-dir wiring ----

def test_results_land_in_the_run_directory_under_the_quick_eval_task(tmp_path):
    """The reason this is a repo task rather than a standalone script: a trial's
    numbers become visible to `python -m experimentation.results` for free."""
    from experimentation.evaluation.quick_eval import TASK, write_quick_eval
    from experimentation.evaluation.result import read_result

    run_dir = tmp_path / "runs" / "study.abcd1234" / "seed42.deadbeef"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump({"run_id": "deadbeef"}))
    ckpt = run_dir / "final" / "model.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"not a real checkpoint")

    metrics = {"full": {"loss": 2.0}, "ska_ablation": {"supported": False}}
    path = write_quick_eval(ckpt, metrics)

    assert path == run_dir / "eval" / "final" / f"{TASK}.json"
    envelope = read_result(path)
    assert envelope["task"] == TASK
    assert envelope["run_id"] == "deadbeef"
    assert envelope["checkpoint"] == "final"
    assert envelope["metrics"]["full"]["loss"] == 2.0
    assert envelope["git_commit"]


def test_aggregation_picks_up_a_quick_eval_result(tmp_path):
    from experimentation.evaluation.quick_eval import write_quick_eval
    from experimentation.results import aggregate

    run_dir = tmp_path / "runs" / "study.abcd1234" / "seed42.deadbeef"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump({
        "name": "study", "run_id": "deadbeef", "group_id": "abcd1234",
        "model": {"d_model": 384, "n_layers": 17},
        "optim": {"lr": 0.0004}, "runtime": {"seed": 42},
        "data": {"kind": "shard"},
    }))
    ckpt = run_dir / "final" / "model.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    write_quick_eval(ckpt, {"full": {"loss": 2.0, "ppl": 7.39},
                            "ska_ablation": {"supported": True, "loss_delta": 0.4}})

    rows = aggregate(tmp_path / "runs")
    assert len(rows) == 1
    assert rows[0]["task"] == "quick_eval"
    assert rows[0]["full.loss"] == 2.0
    assert rows[0]["ska_ablation.loss_delta"] == 0.4


def test_write_quick_eval_returns_none_outside_a_run_directory(tmp_path):
    """A hand-run checkpoint has no spec.yaml above it; the caller falls back to
    an explicit output path rather than inventing a run identity."""
    from experimentation.evaluation.quick_eval import write_quick_eval

    ckpt = tmp_path / "loose" / "model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    assert write_quick_eval(ckpt, {"full": {"loss": 1.0}}) is None
