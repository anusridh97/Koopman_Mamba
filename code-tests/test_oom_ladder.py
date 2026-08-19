"""The OOM ladder: fit a config into memory without changing the science.

`ddp_grad_accum` already refuses to launch when effective_batch is not divisible
by per_device_batch_size x world_size -- it raises rather than silently
truncating, which is right. What the repo lacks is the other half: when a config
OOMs, the microbatch should halve and gradient accumulation should double, so the
effective batch -- the quantity that actually affects the result -- is preserved
exactly.

`batch_plans` enumerates those rungs. It lives beside `ddp_grad_accum` because
they are two halves of one idea: that function validates a chosen split, this one
lists the legal splits in the order to try them.

**The identity consequence, verified and deliberately not fixed here.**
`per_device_batch_size` lives in `OptimSpec`, and `OptimSpec` is hashed whole into
`run_id`. So descending a rung changes the run identity even though
`effective_batch` is unchanged and the science is identical:

    effective_batch=64, pdbs=16  ->  run_id 58674511
    effective_batch=64, pdbs=8   ->  run_id 02705fcd

Arguably it belongs in `RuntimeSpec`, beside `gpus`/`nodes`/`ddp`, which is
explicitly "the same experiment run differently" and is not hashed. But moving it
would move every existing run's identity -- the same class of change as the
precision-policy work, which the backlog treats as a decision requiring its own
commit and a regenerated baseline. So the ladder records which rung succeeded and
this stays a flagged question rather than a silent migration.
"""
import json

import pytest

pytestmark = pytest.mark.correctness


# --------------------------------------------------------- the rungs ----

def test_the_ladder_halves_the_microbatch_and_doubles_accumulation():
    from experimentation.run.train_argv import batch_plans

    assert batch_plans(64, 8) == [(8, 8), (4, 16), (2, 32), (1, 64)]


def test_every_rung_preserves_the_effective_batch_exactly():
    """The whole point. A rung that changed the effective batch would change the
    result, and the trial would be measuring the wrong thing."""
    from experimentation.run.train_argv import batch_plans

    for effective in (16, 64, 96, 512):
        for initial in (1, 8, 16, 32):
            for pdbs, accum in batch_plans(effective, initial):
                assert pdbs * accum == effective, (effective, initial, pdbs, accum)


def test_the_ladder_skips_microbatches_that_do_not_divide_the_effective_batch():
    """96 is not a power of two, so the rungs are its divisors, not 16/8/4/2/1."""
    from experimentation.run.train_argv import batch_plans

    plans = batch_plans(96, 24)
    assert all(96 % pdbs == 0 for pdbs, _ in plans)
    assert (24, 4) in plans


def test_the_ladder_starts_no_higher_than_the_effective_batch():
    from experimentation.run.train_argv import batch_plans

    assert batch_plans(8, 64)[0][0] <= 8


def test_a_minimum_microbatch_truncates_the_ladder():
    """Below some microbatch the run is too slow to be worth finishing; stopping
    is better than descending to 1 and taking a week."""
    from experimentation.run.train_argv import batch_plans

    plans = batch_plans(64, 8, minimum_pdbs=4)
    assert plans == [(8, 8), (4, 16)]


def test_the_ladder_is_never_empty():
    """One sample per device with full accumulation always preserves the
    effective batch, so there is always a last resort."""
    from experimentation.run.train_argv import batch_plans

    assert batch_plans(7, 5) == [(1, 7)]


def test_non_positive_batches_are_rejected():
    from experimentation.run.train_argv import batch_plans

    for args in ((0, 8), (64, 0), (-1, 8)):
        with pytest.raises(ValueError):
            batch_plans(*args)


def test_descending_a_rung_changes_the_run_id(tmp_path):
    """Pinned so the consequence is impossible to forget: this is why the ladder
    records the rung that worked, and why moving per_device_batch_size into
    RuntimeSpec is a separate decision."""
    from koopman_lm.config import build_config
    from experimentation.run.spec import (OptimSpec, RunSpec, RuntimeSpec,
                                          ShardDataSpec, run_id)

    def spec(pdbs):
        return RunSpec(name="x", model=build_config("50m"),
                       data=ShardDataSpec(kind="shard", shard_dir="/tmp/s",
                                          tokenizer="t", mix={"f": 1.0}, n_tokens=10),
                       optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100,
                                       effective_batch=64, per_device_batch_size=pdbs),
                       runtime=RuntimeSpec(seed=42))

    assert run_id(spec(16)) != run_id(spec(8))


# -------------------------------------------------------- oom detection ----

def test_a_cuda_oom_in_a_slurm_log_is_detected(tmp_path):
    from experimentation.sweep.search.metrics import looks_like_oom

    (tmp_path / "slurm-12345.out").write_text(
        "step 10/600 | loss 7.1\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    assert looks_like_oom(tmp_path) is True


def test_oom_detection_is_case_insensitive(tmp_path):
    from experimentation.sweep.search.metrics import looks_like_oom

    (tmp_path / "training.log").write_text("CUBLAS_STATUS_ALLOC_FAILED\n")
    assert looks_like_oom(tmp_path) is True


def test_an_unrelated_failure_is_not_an_oom(tmp_path):
    """Retrying a shape error at a smaller microbatch burns a queue slot to fail
    the same way."""
    from experimentation.sweep.search.metrics import looks_like_oom

    (tmp_path / "slurm-1.out").write_text(
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied\n")
    assert looks_like_oom(tmp_path) is False


def test_no_log_at_all_is_not_an_oom(tmp_path):
    from experimentation.sweep.search.metrics import looks_like_oom

    assert looks_like_oom(tmp_path) is False


# ------------------------------------------------------ objective reading ----

def _write_quick_eval(run_dir, metrics, checkpoint="final"):
    path = run_dir / "eval" / checkpoint / "quick_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": "quick_eval", "metrics": metrics}))
    return path


def test_the_objective_is_read_from_the_run_directory(tmp_path):
    from experimentation.sweep.search.metrics import read_quick_eval_objective

    _write_quick_eval(tmp_path, {"full": {"loss": 2.25},
                                 "ska_ablation": {"supported": False}})
    assert read_quick_eval_objective(tmp_path) == pytest.approx(2.25)


def test_reading_applies_the_configured_penalty_weights(tmp_path):
    from experimentation.sweep.search.metrics import read_quick_eval_objective

    _write_quick_eval(tmp_path, {"full": {"loss": 2.0},
                                 "ska_ablation": {"supported": True, "loss_delta": 0.5}})
    score = read_quick_eval_objective(tmp_path, ska_delta_reward=1.0, ska_delta_cap=0.1)
    assert score == pytest.approx(2.0 - 0.1)


def test_a_missing_result_reads_as_none_rather_than_a_number(tmp_path):
    """The driver turns None into a FAIL. Any number here would be a fiction
    that steers every later proposal."""
    from experimentation.sweep.search.metrics import read_quick_eval_objective

    assert read_quick_eval_objective(tmp_path) is None


def test_the_latest_checkpoint_is_preferred_over_an_earlier_one(tmp_path):
    """A run that saved step_100 and then final should be scored on final."""
    from experimentation.sweep.search.metrics import read_quick_eval_objective

    _write_quick_eval(tmp_path, {"full": {"loss": 5.0}}, checkpoint="step_100")
    _write_quick_eval(tmp_path, {"full": {"loss": 2.0}}, checkpoint="final")
    assert read_quick_eval_objective(tmp_path) == pytest.approx(2.0)


# ---------------------------------------------------------- the retry ----

class _OomThenSucceedLauncher:
    def __init__(self, oom_attempts, marker="CUDA out of memory"):
        self.attempts = []
        self.oom_attempts = oom_attempts
        self.marker = marker

    def submit(self, spec, run_dir, dry_run=False, resume=False):
        self.attempts.append((spec.optim.per_device_batch_size, run_dir))
        if len(self.attempts) <= self.oom_attempts:
            (run_dir / "slurm-1.out").write_text(f"{self.marker}\n")
            raise RuntimeError("child exited non-zero")
        return ["ok"]


def _context(tmp_path):
    import dataclasses

    from koopman_lm.config import build_config
    from experimentation.sweep.search.space import search_space

    shard = tmp_path / "shard"
    shard.mkdir(exist_ok=True)
    (shard / "meta.json").write_text(json.dumps({
        "n_tokens": 1000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0}}))
    cfg = build_config("50m")
    return {
        "base_sections": {
            "model": dataclasses.asdict(cfg),
            "data": {"kind": "shard", "shard_dir": str(shard),
                     "tokenizer": "NousResearch/Llama-2-7b-hf",
                     "mix": {"fineweb": 1.0}, "n_tokens": 1000},
            "optim": {"lr": 4.0e-4, "warmup_steps": 10, "max_steps": 100,
                      "effective_batch": 64, "per_device_batch_size": 16},
            "runtime": {"seed": 42},
        },
        "base_model": cfg,
        "space": search_space(cfg, base_name="50m"),
        "max_steps": 100,
        "run_root": tmp_path / "runs",
        "study_name": "ska-depth",
        "base_lr": 4e-4,
    }


def test_an_oom_descends_one_rung_and_succeeds(tmp_path):
    optuna = pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _OomThenSucceedLauncher(oom_attempts=1)

    outcome = run_trial(study, trial, launcher=launcher,
                        read_objective=lambda run_dir: 2.0,
                        batch_ladder=True, **context)

    assert outcome.state == "complete"
    assert [pdbs for pdbs, _ in launcher.attempts] == [16, 8]
    assert outcome.per_device_batch_size == 8


def test_each_rung_gets_its_own_run_directory(tmp_path):
    """Because per_device_batch_size is hashed, the retry is a different run_id --
    so it must not be written into the first rung's directory."""
    optuna = pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _OomThenSucceedLauncher(oom_attempts=1)
    run_trial(study, trial, launcher=launcher, read_objective=lambda run_dir: 2.0,
              batch_ladder=True, **context)

    directories = {run_dir for _, run_dir in launcher.attempts}
    assert len(directories) == 2


def test_a_non_oom_failure_does_not_descend(tmp_path):
    optuna = pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _OomThenSucceedLauncher(oom_attempts=1, marker="shapes cannot be multiplied")

    outcome = run_trial(study, trial, launcher=launcher,
                        read_objective=lambda run_dir: 2.0,
                        batch_ladder=True, **context)
    assert outcome.state == "failed"
    assert len(launcher.attempts) == 1


def test_the_ladder_gives_up_after_exhausting_every_rung(tmp_path):
    optuna = pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _OomThenSucceedLauncher(oom_attempts=99)

    outcome = run_trial(study, trial, launcher=launcher,
                        read_objective=lambda run_dir: 2.0,
                        batch_ladder=True, **context)
    assert outcome.state == "failed"
    assert [pdbs for pdbs, _ in launcher.attempts] == [16, 8, 4, 2, 1]


def test_without_the_ladder_an_oom_fails_immediately(tmp_path):
    """The ladder is opt-in: a single hand-launched run should fail loudly rather
    than silently spend four more queue slots."""
    optuna = pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _OomThenSucceedLauncher(oom_attempts=1)

    outcome = run_trial(study, trial, launcher=launcher,
                        read_objective=lambda run_dir: 2.0, **context)
    assert outcome.state == "failed"
    assert len(launcher.attempts) == 1
