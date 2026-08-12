"""A spec that cannot launch must be rejected before anything is written.

`kind: synthetic` is legal to *build* (spec.py's SyntheticDataSpec exists and
configs/runs/50m-mqar-smoke.yaml uses it) but cannot be *launched*: train.py has
no synthetic data path, so build_train_argv raises. The problem was where that
raise happened -- step 8 of 8, after run/__main__.py had already:

    created <run_root>/<group>/<run>/
    written spec.yaml            (a fully materialized spec for an unlaunchable run)
    written model_config.json
    appended to attempts.jsonl   (a record claiming a launch was attempted)

Reproduced before the fix, under --dry_run, whose entire purpose is safe
inspection. The attempts.jsonl line carried a real timestamp, host, and code_id.
That is worse than a crash: it is provenance for something that never ran, and
`_existing_code_id` reads that directory on the next launch.

The guard belongs where the shard check already is -- before create_run_dir --
so these tests assert the run_root stays EMPTY, not merely that an error is
raised. build_train_argv keeps its own raise as a backstop for callers that
bypass __main__.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.correctness

SYNTHETIC_SPEC = "configs/runs/50m-mqar-smoke.yaml"


def _run_main(tmp_path, extra=()):
    from experimentation.run.__main__ import main
    return main([SYNTHETIC_SPEC, "--run_root", str(tmp_path),
                 "--allow-dirty", *extra])


def test_synthetic_spec_is_rejected_before_anything_is_written(tmp_path):
    """The load-bearing assertion: run_root is untouched."""
    from experimentation.run.data_verify import UnlaunchableDataSpecError

    with pytest.raises(UnlaunchableDataSpecError):
        _run_main(tmp_path, ["--dry_run"])

    leftovers = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert not leftovers, (
        f"a rejected synthetic spec left {len(leftovers)} path(s) behind: "
        f"{leftovers}. Nothing may be created before the spec is known "
        f"launchable -- an attempts.jsonl entry is provenance for a run that "
        f"never happened.")


def test_synthetic_rejection_also_applies_without_dry_run(tmp_path):
    """dry_run is not what makes this unlaunchable -- the data kind is."""
    from experimentation.run.data_verify import UnlaunchableDataSpecError

    with pytest.raises(UnlaunchableDataSpecError):
        _run_main(tmp_path)
    assert not list(tmp_path.rglob("*"))


def test_the_error_names_the_kind_and_what_is_missing(tmp_path):
    """A user hitting this needs to know it is unimplemented, not malformed."""
    from experimentation.run.data_verify import UnlaunchableDataSpecError

    with pytest.raises(UnlaunchableDataSpecError) as exc:
        _run_main(tmp_path, ["--dry_run"])
    msg = str(exc.value)
    assert "synthetic" in msg
    assert "TrainTask" in msg or "not yet implemented" in msg


def test_verify_data_dispatches_on_kind():
    """Unit-level: shard specs route to the existing check, synthetic raise."""
    from experimentation.run.data_verify import (
        UnlaunchableDataSpecError, verify_data)
    from experimentation.run.spec import ShardDataSpec, SyntheticDataSpec

    with pytest.raises(UnlaunchableDataSpecError):
        verify_data(SyntheticDataSpec(generator="mqar"), dry_run=True)

    # A shard spec pointing nowhere: dry_run=True downgrades the missing
    # meta.json to a warning (existing verify_shard behaviour, unchanged).
    shard = ShardDataSpec(shard_dir="/nonexistent/shard", tokenizer="t",
                          n_tokens=1, mix=None)
    verify_data(shard, dry_run=True)          # must not raise


def test_build_train_argv_keeps_its_own_guard():
    """The backstop stays: callers that skip __main__ still cannot slip through."""
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.train_argv import build_train_argv

    spec = resolve_run_spec(SYNTHETIC_SPEC)
    with pytest.raises(ValueError, match="shard"):
        build_train_argv(spec, "/tmp/does-not-matter")


def test_a_shard_spec_still_launches_dry(tmp_path):
    """Regression: the new dispatch must not break the shard path."""
    from experimentation.run.__main__ import main
    result = main(["configs/runs/50m-first-real.yaml", "--run_root", str(tmp_path),
                   "--allow-dirty", "--dry_run"])
    assert (tmp_path / "50m-first-real.81033b58").is_dir(), (
        "shard specs must still materialize a run directory")
    del result
