"""python -m koopman_lm.results <root>: walk run directories, read each
spec.yaml + eval/**/*.json, and emit one row per (run, checkpoint, task)
(§4.2). The filesystem is the store; this is a walk, not a database.
"""
import pytest

pytestmark = pytest.mark.correctness


def _make_run(run_root, name, group_id, run_id, seed, lr, d_model, results):
    from koopman_lm.run.eval_result import write_result

    run_dir = run_root / f"{name}.{group_id}" / f"seed{seed}.{run_id}"
    run_dir.mkdir(parents=True)
    spec_yaml = run_dir / "spec.yaml"
    spec_yaml.write_text(
        "name: {name}\n"
        "run_id: {run_id}\n"
        "group_id: {group_id}\n"
        "model:\n  d_model: {d_model}\n  n_layers: 17\n"
        "data:\n  kind: shard\n"
        "optim:\n  lr: {lr}\n"
        "runtime:\n  seed: {seed}\n"
        "provenance:\n  git_commit: traincommit\n".format(
            name=name, run_id=run_id, group_id=group_id, d_model=d_model,
            lr=lr, seed=seed))
    for checkpoint, task, metrics in results:
        write_result(run_dir, checkpoint=checkpoint, task=task, metrics=metrics,
                      run_id=run_id, git_commit="evalcommit")
    return run_dir


def test_aggregate_emits_one_row_per_run_checkpoint_task(tmp_path):
    from koopman_lm.results import aggregate

    _make_run(tmp_path, "50m-fineweb-3b", "gA", "rA", seed=42, lr=0.0004,
               d_model=384, results=[
                   ("step_1000", "fineweb_ppl", {"ppl": 20.1}),
                   ("final", "fineweb_ppl", {"ppl": 15.0}),
                   ("final", "zeroshot", {"hellaswag": 0.4}),
               ])
    _make_run(tmp_path, "50m-fineweb-3b", "gB", "rB", seed=1337, lr=0.0003,
               d_model=384, results=[("final", "fineweb_ppl", {"ppl": 16.2})])

    rows = aggregate(tmp_path)
    assert len(rows) == 4
    by_task = {(r["run_id"], r["checkpoint"], r["task"]): r for r in rows}
    row = by_task[("rA", "final", "fineweb_ppl")]
    assert row["ppl"] == 15.0
    assert row["lr"] == 0.0004
    assert row["seed"] == 42
    assert row["d_model"] == 384
    assert row["group_id"] == "gA"


def test_aggregate_skips_runs_with_no_eval_dir(tmp_path):
    from koopman_lm.results import aggregate

    run_dir = tmp_path / "50m.gA" / "seed42.rA"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.yaml").write_text("name: 50m\nrun_id: rA\ngroup_id: gA\n"
                                          "model: {d_model: 384}\ndata: {kind: shard}\n"
                                          "optim: {lr: 0.0004}\nruntime: {seed: 42}\n")
    rows = aggregate(tmp_path)
    assert rows == []


def test_main_prints_a_table(tmp_path, capsys):
    from koopman_lm.results import main

    _make_run(tmp_path, "50m-fineweb-3b", "gA", "rA", seed=42, lr=0.0004,
               d_model=384, results=[("final", "fineweb_ppl", {"ppl": 15.0})])
    rc = main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id" in out
    assert "rA" in out
    assert "fineweb_ppl" in out


def test_main_reports_no_results_found(tmp_path, capsys):
    from koopman_lm.results import main

    rc = main([str(tmp_path)])
    assert rc == 0
    assert "no eval results" in capsys.readouterr().out
