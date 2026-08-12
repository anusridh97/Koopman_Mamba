"""Result envelope + eval output layout (§4.2):
<run_dir>/eval/<checkpoint>/<task>.json. Re-scoring a checkpoint OVERWRITES
its result file -- a deliberate exception to the earned-bytes write policy
(§3.7): eval output is cheap and re-derivable.
"""
import pytest

pytestmark = pytest.mark.correctness


def test_write_result_creates_the_keyed_path(tmp_path):
    from experimentation.evaluation.result import eval_result_path, write_result

    path = write_result(
        tmp_path, checkpoint="step_1000", task="zeroshot",
        metrics={"hellaswag": 0.42}, run_id="deadbeef", git_commit="abc123",
    )
    assert path == eval_result_path(tmp_path, "step_1000", "zeroshot")
    assert path == tmp_path / "eval" / "step_1000" / "zeroshot.json"
    assert path.is_file()


def test_write_result_envelope_has_the_common_fields(tmp_path):
    from experimentation.evaluation.result import read_result, write_result

    path = write_result(
        tmp_path, checkpoint="final", task="fineweb_ppl",
        metrics={"ppl": 12.3, "loss": 2.5}, run_id="deadbeef", git_commit="abc123",
    )
    result = read_result(path)
    assert result["run_id"] == "deadbeef"
    assert result["task"] == "fineweb_ppl"
    assert result["checkpoint"] == "final"
    assert result["git_commit"] == "abc123"
    assert "created_at" in result
    assert result["metrics"] == {"ppl": 12.3, "loss": 2.5}


def test_write_result_overwrites_on_rescoring(tmp_path):
    from experimentation.evaluation.result import read_result, write_result

    path = write_result(tmp_path, checkpoint="final", task="fineweb_ppl",
                          metrics={"ppl": 12.3}, run_id="r1", git_commit="c1")
    write_result(tmp_path, checkpoint="final", task="fineweb_ppl",
                  metrics={"ppl": 11.9}, run_id="r1", git_commit="c2")
    assert read_result(path)["metrics"]["ppl"] == 11.9
    assert read_result(path)["git_commit"] == "c2"


def test_write_result_does_not_collide_across_checkpoints_or_tasks(tmp_path):
    from experimentation.evaluation.result import write_result

    p1 = write_result(tmp_path, checkpoint="step_1000", task="zeroshot",
                        metrics={"a": 1}, run_id="r", git_commit="c")
    p2 = write_result(tmp_path, checkpoint="step_2000", task="zeroshot",
                        metrics={"a": 2}, run_id="r", git_commit="c")
    p3 = write_result(tmp_path, checkpoint="step_1000", task="fineweb_ppl",
                        metrics={"a": 3}, run_id="r", git_commit="c")
    assert len({p1, p2, p3}) == 3
