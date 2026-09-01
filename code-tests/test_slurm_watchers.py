"""Small static gates for the branch-portable Slurm monitoring helpers."""
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.correctness

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["_watch_job.sh", "_watch_array.sh"])
def test_watcher_is_valid_bash(name):
    result = subprocess.run(
        ["bash", "-n", str(REPO / "scripts" / name)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_array_watcher_expands_each_slurm_array_element():
    text = (REPO / "scripts" / "_watch_array.sh").read_text()
    assert text.count(" -r ") >= 2, (
        "without squeue -r a pending array range counts as one task")
