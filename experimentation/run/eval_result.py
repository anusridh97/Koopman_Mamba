"""Result envelope + eval output layout (§4.2):
<run_dir>/eval/<checkpoint>/<task>.json. Every result file carries a common
envelope (run_id, task, checkpoint, git_commit, created_at) with
task-specific numbers nested under `metrics` -- this is what makes the
aggregation walk (experimentation.results) possible at all. Re-scoring a
checkpoint overwrites its result file: a deliberate exception to the
earned-bytes write policy of §3.7.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from experimentation.atomic_io import atomic_write_json


@dataclass(frozen=True)
class ResultEnvelope:
    run_id: str
    task: str
    checkpoint: str
    git_commit: str
    created_at: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def eval_result_path(run_dir, checkpoint: str, task: str) -> Path:
    return Path(run_dir) / "eval" / checkpoint / f"{task}.json"


def write_result(run_dir, *, checkpoint: str, task: str, metrics: Dict[str, Any],
                  run_id: str, git_commit: str,
                  created_at: Optional[str] = None) -> Path:
    envelope = ResultEnvelope(
        run_id=run_id, task=task, checkpoint=checkpoint, git_commit=git_commit,
        created_at=created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metrics=metrics,
    )
    path = eval_result_path(run_dir, checkpoint, task)
    atomic_write_json(path, envelope.to_dict())
    return path


def read_result(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
