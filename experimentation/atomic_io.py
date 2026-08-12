"""Atomic file writes: temp file + os.replace.

Lives at the top of `experimentation/` rather than under `run/` because it is not
run-directory policy -- it is imported by five modules across two packages
(run/{resolve,eval_result,slurm,launch}.py and training/resume.py). Filing
generic write primitives under `run/` made training/resume.py import from the run
layer for something the run layer does not own.

The guarantee: a kill mid-write (a SLURM preemption, say) can leave a `.tmp<pid>`
file behind but never a truncated or half-written target. `atomic_torch_save`
additionally unlinks its temp on failure, so a failed checkpoint save cannot
corrupt the previous good one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

__all__ = ["atomic_write_text", "atomic_write_json", "atomic_torch_save"]


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path, data: Dict[str, Any], indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, default=str))


def atomic_torch_save(path, obj: Any) -> None:
    """Binary counterpart of atomic_write_text: temp file + os.replace, so a
    kill mid-write (a preemption, e.g.) cannot leave a truncated resume.pt.
    torch is imported lazily so this module keeps working without it."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
