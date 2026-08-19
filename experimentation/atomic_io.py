"""Atomic file writes: temp file + os.replace.

Lives at the top of `experimentation/` rather than under `run/` because it is not
run-directory policy -- it is imported by six modules spanning the run layer,
the eval layer and the trainer (run/{resolve,train_argv,write_policy,
launchers}.py, evaluation/result.py, training/resume.py). Filing generic write
primitives under `run/` made training/resume.py import from the run layer for
something the run layer does not own.

The guarantee: a kill mid-write (a SLURM preemption, say) can leave a `.tmp<pid>`
file behind but never a truncated or half-written target. `atomic_torch_save`
and `atomic_write_bytes` additionally unlink their temp on failure, so a failed
write cannot corrupt the previous good file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

__all__ = ["atomic_write_text", "atomic_write_json", "atomic_write_bytes",
           "atomic_torch_save"]


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path, data: Dict[str, Any], indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, default=str))


def atomic_write_bytes(path, data: bytes) -> None:
    """Binary counterpart of atomic_write_text, for payloads that are not text.

    Added for the run-provenance design's `source.tar.gz` (4.2): a gzip stream
    is not valid UTF-8, so routing it through atomic_write_text would raise or
    mangle it. The module had text, json and torch writers but no plain binary
    one.

    Unlinks its temp on failure, like atomic_torch_save. That matters more here
    than for a derived file: the archive is provenance, and its sha256 is meant
    to *be* the behavior id, so a truncated archive is worse than a missing one
    -- it would hash cleanly and silently name the wrong code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


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
