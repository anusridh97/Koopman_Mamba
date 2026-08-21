"""What a finished study hands back: a CSV, a ranked table, a best-trial
pointer, and a promotion sweep.

Two departures from the original harness, both because the run system exists.

**Promotions are a sweep, not a shell script.** The original generated bash --
`cd`, then N train commands and N eval commands in sequence, to be babysat. A
promotion set is a handful of chosen configs re-run longer with several seeds,
which is exactly what a `cells:` sweep is for. Emitting one means the
confirmation runs get content-hashed identities, the dirty-tree gate, and a
single Slurm array, instead of a script whose failure mode is "line 40 died and
lines 41 onward ran anyway". It also means seeds of one promoted config share a
`group_id` and differ by `run_id`, which is how the run system already says "one
experiment, three datapoints" -- a bash loop said nothing at all.

**`best_trial.json` points at the winning run rather than copying its config.**
The full config is already in that run's own materialized `spec.yaml`. A second
copy is a second thing that can disagree with the first, and the run directory is
the thing you actually want to open.

Promotion cells force `exact_invchol` regardless of what was screened, so a
confirmation run confirms the architecture rather than whichever route a screen
happened to use. It is the exact route that costs what the approximation costs
(space.py's docstring has the measurement), which is what lets this be
unconditional instead of a speed/fidelity choice. And `warmup_ratio` is
re-applied against the *promotion* run length, because a warmup baked in at 600
screening steps would occupy a fifth of a 3000-step run.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import optuna
import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.atomic_io import atomic_write_json, atomic_write_text
from experimentation.sweep.search.space import params_to_overrides
from experimentation.sweep.search.study import ANCHOR_ATTR

__all__ = ["trial_row", "write_trials_csv", "write_top_trials_md",
           "write_best_trial", "write_promotion_sweep", "write_report"]


def _completed(study: optuna.study.Study) -> List[Any]:
    """Completed trials, best first."""
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    return sorted(done, key=lambda t: float(t.value))


def trial_row(trial) -> Dict[str, Any]:
    """One trial flattened for a spreadsheet: identity, state, params, attrs."""
    row: Dict[str, Any] = {
        "number": trial.number,
        "state": trial.state.name,
        "objective": trial.value,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": (trial.datetime_complete.isoformat()
                              if trial.datetime_complete else None),
    }
    row.update({f"param_{k}": v for k, v in trial.params.items()})
    row.update({f"attr_{k}": v for k, v in trial.user_attrs.items()})
    return row


def write_trials_csv(study: optuna.study.Study, out_dir) -> Path:
    """Every trial, one row each -- including failures, which are usually the
    interesting ones when a study underperforms."""
    rows = [trial_row(t) for t in study.trials]
    columns = sorted({key for row in rows for key in row}) if rows else [
        "number", "state", "objective"]
    # A study that died before its first trial still gets a header rather than a
    # zero-byte file that reads as "no output" instead of "no trials".
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path = Path(out_dir) / "trials.csv"
    atomic_write_text(path, buffer.getvalue())
    return path


def write_top_trials_md(study: optuna.study.Study, out_dir, *,
                        limit: int = 15) -> Path:
    """A ranked markdown table -- the artefact a human actually reads."""
    completed = _completed(study)
    lines = [
        f"# Search results: {study.study_name}",
        "",
        "Lower is better. The objective is held-out loss unless a penalty weight "
        "was set (see driver.objective_from_metrics).",
        "",
    ]
    if not completed:
        lines += [f"**No completed trials.** {len(study.trials)} trial(s) recorded; "
                  f"see trials.csv for their states and failure reasons.", ""]
        path = Path(out_dir) / "top_trials.md"
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    lines += [
        "| rank | trial | objective | rank | layers | placement | lr | wd | run_id | anchor |",
        "|---:|---:|---:|---:|---:|:--|---:|---:|:--|:--|",
    ]
    for position, trial in enumerate(completed[:limit], start=1):
        params, attrs = trial.params, trial.user_attrs
        lines.append(
            "| {pos} | {num} | {obj:.5f} | {rank} | {layers} | {placement} | "
            "{lr:.3g} | {wd:.3g} | {run_id} | {anchor} |".format(
                pos=position,
                num=trial.number,
                obj=float(trial.value),
                rank=params.get("ska_rank", "-"),
                layers=params.get("n_ska_layers", "-"),
                placement=params.get("placement", "-"),
                lr=float(params.get("learning_rate", 0.0)),
                wd=float(params.get("weight_decay", 0.0)),
                run_id=attrs.get("run_id", "-"),
                anchor=attrs.get(ANCHOR_ATTR, ""),
            ))
    path = Path(out_dir) / "top_trials.md"
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def write_best_trial(study: optuna.study.Study, out_dir) -> Optional[Path]:
    """A pointer to the winning run, or None if nothing completed."""
    completed = _completed(study)
    if not completed:
        return None
    best = completed[0]
    path = Path(out_dir) / "best_trial.json"
    atomic_write_json(path, {
        "study_name": study.study_name,
        "trial_number": best.number,
        "objective": best.value,
        "params": best.params,
        "run_id": best.user_attrs.get("run_id"),
        "run_dir": best.user_attrs.get("run_dir"),
        "anchor": best.user_attrs.get(ANCHOR_ATTR),
    })
    return path


def write_promotion_sweep(study: optuna.study.Study, out_path, *,
                          base_spec,
                          base_model: KoopmanLMConfig,
                          top_k: int,
                          seeds: Sequence[int],
                          max_steps: int,
                          seq_len: Optional[int] = None) -> Optional[Path]:
    """The top `top_k` configs re-run at `max_steps` across `seeds`, as a sweep.

    Returns None (writing nothing) when no trial completed -- there is nothing to
    promote, and an empty sweep would fail at expand time with a less obvious
    message.
    """
    completed = _completed(study)
    if not completed or top_k < 1:
        return None

    cells: List[Dict[str, Any]] = []
    for trial in completed[:top_k]:
        # warmup_ratio re-applied against the PROMOTION length, and the exact
        # route forced: a confirmation run must confirm the architecture, not
        # whichever route a screen used.
        overrides = params_to_overrides(trial.params, base_model,
                                       max_steps=max_steps, seq_len=seq_len,
                                       backend_policy="exact_invchol")
        overrides["optim.max_steps"] = int(max_steps)
        for seed in seeds:
            cells.append({**overrides, "runtime.seed": int(seed)})

    out_path = Path(out_path)
    header = "\n".join([
        "# GENERATED by experimentation.sweep.search.report -- do not hand-edit.",
        f"# Promotion runs for the top {top_k} trial(s) of study "
        f"{study.study_name!r}, {max_steps} steps x {len(seeds)} seed(s).",
        "# Seeds of one promoted config share a group_id and differ by run_id,",
        "# so results.py groups replicates without being told to.",
    ])
    body = yaml.safe_dump({
        "name": f"{study.study_name}-promotions",
        "base": str(base_spec),
        "cells": cells,
    }, sort_keys=False, default_flow_style=False)
    atomic_write_text(out_path, header + "\n\n" + body)
    return out_path


def write_report(study: optuna.study.Study, study_dir, *,
                 base_spec=None,
                 base_model: Optional[KoopmanLMConfig] = None,
                 top_k: int = 3,
                 seeds: Sequence[int] = (42, 43, 44),
                 promotion_max_steps: int = 3000,
                 promotion_seq_len: int = 2048,
                 limit: int = 15) -> Dict[str, Optional[Path]]:
    """Write every artefact; returns a name -> path map (None where skipped)."""
    study_dir = Path(study_dir)
    written: Dict[str, Optional[Path]] = {
        "trials_csv": write_trials_csv(study, study_dir),
        "top_trials": write_top_trials_md(study, study_dir, limit=limit),
        "best_trial": write_best_trial(study, study_dir),
    }
    if base_spec is not None and base_model is not None:
        written["promotions"] = write_promotion_sweep(
            study, study_dir / "promotions.yaml", base_spec=base_spec,
            base_model=base_model, top_k=top_k, seeds=seeds,
            max_steps=promotion_max_steps, seq_len=promotion_seq_len)
    return written
