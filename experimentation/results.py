"""python -m experimentation.results <root>: walk run directories, read each
run's spec.yaml and eval/**/*.json, and emit one table -- a row per (run,
checkpoint, task), with columns for the swept axes (§4.2). No database, no
service: the filesystem is the store, and this is a walk over it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from experimentation.evaluation.result import iter_results


def discover_spec_files(root) -> List[Path]:
    return sorted(Path(root).rglob("spec.yaml"))


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def aggregate(root) -> List[Dict[str, Any]]:
    """One row per (run, checkpoint, task): swept-axis columns from spec.yaml
    plus the flattened metrics of each eval/<checkpoint>/<task>.json."""
    rows: List[Dict[str, Any]] = []
    for spec_path in discover_spec_files(root):
        run_dir = spec_path.parent
        spec = yaml.safe_load(spec_path.read_text()) or {}
        axes = {
            "run": run_dir.name,
            "name": spec.get("name"),
            "run_id": spec.get("run_id"),
            "group_id": spec.get("group_id"),
            "d_model": (spec.get("model") or {}).get("d_model"),
            "n_layers": (spec.get("model") or {}).get("n_layers"),
            "data_kind": (spec.get("data") or {}).get("kind"),
            "lr": (spec.get("optim") or {}).get("lr"),
            "seed": (spec.get("runtime") or {}).get("seed"),
            "sweep_id": spec.get("sweep_id"),
            "sweep_name": spec.get("sweep_name"),
        }
        for checkpoint, task, result_path in iter_results(run_dir):
            result = json.loads(result_path.read_text())
            row = dict(axes)
            row.update({
                "checkpoint": checkpoint,
                "task": task,
                "eval_git_commit": result.get("git_commit"),
                "eval_created_at": result.get("created_at"),
            })
            row.update(_flatten_metrics(result.get("metrics", {})))
            rows.append(row)
    return rows


def _format_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no eval results found)"
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in columns}
    lines = ["  ".join(c.ljust(widths[c]) for c in columns)]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m experimentation.results")
    p.add_argument("root", help="$RUN_ROOT to walk")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = aggregate(args.root)
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
