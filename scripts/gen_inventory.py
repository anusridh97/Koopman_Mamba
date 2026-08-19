#!/usr/bin/env python3
"""Print a one-screen-per-package inventory of the production code: every
module, its length, its classes with public methods, and its functions.

Written during the 2026-08 codebase read-through, to answer "what is actually
in here?" without opening 60 files. Kept because that question recurs every
time someone new picks the repo up, and because a stale hand-written layout
table in a markdown file is exactly the drift this repo keeps finding (see
READING-PROGRESS: "docstrings are accurate, prose docs have drifted").

    python scripts/gen_inventory.py .
    python scripts/gen_inventory.py . --packages koopman_lm

`DEFAULT_PACKAGES` is the load-bearing detail. The original version of this
script walked only `koopman_lm`, which was correct when it was written and
wrong from `e4f5fdc` onward -- the split moved training/, evaluation/,
experiments/ and retrieval/ into a top-level `experimentation` package, so the
script silently under-reported by more than half. test_gen_inventory.py pins
both packages for that reason.

Lives in scripts/ beside check_imports.py and gen_identity_baseline.py, the
repo's other AST/provenance dev utilities. It is a reporting tool: it imports
nothing from the packages it reads, so it stays runnable with no torch and no
GPU.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

DEFAULT_PACKAGES = ("koopman_lm", "experimentation")

_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def summarise_module(source: str, rel_path: str) -> Dict[str, Any]:
    """Summarise one module's top-level shape from its source text.

    Pure: takes text, returns a record. Nothing here touches the filesystem,
    which is what makes the whole script testable on synthetic trees.
    """
    tree = ast.parse(source)
    lines = source.count("\n")

    classes: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        classes.append({
            "name": node.name,
            "bases": [ast.unparse(b) for b in node.bases],
            # Public methods only: the private ones are implementation detail
            # at this altitude, and listing them buries the interface.
            "methods": [m.name for m in node.body
                        if isinstance(m, _FUNC_NODES) and not m.name.startswith("_")],
        })

    functions = [n.name for n in tree.body
                 if isinstance(n, _FUNC_NODES) and not n.name.startswith("_")]
    private = [n.name for n in tree.body
               if isinstance(n, _FUNC_NODES) and n.name.startswith("_")]

    return {
        "path": rel_path,
        "lines": lines,
        # An __init__.py that only re-exports, or a genuinely empty husk, is
        # worth showing as such rather than as "0 lines" noise -- the repo has
        # real empty-orphan directories (finding 4.12) and this is how they surface.
        "empty": not classes and not functions and not private and lines < 3,
        "classes": classes,
        "functions": functions,
        "private_functions": private,
    }


def inventory(root, packages: Sequence[str] = DEFAULT_PACKAGES) -> List[Dict[str, Any]]:
    """Walk `packages` under `root` and summarise every module, sorted by path.

    Deterministic: directories and filenames are sorted, and __pycache__ is
    pruned, so two runs over an unchanged tree produce identical output and a
    diff of two inventories means the code actually moved.
    """
    root = Path(root)
    records: List[Dict[str, Any]] = []
    for package in packages:
        base = root / package
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                rel = path.relative_to(root).as_posix()
                records.append(summarise_module(
                    path.read_text(encoding="utf-8"), rel))
    return sorted(records, key=lambda r: r["path"])


def render(records: Sequence[Dict[str, Any]], width: int = 58) -> str:
    """Render an inventory as the original script's text layout."""
    out: List[str] = []
    for record in records:
        if record["empty"]:
            out.append(f"{record['path']:{width}} (empty)")
            continue
        out.append(f"{record['path']:{width}} {record['lines']:5} lines")
        for cls in record["classes"]:
            out.append(f"    class {cls['name']}({', '.join(cls['bases'])})")
            methods = cls["methods"]
            if methods:
                shown = ", ".join(methods[:9])
                out.append(f"          methods: {shown}"
                           f"{' ...' if len(methods) > 9 else ''}")
        if record["functions"]:
            out.append(f"    fn: {', '.join(record['functions'])}")
        if record["private_functions"]:
            private = record["private_functions"]
            out.append(f"    fn(private): {', '.join(private[:8])}"
                       f"{' ...' if len(private) > 8 else ''}")
    return "\n".join(out) + ("\n" if out else "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/gen_inventory.py", description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".",
                        help="repository root to walk")
    parser.add_argument("--packages", nargs="+", default=list(DEFAULT_PACKAGES),
                        help="packages to inventory")
    args = parser.parse_args(argv)
    sys.stdout.write(render(inventory(args.root, args.packages)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
