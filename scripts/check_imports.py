#!/usr/bin/env python3
"""Static resolution check for intra-package `koopman_lm` imports.

Written during the 2026-08 branch consolidation. It found two real breakages that no
test covered:

  * scripts/inspect_checkpoint.py imported build_mamba_attention / build_mamba_only
    from koopman_lm.evaluation.evaluate (the pre-split path; now
    experimentation.evaluation.evaluate). The script came from phase1-finalize, whose
    evaluate.py re-exported those names at module level; code-refactor's refactor
    (extracting load_model into evaluation/loader.py) removed that import. So the
    break existed on NEITHER parent -- only in the merge of the two, with no conflict
    raised.
  * code-tests/test_training_resume.py imported koopman_lm.globals.config, a module the
    module reorg deleted.

Both are the same bug class: a refactor moves a symbol, and a caller that no test
exercises keeps pointing at the old home. Import errors in scripts don't fail loudly
until someone runs them, often weeks later on a cluster.

Deliberately AST-only -- it never imports the package, so it needs no torch and runs in
seconds on CPU. That makes it usable as a CI gate on every PR.

Usage:
    python scripts/check_imports.py            # from the repo root
    python scripts/check_imports.py --quiet    # only print failures

Both koopman_lm and experimentation are indexed together, so an import crossing the
boundary (experimentation -> koopman_lm) resolves exactly like an internal one. Only
experimentation may import koopman_lm; the reverse is a layering violation, caught by
code-tests/test_package_boundary.py rather than here -- this script answers "does the
symbol exist", not "should it".

Exit status: 0 if every intra-package import resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import glob
import sys

PACKAGES = ("koopman_lm", "experimentation")
SCAN_GLOBS = [
    *(f"{p}/**/*.py" for p in PACKAGES),
    "code-tests/**/*.py",
    "scripts/**/*.py",
]


def _is_local(module: str) -> bool:
    """True for a module inside one of PACKAGES. Compares against `pkg` and
    `pkg.` rather than a bare startswith, so a hypothetical `koopman_lmx` is not
    mistaken for a `koopman_lm` submodule."""
    return any(module == p or module.startswith(p + ".") for p in PACKAGES)


def _parse(path: str) -> ast.Module | None:
    """Parse a file, tolerating a UTF-8 BOM (7 files in code-tests/ carry one)."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return ast.parse(fh.read())
    except (OSError, SyntaxError) as exc:
        print(f"  !! could not parse {path}: {exc}", file=sys.stderr)
        return None


def _module_name(path: str) -> str:
    return path[:-3].replace("/", ".").replace(".__init__", "")


def _exported_names(body: list[ast.stmt]) -> set[str]:
    """Names bound at MODULE level by this statement list.

    Crucially this does NOT use ast.walk. A name imported inside a function body is
    local to that function and is *not* importable from the module -- counting it
    would make the checker report false-clean, which is exactly the failure it exists
    to prevent. (Concretely: evaluate.py imports the baseline builders inside a
    function, so a walk-based version wrongly considers
    `from experimentation.evaluation.evaluate import build_mamba_attention` resolvable.)

    Top-level `if` / `try` / `with` blocks ARE descended into, because the kernels use
    them for optional Triton paths and lazy imports and those bindings really are
    module-level.
    """
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)          # the def itself, not its interior
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # re-exports count: `from x import y` makes y importable from here
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.If):
            names |= _exported_names(node.body) | _exported_names(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _exported_names(node.body) | _exported_names(node.orelse)
            names |= _exported_names(node.finalbody)
            for handler in node.handlers:
                names |= _exported_names(handler.body)
        elif isinstance(node, ast.With):
            names |= _exported_names(node.body)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    args = ap.parse_args()

    modules: dict[str, set[str]] = {}
    for pkg in PACKAGES:
        found = False
        for path in glob.glob(f"{pkg}/**/*.py", recursive=True):
            tree = _parse(path)
            if tree is not None:
                modules[_module_name(path)] = _exported_names(tree.body)
                found = True
        if not found:
            print(f"ERROR: no {pkg}/ modules found -- run this from the repo root.",
                  file=sys.stderr)
            return 1

    sources = sorted({p for g in SCAN_GLOBS for p in glob.glob(g, recursive=True)})
    failures: list[str] = []

    for path in sources:
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import; resolved by Python, not us
                if node.level or not node.module or not _is_local(node.module):
                    continue
                if node.module not in modules:
                    failures.append(
                        f"{path}:{node.lineno}: no such module {node.module!r}")
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    submodule = f"{node.module}.{alias.name}"
                    if alias.name not in modules[node.module] and submodule not in modules:
                        failures.append(
                            f"{path}:{node.lineno}: {node.module!r} exports no "
                            f"{alias.name!r}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_local(alias.name) and alias.name not in modules:
                        failures.append(
                            f"{path}:{node.lineno}: no such module {alias.name!r}")

    if not args.quiet:
        indexed = ", ".join(
            f"{sum(1 for m in modules if m == p or m.startswith(p + '.'))} {p}"
            for p in PACKAGES)
        print(f"indexed {indexed}; scanned {len(sources)} files")

    if failures:
        print(f"\nUNRESOLVED IMPORTS: {len(failures)}")
        for line in sorted(set(failures)):
            print(f"  {line}")
        return 1

    if not args.quiet:
        print("all intra-package imports resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
