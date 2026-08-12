"""archive/ holds code kept for provenance. It must never be on an import path.

This is what makes restoring a file under archive/ a provably behaviour-free act:
if koopman_lm/ cannot import from archive/, then adding, editing, or deleting an
archived file cannot change what a model does. Without this test that guarantee
is a convention, and conventions are what produced two divergent copies of this
project in the first place.

Loading an archived file by PATH is allowed and is deliberately not tested
against -- code-tests/test_jax_reference.py does exactly that with the JAX
oracle. What is forbidden is a package import.
"""
import ast
import pathlib

import pytest

pytestmark = pytest.mark.correctness

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _python_files(root):
    return [p for p in (_REPO / root).rglob("*.py")
            if "__pycache__" not in p.parts]


def _imported_modules(path):
    """Every dotted name this file imports, via either import form."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # node.module is None for `from . import x`; level>0 is relative.
            if node.module:
                names.append(node.module)
    return names


def test_the_package_never_imports_from_archive():
    offenders = []
    for path in _python_files("koopman_lm"):
        for name in _imported_modules(path):
            if name == "archive" or name.startswith("archive."):
                offenders.append(f"{path.relative_to(_REPO)} imports {name}")
    assert not offenders, (
        "koopman_lm/ must not import from archive/ -- archived code is kept for "
        "provenance only, and importing it would make a provably-inert "
        "directory load-bearing:\n  " + "\n  ".join(offenders))


def test_archive_is_not_an_importable_package():
    """No __init__.py at archive/ or its top level, so `import archive.x` cannot
    resolve even by accident."""
    archive = _REPO / "archive"
    if not archive.is_dir():
        pytest.skip("no archive/ directory")
    assert not (archive / "__init__.py").exists()
    stray = [p.relative_to(_REPO) for p in archive.glob("*/__init__.py")]
    # prefix_bench ships its own __init__.py as part of the original package;
    # that is fine because archive/ itself is not a package, so the chain
    # archive.MQAR.prefix_bench is unreachable. Assert the chain is broken.
    for p in stray:
        assert not (_REPO / p.parent.parent / "__init__.py").exists(), (
            f"{p} would become importable because its parent is a package")


def test_archive_is_excluded_from_the_wheel():
    """setuptools must not ship archive/ -- pyproject's package discovery is
    scoped to koopman_lm*, so an archived tree cannot leak into an install."""
    text = (_REPO / "pyproject.toml").read_text()
    assert 'include = ["koopman_lm", "koopman_lm.*"]' in text, (
        "pyproject.toml's package discovery changed; re-verify that archive/ is "
        "still excluded from the built distribution")


def test_every_archived_tree_is_documented():
    """A directory under archive/ with no entry in archive/README.md is how
    'why is this here?' becomes unanswerable a month later."""
    archive = _REPO / "archive"
    if not archive.is_dir():
        pytest.skip("no archive/ directory")
    readme = (archive / "README.md").read_text()
    undocumented = [d.name for d in archive.iterdir()
                    if d.is_dir() and d.name not in readme]
    assert not undocumented, (
        f"archive/ subdirectories missing from archive/README.md: {undocumented}")
