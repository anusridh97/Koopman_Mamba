"""Enforce the koopman_lm / experimentation layering.

koopman_lm/ is the publishable model package: config, models, modules, kernels.
experimentation/ is everything cluster- and study-specific: the trainer, the run
and sweep systems, eval harnesses, retrieval, per-paper experiments.

The dependency edge runs one way -- experimentation -> koopman_lm -- and these
tests are what keep it that way. scripts/check_imports.py answers "does this
symbol exist"; this file answers "should this import exist at all".

Two properties, and the second is the one that actually delivers the goal:

1. No module under koopman_lm/ imports from experimentation/. Checked with
   ast.walk, so a *function-local* import counts too -- the violation this
   split removed (KoopmanLM.encode reaching into retrieval.encoder for
   pool_sequence) was exactly that shape, and a module-level-only check would
   have missed it.

2. koopman_lm/ imports no third-party package beyond torch and yaml. This is
   what makes `pip install koopman-lm` usable by someone who wants a model and
   not a wandb account, an HF datasets stack, or a tokenizer download. If a new
   core dependency is genuinely warranted, add it to CORE_THIRD_PARTY *and* to
   pyproject's [project] dependencies -- deliberately, in one place each.
"""
import ast
import pathlib
import sys

import pytest

pytestmark = pytest.mark.correctness

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "koopman_lm"

#: Hard dependencies: importable at module level, declared in [project] dependencies.
CORE_REQUIRED = {"torch", "yaml"}

#: The [cuda] extra. koopman_lm may use these, but ONLY behind a lazy import --
#: `import koopman_lm` has to succeed on a CPU box with neither installed, which
#: is what the package docstring promises and what CI's import-sanity step
#: checks. A module-level import here would silently make the whole package
#: CUDA-only.
CORE_OPTIONAL = {"triton", "mamba_ssm", "causal_conv1d"}

FIRST_PARTY = {"koopman_lm", "experimentation"}


def _core_files():
    return sorted(CORE.rglob("*.py"))


def _imported_roots(tree):
    """Every top-level module name imported anywhere in `tree`, including inside
    functions and conditionals."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative import; first-party by definition
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_files_exist():
    """Guard against the rglob silently matching nothing and every other test
    in this file passing vacuously."""
    files = _core_files()
    assert len(files) > 20, f"expected the koopman_lm package, found {len(files)} files"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_core_does_not_import_experimentation(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    offenders = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "experimentation" or a.name.startswith("experimentation."):
                    offenders.append((node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mod = node.module or ""
            if mod == "experimentation" or mod.startswith("experimentation."):
                offenders.append((node.lineno, mod))
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} imports from experimentation/ at "
        f"{offenders}. The model core must not depend on the research half -- "
        f"move the shared code into koopman_lm/ instead (as pool_sequence was "
        f"moved to koopman_lm/pooling.py).")


def test_core_third_party_dependencies_are_declared():
    found = {}
    for path in _core_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for root in _imported_roots(tree):
            if root in FIRST_PARTY or root in sys.stdlib_module_names:
                continue
            found.setdefault(root, []).append(str(path.relative_to(REPO_ROOT)))

    allowed = CORE_REQUIRED | CORE_OPTIONAL
    unexpected = {k: v for k, v in found.items() if k not in allowed}
    assert not unexpected, (
        f"koopman_lm/ gained third-party dependencies beyond {sorted(allowed)}: "
        f"{unexpected}. Either move the code to experimentation/, or add the "
        f"dependency here AND to pyproject -- `pip install koopman-lm` must not "
        f"start pulling in the research stack.")


def test_importing_koopman_lm_does_not_reach_the_cuda_extras():
    """`import koopman_lm` must succeed, and stay CPU-only, with none of the
    [cuda] extras installed -- what the package docstring promises and what CI's
    import-sanity step relies on.

    Checked at runtime in a subprocess rather than by AST, deliberately. An AST
    rule ("no module-level `import triton` anywhere in the package") gives false
    positives on modules like kernels/cholesky_update_triton.py, which is a
    triton-only *backend* that is itself only imported lazily by whoever selects
    that backend. What matters is not whether some file imports triton, but
    whether the import graph reachable from `import koopman_lm` does.

    Subprocess so the assertion is about a clean interpreter: by the time this
    test runs, the pytest process has already imported plenty.
    """
    import subprocess
    probe = (
        "import sys, koopman_lm; "
        "print(','.join(m for m in ('triton','mamba_ssm','causal_conv1d') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=REPO_ROOT, check=True)
    reached = [m for m in out.stdout.strip().split(",") if m]
    assert not reached, (
        f"importing koopman_lm pulled in {reached}. These are the [cuda] extra; "
        f"the package must import on a CPU box without them. Make the import "
        f"lazy (inside the function that needs the backend).")


def test_pyproject_does_not_package_experimentation():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    try:
        import tomllib
    except ModuleNotFoundError:                       # py3.10
        assert 'include = ["koopman_lm", "koopman_lm.*"]' in text
        return
    cfg = tomllib.loads(text)
    include = cfg["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["koopman_lm", "koopman_lm.*"], include
    assert "project" in cfg and "scripts" not in cfg["project"], (
        "a console script would resolve at install time and fail at run time if "
        "it pointed into experimentation/, which is not installed")
