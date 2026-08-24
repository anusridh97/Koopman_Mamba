"""Every module gets imported by something, or is statically checked instead.

Motivating failure: commit e2dc98a renamed KoopmanEvalWrapper.__init__'s `dtype`
parameter to `weight_dtype` and left `self._dtype = dtype` behind. `dtype` became
an undefined name, so the constructor raised NameError on every call and lm-eval
was completely broken. 843 tests passed. Nothing imported the module, because
`lm_eval` is not installed in the CPU environment, so nothing could notice.

Measured afterwards: 10 of 95 modules under experimentation/ and koopman_lm/ were
imported by no test at all, and 8 of those imported perfectly well -- no test had
simply ever loaded them. Four of the ten were modified by the branch that
introduced this bug.

So two gates, cheap and broad:

  1. Import every module. Catches NameError at module scope, syntax errors, bad
     imports, and import-time crashes -- in ~7s for the whole repo.
  2. For the handful that genuinely cannot be imported (an optional third-party
     dependency is absent), fall back to a static undefined-name check.

Importing is not exercising: a module that imports cleanly can still be entirely
untested. This is a floor, not a coverage claim. But it is a floor that this
suite did not have, and the bug above lived under it.
"""

import ast
import builtins
import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = ("experimentation", "koopman_lm")

# Third-party packages that are legitimately absent from the CPU environment. A
# module blocked by one of these is skipped by gate 1 and picked up by gate 2.
# Anything blocked by something NOT on this list is a failure, not a skip.
OPTIONAL_DEPS = {"lm_eval", "triton", "mamba_ssm", "causal_conv1d",
                 "optuna", "flash_attn", "lm_eval_harness"}

# Which modules are ALLOWED to require an optional dependency, and which one.
# Not "which are currently unimportable" -- that varies by environment: the search
# modules import fine with optuna on PYTHONPATH and not without, and both runs are
# legitimate. This says instead: only these modules may sit behind an optional
# dependency at all. A new module appearing here is a deliberate, reviewed act,
# because it means the import gate can no longer see it in some environments.
ALLOWED_OPTIONAL_DEPENDENTS = {
    "experimentation.evaluation.lm_harness_eval": "lm_eval",
    "koopman_lm.kernels.cholesky_update_triton": "triton",
    "experimentation.sweep.search.driver": "optuna",
    "experimentation.sweep.search.report": "optuna",
    "experimentation.sweep.search.study": "optuna",
    # Its input IS an optuna journal -- there is no version of "analyse a
    # finished study" that does not read one, so a lazy import would only move
    # the failure from import time to first call. Sits ABOVE the line with
    # report.py, for the same reason: both consume a Study rather than declaring
    # one. `studyspec`, `space`, `anchors` and `geometry` stay below it, which is
    # what keeps `--dry_run` working with optuna absent.
    "experimentation.sweep.search.analysis": "optuna",
}


def _module_names():
    out = []
    for pkg in PACKAGES:
        for path in sorted((REPO / pkg).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            name = str(path.relative_to(REPO))[:-3].replace("/", ".")
            out.append(name.removesuffix(".__init__"))
    return out


MODULES = _module_names()


def _missing_optional(exc):
    """The optional dependency this exception blames, or None."""
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return None
    root = exc.name.split(".")[0]
    return root if root in OPTIONAL_DEPS else None


def _blocked():
    """{module: dep} for modules an optional dependency currently blocks."""
    out = {}
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:                  # noqa: BLE001
            dep = _missing_optional(exc)
            if dep is not None:
                out[name] = dep
    return out


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except Exception as exc:                      # noqa: BLE001 - report anything
        dep = _missing_optional(exc)
        if dep is None:
            raise
        allowed = ALLOWED_OPTIONAL_DEPENDENTS.get(name)
        assert allowed == dep, (
            f"{name} is blocked by optional dependency {dep!r}, which it is not "
            f"declared to use (declared: {allowed!r}). Either drop the import or "
            "add it to ALLOWED_OPTIONAL_DEPENDENTS -- an undeclared optional "
            "import means this module silently left the import gate.")
        pytest.skip(f"{dep} absent; the static check covers it instead")


def test_the_optional_dependency_allowlist_is_not_stale():
    assert set(ALLOWED_OPTIONAL_DEPENDENTS) <= set(MODULES), (
        f"names a module that no longer exists: "
        f"{sorted(set(ALLOWED_OPTIONAL_DEPENDENTS) - set(MODULES))}")


def test_every_blocked_module_is_statically_checked():
    """Nothing escapes both gates: whatever gate 1 skips, gate 2 must inspect."""
    blocked = _blocked()
    unchecked = sorted(set(blocked) - set(_static_targets()))
    assert not unchecked, f"skipped by gate 1 and unchecked by gate 2: {unchecked}"


# --------------------------------------------------------------- static gate ----
# A scope-aware, flow-INSENSITIVE free-variable check. Flow-insensitive on
# purpose: it unions every binding in a scope instead of tracking order, so it
# cannot report use-before-assignment and cannot produce that class of false
# positive. It reports only names never bound at all -- the bug above.

FUNC_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
ALL_SCOPES = FUNC_SCOPES + (ast.ClassDef,)
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
           "__loader__", "__builtins__", "__debug__", "__all__", "__class__"}


def _within(node):
    """Descendants of `node`, stopping at nested scope boundaries.

    ast.walk() cannot be used: it flattens the subtree, so a nested function's
    body would be checked against its parent's names and every parameter would
    look undefined. (The first version of this check did exactly that.)
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        yield cur
        if not isinstance(cur, ALL_SCOPES):
            stack.extend(ast.iter_child_nodes(cur))


def _bound(node):
    out = set()
    if isinstance(node, FUNC_SCOPES):
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            out.add(arg.arg)
        for arg in (a.vararg, a.kwarg):
            if arg is not None:
                out.add(arg.arg)
    for cur in _within(node):
        if isinstance(cur, ast.Name) and isinstance(cur.ctx, (ast.Store, ast.Del)):
            out.add(cur.id)
        elif isinstance(cur, (ast.Import, ast.ImportFrom)):
            for alias in cur.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(cur, (ast.Global, ast.Nonlocal)):
            out.update(cur.names)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            out.add(cur.name)
        elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(cur.name)
    return out


def _scopes(module):
    top = _bound(module) | set(dir(builtins)) | DUNDERS
    yield module, top
    yield from _descend(module, top)


def _descend(node, visible):
    for cur in _within(node):
        if isinstance(cur, FUNC_SCOPES):
            inner = visible | _bound(cur)
            yield cur, inner
            yield from _descend(cur, inner)
        elif isinstance(cur, ast.ClassDef):
            # A class body sees its own assignments; its methods do not.
            yield cur, visible | _bound(cur)
            yield from _descend(cur, visible)


def undefined_names(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node, visible in _scopes(tree):
        for cur in _within(node):
            if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Load):
                if cur.id not in visible:
                    found.append((cur.id, cur.lineno))
    return sorted(set(found), key=lambda p: (p[1], p[0]))


def _static_targets():
    """Every module gets the static check, not just the unimportable ones.

    The import gate only executes module scope, so a NameError inside a function
    body survives it -- and that is exactly what e2dc98a's bug was. Running the
    AST check over all of them costs milliseconds and needs no test to call the
    function.
    """
    return MODULES


def _module_path(name):
    p = REPO / (name.replace(".", "/") + ".py")
    return p if p.exists() else REPO / (name.replace(".", "/") + "/__init__.py")


@pytest.mark.parametrize("name", _static_targets())
def test_module_has_no_undefined_names(name):
    path = _module_path(name)
    assert path.exists(), name
    found = undefined_names(path)
    dep = ALLOWED_OPTIONAL_DEPENDENTS.get(name)
    why = (f" This module sits behind optional dependency {dep!r}, so no test "
           "can even import it in some environments." if dep else "")
    assert not found, (
        f"undefined name(s) in {name}.{why} A NameError in a function body is "
        "invisible to the import gate and to every test that does not call that "
        "function. Undefined: "
        + ", ".join(f"{n!r} (line {ln})" for n, ln in found))


def test_the_static_check_catches_the_bug_it_was_written_for(tmp_path):
    """A check that never fires proves nothing. e2dc98a's defect, reduced: a
    parameter renamed in the signature, still referenced under its old name."""
    src = tmp_path / "regressed.py"
    src.write_text(
        "import torch\n"
        "class W:\n"
        "    def __init__(self, weight_dtype='bf16'):\n"
        "        self._model = torch.nn.Linear(2, 2)\n"
        "        self._dtype = dtype\n")            # <- the bug
    assert ("dtype", 5) in undefined_names(src)


def test_the_static_check_does_not_flag_ordinary_code(tmp_path):
    """Guards the failure mode that gets checks like this muted: everything here
    binds legitimately, so any hit is a false positive."""
    src = tmp_path / "fine.py"
    src.write_text(
        "import os\n"
        "from typing import Optional\n"
        "CONST = 1\n"
        "def outer(a, *args, b=2, **kw):\n"
        "    total = a + b + CONST\n"
        "    squares = [x * x for x in range(total)]\n"
        "    with open(os.devnull) as fh:\n"
        "        data = fh.read()\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as exc:\n"
        "        print(exc)\n"
        "    def inner(y):\n"
        "        return y + total\n"
        "    if (n := len(data)):\n"
        "        total += n\n"
        "    return inner(len(args) + len(kw) + len(squares))\n"
        "class C:\n"
        "    attr = 3\n"
        "    def method(self, z: Optional[int] = None):\n"
        "        return z or self.attr\n"
        "    def other(self):\n"
        "        return [self.method(i) for i in range(3)]\n")
    assert undefined_names(src) == []
