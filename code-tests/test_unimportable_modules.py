"""Static checks for modules no test can import.

experimentation/evaluation/lm_harness_eval.py imports `lm_eval` at module level.
That package is not installed in the CPU test environment and nothing else here
depends on it, so **no test can import this module** and the whole suite is blind
to it.

That blindness is not hypothetical. Commit e2dc98a renamed
KoopmanEvalWrapper.__init__'s `dtype` parameter to `weight_dtype` and left
`self._dtype = dtype` behind, so `dtype` became an undefined name and the
constructor raised NameError on every invocation. 843 tests passed, because
nothing imported the file.

A linter catches this instantly, but the repo has no lint gate and neither
pyflakes nor ruff is installed. So: a scope-aware, flow-INsensitive
free-variable check. Flow-insensitive is deliberate -- it unions every name bound
anywhere in a scope instead of tracking order, so it cannot report
use-before-assignment and cannot produce that entire class of false positive. It
reports only names never bound at all, which is the bug that got through.
"""

import ast
import builtins
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# Modules tests cannot import, and the dependency that blocks it. Add here rather
# than widening to the whole repo: a scoped check that runs beats a broad one that
# gets muted after its first false positive.
UNIMPORTABLE = [("experimentation/evaluation/lm_harness_eval.py", "lm_eval")]

FUNC_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
ALL_SCOPES = FUNC_SCOPES + (ast.ClassDef,)

DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
           "__loader__", "__builtins__", "__debug__", "__all__", "__class__"}


def _within(node):
    """Every descendant of `node`, stopping at nested scope boundaries.

    ast.walk() cannot be used for this: it flattens the whole subtree, so a
    nested function's body would be checked against its parent's names and every
    parameter would look undefined.
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        yield cur
        if not isinstance(cur, ALL_SCOPES):
            stack.extend(ast.iter_child_nodes(cur))


def _bound(node):
    """Names this scope binds, order-insensitively."""
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
    """(scope_node, names_visible_inside_it) for the module and every scope."""
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
            # The class body sees its own assignments; its methods do NOT.
            yield cur, visible | _bound(cur)
            yield from _descend(cur, visible)


def undefined_names(path):
    """[(name, lineno)] for every Load of a name no enclosing scope binds."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node, visible in _scopes(tree):
        for cur in _within(node):
            if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Load):
                if cur.id not in visible:
                    found.append((cur.id, cur.lineno))
    return sorted(set(found), key=lambda p: (p[1], p[0]))


@pytest.mark.parametrize("relpath,dep", UNIMPORTABLE)
def test_unimportable_module_has_no_undefined_names(relpath, dep):
    path = REPO / relpath
    assert path.exists(), relpath
    found = undefined_names(path)
    assert not found, (
        f"{relpath} cannot be imported by any test ({dep} is absent), so a "
        "NameError here reaches production unseen. Undefined: "
        + ", ".join(f"{n!r} (line {ln})" for n, ln in found))


def test_the_checker_catches_the_bug_it_was_written_for(tmp_path):
    """A check that never fires proves nothing. This is e2dc98a's defect,
    reduced: a parameter renamed in the signature, still referenced in the body
    under its old name."""
    src = tmp_path / "regressed.py"
    src.write_text(
        "import torch\n"
        "class W:\n"
        "    def __init__(self, weight_dtype='bf16'):\n"
        "        self._model = torch.nn.Linear(2, 2)\n"
        "        self._dtype = dtype\n")             # <- the bug
    assert ("dtype", 5) in undefined_names(src)


def test_the_checker_does_not_flag_ordinary_code(tmp_path):
    """Guards the failure mode that gets checks like this deleted: every
    construct here binds legitimately, so a hit is a false positive."""
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
