"""scripts/gen_inventory.py: an AST walk that summarises every module in the
production packages.

The script existed as an untracked top-level loop with no function seam, which
made it both untestable and silently stale: it walked only `koopman_lm`, so
after the package split (e4f5fdc) it reported on less than half the code it
claimed to inventory. Both problems are the same problem -- there was nothing
to call, so there was nothing to check.
"""
import importlib.util
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.correctness

_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = _ROOT / "scripts" / "gen_inventory.py"
    spec = importlib.util.spec_from_file_location("gen_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_summarises_classes_and_functions(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(textwrap.dedent('''
        class Thing(Base):
            def public(self):
                pass

            def _private(self):
                pass


        def top():
            pass


        def _helper():
            pass
    '''))

    records = {r["path"]: r for r in _load().inventory(tmp_path, ("mypkg",))}

    mod = records["mypkg/mod.py"]
    assert mod["classes"] == [{"name": "Thing", "bases": ["Base"],
                               "methods": ["public"]}]
    assert mod["functions"] == ["top"]
    assert mod["private_functions"] == ["_helper"]
    assert mod["empty"] is False


def test_inventory_marks_an_empty_module_empty(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    records = {r["path"]: r for r in _load().inventory(tmp_path, ("mypkg",))}
    assert records["mypkg/__init__.py"]["empty"] is True


def test_inventory_skips_pycache(tmp_path):
    pkg = tmp_path / "mypkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__pycache__" / "junk.py").write_text("x = 1\n")
    (pkg / "real.py").write_text("y = 2\n")
    paths = {r["path"] for r in _load().inventory(tmp_path, ("mypkg",))}
    assert paths == {"mypkg/real.py"}


def test_inventory_covers_experimentation_not_only_koopman_lm():
    """The reason this script is being committed rather than deleted: its
    default package list must track the post-split layout, or it under-reports
    by more than half."""
    paths = {r["path"] for r in _load().inventory(_ROOT)}
    assert any(p.startswith("koopman_lm/") for p in paths)
    assert any(p.startswith("experimentation/") for p in paths), (
        "gen_inventory walked only koopman_lm; experimentation/ has been a "
        "production package since e4f5fdc")


def test_render_is_stable_and_mentions_every_module(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("class A:\n    pass\n")
    (pkg / "b.py").write_text("def b():\n    pass\n")
    module = _load()
    records = module.inventory(tmp_path, ("mypkg",))
    text = module.render(records)
    assert "mypkg/a.py" in text and "mypkg/b.py" in text
    assert module.render(records) == text          # deterministic
