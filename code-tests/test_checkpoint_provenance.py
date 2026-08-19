"""Every checkpoint must say which code produced it (run-provenance design 4.1).

The gap this closes: `meta.pt` recorded {step, cfg, cfg_hash, model_type,
model_size, torch_version} and no code identity at all, so a checkpoint
separated from its run directory was unattributable -- and separation is
routine, since `evaluation/evaluate.py` takes a bare `--checkpoint <path>` and
`evaluation/harness.py` rebuilds all its provenance from `meta` alone. Results
were attributable (`evaluation/result.py`'s envelope carries `git_commit`) while
the checkpoints they scored were not.

`meta.pt` is not hashed into anything, so this is purely additive and moves no
identity.

Two kinds of test here, deliberately. `checkpoint_meta` is a pure function and
gets a behavioural test. The other three writers build their dicts inline inside
long training functions that need a GPU and a dataset to reach, so they get a
structural check plus a discovery test asserting the writer set is still exactly
four -- that second one is the part that survives someone adding a fifth writer.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.correctness

_ROOT = Path(__file__).resolve().parent.parent

# The four places that write a checkpoint meta dict (design 4.1). That these
# three build their own dicts rather than calling checkpoint_meta is a
# duplication worth collapsing, but explicitly out of scope for that design.
_META_WRITERS = (
    "experimentation/training/train.py",
    "experimentation/retrieval/adapt.py",
    "experimentation/experiments/table2.py",
    "experimentation/experiments/mqar_finetune.py",
)


def _discover_meta_writers():
    """Files that call torch.save with a 'meta.pt' path.

    AST-based rather than regex: the calls span lines, and a text search cannot
    tell `torch.save(meta, ...)` from `torch.load(... 'meta.pt')` -- four files
    read meta.pt and must not be counted as writers.
    """
    writers = set()
    for path in sorted(_ROOT.glob("experimentation/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "save"):
                continue
            if "meta.pt" in ast.unparse(node):
                writers.add(path.relative_to(_ROOT).as_posix())
    return writers


def test_the_set_of_checkpoint_meta_writers_is_still_the_known_four():
    """If this fails because a fifth writer appeared, the fix is to give that
    writer code_id/dirty too and add it here -- not to relax the assertion."""
    assert _discover_meta_writers() == set(_META_WRITERS)


@pytest.mark.parametrize("relpath", _META_WRITERS)
def test_meta_writer_records_code_id_and_dirty(relpath):
    source = (_ROOT / relpath).read_text(encoding="utf-8")
    assert '"code_id"' in source, (
        f"{relpath} writes a checkpoint meta dict without code_id; a checkpoint "
        f"detached from its run directory would be unattributable")
    assert '"dirty"' in source, (
        f"{relpath} writes a checkpoint meta dict without dirty; a code_id "
        f"recorded from an uncommitted tree does not describe the running code")


def test_checkpoint_meta_records_code_id_and_dirty():
    from koopman_lm.config import build_config
    from experimentation.training.train import checkpoint_meta

    meta = checkpoint_meta(build_config("50m"), step=10,
                           model_type="koopman", model_size="50m")

    assert meta["code_id"], "code_id must be populated (git_commit falls back to 'unknown')"
    assert isinstance(meta["dirty"], bool)


def test_checkpoint_meta_keeps_the_fields_it_already_had():
    """Purely additive: nothing downstream that reads meta.pt may break.
    harness.py and three eval modules key off cfg / cfg_hash / model_type."""
    from koopman_lm.config import build_config, config_hash
    from experimentation.training.train import checkpoint_meta

    cfg = build_config("50m")
    meta = checkpoint_meta(cfg, step=7, model_type="koopman", model_size="50m")

    assert meta["step"] == 7
    assert meta["cfg"] is cfg
    assert meta["cfg_hash"] == config_hash(cfg)
    assert meta["model_type"] == "koopman"
    assert meta["model_size"] == "50m"
    assert meta["torch_version"]


def test_a_dirty_tree_is_recorded_as_dirty(monkeypatch):
    """code_id is a pointer into history; it only describes the running code if
    the tree was clean. Recording the flag beside it is what makes the pointer
    interpretable after the fact."""
    from koopman_lm.config import build_config
    from experimentation.training import train

    monkeypatch.setattr(train, "git_dirty_paths", lambda: [" M koopman_lm/config.py"])
    meta = train.checkpoint_meta(build_config("50m"), step=1,
                                 model_type="koopman", model_size="50m")
    assert meta["dirty"] is True
