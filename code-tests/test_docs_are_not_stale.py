"""Pin the doc claims that have rotted more than once.

READING-PROGRESS's headline finding was "docstrings are accurate, prose docs
have drifted" -- nearly every line number, count and shell command in the
markdown had gone stale while the docstrings held. The difference is that a
docstring sits next to the thing it describes and a prose doc does not.

Correcting the prose is a one-off; this file is the part that lasts. It pins
only *mechanically checkable* claims -- counts, and the existence of paths a
doc tells you to open -- not prose. A test that pinned wording would fail on
every edit and get deleted, which is worse than no test.

The config-count claim is here because it has now been wrong twice: written as
4 when the registry held 4, left at 4 when it grew to 11.
"""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.correctness

_ROOT = Path(__file__).resolve().parent.parent
_GUIDE = _ROOT / "docs" / "CODEBASE_GUIDE.md"


def test_codebase_guide_config_counts_match_reality():
    from koopman_lm.config import CONFIG_REGISTRY

    text = _GUIDE.read_text()
    match = re.search(
        r"registry has exactly (\d+) entries and `configs/` ships exactly (\d+)",
        text)
    assert match, (
        "CODEBASE_GUIDE.md must state the registry/config counts in the form "
        "\"The registry has exactly N entries and `configs/` ships exactly M "
        "YAML files\" so this test can check them. Reword and keep the shape.")
    claimed_registry, claimed_files = int(match.group(1)), int(match.group(2))
    actual_files = len(list((_ROOT / "configs").glob("*.yaml")))
    assert claimed_registry == len(CONFIG_REGISTRY), (
        f"CODEBASE_GUIDE.md claims {claimed_registry} registry entries; there "
        f"are {len(CONFIG_REGISTRY)}")
    assert claimed_files == actual_files, (
        f"CODEBASE_GUIDE.md claims configs/ ships {claimed_files} YAML files; "
        f"it ships {actual_files}")


def test_layout_docs_do_not_advertise_the_pre_split_package_layout():
    """training/, evaluation/, experiments/ and retrieval/ moved out of
    koopman_lm/ into a top-level `experimentation` package in e4f5fdc. A layout
    block that still nests them under koopman_lm/ sends a reader to paths that
    do not exist -- the single most repeated confusion in the handoff docs."""
    for name in ("README.md", "docs/CODEBASE_GUIDE.md"):
        text = (_ROOT / name).read_text()
        assert "experimentation/" in text, (
            f"{name} never mentions experimentation/, so its layout section "
            f"predates the package split (e4f5fdc)")


# Directories a doc may legitimately name that hold no production code.
_PRODUCTION_TREES = ("koopman_lm", "experimentation", "scripts", "code-tests")


def _exists_in_production(token: str) -> bool:
    """Does this doc token name a real production file?

    A layout block writes some entries as full relative paths
    (`koopman_lm/config.py`) and others as bare filenames listed under an
    indented directory heading (`mamba.py, ska.py, ...`). Accept either, but
    only ever look inside the production trees -- archive/ is proven-inert by
    test and deliberately still holds deleted modules, so resolving against it
    would let a doc keep advertising a file the live code no longer has.
    """
    if (_ROOT / token).exists():
        return True
    name = Path(token).name
    return any((_ROOT / tree).is_dir() and any((_ROOT / tree).rglob(name))
               for tree in _PRODUCTION_TREES)


@pytest.mark.parametrize("doc", ["README.md", "docs/CODEBASE_GUIDE.md"])
def test_layout_docs_only_name_modules_that_exist(doc):
    """Every .py token in an indented layout line must resolve to real
    production code. This is what catches modules/seq/fast.py (deleted in
    c34fd46) and modules/mlp/koopman_diag.py (moved into
    koopman_lm/diagnostics/ in 97eb190) still being advertised as if present.

    Several tokens per line, since layout blocks list siblings comma-separated
    under a directory heading rather than one per line.

    Scoped to fenced code blocks. Prose may legitimately name a file that is
    gone -- CODEBASE_GUIDE 297-311 discusses which archived modules were
    restored, and archive/ is deliberately allowed to hold superseded code. A
    fenced layout block is different: it is a map, and a map naming a road that
    does not exist is simply wrong."""
    text = (_ROOT / doc).read_text()
    referenced = set()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or not line.startswith("  "):
            continue
        referenced.update(re.findall(r"[\w./]+\.py", line))
    missing = sorted(t for t in referenced if not _exists_in_production(t))
    assert not missing, (
        f"{doc} names modules that no longer exist in the production trees: "
        f"{missing}")
