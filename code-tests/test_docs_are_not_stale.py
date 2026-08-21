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
import os
import re
import subprocess
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


# --------------------------------------------------------------------------
# Documented shell commands. The .py-token check above is static and covers
# layout blocks; these cover the commands a reader is told to *run*.
#
# Motivating failure: README's "## Train" section documents
# `bash scripts/train_50m.sh`, and it exited non-zero for two independent
# reasons at once. f43a27e deleted the byte-identical `50m_prefix_scan` /
# `180m_prefix_scan` YAMLs and their registry aliases, updated pretrain.sh's
# `case "$SIZE"`, and missed the two wrappers -- which kept exec'ing
# `pretrain.sh 50m_prefix_scan`, hitting `*) unknown production config; exit 1`.
# And `scripts/pretrain.sh` is mode 100644, so `exec "$dir/pretrain.sh"` never
# reached the case statement at all: 126, Permission denied.
#
# Nothing noticed, because nothing ran the documented command. So one test does,
# with a `python` that records its argv instead of training.
# --------------------------------------------------------------------------

_SHIM = """#!/bin/sh
printf '%s\\n' "$@" >> "$ARGV_LOG"
"""


def _run_launcher(script, data_tag, tmp_path):
    """Run a documented launcher for real, with `python` stubbed out.

    Everything up to the training process is genuine: the wrapper, the exec into
    pretrain.sh, the `case "$SIZE"` lookup, the per-size defaults, and the
    shard-exists branches. Only the final `python -m ...` is replaced, by a shim
    that appends its argv to a file -- so this needs no GPU, no network, no data
    and not even a working Python environment, and it still fails if the
    documented command cannot resolve.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "python"
    shim.write_text(_SHIM)
    shim.chmod(0o755)

    data_root = tmp_path / "data"
    for split in ("train", "val"):
        d = data_root / f"fineweb_{data_tag}_{split}"
        d.mkdir(parents=True)
        (d / "train.bin").touch()          # skip both tokenization steps

    argv_log = tmp_path / "argv.txt"
    env = dict(os.environ)
    env.update(PATH=f"{bindir}:{env['PATH']}",
               ARGV_LOG=str(argv_log),
               BUILD_PREFIX_SCAN="0",       # the CUDA build needs a GPU
               DATA_ROOT=str(data_root),
               RUN_ROOT=str(tmp_path / "runs"))

    proc = subprocess.run(["bash", str(_ROOT / script)], env=env, cwd=_ROOT,
                          capture_output=True, text=True, timeout=120)
    argv = argv_log.read_text().splitlines() if argv_log.exists() else []
    return proc, argv


def _flag(argv, name):
    return argv[argv.index(name) + 1]


# The production recipe, recorded here so a future "fix" to the launcher path
# cannot quietly change what a run does. Sourced from scripts/pretrain.sh's
# per-size defaults; if you change a number there on purpose, change it here too
# and say why in the commit message.
_RECIPE = {
    "scripts/train_50m.sh": dict(
        size="50m", data_tag="50m_quality", max_steps="15000",
        learning_rate="4e-4", warmup_steps="300",
        per_device_train_batch_size="16", gradient_accumulation_steps="6"),
    "scripts/train_180m.sh": dict(
        size="180m", data_tag="180m_quality", max_steps="51000",
        learning_rate="3e-4", warmup_steps="1000",
        per_device_train_batch_size="8", gradient_accumulation_steps="12"),
}


@pytest.mark.parametrize("script", sorted(_RECIPE))
def test_documented_launcher_reaches_the_training_command(script, tmp_path):
    recipe = dict(_RECIPE[script])
    size = recipe.pop("size")
    proc, argv = _run_launcher(script, recipe.pop("data_tag"), tmp_path)

    assert proc.returncode == 0, (
        f"README documents `bash {script}` but it exited "
        f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}")
    assert "unknown production config" not in proc.stdout + proc.stderr, (
        f"{script} passes pretrain.sh a size its `case` does not accept")
    assert argv, (
        f"{script} exited 0 but never reached the training command; stdout:\n"
        f"{proc.stdout}")

    assert argv[:2] == ["-m", "experimentation.training.train"], argv[:2]
    assert _flag(argv, "--model_size") == size
    for flag, expected in recipe.items():
        assert _flag(argv, f"--{flag}") == expected, (
            f"{script} now trains with --{flag}={_flag(argv, '--' + flag)}, "
            f"not the recorded production value {expected}")


@pytest.mark.parametrize("script", sorted(_RECIPE))
def test_documented_launcher_names_a_size_the_registry_knows(script, tmp_path):
    """--model_size goes straight into build_config, so a name the registry has
    never heard of is a crash minutes into a job, after tokenization."""
    from koopman_lm.config import CONFIG_REGISTRY

    _, argv = _run_launcher(script, _RECIPE[script]["data_tag"], tmp_path)
    size = _flag(argv, "--model_size")
    assert size in CONFIG_REGISTRY, (
        f"{script} trains --model_size {size!r}, which build_config rejects; "
        f"known: {sorted(CONFIG_REGISTRY)}")


def test_train_pys_own_model_size_default_is_a_real_config(monkeypatch):
    """`python -m experimentation.training.train` with no --model_size must not
    crash. Its default outlived the config it named."""
    import sys

    from koopman_lm.config import CONFIG_REGISTRY
    from experimentation.training.train import parse_args

    monkeypatch.setattr(sys, "argv", ["experimentation.training.train"])
    default = parse_args().model_size
    assert default in CONFIG_REGISTRY, (
        f"--model_size defaults to {default!r}, which build_config rejects")


@pytest.mark.parametrize("doc", ["README.md", "docs/CODEBASE_GUIDE.md"])
def test_documented_shell_commands_name_files_that_exist(doc):
    """Static sweep for the rest: every `bash scripts/X`, `python scripts/X` and
    `python -m pkg.mod` a doc tells you to run must resolve."""
    text = (_ROOT / doc).read_text()
    missing = []
    for path in set(re.findall(r"(?:bash|python) (scripts/[\w./-]+)", text)):
        if not (_ROOT / path).is_file():
            missing.append(path)
    for mod in set(re.findall(r"python -m ([\w.]+)", text)):
        parts = mod.split(".")
        if not ((_ROOT / Path(*parts)).with_suffix(".py").is_file()
                or (_ROOT / Path(*parts) / "__main__.py").is_file()):
            missing.append(f"-m {mod}")
    assert not missing, f"{doc} documents commands that cannot run: {sorted(missing)}"
