"""Build 414632: installing a working CUDA torch, then running
`pip install -e '.[cuda,dev]'` per pyproject.toml's own dependency comment,
silently re-resolved and upgraded torch to a default-index build (cu129 ->
cu130) that the node driver (12.7, forward-compatible only within CUDA 12.x)
can't run -- torch.cuda.is_available() went False with no error anywhere in
the pipeline.

These are static/textual checks, not an actual `pip install` (this suite
runs on a CPU box with no CUDA torch to protect, and scripts/setup_env.sh
needs a real GPU node's driver + an already-installed CUDA torch as its
precondition -- see the script's own first block). They pin the two
structural guarantees the fix relies on: (1) the project itself is installed
with --no-deps so pip can never touch torch to satisfy koopman_lm's own
unbounded "torch>=2.1", and (2) the script ends by asserting CUDA is still
available and still CUDA 12.x, so a future regression fails loudly in the
install step instead of silently in a training job.

The TOML/shell are scanned textually rather than parsed: this project
supports Python 3.10 (pyproject requires-python = ">=3.10") and tomllib is
3.11+; see test_production_configs.py's test_package_data_points_at_the_real_
cuda_sources for the same rationale.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _setup_env_text():
    return (_ROOT / "scripts" / "setup_env.sh").read_text()


def test_setup_env_installs_the_project_with_no_deps():
    text = _setup_env_text()
    m = re.search(r"pip install\s+(?:--no-build-isolation\s+)?--no-deps\s+-e\s+\.\s*$",
                  text, re.MULTILINE)
    assert m, (
        "scripts/setup_env.sh must install koopman_lm itself with --no-deps "
        "so pip never resolves (and can never upgrade) torch to satisfy "
        "koopman_lm's own unbounded 'torch>=2.1' -- see pyproject.toml's "
        "dependency comment for why a plain `pip install -e .` is unsafe on "
        "a box with a working CUDA torch already installed:\n" + text)


def test_setup_env_installs_mamba_ssm_with_no_deps():
    """mamba-ssm/causal-conv1d's own setup.py metadata declares torch as an
    install_requires -- a normal (non --no-deps) install of these would
    silently reintroduce the exact clobber through a different path."""
    text = _setup_env_text()
    m = re.search(r"pip install\s+--no-build-isolation\s+--no-deps\s*\\?\s*\n?"
                  r"\s*\"mamba-ssm[^\"]*\"", text)
    assert m, (
        "scripts/setup_env.sh must install mamba-ssm/causal-conv1d with "
        "--no-deps + --no-build-isolation, not a normal resolved install:\n"
        + text)


def test_setup_env_asserts_cuda_survives_the_install():
    text = _setup_env_text()
    assert "torch.cuda.is_available()" in text, (
        "scripts/setup_env.sh must assert torch.cuda.is_available() is "
        "still True after installing koopman_lm, and fail loudly if not:\n"
        + text)
    assert re.search(r"cuda_version\.startswith\(.12.\)", text), (
        "scripts/setup_env.sh must assert torch.version.cuda still starts "
        "with '12' after installing koopman_lm (this cluster's driver only "
        "forward-compats within CUDA 12.x, not to a 13.x default-index "
        "build) -- see build 414632:\n" + text)
    # The assertion must run AFTER the project install, not before -- it's
    # only meaningful as a check on what pip did.
    install_pos = text.index("-e .")
    assert_pos = text.index("torch.cuda.is_available()", install_pos)
    assert install_pos < assert_pos


def test_pyproject_torch_comment_does_not_claim_install_torch_first_is_sufficient():
    """The comment used to say 'Install torch FIRST for your platform, then
    `pip install -e .`' -- exactly the sequence that killed build 414632.
    It must instead point at scripts/setup_env.sh (the only sequence that
    can't clobber an already-installed torch)."""
    text = (_ROOT / "pyproject.toml").read_text()
    m = re.search(r'dependencies = \[(.*?)"torch>=2\.1",', text, re.DOTALL)
    assert m, "pyproject.toml's dependencies list or its torch comment moved"
    comment = m.group(1)
    assert "setup_env.sh" in comment, (
        "the torch dependency comment must point installers at "
        "scripts/setup_env.sh:\n" + comment)
    assert "no-deps" in comment.replace("--no-deps", "no-deps"), (
        "the torch dependency comment must explain that --no-deps is what "
        "makes the install sequence safe:\n" + comment)
