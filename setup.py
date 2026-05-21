"""
setup.py for the Echo / SKA 440M patch set.

NOTE ON HONESTY: the dependency list below is grounded in the actual imports in
koopman_lm/*.py (verified by static scan), NOT guessed. BUT two of the heaviest
deps -- mamba_ssm and causal-conv1d -- are CUDA-only and could NOT be installed
or import-tested in the environment this package was built in. Their version
compatibility with your torch/CUDA build is the single most likely install
failure, and it cannot be verified here. Pin them against YOUR CUDA + torch
versions; see the comments on each.

Install (editable, recommended so it overlays your existing koopman-lm repo):
    pip install -e .
Then also install the CUDA-only extras that pip cannot resolve cleanly:
    pip install mamba-ssm causal-conv1d   # MUST match your torch/CUDA -- see below

This is a PATCH SET: it expects your repo to already provide koopman_lm.baselines,
koopman_lm.koopman_mlp, and (optionally) koopman_lm.adaptive_chunking. setup.py
does not vendor those.
"""
from setuptools import setup, find_packages

# --- Hard imports, verified present by static scan of koopman_lm/*.py ---
# torch:        every module. transformers: pretokenize.py (Llama-2 tokenizer).
# numpy:        dataset_weighted.py / pretokenize.py. triton: cholesky_update_triton.py
#               (lazily imported; only needed for the GPU Cholesky kernel path).
INSTALL_REQUIRES = [
    # torch is intentionally NOT version-pinned here: it must match your CUDA
    # and your mamba_ssm/triton build. Install torch FIRST, for your CUDA, then
    # `pip install -e .`. A pin here would fight your CUDA wheel.
    "torch>=2.1",
    "numpy>=1.24",
    "transformers>=4.40",   # AutoTokenizer for meta-llama/Llama-2 (or NousResearch mirror)
]

# --- CUDA-only, version-sensitive: NOT auto-installed (pip can't resolve the
#     CUDA match reliably). Listed as an extra so the failure is explicit. ---
# triton ships with recent torch CUDA wheels; if yours doesn't have it, add it.
EXTRAS_REQUIRE = {
    "cuda": [
        # mamba_ssm: REQUIRED to build the model (model.py: `from mamba_ssm import
        # Mamba2`). Must match your torch+CUDA. Build-from-source if no matching
        # wheel. This is the most common install failure -- pin to a version
        # tested against your torch.
        "mamba-ssm>=2.2.2",
        "causal-conv1d>=1.4.0",  # mamba_ssm's fast conv dep; same CUDA-match caveat
        "triton>=2.2",           # GPU Cholesky kernel (cholesky_update_triton.py)
    ],
    "data": [
        "datasets>=2.18",        # if you pull FineWeb-Edu/PG-19/SCROLLS via HF
    ],
}

setup(
    name="koopman_lm_echo_ska_440m",
    version="0.4.0",
    description="Echo/SKA 440M patch set: Spectral Koopman Attention LM + "
                "inference-time ridge memory (overlay on koopman-lm-fast).",
    packages=find_packages(include=["koopman_lm", "koopman_lm.*"]),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
