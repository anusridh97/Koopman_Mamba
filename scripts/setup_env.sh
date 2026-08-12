#!/usr/bin/env bash
# Install the project after a CUDA-enabled, SM100-capable PyTorch build is
# already present. This script deliberately does not pin a PyTorch wheel: the
# correct index depends on the target cluster and driver stack.
set -euo pipefail

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Install a CUDA-enabled PyTorch build for the target machine first"
    ) from exc
if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see an NVIDIA CUDA device")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

# --no-deps: koopman_lm's own pyproject.toml lists "torch>=2.1" (intentionally
# unbounded -- see the comment there). A normal `pip install -e '.[cuda,dev]'`
# re-resolves that whole dependency list, including torch, against the
# default index -- which on this box means silently replacing the
# CUDA-enabled torch build just verified above with whatever build the
# default index prefers (this is what killed build 414632). --no-deps makes
# that structurally impossible: pip installs koopman_lm's own code without
# ever looking at its dependency list, so torch is never a resolution
# candidate here. Every other dependency is installed explicitly below.
python -m pip install --no-build-isolation --no-deps -e .

# Every OTHER base + [dev] dependency, EXCLUDING torch, installed normally.
# None of these declare torch as a hard install-time dependency (it's
# optional for the HF libraries), so a normal, fully-resolved install here is
# safe and correctly pulls in their own legitimate sub-dependencies
# (huggingface_hub, tokenizers, pyarrow, ...) that --no-deps would otherwise
# silently drop.
python -m pip install \
    "numpy>=1.24" "transformers>=4.40,<5.0" "datasets>=2.18" \
    safetensors wandb "pyyaml>=6.0" "pytest>=7.0"

# mamba-ssm / causal-conv1d are the risky ones in [cuda]: their own setup.py
# metadata declares "torch" as an install_requires (it has to -- it
# type-checks against the installed torch at import time), so a normal
# resolved install of these two would silently reintroduce the exact same
# clobber through a different path. --no-deps + --no-build-isolation: build
# against the torch already verified on this box, and never let pip touch
# torch to satisfy their metadata. ninja/packaging (installed first, ordinary
# resolution) are their build-time requirements.
python -m pip install "ninja>=1.11" "packaging>=23"
python -m pip install --no-build-isolation --no-deps \
    "mamba-ssm>=2.2.2" "causal-conv1d>=1.4.0"

bash scripts/build_b200_prefix_scan.sh

# The whole point of the sequence above: prove it did NOT clobber the
# CUDA-enabled torch verified at the top. Fail loudly here (not silently,
# downstream, in the middle of a training job) if it did.
python - <<'PY'
import torch

assert torch.cuda.is_available(), (
    "torch.cuda.is_available() is False after installing koopman_lm -- the "
    "install sequence clobbered the CUDA-enabled torch build. See "
    "scripts/setup_env.sh and pyproject.toml's dependency comments."
)
cuda_version = torch.version.cuda or ""
assert cuda_version.startswith("12"), (
    f"torch.version.cuda={cuda_version!r} does not start with '12' after "
    "installing koopman_lm -- the install sequence upgraded torch to an "
    "incompatible CUDA major version (this cluster's driver only forward-"
    "compats within CUDA 12.x, not to 13.x). See scripts/setup_env.sh and "
    "pyproject.toml's dependency comments."
)
print(f"OK: torch={torch.__version__} cuda={cuda_version} still CUDA-capable after install")
PY

echo "Environment installed and fused prefix scan validated."
