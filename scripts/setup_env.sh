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

python -m pip install --no-build-isolation -e '.[cuda,dev]'
bash scripts/build_b200_prefix_scan.sh

echo "Environment installed and fused prefix scan validated."
