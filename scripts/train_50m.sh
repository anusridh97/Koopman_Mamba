#!/usr/bin/env bash
set -euo pipefail
# Native B200 defaults; set SKA_REQUIRE_B200=0 and override the architecture to
# use another supported NVIDIA GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export SKA_REQUIRE_B200="${SKA_REQUIRE_B200:-1}"
exec "$(dirname "$0")/pretrain.sh" 50m_prefix_scan
