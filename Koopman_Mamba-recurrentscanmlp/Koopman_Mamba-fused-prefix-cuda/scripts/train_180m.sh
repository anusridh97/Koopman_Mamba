#!/usr/bin/env bash
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export SKA_REQUIRE_B200="${SKA_REQUIRE_B200:-1}"
exec "$(dirname "$0")/pretrain.sh" 180m_prefix_scan
