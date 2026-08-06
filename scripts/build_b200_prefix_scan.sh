#!/usr/bin/env bash
set -euo pipefail

# Native NVIDIA B200 / datacenter Blackwell build. CUDA 12.8+ is required to
# produce an sm_100 cubin. Override MAX_JOBS or TORCH_EXTENSIONS_DIR as needed.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export SKA_PREFIX_SCAN_BUILD_VERBOSE="${SKA_PREFIX_SCAN_BUILD_VERBOSE:-1}"
export SKA_REQUIRE_B200="${SKA_REQUIRE_B200:-1}"

python scripts/build_prefix_scan_cuda.py
