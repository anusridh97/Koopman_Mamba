#!/usr/bin/env bash
set -euo pipefail
# Override TOKENS/STEPS/PDBS/GA/LR/WARMUP/RUN_NAME through the environment.
exec "$(dirname "$0")/pretrain.sh" 50m_prefix_scan
