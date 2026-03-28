#!/bin/bash
set -euo pipefail

# ============================================================================
# assemble.sh -- Build the final tarball from original repo + new fast files.
#
# Run from the directory containing both:
#   - The original koopman-lm repo (with koopman_lm/, train.py, evaluate.py, etc.)
#   - The new fast files (train_fast.py, pretokenize.py, etc.)
#
# Usage:
#   bash assemble.sh /path/to/original/repo
# ============================================================================

ORIG="${1:-.}"
DEST="./koopman-lm-fast"

echo "Assembling from original repo: ${ORIG}"
echo "Output: ${DEST}"

rm -rf "${DEST}"
mkdir -p "${DEST}/koopman_lm" "${DEST}/evals"

# ---- Copy unchanged original files ----
echo "Copying original files..."

# Core modules (unchanged)
cp "${ORIG}/koopman_lm/model.py"              "${DEST}/koopman_lm/"
cp "${ORIG}/koopman_lm/ska.py"                "${DEST}/koopman_lm/"
cp "${ORIG}/koopman_lm/koopman_mlp.py"        "${DEST}/koopman_lm/"
cp "${ORIG}/koopman_lm/recurrent.py"          "${DEST}/koopman_lm/"
cp "${ORIG}/koopman_lm/adaptive_chunking.py"  "${DEST}/koopman_lm/"

# Eval harness wrapper (unchanged)
if [ -d "${ORIG}/evals" ]; then
    cp "${ORIG}/evals/lm_harness_eval.py"     "${DEST}/evals/"
elif [ -f "${ORIG}/koopman_lm/eval_tasks.py" ]; then
    cp "${ORIG}/koopman_lm/eval_tasks.py"     "${DEST}/evals/lm_harness_eval.py"
fi

# Keep original train.py and evaluate.py as reference
cp "${ORIG}/train.py"                         "${DEST}/train_original.py" 2>/dev/null || true
cp "${ORIG}/evaluate.py"                      "${DEST}/evaluate_original.py" 2>/dev/null || true

echo "Done assembling. Creating tarball..."
tar czf koopman-lm-fast.tar.gz -C "$(dirname ${DEST})" "$(basename ${DEST})"
echo "Created: koopman-lm-fast.tar.gz"
ls -lh koopman-lm-fast.tar.gz
