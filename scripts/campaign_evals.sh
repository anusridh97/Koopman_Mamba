#!/bin/bash
#SBATCH --job-name=mamba3Eval
#SBATCH --partition=preempt
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --export=ALL
# Score every finished model against the Llama-3.1 held-out shard.
#
# Submitted by campaign_decide.sh with a dependency on the training jobs, NOT
# run by hand at the end -- running them by hand is what failed twice: an
# invented --eval_data_dir killed 39 evals, and a run-dir path one level too
# shallow killed the retry. Idempotent, so a requeue re-scores nothing.
set -uo pipefail
REPO=${REPO:-/users/cody1212/Koopman_Mamba}
RUN_ROOT=${RUN_ROOT:-/scratch/m000151-pm06/cqiu/mamba3/prod}
VAL=${VAL:-/scratch/m000151-pm06/cqiu/tok100b/val}
V=/scratch/m000151-pm06/jkli/venvs/koopman-cuda/bin/python
cd "$REPO"
export PYTHONPATH=$REPO:/scratch/m000151-pm06/cqiu/pylibs
# HF_HOME explicit and config.json pre-cached: AutoTokenizer resolves its class
# through AutoConfig, so offline failed on a cache holding tokenizer.json but
# no config.json.
export HF_HOME=/scratch/m000151-pm06/cqiu/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
rc=0; n=0
for rd in "$RUN_ROOT"/*/seed42.*; do
  [ -d "$rd" ] || continue
  [ -f "$rd/final/model.pt" ] || { echo "SKIP no final/: $rd"; continue; }
  [ -f "$rd/eval/final/fineweb_ppl.json" ] && { echo "already scored: $rd"; continue; }
  echo "=== scoring $rd"
  # --held_out_data_dir, NOT --eval_data_dir (does not exist).
  # --mode fineweb_ppl, NOT ppl (ppl is WikiText over HF load_dataset; we are offline).
  "$V" -m experimentation.evaluation.evaluate \
      --checkpoint "$rd/final/model.pt" --mode fineweb_ppl \
      --held_out_data_dir "$VAL" || { echo "FAILED: $rd"; rc=1; }
  n=$((n+1))
done
# LOUD accounting: name every model that has no final/ so a missed one is
# visible in the log and in `campaign.sh --status`, rather than silently absent
# from the results. The dependency should make this impossible; this is the
# second line of defence.
missing=""
for want in mamba3-180m mamba3-440m mamba3-1p5b; do
  found=0
  for rd in "$RUN_ROOT"/$want*/seed42.*; do
    [ -f "$rd/eval/final/fineweb_ppl.json" ] && found=1
  done
  [ "$found" = "0" ] && missing="${missing:+$missing }$want"
done
if [ -n "$missing" ]; then
  echo "UNSCORED MODELS: $missing" | tee "$RUN_ROOT/_campaign/UNSCORED"
  rc=1
else
  rm -f "$RUN_ROOT/_campaign/UNSCORED"
  echo "all three models scored"
fi
echo "eval stage: scored $n, rc=$rc"
exit $rc
