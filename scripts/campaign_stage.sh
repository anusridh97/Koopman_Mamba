#!/bin/bash
#SBATCH --job-name=campStage
#SBATCH --partition=preempt
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --export=ALL
# One decision stage of the campaign, run as a Slurm job so nobody has to be
# awake for it. STAGE comes from the job name suffix written by campaign.sh
# into $STATE/stage, NOT from --export: an --export variable was lost on requeue
# earlier today and parked a job `user_env_retrieval_failed_requeued_held`.
#
# STAGE 2  ten sweep points are done -> fit lr ~ width^p across 180M and 440M,
#          submit the 1.5B bracket straddling the extrapolation, submit the full
#          180M and 440M runs at their OWN measured optima, and chain stage 3.
# STAGE 3  the 1.5B bracket is done -> pick its winner, chain its continuation,
#          and submit the evals.
set -uo pipefail
REPO=${REPO:-/users/cody1212/Koopman_Mamba}
RUN_ROOT=${RUN_ROOT:-/scratch/m000151-pm06/cqiu/mamba3/prod}
ACCT=${ACCT:-marlowe-m000151-pm06}            # batch  + qos medium
PREEMPT_ACCT=${PREEMPT_ACCT:-marlowe-m000151} # preempt + qos normal
SHARD=${SHARD:-/scratch/m000151-pm06/cqiu/tok100b/shard}
STATE=$RUN_ROOT/_campaign
V=/scratch/m000151-pm06/jkli/venvs/koopman-cuda/bin/python
cd "$REPO"; export PYTHONPATH=$REPO:/scratch/m000151-pm06/cqiu/pylibs
mkdir -p "$STATE"
STAGE=$(cat "$STATE/stage" 2>/dev/null || echo 2)

# IDEMPOTENCE. These stages run on `preempt` with --requeue, so a preemption
# re-executes them from the top. Without a guard that is actively destructive:
# re-running stage 2 after it had already submitted would put TWO jobs on the
# same identity-derived run_dir, both writing the same checkpoints. The marker
# is written as the stage's LAST action, so its presence means "fully done".
if [ -f "$STATE/stage${STAGE}_done" ]; then
  echo "stage $STAGE already completed (marker present) -- nothing to do"
  exit 0
fi

# Belt and braces: never submit a config whose job name is already queued or
# running for this account. Catches a requeue that happened mid-stage, where
# the marker is absent but some jobs are already in flight.
already_queued() {
  squeue -A "$ACCT" -h -o "%j" 2>/dev/null | grep -qx "$1"
}

launch() {  # launch <config> <acct> -> prints job id on stdout, logs elsewhere
  local cfg=$1 acct=$2 rd
  RUN_ROOT=$RUN_ROOT "$V" -m experimentation.run "$cfg" --launcher slurm \
      --dry_run --allow-dirty >/dev/null 2>&1
  rd=$(ls -dt "$RUN_ROOT"/"$(basename "${cfg%.yaml}")".*/seed42.* 2>/dev/null | head -1)
  [ -n "$rd" ] || { echo "MATERIALIZE FAILED $cfg" >&2; return 1; }
  sbatch --parsable --account="$acct" "$rd/launch.sbatch"
}

if [ "$STAGE" = "2" ]; then
  echo "=== STAGE 2: fit the two-width sweep and launch the rest ==="
  "$V" scripts/campaign_stage2.py --run_root "$RUN_ROOT" --shard "$SHARD" \
      --acct "$ACCT" --submit || { echo "STAGE2 FAILED" | tee "$STATE/HALTED"; exit 1; }

  IDS=""
  # Full-length 180M and 440M at their own measured optima. 440M is ~52h at
  # 32 GPUs, past the 2-day cap, so it gets a chain.
  for sz in 440m 180m; do
    if already_queued "mamba3-$sz"; then
      echo "  full $sz already in flight -- skipping"; continue; fi
    jid=$(launch "$REPO/configs/runs/mamba3/mamba3-$sz.yaml" "$ACCT") || continue
    echo "  full $sz -> $jid"; echo "$jid mamba3-$sz" >> "$STATE/jobs.txt"
    IDS="${IDS:+$IDS:}$jid"
    # PERSIST the primary id. The eval dependency used to be built only from
    # chain.log, which records CHAINED chunks -- so a job with no chain (the
    # 180M) never appeared in it, and the evals could start before it finished
    # and silently SKIP it, since campaign_evals.sh passes over any run dir
    # lacking final/model.pt. Found by executing stage 2 and 3 end to end.
    echo "$jid" >> "$STATE/train_ids"
    rd=$(ls -dt "$RUN_ROOT"/mamba3-$sz.*/seed42.* | head -1)
    [ "$sz" = "440m" ] && "$V" scripts/chain_run.py "$rd" --chunks 2 \
        --after "$jid" --account "$ACCT" | tee -a "$STATE/chain.log"
  done

  # The 1.5B bracket, at production width with a 7h cap so both arms stop
  # themselves at ~8% having written resume.pt. Nothing needs killing.
  ARMS=""
  for cfg in "$REPO"/configs/runs/mamba3/mamba3-1p5b-lr*.yaml; do
    if already_queued "$(basename "${cfg%.yaml}")"; then
      echo "  $(basename "$cfg") already in flight -- skipping"; continue; fi
    RUN_ROOT=$RUN_ROOT "$V" -m experimentation.run "$cfg" --launcher slurm \
        --dry_run --allow-dirty >/dev/null 2>&1
    rd=$(ls -dt "$RUN_ROOT"/"$(basename "${cfg%.yaml}")".*/seed42.* | head -1)
    jid=$(sbatch --parsable --account="$ACCT" --time=07:00:00 "$rd/launch.sbatch") || continue
    echo "  1p5b arm -> $jid ($(basename "$cfg"))"
    echo "$jid $(basename "${cfg%.yaml}")" >> "$STATE/jobs.txt"
    ARMS="${ARMS:+$ARMS:}$jid"
  done
  [ -n "$ARMS" ] || { echo "no 1.5B arms submitted" | tee "$STATE/HALTED"; exit 1; }

  s3=$(sbatch --parsable --account="$PREEMPT_ACCT" --dependency="afterany:$ARMS" \
       "$REPO/scripts/campaign_stage.sh")
  echo "$s3 stage3" >> "$STATE/jobs.txt"
  echo "  stage3 -> $s3 (after $ARMS)"
  # ORDER MATTERS: flip the pointer and mark done only once stage 3 exists. The
  # earlier version wrote `3` before submitting, so a preemption in between left
  # the pointer at 3 with no stage-3 job -- a campaign that silently stopped.
  echo 3 > "$STATE/stage"
  touch "$STATE/stage2_done"
  exit 0
fi

echo "=== STAGE 3: pick the 1.5B winner, continue it, then evaluate ==="
A=$(ls -d "$RUN_ROOT"/mamba3-1p5b-lr*/seed42.* 2>/dev/null)
[ -n "$A" ] || { echo "no 1.5B arms found" | tee "$STATE/HALTED"; exit 1; }
# shellcheck disable=SC2086
"$V" scripts/pick_lr_winner.py $A --json "$STATE/decision_1p5b.json" || {
  echo "NO 1.5B WINNER -- halted; both arms diverged." | tee "$STATE/HALTED"; exit 1; }
WIN=$("$V" -c "import json;print(json.load(open('$STATE/decision_1p5b.json'))['winner']['run_dir'])")
echo "1.5B winner: $WIN"
"$V" scripts/chain_run.py "$WIN" --chunks 3 --account "$ACCT" | tee -a "$STATE/chain.log"
CHAIN=$(grep -oE "chunk [0-9]+: [0-9]+" "$STATE/chain.log" 2>/dev/null \
        | grep -oE "[0-9]+$" | tr '\n' ' ')
PRIM=$(cat "$STATE/train_ids" 2>/dev/null | tr '\n' ' ')
# Union of PRIMARY training jobs and every chained chunk, so the evals cannot
# start while any model is still training.
ALL=$(printf "%s %s\n" "$PRIM" "$CHAIN" | tr ' ' '\n' | grep -E "^[0-9]+$" \
      | sort -u | tr '\n' ':' | sed 's/:$//')
ev=$(sbatch --parsable --account="$PREEMPT_ACCT" --dependency="afterany:$ALL" \
     "$REPO/scripts/campaign_evals.sh")
echo "$ev mamba3-evals" >> "$STATE/jobs.txt"
echo "evals -> $ev (after $ALL)"
touch "$STATE/stage3_done"
