#!/bin/bash
# Submit the whole FineWeb-Edu 100B campaign so it drives ITSELF.
#
#   scripts/campaign.sh --dry-run     # print every sbatch, submit nothing
#   scripts/campaign.sh --go          # submit
#   scripts/campaign.sh --status      # one-line-per-job status
#
# WHY IT IS BUILT THIS WAY. The plan used to have manual decision points -- pick
# the LR winner at 8%, then launch the siblings, then launch the evals. Each is
# a moment where nobody may be awake, and the GPU cap means a missed decision
# BLOCKS the other two models rather than merely wasting time. So every decision
# is a Slurm job with a dependency, and the only human action is reading
# --status.
#
# STAGE 1  two 1.5B arms at 64 GPUs each (= the 128-GPU account cap), each with
#          time_limit 7h. They are FULL-length runs that simply run out of wall
#          clock at ~8%, writing resume.pt on the way out. Nothing is thrown
#          away and nothing needs killing: the failure mode if the next stage
#          never fires is "paused", not "wasted" and not "wrong".
#
# STAGE 2  a CPU-only decider, --dependency=afterany on both arms. It runs
#          pick_lr_winner.py, then submits: the winner's continuation as a
#          2-day chain (--resume_if_available picks up its 8%), 440M and 180M at
#          the winning LR, and an eval job per model dependent on that model
#          finishing. afterany, not afterok, because a wall-clock kill exits
#          non-zero and that is exactly the case we are continuing from.
#
# Two points at 1.5B rather than three: 3 x 64 = 192 GPUs exceeds the medium QOS
# cap (gres/gpu=128 per ACCOUNT, shared with four other users). It has to be
# production width because resume is invalid across a world-size change.
set -uo pipefail
REPO=/users/cody1212/Koopman_Mamba
V=/scratch/m000151-pm06/jkli/venvs/koopman-cuda/bin/python
ACCT=${ACCT:-marlowe-m000151-pm06}
# TWO ACCOUNTS, because QOS is per-association: marlowe-m000151-pm06 holds
# `medium` (the only QOS `batch` allows) while marlowe-m000151 holds `normal`
# (the only one `preempt` allows). Training goes to batch under ACCT; the
# decider and eval helper jobs run on preempt under PREEMPT_ACCT. Submitting a
# preempt/normal job under the pm06 account fails with
# `Invalid qos specification`.
PREEMPT_ACCT=${PREEMPT_ACCT:-marlowe-m000151}
RUN_ROOT=${RUN_ROOT:-/scratch/m000151-pm06/cqiu/mamba3/prod}
STATE=$RUN_ROOT/_campaign
export PYTHONPATH=$REPO:/scratch/m000151-pm06/cqiu/pylibs

MODE=${1:---status}
mkdir -p "$STATE"

submit() {  # submit <label> <account> <extra sbatch args...> -- prints job id
  local label=$1 acct=$2; shift 2
  if [ "$MODE" = "--dry-run" ]; then echo "DRYRUN[$label]: $*" >&2; echo "000000"; return; fi
  local id
  id=$(sbatch --parsable --account="$acct" "$@") || { echo "SUBMIT FAILED: $label" >&2; return 1; }
  echo "$id"
}

case "$MODE" in
--status)
  echo "=== campaign status $(date '+%F %H:%M %Z') ==="
  if [ -f "$STATE/jobs.txt" ]; then
    while read -r id label; do
      st=$(sacct -j "$id" --format=State -nP 2>/dev/null | head -1)
      el=$(sacct -j "$id" --format=Elapsed -nP 2>/dev/null | head -1)
      printf "  %-28s %-12s %-12s %s\n" "$label" "$id" "${st:-UNKNOWN}" "${el:-}"
    done < "$STATE/jobs.txt"
  else
    echo "  not launched yet"
  fi
  [ -f "$STATE/decision.json" ] && { echo "--- LR decision ---"; cat "$STATE/decision.json"; }
  echo "--- queue ---"
  squeue -A "$ACCT" -h -o "  %.12i %.18j %.2t %.6M %R" 2>/dev/null
  echo "--- progress (latest step line per run) ---"
  for d in "$RUN_ROOT"/*/seed*/; do
    [ -d "$d" ] || continue
    line=$(grep -hE "^step " "$d"slurm-*.out 2>/dev/null | tail -1)
    [ -n "$line" ] && printf "  %-34s %s\n" "$(basename "$(dirname "$d")" | cut -c1-34)" "$line"
  done
  exit 0
  ;;
--dry-run|--go) ;;
*) echo "usage: $0 [--dry-run|--go|--status]"; exit 2 ;;
esac

: > "$STATE/jobs.txt"
cd "$REPO"

# ---- STAGE 1: the two-width LR sweep ------------------------------------
# Ten points: 180M x6 at 8 GPUs (48) + 440M x4 at 16 GPUs (64) = 112 of the
# 128-GPU account cap. Each is a full-length run stopped by an 8h30 wall clock
# at ~8-10% of its own schedule, so every point writes resume.pt and nothing
# needs killing.
#
# SWEEP AT TWO WIDTHS, rather than bracketing the 1.5B directly on a one-point
# prior. Our only uncensored LR measurement is 180M = 0.00256 -- the 3M, 10M and
# 50M optima all sat at their search ceilings -- and it came from a different
# protocol (eb 96, 65 tok/param vs eb 256, 545 here). Two points on that prior
# picks an endpoint; six points at 576 and four at 832 locate interior optima on
# the protocol we are actually running, and give a trend to extrapolate from.
#
# The points are throwaway by design: resume is invalid across a world-size
# change, so a narrow sweep point cannot hand off to a full-width run. That is
# affordable precisely because they are narrow -- ~744 GPU-h, 9% of the campaign.
ARM_IDS=()
for cfg in "$REPO"/configs/runs/mamba3/mamba3-180m-lr*.yaml \
           "$REPO"/configs/runs/mamba3/mamba3-440m-lr*.yaml; do
  [ -f "$cfg" ] || { echo "missing sweep configs -- run make_mamba3_runs.py --lr_bracket"; exit 1; }
  RUN_ROOT="$RUN_ROOT" $V -m experimentation.run "$cfg" --launcher slurm \
      --dry_run --allow-dirty >/dev/null 2>&1
  rd=$(ls -dt "$RUN_ROOT"/"$(basename "${cfg%.yaml}")".*/seed42.* 2>/dev/null | head -1)
  [ -n "$rd" ] || { echo "materialize failed: $cfg"; exit 1; }
  id=$(submit "sweep-$(basename "${cfg%.yaml}" | sed 's/mamba3-//')" "$ACCT" \
       --time=08:30:00 "$rd/launch.sbatch") || exit 1
  echo "$id $(basename "${cfg%.yaml}")" >> "$STATE/jobs.txt"
  ARM_IDS+=("$id")
  echo "stage1 $(basename "${cfg%.yaml}") -> $id"
done

# ---- STAGE 2 and 3 fire themselves --------------------------------------
echo 2 > "$STATE/stage"
DEP=$(IFS=:; echo "${ARM_IDS[*]}")
id=$(submit "stage2" "$PREEMPT_ACCT" --dependency="afterany:$DEP" \
     "$REPO/scripts/campaign_stage.sh") || exit 1
echo "$id stage2" >> "$STATE/jobs.txt"
echo "stage2 -> $id (after ${#ARM_IDS[@]} sweep points)"
echo
echo "Stages 2 and 3 fire by themselves. The only thing left to do is:"
echo "  scripts/campaign.sh --status"
