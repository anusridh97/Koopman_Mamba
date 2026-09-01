#!/bin/bash
# Emit a line whenever a Slurm array's running/pending/done counts change, then
# exit when it drains. Monitoring helper; not part of the experiment.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 JOB_ID [TOTAL_TASKS]" >&2
    exit 2
fi

# `squeue -r` is load-bearing: without it, pending array tasks are collapsed
# into one range line (for example 446697_[4-27]), so wc -l reports one pending
# task when there are actually 24.
job="$1"
total="${2:-28}"
last=""
while true; do
    running=$(squeue -j "$job" -h -r -t RUNNING 2>/dev/null | wc -l)
    pending=$(squeue -j "$job" -h -r -t PENDING,CONFIGURING,REQUEUED \
        2>/dev/null | wc -l)
    if [ "$running" = "0" ] && [ "$pending" = "0" ]; then
        echo "array $job drained: no running or pending tasks left"
        exit 0
    fi
    done_count=$(( total - running - pending ))
    current="running=$running pending=$pending done=$done_count"
    if [ "$current" != "$last" ]; then
        echo "array $job $current (of $total)"
        last="$current"
    fi
    sleep 60
done
