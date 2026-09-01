#!/bin/bash
# Emit one line per state change for a Slurm job, then exit when it leaves the
# queue. Monitoring helper; not part of the experiment.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 JOB_ID" >&2
    exit 2
fi

job="$1"
last=""
while true; do
    state=$(squeue -j "$job" -h -o "%T" 2>/dev/null)
    if [ -z "$state" ]; then
        echo "job $job left the queue (finished, failed or cancelled)"
        exit 0
    fi
    if [ "$state" != "$last" ]; then
        echo "job $job state=$state node=$(squeue -j "$job" -h -o '%N')"
        last="$state"
    fi
    sleep 30
done
