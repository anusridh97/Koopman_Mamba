#!/bin/bash
# Emit one line per state change for a slurm job, then exit when it leaves the
# queue. Used by the agent's Monitor tool; a scratch helper, not part of the
# experiment.
job="$1"
last=""
while true; do
    s=$(squeue -j "$job" -h -o "%T" 2>/dev/null)
    if [ -z "$s" ]; then
        echo "job $job left the queue (finished, failed or cancelled)"
        exit 0
    fi
    if [ "$s" != "$last" ]; then
        echo "job $job state=$s node=$(squeue -j "$job" -h -o '%N')"
        last="$s"
    fi
    sleep 30
done
