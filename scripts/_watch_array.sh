#!/bin/bash
# Emit a line whenever a slurm array's (running, pending, done) counts change,
# and exit when it drains. Agent monitoring helper; not part of the experiment.
job="$1"
total="${2:-28}"
last=""
while true; do
    r=$(squeue -j "$job" -h -t RUNNING 2>/dev/null | wc -l)
    p=$(squeue -j "$job" -h -t PENDING,CONFIGURING 2>/dev/null | wc -l)
    if [ "$r" = "0" ] && [ "$p" = "0" ]; then
        echo "array $job drained: no running or pending tasks left"
        exit 0
    fi
    cur="running=$r pending=$p"
    if [ "$cur" != "$last" ]; then
        echo "array $job $cur (of $total)"
        last="$cur"
    fi
    sleep 60
done
