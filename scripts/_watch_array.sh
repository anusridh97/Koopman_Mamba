#!/bin/bash
# Emit a line whenever a slurm array's (running, pending, done) counts change,
# and exit when it drains. Agent monitoring helper; not part of the experiment.
#
# `squeue -r` is load-bearing. WITHOUT it, squeue collapses all pending array
# tasks into a SINGLE range line (`446697_[4-27]`), so `wc -l` reports 1 pending
# no matter how many are queued -- which reads as "almost done" when 23 tasks
# have not started. Observed misreporting `running=5 pending=1 (of 28)` at a
# point when nothing had completed at all.
job="$1"
total="${2:-28}"
last=""
while true; do
    r=$(squeue -j "$job" -h -r -t RUNNING 2>/dev/null | wc -l)
    p=$(squeue -j "$job" -h -r -t PENDING,CONFIGURING,REQUEUED 2>/dev/null | wc -l)
    if [ "$r" = "0" ] && [ "$p" = "0" ]; then
        echo "array $job drained: no running or pending tasks left"
        exit 0
    fi
    done_n=$(( total - r - p ))
    cur="running=$r pending=$p done=$done_n"
    if [ "$cur" != "$last" ]; then
        echo "array $job $cur (of $total)"
        last="$cur"
    fi
    sleep 60
done
