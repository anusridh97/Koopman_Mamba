"""Early-warning checks for a running search study. Read the journal, not the logs.

    python scripts/study_health.py <study_dir> [--study-name NAME] [--strict]

Exits 0 if everything passes, 1 if any check FAILS.

RUN IT REPEATEDLY, NOT ONCE. A single check a few hours after launch would have
PASSED on the 180M study: the leader started 20:23 and ran one clean anchor for
eleven hours, and the fleet only scaled to 16 workers at 07:30 the next morning
-- the duplicate-config bug could not appear until then, and the first duplicate
started at 11:16. The failure mode arrives when the WORKER COUNT changes, which
on a contended cluster is whenever the backfill happens to let your jobs in. So
schedule this every few hours for the life of the study, not once at launch+3h.

Each check names the incident that motivated it. That is deliberate: a check
whose failure mode nobody can describe gets deleted the first time it is noisy.

  DUPLICATE_CONFIGS   16 single-worker jobs each computed `sampler_seed_for(seed, 0)`
                      because `worker_id` is scoped to one supervisor's fanout.
                      All 16 drew the identical stream, so 24 sampled trials were
                      24 copies of ONE config. 335 GPU-h. Detectable in <1 h:
                      the 2nd and 3rd sampled trials were already identical.
  SAMPLER_SEEDS       the root cause of the above, visible directly in the journal
                      as one distinct `sampler_seed` where there should be N.
  SHARED_RUN_DIR      those duplicates collided into two run dirs, one with 13
                      distinct job_ids interleaving train.log and checkpoints.
  NONFINITE_LOSS      `lr-3.85x` first went NaN at step 17,950 of 50,862 (35.3%,
                      ~3.8 h in) and still ran the remaining 33,000 steps to
                      completion. ~28 GPU-h spent training on NaN.
  FAILURE_RATE        a bare `torchrun` not on PATH failed 25 of 25 trials in 22
                      seconds while the job reported COMPLETED 0:0.
  STALLED             `wait_for_objective` returns when quick_eval.json appears,
                      which train.py writes BEFORE exiting; a 0.5 s teardown race
                      once produced an OOM cascade.
  ANCHOR_PROGRESS     the anchors are the deliverable and are pruning-exempt; if
                      they are not advancing, nothing else matters.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_ORDER = {FAIL: 0, WARN: 1, PASS: 2}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append((name, status, detail))

    def worst(self) -> str:
        return min((r[1] for r in self.rows), key=lambda s: _ORDER[s], default=PASS)

    def render(self) -> str:
        w = max((len(r[0]) for r in self.rows), default=10)
        out = []
        for name, status, detail in sorted(self.rows, key=lambda r: _ORDER[r[1]]):
            out.append(f"  [{status}] {name.ljust(w)}  {detail}")
        return "\n".join(out)


def _params_key(t) -> str:
    return json.dumps(dict(sorted(t.params.items())), sort_keys=True, default=str)


def _load(study_dir: Path, study_name: str | None):
    journals = sorted(study_dir.glob("*journal*.log"))
    # Existence check FIRST: JournalFileBackend CREATES the file when absent, so
    # opening a wrong path manufactures an empty study and every check "passes".
    journals = [j for j in journals if j.is_file() and j.stat().st_size > 0]
    if not journals:
        raise SystemExit(f"no non-empty *journal*.log under {study_dir}")
    journal = max(journals, key=lambda p: p.stat().st_size)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal)))
    if study_name is None:
        summaries = optuna.get_all_study_summaries(storage=storage)
        if not summaries:
            raise SystemExit(f"{journal} contains no study")
        study_name = summaries[0].study_name
    return optuna.load_study(study_name=study_name, storage=storage), journal


def _run_root(study_dir: Path) -> Path:
    """The directory holding the RUN dirs, given the study dir.

    Layout is `<run_root>/_studies/<name>.<study_id>.<macro_id>/` for the study
    and `<run_root>/<name>.<group_id>/seed<n>.<run_id>/` for each run, so the
    root is the parent of `_studies`. Found by NAME rather than by counting
    `..`, because the first version guessed from the leading character of the
    study dir's name and silently looked in the wrong place -- which made both
    filesystem checks report PASS on a study that was actively broken.
    """
    for p in (study_dir, *study_dir.parents):
        if p.name == "_studies":
            return p.parent
    return study_dir.parent


def check_duplicate_configs(trials, rep: Report) -> None:
    live = [t for t in trials
            if t.state.name in ("RUNNING", "COMPLETE", "PRUNED")]
    if len(live) < 3:
        rep.add("DUPLICATE_CONFIGS", PASS, f"only {len(live)} live trial(s), too early")
        return
    groups = collections.Counter(_params_key(t) for t in live)
    worst_n = max(groups.values())
    distinct = len(groups)
    # Anchors legitimately repeat params across the reference group (same config,
    # different SEED), so compare on (params, model_seed) for those.
    seeded = collections.Counter(
        (_params_key(t), t.user_attrs.get("model_seed")) for t in live)
    worst_seeded = max(seeded.values())
    detail = (f"{len(live)} live trials -> {distinct} distinct configs; "
              f"largest identical group {worst_n} "
              f"({worst_seeded} sharing params AND seed)")
    if worst_seeded >= 3:
        rep.add("DUPLICATE_CONFIGS", FAIL, detail +
                " -- >=3 trials with identical params AND seed is a sampler-seed bug,"
                " not chance")
    elif worst_seeded == 2:
        rep.add("DUPLICATE_CONFIGS", WARN, detail + " -- watch for a third")
    else:
        rep.add("DUPLICATE_CONFIGS", PASS, detail)


def check_sampler_seeds(trials, rep: Report) -> None:
    seeds = {t.user_attrs.get("sampler_seed") for t in trials
             if t.user_attrs.get("sampler_seed") is not None}
    workers = {t.user_attrs.get("worker_id") for t in trials
               if t.user_attrs.get("worker_id") is not None}
    sampled = [t for t in trials if not t.user_attrs.get("anchor_name")]
    detail = f"{len(seeds)} distinct sampler_seed(s), {len(workers)} distinct worker_id(s)"
    if len(seeds) == 1 and len(sampled) >= 3:
        rep.add("SAMPLER_SEEDS", FAIL, detail +
                " -- one seed across the whole fleet: every worker draws the SAME"
                " stream. Set KOOPMAN_SEARCH_SAMPLER_OFFSET per job.")
    else:
        rep.add("SAMPLER_SEEDS", PASS, detail)


def check_shared_run_dir(study_dir: Path, rep: Report) -> None:
    root = _run_root(study_dir)
    worst = (0, None)
    checked = 0
    for att in root.glob("*/seed*/attempts.jsonl"):
        checked += 1
        jobs = set()
        try:
            for line in att.read_text().splitlines():
                try:
                    jobs.add(json.loads(line).get("job_id"))
                except Exception:
                    continue
        except OSError:
            continue
        jobs.discard(None)
        if len(jobs) > worst[0]:
            worst = (len(jobs), att.parent)
    if not checked:
        rep.add("SHARED_RUN_DIR", PASS, "no run dirs found yet")
    elif worst[0] > 1:
        rep.add("SHARED_RUN_DIR", FAIL,
                f"{worst[0]} distinct job_ids write {worst[1].name} "
                f"-- concurrent writers interleaving train.log/checkpoints")
    else:
        rep.add("SHARED_RUN_DIR", PASS, f"{checked} run dir(s), one writer each")


def check_nonfinite_loss(study_dir: Path, rep: Report) -> None:
    root = _run_root(study_dir)
    bad = []
    for log in root.glob("*/seed*/train.log"):
        try:
            # Tail only: a NaN that starts mid-run stays for the rest of it.
            with log.open("rb") as fh:
                fh.seek(max(0, log.stat().st_size - 20000))
                tail = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "loss nan" in tail or "loss inf" in tail:
            bad.append(log.parent.name)
    if bad:
        rep.add("NONFINITE_LOSS", FAIL,
                f"{len(bad)} run(s) training on nan/inf: {bad[:4]}"
                " -- kill them, they cannot produce an objective")
    else:
        rep.add("NONFINITE_LOSS", PASS, "no nan/inf in any train.log tail")


def check_failure_rate(trials, rep: Report) -> None:
    counts = collections.Counter(t.state.name for t in trials)
    term = sum(counts[k] for k in ("COMPLETE", "PRUNED", "FAIL"))
    fails = counts["FAIL"]
    if term == 0:
        rep.add("FAILURE_RATE", PASS, "nothing terminal yet")
        return
    rate = fails / term
    detail = f"{fails}/{term} terminal trials failed ({rate:.0%})"
    reasons = collections.Counter(
        str(t.user_attrs.get("failure", ""))[:60] for t in trials
        if t.state.name == "FAIL")
    if reasons:
        detail += " | " + "; ".join(f"{v}x {k!r}" for k, v in reasons.most_common(2))
    if rate > 0.20:
        rep.add("FAILURE_RATE", FAIL, detail)
    elif fails:
        rep.add("FAILURE_RATE", WARN, detail)
    else:
        rep.add("FAILURE_RATE", PASS, detail)


def check_stalled(trials, rep: Report) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stalled = []
    for t in trials:
        if t.state.name != "RUNNING" or t.datetime_start is None:
            continue
        age_h = (now - t.datetime_start).total_seconds() / 3600
        if age_h > 1.0 and not t.intermediate_values:
            stalled.append((t.number, age_h))
    if stalled:
        rep.add("STALLED", FAIL,
                f"{len(stalled)} trial(s) RUNNING >1 h with zero reported steps: "
                + ", ".join(f"#{n} ({h:.1f} h)" for n, h in stalled[:4]))
    else:
        rep.add("STALLED", PASS, "every RUNNING trial is reporting steps")


def check_anchor_progress(trials, rep: Report) -> None:
    anchors = [t for t in trials if t.user_attrs.get("anchor_name")]
    if not anchors:
        rep.add("ANCHOR_PROGRESS", WARN, "no anchors in this study")
        return
    done = sum(1 for t in anchors if t.state.name == "COMPLETE")
    failed = [t.user_attrs["anchor_name"] for t in anchors if t.state.name == "FAIL"]
    running = sum(1 for t in anchors if t.state.name == "RUNNING")
    detail = f"{done}/{len(anchors)} complete, {running} running"
    if failed:
        detail += f", FAILED: {failed}"
    ref = [t.value for t in anchors
           if t.user_attrs.get("reference_group") and t.state.name == "COMPLETE"]
    if len(ref) >= 2:
        import statistics
        detail += f" | reference n={len(ref)} sigma={statistics.stdev(ref):.5f}"
    rep.add("ANCHOR_PROGRESS", FAIL if failed else PASS, detail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir", type=Path)
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="treat WARN as failure too")
    args = ap.parse_args(argv)

    study, journal = _load(args.study_dir, args.study_name)
    trials = study.trials
    rep = Report()
    check_anchor_progress(trials, rep)
    check_duplicate_configs(trials, rep)
    check_sampler_seeds(trials, rep)
    check_shared_run_dir(args.study_dir, rep)
    check_nonfinite_loss(args.study_dir, rep)
    check_failure_rate(trials, rep)
    check_stalled(trials, rep)

    counts = collections.Counter(t.state.name for t in trials)
    print(f"study   {study.study_name}")
    print(f"journal {journal}")
    print(f"trials  {len(trials)}  {dict(counts)}")
    print(f"checked {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(rep.render())
    worst = rep.worst()
    print(f"\nVERDICT: {worst}")
    if worst == FAIL or (args.strict and worst == WARN):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
