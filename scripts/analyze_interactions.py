#!/usr/bin/env python
"""Analyse a finished search study. READ-ONLY.

    python scripts/analyze_interactions.py <run_root>/_studies/<name>.<id>
    python scripts/analyze_interactions.py <study_dir> --study-name <name>
    python scripts/analyze_interactions.py <study_dir> --out /tmp/report

Writes `<study_dir>/analysis/`: noise_floor.csv, anchor_contrasts.csv,
main_effects.csv, conditional_effects.csv, summary.json, importances.csv,
interactions.md, pareto.csv, throughput_pareto.csv, shortlist.csv,
top_by_loss.csv, top_by_ska_delta.csv, objective_vs_params.csv, rank_curve.csv,
sampler_correlations.csv.

`noise_floor.csv` is listed first because it is the scale for every other file:
it is the spread of one configuration across training seeds, and an effect
smaller than it is not a small effect but an unmeasurable one. This command
prints it to stdout for the same reason -- a reader who runs this and reads only
the terminal should still be handed the number that qualifies every table.

A thin shell over `experimentation.sweep.search.analysis`, which is where the
reasoning lives and where the tests point. What is genuinely this file's job is
the part that is easy to get wrong: FINDING the journal.

Two failures it exists to prevent, both of which have already happened here.

`JournalFileBackend(path)` CREATES the file when it does not exist. So pointing
at a wrong or empty directory silently makes an empty journal, and
`optuna.load_study` then raises `KeyError: Record does not exist` -- a crash
where a message belongs, and one that leaves behind the very file whose absence
was the actual problem. `scripts/verify_study_e2e.sbatch` had exactly this bug
(it said `journal.log`; `study.py` writes `optuna_journal.log`). So this checks
for the file with `pathlib` BEFORE handing the path to optuna.

And the study name. A study directory holds one study, so requiring the name on
the command line is a papercut whose failure mode is a KeyError from inside
optuna rather than "that study is not in this journal". `--study-name` stays for
the ambiguous case, and is otherwise discovered.

Artefacts go in a SUBDIRECTORY rather than beside the journal, because
`report.py` already writes `trials.csv`, `top_trials.md` and `best_trial.json`
there and a reader could not tell which run produced which file.

Safe to run against a study that is still going. Nothing here calls `tell`,
`set_user_attr` or `enqueue_trial`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

JOURNAL_NAME = "optuna_journal.log"


def find_journal(study_dir: Path) -> Path:
    """The journal in `study_dir`, or a message saying where it was not.

    Checked with `pathlib` and not by trying to open it: `JournalFileBackend`
    creates a missing file, so "let optuna tell us" would leave an empty journal
    behind and report the wrong error.
    """
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        raise SystemExit(f"FATAL: {study_dir} is not a directory.")
    journal = study_dir / JOURNAL_NAME
    if journal.is_file():
        return journal

    # A run_root rather than a study dir is the likely mistake, so say so.
    candidates = sorted((study_dir / "_studies").glob(f"*/{JOURNAL_NAME}"))
    hint = ""
    if candidates:
        hint = ("\n       This looks like a run_root. Did you mean one of:\n"
                + "\n".join(f"         {c.parent}" for c in candidates))
    raise SystemExit(
        f"FATAL: no {JOURNAL_NAME} in {study_dir}\n"
        f"       Contents: {sorted(p.name for p in study_dir.iterdir())}\n"
        f"       NOT creating it: JournalFileBackend would happily make an "
        f"empty one, and optuna would then raise 'Record does not exist' -- a "
        f"crash instead of this message, plus a stray file.{hint}")


def load(journal: Path, study_name: str | None):
    """Open the journal read-only and return the study it holds."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal)))
    summaries = optuna.get_all_study_summaries(storage=storage)
    names = [s.study_name for s in summaries]
    if not names:
        raise SystemExit(f"FATAL: {journal} holds no studies.")
    if study_name is None:
        if len(names) > 1:
            raise SystemExit(
                f"FATAL: {journal} holds {len(names)} studies {names}; pass "
                f"--study-name to choose one.")
        study_name = names[0]
    elif study_name not in names:
        raise SystemExit(
            f"FATAL: study {study_name!r} is not in {journal}. It holds "
            f"{names}.")
    return optuna.load_study(study_name=study_name, storage=storage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study_dir",
                        help="the directory holding optuna_journal.log")
    parser.add_argument("--study-name", default=None,
                        help="only needed when the journal holds several")
    parser.add_argument("--out", default=None,
                        help="where to write (default: <study_dir>/analysis)")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--bins", type=int, default=3,
                        help="bins for a continuous axis in a 2-way table")
    args = parser.parse_args(argv)

    study_dir = Path(args.study_dir)
    journal = find_journal(study_dir)
    study = load(journal, args.study_name)

    from experimentation.sweep.search.analysis import analyse, write_analysis

    out_dir = Path(args.out) if args.out else study_dir / "analysis"
    result = analyse(study, top_k=args.top_k, bins=args.bins)

    print(f"[analysis] study     {study.study_name}")
    print(f"[analysis] journal   {journal}")
    counts = result["counts"]
    print(f"[analysis] trials    {counts['total']} recorded: "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())
                      if k != "total"))
    print(f"[analysis] anchors   {result['n_anchors']}")

    # The headline, before the importances, because it is what qualifies them.
    floor = result["noise_floor"]
    if floor["available"]:
        print(f"[analysis] NOISE FLOOR sigma={floor['sigma']:.6g} over "
              f"{floor['dof']} dof; smallest resolvable effect "
              f"{floor['min_resolvable_effect']:.6g}")
        for group in floor["groups"]:
            print(f"[analysis]   {group['group']}: n={group['n']} "
                  f"seeds={group['seeds']} mean={group['mean']:.6g} "
                  f"sd={group['sd']:.4g} range={group['range']:.4g}")
        resolved = sum(1 for r in result["conditional_effects"] if r["resolved"])
        print(f"[analysis]   {resolved} of {len(result['conditional_effects'])} "
              f"prespecified interaction(s) exceed it")
        unresolved = [r["anchor"] for r in result["anchor_contrasts"]
                      if r["resolved"] is False]
        if unresolved:
            print(f"[analysis]   anchors INSIDE the floor (not rankable): "
                  + ", ".join(unresolved))
    else:
        print("[analysis] NOISE FLOOR unavailable -- every magnitude below is "
              "reported WITHOUT A SCALE")
        print(f"[analysis]   {floor['reason']}")

    importances = result["importances"]
    if importances["available"]:
        top = sorted(importances["values"].items(), key=lambda kv: -kv[1])[:5]
        print(f"[analysis] top axes  "
              + ", ".join(f"{k}={v:.3f}" for k, v in top))
    else:
        print(f"[analysis] top axes  unavailable ({importances['reason']})")
    for reason, count in sorted(result["failures"].items(),
                                key=lambda kv: -kv[1])[:3]:
        print(f"[analysis]   failed {count}x: {reason}")

    written = write_analysis(study, out_dir, top_k=args.top_k, bins=args.bins)
    for name, path in sorted(written.items()):
        print(f"[analysis] wrote {name:22} {path}")

    print()
    print("[analysis] READ THIS BEFORE QUOTING A NUMBER.")
    print("           1. Compare every magnitude to the noise floor above. An")
    print("              effect smaller than it is not a small effect, it is an")
    print("              unmeasurable one, and more trials will not change that.")
    print("           2. `anchor_contrasts.csv` is the only CONTROLLED table:")
    print("              one factor moved from one reference point. Read it")
    print("              first.")
    print("           3. `importances.csv` is a PED-ANOVA divergence, not a")
    print("              variance decomposition, and its `local` column is")
    print("              partly a description of the search path. See section 1")
    print("              of interactions.md.")
    print("           4. `sampler_correlations.csv` describes where the sampler")
    print("              WENT, not what the model prefers. It is not evidence.")
    print("           5. `shortlist.csv` is a set of candidates for confirmation")
    print("              at larger scale, never a winner, and interactions.md")
    print("              section 6 lists what this study cannot establish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
