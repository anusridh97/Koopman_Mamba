"""scripts/run_proxy_256x17_interactions.sbatch: the invariants a GPU job hides.

A launcher script is the one artifact here whose only consumer is Slurm, so
nothing catches a mistake in it until eight GPUs are already allocated. Three
classes of mistake are checkable on a CPU box, and all three have already
happened in this repo:

**A removed CLI flag.** `--run_root`, `--launcher` and `--n_trials` were removed
because they resolved before `study_id` was computed but were not part of it, so
a flag could change the study while leaving its identity alone.
`scripts/verify_study_e2e.sbatch` still passes `--run_root` and would now die at
argparse -- which is exactly the rot this test catches, and it is asserted for
EVERY sbatch script rather than only the new one.

**A second implementation of the fanout.** The obvious way to write this file is
a loop starting eight processes with `CUDA_VISIBLE_DEVICES` set per worker. That
is what `_fanout` already does, tested in `test_search_fanout.py`, with the
per-worker sampler seeding a hand-rolled loop would silently omit. So the script
must invoke the CLI exactly once and let it spawn.

**A syntax error, or a stale filename.** `bash -n` costs milliseconds. And the
study config, base spec and design file it names have to exist -- a renamed
config strands the script with a shell-level failure at 3am.

Deliberately NOT asserted: that the resources are correctly sized, that the
walltime is enough, or that the scratch paths exist. Those need a GPU, a
measurement, or a filesystem this box does not have.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

SCRIPT = REPO / "scripts/run_proxy_256x17_interactions.sbatch"
STUDY = "configs/search/proxy-256x17-interactions-v1.yaml"


@pytest.fixture(scope="module")
def text():
    return SCRIPT.read_text()


def _directives(text):
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", text, re.MULTILINE))


def _commands(text):
    """The script's EXECUTABLE lines, with continuations joined.

    Two things this fixes, both of which made an earlier version of this file
    wrong in opposite directions.

    Comment lines are dropped. These scripts explain at length why a removed
    flag is NOT passed, and a naive substring search over the whole file finds
    the explanation and calls it a violation -- so the test would forbid
    documenting the thing it is checking for.

    Backslash continuations are joined, and this is the one that mattered.
    `verify_study_e2e.sbatch` invokes the CLI as

        python -m experimentation.sweep.search "$STUDY_CONFIG" \\
            --dry_run --run_root "$RUN_ROOT"

    so the command and the removed flag are on DIFFERENT physical lines. A
    per-line check requiring both together passed on it -- vacuously, against
    the exact file the check was written to catch. `HANDOFF-2026-08-21.md` §6
    calls this "a test that passes for the wrong reason", and it is why the
    parametrised test below asserts a KNOWN violation as well as the absence of
    new ones.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


# ------------------------------------------------------------------ it exists ----

def test_the_script_exists_and_is_a_bash_script(text):
    assert text.startswith("#!/bin/bash")


def test_it_parses_as_bash():
    """`bash -n` catches an unclosed heredoc or quote for free. Discovering one
    of those from a queued job is a wasted allocation."""
    bash = shutil.which("bash")
    assert bash, "no bash on PATH"
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


def test_it_says_it_has_not_been_run(text):
    """It has not. Committed for review, not launched -- and the file has to say
    so, because the next reader will assume results exist."""
    assert "HAS NOT BEEN RUN" in text


# ------------------------------------------------------- the allocation shape ----

def test_it_asks_for_one_node_and_eight_gpus(text):
    directives = _directives(text)
    assert directives["nodes"] == "1"
    assert directives["gpus-per-node"] == "8"


def test_one_node_is_deliberate_and_explained(text):
    """A file-backed JournalStorage across nodes is the failure mode the study
    config warns about. If someone raises `--nodes`, the reason has to be in
    front of them."""
    assert "JournalStorage" in text or "JournalFileBackend" in text
    assert "one node" in text.lower()


def test_the_account_partition_and_qos_match_the_other_h100_scripts(text):
    """`RuntimeSpec.__post_init__` enforces the same triple for a run spec:
    partition 'batch' requires this account and qos 'medium'. A script that
    disagrees with the spec it launches gets rejected by Slurm, or worse, runs
    under an account with no allocation."""
    directives = _directives(text)
    reference = _directives(
        (REPO / "scripts/verify_study_e2e.sbatch").read_text())
    for key in ("account", "partition", "qos"):
        assert directives[key] == reference[key], key
    assert directives["account"] == "marlowe-m000151-pm06"
    assert directives["qos"] == "medium"


def test_there_are_enough_cpus_and_memory_for_eight_trainers(text):
    """Eight independent trainers, each with its own dataloader workers. At the
    16 CPUs a single-GPU script asks for, the dataloaders would contend and the
    study would be CPU-bound on a node with eight H100s."""
    directives = _directives(text)
    cpus = int(directives["cpus-per-task"])
    assert cpus >= 8 * 4, f"{cpus} CPUs for 8 trainers x 4 dataloader workers"
    mem = directives["mem"]
    assert mem.endswith("G")
    assert int(mem[:-1]) >= 8 * 32, f"{mem} for eight trainers"


def test_the_cpu_count_covers_the_base_specs_own_dataloader_request(text):
    """The number that has to be covered is on the SPEC, not chosen here."""
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = resolve_run_spec(REPO / load_study_spec(REPO / STUDY).base)
    fleet = load_study_spec(REPO / STUDY).concurrent_trials
    needed = fleet * (spec.runtime.workers + 1)
    assert int(_directives(text)["cpus-per-task"]) >= needed, (
        f"{fleet} trainers x ({spec.runtime.workers} dataloader workers + 1 "
        f"main) = {needed}")


def test_it_requests_a_walltime(text):
    directives = _directives(text)
    assert "time" in directives
    hours = int(directives["time"].split(":")[0])
    assert hours >= 4, "a 256-trial study will not finish in under 4 hours"


# ------------------------------------------ ONE invocation, not a second fanout ----

def _cli_invocations(text):
    return re.findall(r"^\s*(?:stdbuf[^\n|]*?)?python -m experimentation\.sweep\.search.*$",
                      text, re.MULTILINE)


def test_it_invokes_the_cli_exactly_twice_dry_then_real(text):
    """Once for the dry run and once for the study. Any more means a hand-rolled
    fleet -- a second, untested implementation of `_fanout`."""
    invocations = _cli_invocations(text)
    assert len(invocations) == 2, invocations
    assert "--dry_run" in invocations[0]
    assert "--dry_run" not in invocations[1]


def test_it_does_not_set_cuda_visible_devices_itself(text):
    """`_fanout` does the pinning, per worker, round-robin over what the
    allocation actually granted -- and it is tested. Setting it here would either
    fight that or narrow the allocation the fleet can see."""
    assignments = re.findall(r"^\s*(?:export\s+)?CUDA_VISIBLE_DEVICES=",
                             text, re.MULTILINE)
    assert assignments == [], assignments


def test_it_does_not_loop_over_workers(text):
    """The specific wrong shape: `for worker in $(seq 0 7)`."""
    body = text.split("set -uo pipefail", 1)[1]
    assert not re.search(r"^\s*for\s+\w*worker", body, re.MULTILINE | re.IGNORECASE)
    assert "seq 0 7" not in body


def test_it_explains_why_there_is_no_loop(text):
    """A future editor's first instinct will be to add one. The reason has to be
    in the file, not only in a commit message."""
    lowered = text.lower()
    assert "_fanout" in text
    assert "do not" in lowered and "loop" in lowered


def test_it_does_not_pass_the_fleet_size_on_the_command_line(text):
    """The fleet size is a StudySpec field. A flag could disagree with the spec
    while sharing its journal, which is the drift that got three flags removed."""
    for flag in ("--concurrent_trials", "--concurrent-trials", "--workers",
                 "--n_jobs"):
        assert flag not in text, flag


# ---------------------------------------------------- no removed CLI flags ----

REMOVED_FLAGS = ("--run_root", "--launcher", "--n_trials")


def _removed_flag_violations(path):
    """Search-CLI invocations in `path` that pass a flag the CLI removed."""
    found = []
    for command in _commands(path.read_text()):
        if "experimentation.sweep.search" not in command:
            continue
        for flag in REMOVED_FLAGS:
            # Word-boundary, so `--n_trials` does not match `--n_trials_extra`
            # and a longer flag containing a shorter one is not a false hit.
            if re.search(rf"{re.escape(flag)}(\s|=|$)", command):
                found.append((flag, command.strip()))
    return found


def test_this_script_passes_no_removed_flag():
    assert _removed_flag_violations(SCRIPT) == []


def _sbatch_scripts():
    return sorted((REPO / "scripts").glob("*.sbatch"))


def test_there_are_sbatch_scripts_to_check():
    """Guards the guard: a glob matching nothing makes the check below vacuous."""
    assert len(_sbatch_scripts()) > 5


def test_the_removed_flag_check_catches_a_known_violation():
    """The most important test in this file, because it is the one that proves
    the others are not vacuous.

    `verify_study_e2e.sbatch` really does pass `--run_root`, on a line
    CONTINUED from the invocation:

        python -m experimentation.sweep.search "$STUDY_CONFIG" \\
            --dry_run --run_root "$RUN_ROOT"

    A per-physical-line check requiring the command and the flag together
    reported that file as clean -- passing, against the exact violation it was
    written to find. So the detector is pointed at a known-bad file here, and if
    that script is ever fixed this test fails and says to pick a new fixture
    rather than silently going vacuous again.
    """
    stale = REPO / "scripts/verify_study_e2e.sbatch"
    violations = _removed_flag_violations(stale)
    assert violations, (
        f"{stale.name} no longer passes a removed flag. Good -- but this test "
        f"was the only proof the detector works. Point it at another script "
        f"that does, or construct the fixture inline.")
    assert any(flag == "--run_root" for flag, _ in violations)


@pytest.mark.parametrize("path", _sbatch_scripts(), ids=lambda p: p.name)
def test_no_committed_sbatch_script_passes_a_removed_search_flag(path):
    """Asserted for EVERY script, not only the new one.

    `verify_study_e2e.sbatch` is a KNOWN, pre-existing failure: it passes
    `--run_root` twice and would now die at argparse. It is xfailed rather than
    fixed, because fixing it means deciding where that verification job's run
    root should come from now that a flag cannot supply one -- a study-design
    call, and out of scope here. Left visible rather than silenced so it is
    found and fixed, which is the entire point of the check.
    """
    violations = _removed_flag_violations(path)
    if path.name == "verify_study_e2e.sbatch":
        pytest.xfail(
            "pre-existing: passes --run_root, which the CLI removed. Fixing it "
            "means choosing where that job's run root comes from instead, which "
            "is a study-design decision.")
    assert violations == [], (
        f"{path.name} passes a removed flag to the search CLI; the job will die "
        f"at argparse: {violations}")


# -------------------------------------------------------------- the gates ----

def test_it_validates_cuda_before_spending(text):
    """Job 439811 landed on a node where nvidia-smi saw an H100 and
    torch.cuda.is_available() was False."""
    assert "torch.cuda.is_available()" in text


def test_it_validates_that_eight_devices_are_visible(text):
    """The whole shape of this study is eight concurrent trials on eight GPUs.
    With one visible, `_fanout` wraps and eight workers time-slice one device --
    it would run, and the walltime was sized for eight."""
    assert "device_count()" in text
    assert re.search(r"count\s*<\s*8", text)


def test_it_validates_optuna_is_importable(text):
    """optuna is SIDE-LOADED from a target directory, not installed. A missing
    PYTHONPATH entry surfaces as an ImportError two hours in, after the dry run
    (which does not need optuna) passed."""
    assert "import optuna" in text
    assert "OPTUNA_SITE" in text


def test_it_dry_runs_first_and_aborts_on_failure(text):
    """The gate that costs nothing and catches a bad axis, an undeclared anchor
    power_K, a retired backend policy or a missing base spec."""
    assert "--dry_run" in text
    dry_index = text.index("--dry_run")
    tail = text[dry_index:]
    assert "STATUS dry_run=" in tail
    # An abort between the dry run and the real invocation.
    real_index = tail.index("stdbuf -oL -eL python -m experimentation.sweep.search")
    assert "exit 1" in tail[:real_index], (
        "the dry run's failure does not abort before the real launch")


def test_every_gate_aborts_rather_than_continuing(text):
    """A job that continues past its own failed precondition produces evidence
    about nothing -- job 439811 ran a whole study after its CUDA check failed and
    reported '4 trials: 4 failed', indistinguishable from a real wiring bug."""
    for status in ("gate1", "dry_run"):
        assert f"STATUS {status}=" in text
    assert text.count("exit 1") >= 3


def test_it_loads_the_known_cuda_and_gcc_modules(text):
    reference = (REPO / "scripts/verify_study_e2e.sbatch").read_text()
    for line in reference.splitlines():
        if line.startswith("module load"):
            assert line in text, f"missing: {line}"


def test_it_activates_the_existing_cuda_environment(text):
    assert "$SCRATCH/venvs/koopman-cuda" in text
    assert "source \"$VENV/bin/activate\"" in text


# ------------------------------------------------- provenance and the run root ----

def test_it_does_not_pass_allow_dirty(text):
    """A real launch refuses a dirty tree. 256 trials whose code state is an
    uncommitted mixture are 256 unreproducible results.

    Checked against executable lines: the script explains at length why the flag
    is absent, and a whole-file substring search would find the explanation and
    call it a violation -- forbidding the documentation of the very thing being
    checked.
    """
    offenders = [c for c in _commands(text) if "--allow-dirty" in c]
    assert offenders == [], offenders


def test_it_resolves_the_repo_root_from_slurm_submit_dir(text):
    """sbatch COPIES the script to a spool dir, so BASH_SOURCE is
    /cm/local/apps/slurm/var/spool/... Measured in job 440764."""
    assert "SLURM_SUBMIT_DIR" in text
    assert "BASH_SOURCE" in text, "the direct-execution fallback"
    assert text.index("SLURM_SUBMIT_DIR") < text.index(
        '$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)'), (
        "BASH_SOURCE must be the FALLBACK, not the first choice")


def test_it_validates_the_resolved_root_is_a_checkout(text):
    """Without this a submit from the wrong cwd runs against the wrong tree."""
    assert '$REPO_ROOT/experimentation' in text
    assert '$REPO_ROOT/code-tests' in text


def test_the_run_root_is_not_derived_from_the_job_id(text):
    """It comes from the study FILE, so a resubmission attaches to the same
    study. A job-id-derived root starts a fresh study on every requeue -- and
    `--requeue` is in the run system's own sbatch template."""
    assert not re.search(r"RUN_ROOT=.*SLURM_JOB_ID", text)
    assert "run_root" in text.lower()


def test_the_run_root_comes_from_the_spec_and_is_stable():
    """Asserted against the spec, not the script: the script must not carry a
    run root at all, and the spec's must not mention a job id."""
    from experimentation.sweep.search.studyspec import load_study_spec

    root = load_study_spec(REPO / STUDY).run_root
    assert root == "/scratch/m000151-pm06/jkli/study-proxy-256x17-v1"
    assert "JOB_ID" not in root and "%j" not in root


def test_resubmission_reuse_is_explained(text):
    assert "load_if_exists" in text or "ATTACHES" in text
    assert "skip_if_exists" in text


# ------------------------------------------------------------- worker logs ----

def test_each_worker_gets_a_distinct_log_destination(text):
    """Eight trainers inheriting one stdout interleave, and a traceback's first
    line lands under another worker's progress."""
    assert "WORKER_LOG_DIR" in text
    assert "mkdir -p \"$WORKER_LOG_DIR\"" in text


def test_the_log_directory_is_per_job(text):
    """Two submissions must not overwrite each other's logs, even though they
    deliberately share the run root and the journal.

    Asserted on the EXECUTABLE assignment rather than a character window around
    the first mention. The window version broke the moment a comment above the
    line grew -- it was testing the prose, not the code.
    """
    assignments = [c for c in _commands(text)
                   if "KOOPMAN_SEARCH_WORKER_LOG_DIR=" in c]
    assert assignments, "the script does not set the log directory the CLI reads"
    assert any("SLURM_JOB_ID" in c for c in assignments), assignments


def test_it_exports_the_variable_the_cli_actually_READS(text):
    """The bug this replaces: the script exported `WORKER_LOG_DIR` and nothing
    read it, so eight workers shared one stream while the comment claimed
    otherwise. The name is asserted against the constant, not spelled twice."""
    from experimentation.sweep.search.__main__ import WORKER_LOG_DIR_ENV

    assignments = [c for c in _commands(text)
                   if f"{WORKER_LOG_DIR_ENV}=" in c]
    assert assignments, (
        f"the script must export {WORKER_LOG_DIR_ENV}, which is what _fanout "
        f"reads; any other name is a directory nothing writes to")


def test_the_log_directory_creation_is_checked(text):
    """`mkdir -p` can fail (quota, read-only mount). Unchecked, the workers then
    fail one by one on an IOError with the real cause three screens up.

    The guard must be on the mkdir ITSELF -- `if ! mkdir ...` or `mkdir ... ||`.
    An earlier version searched for `exit 1` within 200 characters and passed on
    an unrelated abort further down the file, which is a test satisfied by
    proximity rather than by structure.
    """
    guarded = [c for c in _commands(text)
               if "mkdir -p" in c and "WORKER_LOG_DIR" in c
               and (c.lstrip().startswith("if ! mkdir")
                    or "||" in c.split("mkdir", 1)[1])]
    assert guarded, (
        "the WORKER_LOG_DIR mkdir is not guarded on its own exit status; write "
        "`if ! mkdir -p ...; then ... exit 1; fi` or `mkdir -p ... || exit 1`")


def test_it_fails_when_a_worker_fails(text):
    """`_fanout` returns non-zero if any child did, and the script must not
    report success over it."""
    assert "PIPESTATUS" in text
    assert 'exit "$STUDY_STATUS"' in text


def test_it_does_not_reimplement_waiting_for_workers(text):
    """The supervisor already waited. A `wait` here would be a second,
    disagreeing account of the same thing."""
    body = text.split("THE STUDY:", 1)[1]
    assert not re.search(r"^\s*wait\s*$", body, re.MULTILINE)


# --------------------------------------------------------------- the objective ----

def test_it_explains_why_throughput_is_not_in_the_objective(text):
    """Eight trainers on one node contend, so a trial's tokens/sec depends on
    what its neighbours were doing. A throughput term would score the scheduler."""
    assert "throughput" in text.lower()
    assert "contend" in text.lower() or "contention" in text.lower()


def test_the_study_it_launches_really_has_an_empty_objective():
    """The script's claim, checked against the file rather than trusted."""
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(REPO / STUDY)
    assert spec.objective == {}


def test_it_documents_the_overshoot_and_checks_the_bound(text):
    """Concurrency means up to concurrent_trials-1 extra trials. The script has
    to say so, or someone counts 263 rows in a 256-trial study and calls it a
    bug."""
    assert "overshoot" in text.lower()
    assert "concurrent_trials - 1" in text


# ----------------------------------------------------- the files it names exist ----

def test_the_study_config_it_defaults_to_is_committed(text):
    assert STUDY in text
    assert (REPO / STUDY).is_file()


def test_the_configs_that_study_names_are_committed():
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(REPO / STUDY)
    assert (REPO / spec.base).is_file()
    assert (REPO / spec.design_file).is_file()


def test_the_analysis_script_it_calls_is_committed(text):
    assert "scripts/analyze_interactions.py" in text
    assert (REPO / "scripts/analyze_interactions.py").is_file()


def test_every_repo_relative_script_it_calls_exists(text):
    """A renamed helper strands this file with a shell-level failure at 3am."""
    for match in re.finditer(r"(scripts/[\w./-]+\.py)", text):
        assert (REPO / match.group(1)).is_file(), match.group(1)


# --------------------------------------- the job must be able to START ----

def _log_dirs(text):
    """The directories --output/--error point at, with %x/%j stripped."""
    out = set()
    for match in re.finditer(r"^#SBATCH --(?:output|error)=(\S+)", text,
                             re.MULTILINE):
        out.add(str(pathlib.Path(match.group(1)).parent))
    return sorted(out)


def test_it_names_a_log_directory(text):
    assert _log_dirs(text), "no --output/--error directive"


@pytest.mark.parametrize("path", _sbatch_scripts(), ids=lambda p: p.name)
def test_every_sbatch_log_directory_exists(path):
    """Slurm does NOT create the directory for --output/--error. slurmstepd fails
    to open stdout and the batch step dies BEFORE LINE 1 -- so every gate is
    bypassed and there is no log anywhere saying why.

    Skipped rather than failed when scratch is absent: this box may not have it,
    and a test that cannot see the filesystem must not claim the directory is
    missing. Guarded below so the skip cannot become universal silently.
    """
    text = path.read_text()
    dirs = _log_dirs(text)
    if not dirs:
        pytest.skip(f"{path.name} declares no log directory")
    for directory in dirs:
        root = pathlib.Path(directory).parents[-2] if pathlib.Path(
            directory).is_absolute() else None
        if not pathlib.Path("/scratch/m000151-pm06").is_dir():
            pytest.skip("scratch is not mounted on this host")
        assert pathlib.Path(directory).is_dir(), (
            f"{path.name} writes its job log to {directory}, which does not "
            f"exist. Slurm will not create it -- the job dies before line 1 "
            f"with no log. `mkdir -p` it, or point at a directory that exists.")


def test_the_log_directory_check_is_not_universally_skipped():
    """Guards the guard: if scratch were never mounted the parametrised test
    above would skip for every script and prove nothing. On a host with scratch,
    at least one script must be really checked."""
    if not pathlib.Path("/scratch/m000151-pm06").is_dir():
        pytest.skip("scratch is not mounted on this host")
    checked = [p for p in _sbatch_scripts() if _log_dirs(p.read_text())]
    assert len(checked) >= 5, checked
