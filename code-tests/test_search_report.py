"""experimentation/sweep/search/report.py -- what a finished study hands back.

Four artefacts: a row-per-trial CSV, a ranked markdown table, a pointer to the
best trial, and a promotion sweep for the top few.

The promotion artefact is where this deviates from the original harness on
purpose. That version generated a bash script -- `cd`, then N train commands and
N eval commands in sequence, to be babysat. There is no longer any reason for
that: a promotion set is a handful of chosen configs at a longer run length,
which is precisely a `cells:` sweep. Emitting one means the confirmation runs get
the same content-hashed identities, dirty-tree gate and single Slurm array as
everything else, instead of a script whose failure mode is "line 40 died and
lines 41+ ran anyway".

`best_trial.json` records the winning trial's run_id and run_dir rather than
copying its model config. The config is already in that run's own materialized
spec.yaml, and a second copy is a second thing that can disagree.
"""
import csv
import json
import re

import pytest
import yaml

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _quiet_optuna():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


def _space():
    from koopman_lm.config import build_config
    from experimentation.sweep.search.space import search_space
    return search_space(build_config("50m"), base_name="50m")


def _data_rows(text):
    """Ranked rows only. Counting lines that merely start with "| " also catches
    the table header, which is what the first version of these assertions did."""
    return [line for line in text.splitlines()
            if re.match(r"^\|\s*\d+\s*\|", line)]


def _study_with_results(tmp_path, values, anchors=()):
    """A study whose trials have completed with the given objective values."""
    from experimentation.sweep.search.study import create_study, to_distributions

    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    distributions = to_distributions(_space())
    for i, value in enumerate(values):
        trial = study.ask(distributions)
        # Public API: set_trial_user_attr on the storage would work but reaches
        # through two layers of private attribute to get there.
        if i < len(anchors):
            trial.set_user_attr("anchor_name", anchors[i])
        trial.set_user_attr("run_id", f"run{i:05d}")
        if value is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, value)
    return study


# ---------------------------------------------------------------- csv ----

def test_trials_csv_has_one_row_per_trial(tmp_path):
    from experimentation.sweep.search.report import write_trials_csv

    study = _study_with_results(tmp_path, [2.0, 1.5, None])
    path = write_trials_csv(study, tmp_path)
    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert len(rows) == 3
    assert {r["state"] for r in rows} == {"COMPLETE", "FAIL"}


def test_trials_csv_flattens_params_and_attrs_into_columns(tmp_path):
    from experimentation.sweep.search.report import write_trials_csv

    study = _study_with_results(tmp_path, [2.0])
    rows = list(csv.DictReader(write_trials_csv(study, tmp_path).read_text().splitlines()))
    assert "param_ska_rank" in rows[0]
    assert "attr_run_id" in rows[0]
    assert rows[0]["attr_run_id"] == "run00000"


def test_trials_csv_of_an_empty_study_still_has_a_header(tmp_path):
    """A study that crashed before its first trial must still produce a readable
    file rather than a zero-byte one."""
    from experimentation.sweep.search.report import write_trials_csv
    from experimentation.sweep.search.study import create_study

    study = create_study(study_name="empty", study_dir=tmp_path, seed=1)
    path = write_trials_csv(study, tmp_path)
    assert path.read_text().strip().startswith("number")


# ----------------------------------------------------------- markdown ----

def test_top_trials_are_ranked_best_first(tmp_path):
    from experimentation.sweep.search.report import write_top_trials_md

    study = _study_with_results(tmp_path, [2.0, 1.5, 3.0])
    text = write_top_trials_md(study, tmp_path).read_text()
    body = _data_rows(text)
    assert body[0].split("|")[3].strip().startswith("1.5")


def test_top_trials_omits_failed_trials(tmp_path):
    from experimentation.sweep.search.report import write_top_trials_md

    study = _study_with_results(tmp_path, [2.0, None])
    text = write_top_trials_md(study, tmp_path).read_text()
    assert len(_data_rows(text)) == 1, "one ranked row, the failure excluded"


def test_top_trials_marks_which_rows_were_curated_anchors(tmp_path):
    """Reading the table, the first question is whether a hand-chosen point beat
    the sampled ones."""
    from experimentation.sweep.search.report import write_top_trials_md

    study = _study_with_results(tmp_path, [1.0, 2.0], anchors=("baseline",))
    text = write_top_trials_md(study, tmp_path).read_text()
    assert "baseline" in text


def test_top_trials_respects_the_limit(tmp_path):
    from experimentation.sweep.search.report import write_top_trials_md

    study = _study_with_results(tmp_path, [3.0, 1.0, 2.0, 4.0])
    text = write_top_trials_md(study, tmp_path, limit=2).read_text()
    assert len(_data_rows(text)) == 2


def test_top_trials_says_so_when_nothing_completed(tmp_path):
    from experimentation.sweep.search.report import write_top_trials_md

    study = _study_with_results(tmp_path, [None])
    assert "no completed trials" in write_top_trials_md(study, tmp_path).read_text().lower()


# --------------------------------------------------------------- best ----

def test_best_trial_points_at_the_run_rather_than_copying_its_config(tmp_path):
    from experimentation.sweep.search.report import write_best_trial

    study = _study_with_results(tmp_path, [2.0, 1.25])
    payload = json.loads(write_best_trial(study, tmp_path).read_text())
    assert payload["objective"] == pytest.approx(1.25)
    assert payload["run_id"] == "run00001"
    assert "params" in payload
    assert "model" not in payload, (
        "the config lives in that run's own spec.yaml; a second copy is a second "
        "thing that can disagree")


def test_best_trial_is_absent_when_nothing_completed(tmp_path):
    from experimentation.sweep.search.report import write_best_trial

    study = _study_with_results(tmp_path, [None])
    assert write_best_trial(study, tmp_path) is None


# --------------------------------------------------------- promotions ----

def _base_spec(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir(exist_ok=True)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({
        "name": "50m-fineweb-3b", "model": "50m",
        "data": {"kind": "shard", "shard_dir": str(shard),
                 "tokenizer": "NousResearch/Llama-2-7b-hf",
                 "mix": {"fineweb": 1.0}, "n_tokens": 3000000000},
        "optim": {"lr": 4.0e-4, "warmup_steps": 300, "max_steps": 15000,
                  "effective_batch": 96},
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }))
    return path


def test_promotion_sweep_is_a_loadable_cells_sweep(tmp_path):
    """The deviation from the original: a sweep, not a bash script."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_promotion_sweep
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    study = _study_with_results(tmp_path, [3.0, 1.0, 2.0])
    out = tmp_path / "promotions.yaml"
    write_promotion_sweep(study, out, base_spec=_base_spec(tmp_path),
                          base_model=build_config("50m"), top_k=2,
                          seeds=(42, 43, 44), max_steps=3000, seq_len=2048)

    sweep = load_sweep_spec(out)
    assert sweep.cells is not None
    assert len(sweep.cells) == 2 * 3, "top_k x seeds"
    assert len(expand_cells(sweep)) == 6


def test_promotion_cells_vary_the_seed_and_extend_the_run(tmp_path):
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_promotion_sweep
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    study = _study_with_results(tmp_path, [1.0])
    out = tmp_path / "promotions.yaml"
    write_promotion_sweep(study, out, base_spec=_base_spec(tmp_path),
                          base_model=build_config("50m"), top_k=1,
                          seeds=(42, 43), max_steps=3000, seq_len=2048)

    specs = [c.spec for c in expand_cells(load_sweep_spec(out))]
    assert {s.runtime.seed for s in specs} == {42, 43}
    assert all(s.optim.max_steps == 3000 for s in specs)
    assert all(s.model.max_seq_len == 2048 for s in specs)


def test_promotion_replicates_share_a_group_id_and_differ_by_run_id(tmp_path):
    """Seeds of one config are one experiment with several datapoints; the run
    system already encodes that, and promotions must not accidentally break it."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_promotion_sweep
    from experimentation.run.spec import group_id, run_id
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    study = _study_with_results(tmp_path, [1.0])
    out = tmp_path / "promotions.yaml"
    write_promotion_sweep(study, out, base_spec=_base_spec(tmp_path),
                          base_model=build_config("50m"), top_k=1,
                          seeds=(42, 43, 44), max_steps=3000, seq_len=2048)

    specs = [c.spec for c in expand_cells(load_sweep_spec(out))]
    assert len({group_id(s) for s in specs}) == 1
    assert len({run_id(s) for s in specs}) == 3


def test_promotion_warmup_is_rescaled_to_the_longer_run(tmp_path):
    """warmup_ratio is a ratio; promoting a 600-step screen to 3000 steps with a
    600-step warmup baked in would warm up for a fifth of the run."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_promotion_sweep
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    study = _study_with_results(tmp_path, [1.0])
    out = tmp_path / "promotions.yaml"
    write_promotion_sweep(study, out, base_spec=_base_spec(tmp_path),
                          base_model=build_config("50m"), top_k=1,
                          seeds=(42,), max_steps=3000, seq_len=2048)
    spec = expand_cells(load_sweep_spec(out))[0].spec
    assert spec.optim.warmup_steps < 3000 // 4


def test_no_promotion_sweep_when_nothing_completed(tmp_path):
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_promotion_sweep

    study = _study_with_results(tmp_path, [None])
    out = tmp_path / "promotions.yaml"
    assert write_promotion_sweep(study, out, base_spec=_base_spec(tmp_path),
                                 base_model=build_config("50m"), top_k=3,
                                 seeds=(42,), max_steps=3000, seq_len=2048) is None
    assert not out.exists()


# -------------------------------------------------------- everything ----

def test_write_report_produces_every_artefact(tmp_path):
    from koopman_lm.config import build_config
    from experimentation.sweep.search.report import write_report

    study = _study_with_results(tmp_path, [2.0, 1.0])
    written = write_report(study, tmp_path, base_spec=_base_spec(tmp_path),
                           base_model=build_config("50m"))
    assert set(written) >= {"trials_csv", "top_trials", "best_trial"}
    for path in written.values():
        assert path is None or path.is_file()
