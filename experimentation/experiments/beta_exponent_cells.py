"""The key/value exponent arm: its cells, its gates, and its array-index map.

`docs/beta-exponent-preregistration.md` fixes the design; this is that design as
importable data, so the login-node preparer and each GPU array task read ONE copy
of it.

## Why an array of 1-GPU tasks, not one whole-node job

The cluster is GPU-saturated: every node reports 8/8 GRES allocated. A whole-node
8-GPU request queued ~12 hours out while a 1-GPU request BACKFILLED in ten minutes
(job 446640, observed the same morning). A slurm array of 1-GPU tasks backfills
into the gaps; one 8-GPU job waits for a node to drain.

That splits the arm across processes that never see each other, which changes what
the gates have to be. An inline shell block checked the cells once, in the job that
then ran them. An array task has to re-check ITS OWN cell, because nothing
guarantees it is running the arm that was prepared: a stale `RUN_ROOT`, an array
re-submitted against an edited cell list, or an index off the end of the manifest
each produce a silently wrong run rather than an error. `preflight_failures` is
therefore called twice -- once at prepare time over the whole arm, once per task
over its own cell -- and it is the same function both times.

## The gates, and what each is protecting

  * **Route.** On a CHUNKED route `beta_proj.bias` gradient cosine is ~0.00, so a
    beta comparison there measures a gate receiving no usable gradient. It would
    return numbers and they would mean nothing. `mqar_finetune`'s default
    `--model_size 50m` is prefix_scan; table2's default `1m` is chunked.
  * **gamma == 1, K == 1.** gamma > 1 could rescue a weak cell by amplification,
    which confounds function with scale.
  * **Ridge.** The ridge-matched controls are the arm's mandatory confound
    control. A control that silently ran at the base ridge would be a duplicate
    of its own treatment, and the arm would report "ridge does not explain it"
    having never varied ridge.
  * **Distinct identity and distinct name.** Two cells sharing a `config_hash`
    are the same experiment; two sharing a NAME share a run directory even when
    their configs differ, because the name is what the path is built from.
  * **The cells exist.** A checkout predating the exponent decomposition would run
    a five-cell arm and the log would call it seven.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import yaml

from koopman_lm.config import BETA_POLICIES, KoopmanLMConfig, config_hash


@dataclass(frozen=True)
class Cell:
    """One screening cell.

    `name` is distinct from `policy` on purpose: the two ridge controls run the
    SAME policy at a different ridge, and the report groups on `name`. Grouping
    on `policy` would average each control into the treatment it exists to
    control for, which is the one thing it must not do.
    """
    name: str
    policy: str
    ridge: float


#: The arm. `head_scalar` is absent by instruction -- see the pre-registration's
#: note that its "0/3 grokked" is a 4000-step artefact (it groks at step 12000 at
#: the 24000-step budget), so the exclusion is honoured but its stated reason is
#: not the reason.
CELLS: tuple = (
    Cell("learned", "learned", 0.01),
    Cell("one", "one", 0.01),
    Cell("linear", "linear", 0.01),
    Cell("key_linear_value_sqrt", "key_linear_value_sqrt", 0.01),
    Cell("key_sqrt_value_linear", "key_sqrt_value_linear", 0.01),
    # The mandatory ridge controls, bracketing the base from BOTH sides so the
    # convergence test does not depend on which direction was chosen. `linear`
    # puts beta^2 in G, so at the shared beta = 0.5 init its own-weight is 0.25
    # against `learned`'s 0.5: exactly twice as ridge-regularised against a fixed
    # ridge (factor measured by test_beta_exponent_decomposition.py).
    Cell("learned-ridge2x", "learned", 0.02),
    Cell("linear-ridge0.5x", "linear", 0.005),
)

#: Paired across cells. Six rather than three because measured throughput permits
#: it -- job 446145's log gives ~8.6 steps/s at this geometry, so a 24000-step run
#: is ~47 minutes -- and six paired seeds is near confirmation-grade rather than
#: screening-grade.
SEEDS: tuple = (42, 43, 44, 45, 46, 47)

#: AMENDMENT 1 of the pre-registration. Job 446106 (~4000 steps) is the source of
#: "linear 3/3, learned 1/3, head_scalar 0/3"; job 446145 ran the same cells at
#: 24000 and got linear 3/3, learned 2/3, head_scalar 1/3. The horizon was
#: carrying most of the separation, so this matches 446145 exactly and the arm
#: REPLICATES it rather than being compared across horizons.
STEPS = 24000

#: 96 evals over the budget, against 446145's 8. `grok_step` and `lc_area`
#: resolution is bounded by the cadence, and a 4000-step cadence cannot tell
#: grokking at 4001 from grokking at 7999.
EVAL_EVERY = 250

#: The cell, reused UNCHANGED from job 446102/446106/446145 so it cannot have been
#: chosen to favour a policy that did not exist when it was picked.
KV = 8
GAP = 128
TASK_VOCAB = 128


@dataclass(frozen=True)
class Task:
    """One array task: a (cell, seed) pair at a fixed array index."""
    index: int
    cell: Cell
    seed: int

    @property
    def run_name(self) -> str:
        """`mqar-<cell>-seed<N>`, which is what
        `scripts/report_beta_exponent_arm.py` greps for. A task writing any other
        name would be invisible to the report -- and that reads as a missing run
        rather than as a naming bug."""
        return f"mqar-{self.cell.name}-seed{self.seed}"


def resolve_cell(cell: Cell, base_model: KoopmanLMConfig) -> KoopmanLMConfig:
    """The base proxy model with this cell's policy and ridge applied."""
    return dataclasses.replace(
        base_model, ska_beta_policy=cell.policy, ska_ridge=float(cell.ridge))


def preflight_failures(cells: Sequence[Cell],
                       base_model: KoopmanLMConfig) -> List[str]:
    """Every reason this arm must not launch. Empty list means "go".

    Returns a list rather than raising so a caller can report ALL the problems at
    once. Discovering the second gate failure only after fixing the first costs
    another multi-hour queue wait on a saturated cluster.
    """
    failures: List[str] = []

    for needed in ("key_linear_value_sqrt", "key_sqrt_value_linear"):
        if needed not in BETA_POLICIES:
            failures.append(
                f"{needed} is not in BETA_POLICIES -- this checkout predates the "
                f"exponent decomposition, so a 7-cell arm would silently run 5")

    seen_names, seen_hashes = {}, {}
    for cell in cells:
        if cell.policy not in BETA_POLICIES:
            failures.append(
                f"cell {cell.name}: policy {cell.policy!r} is not a "
                f"BETA_POLICIES member; declared set is {sorted(BETA_POLICIES)}")
            continue
        if cell.name in seen_names:
            failures.append(
                f"cell name {cell.name!r} is used twice -- the run directory is "
                f"built from the name, so the second would overwrite the first")
        seen_names[cell.name] = True

        cfg = resolve_cell(cell, base_model)

        if not (cfg.ska_prefix_scan or cfg.ska_inverse_cholesky
                or cfg.ska_exact_intrachunk):
            failures.append(
                f"cell {cell.name}: resolved route is CHUNKED. On a chunked route "
                f"beta_proj.bias gradient cosine is ~0.00, so this cell would "
                f"measure a gate receiving no usable gradient. REFUSED.")
        if float(cfg.ska_gamma_value) != 1.0 or cfg.ska_gamma_learnable:
            failures.append(
                f"cell {cell.name}: gamma is {cfg.ska_gamma_value} "
                f"(learnable={cfg.ska_gamma_learnable}); the arm pins gamma = 1 "
                f"so no cell can be rescued by amplification")
        # `ska_gamma_bounds` is checked SEPARATELY from `ska_gamma_learnable`
        # because it is a second, independent way to make gamma trainable:
        # `modules/seq/ska.py` promotes gamma to an nn.Parameter whenever
        # `gamma_bounds is not None`, REGARDLESS of the learnable flag. Without
        # this line a config with bounds set and learnable false passes a gate
        # whose message claims "the arm pins gamma = 1" and then trains a
        # drifting gamma. Latent at the proxy (its bounds are null) and one
        # config edit away from live.
        if getattr(cfg, "ska_gamma_bounds", None) is not None:
            failures.append(
                f"cell {cell.name}: ska_gamma_bounds is "
                f"{cfg.ska_gamma_bounds!r}, which makes gamma an nn.Parameter "
                f"even at gamma_learnable=False. The arm pins gamma = 1.")
        if int(cfg.ska_power_K) != 1:
            failures.append(
                f"cell {cell.name}: power_K is {cfg.ska_power_K}, not 1")
        # NOT `float(cfg.ska_ridge) != cell.ridge` -- `resolve_cell` just
        # assigned it, so that comparison cannot fire and would be a gate in
        # name only. What can actually go wrong is the value not surviving
        # `KoopmanLMConfig.__post_init__`'s coercion, or not surviving the round
        # trip through the flat YAML the cell trains from. The first is checked
        # here; the second is `write_model_yaml` + the preparer's `load_config`
        # comparison, which is the check that protects the ridge control.
        if not isinstance(cfg.ska_ridge, float) or cfg.ska_ridge <= 0:
            failures.append(
                f"cell {cell.name}: ska_ridge came out of the dataclass as "
                f"{cfg.ska_ridge!r}; the ridge control needs a positive float")

        h = config_hash(cfg)
        if h in seen_hashes:
            failures.append(
                f"cells {cell.name} and {seen_hashes[h]} share config_hash "
                f"{h[:8]} -- they are the same experiment, so they would claim "
                f"one run directory")
        seen_hashes[h] = cell.name

    return failures


def route_of(cfg: KoopmanLMConfig) -> str:
    """The resolved backend as a word. `CHUNKED` is spelled loudly rather than as
    a False triple, because it is the one that invalidates a beta measurement."""
    if cfg.ska_prefix_scan:
        return "prefix_scan"
    if cfg.ska_inverse_cholesky:
        return "inverse_cholesky"
    if cfg.ska_exact_intrachunk:
        return "exact_intrachunk"
    return "CHUNKED"


def write_model_yaml(cell: Cell, cfg: KoopmanLMConfig, out_dir: Path) -> Path:
    """The flat model YAML this cell trains from.

    Generated per run rather than committed, so it cannot drift from
    `configs/runs/proxy-256x17.yaml` the way a committed copy would. Named after
    the CELL, so a task loading the wrong file is a missing-file error rather than
    a silently wrong policy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model-{cell.name}.yaml"
    path.write_text(
        f"# GENERATED for exponent-arm cell {cell.name!r} "
        f"(policy={cell.policy}, ridge={cell.ridge}).\n"
        "# The model section of configs/runs/proxy-256x17.yaml with this cell's\n"
        "# beta policy and ridge applied, flattened because build_config takes a\n"
        "# FLAT model config while the proxy is a run spec. Generated per run so\n"
        "# it cannot drift from the proxy spec the LM arm uses.\n"
        f"# config_hash {config_hash(cfg)}\n"
        + yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True))
    return path


def manifest(cells: Sequence[Cell] = CELLS,
             seeds: Sequence[int] = SEEDS) -> List[Task]:
    """Array index -> (cell, seed), GROUPED BY SEED.

    Seed-major, not cell-major, and that ordering is the point: if the array is
    cut short -- cancelled, preempted, or the allocation ends -- the completed
    prefix is N COMPLETE paired seeds across all cells rather than a ragged set
    with some cells at six seeds and others at one. A ragged set is not a paired
    design, and pairing is what the arm's comparison rests on.
    """
    out: List[Task] = []
    i = 0
    for seed in seeds:
        for cell in cells:
            out.append(Task(i, cell, seed))
            i += 1
    return out


def task_count(cells: Sequence[Cell] = CELLS,
               seeds: Sequence[int] = SEEDS) -> int:
    return len(cells) * len(seeds)


def task_for_index(index: int, cells: Sequence[Cell] = CELLS,
                   seeds: Sequence[int] = SEEDS) -> Task:
    """The task at `index`, or `IndexError`.

    Raises rather than clamping or wrapping. An array submitted wider than the
    manifest has to fail loudly: a task that silently exited 0 would report as a
    completed run with no data, which is indistinguishable from a training
    failure in the final table.
    """
    tasks = manifest(cells, seeds)
    if index < 0 or index >= len(tasks):
        raise IndexError(
            f"array task index {index} is outside the manifest (0..{len(tasks)-1}"
            f" for {len(cells)} cells x {len(seeds)} seeds). The array width and "
            f"the cell list disagree; one of them is stale.")
    return tasks[index]
