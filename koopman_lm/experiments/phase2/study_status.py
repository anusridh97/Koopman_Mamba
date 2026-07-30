"""Plan the next bounded Phase 2a Slurm batch from study and claim state."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import git_identity
    from .run_study import (
        _recorded_active_gpu_hours,
        build_ledger_contract,
        build_storage,
        count_fresh_optuna_trials,
        count_fresh_trial_target_guards,
        ensure_ledger_contract,
    )
    from .spec import load_spec, stable_hash
except ImportError:
    from manifest import git_identity  # type: ignore
    from run_study import (  # type: ignore
        _recorded_active_gpu_hours,
        build_ledger_contract,
        build_storage,
        count_fresh_optuna_trials,
        count_fresh_trial_target_guards,
        ensure_ledger_contract,
    )
    from spec import load_spec, stable_hash  # type: ignore


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _claim_snapshot(output_root: str | Path) -> dict[str, Any]:
    claims_dir = Path(output_root).expanduser().resolve() / "claims"
    states: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    terminal_actual = 0.0
    active_reserved = 0.0
    budget_violation_count = 0
    if not claims_dir.is_dir():
        return {
            "claim_count": 0,
            "states": {},
            "outcomes": {},
            "terminal_gpu_hours_actual": 0.0,
            "active_gpu_hours_reserved": 0.0,
            "compute_budget_violation_count": 0,
        }
    for path in sorted(claims_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable claim {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Claim must be an object: {path}")
        state = payload.get("state")
        if state not in {"running", "recovery_queued", "terminal"}:
            raise ValueError(f"Invalid claim state in {path}: {state!r}")
        states[str(state)] += 1
        if state == "terminal":
            outcome = payload.get("outcome")
            outcomes[str(outcome)] += 1
            terminal_actual += _finite_nonnegative(
                payload.get("gpu_hours_actual"),
                f"{path}: gpu_hours_actual",
            )
            budget_violation_count += (
                payload.get("compute_budget_violation") is not None
            )
        else:
            reservation = _finite_nonnegative(
                payload.get("gpu_hours_reserved"),
                f"{path}: gpu_hours_reserved",
            )
            active_reserved += max(
                reservation,
                _recorded_active_gpu_hours(payload),
            )
    return {
        "claim_count": sum(states.values()),
        "states": dict(sorted(states.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "terminal_gpu_hours_actual": terminal_actual,
        "active_gpu_hours_reserved": active_reserved,
        "compute_budget_violation_count": budget_violation_count,
    }


def build_status_report(
    spec: Mapping[str, Any],
    *,
    optuna_trials: list[Any],
    output_root: str | Path,
    study_identity: Mapping[str, Any] | None = None,
    storage_backend: str | None = None,
    storage_identity_sha256: str | None = None,
    ledger_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed next-batch recommendation without submitting jobs."""

    claim_state = _claim_snapshot(output_root)
    trial_states = Counter(
        str(getattr(getattr(trial, "state", None), "name", "UNKNOWN"))
        for trial in optuna_trials
    )
    requested = int(spec["study"]["requested_trials"])
    budget = spec["study"]["compute_budget"]
    maximum_total = budget.get("maximum_total_gpu_hours")
    maximum_full = budget.get("maximum_full_trial_gpu_hours")
    maximum_concurrent = int(budget["maximum_concurrent_trials"])
    reasons: list[str] = []
    if spec.get("status") != "scientific_ready":
        reasons.append("spec_not_scientific_ready")
    if budget.get("status") != "approved":
        reasons.append("compute_budget_not_approved")
    try:
        maximum_total_value = _finite_nonnegative(
            maximum_total, "maximum_total_gpu_hours"
        )
        maximum_full_value = _finite_nonnegative(
            maximum_full, "maximum_full_trial_gpu_hours"
        )
        if maximum_total_value <= 0.0 or maximum_full_value <= 0.0:
            raise ValueError("GPU-hour limits must be positive")
    except ValueError:
        maximum_total_value = 0.0
        maximum_full_value = 0.0
        reasons.append("gpu_hour_limits_unresolved")

    terminal_actual = float(claim_state["terminal_gpu_hours_actual"])
    active_reserved = float(claim_state["active_gpu_hours_reserved"])
    remaining_gpu_hours = max(
        0.0, maximum_total_value - terminal_actual - active_reserved
    )
    fresh_budget_slots = (
        int(remaining_gpu_hours // maximum_full_value)
        if maximum_full_value > 0.0
        else 0
    )
    running_claims = int(claim_state["states"].get("running", 0))
    queued_recoveries = int(
        claim_state["states"].get("recovery_queued", 0)
    )
    recovery_worker_slots = min(
        queued_recoveries,
        max(0, maximum_concurrent - running_claims),
    )
    fresh_concurrency_slots = max(
        0,
        maximum_concurrent - running_claims - queued_recoveries,
    )
    fresh_trial_count = count_fresh_optuna_trials(optuna_trials)
    target_guard_count = count_fresh_trial_target_guards(optuna_trials)
    remaining_trial_slots = max(0, requested - fresh_trial_count)
    recommended_fresh = min(
        fresh_budget_slots,
        fresh_concurrency_slots,
        remaining_trial_slots,
    )
    recommended = recovery_worker_slots + recommended_fresh
    if remaining_trial_slots == 0 and recovery_worker_slots == 0:
        reasons.append("requested_trial_target_reached")
    if (
        recovery_worker_slots == 0
        and fresh_concurrency_slots == 0
    ):
        reasons.append("concurrency_ceiling_reached")
    if fresh_budget_slots == 0 and recovery_worker_slots == 0:
        reasons.append("insufficient_unreserved_gpu_hours_for_one_full_trial")
    if int(claim_state["compute_budget_violation_count"]) > 0:
        reasons.append("compute_budget_violation_requires_reapproval")

    return {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "study_identity": (
            dict(study_identity) if study_identity is not None else None
        ),
        "storage_backend": storage_backend,
        "storage_identity_sha256": storage_identity_sha256,
        "ledger_contract_sha256": ledger_contract_sha256,
        "claims_output_root": str(Path(output_root).expanduser().resolve()),
        "requested_trials": requested,
        "optuna_trial_count": len(optuna_trials),
        "fresh_optuna_trial_count": fresh_trial_count,
        "fresh_trial_target_guard_count": target_guard_count,
        "queued_recovery_count": queued_recoveries,
        "optuna_states": dict(sorted(trial_states.items())),
        "claims": claim_state,
        "compute_budget": {
            "maximum_total_gpu_hours": maximum_total,
            "maximum_full_trial_gpu_hours": maximum_full,
            "terminal_gpu_hours_actual": terminal_actual,
            "active_gpu_hours_reserved": active_reserved,
            "remaining_unreserved_gpu_hours": remaining_gpu_hours,
            "maximum_concurrent_trials": maximum_concurrent,
        },
        "recommended_recovery_tasks": recovery_worker_slots,
        "recommended_fresh_tasks": recommended_fresh,
        "recommended_next_array_tasks": recommended,
        "recommended_slurm_array": (
            f"0-{recommended - 1}" if recommended > 0 else None
        ),
        "launch_allowed": recommended > 0 and not reasons,
        "hold_reasons": sorted(set(reasons)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name")
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--print-next-array",
        action="store_true",
        help="Print only the next Slurm array expression; exits nonzero on hold.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = load_spec(args.spec)
    study_name = args.study_name or spec["study_name"]
    if study_name != spec["study_name"]:
        raise SystemExit(
            "--study-name must exactly match the frozen spec study_name: "
            f"{spec['study_name']}"
        )
    if args.storage.startswith("postgresql+psycopg://"):
        backend = "postgresql"
    elif args.storage.startswith("journal://"):
        backend = "scg_validated_journal_storage"
    else:
        raise SystemExit(
            "Storage must use postgresql+psycopg:// or journal://"
        )
    if backend != spec["storage"]["backend"]:
        raise SystemExit(
            f"Storage URL backend {backend!r} does not match frozen spec "
            f"backend {spec['storage']['backend']!r}"
        )
    try:
        capabilities = json.loads(Path(args.capabilities).read_text())
        data_manifest = json.loads(Path(args.data_manifest).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not load status identity inputs: {exc}") from exc
    if not isinstance(capabilities, Mapping) or not isinstance(
        data_manifest, Mapping
    ):
        raise SystemExit("Capabilities and data manifest must be JSON objects")
    code = git_identity(spec["_repo_root"])
    ledger_contract = build_ledger_contract(
        spec,
        output_root=Path(args.output_root),
        storage=args.storage,
        code_identity=code,
        data_identity=data_manifest,
        capability_identity=capabilities,
    )
    expected_identity = {
        key: ledger_contract[key]
        for key in (
            "study_name",
            "spec_hash",
            "code_commit",
            "code_dirty",
            "data_hash",
            "capability_hash",
        )
    }
    expected_contract = {
        key: ledger_contract[key]
        for key in (
            *expected_identity,
            "claims_output_root",
            "storage_identity_sha256",
            "ledger_contract_sha256",
        )
    }
    ensure_ledger_contract(
        Path(args.output_root),
        ledger_contract,
        create=False,
    )
    storage = build_storage(args.storage, approved_backend=backend)
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Install optional nas dependencies") from exc
    study = optuna.load_study(study_name=study_name, storage=storage)
    observed_identity = study.system_attrs.get("echo_phase2_contract_v1")
    if observed_identity != expected_contract:
        raise SystemExit(
            "Optuna study contract does not match the supplied frozen "
            f"spec/code/data/capabilities: {observed_identity!r}"
        )
    report = build_status_report(
        spec,
        optuna_trials=list(study.get_trials(deepcopy=False)),
        output_root=args.output_root,
        study_identity=expected_identity,
        storage_backend=backend,
        storage_identity_sha256=ledger_contract[
            "storage_identity_sha256"
        ],
        ledger_contract_sha256=ledger_contract[
            "ledger_contract_sha256"
        ],
    )
    if args.output:
        try:
            from .manifest import atomic_write_json
        except ImportError:
            from manifest import atomic_write_json  # type: ignore
        atomic_write_json(args.output, report)
    if args.print_next_array:
        expression = report["recommended_slurm_array"]
        if not report["launch_allowed"] or expression is None:
            raise SystemExit(
                "HOLD: " + ", ".join(report["hold_reasons"])
            )
        print(expression)
        return
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
