from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.autocode_state_store import AutoCodeStateStore

_REQUIRED_TOP_LEVEL: dict[str, type] = {
    "polling": dict,
    "issue_runs": dict,
    "issue_plans": dict,
    "issues": dict,
    "issue_events": dict,
    "issue_comments": dict,
    "review_feedback": dict,
    "paused_issues": dict,
    "paused_tracked_prs": dict,
    "backlog_items": dict,
    "tracked_pull_requests": dict,
    "runs": dict,
    "task_queue": dict,
    "observability": dict,
}
_TASK_STATUSES = {"queued", "running", "retry_waiting", "dead_letter", "done"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unwrap_state_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a JSON object")
    if isinstance(payload.get("state"), dict):
        return dict(payload["state"])
    return dict(payload)


def validate_state_payload(payload: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        state = _unwrap_state_payload(payload)
    except ValueError as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "summary": {},
        }

    schema_version = state.get("schema_version", 0)
    if not isinstance(schema_version, int):
        errors.append("schema_version must be an integer")

    for key, expected_type in _REQUIRED_TOP_LEVEL.items():
        value = state.get(key)
        if value is None:
            errors.append(f"missing top-level section: {key}")
            continue
        if not isinstance(value, expected_type):
            errors.append(f"top-level section {key} must be {expected_type.__name__}")

    task_queue = state.get("task_queue", {})
    queue_status_counts = {status: 0 for status in sorted(_TASK_STATUSES)}
    if isinstance(task_queue, dict):
        for task_id, record in task_queue.items():
            if not isinstance(record, dict):
                errors.append(f"task_queue[{task_id}] must be an object")
                continue
            if str(record.get("task_id", "") or "") != str(task_id):
                warnings.append(f"task_queue[{task_id}] task_id mismatch")
            status = str(record.get("status", "") or "")
            if status not in _TASK_STATUSES:
                errors.append(f"task_queue[{task_id}] has unsupported status: {status}")
                continue
            queue_status_counts[status] += 1
            if status == "running":
                if not str(record.get("lease_owner", "") or ""):
                    warnings.append(f"task_queue[{task_id}] is running without lease_owner")
                if not str(record.get("lease_acquired_at", "") or ""):
                    warnings.append(f"task_queue[{task_id}] is running without lease_acquired_at")
            max_attempts = int(record.get("max_attempts", 0) or 0)
            attempt_count = int(record.get("attempt_count", 0) or 0)
            if max_attempts < 1:
                errors.append(f"task_queue[{task_id}] max_attempts must be >= 1")
            if attempt_count < 0:
                errors.append(f"task_queue[{task_id}] attempt_count must be >= 0")
            if status == "dead_letter":
                warnings.append(f"task_queue[{task_id}] is in dead_letter")

    observability = state.get("observability", {})
    recent_events = observability.get("recent_events", []) if isinstance(observability, dict) else []
    cycles = observability.get("cycles", {}) if isinstance(observability, dict) else {}
    counters = observability.get("counters", {}) if isinstance(observability, dict) else {}
    if isinstance(recent_events, list) and len(recent_events) > 30:
        warnings.append("recent_events exceeds in-memory retention limit 30")

    summary = {
        "schema_version": schema_version if isinstance(schema_version, int) else 0,
        "tasks": len(task_queue) if isinstance(task_queue, dict) else 0,
        "queued_tasks": queue_status_counts.get("queued", 0),
        "running_tasks": queue_status_counts.get("running", 0),
        "retry_waiting_tasks": queue_status_counts.get("retry_waiting", 0),
        "dead_letter_tasks": queue_status_counts.get("dead_letter", 0),
        "tracked_prs": len(state.get("tracked_pull_requests", {})) if isinstance(state.get("tracked_pull_requests"), dict) else 0,
        "runs": len(state.get("runs", {})) if isinstance(state.get("runs"), dict) else 0,
        "observability_counters": len(counters) if isinstance(counters, dict) else 0,
        "observed_cycles": len(cycles) if isinstance(cycles, dict) else 0,
        "recent_events": len(recent_events) if isinstance(recent_events, list) else 0,
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def validate_state_file(path: str) -> dict[str, Any]:
    state_path = Path(path).resolve()
    if not state_path.exists():
        return {
            "ok": False,
            "errors": [f"state file does not exist: {state_path}"],
            "warnings": [],
            "summary": {},
        }
    try:
        payload = _read_json(state_path)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "errors": [f"invalid json: {exc.msg}"],
            "warnings": [],
            "summary": {},
        }
    result = validate_state_payload(payload)
    result["path"] = str(state_path)
    return result


def export_snapshot_file(state_file: str, output_file: str) -> dict[str, Any]:
    store = AutoCodeStateStore(state_file)
    snapshot = store.export_snapshot()
    output_path = Path(output_file).resolve()
    _write_json(output_path, snapshot)
    return {
        "output_path": str(output_path),
        "snapshot": snapshot,
    }


def restore_snapshot_file(
    state_file: str,
    snapshot_file: str,
    *,
    create_backup: bool = True,
) -> dict[str, Any]:
    snapshot_path = Path(snapshot_file).resolve()
    payload = _read_json(snapshot_path)
    store = AutoCodeStateStore(state_file)
    result = store.restore_snapshot(payload, create_backup=create_backup)
    result["snapshot_path"] = str(snapshot_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoCode state administration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a state file")
    validate_parser.add_argument("--state-file", required=True)
    validate_parser.add_argument("--json", action="store_true", dest="json_output")

    snapshot_parser = subparsers.add_parser("snapshot", help="Export a state snapshot")
    snapshot_parser.add_argument("--state-file", required=True)
    snapshot_parser.add_argument("--output", required=True)

    restore_parser = subparsers.add_parser("restore", help="Restore a state snapshot")
    restore_parser.add_argument("--state-file", required=True)
    restore_parser.add_argument("--snapshot-file", required=True)
    restore_parser.add_argument("--yes", action="store_true")
    restore_parser.add_argument("--no-backup", action="store_true")
    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_state_file(args.state_file)
        if args.json_output:
            _print_json(result)
        else:
            print(f"state_file: {result.get('path', '')}")
            print(f"ok: {result['ok']}")
            print(f"errors: {len(result['errors'])}")
            print(f"warnings: {len(result['warnings'])}")
            for item in result["errors"]:
                print(f"ERROR: {item}")
            for item in result["warnings"]:
                print(f"WARNING: {item}")
        return 0 if result["ok"] else 1

    if args.command == "snapshot":
        result = export_snapshot_file(args.state_file, args.output)
        print(result["output_path"])
        return 0

    if args.command == "restore":
        if not args.yes:
            print("--yes is required for restore", file=sys.stderr)
            return 2
        result = restore_snapshot_file(
            args.state_file,
            args.snapshot_file,
            create_backup=not args.no_backup,
        )
        _print_json(result)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
