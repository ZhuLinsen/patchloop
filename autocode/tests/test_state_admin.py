import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent.autocode_state_store import AutoCodeStateStore
from tools.state_admin import export_snapshot_file, restore_snapshot_file, validate_state_file, validate_state_payload


class StateAdminTests(unittest.TestCase):
    def test_validate_state_payload_reports_dead_letter_warning(self):
        payload = {
            "schema_version": 5,
            "polling": {},
            "issue_runs": {},
            "issue_plans": {},
            "issues": {},
            "issue_events": {},
            "issue_comments": {},
            "review_feedback": {},
            "paused_issues": {},
            "paused_tracked_prs": {},
            "backlog_items": {},
            "tracked_pull_requests": {},
            "runs": {},
            "task_queue": {
                "issue-task:1": {
                    "task_id": "issue-task:1",
                    "task_type": "issue_execution",
                    "status": "dead_letter",
                    "attempt_count": 1,
                    "max_attempts": 1,
                }
            },
            "observability": {"counters": {}, "cycles": {}, "recent_events": []},
        }

        result = validate_state_payload(payload)

        self.assertTrue(result["ok"])
        self.assertEqual(1, result["summary"]["dead_letter_tasks"])
        self.assertTrue(any("dead_letter" in item for item in result["warnings"]))

    def test_export_and_restore_snapshot_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            snapshot_path = Path(tmpdir) / "snapshot.json"
            store = AutoCodeStateStore(str(state_path))
            store.increment_counter("webhook.received")
            store.enqueue_task(
                task_id="issue-task:round-trip",
                task_type="issue_execution",
                target_number=88,
                source="webhook",
            )

            export_result = export_snapshot_file(str(state_path), str(snapshot_path))
            self.assertTrue(snapshot_path.exists())
            snapshot = export_result["snapshot"]
            snapshot["state"]["observability"]["counters"]["webhook.received"] = 9
            snapshot["state"]["task_queue"]["issue-task:round-trip"]["status"] = "running"
            snapshot["state"]["task_queue"]["issue-task:round-trip"]["lease_owner"] = "worker-x"
            snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

            restore_result = restore_snapshot_file(str(state_path), str(snapshot_path))

            restored_store = AutoCodeStateStore(str(state_path))
            self.assertEqual(9, restored_store.observability_summary()["counters"]["webhook.received"])
            self.assertEqual("queued", restored_store.get_task("issue-task:round-trip")["status"])
            self.assertEqual(1, restore_result["recovered_running_tasks"])
            self.assertTrue(Path(restore_result["backup_path"]).exists())

    def test_validate_state_file_returns_non_zero_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "broken.json"
            state_path.write_text("{broken", encoding="utf-8")

            result = validate_state_file(str(state_path))

            self.assertFalse(result["ok"])
            self.assertTrue(any("invalid json" in item for item in result["errors"]))

    def test_cli_validate_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = AutoCodeStateStore(str(state_path))
            store.increment_counter("webhook.received")
            script_path = Path(__file__).resolve().parent.parent / "tools" / "state_admin.py"

            completed = subprocess.run(
                [sys.executable, str(script_path), "validate", "--state-file", str(state_path), "--json"],
                cwd=str(Path(__file__).resolve().parent.parent),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(1, payload["summary"]["observability_counters"])


if __name__ == "__main__":
    unittest.main()
