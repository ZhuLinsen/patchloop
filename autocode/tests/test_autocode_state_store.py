import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.autocode_state_store import (
    AutoCodeRunRecord,
    AutoCodeStateStore,
    build_feedback_item_fingerprint,
    build_feedback_key,
    build_issue_plan_key,
)


class AutoCodeStateStoreTests(unittest.TestCase):
    def test_polling_baseline_and_issue_comment_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))

            self.assertTrue(store.ensure_polling_baseline())
            self.assertTrue(store.issue_baseline())
            self.assertTrue(store.issue_comment_baseline())
            self.assertEqual((store.issue_baseline(), 0), store.issue_event_cursor())
            self.assertEqual((store.issue_comment_baseline(), 0), store.issue_comment_cursor())
            self.assertEqual("claimed", store.try_claim_issue_event(101))
            store.mark_issue_event_processed(
                event_id=101,
                issue_number=123,
                action="opened",
                created_at="2026-03-23T00:00:00Z",
                source="polling",
            )
            store.release_issue_event(101)
            self.assertTrue(store.has_processed_issue_event(101))
            self.assertEqual("processed", store.try_claim_issue_event(101))
            self.assertEqual("claimed", store.try_claim_issue_comment(9001, "2026-03-23T00:00:00Z"))
            store.mark_issue_comment_processed(
                comment_id=9001,
                issue_number=123,
                updated_at="2026-03-23T00:00:00Z",
                command="implement",
                source="polling",
            )
            store.release_issue_comment(9001)
            self.assertTrue(store.has_processed_issue_comment(9001))
            self.assertTrue(store.has_processed_issue_comment(9001, "2026-03-23T00:00:00Z"))
            self.assertFalse(store.has_processed_issue_comment(9001, "2026-03-23T00:05:00Z"))
            self.assertEqual("processed", store.try_claim_issue_comment(9001, "2026-03-23T00:00:00Z"))
            self.assertEqual("claimed", store.try_claim_issue_comment(9001, "2026-03-23T00:05:00Z"))
            store.release_issue_comment(9001)
            store.advance_issue_baseline("2026-03-24T00:00:00Z", 101)
            store.advance_issue_comment_baseline("2026-03-24T00:01:00Z", 9001)
            self.assertEqual(("2026-03-24T00:00:00Z", 101), store.issue_event_cursor())
            self.assertEqual(("2026-03-24T00:01:00Z", 9001), store.issue_comment_cursor())

    def test_issue_plan_key_and_claim_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_key = build_issue_plan_key(123, "2026-03-23T00:00:00Z", "bug_fix")

            self.assertEqual("claimed", store.try_claim_issue_plan(issue_key))
            store.mark_issue_plan(
                issue_key=issue_key,
                issue_number=123,
                task_type="bug_fix",
                updated_at="2026-03-23T00:00:00Z",
                summary="fix webhook",
            )
            store.release_issue_plan(issue_key)
            self.assertEqual("processed", store.try_claim_issue_plan(issue_key))
            store.save_issue_plan_snapshot(
                123,
                {
                    "issue_number": 123,
                    "title": "Fix webhook",
                    "body": "Need a stable plan",
                    "labels": ["bug"],
                    "updated_at": "2026-03-23T00:00:00Z",
                    "comment": "plan comment",
                    "triage": {"task_type": "bug_fix", "action": "auto_fix", "risk_level": "low"},
                    "plan": {"goal": "fix webhook", "acceptance_criteria": ["tests pass"]},
                    "issue_key": issue_key,
                },
            )
            snapshot = store.get_issue_plan_snapshot(123)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual("Fix webhook", snapshot["title"])
            self.assertEqual(1, store.summary()["frozen_issue_plans"])

    def test_processed_issue_version_history_keeps_recent_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))

            store.mark_issue_plan(
                issue_key="issue:123:old",
                issue_number=123,
                task_type="bug_fix",
                updated_at="2026-03-23T00:00:00Z",
                summary="first pass",
            )
            store.mark_issue_plan(
                issue_key="issue:123:new",
                issue_number=123,
                task_type="bug_fix",
                updated_at="2026-03-24T00:00:00Z",
                summary="second pass",
            )

            self.assertTrue(store.has_processed_issue_version(123, "2026-03-23T00:00:00Z"))
            self.assertTrue(store.has_processed_issue_version(123, "2026-03-24T00:00:00Z"))
            tracked_issue = store._data["issues"]["123"]
            self.assertEqual(
                ["issue:123:old", "issue:123:new"],
                tracked_issue["processed_issue_keys"],
            )
            self.assertEqual(
                ["2026-03-23T00:00:00Z", "2026-03-24T00:00:00Z"],
                tracked_issue["processed_updated_ats"],
            )

    def test_feedback_key_dedupes_processed_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            feedback_key = build_feedback_key(88, [3, 1], ["2026-03-23T00:00:00Z"])

            self.assertEqual("claimed", store.try_claim_feedback(feedback_key))
            store.mark_feedback_processed(feedback_key=feedback_key, pr_number=88, summary="updated branch")
            store.release_feedback(feedback_key)
            self.assertEqual("processed", store.try_claim_feedback(feedback_key))

    def test_feedback_key_distinguishes_edited_feedback_with_same_ids_and_timestamps(self):
        original = {
            "id": 100,
            "body": "please add tests",
            "state": "CHANGES_REQUESTED",
            "updated_at": "2026-03-24T01:00:00Z",
            "user": {"login": "reviewer"},
        }
        edited = {
            "id": 100,
            "body": "please add tests and docs",
            "state": "CHANGES_REQUESTED",
            "updated_at": "2026-03-24T01:00:00Z",
            "user": {"login": "reviewer"},
        }

        original_key = build_feedback_key(
            88,
            [100],
            ["2026-03-24T01:00:00Z"],
            item_fingerprints=[build_feedback_item_fingerprint(original)],
        )
        edited_key = build_feedback_key(
            88,
            [100],
            ["2026-03-24T01:00:00Z"],
            item_fingerprints=[build_feedback_item_fingerprint(edited)],
        )

        self.assertNotEqual(original_key, edited_key)

    def test_try_claim_feedback_accepts_legacy_processed_key_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            legacy_key = build_feedback_key(88, [100], ["2026-03-24T01:00:00Z"])
            new_key = build_feedback_key(
                88,
                [100],
                ["2026-03-24T01:00:00Z"],
                item_fingerprints=["review:100:newfingerprint"],
            )

            store.mark_feedback_processed(feedback_key=legacy_key, pr_number=88, summary="updated branch")

            self.assertEqual(
                "processed",
                store.try_claim_feedback(new_key, extra_feedback_keys=[legacy_key]),
            )
            self.assertEqual(1, store.summary()["processed_feedback_items"])

    def test_feedback_claim_owner_can_be_transferred_and_released_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            feedback_key = build_feedback_key(88, [100], ["2026-03-24T01:00:00Z"])

            self.assertEqual("claimed", store.try_claim_feedback(feedback_key, owner="task-a"))
            self.assertTrue(store.feedback_claim_owned_by(feedback_key, owner="task-a"))

            store.set_feedback_claim_owner(feedback_key, owner="task-b")

            self.assertFalse(store.feedback_claim_owned_by(feedback_key, owner="task-a"))
            self.assertTrue(store.feedback_claim_owned_by(feedback_key, owner="task-b"))

            store.release_feedback(feedback_key, owner="task-a")
            self.assertEqual("inflight", store.try_claim_feedback(feedback_key))

            store.release_feedback(feedback_key, owner="task-b")
            self.assertEqual("claimed", store.try_claim_feedback(feedback_key, owner="task-c"))

    def test_tracked_pr_queue_state_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.upsert_tracked_pr(
                pr_number=88,
                branch_name="feature/demo",
                base_ref="release",
                head_sha="abc123",
                author_login="mumu",
                pr_url="https://example.com/pr/88",
                queue_source="author_allowlist",
            )
            store.mark_tracked_pr_status(
                88,
                status="review_feedback_pending",
                summary="waiting for queue",
                review_state="changes_requested",
            )

            tracked_prs = store.list_tracked_prs()
            self.assertEqual(1, len(tracked_prs))
            self.assertEqual("review_feedback_pending", tracked_prs[0]["status"])
            store.update_tracked_pr_feedback_cursor(
                88,
                updated_at="2026-03-24T00:00:00Z",
                item_ids=[12, 13],
            )
            tracked_record = store.get_tracked_pr(88)
            self.assertIsNotNone(tracked_record)
            assert tracked_record is not None
            self.assertEqual("2026-03-24T00:00:00Z", tracked_record["last_feedback_updated_at"])
            self.assertEqual([12, 13], tracked_record["last_feedback_item_ids"])
            store.reset_tracked_pr_feedback_cursor(88)
            reset_record = store.get_tracked_pr(88)
            self.assertIsNotNone(reset_record)
            assert reset_record is not None
            self.assertNotIn("last_feedback_updated_at", reset_record)
            self.assertNotIn("last_feedback_item_ids", reset_record)
            self.assertEqual("review_feedback_pending", reset_record["status"])
            self.assertEqual(1, store.summary()["tracked_prs"])
            self.assertEqual(1, store.summary()["pending_tracked_prs"])

    def test_waiting_review_tracked_pr_does_not_count_as_pending_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.upsert_tracked_pr(
                pr_number=89,
                branch_name="autocode/issue-89-demo",
                base_ref="release",
                head_sha="sha-89",
                author_login="mumu",
                pr_url="https://example.com/pr/89",
                queue_source="autocode_branch",
            )
            store.mark_tracked_pr_status(
                89,
                status="waiting_review",
                summary="waiting for reviewer",
                review_state="commented",
            )

            self.assertEqual(1, store.summary()["tracked_prs"])
            self.assertEqual(0, store.summary()["pending_tracked_prs"])

    def test_run_records_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-1",
                    status="done",
                    kind="issue",
                    target_number=101,
                    started_at="2026-03-23T00:00:00Z",
                    finished_at="2026-03-23T00:01:00Z",
                )
            )

            self.assertEqual(1, store.summary()["runs"])

    def test_observability_records_counters_events_and_cycles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.increment_counter("webhook.received")
            for index in range(35):
                store.record_event(
                    category="webhook",
                    message=f"event {index}",
                    details={"index": index},
                )
            store.mark_cycle_started("polling-cycle")
            store.record_cycle_result("polling-cycle", status="succeeded", stats={"processed": 2})
            store.mark_cycle_started("idle-scan-cycle")
            store.record_cycle_result("idle-scan-cycle", status="failed", error="boom")

            summary = store.summary()
            observability = store.observability_summary()

            self.assertEqual(2, summary["observed_cycles"])
            self.assertEqual(30, summary["recent_events"])
            self.assertEqual(1, observability["counters"]["webhook.received"])
            self.assertEqual(30, len(observability["recent_events"]))
            self.assertEqual("event 5", observability["recent_events"][0]["message"])
            self.assertEqual("event 34", observability["recent_events"][-1]["message"])
            cycles = {item["name"]: item for item in observability["cycles"]}
            self.assertEqual("succeeded", cycles["polling-cycle"]["last_status"])
            self.assertEqual({"processed": 2}, cycles["polling-cycle"]["last_stats"])
            self.assertEqual("failed", cycles["idle-scan-cycle"]["last_status"])
            self.assertEqual("boom", cycles["idle-scan-cycle"]["last_error"])

    def test_export_and_restore_snapshot_recover_running_tasks_and_write_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "autocode-state.json"
            store = AutoCodeStateStore(str(state_path))
            store.ensure_polling_baseline()
            store.increment_counter("webhook.received")
            record, created = store.enqueue_task(
                task_id="issue-task:restore",
                task_type="issue_execution",
                target_number=99,
                source="webhook",
            )
            self.assertTrue(created)
            self.assertEqual("queued", record["status"])

            snapshot = store.export_snapshot()
            snapshot["state"]["task_queue"]["issue-task:restore"]["status"] = "running"
            snapshot["state"]["task_queue"]["issue-task:restore"]["lease_owner"] = "worker-x"
            snapshot["state"]["task_queue"]["issue-task:restore"]["lease_acquired_at"] = "2026-03-25T00:00:00Z"
            snapshot["state"]["observability"]["counters"]["state.restore"] = 3

            result = store.restore_snapshot(snapshot)

            self.assertTrue(result["backup_path"].endswith(".json"))
            self.assertTrue(Path(result["backup_path"]).exists())
            restored = store.get_task("issue-task:restore")
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual("queued", restored["status"])
            self.assertEqual("", restored["lease_owner"])
            self.assertEqual(1, result["recovered_running_tasks"])
            self.assertEqual(3, store.observability_summary()["counters"]["state.restore"])

    def test_task_queue_persists_claim_retry_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            record, created = store.enqueue_task(
                task_id="issue-task:1:a",
                task_type="issue_execution",
                target_number=1,
                source="webhook",
                command="implement",
                payload={"explicit_command": True},
                max_attempts=2,
            )
            self.assertTrue(created)
            self.assertEqual("queued", record["status"])

            duplicate, created = store.enqueue_task(
                task_id="issue-task:1:a",
                task_type="issue_execution",
                target_number=1,
                source="webhook",
            )
            self.assertFalse(created)
            self.assertEqual("issue-task:1:a", duplicate["task_id"])

            claimed = store.claim_next_task(task_type="issue_execution", owner="worker-1")
            self.assertIsNotNone(claimed)
            assert claimed is not None
            self.assertEqual("running", claimed["status"])
            self.assertEqual(1, claimed["attempt_count"])

            failed = store.mark_task_failed("issue-task:1:a", error="boom", retry_delay_seconds=1)
            self.assertIsNotNone(failed)
            assert failed is not None
            self.assertEqual("retry_waiting", failed["status"])

            retried = store.retry_task_queue_item("issue-task:1:a")
            self.assertEqual("queued", retried["status"])
            self.assertEqual(0, retried["attempt_count"])

            claimed_again = store.claim_next_task(task_type="issue_execution", owner="worker-2")
            assert claimed_again is not None
            dead = store.mark_task_failed("issue-task:1:a", error="boom-again", retry_delay_seconds=1)
            assert dead is not None
            self.assertEqual("retry_waiting", dead["status"])

            store._data["task_queue"]["issue-task:1:a"]["available_at"] = "2000-01-01T00:00:00Z"
            claimed_third = store.claim_next_task(task_type="issue_execution", owner="worker-3")
            assert claimed_third is not None
            dead = store.mark_task_failed("issue-task:1:a", error="boom-final", retry_delay_seconds=1)
            assert dead is not None
            self.assertEqual("dead_letter", dead["status"])

            recovered = store.retry_task_queue_item("issue-task:1:a")
            self.assertEqual("queued", recovered["status"])
            claimed_recovered = store.claim_next_task(task_type="issue_execution", owner="worker-4")
            assert claimed_recovered is not None
            self.assertEqual("running", claimed_recovered["status"])
            self.assertEqual(1, claimed_recovered["attempt_count"])

            second, created = store.enqueue_task(
                task_id="issue-task:2:b",
                task_type="issue_execution",
                target_number=2,
                source="polling",
            )
            self.assertTrue(created)
            self.assertEqual("issue-task:2:b", second["task_id"])
            running = store.claim_next_task(task_type="issue_execution", owner="worker-5")
            assert running is not None
            recovered_count = store.recover_running_tasks(task_type="issue_execution")
            self.assertEqual(2, recovered_count)
            summary = store.summary()
            self.assertEqual(2, summary["queued_tasks"])

    def test_count_active_tasks_filters_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.enqueue_task(
                task_id="issue-task:1:a",
                task_type="issue_execution",
                target_number=1,
                source="polling",
            )
            store.enqueue_task(
                task_id="pr-task:pr:100:a",
                task_type="pr_feedback_repair",
                target_number=100,
                source="pr_queue",
            )
            store.enqueue_task(
                task_id="source-task:x",
                task_type="source_issue_create",
                target_number=0,
                source="backlog",
            )
            # All active
            self.assertEqual(3, store.count_active_tasks())
            # Exclude PR
            self.assertEqual(2, store.count_active_tasks(exclude_types={"pr_feedback_repair"}))
            # Include only issue
            self.assertEqual(1, store.count_active_tasks(include_types={"issue_execution"}))
            # Include only PR
            self.assertEqual(1, store.count_active_tasks(include_types={"pr_feedback_repair"}))
            # Mark issue task as done — should not be counted
            store.claim_next_task(task_type="issue_execution", owner="w")
            store.mark_task_succeeded("issue-task:1:a")
            self.assertEqual(2, store.count_active_tasks())
            self.assertEqual(1, store.count_active_tasks(exclude_types={"pr_feedback_repair"}))

    def test_cleanup_completed_tasks_removes_old(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            # Create 3 done tasks with old timestamps
            for i in range(3):
                store.enqueue_task(
                    task_id=f"old-task:{i}",
                    task_type="issue_execution",
                    target_number=i + 1,
                    source="test",
                )
                store.claim_next_task(task_type="issue_execution", owner="w")
                store.mark_task_succeeded(f"old-task:{i}")
            # Manually set old finished_at
            with store._lock:
                for i in range(3):
                    store._data["task_queue"][f"old-task:{i}"]["finished_at"] = "2020-01-01T00:00:00Z"
                store._save_unlocked()
            # Create 1 recent done task
            store.enqueue_task(
                task_id="new-task:0",
                task_type="issue_execution",
                target_number=10,
                source="test",
            )
            store.claim_next_task(task_type="issue_execution", owner="w")
            store.mark_task_succeeded("new-task:0")
            # Cleanup with keep_recent=1 (keep the most recent done task)
            pruned = store.cleanup_completed_tasks(max_age_hours=1, keep_recent=1)
            self.assertEqual(3, pruned)
            # Only the recent task should remain
            self.assertIn("new-task:0", store._data["task_queue"])
            self.assertNotIn("old-task:0", store._data["task_queue"])

    def test_has_active_tasks_for_target(self):
        """has_active_tasks_for_target returns True only when matching active tasks exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            # No tasks → False
            self.assertFalse(store.has_active_tasks_for_target(42))
            # Add a queued task for PR 42
            store.enqueue_task(
                task_id="pr-task:42:a",
                task_type="pr_feedback_repair",
                target_number=42,
                source="test",
            )
            self.assertTrue(store.has_active_tasks_for_target(42))
            self.assertTrue(store.has_active_tasks_for_target(42, task_type="pr_feedback_repair"))
            self.assertFalse(store.has_active_tasks_for_target(42, task_type="issue_execution"))
            self.assertFalse(store.has_active_tasks_for_target(99))
            # Complete the task → False
            store.claim_next_task(task_type="pr_feedback_repair", owner="w")
            store.mark_task_succeeded("pr-task:42:a")
            self.assertFalse(store.has_active_tasks_for_target(42))

    def test_enqueue_task_replace_existing_upgrades_pending_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            original, created = store.enqueue_task(
                task_id="issue-task:auto",
                task_type="issue_execution",
                target_number=13,
                source="polling",
                command="issue_open",
                payload={"explicit_command": False},
            )
            self.assertTrue(created)
            self.assertEqual("issue_open", original["command"])

            upgraded, created = store.enqueue_task(
                task_id="issue-task:comment",
                task_type="issue_execution",
                target_number=13,
                source="comment",
                priority="urgent",
                command="implement",
                payload={"explicit_command": True},
                replace_existing=True,
            )

            self.assertFalse(created)
            self.assertEqual("issue-task:auto", upgraded["task_id"])
            self.assertEqual("implement", upgraded["command"])
            self.assertEqual({"explicit_command": True}, upgraded["payload"])
            self.assertEqual("urgent", upgraded["priority"])
            self.assertEqual("comment", upgraded["source"])

            claimed = store.claim_next_task(task_type="issue_execution", owner="worker-1")
            assert claimed is not None
            failed = store.mark_task_failed("issue-task:auto", error="boom", retry_delay_seconds=60)
            assert failed is not None
            self.assertEqual("retry_waiting", failed["status"])

            retried, created = store.enqueue_task(
                task_id="issue-task:comment:retry",
                task_type="issue_execution",
                target_number=13,
                source="comment",
                priority="urgent",
                command="retry",
                payload={"explicit_command": True},
                replace_existing=True,
            )

            self.assertFalse(created)
            self.assertEqual("queued", retried["status"])
            self.assertEqual(0, retried["attempt_count"])
            self.assertEqual("retry", retried["command"])
            self.assertEqual("urgent", retried["priority"])

    def test_enqueue_task_replace_existing_creates_follow_up_when_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.enqueue_task(
                task_id="issue-task:auto",
                task_type="issue_execution",
                target_number=42,
                source="polling",
                command="issue_open",
                payload={"explicit_command": False},
            )
            claimed = store.claim_next_task(task_type="issue_execution", owner="worker-1")
            self.assertIsNotNone(claimed)

            follow_up, created = store.enqueue_task(
                task_id="issue-task:manual",
                task_type="issue_execution",
                target_number=42,
                source="comment",
                priority="urgent",
                command="implement",
                payload={"explicit_command": True},
                replace_existing=True,
            )

            self.assertTrue(created)
            self.assertEqual("issue-task:manual", follow_up["task_id"])
            self.assertEqual("queued", follow_up["status"])
            self.assertEqual("implement", follow_up["command"])
            self.assertEqual({"explicit_command": True}, follow_up["payload"])

            tasks = store.list_tasks(task_type="issue_execution")
            self.assertEqual(2, len(tasks))
            statuses = {task["task_id"]: task["status"] for task in tasks}
            self.assertEqual("running", statuses["issue-task:auto"])
            self.assertEqual("queued", statuses["issue-task:manual"])

    def test_enqueue_task_replace_existing_requeues_exact_completed_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            record, created = store.enqueue_task(
                task_id="source-task:item-1",
                task_type="source_issue_create",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={"item_key": "backlog:1", "title": "[TODO] old"},
                max_attempts=2,
            )
            self.assertTrue(created)
            store.mark_task_succeeded("source-task:item-1", result_summary="dry_run")

            updated, created = store.enqueue_task(
                task_id="source-task:item-1",
                task_type="source_issue_create",
                target_number=0,
                source="backlog",
                priority="urgent",
                command="create_issue",
                payload={"item_key": "backlog:1", "title": "[TODO] new"},
                max_attempts=3,
                replace_existing=True,
            )

            self.assertFalse(created)
            self.assertEqual("queued", updated["status"])
            self.assertEqual(0, updated["attempt_count"])
            self.assertEqual("", updated["result_summary"])
            self.assertEqual({"item_key": "backlog:1", "title": "[TODO] new"}, updated["payload"])
            self.assertEqual("urgent", updated["priority"])

    def test_save_uses_unique_temp_file_per_store_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "autocode-state.json"
            primary = AutoCodeStateStore(str(state_path))
            secondary = AutoCodeStateStore(str(state_path))
            original_replace = os.replace
            injected = False

            def interleaved_replace(src, dst, *args, **kwargs):
                nonlocal injected
                src_path = Path(src)
                if not injected and src_path.parent == state_path.parent and src_path.name.endswith(".tmp"):
                    injected = True
                    secondary.ensure_polling_baseline()
                return original_replace(src, dst, *args, **kwargs)

            with mock.patch("agent.autocode_state_store.os.replace", side_effect=interleaved_replace):
                primary.ensure_polling_baseline()

            restored = AutoCodeStateStore(str(state_path))
            self.assertTrue(restored.issue_baseline())

    def test_has_backlog_item_can_ignore_dry_run_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.mark_backlog_item_synced(
                item_key="plan:dry-run",
                source_name="plan",
                source_path="PLAN.md",
                line_number=3,
                title="[Plan] dry run",
                issue_number=0,
                issue_url="",
                dry_run=True,
            )

            self.assertTrue(store.has_backlog_item("plan:dry-run"))
            self.assertFalse(store.has_backlog_item("plan:dry-run", require_real_issue=True))

    def test_pause_and_backlog_state_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            store.set_issue_paused(123, paused=True, reason="manual abort")
            store.set_tracked_pr_paused(88, paused=True, reason="pause pr")
            store.mark_backlog_item_synced(
                item_key="backlog:1",
                source_name="backlog",
                source_path="TODO.md",
                line_number=4,
                title="[Backlog] add tests",
                issue_title="[Bug] add tests",
                issue_number=501,
                issue_url="https://example.com/issues/501",
            )
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-2",
                    status="failed",
                    kind="issue",
                    target_number=123,
                    failure_reason="boom",
                    started_at="2026-03-23T00:00:00Z",
                    finished_at="2026-03-23T00:02:00Z",
                )
            )

            self.assertTrue(store.is_issue_paused(123))
            self.assertTrue(store.is_tracked_pr_paused(88))
            self.assertTrue(store.has_backlog_item("backlog:1"))
            latest_run = store.latest_run_for_target("issue", 123)
            self.assertIsNotNone(latest_run)
            assert latest_run is not None
            self.assertEqual("failed", latest_run["status"])
            backlog_record = store.get_backlog_item("backlog:1")
            self.assertIsNotNone(backlog_record)
            assert backlog_record is not None
            self.assertEqual("[Backlog] add tests", backlog_record["source_title"])
            self.assertEqual("[Bug] add tests", backlog_record["issue_title"])
            summary = store.summary()
            self.assertEqual(1, summary["paused_issues"])
            self.assertEqual(1, summary["paused_tracked_prs"])
            self.assertEqual(1, summary["backlog_items"])

    def test_list_failed_runs_for_target_returns_only_failed_with_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AutoCodeStateStore(str(Path(tmpdir) / ".autocode-state.json"))
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-ok", status="done", kind="issue", target_number=42,
                    started_at="2026-03-23T00:00:00Z", finished_at="2026-03-23T00:01:00Z",
                )
            )
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-fail-1", status="failed", kind="issue", target_number=42,
                    failure_reason="验证失败: lint error",
                    started_at="2026-03-23T00:02:00Z", finished_at="2026-03-23T00:03:00Z",
                )
            )
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-fail-2", status="failed", kind="issue", target_number=42,
                    failure_reason="补丁超出范围限制",
                    started_at="2026-03-23T00:04:00Z", finished_at="2026-03-23T00:05:00Z",
                )
            )
            store.mark_run(
                AutoCodeRunRecord(
                    run_id="run-fail-other", status="failed", kind="issue", target_number=99,
                    failure_reason="unrelated",
                    started_at="2026-03-23T00:06:00Z", finished_at="2026-03-23T00:07:00Z",
                )
            )

            results = store.list_failed_runs_for_target("issue", 42)

            self.assertEqual(2, len(results))
            self.assertEqual("验证失败: lint error", results[0]["failure_reason"])
            self.assertEqual("补丁超出范围限制", results[1]["failure_reason"])

    def test_recover_stale_runs_on_load(self):
        """loading state should mark running runs older than max_age_hours as failed."""
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "autocode-state.json"
            # Build state with a stale running record and a fresh running record
            stale_time = "2020-01-01T00:00:00Z"
            fresh_time = "2099-12-31T00:00:00Z"
            raw_state = {
                "runs": {
                    "stale-run-1": {
                        "run_id": "stale-run-1",
                        "status": "running",
                        "kind": "issue",
                        "target_number": 100,
                        "branch_name": "auto/100",
                        "started_at": stale_time,
                    },
                    "fresh-run-2": {
                        "run_id": "fresh-run-2",
                        "status": "running",
                        "kind": "review_feedback",
                        "target_number": 200,
                        "branch_name": "auto/200",
                        "started_at": fresh_time,
                    },
                    "done-run-3": {
                        "run_id": "done-run-3",
                        "status": "done",
                        "kind": "issue",
                        "target_number": 300,
                        "branch_name": "auto/300",
                        "started_at": stale_time,
                    },
                },
            }
            state_path.write_text(json.dumps(raw_state), encoding="utf-8")
            store = AutoCodeStateStore(str(state_path))

            # stale-run-1 should be marked as failed
            stale = store._data["runs"]["stale-run-1"]
            self.assertEqual("failed", stale["status"])
            self.assertEqual("orphaned_on_restart", stale["failure_reason"])
            self.assertTrue(stale.get("finished_at"))

            # fresh-run-2 should remain running
            fresh = store._data["runs"]["fresh-run-2"]
            self.assertEqual("running", fresh["status"])

            # done-run-3 should remain done
            done = store._data["runs"]["done-run-3"]
            self.assertEqual("done", done["status"])


if __name__ == "__main__":
    unittest.main()
