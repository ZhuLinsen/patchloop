import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.state_store import (
    LocalStateStore,
    build_activity_fingerprint,
    build_issue_processing_key,
    build_pr_review_key,
    build_pr_review_metadata,
)


class StateStoreReviewKeyTests(unittest.TestCase):
    def test_review_key_changes_when_pr_body_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            first_key = build_pr_review_key("abc123", "Title", "Body v1")
            second_key = build_pr_review_key("abc123", "Title", "Body v2")
            first_metadata = build_pr_review_metadata("Title", "Body v1")
            second_metadata = build_pr_review_metadata("Title", "Body v2")

            self.assertEqual(first_key.rsplit(":", 1)[-1], first_metadata["digest"])
            self.assertNotEqual(first_metadata["body_hash"], second_metadata["body_hash"])
            self.assertNotEqual(first_metadata["digest"], second_metadata["digest"])

            self.assertEqual("claimed", store.try_claim_pr_review(714, first_key, head_sha="abc123"))
            store.release_pr_review(714, first_key, head_sha="abc123")

            store.mark_pr_processed(
                714,
                review_key=first_key,
                head_sha="abc123",
                source="test",
                ci_state="success",
                review_metadata=first_metadata,
            )

            self.assertTrue(store.is_pr_review_processed(714, first_key, head_sha="abc123"))
            self.assertFalse(store.is_pr_review_processed(714, second_key, head_sha="abc123"))
            self.assertEqual(first_metadata, store.pr_processed_review_metadata(714))
            self.assertEqual("claimed", store.try_claim_pr_review(714, second_key, head_sha="abc123"))

    def test_review_key_changes_when_base_ref_changes(self):
        first_key = build_pr_review_key("abc123", "Title", "Body v1", "main")
        second_key = build_pr_review_key("abc123", "Title", "Body v1", "release")

        self.assertNotEqual(first_key, second_key)

    def test_legacy_processed_entry_is_rechecked_after_pr_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            old_updated_at = "2026-03-17T00:00:00Z"
            new_updated_at = "2026-03-18T00:00:00Z"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {
                            "714": {
                                "last_processed_head_sha": "abc123",
                                "updated_at": old_updated_at,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            unchanged_key = build_pr_review_key("abc123", "Title", "Body v1", "main")
            self.assertTrue(
                store.is_pr_review_processed(
                    714,
                    unchanged_key,
                    head_sha="abc123",
                    updated_at=old_updated_at,
                )
            )

            store.mark_pr_seen(
                714,
                head_sha="abc123",
                ci_state="pending",
                updated_at=new_updated_at,
            )

            edited_key = build_pr_review_key("abc123", "Title", "Body v2", "main")
            self.assertFalse(
                store.is_pr_review_processed(
                    714,
                    edited_key,
                    head_sha="abc123",
                    updated_at=new_updated_at,
                )
            )
            self.assertEqual(
                "claimed",
                store.try_claim_pr_review(
                    714,
                    edited_key,
                    head_sha="abc123",
                    updated_at=new_updated_at,
                ),
            )

    def test_review_key_changes_when_discussion_changes(self):
        first_key = build_pr_review_key("abc123", "Title", "Body v1", "main")
        followup_key = build_pr_review_key(
            "abc123",
            "Title",
            "Body v1",
            "main",
            discussion_fingerprint=build_activity_fingerprint(
                "review_comment",
                99,
                "这个点已经处理了",
                "2026-03-25T10:00:00Z",
            ),
        )

        self.assertNotEqual(first_key, followup_key)

    def test_same_pr_version_can_fall_back_to_head_and_updated_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_processed(
                715,
                review_key="abc123:legacy-mismatch",
                head_sha="abc123",
                source="test",
                ci_state="success",
                updated_at="2026-03-18T00:00:00Z",
            )

            self.assertTrue(
                store.is_pr_same_version_processed(
                    715,
                    head_sha="abc123",
                    updated_at="2026-03-18T00:00:00Z",
                )
            )
            self.assertFalse(
                store.is_pr_same_version_processed(
                    715,
                    head_sha="abc123",
                    updated_at="2026-03-19T00:00:00Z",
                )
            )
            self.assertFalse(
                store.is_pr_same_version_processed(
                    715,
                    head_sha="def456",
                    updated_at="2026-03-18T00:00:00Z",
                )
            )

    def test_legacy_last_review_key_does_not_count_as_processing_key_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {
                            "759": {
                                "last_processed_at": "2026-03-18T12:46:02Z",
                                "last_processed_head_sha": "sha-759",
                                "last_processed_review_key": "sha-759:legacy-key",
                                "updated_at": "2026-03-18T13:19:22Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            self.assertFalse(store.has_pr_review_processing_keys(759))

    def test_same_pr_blocks_other_review_key_while_inflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            initial_key = build_pr_review_key("abc123", "Title", "Body v1", "main")
            followup_key = build_pr_review_key(
                "abc123",
                "Title",
                "Body v1",
                "main",
                discussion_fingerprint=build_activity_fingerprint(
                    "review_comment",
                    99,
                    "这个点已经处理了",
                    "2026-03-25T10:00:00Z",
                ),
            )

            self.assertEqual("claimed", store.try_claim_pr_review(714, initial_key, head_sha="abc123"))
            self.assertEqual("inflight", store.try_claim_pr_review(714, followup_key, head_sha="abc123"))


class StateStoreIssueKeyTests(unittest.TestCase):
    def test_issue_key_changes_when_followup_comment_changes(self):
        initial_key = build_issue_processing_key("Issue title", "Issue body")
        followup_key = build_issue_processing_key(
            "Issue title",
            "Issue body",
            build_activity_fingerprint(
                "issue_comment",
                123,
                "补充了复现条件",
                "2026-03-25T09:00:00Z",
            ),
        )

        self.assertNotEqual(initial_key, followup_key)

    def test_issue_followup_can_be_processed_after_initial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            initial_key = build_issue_processing_key("Issue title", "Issue body")
            followup_key = build_issue_processing_key(
                "Issue title",
                "Issue body",
                build_activity_fingerprint(
                    "issue_comment",
                    123,
                    "补充了复现条件",
                    "2026-03-25T09:00:00Z",
                ),
            )

            self.assertEqual("claimed", store.try_claim_issue_key(816, initial_key))
            store.release_issue(816, initial_key)

            store.mark_issue_processed(
                816,
                processing_key=initial_key,
                source="test",
                updated_at="2026-03-25T08:00:00Z",
            )

            self.assertTrue(store.is_issue_processed(816, initial_key))
            self.assertFalse(store.is_issue_processed(816, followup_key))
            self.assertEqual("claimed", store.try_claim_issue_key(816, followup_key))

    def test_issue_processed_alias_keys_cover_comment_triggered_initial_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            initial_key = build_issue_processing_key("Issue title", "Issue body")
            comment_key = build_issue_processing_key(
                "Issue title",
                "Issue body",
                build_activity_fingerprint(
                    "issue_comment",
                    456,
                    "补充评论",
                    "2026-03-25T10:00:00Z",
                ),
            )

            store.mark_issue_processed(
                817,
                processing_key=comment_key,
                extra_processing_keys=[initial_key],
                source="test",
                updated_at="2026-03-25T10:00:00Z",
            )

            self.assertTrue(store.is_issue_processed(817, initial_key))
            self.assertTrue(store.is_issue_processed(817, comment_key))

    def test_legacy_issue_processed_entry_falls_back_to_updated_at_without_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            updated_at = "2026-03-25T10:00:00Z"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {
                            "817": {
                                "created_at": "2026-03-25T09:00:00Z",
                                "processed_at": "2026-03-25T10:05:00Z",
                                "updated_at": updated_at,
                                "source": "polling",
                            }
                        },
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            processing_key = build_issue_processing_key("Issue title", "Issue body")

            self.assertTrue(store.is_issue_processed(817, processing_key, updated_at=updated_at))
            self.assertEqual(
                "processed",
                store.try_claim_issue_key(817, processing_key, updated_at=updated_at),
            )
            self.assertFalse(
                store.is_issue_processed(817, processing_key, updated_at="2026-03-25T11:00:00Z")
            )

    def test_pr_processed_alias_keys_cover_comment_triggered_initial_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            initial_key = build_pr_review_key("abc123", "Title", "Body", "main")
            comment_key = build_pr_review_key(
                "abc123",
                "Title",
                "Body",
                "main",
                discussion_fingerprint=build_activity_fingerprint(
                    "review_comment",
                    789,
                    "补充评审意见",
                    "2026-03-25T11:00:00Z",
                ),
            )

            store.mark_pr_processed(
                818,
                review_key=comment_key,
                extra_review_keys=[initial_key],
                head_sha="abc123",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T11:00:00Z",
            )

            self.assertTrue(store.is_pr_review_processed(818, initial_key, head_sha="abc123"))
            self.assertTrue(store.is_pr_review_processed(818, comment_key, head_sha="abc123"))

    def test_same_issue_blocks_other_key_while_inflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            initial_key = build_issue_processing_key("Issue title", "Issue body")
            followup_key = build_issue_processing_key(
                "Issue title",
                "Issue body",
                build_activity_fingerprint(
                    "issue_comment",
                    456,
                    "补充评论",
                    "2026-03-25T10:00:00Z",
                ),
            )

            self.assertEqual("claimed", store.try_claim_issue_key(817, initial_key))
            self.assertEqual("inflight", store.try_claim_issue_key(817, followup_key))


class StateStorePollingCursorTests(unittest.TestCase):
    def test_missing_updated_cursors_are_initialized_from_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {
                            "bootstrapped_at": "2026-03-20T00:00:00Z",
                            "issue_created_after": "2026-03-20T00:00:00Z",
                            "pr_created_after": "2026-03-21T00:00:00Z",
                        },
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            store.ensure_polling_baseline()

            self.assertEqual("2026-03-20T00:00:00Z", store.issue_cursor())
            self.assertEqual("2026-03-21T00:00:00Z", store.pr_cursor())
            self.assertEqual("2026-03-20T00:00:00Z", store.issue_comment_cursor())
            self.assertEqual("2026-03-21T00:00:00Z", store.review_comment_cursor())

    def test_legacy_state_seeds_new_cursors_from_latest_tracked_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {
                            "bootstrapped_at": "2026-03-11T15:11:57Z",
                            "issue_created_after": "2026-03-11T15:11:57Z",
                            "pr_created_after": "2026-03-11T15:11:57Z",
                        },
                        "thresholds": {},
                        "issues": {
                            "610": {
                                "created_at": "2026-03-11T15:58:43Z",
                                "processed_at": "2026-03-11T16:15:00Z",
                                "source": "polling",
                                "updated_at": "2026-03-26T07:32:40Z",
                            }
                        },
                        "pull_requests": {
                            "609": {
                                "created_at": "2026-03-11T15:33:12Z",
                                "last_processed_at": "2026-03-26T04:40:00Z",
                                "last_processed_head_sha": "8921c70d1c661a082fae5edbbe410af1920b4529",
                                "last_seen_at": "2026-03-26T04:41:00Z",
                                "last_seen_ci_state": "success",
                                "last_seen_head_sha": "8921c70d1c661a082fae5edbbe410af1920b4529",
                                "source": "polling",
                                "updated_at": "2026-03-26T04:27:04Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            self.assertEqual("2026-03-26T07:32:40Z", store.issue_cursor())
            self.assertEqual("2026-03-26T07:32:40Z", store.issue_comment_cursor())
            self.assertEqual("2026-03-26T04:27:04Z", store.pr_cursor())
            self.assertEqual("2026-03-26T04:27:04Z", store.review_comment_cursor())
            self.assertEqual("2026-03-26T04:27:04Z", store.pr_review_polling_cutoff(609))
            self.assertEqual(
                "2026-03-26T04:27:04Z",
                store._data["pull_requests"]["609"]["last_processed_updated_at"],
            )
            self.assertEqual(
                "2026-03-26T04:27:04Z",
                store._data["pull_requests"]["609"]["last_review_submission_cursor_at"],
            )
            self.assertEqual(
                "2026-03-26T04:41:00Z",
                store._data["pull_requests"]["609"]["last_review_scan_at"],
            )

    def test_advance_issue_cursor_rewinds_overlap_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "polling": {
                            "bootstrapped_at": "2026-03-25T00:00:00Z",
                            "issue_created_after": "2026-03-25T00:00:00Z",
                            "pr_created_after": "2026-03-25T00:00:00Z",
                            "issue_updated_after": "2026-03-25T00:00:00Z",
                            "pr_updated_after": "2026-03-25T00:00:00Z",
                            "issue_comment_updated_after": "2026-03-25T00:00:00Z",
                            "review_comment_updated_after": "2026-03-25T00:00:00Z",
                        },
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )
            store = LocalStateStore(str(state_path))

            store.advance_issue_cursor("2026-03-25T10:00:00Z", overlap_seconds=5)

            self.assertEqual("2026-03-25T09:59:55Z", store.issue_cursor())

    def test_pending_pr_numbers_for_polling_returns_unfinished_tracked_prs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_seen(
                820,
                head_sha="sha-pending",
                ci_state="pending",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_seen(
                821,
                head_sha="sha-success",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_processed(
                821,
                review_key="sha-success:key",
                head_sha="sha-success",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_seen(
                822,
                head_sha="sha-pending-reviewed",
                ci_state="pending",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_processed(
                822,
                review_key="sha-pending-reviewed:key",
                head_sha="sha-pending-reviewed",
                source="test",
                ci_state="pending",
                updated_at="2026-03-25T10:00:00Z",
            )

            self.assertEqual([820], store.pending_pr_numbers_for_polling(max_age_days=3650))

    def test_pending_pr_numbers_for_polling_skips_stale_backlog_prs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_seen(
                650,
                head_sha="sha-old",
                ci_state="pending",
                updated_at="2026-01-01T10:00:00Z",
            )

            self.assertEqual([], store.pending_pr_numbers_for_polling(max_age_days=7))

    def test_pr_numbers_for_review_polling_returns_recent_tracked_prs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_seen(
                830,
                head_sha="sha-830",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_processed(
                830,
                review_key="sha-830:key",
                head_sha="sha-830",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )

            self.assertEqual([830], store.pr_numbers_for_review_polling(limit=5, max_age_days=3650))

    def test_pr_review_polling_cutoff_uses_stable_review_cursor_until_scan_advances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_seen(
                831,
                head_sha="sha-831",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )
            store.mark_pr_processed(
                831,
                review_key="sha-831:initial",
                head_sha="sha-831",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T10:00:00Z",
            )

            self.assertEqual("2026-03-25T10:00:00Z", store.pr_review_polling_cutoff(831))

            store.mark_pr_processed(
                831,
                review_key="sha-831:followup",
                head_sha="sha-831",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T12:00:00Z",
            )

            self.assertEqual("2026-03-25T10:00:00Z", store.pr_review_polling_cutoff(831))

            store.mark_pr_review_polling_scanned(
                831,
                scanned_at="2026-03-25T12:30:00Z",
                latest_review_updated_at="2026-03-25T11:00:00Z",
            )

            self.assertEqual("2026-03-25T11:00:00Z", store.pr_review_polling_cutoff(831))

    def test_pr_review_polling_rotates_after_a_batch_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:00:00Z"):
                for pr_number in range(1, 41):
                    store.mark_pr_seen(
                        pr_number,
                        head_sha=f"sha-{pr_number}",
                        ci_state="success",
                        updated_at="2026-03-25T10:00:00Z",
                    )

            first_batch = store.pr_numbers_for_review_polling(limit=20, max_age_days=3650)
            self.assertEqual(list(range(40, 20, -1)), first_batch)

            for pr_number in first_batch:
                store.mark_pr_review_polling_scanned(pr_number, scanned_at="2026-03-25T12:00:00Z")

            second_batch = store.pr_numbers_for_review_polling(limit=20, max_age_days=3650)
            self.assertEqual(list(range(20, 0, -1)), second_batch)

    def test_pr_review_polling_skips_prs_already_scanned_since_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            store = LocalStateStore(str(state_path))

            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:00:00Z"):
                store.mark_pr_seen(
                    841,
                    head_sha="sha-841",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            self.assertEqual([841], store.pr_numbers_for_review_polling(limit=5, max_age_days=3650))

            store.mark_pr_review_polling_scanned(841, scanned_at="2026-03-25T10:05:00Z")
            self.assertEqual([], store.pr_numbers_for_review_polling(limit=5, max_age_days=3650))

            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:10:00Z"):
                store.mark_pr_seen(
                    841,
                    head_sha="sha-841",
                    ci_state="failure",
                    updated_at="2026-03-25T10:10:00Z",
                )
            self.assertEqual([841], store.pr_numbers_for_review_polling(limit=5, max_age_days=3650))

    def test_runtime_shadow_recovers_newer_pr_progress_when_primary_regresses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / ".openreview-state.json"
            runtime_path = Path(tmpdir) / ".openreview-state.runtime.json"

            primary_state = {
                "schema_version": 3,
                "polling": {
                    "bootstrapped_at": "2026-03-18T12:38:22Z",
                    "issue_created_after": "2026-03-18T12:38:22Z",
                    "pr_created_after": "2026-03-18T12:38:22Z",
                    "issue_updated_after": "2026-03-18T12:38:22Z",
                    "pr_updated_after": "2026-03-18T13:19:22Z",
                    "issue_comment_updated_after": "2026-03-18T12:38:22Z",
                    "review_comment_updated_after": "2026-03-18T13:19:22Z",
                },
                "thresholds": {},
                "issues": {},
                "pull_requests": {
                    "759": {
                        "created_at": "2026-03-18T12:38:22Z",
                        "last_processed_at": "2026-03-26T11:23:13Z",
                        "last_processed_head_sha": "7614e19601c5f92c424f6075192038843b763471",
                        "last_processed_review_key": "7614e19601c5f92c424f6075192038843b763471:fda682e7361874d8",
                        "last_processed_updated_at": "2026-03-18T13:19:22Z",
                        "last_review_scan_at": "2026-03-18T12:46:02Z",
                        "last_review_submission_cursor_at": "2026-03-18T12:39:41Z",
                        "last_seen_at": "2026-03-26T11:23:13Z",
                        "last_seen_ci_state": "success",
                        "last_seen_head_sha": "7614e19601c5f92c424f6075192038843b763471",
                        "processed_review_keys": [
                            "7614e19601c5f92c424f6075192038843b763471:d3121a47185f5d4e",
                            "7614e19601c5f92c424f6075192038843b763471:fda682e7361874d8",
                        ],
                        "source": "polling",
                        "updated_at": "2026-03-18T13:19:22Z",
                    }
                },
            }
            runtime_state = {
                "schema_version": 3,
                "polling": {
                    "bootstrapped_at": "2026-03-18T12:38:22Z",
                    "issue_created_after": "2026-03-18T12:38:22Z",
                    "pr_created_after": "2026-03-18T12:38:22Z",
                    "issue_updated_after": "2026-03-26T11:19:00Z",
                    "pr_updated_after": "2026-03-26T11:19:00Z",
                    "issue_comment_updated_after": "2026-03-26T11:19:00Z",
                    "review_comment_updated_after": "2026-03-26T11:19:00Z",
                },
                "thresholds": {},
                "issues": {},
                "pull_requests": {
                    "759": {
                        "created_at": "2026-03-18T12:38:22Z",
                        "last_processed_at": "2026-03-26T12:03:06Z",
                        "last_processed_head_sha": "7614e19601c5f92c424f6075192038843b763471",
                        "last_processed_review_key": "7614e19601c5f92c424f6075192038843b763471:6afd550d9800f88e",
                        "last_processed_updated_at": "2026-03-26T11:23:13Z",
                        "last_review_scan_at": "2026-03-26T12:00:37Z",
                        "last_review_submission_cursor_at": "2026-03-26T11:23:13Z",
                        "last_seen_at": "2026-03-26T12:03:06Z",
                        "last_seen_ci_state": "success",
                        "last_seen_head_sha": "7614e19601c5f92c424f6075192038843b763471",
                        "processed_review_keys": [
                            "7614e19601c5f92c424f6075192038843b763471:d3121a47185f5d4e",
                            "7614e19601c5f92c424f6075192038843b763471:6afd550d9800f88e",
                            "7614e19601c5f92c424f6075192038843b763471:fda682e7361874d8",
                        ],
                        "source": "polling",
                        "updated_at": "2026-03-26T11:23:13Z",
                    }
                },
            }
            state_path.write_text(json.dumps(primary_state), encoding="utf-8")
            runtime_path.write_text(json.dumps(runtime_state), encoding="utf-8")

            store = LocalStateStore(str(state_path))

            self.assertEqual("2026-03-26T11:23:13Z", store.pr_processed_updated_at(759))
            self.assertTrue(
                store.is_pr_same_version_processed(
                    759,
                    head_sha="7614e19601c5f92c424f6075192038843b763471",
                    updated_at="2026-03-26T11:23:13Z",
                )
            )
            self.assertEqual(
                "7614e19601c5f92c424f6075192038843b763471:6afd550d9800f88e",
                store._data["pull_requests"]["759"]["last_processed_review_key"],
            )
            self.assertEqual("2026-03-26T11:19:00Z", store.pr_cursor())

    def test_mark_pr_processed_updates_runtime_shadow_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / ".openreview-state.json"
            store = LocalStateStore(str(state_path))

            store.mark_pr_processed(
                759,
                review_key="sha-759:initial",
                head_sha="sha-759",
                source="test",
                ci_state="success",
                updated_at="2026-03-26T11:23:13Z",
            )

            runtime_path = Path(tmpdir) / ".openreview-state.runtime.json"
            self.assertTrue(runtime_path.exists())
            runtime_state = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "sha-759:initial",
                runtime_state["pull_requests"]["759"]["last_processed_review_key"],
            )


class HeadStabilizationTests(unittest.TestCase):
    """Head SHA debounce / stabilization tests."""

    def test_head_stable_when_pr_not_tracked(self):
        """未跟踪的 PR 不会进入 stabilization 检查（由 main.py has_processed_pr 守护）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            # 未跟踪的 PR last_seen_head_sha 为空，head 不匹配 → 返回 False
            self.assertFalse(store.is_pr_head_stable(900, "sha-abc", stabilize_seconds=300))

    def test_head_not_stable_when_just_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(901, head_sha="sha-new", ci_state="success", updated_at="2026-03-25T10:00:00Z")
            self.assertFalse(store.is_pr_head_stable(901, "sha-new", stabilize_seconds=300))

    def test_head_stable_after_elapsed_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(902, head_sha="sha-old", ci_state="success", updated_at="2026-03-25T10:00:00Z")
            past = "2026-03-25T09:50:00Z"
            with store._lock:
                store._data["pull_requests"]["902"]["head_sha_first_seen_at"] = past
                store._save_unlocked()
            self.assertTrue(store.is_pr_head_stable(902, "sha-old", stabilize_seconds=300))

    def test_head_not_stable_when_sha_differs_from_last_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(903, head_sha="sha-A", ci_state="success", updated_at="2026-03-25T10:00:00Z")
            self.assertFalse(store.is_pr_head_stable(903, "sha-B", stabilize_seconds=0))

    def test_mark_pr_seen_resets_first_seen_on_head_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(904, head_sha="sha-v1", ci_state="success", updated_at="2026-03-25T10:00:00Z")
            # 手动设置一个较早的 first_seen 时间
            with store._lock:
                store._data["pull_requests"]["904"]["head_sha_first_seen_at"] = "2026-03-25T09:00:00Z"
                store._save_unlocked()
            store.mark_pr_seen(904, head_sha="sha-v2", ci_state="success", updated_at="2026-03-25T10:05:00Z")
            new_ts = store._data["pull_requests"]["904"]["head_sha_first_seen_at"]
            # head 变化后 first_seen 应被重置为当前时间，不再是手动设置的旧值
            self.assertNotEqual("2026-03-25T09:00:00Z", new_ts)

    def test_mark_pr_seen_preserves_first_seen_when_head_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(905, head_sha="sha-stable", ci_state="pending", updated_at="2026-03-25T10:00:00Z")
            first_ts = store._data["pull_requests"]["905"]["head_sha_first_seen_at"]
            store.mark_pr_seen(905, head_sha="sha-stable", ci_state="success", updated_at="2026-03-25T10:10:00Z")
            second_ts = store._data["pull_requests"]["905"]["head_sha_first_seen_at"]
            self.assertEqual(first_ts, second_ts)

    def test_head_stable_with_zero_stabilize(self):
        """stabilize_seconds=0 时只要 head 和 last_seen 一致就视为稳定。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            store.mark_pr_seen(906, head_sha="sha-x", ci_state="success", updated_at="2026-03-25T10:00:00Z")
            self.assertTrue(store.is_pr_head_stable(906, "sha-x", stabilize_seconds=0))


if __name__ == "__main__":
    unittest.main()
