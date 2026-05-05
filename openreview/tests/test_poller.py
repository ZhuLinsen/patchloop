import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import agent.poller as poller_module
import agent.state_store as state_store_module
from agent.poller import GitHubPoller
from agent.state_store import LocalStateStore


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 3, 26, tzinfo=timezone.utc)
        return value if tz is None else value.astimezone(tz)


class _StubGitHubClient:
    def __init__(
        self,
        *,
        issues=None,
        prs=None,
        pr_details=None,
        pr_reviews=None,
        repo_issue_comments=None,
        repo_review_comments=None,
        issue_details=None,
        ci_status=None,
    ):
        self.issues = issues or []
        self.prs = prs or []
        self.pr_details = pr_details or {}
        self.pr_reviews = pr_reviews or {}
        self.repo_issue_comments = repo_issue_comments or []
        self.repo_review_comments = repo_review_comments or []
        self.issue_details = issue_details or {}
        self.ci_status = ci_status or {"state": "success", "ready": True, "details": []}
        self.issue_list_calls = []
        self.pr_review_calls = []

    def list_issues(self, **kwargs):
        self.issue_list_calls.append(kwargs)
        return self.issues if kwargs.get("page", 1) == 1 else []

    def list_pull_requests(self, **kwargs):
        return self.prs if kwargs.get("page", 1) == 1 else []

    def list_repository_issue_comments(self, **kwargs):
        return self.repo_issue_comments if kwargs.get("page", 1) == 1 else []

    def list_repository_pr_review_comments(self, **kwargs):
        return self.repo_review_comments if kwargs.get("page", 1) == 1 else []

    def list_pr_reviews(self, pr_number: int, **kwargs):
        self.pr_review_calls.append((pr_number, kwargs))
        reviews = self.pr_reviews.get(pr_number, [])
        return reviews if kwargs.get("page", 1) == 1 else []

    def get_issue(self, issue_number: int):
        return self.issue_details[issue_number]

    def get_pr(self, pr_number: int):
        if pr_number in self.pr_details:
            return self.pr_details[pr_number]
        for pr in self.prs:
            if pr.get("number") == pr_number:
                return pr
        raise KeyError(pr_number)

    def get_commit_ci_status(self, _ref: str):
        return self.ci_status


class PollerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        to_thread_patcher = patch.object(poller_module.asyncio, "to_thread", new=_run_inline)
        datetime_patcher = patch.object(state_store_module, "datetime", _FrozenDateTime)
        to_thread_patcher.start()
        datetime_patcher.start()
        self.addCleanup(to_thread_patcher.stop)
        self.addCleanup(datetime_patcher.stop)

    def _make_store(self, tmpdir: str, *, baseline: str) -> LocalStateStore:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "polling": {
                        "bootstrapped_at": baseline,
                        "issue_created_after": baseline,
                        "pr_created_after": baseline,
                        "issue_comment_updated_after": baseline,
                        "review_comment_updated_after": baseline,
                    },
                    "thresholds": {
                        "issue_min_number": 603,
                        "pr_min_number": 603,
                    },
                    "issues": {},
                    "pull_requests": {},
                }
            ),
            encoding="utf-8",
        )
        return LocalStateStore(str(state_path))

    async def test_polling_issue_uses_updated_at_for_prebaseline_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                issues=[
                    {
                        "number": 900,
                        "state": "open",
                        "title": "old issue revived",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "labels": [],
                        "user": {"login": "alice", "type": "User"},
                    }
                ]
            )
            issue_calls = []

            async def issue_processor(issue, source, **kwargs):
                issue_calls.append((issue["number"], source, kwargs))
                return True

            async def pr_processor(_pr, _source, _ci_status=None, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual([(900, "polling", {})], issue_calls)
            self.assertEqual("2026-03-25T09:59:55Z", store.issue_cursor())
            self.assertEqual("2026-03-25T00:00:00Z", gh.issue_list_calls[0]["since"])

    async def test_polling_issue_comments_trigger_issue_followup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                repo_issue_comments=[
                    {
                        "id": 101,
                        "body": "补充复现步骤",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "issue_url": "https://api.github.com/repos/demo/repo/issues/910",
                        "user": {"login": "alice", "type": "User"},
                    }
                ],
                issue_details={
                    910: {
                        "number": 910,
                        "state": "open",
                        "title": "old issue",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "labels": [],
                        "user": {"login": "alice", "type": "User"},
                    }
                },
            )
            issue_calls = []

            async def issue_processor(issue, source, **kwargs):
                issue_calls.append((issue["number"], source, kwargs))
                return True

            async def pr_processor(_pr, _source, _ci_status=None, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual(1, len(issue_calls))
            self.assertEqual(910, issue_calls[0][0])
            self.assertEqual("polling", issue_calls[0][1])
            self.assertIn("activity_fingerprint", issue_calls[0][2])
            self.assertFalse(issue_calls[0][2]["allow_unanswerable_reply"])

    async def test_polling_issue_comments_from_self_login_without_marker_still_trigger_followup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                repo_issue_comments=[
                    {
                        "id": 103,
                        "body": "我补充了新的复现路径",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "issue_url": "https://api.github.com/repos/demo/repo/issues/913",
                        "user": {"login": "zhulinsen", "type": "User"},
                    }
                ],
                issue_details={
                    913: {
                        "number": 913,
                        "state": "open",
                        "title": "self authored followup",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "labels": [],
                        "user": {"login": "alice", "type": "User"},
                    }
                },
            )
            issue_calls = []

            async def issue_processor(issue, source, **kwargs):
                issue_calls.append((issue["number"], source, kwargs))
                return True

            async def pr_processor(_pr, _source, _ci_status=None, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
                ignored_logins={"zhulinsen"},
            )

            await poller.poll_once()

            self.assertEqual(1, len(issue_calls))
            self.assertEqual(913, issue_calls[0][0])

    async def test_polling_issue_comments_ignore_autocode_slash_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                repo_issue_comments=[
                    {
                        "id": 102,
                        "body": "/implement",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "issue_url": "https://api.github.com/repos/demo/repo/issues/912",
                        "user": {"login": "alice", "type": "User"},
                    }
                ]
            )
            issue_calls = []
            pr_calls = []

            async def issue_processor(issue, source, **kwargs):
                issue_calls.append((issue["number"], source, kwargs))
                return True

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual([], issue_calls)
            self.assertEqual([], pr_calls)

    async def test_polling_issue_comment_failure_rewinds_from_earliest_failed_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                repo_issue_comments=[
                    {
                        "id": 111,
                        "body": "后面的评论成功处理",
                        "created_at": "2026-03-25T10:05:00Z",
                        "updated_at": "2026-03-25T10:05:00Z",
                        "issue_url": "https://api.github.com/repos/demo/repo/issues/911",
                        "user": {"login": "alice", "type": "User"},
                    },
                    {
                        "id": 112,
                        "body": "前面的评论这轮失败",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "issue_url": "https://api.github.com/repos/demo/repo/issues/910",
                        "user": {"login": "bob", "type": "User"},
                    },
                ],
                issue_details={
                    910: {
                        "number": 910,
                        "state": "open",
                        "title": "old issue 910",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "labels": [],
                        "user": {"login": "bob", "type": "User"},
                    },
                    911: {
                        "number": 911,
                        "state": "open",
                        "title": "old issue 911",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:05:00Z",
                        "labels": [],
                        "user": {"login": "alice", "type": "User"},
                    },
                },
            )

            async def issue_processor(issue, _source, **_kwargs):
                if issue["number"] == 910:
                    raise RuntimeError("temporary failure")
                return True

            async def pr_processor(_pr, _source, _ci_status=None, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual("2026-03-25T09:59:30Z", store.issue_comment_cursor())

    async def test_polling_review_comments_trigger_pr_followup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                repo_review_comments=[
                    {
                        "id": 202,
                        "body": "这个点我已经修了",
                        "created_at": "2026-03-25T11:00:00Z",
                        "updated_at": "2026-03-25T11:00:00Z",
                        "pull_request_url": "https://api.github.com/repos/demo/repo/pulls/920",
                        "user": {"login": "bob", "type": "User"},
                    }
                ]
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual(1, len(pr_calls))
            self.assertEqual(920, pr_calls[0][0])
            self.assertEqual("polling", pr_calls[0][1])
            self.assertIsNone(pr_calls[0][2])
            self.assertIn("activity_fingerprint", pr_calls[0][3])

    async def test_polling_issue_failure_rewinds_cursor_more_aggressively(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                issues=[
                    {
                        "number": 930,
                        "state": "open",
                        "title": "issue fails once",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "labels": [],
                        "user": {"login": "alice", "type": "User"},
                    }
                ]
            )

            async def issue_processor(_issue, _source, **_kwargs):
                raise RuntimeError("temporary failure")

            async def pr_processor(_pr, _source, _ci_status=None, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual("2026-03-25T09:59:30Z", store.issue_cursor())

    async def test_polling_still_reviews_open_pr_when_ci_is_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            gh = _StubGitHubClient(
                prs=[
                    {
                        "number": 935,
                        "state": "open",
                        "title": "pending ci pr",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "head": {"sha": "sha-935"},
                    }
                ],
                ci_status={
                    "state": "pending",
                    "ready": False,
                    "has_checks": True,
                    "details": [{"name": "unit", "status": "in_progress", "conclusion": ""}],
                },
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertEqual(1, len(pr_calls))
            self.assertEqual(935, pr_calls[0][0])
            self.assertEqual("pending", pr_calls[0][2]["state"])

    async def test_polling_revisits_tracked_pending_pr_even_when_list_scan_no_longer_covers_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            store.mark_pr_seen(
                940,
                head_sha="sha-940",
                ci_state="pending",
                created_at="2026-03-20T00:00:00Z",
                updated_at="2026-03-20T10:00:00Z",
            )
            store.advance_pr_cursor("2026-03-25T12:00:00Z", overlap_seconds=0)

            gh = _StubGitHubClient(
                prs=[
                    {
                        "number": 941,
                        "state": "open",
                        "title": "new unrelated pr",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-20T11:00:00Z",
                        "head": {"sha": "sha-941"},
                    }
                ],
                pr_details={
                    940: {
                        "number": 940,
                        "state": "open",
                        "title": "tracked pending pr",
                        "body": "body",
                        "created_at": "2026-03-20T00:00:00Z",
                        "updated_at": "2026-03-20T10:00:00Z",
                        "head": {"sha": "sha-940"},
                    }
                },
                ci_status={"state": "success", "ready": True, "details": []},
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            self.assertTrue(any(call[0] == 940 for call in pr_calls))

    async def test_polling_review_submissions_trigger_followup_for_recent_tracked_pr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                store.mark_pr_seen(
                    950,
                    head_sha="sha-950",
                    ci_state="success",
                    created_at="2026-03-20T00:00:00Z",
                    updated_at="2026-03-25T10:00:00Z",
                )
                store.mark_pr_processed(
                    950,
                    review_key="sha-950:initial",
                    head_sha="sha-950",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            gh = _StubGitHubClient(
                pr_reviews={
                    950: [
                        {
                            "id": 303,
                            "body": "这个问题已经修复，可以重新看一下",
                            "submitted_at": "2026-03-25T11:00:00Z",
                            "user": {"login": "carol", "type": "User"},
                        }
                    ]
                }
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            matched = [call for call in pr_calls if call[0] == 950]
            self.assertEqual(1, len(matched))
            self.assertIsNone(matched[0][2])
            self.assertIn("activity_fingerprint", matched[0][3])

    async def test_review_submission_scan_uses_configured_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                for pr_number in range(1, 4):
                    store.mark_pr_seen(
                        pr_number,
                        head_sha=f"sha-{pr_number}",
                        ci_state="success",
                        created_at="2026-03-20T00:00:00Z",
                        updated_at="2026-03-25T10:00:00Z",
                    )
                    store.mark_pr_processed(
                        pr_number,
                        review_key=f"sha-{pr_number}:initial",
                        head_sha=f"sha-{pr_number}",
                        source="test",
                        ci_state="success",
                        updated_at="2026-03-25T10:00:00Z",
                    )
            gh = _StubGitHubClient()

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(_pr, _source, **_kwargs):
                return False

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
                items_per_page=1,
                review_submission_batch_size=2,
                review_submission_max_age_days=3650,
            )

            await poller.poll_once()

            self.assertEqual(2, len(gh.pr_review_calls))

    async def test_polling_review_submissions_ignore_markered_self_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                store.mark_pr_seen(
                    954,
                    head_sha="sha-954",
                    ci_state="success",
                    created_at="2026-03-20T00:00:00Z",
                    updated_at="2026-03-25T10:00:00Z",
                )
                store.mark_pr_processed(
                    954,
                    review_key="sha-954:initial",
                    head_sha="sha-954",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            gh = _StubGitHubClient(
                pr_reviews={
                    954: [
                        {
                            "id": 307,
                            "body": "自动 review\n\n<!-- openreview:idempotency:pr-review:abc -->",
                            "submitted_at": "2026-03-25T11:00:00Z",
                            "user": {"login": "zhulinsen", "type": "User"},
                        }
                    ]
                }
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
                ignored_logins={"zhulinsen"},
            )

            await poller.poll_once()

            matched = [call for call in pr_calls if call[0] == 954]
            self.assertEqual([], matched)

    async def test_polling_review_submissions_ignore_reviews_older_than_last_processed_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                store.mark_pr_seen(
                    951,
                    head_sha="sha-951",
                    ci_state="success",
                    created_at="2026-03-20T00:00:00Z",
                    updated_at="2026-03-25T10:00:00Z",
                )
                store.mark_pr_processed(
                    951,
                    review_key="sha-951:initial",
                    head_sha="sha-951",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            gh = _StubGitHubClient(
                pr_reviews={
                    951: [
                        {
                            "id": 304,
                            "body": "这是 bot 初评之前就存在的 review",
                            "submitted_at": "2026-03-25T10:00:00Z",
                            "user": {"login": "dave", "type": "User"},
                        }
                    ]
                }
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            matched = [call for call in pr_calls if call[0] == 951]
            self.assertEqual([], matched)

    async def test_polling_review_submissions_use_updated_at_for_edited_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                store.mark_pr_seen(
                    952,
                    head_sha="sha-952",
                    ci_state="success",
                    created_at="2026-03-20T00:00:00Z",
                    updated_at="2026-03-25T10:00:00Z",
                )
                store.mark_pr_processed(
                    952,
                    review_key="sha-952:initial",
                    head_sha="sha-952",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            gh = _StubGitHubClient(
                pr_reviews={
                    952: [
                        {
                            "id": 305,
                            "body": "我编辑了 review，请重新处理",
                            "submitted_at": "2026-03-25T09:00:00Z",
                            "updated_at": "2026-03-25T11:00:00Z",
                            "user": {"login": "erin", "type": "User"},
                        }
                    ]
                }
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            matched = [call for call in pr_calls if call[0] == 952]
            self.assertEqual(1, len(matched))
            self.assertIsNone(matched[0][2])
            self.assertIn("activity_fingerprint", matched[0][3])

    async def test_polling_review_submissions_do_not_skip_unscanned_review_after_later_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir, baseline="2026-03-25T00:00:00Z")
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T10:30:00Z"):
                store.mark_pr_seen(
                    953,
                    head_sha="sha-953",
                    ci_state="success",
                    created_at="2026-03-20T00:00:00Z",
                    updated_at="2026-03-25T10:00:00Z",
                )
                store.mark_pr_processed(
                    953,
                    review_key="sha-953:initial",
                    head_sha="sha-953",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T10:00:00Z",
                )
            with patch("agent.state_store.utc_now_iso", return_value="2026-03-25T12:30:00Z"):
                store.mark_pr_processed(
                    953,
                    review_key="sha-953:followup",
                    head_sha="sha-953",
                    source="test",
                    ci_state="success",
                    updated_at="2026-03-25T12:00:00Z",
                )
            gh = _StubGitHubClient(
                pr_reviews={
                    953: [
                        {
                            "id": 306,
                            "body": "这条 review 在后续处理之前还没被 poll 到",
                            "submitted_at": "2026-03-25T11:00:00Z",
                            "user": {"login": "frank", "type": "User"},
                        }
                    ]
                }
            )
            pr_calls = []

            async def issue_processor(_issue, _source, **_kwargs):
                return False

            async def pr_processor(pr, source, ci_status=None, **kwargs):
                pr_calls.append((pr["number"], source, ci_status, kwargs))
                return True

            poller = GitHubPoller(
                gh=gh,
                state_store=store,
                issue_processor=issue_processor,
                pr_processor=pr_processor,
            )

            await poller.poll_once()

            matched = [call for call in pr_calls if call[0] == 953]
            self.assertEqual(1, len(matched))
            self.assertIsNone(matched[0][2])
            self.assertIn("activity_fingerprint", matched[0][3])


if __name__ == "__main__":
    unittest.main()
