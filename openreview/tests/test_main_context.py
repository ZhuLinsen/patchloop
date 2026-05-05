import json
import re
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from agent.state_store import (
    LocalStateStore,
    build_activity_fingerprint,
    build_issue_processing_key,
    build_pr_review_key,
)


_IDEMPOTENCY_MARKER_RE = re.compile(r"\s*<!-- openreview:idempotency:[^>]+ -->\s*$")


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _patch_to_thread(testcase: unittest.TestCase):
    patcher = mock.patch.object(main.asyncio, "to_thread", new=_run_inline)
    patcher.start()
    testcase.addCleanup(patcher.stop)


def _strip_test_idempotency_marker(body: str) -> str:
    return _IDEMPOTENCY_MARKER_RE.sub("", body).rstrip()


class _IssueContextGitHubClient:
    def __init__(self, *, similar_issues=None, related_prs=None, pr_details=None):
        self.recent_issue_comment_calls = []
        self.search_queries = []
        self.similar_issues = similar_issues or []
        self.related_prs = related_prs or []
        self.pr_details = pr_details or {}

    def list_recent_issue_comments(self, issue_number: int, *, limit: int, per_page: int):
        self.recent_issue_comment_calls.append((issue_number, limit, per_page))
        return [
            {
                "id": 901,
                "body": "这是最新的 issue 评论",
                "created_at": "2026-03-25T11:00:00Z",
                "updated_at": "2026-03-25T11:00:00Z",
                "user": {"login": "alice", "type": "User"},
            }
        ]

    def search_issues(self, query: str, *, per_page: int, page: int):
        self.search_queries.append((query, per_page, page))
        if "is:pr" in query:
            return list(self.related_prs)
        return list(self.similar_issues)

    def get_pr(self, pr_number: int):
        return self.pr_details[pr_number]

    def list_issue_comments(self, *_args, **_kwargs):
        raise AssertionError("should not read only the first issue comments page")


class _PRContextGitHubClient:
    def __init__(self):
        self.issue_comment_calls = []
        self.review_calls = []
        self.review_comment_calls = []

    def list_recent_issue_comments(self, pr_number: int, *, limit: int, per_page: int):
        self.issue_comment_calls.append((pr_number, limit, per_page))
        return [
            {
                "id": 1001,
                "body": "旧的会话评论",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T09:00:00Z",
                "user": {"login": "alice", "type": "User"},
            }
        ]

    def list_recent_pr_reviews(self, pr_number: int, *, limit: int, per_page: int):
        self.review_calls.append((pr_number, limit, per_page))
        return [
            {
                "id": 1002,
                "body": "这是最新的 review submission",
                "submitted_at": "2026-03-25T11:30:00Z",
                "user": {"login": "bob", "type": "User"},
            }
        ]

    def list_recent_pr_review_comments(self, pr_number: int, *, limit: int, per_page: int):
        self.review_comment_calls.append((pr_number, limit, per_page))
        return [
            {
                "id": 1003,
                "body": "较早的 inline comment",
                "created_at": "2026-03-25T10:00:00Z",
                "updated_at": "2026-03-25T10:00:00Z",
                "user": {"login": "carol", "type": "User"},
            }
        ]

    def list_issue_comments(self, *_args, **_kwargs):
        raise AssertionError("should not read only the first issue comments page")

    def list_pr_reviews(self, *_args, **_kwargs):
        raise AssertionError("should not read only the first reviews page")

    def list_pr_review_comments(self, *_args, **_kwargs):
        raise AssertionError("should not read only the first review comments page")


class _ProcessPRGitHubClient:
    def __init__(self, *, pr: dict, issue_comments=None, reviews=None, review_comments=None, ci_status=None):
        self.pr = pr
        self.issue_comments = issue_comments or []
        self.reviews = reviews or []
        self.review_comments = review_comments or []
        self.ci_status = ci_status or {"state": "success", "ready": True, "details": []}
        self.posted_reviews = []
        self.posted_comments = []

    def get_pr(self, pr_number: int):
        if pr_number != self.pr["number"]:
            raise KeyError(pr_number)
        return self.pr

    def get_commit_ci_status(self, _ref: str):
        return self.ci_status

    def post_pr_review_comment(self, pr_number: int, body: str):
        self.posted_reviews.append((pr_number, _strip_test_idempotency_marker(body)))

    def post_issue_comment(self, issue_number: int, body: str):
        self.posted_comments.append((issue_number, _strip_test_idempotency_marker(body)))

    def list_recent_issue_comments(self, pr_number: int, *, limit: int, per_page: int):
        assert pr_number == self.pr["number"]
        return list(self.issue_comments)

    def list_recent_pr_reviews(self, pr_number: int, *, limit: int, per_page: int):
        assert pr_number == self.pr["number"]
        return list(self.reviews)

    def list_recent_pr_review_comments(self, pr_number: int, *, limit: int, per_page: int):
        assert pr_number == self.pr["number"]
        return list(self.review_comments)


class _ProcessPRAgent:
    def __init__(self):
        self.handle_pr_calls = []
        self.handle_pr_followup_calls = []

    def handle_pr(
        self,
        title: str,
        body: str,
        base_ref: str,
        head_ref: str = "",
        head_repo_clone_url: str = "",
        head_sha: str = "",
        discussion_context: str = "",
        review_facts=None,
    ):
        self.handle_pr_calls.append(
            {
                "title": title,
                "body": body,
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "discussion_context": discussion_context,
                "review_facts": list(review_facts or []),
            }
        )
        return types.SimpleNamespace(reply="首轮评审", blocked_reason="")

    def handle_pr_followup(
        self,
        title: str,
        body: str,
        base_ref: str,
        discussion_context: str,
        head_ref: str = "",
        head_repo_clone_url: str = "",
        head_sha: str = "",
        review_facts=None,
    ):
        self.handle_pr_followup_calls.append(
            {
                "title": title,
                "body": body,
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "discussion_context": discussion_context,
                "review_facts": list(review_facts or []),
            }
        )
        return types.SimpleNamespace(reply="跟进评审", blocked_reason="")


class _FailingProcessPRAgent(_ProcessPRAgent):
    def handle_pr(
        self,
        title: str,
        body: str,
        base_ref: str,
        head_ref: str = "",
        head_repo_clone_url: str = "",
        head_sha: str = "",
        discussion_context: str = "",
        review_facts=None,
    ):
        super().handle_pr(
            title,
            body,
            base_ref,
            head_ref=head_ref,
            head_repo_clone_url=head_repo_clone_url,
            head_sha=head_sha,
            discussion_context=discussion_context,
            review_facts=review_facts,
        )
        raise RuntimeError("pr boom")


class _ProcessIssueGitHubClient:
    def __init__(
        self,
        *,
        issue: dict,
        issue_comments=None,
        similar_issues=None,
        related_prs=None,
        pr_details=None,
    ):
        self.issue = issue
        self.issue_comments = issue_comments or []
        self.similar_issues = similar_issues or []
        self.related_prs = related_prs or []
        self.pr_details = pr_details or {}
        self.posted_comments = []

    def list_recent_issue_comments(self, issue_number: int, *, limit: int, per_page: int):
        assert issue_number == self.issue["number"]
        return list(self.issue_comments)

    def search_issues(self, query: str, *, per_page: int, page: int):
        if "is:pr" in query:
            return list(self.related_prs)
        return list(self.similar_issues)

    def get_pr(self, pr_number: int):
        return self.pr_details[pr_number]

    def post_issue_comment(self, issue_number: int, body: str):
        self.posted_comments.append((issue_number, _strip_test_idempotency_marker(body)))


class _ProcessIssueAgent:
    def __init__(self):
        self.handle_issue_calls = []

    def handle_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
        *,
        triage_context: str = "",
        discussion_context: str = "",
        allow_unanswerable_reply: bool = True,
        author_association: str = "",
    ):
        self.handle_issue_calls.append(
            {
                "title": title,
                "body": body,
                "labels": labels,
                "triage_context": triage_context,
                "discussion_context": discussion_context,
                "allow_unanswerable_reply": allow_unanswerable_reply,
                "author_association": author_association,
            }
        )
        return "Issue 回复", types.SimpleNamespace(classification="ANSWERABLE")


class _FailingProcessIssueAgent(_ProcessIssueAgent):
    def handle_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
        *,
        triage_context: str = "",
        discussion_context: str = "",
        allow_unanswerable_reply: bool = True,
        author_association: str = "",
    ):
        super().handle_issue(
            title,
            body,
            labels,
            triage_context=triage_context,
            discussion_context=discussion_context,
            allow_unanswerable_reply=allow_unanswerable_reply,
            author_association=author_association,
        )
        raise RuntimeError("issue boom")


class MainDiscussionContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _patch_to_thread(self)
        self.original_gh = main.gh
        self.original_config = main.config
        self.original_self_login = main.self_login
        main.self_login = ""

    def tearDown(self):
        main.gh = self.original_gh
        main.config = self.original_config
        main.self_login = self.original_self_login

    async def test_issue_runtime_context_uses_recent_comment_loader(self):
        gh = _IssueContextGitHubClient()
        main.gh = gh
        main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

        (
            discussion_context,
            triage_context,
            latest_fingerprint,
            latest_activity_updated_at,
            latest_comment_updated_at,
        ) = await main._load_issue_runtime_context(
            816,
            "issue title",
            "issue body",
        )

        self.assertEqual([(816, 60, 100)], gh.recent_issue_comment_calls)
        self.assertIn("这是最新的 issue 评论", discussion_context)
        self.assertEqual("", triage_context)
        self.assertTrue(latest_fingerprint.startswith("issue_comment:901:"))
        self.assertEqual("2026-03-25T11:00:00Z", latest_activity_updated_at)
        self.assertEqual("2026-03-25T11:00:00Z", latest_comment_updated_at)

    async def test_issue_runtime_context_includes_related_prs_and_resolution_hints(self):
        gh = _IssueContextGitHubClient(
            related_prs=[
                {
                    "number": 77,
                    "state": "closed",
                    "title": "fix config loading",
                    "body": "Fixes #816",
                    "updated_at": "2026-03-25T12:00:00Z",
                }
            ],
            pr_details={
                77: {
                    "number": 77,
                    "state": "closed",
                    "title": "fix config loading",
                    "body": "Fixes #816",
                    "updated_at": "2026-03-25T12:00:00Z",
                    "merged_at": "2026-03-25T12:30:00Z",
                    "mergeable_state": "clean",
                    "merge_commit_sha": "abcdef1234567890",
                }
            },
        )
        main.gh = gh
        main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

        (
            _discussion_context,
            triage_context,
            _latest_fingerprint,
            _latest_activity_updated_at,
            _latest_comment_updated_at,
        ) = await main._load_issue_runtime_context(
            816,
            "config not loading",
            "这个配置项到底在哪里生效？",
        )

        self.assertIn("可能相关的 PR", triage_context)
        self.assertIn("PR #77 [merged]", triage_context)
        self.assertIn("merged_sha: abcdef1", triage_context)
        self.assertIn("修复状态线索", triage_context)
        self.assertIn("已合并", triage_context)
        self.assertIn("abcdef1", triage_context)

    async def test_issue_runtime_context_highlights_duplicate_issues_separately(self):
        gh = _IssueContextGitHubClient(
            similar_issues=[
                {
                    "number": 12,
                    "state": "closed",
                    "title": "login timeout after retry on release branch",
                    "body": "Same stack trace on release branch. Fixed by #20",
                    "updated_at": "2026-03-24T00:00:00Z",
                },
                {
                    "number": 19,
                    "state": "open",
                    "title": "config path question",
                    "body": "How to set LOCAL_REPO_PATH?",
                    "updated_at": "2026-03-25T00:00:00Z",
                },
            ],
        )
        main.gh = gh
        main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

        (
            _discussion_context,
            triage_context,
            _latest_fingerprint,
            _latest_activity_updated_at,
            _latest_comment_updated_at,
        ) = await main._load_issue_runtime_context(
            816,
            "login timeout after retry on release branch",
            "release branch 上也会出现 login timeout after retry，错误栈完全一致",
        )

        self.assertIn("疑似重复 Issue（高置信）", triage_context)
        self.assertIn("#12", triage_context)
        self.assertIn("相关 Issue / 使用讨论", triage_context)
        self.assertIn("#19", triage_context)

    async def test_pr_discussion_context_uses_recent_discussion_loaders(self):
        gh = _PRContextGitHubClient()
        main.gh = gh

        discussion_context, latest_fingerprint, latest_updated_at = await main._load_pr_discussion_context(900)

        self.assertEqual([(900, 60, 100)], gh.issue_comment_calls)
        self.assertEqual([(900, 60, 100)], gh.review_calls)
        self.assertEqual([(900, 60, 100)], gh.review_comment_calls)
        self.assertIn("这是最新的 review submission", discussion_context)
        self.assertTrue(latest_fingerprint.startswith("review:1002:"))
        self.assertEqual("2026-03-25T11:30:00Z", latest_updated_at)

    async def test_issue_marker_lookup_uses_wider_recent_comment_window(self):
        marker = "<!-- openreview:idempotency:issue:abc -->"

        class _MarkerIssueGitHubClient:
            def __init__(self):
                self.calls = []

            def list_recent_issue_comments(self, issue_number: int, *, limit: int, per_page: int):
                self.calls.append((issue_number, limit, per_page))
                return [{"body": f"已有回复\n\n{marker}"}]

        gh = _MarkerIssueGitHubClient()
        main.gh = gh

        exists = await main._issue_comment_marker_exists(901, marker)

        self.assertTrue(exists)
        self.assertEqual([(901, 200, 100)], gh.calls)

    async def test_pr_review_marker_lookup_uses_wider_recent_review_window(self):
        marker = "<!-- openreview:idempotency:pr-review:def -->"

        class _MarkerPRGitHubClient:
            def __init__(self):
                self.calls = []

            def list_recent_pr_reviews(self, pr_number: int, *, limit: int, per_page: int):
                self.calls.append((pr_number, limit, per_page))
                return [{"body": f"已有 review\n\n{marker}"}]

        gh = _MarkerPRGitHubClient()
        main.gh = gh

        exists = await main._pr_review_marker_exists(902, marker)

        self.assertTrue(exists)
        self.assertEqual([(902, 200, 100)], gh.calls)


class MainSingleInstanceLockTests(unittest.TestCase):
    def setUp(self):
        self.original_lock_handle = main._single_instance_lock_handle
        self.original_lock_path = main._single_instance_lock_path
        main._single_instance_lock_handle = None
        main._single_instance_lock_path = ""

    def tearDown(self):
        main._release_single_instance_lock()
        main._single_instance_lock_handle = self.original_lock_handle
        main._single_instance_lock_path = self.original_lock_path

    def test_acquire_single_instance_lock_records_owner_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")

            main._acquire_single_instance_lock(state_path)

            self.assertIsNotNone(main._single_instance_lock_handle)
            lock_path = main._single_instance_lockfile_path(state_path)
            self.assertEqual(str(lock_path), main._single_instance_lock_path)
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(state_path, payload["state_file"])

    def test_acquire_single_instance_lock_raises_when_lock_is_held(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "state.json")
            lock_path = main._single_instance_lockfile_path(state_path)
            lock_path.write_text('{"pid": 1234}', encoding="utf-8")

            with mock.patch.object(main.fcntl, "flock", side_effect=BlockingIOError):
                with self.assertRaisesRegex(RuntimeError, "已有 OpenReview 实例正在运行"):
                    main._acquire_single_instance_lock(state_path)

            self.assertIsNone(main._single_instance_lock_handle)

    def test_release_single_instance_lock_is_noop_without_handle(self):
        main._release_single_instance_lock()
        self.assertIsNone(main._single_instance_lock_handle)


class MainProcessPRTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _patch_to_thread(self)
        self.original_agent = main.agent
        self.original_gh = main.gh
        self.original_state_store = main.state_store
        self.original_self_login = main.self_login
        self.original_pr_locks = main._pr_processing_locks
        self.original_issue_locks = main._issue_processing_locks
        self.original_pr_lock_refs = main._pr_processing_lock_refs
        self.original_issue_lock_refs = main._issue_processing_lock_refs
        main.self_login = ""
        main._pr_processing_locks = {}
        main._issue_processing_locks = {}
        main._pr_processing_lock_refs = {}
        main._issue_processing_lock_refs = {}

    def tearDown(self):
        main.agent = self.original_agent
        main.gh = self.original_gh
        main.state_store = self.original_state_store
        main.self_login = self.original_self_login
        main._pr_processing_locks = self.original_pr_locks
        main._issue_processing_locks = self.original_issue_locks
        main._pr_processing_lock_refs = self.original_pr_lock_refs
        main._issue_processing_lock_refs = self.original_issue_lock_refs

    async def test_initial_review_includes_existing_discussion_and_marks_latest_discussion_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 920,
                "state": "open",
                "title": "Test PR",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-920", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            latest_review = {
                "id": 1002,
                "body": "这是最新的 review submission",
                "submitted_at": "2026-03-25T11:30:00Z",
                "user": {"login": "bob", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(pr=pr, reviews=[latest_review])
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 920}, "webhook")

            self.assertTrue(processed)
            self.assertEqual(1, len(main.agent.handle_pr_calls))
            self.assertEqual([], main.agent.handle_pr_followup_calls)
            self.assertIn("这是最新的 review submission", main.agent.handle_pr_calls[0]["discussion_context"])
            self.assertEqual([(920, "首轮评审")], main.gh.posted_reviews)

            initial_key = build_pr_review_key("sha-920", "Test PR", "Body", "main")
            latest_fingerprint = build_activity_fingerprint(
                "review",
                1002,
                "这是最新的 review submission",
                "2026-03-25T11:30:00Z",
            )
            latest_key = build_pr_review_key(
                "sha-920",
                "Test PR",
                "Body",
                "main",
                discussion_fingerprint=latest_fingerprint,
            )
            self.assertTrue(main.state_store.is_pr_review_processed(920, initial_key, head_sha="sha-920"))
            self.assertTrue(main.state_store.is_pr_review_processed(920, latest_key, head_sha="sha-920"))

    async def test_initial_review_tolerates_missing_head_repo_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 924,
                "state": "open",
                "title": "Test PR Missing Repo",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-924", "repo": None},
                "base": {"ref": "main"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(pr=pr)
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 924}, "webhook")

            self.assertTrue(processed)
            self.assertEqual([(924, "首轮评审")], main.gh.posted_reviews)
            self.assertEqual(1, len(main.agent.handle_pr_calls))

    async def test_initial_review_allows_pr_authored_by_self_login(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 925,
                "state": "open",
                "title": "Self Authored PR",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "zhulinsen", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-925", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.self_login = "zhulinsen"
            main.gh = _ProcessPRGitHubClient(pr=pr)
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 925}, "webhook")

            self.assertTrue(processed)
            self.assertEqual([(925, "首轮评审")], main.gh.posted_reviews)
            self.assertEqual(1, len(main.agent.handle_pr_calls))

    async def test_initial_review_is_skipped_when_ci_is_pending_without_discussion_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 926,
                "state": "open",
                "title": "Test PR Pending CI",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-926", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
                "mergeable": False,
                "mergeable_state": "dirty",
                "draft": False,
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                ci_status={
                    "state": "pending",
                    "ready": False,
                    "has_checks": True,
                    "details": [{"name": "unit", "status": "in_progress", "conclusion": ""}],
                },
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 926}, "webhook")

            self.assertFalse(processed)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertEqual([], main.agent.handle_pr_calls)

    async def test_initial_review_runs_when_ci_is_final_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 936,
                "state": "open",
                "title": "Test PR Failed CI",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-936", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
                "mergeable": True,
                "mergeable_state": "clean",
                "draft": False,
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                ci_status={
                    "state": "failure",
                    "ready": False,
                    "finalized": True,
                    "has_checks": True,
                    "details": [{"name": "backend-gate", "status": "completed", "conclusion": "failure"}],
                },
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 936}, "webhook")

            self.assertTrue(processed)
            self.assertEqual([(936, "首轮评审")], main.gh.posted_reviews)
            self.assertEqual(1, len(main.agent.handle_pr_calls))
            self.assertIn("当前 CI 状态：failure", main.agent.handle_pr_calls[0]["review_facts"])

    async def test_polled_processed_pr_follows_up_on_ci_failure_without_new_discussion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 937,
                "state": "open",
                "title": "Processed PR Failed CI",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-937", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
                "mergeable": True,
                "mergeable_state": "clean",
                "draft": False,
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.state_store.mark_pr_processed(
                937,
                review_key=build_pr_review_key("sha-937", "Processed PR Failed CI", "Body", "main"),
                head_sha="sha-937",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T11:30:00Z",
            )
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                ci_status={
                    "state": "failure",
                    "ready": False,
                    "finalized": True,
                    "has_checks": True,
                    "details": [{"name": "backend-gate", "status": "completed", "conclusion": "failure"}],
                },
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 937}, "polling")

            self.assertTrue(processed)
            self.assertEqual([(937, "跟进评审")], main.gh.posted_reviews)
            self.assertEqual(1, len(main.agent.handle_pr_followup_calls))
            self.assertIn("CI 状态变化触发复核", main.agent.handle_pr_followup_calls[0]["discussion_context"])

    async def test_initial_review_can_run_on_pending_ci_when_comment_provides_discussion_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 927,
                "state": "open",
                "title": "Test PR Pending CI Comment",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-927", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
                "mergeable": False,
                "mergeable_state": "dirty",
                "draft": False,
            }
            comment = {
                "id": 1101,
                "body": "这个 PR 能先帮忙看下设计吗？",
                "created_at": "2026-03-25T11:00:00Z",
                "updated_at": "2026-03-25T11:00:00Z",
                "user": {"login": "bob", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                issue_comments=[comment],
                ci_status={
                    "state": "pending",
                    "ready": False,
                    "has_checks": True,
                    "details": [{"name": "unit", "status": "in_progress", "conclusion": ""}],
                },
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr(
                {"number": 927},
                "webhook:issue_comment",
                activity_fingerprint=build_activity_fingerprint(
                    "issue_comment",
                    1101,
                    "这个 PR 能先帮忙看下设计吗？",
                    "2026-03-25T11:00:00Z",
                ),
            )

            self.assertTrue(processed)
            self.assertEqual([(927, "首轮评审")], main.gh.posted_reviews)
            self.assertEqual(1, len(main.agent.handle_pr_calls))
            self.assertIn("当前 CI 状态：pending", main.agent.handle_pr_calls[0]["review_facts"])
            self.assertIn("当前 merge 状态：存在冲突", "\n".join(main.agent.handle_pr_calls[0]["review_facts"]))

    async def test_initial_review_is_skipped_for_draft_pr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 928,
                "state": "open",
                "title": "Draft PR",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-928", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
                "draft": True,
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(pr=pr)
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 928}, "webhook")

            self.assertFalse(processed)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertEqual([], main.agent.handle_pr_calls)

    async def test_followup_marks_latest_loaded_discussion_key_to_avoid_duplicate_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 921,
                "state": "open",
                "title": "Test PR Followup",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-921", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            older_comment = {
                "id": 1001,
                "body": "旧的讨论点",
                "created_at": "2026-03-25T10:00:00Z",
                "updated_at": "2026-03-25T10:00:00Z",
                "user": {"login": "bob", "type": "User"},
            }
            latest_review_comment = {
                "id": 1003,
                "body": "更晚的补充说明",
                "created_at": "2026-03-25T11:00:00Z",
                "updated_at": "2026-03-25T11:00:00Z",
                "user": {"login": "carol", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            initial_key = build_pr_review_key("sha-921", "Test PR Followup", "Body", "main")
            main.state_store.mark_pr_processed(
                921,
                review_key=initial_key,
                head_sha="sha-921",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T11:30:00Z",
            )
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                issue_comments=[older_comment],
                review_comments=[latest_review_comment],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr(
                {"number": 921},
                "webhook:issue_comment",
                activity_fingerprint=build_activity_fingerprint(
                    "issue_comment",
                    1001,
                    "旧的讨论点",
                    "2026-03-25T10:00:00Z",
                ),
            )

            self.assertTrue(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual(1, len(main.agent.handle_pr_followup_calls))
            self.assertIn("更晚的补充说明", main.agent.handle_pr_followup_calls[0]["discussion_context"])
            self.assertEqual([(921, "跟进评审")], main.gh.posted_reviews)

            latest_fingerprint = build_activity_fingerprint(
                "review_comment",
                1003,
                "更晚的补充说明",
                "2026-03-25T11:00:00Z",
            )
            latest_key = build_pr_review_key(
                "sha-921",
                "Test PR Followup",
                "Body",
                "main",
                discussion_fingerprint=latest_fingerprint,
            )
            self.assertTrue(main.state_store.is_pr_review_processed(921, latest_key, head_sha="sha-921"))

    async def test_stale_pr_discussion_event_is_skipped_after_newer_discussion_was_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 922,
                "state": "open",
                "title": "Test PR Replay",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-922", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            older_comment_fingerprint = build_activity_fingerprint(
                "issue_comment",
                1001,
                "较早的讨论点",
                "2026-03-25T10:00:00Z",
            )
            latest_review_fingerprint = build_activity_fingerprint(
                "review",
                1004,
                "更新后的最终说明",
                "2026-03-25T11:30:00Z",
            )
            initial_key = build_pr_review_key("sha-922", "Test PR Replay", "Body", "main")
            latest_key = build_pr_review_key(
                "sha-922",
                "Test PR Replay",
                "Body",
                "main",
                discussion_fingerprint=latest_review_fingerprint,
            )

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.state_store.mark_pr_processed(
                922,
                review_key=latest_key,
                extra_review_keys=[initial_key],
                head_sha="sha-922",
                source="test",
                ci_state="success",
                updated_at="2026-03-25T11:30:00Z",
            )
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                issue_comments=[
                    {
                        "id": 1001,
                        "body": "较早的讨论点",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "user": {"login": "bob", "type": "User"},
                    }
                ],
                reviews=[
                    {
                        "id": 1004,
                        "body": "更新后的最终说明",
                        "submitted_at": "2026-03-25T11:30:00Z",
                        "user": {"login": "carol", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr(
                {"number": 922},
                "webhook:issue_comment",
                activity_fingerprint=older_comment_fingerprint,
            )

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual([], main.agent.handle_pr_followup_calls)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertEqual({}, main._pr_processing_locks)

    async def test_polled_pr_skips_rereview_when_same_version_was_already_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 807,
                "state": "open",
                "title": "Test PR Same Version",
                "body": "Body",
                "created_at": "2026-03-22T10:55:05Z",
                "updated_at": "2026-03-22T15:34:56Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-807", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "polling": {},
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {
                            "807": {
                                "created_at": "2026-03-22T10:55:05Z",
                                "last_processed_at": "2026-03-22T10:59:24Z",
                                "last_processed_head_sha": "sha-807",
                                "last_processed_review_key": "sha-807:legacy-mismatch",
                                "last_processed_updated_at": "2026-03-22T15:34:56Z",
                                "last_seen_at": "2026-03-22T10:59:24Z",
                                "last_seen_ci_state": "success",
                                "last_seen_head_sha": "sha-807",
                                "source": "polling",
                                "updated_at": "2026-03-22T15:34:56Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessPRGitHubClient(pr=pr)
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 807}, "polling")

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual([], main.agent.handle_pr_followup_calls)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertEqual({}, main._pr_processing_locks)

    async def test_polled_legacy_pr_skips_rereview_when_only_head_was_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 759,
                "state": "open",
                "title": "Legacy PR Head Only",
                "body": "Body",
                "created_at": "2026-03-18T12:38:22Z",
                "updated_at": "2026-03-18T13:19:22Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-759", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
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
                                "created_at": "2026-03-18T12:38:22Z",
                                "last_processed_at": "2026-03-18T12:46:02Z",
                                "last_processed_head_sha": "sha-759",
                                "last_seen_at": "2026-03-26T09:27:02Z",
                                "last_seen_ci_state": "success",
                                "last_seen_head_sha": "sha-759",
                                "source": "polling",
                                "updated_at": "2026-03-18T13:19:22Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                reviews=[
                    {
                        "id": 3201,
                        "body": "这条 review 在首轮处理之前就已经存在",
                        "submitted_at": "2026-03-18T12:39:41Z",
                        "user": {"login": "bob", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 759}, "polling")

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual([], main.agent.handle_pr_followup_calls)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertEqual({}, main._pr_processing_locks)

    async def test_polled_legacy_pr_follows_up_when_external_discussion_is_newer_than_last_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 760,
                "state": "open",
                "title": "Legacy PR With New Discussion",
                "body": "Body",
                "created_at": "2026-03-18T12:38:22Z",
                "updated_at": "2026-03-18T13:19:22Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-760", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {
                            "760": {
                                "created_at": "2026-03-18T12:38:22Z",
                                "last_processed_at": "2026-03-18T12:46:02Z",
                                "last_processed_head_sha": "sha-760",
                                "last_processed_review_key": "sha-760:legacy-initial",
                                "last_processed_updated_at": "2026-03-18T12:46:02Z",
                                "last_seen_at": "2026-03-26T09:27:02Z",
                                "last_seen_ci_state": "success",
                                "last_seen_head_sha": "sha-760",
                                "source": "polling",
                                "updated_at": "2026-03-18T13:19:22Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                reviews=[
                    {
                        "id": 3202,
                        "body": "这条 review 是后面新补充的讨论",
                        "submitted_at": "2026-03-18T12:50:00Z",
                        "user": {"login": "bob", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 760}, "polling")

            self.assertTrue(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual(1, len(main.agent.handle_pr_followup_calls))
            self.assertEqual([(760, "跟进评审")], main.gh.posted_reviews)

    async def test_polled_pr_uses_processed_updated_at_instead_of_local_processed_at_for_followup_cutoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 761,
                "state": "open",
                "title": "PR Discussion During Review",
                "body": "Body",
                "created_at": "2026-03-18T12:38:22Z",
                "updated_at": "2026-03-18T13:19:22Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-761", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "polling": {},
                        "thresholds": {},
                        "issues": {},
                        "pull_requests": {
                            "761": {
                                "created_at": "2026-03-18T12:38:22Z",
                                "last_processed_at": "2026-03-18T12:46:02Z",
                                "last_processed_head_sha": "sha-761",
                                "last_processed_review_key": "sha-761:initial",
                                "last_processed_updated_at": "2026-03-18T12:39:41Z",
                                "last_seen_at": "2026-03-18T12:46:02Z",
                                "last_seen_ci_state": "success",
                                "last_seen_head_sha": "sha-761",
                                "source": "polling",
                                "updated_at": "2026-03-18T13:19:22Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                reviews=[
                    {
                        "id": 3203,
                        "body": "这条 review 在上一轮分析进行期间才出现",
                        "submitted_at": "2026-03-18T12:45:00Z",
                        "user": {"login": "bob", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 761}, "polling")

            self.assertTrue(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual(1, len(main.agent.handle_pr_followup_calls))
            self.assertEqual([(761, "跟进评审")], main.gh.posted_reviews)

    async def test_existing_review_marker_reconciles_state_without_rereview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 930,
                "state": "open",
                "title": "Existing Review Marker",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-930", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }

            review_key = build_pr_review_key("sha-930", "Existing Review Marker", "Body", "main")
            marker = main._build_idempotency_marker("pr-review", review_key)

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                reviews=[
                    {
                        "id": 1201,
                        "body": f"首轮评审\n\n{marker}",
                        "submitted_at": "2026-03-25T11:30:00Z",
                        "user": {"login": "zhulinsen", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessPRAgent()

            processed = await main._process_pr({"number": 930}, "webhook")

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_pr_calls)
            self.assertEqual([], main.agent.handle_pr_followup_calls)
            self.assertEqual([], main.gh.posted_reviews)
            self.assertTrue(
                main.state_store.is_pr_review_processed(
                    930,
                    review_key,
                    head_sha="sha-930",
                    updated_at="2026-03-25T11:30:00Z",
                )
            )
            self.assertEqual({}, main._pr_processing_locks)

    async def test_existing_blocked_comment_marker_reconciles_state_without_renotify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 931,
                "state": "open",
                "title": "Blocked PR",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "user": {"login": "alice", "type": "User"},
                "head": {"ref": "feature/test", "sha": "sha-931", "repo": {"clone_url": "https://example.com/repo.git"}},
                "base": {"ref": "main"},
            }

            class _BlockedAgent(_ProcessPRAgent):
                def handle_pr(self, *args, **kwargs):
                    self.handle_pr_calls.append({"args": args, "kwargs": kwargs})
                    return types.SimpleNamespace(reply="", blocked_reason="sync failed")

            marker = main._build_idempotency_marker("pr-blocked", "sha-931\nsync failed")

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(
                pr=pr,
                issue_comments=[
                    {
                        "id": 1301,
                        "body": f"自动评审已跳过。\n\n{marker}",
                        "created_at": "2026-03-25T11:35:00Z",
                        "updated_at": "2026-03-25T11:35:00Z",
                        "user": {"login": "zhulinsen", "type": "User"},
                    }
                ],
            )
            main.agent = _BlockedAgent()

            processed = await main._process_pr({"number": 931}, "webhook")

            self.assertFalse(processed)
            self.assertEqual([], main.gh.posted_comments)
            self.assertTrue(bool(main.state_store._data["pull_requests"]["931"].get("last_blocked_at")))
            self.assertEqual({}, main._pr_processing_locks)


class MainProcessIssueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _patch_to_thread(self)
        self.original_agent = main.agent
        self.original_gh = main.gh
        self.original_state_store = main.state_store
        self.original_config = main.config
        self.original_self_login = main.self_login
        self.original_pr_locks = main._pr_processing_locks
        self.original_issue_locks = main._issue_processing_locks
        self.original_pr_lock_refs = main._pr_processing_lock_refs
        self.original_issue_lock_refs = main._issue_processing_lock_refs
        main.self_login = ""
        main._pr_processing_locks = {}
        main._issue_processing_locks = {}
        main._pr_processing_lock_refs = {}
        main._issue_processing_lock_refs = {}

    def tearDown(self):
        main.agent = self.original_agent
        main.gh = self.original_gh
        main.state_store = self.original_state_store
        main.config = self.original_config
        main.self_login = self.original_self_login
        main._pr_processing_locks = self.original_pr_locks
        main._issue_processing_locks = self.original_issue_locks
        main._pr_processing_lock_refs = self.original_pr_lock_refs
        main._issue_processing_lock_refs = self.original_issue_lock_refs

    async def test_stale_issue_discussion_event_is_skipped_after_newer_discussion_was_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 830,
                "state": "open",
                "title": "Issue Replay",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T11:30:00Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }
            older_comment_fingerprint = build_activity_fingerprint(
                "issue_comment",
                2001,
                "较早的 issue 讨论",
                "2026-03-25T10:00:00Z",
            )
            latest_comment_fingerprint = build_activity_fingerprint(
                "issue_comment",
                2002,
                "更新后的 issue 结论",
                "2026-03-25T11:30:00Z",
            )
            initial_key = build_issue_processing_key("Issue Replay", "Body")
            latest_key = build_issue_processing_key("Issue Replay", "Body", latest_comment_fingerprint)

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.state_store.mark_issue_processed(
                830,
                processing_key=latest_key,
                extra_processing_keys=[initial_key],
                source="test",
                updated_at="2026-03-25T11:30:00Z",
            )
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2001,
                        "body": "较早的 issue 讨论",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "user": {"login": "bob", "type": "User"},
                    },
                    {
                        "id": 2002,
                        "body": "更新后的 issue 结论",
                        "created_at": "2026-03-25T11:30:00Z",
                        "updated_at": "2026-03-25T11:30:00Z",
                        "user": {"login": "carol", "type": "User"},
                    },
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(
                issue,
                "webhook:issue_comment",
                activity_fingerprint=older_comment_fingerprint,
                allow_unanswerable_reply=False,
            )

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_issue_calls)
            self.assertEqual([], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)

    async def test_first_issue_processing_from_comment_still_allows_initial_reply_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 831,
                "state": "open",
                "title": "Issue First Comment",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T10:00:00Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2003,
                        "body": "我再补充一下现象",
                        "created_at": "2026-03-25T10:00:00Z",
                        "updated_at": "2026-03-25T10:00:00Z",
                        "user": {"login": "bob", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(
                issue,
                "webhook:issue_comment",
                activity_fingerprint=build_activity_fingerprint(
                    "issue_comment",
                    2003,
                    "我再补充一下现象",
                    "2026-03-25T10:00:00Z",
                ),
                allow_unanswerable_reply=False,
            )

            self.assertTrue(processed)
            self.assertEqual(1, len(main.agent.handle_issue_calls))
            self.assertTrue(main.agent.handle_issue_calls[0]["allow_unanswerable_reply"])
            self.assertEqual([(831, "Issue 回复")], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)

    async def test_issue_processing_exception_still_releases_processing_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 832,
                "state": "open",
                "title": "Issue Failure",
                "body": "Body",
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T10:00:00Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessIssueGitHubClient(issue=issue)
            main.agent = _FailingProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            with self.assertRaisesRegex(RuntimeError, "issue boom"):
                await main._process_issue(issue, "webhook")

            self.assertEqual({}, main._issue_processing_locks)

    async def test_legacy_processed_issue_skips_rereply_when_only_markered_self_comment_bumped_updated_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 866,
                "state": "open",
                "title": "Legacy Issue",
                "body": "Body",
                "created_at": "2026-03-26T08:14:15Z",
                "updated_at": "2026-03-26T08:20:41Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {
                            "866": {
                                "created_at": "2026-03-26T08:14:15Z",
                                "processed_at": "2026-03-26T08:20:42Z",
                                "updated_at": "2026-03-26T08:14:15Z",
                                "source": "polling",
                            }
                        },
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.self_login = "zhulinsen"
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2101,
                        "body": "最初的功能请求",
                        "created_at": "2026-03-26T08:14:15Z",
                        "updated_at": "2026-03-26T08:14:15Z",
                        "user": {"login": "alice", "type": "User"},
                    },
                    {
                        "id": 2102,
                        "body": "自动回复\n\n<!-- openreview:idempotency:issue:legacy -->",
                        "created_at": "2026-03-26T08:20:41Z",
                        "updated_at": "2026-03-26T08:20:41Z",
                        "user": {"login": "zhulinsen", "type": "User"},
                    },
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(issue, "polling")

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_issue_calls)
            self.assertEqual([], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)

    async def test_existing_issue_marker_reconciles_state_without_rereply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 867,
                "state": "open",
                "title": "Issue Marker",
                "body": "Body",
                "created_at": "2026-03-26T08:14:15Z",
                "updated_at": "2026-03-26T08:20:41Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }
            processing_key = build_issue_processing_key("Issue Marker", "Body")
            marker = main._build_idempotency_marker("issue", processing_key)

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.self_login = "zhulinsen"
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2201,
                        "body": "最初的问题描述",
                        "created_at": "2026-03-26T08:14:15Z",
                        "updated_at": "2026-03-26T08:14:15Z",
                        "user": {"login": "alice", "type": "User"},
                    },
                    {
                        "id": 2202,
                        "body": f"Issue 回复\n\n{marker}",
                        "created_at": "2026-03-26T08:20:41Z",
                        "updated_at": "2026-03-26T08:20:41Z",
                        "user": {"login": "zhulinsen", "type": "User"},
                    },
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(issue, "polling")

            self.assertFalse(processed)
            self.assertEqual([], main.agent.handle_issue_calls)
            self.assertEqual([], main.gh.posted_comments)
            self.assertTrue(main.state_store.is_issue_processed(867, processing_key, updated_at="2026-03-26T08:20:41Z"))
            self.assertEqual({}, main._issue_processing_locks)

    async def test_legacy_issue_uses_issue_updated_at_instead_of_local_processed_at_for_followup_cutoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 868,
                "state": "open",
                "title": "Legacy Issue Followup",
                "body": "Body",
                "created_at": "2026-03-26T08:14:15Z",
                "updated_at": "2026-03-26T08:20:41Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {
                            "868": {
                                "created_at": "2026-03-26T08:14:15Z",
                                "processed_at": "2026-03-26T08:20:42Z",
                                "updated_at": "2026-03-26T08:14:15Z",
                                "source": "polling",
                            }
                        },
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2301,
                        "body": "最初的问题描述",
                        "created_at": "2026-03-26T08:14:15Z",
                        "updated_at": "2026-03-26T08:14:15Z",
                        "user": {"login": "alice", "type": "User"},
                    },
                    {
                        "id": 2302,
                        "body": "这是分析进行期间新补充的评论",
                        "created_at": "2026-03-26T08:20:41Z",
                        "updated_at": "2026-03-26T08:20:41Z",
                        "user": {"login": "bob", "type": "User"},
                    },
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(issue, "polling")

            self.assertTrue(processed)
            self.assertEqual(1, len(main.agent.handle_issue_calls))
            self.assertEqual([(868, "Issue 回复")], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)

    async def test_legacy_issue_still_processes_when_body_changes_without_new_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 869,
                "state": "open",
                "title": "Legacy Issue Edited",
                "body": "Body updated by author",
                "created_at": "2026-03-26T08:14:15Z",
                "updated_at": "2026-03-26T08:30:00Z",
                "labels": [],
                "user": {"login": "alice", "type": "User"},
            }
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "polling": {},
                        "thresholds": {},
                        "issues": {
                            "869": {
                                "created_at": "2026-03-26T08:14:15Z",
                                "processed_at": "2026-03-26T08:20:42Z",
                                "updated_at": "2026-03-26T08:14:15Z",
                                "source": "polling",
                            }
                        },
                        "pull_requests": {},
                    }
                ),
                encoding="utf-8",
            )

            main.state_store = LocalStateStore(str(state_path))
            main.gh = _ProcessIssueGitHubClient(
                issue=issue,
                issue_comments=[
                    {
                        "id": 2401,
                        "body": "最初的问题描述",
                        "created_at": "2026-03-26T08:14:15Z",
                        "updated_at": "2026-03-26T08:14:15Z",
                        "user": {"login": "alice", "type": "User"},
                    }
                ],
            )
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(issue, "polling")

            self.assertTrue(processed)
            self.assertEqual(1, len(main.agent.handle_issue_calls))
            self.assertEqual([(869, "Issue 回复")], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)

    async def test_issue_authored_by_self_login_is_still_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            issue = {
                "number": 870,
                "state": "open",
                "title": "Self Authored Issue",
                "body": "Body",
                "created_at": "2026-03-26T08:14:15Z",
                "updated_at": "2026-03-26T08:14:15Z",
                "labels": [],
                "user": {"login": "zhulinsen", "type": "User"},
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.self_login = "zhulinsen"
            main.gh = _ProcessIssueGitHubClient(issue=issue)
            main.agent = _ProcessIssueAgent()
            main.config = types.SimpleNamespace(github=types.SimpleNamespace(repo="demo/repo"))

            processed = await main._process_issue(issue, "polling")

            self.assertTrue(processed)
            self.assertEqual(1, len(main.agent.handle_issue_calls))
            self.assertEqual([(870, "Issue 回复")], main.gh.posted_comments)
            self.assertEqual({}, main._issue_processing_locks)


class MainProcessPRExceptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _patch_to_thread(self)
        self.original_agent = main.agent
        self.original_gh = main.gh
        self.original_state_store = main.state_store
        self.original_config = main.config
        self.original_self_login = main.self_login
        self.original_pr_locks = main._pr_processing_locks
        self.original_issue_locks = main._issue_processing_locks
        self.original_pr_lock_refs = main._pr_processing_lock_refs
        self.original_issue_lock_refs = main._issue_processing_lock_refs
        main.self_login = ""
        main._pr_processing_locks = {}
        main._issue_processing_locks = {}
        main._pr_processing_lock_refs = {}
        main._issue_processing_lock_refs = {}

    def tearDown(self):
        main.agent = self.original_agent
        main.gh = self.original_gh
        main.state_store = self.original_state_store
        main.config = self.original_config
        main.self_login = self.original_self_login
        main._pr_processing_locks = self.original_pr_locks
        main._issue_processing_locks = self.original_issue_locks
        main._pr_processing_lock_refs = self.original_pr_lock_refs
        main._issue_processing_lock_refs = self.original_issue_lock_refs

    async def test_pr_processing_exception_still_releases_processing_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pr = {
                "number": 923,
                "state": "open",
                "draft": False,
                "mergeable": True,
                "title": "PR Failure",
                "body": "Body",
                "base": {"ref": "main"},
                "head": {
                    "ref": "feature/failure",
                    "sha": "abc123",
                    "repo": {"clone_url": "https://example.com/repo.git"},
                },
                "created_at": "2026-03-25T09:00:00Z",
                "updated_at": "2026-03-25T10:00:00Z",
            }

            main.state_store = LocalStateStore(str(Path(tmpdir) / "state.json"))
            main.gh = _ProcessPRGitHubClient(pr=pr)
            main.agent = _FailingProcessPRAgent()

            with self.assertRaisesRegex(RuntimeError, "pr boom"):
                await main._process_pr({"number": 923}, "webhook")

            self.assertEqual({}, main._pr_processing_locks)


class MainPRMergeabilitySummaryTests(unittest.TestCase):
    def test_blocked_state_is_not_described_as_conflict(self):
        summary = main._summarize_pr_mergeability(
            {"mergeable": False, "mergeable_state": "blocked"}
        )

        self.assertIn("mergeable_state=blocked", summary)
        self.assertIn("不等同于存在冲突", summary)
        self.assertNotIn("当前不能直接自动合并", summary)
        self.assertNotIn("常见于", summary)

    def test_dirty_state_is_still_described_as_conflict(self):
        summary = main._summarize_pr_mergeability(
            {"mergeable": False, "mergeable_state": "dirty"}
        )

        self.assertEqual("存在冲突，当前不能直接合并", summary)

    def test_non_blocked_false_state_keeps_non_mergeable_signal(self):
        summary = main._summarize_pr_mergeability(
            {"mergeable": False, "mergeable_state": "behind"}
        )

        self.assertEqual("当前不能直接自动合并（mergeable_state=behind）", summary)


if __name__ == "__main__":
    unittest.main()
