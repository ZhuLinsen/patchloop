import asyncio
import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from fastapi import HTTPException

from starlette.requests import Request

from agent.backlog_source import BacklogItem, BacklogSource
from agent.autocode_state_store import AutoCodeStateStore
from agent.task_queue import TaskQueue
from agent.task_scheduler import TaskLease, TaskScheduler, utc_now_iso
import autocode as main


class FakeGitHubClient:
    def __init__(
        self,
        *,
        issue_events: list[dict] | None = None,
        issue_comments: list[dict] | None = None,
        issues_by_number: dict[int, dict] | None = None,
        listed_issues: list[dict] | None = None,
    ):
        self.issue_events = issue_events or []
        self.issue_comments = issue_comments or []
        self.issues_by_number = issues_by_number or {}
        self.listed_issues = listed_issues or []
        self.issue_comment_since_calls: list[str | None] = []
        self.posted_comments: list[tuple[int, str]] = []

    def list_repo_issue_events(self, *, page: int = 1, **kwargs) -> list[dict]:
        del kwargs
        return list(self.issue_events) if page == 1 else []

    def list_repo_issue_comments(self, *, since: str | None = None, page: int = 1, **kwargs) -> list[dict]:
        del kwargs
        self.issue_comment_since_calls.append(since)
        return list(self.issue_comments) if page == 1 else []

    def get_issue(self, issue_number: int) -> dict:
        return dict(self.issues_by_number[issue_number])

    def get_pr(self, pr_number: int) -> dict:
        return dict(self.issues_by_number[pr_number])

    def list_issues(self, *, page: int = 1, **kwargs) -> list[dict]:
        del kwargs
        return list(self.listed_issues) if page == 1 else []

    def post_issue_comment(self, issue_number: int, body: str) -> dict:
        self.posted_comments.append((issue_number, body))
        return {"issue_number": issue_number, "body": body}


class FakeGitHubWriter:
    def __init__(self):
        self.created_issues: list[dict] = []

    def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict:
        payload = {"title": title, "body": body, "labels": list(labels or [])}
        self.created_issues.append(payload)
        return {
            "number": 701 + len(self.created_issues),
            "html_url": f"https://example.com/issues/{701 + len(self.created_issues)}",
            "state": "open",
            "updated_at": "2026-03-27T08:58:41Z",
            "created_at": "2026-03-27T08:58:41Z",
            "user": {"login": "maintainer", "type": "User"},
        }


class FakeBacklogSource:
    def __init__(self, items: list[BacklogItem]):
        self.items = items

    def collect_items(self) -> list[BacklogItem]:
        return list(self.items)

    def has_item_key(self, item_key: str) -> bool:
        return any(item.key == item_key for item in self.items)

    def get_item_by_key(self, item_key: str) -> BacklogItem | None:
        return next((item for item in self.items if item.key == item_key), None)


class FakePlanner:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def analyze(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        return self.response


def _build_request(payload: dict, *, secret: str, event_type: str) -> Request:
    body = json.dumps(payload).encode("utf-8")
    signature = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    headers = [
        (b"x-hub-signature-256", signature.encode("utf-8")),
        (b"x-github-event", event_type.encode("utf-8")),
        (b"content-type", b"application/json"),
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/webhook",
        "headers": headers,
    }

    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


class MainTests(unittest.TestCase):
    def test_github_webhook_returns_non_2xx_on_handler_failure(self):
        secret = "topsecret"
        request = _build_request(
            {"action": "opened", "issue": {"number": 1}},
            secret=secret,
            event_type="issues",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(enable_webhook=True),
                    github=SimpleNamespace(webhook_secret=secret),
                ),
            ), mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "_handle_issue_event", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
            ):
                response = asyncio.run(main.github_webhook(request))

            self.assertEqual(500, response.status_code)
            observability = state_store.observability_summary()
            self.assertEqual(1, observability["counters"]["webhook.received"])
            self.assertEqual(1, observability["counters"]["webhook.failed"])
            self.assertEqual("webhook", observability["recent_events"][-1]["category"])
            self.assertEqual("error", observability["recent_events"][-1]["level"])

    def test_poll_issues_once_processes_opened_and_reopened_events_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.ensure_polling_baseline()
            state_store._data["polling"]["issue_created_after"] = "2026-03-24T12:00:00Z"
            events = [
                {
                    "id": 200,
                    "event": "labeled",
                    "created_at": "2026-03-24T12:30:00Z",
                    "issue": {"number": 5},
                },
                {
                    "id": 201,
                    "event": "reopened",
                    "created_at": "2026-03-24T13:00:00Z",
                    "issue": {"number": 7},
                },
            ]
            reopened_issue = {
                "number": 7,
                "state": "open",
                "updated_at": "2026-03-24T13:00:00Z",
            }

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(server=SimpleNamespace(poll_items_per_page=10)),
            ), mock.patch.object(
                main,
                "gh",
                FakeGitHubClient(issue_events=events, issues_by_number={7: reopened_issue, 5: {"number": 5}}),
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "_handle_issue_event", new=mock.AsyncMock()) as handle_issue, mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._poll_issues_once())

            handle_issue.assert_awaited_once_with(reopened_issue, source="polling")
            self.assertEqual(1, stats["eligible"])
            self.assertEqual(1, stats["processed"])
            self.assertTrue(state_store.has_processed_issue_event(201))
            self.assertEqual(("2026-03-24T13:00:00Z", 201), state_store.issue_event_cursor())

    def test_poll_issues_once_uses_event_id_to_break_same_timestamp_ties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.ensure_polling_baseline()
            state_store.advance_issue_baseline("2026-03-24T12:00:00Z", 300)
            events = [
                {
                    "id": 300,
                    "event": "opened",
                    "created_at": "2026-03-24T12:00:00Z",
                    "issue": {"number": 3},
                },
                {
                    "id": 301,
                    "event": "reopened",
                    "created_at": "2026-03-24T12:00:00Z",
                    "issue": {"number": 4},
                },
            ]
            issue = {"number": 4, "state": "open", "updated_at": "2026-03-24T12:00:00Z"}

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(server=SimpleNamespace(poll_items_per_page=10)),
            ), mock.patch.object(
                main,
                "gh",
                FakeGitHubClient(issue_events=events, issues_by_number={4: issue, 3: {"number": 3, "state": "open"}}),
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "_handle_issue_event", new=mock.AsyncMock()) as handle_issue, mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._poll_issues_once())

            handle_issue.assert_awaited_once_with(issue, source="polling")
            self.assertEqual(1, stats["processed"])
            self.assertEqual(("2026-03-24T12:00:00Z", 301), state_store.issue_event_cursor())

    def test_reconcile_open_autocode_issues_queues_unprocessed_open_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            issue = {
                "number": 892,
                "state": "open",
                "updated_at": "2026-03-28T12:00:00Z",
                "labels": [{"name": "autocode"}, {"name": "enhancement"}],
                "user": {"login": "autocode-bot[bot]", "type": "Bot"},
            }

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=[issue])
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_autocode_issues_once())

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual({"fetched": 1, "eligible": 1, "processed": 1}, stats)
            self.assertEqual(1, len(tasks))
            self.assertEqual(892, tasks[0]["target_number"])
            self.assertEqual("reconcile", tasks[0]["source"])

    def test_reconcile_open_issues_queues_unprocessed_manual_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            issue = {
                "number": 894,
                "state": "open",
                "updated_at": "2026-03-28T12:10:00Z",
                "labels": [{"name": "bug"}],
                "user": {"login": "reporter", "type": "User"},
            }

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=[issue])
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_issues_once())

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual({"fetched": 1, "eligible": 1, "processed": 1}, stats)
            self.assertEqual(1, len(tasks))
            self.assertEqual(894, tasks[0]["target_number"])
            self.assertEqual("reconcile", tasks[0]["source"])

    def test_reconcile_open_issues_still_skips_non_autocode_bot_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            issue = {
                "number": 895,
                "state": "open",
                "updated_at": "2026-03-28T12:12:00Z",
                "labels": [{"name": "bug"}],
                "user": {"login": "helper-bot[bot]", "type": "Bot"},
            }

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=[issue])
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_issues_once())

            self.assertEqual({"fetched": 1, "eligible": 1, "processed": 0}, stats)
            self.assertEqual([], state_store.list_tasks(task_type="issue_execution"))

    def test_reconcile_open_autocode_issues_skips_processed_issue_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            state_store.mark_issue_plan(
                issue_key="issue:895:processed",
                issue_number=895,
                task_type="feature",
                updated_at="2026-03-28T12:00:00Z",
                summary="already planned",
            )
            issue = {
                "number": 895,
                "state": "open",
                "updated_at": "2026-03-28T12:00:00Z",
                "labels": [{"name": "autocode"}, {"name": "enhancement"}],
                "user": {"login": "maintainer", "type": "User"},
            }

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=[issue])
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_autocode_issues_once())

            self.assertEqual({"fetched": 1, "eligible": 0, "processed": 0}, stats)
            self.assertEqual([], state_store.list_tasks(task_type="issue_execution"))

    def test_reconcile_skips_blocked_and_failed_outcomes(self):
        """Issues with 'blocked' or 'failed' outcomes should be skipped by reconcile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            # Mark issue as processed with 'blocked' outcome
            state_store.mark_issue_plan(
                issue_key="issue:892:blocked",
                issue_number=892,
                task_type="feature",
                updated_at="2026-03-28T12:00:00Z",
                summary="planned",
                last_outcome="blocked",
            )
            # Mark issue as processed with 'failed' outcome
            state_store.mark_issue_plan(
                issue_key="issue:893:failed",
                issue_number=893,
                task_type="feature",
                updated_at="2026-03-28T12:00:00Z",
                summary="planned",
                last_outcome="failed",
            )
            issues = [
                {
                    "number": 892,
                    "state": "open",
                    "updated_at": "2026-03-28T12:00:00Z",
                    "labels": [{"name": "autocode"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
                {
                    "number": 893,
                    "state": "open",
                    "updated_at": "2026-03-28T12:00:00Z",
                    "labels": [{"name": "autocode"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
            ]
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=issues)
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_autocode_issues_once())

            self.assertEqual(0, stats["eligible"])
            self.assertEqual(0, stats["processed"])

    def test_reconcile_skips_planned_outcome(self):
        """Issues with 'planned' outcome should be skipped (needs explicit implement)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            state_store.mark_issue_plan(
                issue_key="issue:900:planned",
                issue_number=900,
                task_type="feature",
                updated_at="2026-03-28T12:00:00Z",
                summary="plan generated",
                last_outcome="planned",
            )
            issues = [
                {
                    "number": 900,
                    "state": "open",
                    "updated_at": "2026-03-28T12:00:00Z",
                    "labels": [{"name": "autocode"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
            ]
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=issues)
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_autocode_issues_once())

            self.assertEqual(0, stats["eligible"])
            self.assertEqual(0, stats["processed"])

    def test_safe_poll_step_isolates_errors(self):
        """_safe_poll_step should catch errors and return defaults."""

        async def _failing_handler():
            raise RuntimeError("API timeout")

        with mock.patch.object(main, "_record_state_event"):
            result = asyncio.run(main._safe_poll_step("test_step", _failing_handler))

        self.assertEqual(0, result["fetched"])
        self.assertEqual(0, result["eligible"])
        self.assertEqual(0, result["processed"])

    def test_safe_poll_step_passes_through_success(self):
        """_safe_poll_step should return handler result on success."""

        async def _ok_handler():
            return {"fetched": 5, "eligible": 3, "processed": 2}

        result = asyncio.run(main._safe_poll_step("test_step", _ok_handler))
        self.assertEqual(5, result["fetched"])
        self.assertEqual(3, result["eligible"])

    def test_reconcile_open_autocode_issues_scans_beyond_processed_front_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            processed_timestamp = "2026-03-28T12:00:00Z"
            for issue_number in (891, 892):
                state_store.mark_issue_plan(
                    issue_key=f"issue:{issue_number}:processed",
                    issue_number=issue_number,
                    task_type="feature",
                    updated_at=processed_timestamp,
                    summary="already planned",
                )
            issues = [
                {
                    "number": 891,
                    "state": "open",
                    "updated_at": processed_timestamp,
                    "labels": [{"name": "autocode"}, {"name": "enhancement"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
                {
                    "number": 892,
                    "state": "open",
                    "updated_at": processed_timestamp,
                    "labels": [{"name": "autocode"}, {"name": "enhancement"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
                {
                    "number": 893,
                    "state": "open",
                    "updated_at": "2026-03-28T12:05:00Z",
                    "labels": [{"name": "autocode"}, {"name": "enhancement"}],
                    "user": {"login": "maintainer", "type": "User"},
                },
            ]

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=2),
                    autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
                ),
            ), mock.patch.object(
                main, "gh", FakeGitHubClient(listed_issues=issues)
            ), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._reconcile_open_autocode_issues_once())

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual({"fetched": 3, "eligible": 1, "processed": 1}, stats)
            self.assertEqual(1, len(tasks))
            self.assertEqual(893, tasks[0]["target_number"])

    def test_list_open_autocode_issues_is_not_limited_to_five_pages(self):
        class PagedGitHubClient:
            def list_issues(self, *, page: int = 1, **kwargs) -> list[dict]:
                del kwargs
                if page > 6:
                    return []
                page_size = 100 if page < 6 else 1
                return [
                    {
                        "number": 800 + ((page - 1) * 100) + index,
                        "state": "open",
                        "updated_at": f"2026-03-28T12:{page:02d}:00Z",
                        "labels": [{"name": "autocode"}],
                    }
                    for index in range(1, page_size + 1)
                ]

        with mock.patch.object(main, "gh", PagedGitHubClient()):
            issues = main._list_open_autocode_issues()

        self.assertEqual(501, len(issues))
        self.assertEqual(1301, issues[-1]["number"])

    def test_list_open_autocode_issues_filters_non_autocode_issues(self):
        fake_gh = FakeGitHubClient(
            listed_issues=[
                {
                    "number": 876,
                    "state": "open",
                    "title": "[Bug] autocode issue",
                    "labels": [{"name": "autocode"}, {"name": "bug"}],
                },
                {
                    "number": 877,
                    "state": "open",
                    "title": "manual issue",
                    "labels": [{"name": "bug"}],
                },
            ]
        )

        with mock.patch.object(main, "gh", fake_gh):
            issues = main._list_open_autocode_issues()

        self.assertEqual([876], [int(issue["number"]) for issue in issues])

    def test_github_webhook_dispatches_pull_request_event(self):
        secret = "topsecret"
        request = _build_request(
            {"action": "synchronize", "pull_request": {"number": 11}},
            secret=secret,
            event_type="pull_request",
        )

        with mock.patch.object(
            main,
            "config",
            SimpleNamespace(
                server=SimpleNamespace(enable_webhook=True),
                github=SimpleNamespace(webhook_secret=secret),
            ),
        ), mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
            main, "state_store", object()
        ), mock.patch.object(main, "_handle_pull_request_event", new=mock.AsyncMock()) as handler:
            response = asyncio.run(main.github_webhook(request))

        self.assertEqual({"status": "ok"}, response)
        handler.assert_awaited_once()

    def test_github_webhook_dispatches_issue_comment_event(self):
        secret = "topsecret"
        request = _build_request(
            {
                "action": "edited",
                "issue": {"number": 8},
                "comment": {"id": 1001, "body": "实现"},
            },
            secret=secret,
            event_type="issue_comment",
        )

        with mock.patch.object(
            main,
            "config",
            SimpleNamespace(
                server=SimpleNamespace(enable_webhook=True),
                github=SimpleNamespace(webhook_secret=secret),
            ),
        ), mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
            main, "state_store", object()
        ), mock.patch.object(main, "_handle_issue_comment_event", new=mock.AsyncMock()) as handler:
            response = asyncio.run(main.github_webhook(request))

        self.assertEqual({"status": "ok"}, response)
        handler.assert_awaited_once()

    def test_handle_issue_comment_ignores_non_implement_comment(self):
        """普通评论（不含"实现"/"implement"）不触发执行。"""
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3),
        )):
            issue = {"number": 9, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 1002,
                "body": "这个看起来不错",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="polling"))
            queue_fn.assert_not_called()

    def test_handle_issue_comment_ignores_non_owner(self):
        """非 owner 的"实现"评论不触发执行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "config", SimpleNamespace(
                github=SimpleNamespace(owner="repoowner"),
                autocode=SimpleNamespace(allow_feature_on_comment=True),
            )), mock.patch.object(main, "state_store", state_store):
                issue = {"number": 9, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
                comment = {
                    "id": 1002,
                    "body": "实现",
                    "user": {"login": "random_user", "type": "User"},
                }
                with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                    asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="polling"))
                queue_fn.assert_not_called()
                self.assertTrue(state_store.has_processed_issue_comment(1002))

    def test_handle_issue_comment_triggers_on_implement_keyword(self):
        """Owner 评论"实现"应触发 explicit_command=True 的 issue 执行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "config", SimpleNamespace(
                github=SimpleNamespace(owner="repoowner"),
                autocode=SimpleNamespace(allow_feature_on_comment=True),
            )), mock.patch.object(main, "state_store", state_store):
                issue = {"number": 10, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
                comment = {
                    "id": 1003,
                    "body": "实现",
                    "created_at": "2026-03-24T00:01:00Z",
                    "user": {"login": "repoowner", "type": "User"},
                }
                with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                    asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
                queue_fn.assert_called_once()
                call_kwargs = queue_fn.call_args
                self.assertTrue(call_kwargs.kwargs.get("explicit_command", False) or call_kwargs[1].get("explicit_command", False))
                self.assertTrue(state_store.has_processed_issue_comment(1003, "2026-03-24T00:01:00Z"))

    def test_handle_issue_comment_promotes_existing_pending_issue_task_to_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            base_issue = {
                "number": 10,
                "state": "open",
                "updated_at": "2026-03-24T00:00:00Z",
                "user": {"login": "reporter", "type": "User"},
            }
            approved_issue = {
                **base_issue,
                "updated_at": "2026-03-24T00:01:00Z",
            }
            comment = {
                "id": 10031,
                "body": "实现",
                "updated_at": "2026-03-24T00:01:00Z",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "config", SimpleNamespace(
                github=SimpleNamespace(owner="repoowner"),
                autocode=SimpleNamespace(
                    auto_plan_on_issue_open=True,
                    issue_task_max_attempts=3,
                    allow_feature_on_comment=True,
                ),
            )), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ):
                main._queue_issue_execution_task(
                    base_issue,
                    source="polling",
                    trigger="issue_open",
                    explicit_command=False,
                )
                asyncio.run(main._handle_issue_comment_event({"issue": approved_issue, "comment": comment}, source="webhook"))

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(tasks))
            self.assertEqual("webhook", tasks[0]["source"])
            self.assertTrue(bool(tasks[0]["payload"].get("explicit_command", False)))

    def test_handle_issue_comment_triggers_on_english_implement(self):
        """Owner 评论"implement"(英文)也应触发执行。"""
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 11, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 1004,
                "body": "implement",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_called_once()

    def test_handle_issue_comment_triggers_on_please_implement_chinese(self):
        """Owner 评论"请实现"也应触发执行。"""
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 12, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 1005,
                "body": "请实现",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_called_once()

    def test_handle_issue_comment_triggers_on_semantic_implement_phrase(self):
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 12, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 10052,
                "body": "进行实现",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_called_once()

    def test_handle_issue_comment_triggers_on_not_resolved_followup(self):
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 12, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 10054,
                "body": "还是不行，继续修",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_called_once()
            self.assertTrue(queue_fn.call_args.kwargs.get("explicit_command"))

    def test_handle_issue_comment_ignores_conditional_multiline_implement(self):
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 12, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 10051,
                "body": "实现\n但记得补测试",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_not_called()

    def test_handle_issue_comment_ignores_owner_bot_marked_comment(self):
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 12, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            comment = {
                "id": 10053,
                "body": "这个方向可以直接实现。\n\n<!-- openreview:idempotency:issue:abc123 -->",
                "user": {"login": "repoowner", "type": "User"},
            }
            with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
            queue_fn.assert_not_called()

    def test_handle_issue_comment_ignores_negation(self):
        """包含"未实现"等否定形式不应触发执行。"""
        with mock.patch.object(main, "config", SimpleNamespace(
            github=SimpleNamespace(owner="repoowner"),
            autocode=SimpleNamespace(allow_feature_on_comment=True),
        )):
            issue = {"number": 13, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
            for body in ("未实现", "已实现", "难实现", "这个功能还未实现"):
                comment = {
                    "id": 1006,
                    "body": body,
                    "user": {"login": "repoowner", "type": "User"},
                }
                with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                    asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))
                queue_fn.assert_not_called(), f"'{body}' should NOT trigger implement"

    def test_handle_issue_comment_respects_comment_approval_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "config", SimpleNamespace(
                github=SimpleNamespace(owner="repoowner"),
                autocode=SimpleNamespace(allow_feature_on_comment=False),
            )), mock.patch.object(main, "state_store", state_store):
                issue = {"number": 14, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}
                comment = {
                    "id": 1007,
                    "body": "实现",
                    "updated_at": "2026-03-24T00:01:00Z",
                    "user": {"login": "repoowner", "type": "User"},
                }
                with mock.patch.object(main, "_queue_issue_execution_task") as queue_fn:
                    asyncio.run(main._handle_issue_comment_event({"issue": issue, "comment": comment}, source="webhook"))

                queue_fn.assert_not_called()
                self.assertTrue(state_store.has_processed_issue_comment(1007, "2026-03-24T00:01:00Z"))
                observability = state_store.observability_summary()
                self.assertEqual("issue_comment", observability["recent_events"][-1]["category"])
                self.assertEqual("comment_approval_disabled", observability["recent_events"][-1]["details"]["reason"])

    def test_poll_issue_comments_once_queues_open_issue_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.advance_issue_comment_baseline("2026-03-23T00:00:00Z", 0)
            gh = FakeGitHubClient(
                issue_comments=[
                    {
                        "id": 1101,
                        "body": "请实现",
                        "updated_at": "2026-03-24T00:01:00Z",
                        "issue_url": "https://github.com/octo-org/example-repo/issues/42",
                        "user": {"login": "repoowner", "type": "User"},
                    }
                ],
                issues_by_number={
                    42: {
                        "number": 42,
                        "state": "open",
                        "updated_at": "2026-03-24T00:00:00Z",
                        "labels": [{"name": "enhancement"}],
                    }
                },
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    github=SimpleNamespace(owner="repoowner"),
                ),
            ), mock.patch.object(main, "gh", gh), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ), mock.patch.object(
                main, "_queue_issue_execution_task"
            ) as queue_fn:
                stats = asyncio.run(main._poll_issue_comments_once())

            self.assertEqual({"fetched": 1, "eligible": 1, "processed": 1}, stats)
            queue_fn.assert_called_once()
            self.assertEqual(["2026-03-23T00:00:00Z"], gh.issue_comment_since_calls)
            self.assertTrue(state_store.has_processed_issue_comment(1101, "2026-03-24T00:01:00Z"))
            self.assertEqual(("2026-03-24T00:01:00Z", 1101), state_store.issue_comment_cursor())

    def test_poll_issue_comments_once_skips_pull_request_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.advance_issue_comment_baseline("2026-03-23T00:00:00Z", 0)
            gh = FakeGitHubClient(
                issue_comments=[
                    {
                        "id": 1102,
                        "body": "实现",
                        "updated_at": "2026-03-24T00:01:00Z",
                        "issue_url": "https://github.com/octo-org/example-repo/issues/77",
                        "user": {"login": "repoowner", "type": "User"},
                    }
                ],
                issues_by_number={
                    77: {
                        "number": 77,
                        "state": "open",
                        "updated_at": "2026-03-24T00:00:00Z",
                        "pull_request": {"url": "https://api.github.com/repos/x/y/pulls/77"},
                    }
                },
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    github=SimpleNamespace(owner="repoowner"),
                ),
            ), mock.patch.object(main, "gh", gh), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ), mock.patch.object(
                main, "_queue_issue_execution_task"
            ) as queue_fn:
                stats = asyncio.run(main._poll_issue_comments_once())

            self.assertEqual({"fetched": 1, "eligible": 0, "processed": 0}, stats)
            queue_fn.assert_not_called()
            self.assertTrue(state_store.has_processed_issue_comment(1102, "2026-03-24T00:01:00Z"))

    def test_poll_issue_comments_once_marks_invalid_issue_url_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.advance_issue_comment_baseline("2026-03-23T00:00:00Z", 0)
            gh = FakeGitHubClient(
                issue_comments=[
                    {
                        "id": 1103,
                        "body": "实现",
                        "updated_at": "2026-03-24T00:01:00Z",
                        "issue_url": "",
                        "user": {"login": "repoowner", "type": "User"},
                    }
                ]
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    github=SimpleNamespace(owner="repoowner"),
                ),
            ), mock.patch.object(main, "gh", gh), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ), mock.patch.object(
                main, "_queue_issue_execution_task"
            ) as queue_fn:
                stats = asyncio.run(main._poll_issue_comments_once())

            self.assertEqual({"fetched": 1, "eligible": 0, "processed": 0}, stats)
            queue_fn.assert_not_called()
            self.assertTrue(state_store.has_processed_issue_comment(1103, "2026-03-24T00:01:00Z"))
            self.assertEqual(("2026-03-24T00:01:00Z", 1103), state_store.issue_comment_cursor())

    def test_poll_issue_comments_once_handles_issue_fetch_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.advance_issue_comment_baseline("2026-03-23T00:00:00Z", 0)
            gh = FakeGitHubClient(
                issue_comments=[
                    {
                        "id": 1104,
                        "body": "实现",
                        "updated_at": "2026-03-24T00:01:00Z",
                        "issue_url": "https://github.com/octo-org/example-repo/issues/404",
                        "user": {"login": "repoowner", "type": "User"},
                    }
                ]
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    github=SimpleNamespace(owner="repoowner"),
                ),
            ), mock.patch.object(main, "gh", gh), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ), mock.patch.object(
                main, "_queue_issue_execution_task"
            ) as queue_fn:
                stats = asyncio.run(main._poll_issue_comments_once())

            self.assertEqual({"fetched": 1, "eligible": 0, "processed": 0}, stats)
            queue_fn.assert_not_called()
            self.assertTrue(state_store.has_processed_issue_comment(1104, "2026-03-24T00:01:00Z"))
            observability = state_store.observability_summary()
            self.assertEqual("issue_comment", observability["recent_events"][-1]["category"])
            self.assertEqual("error", observability["recent_events"][-1]["level"])

    def test_handle_issue_event_enqueues_background_issue_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            issue = {"number": 12, "updated_at": "2026-03-24T00:00:00Z", "user": {"login": "mumu", "type": "User"}}
            with mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "issue_task_queue", queue), mock.patch.object(
                main,
                "config",
                SimpleNamespace(autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3)),
            ):
                asyncio.run(main._handle_issue_event(issue, source="polling"))

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(tasks))
            self.assertEqual("queued", tasks[0]["status"])
            self.assertEqual(12, tasks[0]["target_number"])

    def test_handle_issue_event_skips_already_processed_issue_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            state_store.mark_issue_plan(
                issue_key="issue:12:processed",
                issue_number=12,
                task_type="bug_fix",
                updated_at="2026-03-24T00:00:00Z",
                summary="already planned",
            )
            issue = {"number": 12, "updated_at": "2026-03-24T00:00:00Z", "user": {"login": "mumu", "type": "User"}}
            with mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "issue_task_queue", queue), mock.patch.object(
                main,
                "config",
                SimpleNamespace(autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3)),
            ):
                asyncio.run(main._handle_issue_event(issue, source="webhook"))

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual([], tasks)

    def test_handle_issue_event_skips_duplicate_older_processed_version_after_newer_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            state_store.mark_issue_plan(
                issue_key="issue:12:old",
                issue_number=12,
                task_type="bug_fix",
                updated_at="2026-03-24T00:00:00Z",
                summary="old plan",
            )
            state_store.mark_issue_plan(
                issue_key="issue:12:new",
                issue_number=12,
                task_type="bug_fix",
                updated_at="2026-03-25T00:00:00Z",
                summary="new plan",
            )
            delayed_issue = {
                "number": 12,
                "updated_at": "2026-03-24T00:00:00Z",
                "user": {"login": "mumu", "type": "User"},
            }
            with mock.patch.object(main, "controller", object()), mock.patch.object(main, "gh", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "issue_task_queue", queue), mock.patch.object(
                main,
                "config",
                SimpleNamespace(autocode=SimpleNamespace(auto_plan_on_issue_open=True, issue_task_max_attempts=3)),
            ):
                asyncio.run(main._handle_issue_event(delayed_issue, source="webhook"))

            tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual([], tasks)

    def test_sync_backlog_once_creates_issue_and_persists_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            item = BacklogItem(
                key="backlog:1",
                title="[Backlog] add tests",
                body="body",
                source_path="TODO.md",
                line_number=3,
                labels=["autocode", "backlog"],
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(autocode=SimpleNamespace(enable_backlog_source=True)),
            ), mock.patch.object(main, "source_task_queue", queue), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "backlog_source", FakeBacklogSource([item])
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._sync_backlog_once())

            self.assertEqual(1, stats["queued"])
            tasks = state_store.list_tasks(task_type="source_issue_create")
            self.assertEqual(1, len(tasks))
            self.assertEqual("create_issue", tasks[0]["command"])

    def test_sync_backlog_once_requeues_completed_dry_run_task_when_real_issue_is_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            item = BacklogItem(
                key="backlog:1",
                title="[Backlog] add tests",
                body="body",
                source_path="TODO.md",
                line_number=3,
                labels=["autocode", "backlog"],
            )
            task_id = main._build_source_task_id(item.key)
            queue.enqueue(
                task_id=task_id,
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={"item_key": "backlog:1", "title": "[Backlog] old title"},
            )
            state_store.mark_task_succeeded(task_id, result_summary="dry_run")
            state_store.mark_backlog_item_synced(
                item_key="backlog:1",
                source_name="backlog",
                source_path="TODO.md",
                line_number=3,
                title="[Backlog] add tests",
                issue_number=0,
                issue_url="",
                dry_run=True,
            )

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(enable_backlog_source=True, source_task_max_attempts=3),
                ),
            ), mock.patch.object(main, "source_task_queue", queue), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "backlog_source", FakeBacklogSource([item])
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._sync_backlog_once())

            self.assertEqual({"fetched": 1, "queued": 1}, stats)
            task = state_store.get_task(task_id)
            assert task is not None
            self.assertEqual("queued", task["status"])
            self.assertEqual("[Backlog] add tests", task["payload"]["title"])

    def test_sync_plan_files_once_creates_issue_and_persists_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            item = BacklogItem(
                key="plan:1",
                title="[Plan] add coverage",
                body="body",
                source_path="PLAN.md",
                line_number=3,
                labels=["autocode", "plan"],
            )
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(autocode=SimpleNamespace(enable_plan_source=True)),
            ), mock.patch.object(main, "source_task_queue", queue), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "plan_source", FakeBacklogSource([item])
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._sync_plan_files_once())

            self.assertEqual(1, stats["queued"])
            tasks = state_store.list_tasks(task_type="source_issue_create")
            self.assertEqual(1, len(tasks))
            self.assertEqual("create_issue", tasks[0]["command"])

    def test_build_background_scheduler_registers_polling_and_pr_queue(self):
        with mock.patch.object(
            main,
            "config",
            SimpleNamespace(
                server=SimpleNamespace(event_source="polling", poll_interval_seconds=30),
                autocode=SimpleNamespace(
                    poll_review_feedback=True,
                    enable_pr_queue=True,
                    enable_plan_source=False,
                    enable_backlog_source=False,
                    enable_idle_scanner=True,
                    issue_task_interval_seconds=5,
                    pr_task_interval_seconds=5,
                    source_task_interval_seconds=5,
                    review_feedback_interval_seconds=120,
                    idle_scanner_interval_seconds=600,
                ),
            ),
        ), mock.patch.object(main, "controller", object()):
            scheduler = main._build_background_scheduler()

        summary = scheduler.summary()
        self.assertEqual(6, summary["registered"])
        self.assertEqual(
            ["idle-scan-cycle", "issue-task-cycle", "polling-cycle", "pr-queue-cycle", "pr-task-cycle", "source-task-cycle"],
            [item["name"] for item in summary["tasks"]],
        )

    def test_should_run_idle_scan_returns_false_when_pr_work_is_pending(self):
        state_store = SimpleNamespace(
            summary=lambda: {"pending_tracked_prs": 1},
            count_active_tasks=lambda **kw: 0,
        )
        fake_scheduler = SimpleNamespace(summary=lambda: {"tasks": [{"name": "pr-queue-cycle", "status": "succeeded"}]})

        with mock.patch.object(main, "state_store", state_store), mock.patch.object(main, "scheduler", fake_scheduler):
            self.assertFalse(main._should_run_idle_scan())

    def test_should_run_idle_scan_returns_false_when_queue_has_pending_items(self):
        state_store = SimpleNamespace(
            summary=lambda: {
                "pending_tracked_prs": 0,
            },
            count_active_tasks=lambda **kw: 1,
        )
        fake_scheduler = SimpleNamespace(summary=lambda: {"tasks": [{"name": "issue-task-cycle", "status": "succeeded"}]})

        with mock.patch.object(main, "state_store", state_store), mock.patch.object(main, "scheduler", fake_scheduler):
            self.assertFalse(main._should_run_idle_scan())

    def test_should_run_idle_scan_ignores_pr_repair_tasks(self):
        """Idle scan should NOT be blocked by PR repair tasks or PR scheduler cycles."""
        state_store = SimpleNamespace(
            summary=lambda: {"pending_tracked_prs": 0},
            count_active_tasks=lambda **kw: 0,
        )
        fake_scheduler = SimpleNamespace(
            summary=lambda: {
                "tasks": [
                    {"name": "pr-task-cycle", "status": "running"},
                    {"name": "pr-queue-cycle", "status": "succeeded"},
                    {"name": "issue-task-cycle", "status": "succeeded"},
                ]
            }
        )

        with mock.patch.object(main, "state_store", state_store), mock.patch.object(main, "scheduler", fake_scheduler):
            self.assertTrue(main._should_run_idle_scan())

    def test_should_run_idle_scan_blocked_by_issue_scheduler_cycle(self):
        """Idle scan SHOULD be blocked when issue-task-cycle is running."""
        state_store = SimpleNamespace(
            summary=lambda: {"pending_tracked_prs": 0},
            count_active_tasks=lambda **kw: 0,
        )
        fake_scheduler = SimpleNamespace(
            summary=lambda: {
                "tasks": [
                    {"name": "pr-task-cycle", "status": "running"},
                    {"name": "issue-task-cycle", "status": "running"},
                ]
            }
        )

        with mock.patch.object(main, "state_store", state_store), mock.patch.object(main, "scheduler", fake_scheduler):
            self.assertFalse(main._should_run_idle_scan())

    def test_run_idle_scan_cycle_skips_when_non_idle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "_should_run_idle_scan", return_value=False), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "source_task_queue", object()), mock.patch.object(
                main, "idle_scanner", object()
            ), mock.patch.object(main, "_enqueue_plan_source_items_once", new=mock.AsyncMock()) as enqueue_once:
                asyncio.run(main._run_idle_scan_cycle())

            enqueue_once.assert_not_awaited()
            observability = state_store.observability_summary()
            cycles = {item["name"]: item for item in observability["cycles"]}
            self.assertEqual("skipped", cycles["idle-scan-cycle"]["last_status"])
            self.assertEqual({"reason": "active_work_pending"}, cycles["idle-scan-cycle"]["last_stats"])

    def test_health_includes_scheduler_summary(self):
        fake_scheduler = SimpleNamespace(summary=lambda: {"registered": 1, "tasks": [{"name": "polling-cycle"}]})
        fake_issue_queue = SimpleNamespace(list_tasks=lambda: [])
        fake_pr_queue = SimpleNamespace(list_tasks=lambda: [])
        fake_source_queue = SimpleNamespace(list_tasks=lambda: [])
        with mock.patch.object(
            main,
            "config",
            SimpleNamespace(
                github=SimpleNamespace(repo="demo/repo"),
                server=SimpleNamespace(event_source="polling", enable_webhook=True),
                autocode=SimpleNamespace(
                    enabled=True,
                    mode="manual",
                    pr_base_branch="main",
                    forbid_main_base_branch=False,
                    force_draft_pr=False,
                    enable_plan_source=True,
                    enable_pr_queue=True,
                    poll_review_feedback=True,
                    enable_backlog_source=False,
                    enable_idle_scanner=True,
                ),
            ),
        ), mock.patch.object(
            main,
            "state_store",
            SimpleNamespace(
                summary=lambda: {"runs": 3},
                observability_summary=lambda: {"counters": {"webhook.received": 2}},
            ),
        ), mock.patch.object(main, "issue_task_queue", fake_issue_queue), mock.patch.object(
            main, "pr_task_queue", fake_pr_queue
        ), mock.patch.object(
            main, "source_task_queue", fake_source_queue
        ), mock.patch.object(
            main, "scheduler", fake_scheduler
        ):
            result = asyncio.run(main.health())

        self.assertEqual({"registered": 1, "tasks": [{"name": "polling-cycle"}]}, result["scheduler"])
        empty_queue = {
            "total": 0,
            "queued": 0,
            "running": 0,
            "retry_waiting": 0,
            "dead_letter": 0,
            "dead_letter_action_required": 0,
            "dead_letter_superseded": 0,
            "problem": 0,
            "tasks": [],
        }
        self.assertEqual(empty_queue, result["issue_task_queue"])
        self.assertEqual(empty_queue, result["pr_task_queue"])
        self.assertEqual(empty_queue, result["source_task_queue"])
        self.assertEqual({"runs": 3}, result["state"])
        self.assertEqual({"counters": {"webhook.received": 2}}, result["observability"])
        self.assertTrue(result["plan_source_enabled"])
        self.assertTrue(result["idle_scanner_enabled"])

    def test_observability_endpoint_returns_observability_payload(self):
        fake_scheduler = SimpleNamespace(summary=lambda: {"registered": 1})
        fake_issue_queue = SimpleNamespace(list_tasks=lambda: [])
        fake_pr_queue = SimpleNamespace(list_tasks=lambda: [])
        fake_source_queue = SimpleNamespace(list_tasks=lambda: [])
        fake_state_store = SimpleNamespace(
            summary=lambda: {"runs": 4},
            observability_summary=lambda: {"recent_events": [{"message": "ok"}]},
        )
        with mock.patch.object(
            main,
            "config",
            SimpleNamespace(github=SimpleNamespace(repo="demo/repo")),
        ), mock.patch.object(main, "scheduler", fake_scheduler), mock.patch.object(
            main, "issue_task_queue", fake_issue_queue
        ), mock.patch.object(
            main, "pr_task_queue", fake_pr_queue
        ), mock.patch.object(
            main, "source_task_queue", fake_source_queue
        ), mock.patch.object(
            main, "state_store", fake_state_store
        ):
            result = asyncio.run(main.observability())

        self.assertEqual("demo/repo", result["repo"])
        self.assertEqual({"registered": 1}, result["scheduler"])
        empty_queue = {
            "total": 0,
            "queued": 0,
            "running": 0,
            "retry_waiting": 0,
            "dead_letter": 0,
            "dead_letter_action_required": 0,
            "dead_letter_superseded": 0,
            "problem": 0,
            "tasks": [],
        }
        self.assertEqual(empty_queue, result["issue_task_queue"])
        self.assertEqual(empty_queue, result["pr_task_queue"])
        self.assertEqual(empty_queue, result["source_task_queue"])
        self.assertEqual({"runs": 4}, result["state"])
        self.assertEqual({"recent_events": [{"message": "ok"}]}, result["observability"])

    def test_metrics_endpoint_returns_prometheus_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.increment_counter("webhook.received")
            state_store.record_event(category="webhook", level="error", message="webhook failed", source="webhook")
            state_store.mark_cycle_started("polling-cycle")
            state_store.record_cycle_result("polling-cycle", status="succeeded", stats={"processed": 2})
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            issue_queue.enqueue(task_id="issue-task:31:test", target_number=31, source="polling", command="implement")
            scheduler = TaskScheduler()

            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(enable_webhook=True),
                    autocode=SimpleNamespace(
                        enabled=True,
                        enable_pr_queue=True,
                        enable_plan_source=False,
                        enable_backlog_source=True,
                        enable_idle_scanner=False,
                    ),
                ),
            ), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "scheduler", scheduler
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "pr_task_queue", SimpleNamespace(list_tasks=lambda *args, **kwargs: [])
            ), mock.patch.object(
                main, "source_task_queue", SimpleNamespace(list_tasks=lambda *args, **kwargs: [])
            ):
                response = asyncio.run(main.metrics())

        self.assertIn("autocode_up 1", response.body.decode("utf-8"))
        self.assertIn('autocode_webhook_enabled 1', response.body.decode("utf-8"))
        self.assertIn('autocode_task_queue_tasks{queue="issue",status="queued"} 1', response.body.decode("utf-8"))
        self.assertIn('autocode_observability_counter_total{name="webhook.received"} 1', response.body.decode("utf-8"))
        self.assertIn('autocode_cycle_success_total{cycle="polling-cycle"} 1', response.body.decode("utf-8"))
        self.assertIn('autocode_recent_events_by_category{category="webhook"} 1', response.body.decode("utf-8"))
        self.assertIn('autocode_recent_events_by_level{level="error"} 1', response.body.decode("utf-8"))

    def test_state_snapshot_endpoint_exports_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.increment_counter("webhook.received")
            with mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.export_state_snapshot())

        self.assertEqual("ok", result["status"])
        self.assertEqual(1, result["snapshot"]["snapshot_version"])
        self.assertEqual(1, result["snapshot"]["state"]["observability"]["counters"]["webhook.received"])

    def test_logs_events_endpoint_filters_recent_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.record_event(category="webhook", level="info", message="accepted", source="webhook")
            state_store.record_event(category="webhook", level="error", message="failed", source="webhook")
            state_store.record_event(category="issue_task_queue", level="error", message="retry", source="polling")

            with mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.list_log_events(category="webhook", level="error", source="webhook", limit=10))

        self.assertEqual("ok", result["status"])
        self.assertEqual(1, result["count"])
        self.assertEqual("failed", result["events"][0]["message"])
        self.assertEqual(["webhook"], result["filters"]["category"])
        self.assertEqual(["error"], result["filters"]["level"])
        self.assertEqual(["webhook"], result["filters"]["source"])

    def test_restore_state_snapshot_restores_data_and_records_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            snapshot = state_store.export_snapshot()
            snapshot["state"]["observability"]["counters"]["webhook.received"] = 5
            request = _build_request({"confirm_restore": True, "snapshot": snapshot}, secret="unused", event_type="unused")
            request._body = json.dumps({"confirm_restore": True, "snapshot": snapshot}).encode("utf-8")

            async def receive():
                return {"type": "http.request", "body": request._body, "more_body": False}

            restore_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/state/restore",
                    "headers": [(b"content-type", b"application/json")],
                },
                receive,
            )
            scheduler = TaskScheduler()
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "scheduler", scheduler
            ), mock.patch.object(main, "scheduler_task", None):
                result = asyncio.run(main.restore_state_snapshot(restore_request))

        self.assertEqual("ok", result["status"])
        self.assertEqual(5, state_store.observability_summary()["counters"]["webhook.received"])
        self.assertEqual(1, state_store.observability_summary()["counters"]["state.restore"])
        self.assertEqual("state_restore", state_store.observability_summary()["recent_events"][-1]["category"])

    def test_restore_state_snapshot_rejects_when_background_work_is_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            snapshot = state_store.export_snapshot()

            async def receive():
                return {
                    "type": "http.request",
                    "body": json.dumps({"confirm_restore": True, "snapshot": snapshot}).encode("utf-8"),
                    "more_body": False,
                }

            restore_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/state/restore",
                    "headers": [(b"content-type", b"application/json")],
                },
                receive,
            )
            fake_scheduler = SimpleNamespace(summary=lambda: {"running": 1})
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(main, "scheduler", fake_scheduler):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(main.restore_state_snapshot(restore_request))

        self.assertEqual(409, ctx.exception.status_code)

    def test_restore_state_snapshot_rejects_when_queued_work_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(task_id="issue-task:queued", target_number=9, source="polling")
            snapshot = state_store.export_snapshot()

            async def receive():
                return {
                    "type": "http.request",
                    "body": json.dumps({"confirm_restore": True, "snapshot": snapshot}).encode("utf-8"),
                    "more_body": False,
                }

            restore_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/state/restore",
                    "headers": [(b"content-type", b"application/json")],
                },
                receive,
            )
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "pr_task_queue", None), mock.patch.object(
                main, "source_task_queue", None
            ), mock.patch.object(
                main, "scheduler", None
            ):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(main.restore_state_snapshot(restore_request))

        self.assertEqual(409, ctx.exception.status_code)

    def test_restore_state_snapshot_rejects_invalid_json_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))

            async def receive():
                return {
                    "type": "http.request",
                    "body": b"{not-json",
                    "more_body": False,
                }

            restore_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/state/restore",
                    "headers": [(b"content-type", b"application/json")],
                },
                receive,
            )
            with mock.patch.object(main, "state_store", state_store):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(main.restore_state_snapshot(restore_request))

        self.assertEqual(400, ctx.exception.status_code)

    def test_list_all_tasks_returns_aggregated_summary_and_problem_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="pr-test")
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")

            issue_queue.enqueue(
                task_id="issue-task:21:test",
                target_number=21,
                source="polling",
                priority="urgent",
                command="implement",
            )
            pr_queue.enqueue(
                task_id="pr-task:22:test",
                target_number=22,
                source="webhook",
                command="review_feedback",
                max_attempts=1,
            )
            self.assertIsNotNone(pr_queue.claim_next())
            pr_queue.mark_failed("pr-task:22:test", error="boom", retry_delay_seconds=1)
            source_queue.enqueue(
                task_id="source-task:test",
                target_number=0,
                source="plan",
                priority="background",
                command="create_issue",
                max_attempts=2,
            )
            self.assertIsNotNone(source_queue.claim_next())
            source_queue.mark_failed("source-task:test", error="later", retry_delay_seconds=60)

            with mock.patch.object(main, "issue_task_queue", issue_queue), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(main, "source_task_queue", source_queue):
                result = asyncio.run(main.list_all_tasks())

        self.assertEqual(
            {
                "queues": 3,
                "total": 3,
                "queued": 1,
                "running": 0,
                "retry_waiting": 1,
                "dead_letter": 1,
                "dead_letter_action_required": 1,
                "dead_letter_superseded": 0,
                "problem": 2,
            },
            result["summary"],
        )
        self.assertEqual(1, result["queues"]["issue"]["queued"])
        self.assertEqual(1, result["queues"]["pr"]["dead_letter"])
        self.assertEqual(1, result["queues"]["source"]["retry_waiting"])
        self.assertEqual(2, len(result["active_tasks"]))
        self.assertEqual(2, len(result["problem_tasks"]))
        statuses_by_queue = {item["queue"]: item["status"] for item in result["problem_tasks"]}
        self.assertEqual({"pr": "dead_letter", "source": "retry_waiting"}, statuses_by_queue)

    def test_dead_letter_superseded_by_later_done_task_is_not_action_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            issue_queue.enqueue(task_id="issue-task:71:dead", target_number=71, source="polling", max_attempts=1)
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_failed("issue-task:71:dead", error="old failure", retry_delay_seconds=1)
            issue_queue.enqueue(task_id="issue-task:71:done", target_number=71, source="polling")
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_succeeded("issue-task:71:done", result_summary="answered")
            state_store._data["task_queue"]["issue-task:71:dead"]["last_finished_at"] = "2026-04-01T00:00:00Z"
            state_store._data["task_queue"]["issue-task:71:dead"]["updated_at"] = "2026-04-01T00:00:00Z"
            state_store._data["task_queue"]["issue-task:71:done"]["last_finished_at"] = "2026-04-01T00:01:00Z"
            state_store._data["task_queue"]["issue-task:71:done"]["updated_at"] = "2026-04-01T00:01:00Z"

            with mock.patch.object(main, "issue_task_queue", issue_queue), mock.patch.object(
                main, "pr_task_queue", None
            ), mock.patch.object(main, "source_task_queue", None):
                tasks_result = asyncio.run(main.list_all_tasks())
                dead_result = asyncio.run(main.export_dead_letter_tasks(queue="issue", limit=10))
                filtered_dead_result = asyncio.run(
                    main.export_dead_letter_tasks(queue="issue", limit=10, include_superseded=False)
                )

        self.assertEqual(1, tasks_result["summary"]["dead_letter"])
        self.assertEqual(0, tasks_result["summary"]["dead_letter_action_required"])
        self.assertEqual(1, tasks_result["summary"]["dead_letter_superseded"])
        self.assertEqual(0, tasks_result["summary"]["problem"])
        self.assertEqual([], tasks_result["problem_tasks"])
        self.assertEqual(1, dead_result["count"])
        self.assertEqual(0, dead_result["action_required_count"])
        self.assertEqual(1, dead_result["superseded_count"])
        self.assertTrue(dead_result["tasks"][0]["superseded"])
        self.assertEqual("issue-task:71:done", dead_result["tasks"][0]["superseded_by_task_id"])
        self.assertEqual(0, filtered_dead_result["count"])

    def test_list_issue_tasks_filters_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            issue_queue.enqueue(task_id="issue-task:41:done", target_number=41, source="polling", command="implement")
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_succeeded("issue-task:41:done", result_summary="ok")
            issue_queue.enqueue(task_id="issue-task:42:dead", target_number=42, source="polling", command="implement", max_attempts=1)
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_failed("issue-task:42:dead", error="boom", retry_delay_seconds=1)

            with mock.patch.object(main, "issue_task_queue", issue_queue):
                result = asyncio.run(main.list_issue_tasks(status="dead_letter"))

        self.assertEqual("ok", result["status"])
        self.assertEqual(1, len(result["tasks"]))
        self.assertEqual("issue-task:42:dead", result["tasks"][0]["task_id"])

    def test_export_dead_letter_tasks_returns_filtered_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="pr-test")
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue.enqueue(task_id="issue-task:51:dead", target_number=51, source="polling", max_attempts=1)
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_failed("issue-task:51:dead", error="boom", retry_delay_seconds=1)
            pr_queue.enqueue(task_id="pr-task:52:dead", target_number=52, source="webhook", max_attempts=1)
            self.assertIsNotNone(pr_queue.claim_next())
            pr_queue.mark_failed("pr-task:52:dead", error="boom", retry_delay_seconds=1)
            source_queue.enqueue(task_id="source-task:53:wait", target_number=0, source="plan", max_attempts=2)
            self.assertIsNotNone(source_queue.claim_next())
            source_queue.mark_failed("source-task:53:wait", error="later", retry_delay_seconds=60)

            with mock.patch.object(main, "issue_task_queue", issue_queue), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(main, "source_task_queue", source_queue):
                result = asyncio.run(main.export_dead_letter_tasks(queue="issue,pr", limit=10))

        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["count"])
        self.assertEqual(["issue", "pr"], result["filters"]["queue"])
        self.assertEqual({"issue", "pr"}, {item["queue"] for item in result["tasks"]})

    def test_retry_tasks_batch_requeues_problem_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="pr-test")
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue.enqueue(task_id="issue-task:61:dead", target_number=61, source="polling", max_attempts=1)
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_failed("issue-task:61:dead", error="boom", retry_delay_seconds=1)
            pr_queue.enqueue(task_id="pr-task:62:wait", target_number=62, source="webhook", max_attempts=2)
            self.assertIsNotNone(pr_queue.claim_next())
            pr_queue.mark_failed("pr-task:62:wait", error="later", retry_delay_seconds=60)
            source_queue.enqueue(task_id="source-task:63:done", target_number=0, source="plan")

            with mock.patch.object(main, "issue_task_queue", issue_queue), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(main, "source_task_queue", source_queue), mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.retry_tasks(queue="issue,pr", status="dead_letter,retry_waiting", limit=10))

        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["matched"])
        self.assertEqual(2, len(result["retried"]))
        self.assertEqual({"issue", "pr"}, {item["queue"] for item in result["retried"]})
        self.assertEqual("queued", state_store.get_task("issue-task:61:dead")["status"])
        self.assertEqual("queued", state_store.get_task("pr-task:62:wait")["status"])
        observability = state_store.observability_summary()
        self.assertEqual(1, observability["counters"]["tasks.batch_retry"])

    def test_drain_issue_task_queue_executes_task_and_marks_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(
                task_id="issue-task:14:test",
                target_number=14,
                source="polling",
                command="implement",
                payload={"explicit_command": True},
            )
            fake_gh = FakeGitHubClient(
                issues_by_number={14: {"number": 14, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )
            fake_controller = SimpleNamespace(
                execute_issue=lambda issue, explicit_command: SimpleNamespace(
                    comment="## done",
                    run_record=SimpleNamespace(status="done", run_id="run-14"),
                )
            )
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_issue_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("issue-task:14:test")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual([], fake_gh.posted_comments)

    def test_drain_issue_task_queue_marks_failed_results_for_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(
                task_id="issue-task:18:test",
                target_number=18,
                source="polling",
                command="implement",
                payload={"explicit_command": True},
            )
            fake_gh = FakeGitHubClient(
                issues_by_number={18: {"number": 18, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )
            fake_controller = SimpleNamespace(
                execute_issue=lambda issue, explicit_command: SimpleNamespace(
                    comment="## failed",
                    run_record=SimpleNamespace(
                        status="failed",
                        run_id="run-18",
                        failure_reason="validation failed",
                    ),
                )
            )
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_issue_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 0, "failed": 1}, stats)
            task = state_store.get_task("issue-task:18:test")
            assert task is not None
            self.assertEqual("retry_waiting", task["status"])
            self.assertEqual("validation failed", task["last_error"])

    def test_drain_issue_task_queue_treats_blocked_results_as_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(
                task_id="issue-task:19:test",
                target_number=19,
                source="polling",
                command="implement",
                payload={"explicit_command": True},
            )
            fake_gh = FakeGitHubClient(
                issues_by_number={19: {"number": 19, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )
            fake_controller = SimpleNamespace(
                execute_issue=lambda issue, explicit_command: SimpleNamespace(
                    comment="## blocked",
                    run_record=SimpleNamespace(
                        status="blocked",
                        run_id="run-19",
                        failure_reason="本地验证环境未就绪",
                    ),
                )
            )
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_issue_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("issue-task:19:test")
            assert task is not None
            self.assertEqual("done", task["status"])

    def test_drain_issue_task_queue_skips_paused_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.set_issue_paused(16, paused=True, reason="manual abort")
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(task_id="issue-task:16:test", target_number=16, source="polling", command="implement")
            fake_gh = FakeGitHubClient(issues_by_number={16: {"number": 16, "updated_at": "2026-03-24T00:00:00Z"}})
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", object()
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_issue_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("issue-task:16:test")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("skipped_paused", task["result_summary"])
            self.assertEqual([], fake_gh.posted_comments)

    def test_drain_issue_task_queue_skips_closed_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(task_id="issue-task:17:test", target_number=17, source="polling", command="implement")
            fake_gh = FakeGitHubClient(
                issues_by_number={17: {"number": 17, "state": "closed", "updated_at": "2026-03-24T00:00:00Z"}}
            )
            fake_controller = SimpleNamespace(execute_issue=mock.Mock())
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_issue_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("issue-task:17:test")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("skipped_closed", task["result_summary"])
            fake_controller.execute_issue.assert_not_called()
            self.assertEqual([], fake_gh.posted_comments)

    def test_retry_issue_task_endpoint_requeues_dead_letter_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            queue.enqueue(task_id="issue-task:15:test", target_number=15, source="polling", max_attempts=1)
            claimed = state_store.claim_next_task(task_type="issue_execution", owner="test")
            self.assertIsNotNone(claimed)
            state_store.mark_task_failed("issue-task:15:test", error="boom", retry_delay_seconds=1)
            with mock.patch.object(main, "issue_task_queue", queue), mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.retry_issue_task("issue-task:15:test"))

            self.assertEqual("ok", result["status"])
            self.assertEqual("queued", result["task"]["status"])

    def test_retry_issue_task_endpoint_rejects_task_from_wrong_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            pr_queue.enqueue(task_id="pr-task:15:test", target_number=15, source="polling", max_attempts=1)

            with mock.patch.object(main, "issue_task_queue", issue_queue), mock.patch.object(main, "state_store", state_store):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(main.retry_issue_task("pr-task:15:test"))

            self.assertEqual(404, ctx.exception.status_code)

    def test_implement_issue_endpoint_queues_explicit_task_and_resumes_paused_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.set_issue_paused(21, paused=True, reason="manual abort")
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            fake_gh = FakeGitHubClient(
                issues_by_number={21: {"number": 21, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                result = asyncio.run(main.implement_issue(21))

            self.assertEqual("ok", result["status"])
            self.assertTrue(result["created"])
            self.assertTrue(result["resumed"])
            self.assertFalse(state_store.is_issue_paused(21))
            task_id = result["task"]["task_id"]
            task = state_store.get_task(task_id)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual("implement", task["command"])
            self.assertEqual(True, task["payload"]["explicit_command"])

    def test_implement_issue_endpoint_allows_requeueing_same_issue_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            fake_gh = FakeGitHubClient(
                issues_by_number={21: {"number": 21, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                first = asyncio.run(main.implement_issue(21))
                state_store.mark_task_succeeded(first["task"]["task_id"], result_summary="done")
                second = asyncio.run(main.implement_issue(21))

            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["task"]["task_id"], second["task"]["task_id"])
            self.assertEqual("queued", state_store.get_task(first["task"]["task_id"])["status"])

    def test_implement_issue_endpoint_reuses_running_task_for_same_issue_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            fake_gh = FakeGitHubClient(
                issues_by_number={21: {"number": 21, "state": "open", "updated_at": "2026-03-24T00:00:00Z"}}
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                first = asyncio.run(main.implement_issue(21))
                claimed = queue.claim_next()
                self.assertIsNotNone(claimed)
                second = asyncio.run(main.implement_issue(21))

            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["task"]["task_id"], second["task"]["task_id"])
            self.assertEqual(1, len(state_store.list_tasks(task_type="issue_execution")))

    def test_implement_issue_endpoint_rejects_closed_issue(self):
        fake_gh = FakeGitHubClient(
            issues_by_number={22: {"number": 22, "state": "closed", "updated_at": "2026-03-24T00:00:00Z"}}
        )

        with mock.patch.object(main, "gh", fake_gh), mock.patch.object(
            main, "state_store", object()
        ), mock.patch.object(main, "issue_task_queue", object()), mock.patch.object(
            main, "config", SimpleNamespace(autocode=SimpleNamespace(issue_task_max_attempts=3))
        ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(main.implement_issue(22))

        self.assertEqual(409, ctx.exception.status_code)

    def test_recheck_pull_request_feedback_resets_cursor_and_schedules_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=41,
                branch_name="feature/pr-41",
                base_ref="main",
                head_sha="abc123",
                author_login="maintainer",
                pr_url="https://example.com/pulls/41",
                queue_source="author_allowlist",
            )
            state_store.update_tracked_pr_feedback_cursor(
                41,
                updated_at="2026-04-25T10:00:00Z",
                item_ids=[100],
            )
            queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            fake_gh = FakeGitHubClient(
                issues_by_number={
                    41: {
                        "number": 41,
                        "state": "open",
                        "updated_at": "2026-04-25T10:01:00Z",
                    }
                }
            )

            async def fake_schedule(pr, *, source: str, priority: str):
                self.assertEqual(41, pr["number"])
                self.assertEqual("api", source)
                self.assertEqual("urgent", priority)
                tracked = state_store.get_tracked_pr(41) or {}
                self.assertNotIn("last_feedback_updated_at", tracked)
                self.assertNotIn("last_feedback_item_ids", tracked)
                task, _ = queue.enqueue(
                    task_id="pr-task:41:recheck",
                    target_number=41,
                    source=source,
                    priority=priority,
                    command="review_feedback",
                    payload={"feedback_key": "pr:41:recheck"},
                )
                return "PR #41 已进入修复队列，等待后台任务消费 (feedback_items=1)", task

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "pr_task_queue", queue
            ), mock.patch.object(main, "controller", object()), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(pr_task_max_attempts=3))
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "_schedule_pull_request_repair", new=fake_schedule
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                result = asyncio.run(main.recheck_pull_request_feedback(41))

            self.assertEqual("ok", result["status"])
            self.assertEqual("pr-task:41:recheck", result["task"]["task_id"])
            self.assertEqual("queued", state_store.get_task("pr-task:41:recheck")["status"])

    def test_recheck_pull_request_feedback_rejects_closed_pr(self):
        fake_gh = FakeGitHubClient(
            issues_by_number={42: {"number": 42, "state": "closed", "updated_at": "2026-04-25T10:00:00Z"}}
        )
        with mock.patch.object(main, "gh", fake_gh), mock.patch.object(
            main, "state_store", object()
        ), mock.patch.object(main, "pr_task_queue", object()), mock.patch.object(
            main, "controller", object()
        ), mock.patch.object(main, "config", object()), mock.patch.object(
            main.asyncio, "to_thread", new=_run_inline
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(main.recheck_pull_request_feedback(42))

        self.assertEqual(409, ctx.exception.status_code)

    def test_handle_pull_request_event_enqueues_repair_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            task = SimpleNamespace(task_id="pr-task:31:test")
            with mock.patch.object(main, "controller", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(main, "pr_task_queue", pr_queue), mock.patch.object(
                main,
                "_schedule_pull_request_repair",
                new=mock.AsyncMock(return_value=("PR #31 已进入修复队列，等待后台任务消费", task)),
            ) as schedule:
                asyncio.run(main._handle_pull_request_event({"number": 31}, source="webhook", action="synchronize"))

            schedule.assert_awaited_once()

    def test_schedule_pull_request_repair_requeues_retry_waiting_task_for_new_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            pr_queue.enqueue(
                task_id="pr-task:41:old",
                target_number=41,
                source="polling",
                priority="normal",
                command="review_feedback",
                payload={"feedback_key": "pr:41:old"},
                max_attempts=3,
            )
            self.assertIsNotNone(pr_queue.claim_next())
            state_store.mark_task_failed("pr-task:41:old", error="boom", retry_delay_seconds=60)
            self.assertEqual("claimed", state_store.try_claim_feedback("pr:41:new"))
            candidate = SimpleNamespace(
                task_id="pr-task:41:new",
                pr_number=41,
                queue_source="webhook",
                feedback_key="pr:41:new",
                feedback_updated_at="2026-03-24T01:00:00Z",
                feedback_item_ids=[201],
                summary="queued",
            )
            fake_controller = SimpleNamespace(plan_pull_request_repair=lambda pr: ("queued", candidate))

            with mock.patch.object(main, "controller", fake_controller), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(pr_task_max_attempts=3))
            ), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                summary, task = asyncio.run(
                    main._schedule_pull_request_repair({"number": 41}, source="webhook", priority="urgent")
                )

            self.assertEqual("queued", summary)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual("pr-task:41:old", task.task_id)
            updated = state_store.get_task("pr-task:41:old")
            assert updated is not None
            self.assertEqual("queued", updated["status"])
            self.assertEqual("urgent", updated["priority"])
            self.assertEqual("webhook", updated["source"])
            self.assertEqual("pr:41:new", updated["payload"]["feedback_key"])
            self.assertEqual("", updated["last_error"])
            self.assertEqual(0, updated["attempt_count"])
            self.assertTrue(state_store.feedback_claim_owned_by("pr:41:new", owner="pr-task:41:old"))

    def test_schedule_pull_request_repair_keeps_feedback_claim_owned_by_enqueued_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            feedback_key = "pr:51:test"
            self.assertEqual("claimed", state_store.try_claim_feedback(feedback_key))
            candidate = SimpleNamespace(
                task_id="pr-task:51:test",
                pr_number=51,
                queue_source="webhook",
                feedback_key=feedback_key,
                feedback_updated_at="2026-03-24T02:00:00Z",
                feedback_item_ids=[401],
                summary="queued",
            )
            fake_controller = SimpleNamespace(plan_pull_request_repair=lambda pr: ("queued", candidate))

            with mock.patch.object(main, "controller", fake_controller), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(pr_task_max_attempts=3))
            ), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                summary, task = asyncio.run(
                    main._schedule_pull_request_repair({"number": 51}, source="webhook", priority="urgent")
                )

            self.assertEqual("queued", summary)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual("inflight", state_store.try_claim_feedback(feedback_key))
            self.assertTrue(state_store.feedback_claim_owned_by(feedback_key, owner=task.task_id))

    def test_schedule_pull_request_repair_creates_follow_up_task_when_old_one_is_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            pr_queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            pr_queue.enqueue(
                task_id="pr-task:42:old",
                target_number=42,
                source="polling",
                priority="normal",
                command="review_feedback",
                payload={"feedback_key": "pr:42:old"},
                max_attempts=3,
            )
            self.assertIsNotNone(pr_queue.claim_next())
            candidate = SimpleNamespace(
                task_id="pr-task:42:new",
                pr_number=42,
                queue_source="webhook",
                feedback_key="pr:42:new",
                feedback_updated_at="2026-03-24T02:00:00Z",
                feedback_item_ids=[301],
                summary="queued",
            )
            fake_controller = SimpleNamespace(plan_pull_request_repair=lambda pr: ("queued", candidate))

            with mock.patch.object(main, "controller", fake_controller), mock.patch.object(
                main, "pr_task_queue", pr_queue
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(pr_task_max_attempts=3))
            ), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                summary, task = asyncio.run(
                    main._schedule_pull_request_repair({"number": 42}, source="webhook", priority="urgent")
                )

            self.assertEqual("queued", summary)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual("pr-task:42:new", task.task_id)
            tasks = {item["task_id"]: item for item in state_store.list_tasks(task_type="pr_feedback_repair")}
            self.assertEqual("running", tasks["pr-task:42:old"]["status"])
            self.assertEqual("queued", tasks["pr-task:42:new"]["status"])
            self.assertEqual("pr:42:new", tasks["pr-task:42:new"]["payload"]["feedback_key"])

    def test_scan_pull_request_repairs_once_suppresses_unchanged_waiting_review_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=822,
                branch_name="feature/pr-822",
                base_ref="main",
                head_sha="abc123",
                author_login="maintainer",
                pr_url="https://example.com/pulls/822",
                queue_source="author_allowlist",
            )
            summary = "PR #822 暂无新的可执行检视意见，等待 reviewer 进一步反馈"
            state_store.mark_tracked_pr_status(
                822,
                status="waiting_review",
                summary=summary,
                review_state="commented",
            )

            def plan_pull_request_repair(pr):
                state_store.mark_tracked_pr_status(
                    int(pr.get("number", 0) or 0),
                    status="waiting_review",
                    summary=summary,
                    review_state="commented",
                )
                return summary, None

            fake_controller = SimpleNamespace(
                collect_tracked_pull_requests=lambda: [{"number": 822}],
                plan_pull_request_repair=plan_pull_request_repair,
                rebase_tracked_pr_if_needed=lambda pr: None,
            )
            fake_logger = mock.Mock()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(
                main, "pr_task_queue", object()
            ), mock.patch.object(
                main, "logger", fake_logger
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._scan_pull_request_repairs_once())

            self.assertEqual({"scanned": 1, "queued": 0, "skipped": 1, "rebased": 0}, stats)
            fake_logger.info.assert_any_call("pr-queue: 开始扫描 %d 个待跟踪 PR", 1)
            fake_logger.info.assert_any_call(
                "pr-queue: 扫描完成 total=%d queued=%d skipped_unchanged=%d", 1, 0, 1,
            )

    def test_scan_pull_request_repairs_once_still_rechecks_waiting_review_pr_after_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=900,
                branch_name="autocode/issue-880-bug",
                base_ref="main",
                head_sha="abc123",
                author_login="maintainer",
                pr_url="https://example.com/pulls/900",
                queue_source="autocode_branch",
                pr_updated_at="2026-03-28T01:57:32Z",
            )
            state_store.mark_tracked_pr_status(
                900,
                status="waiting_review",
                summary="PR #900 暂无新的可执行检视意见，等待 reviewer 进一步反馈",
                review_state="commented",
            )

            calls: list[int] = []

            def plan_pull_request_repair(pr):
                calls.append(int(pr.get("number", 0) or 0))
                state_store.mark_tracked_pr_status(
                    900,
                    status="review_feedback_pending",
                    summary="PR #900 已进入修复队列，等待后台任务消费 (feedback_items=1)",
                    review_state="commented",
                )
                return (
                    "PR #900 已进入修复队列，等待后台任务消费 (feedback_items=1)",
                    SimpleNamespace(
                        pr_number=900,
                        task_id="pr-task:pr:900:test",
                        queue_source="autocode_branch",
                        feedback_key="pr:900:test",
                        feedback_updated_at="2026-03-28T01:57:32Z",
                        feedback_item_ids=[100],
                    ),
                )

            fake_controller = SimpleNamespace(
                collect_tracked_pull_requests=lambda: [
                    {"number": 900, "updated_at": "2026-03-28T01:57:32Z"}
                ],
                plan_pull_request_repair=plan_pull_request_repair,
                rebase_tracked_pr_if_needed=lambda pr: None,
            )
            fake_logger = mock.Mock()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(
                main, "pr_task_queue", object()
            ), mock.patch.object(
                main, "logger", fake_logger
            ), mock.patch.object(
                main, "_enqueue_pull_request_repair_candidate", return_value=(SimpleNamespace(task_id="pr-task:pr:900:test", status="queued"), True)
            ), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._scan_pull_request_repairs_once())

            self.assertEqual([900], calls)
            self.assertEqual({"scanned": 1, "queued": 1, "skipped": 0, "rebased": 0}, stats)

    def test_drain_pr_task_queue_executes_task_and_marks_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            queue.enqueue(task_id="pr-task:33:test", target_number=33, source="polling", command="review_feedback")
            fake_gh = FakeGitHubClient(
                issues_by_number={
                    33: {
                        "number": 33,
                        "state": "open",
                        "updated_at": "2026-03-24T00:00:00Z",
                    }
                }
            )
            process_calls: list[tuple[bool, str, str]] = []

            def process_pull_request(pr, allow_repair, current_task_id="", current_feedback_key="", **kwargs):
                del pr, kwargs
                process_calls.append((allow_repair, current_task_id, current_feedback_key))
                return "PR #33 已根据 review 反馈更新分支"

            fake_controller = SimpleNamespace(process_pull_request=process_pull_request)
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "pr_task_queue", queue
            ), mock.patch.object(main, "gh", fake_gh), mock.patch.object(
                main, "controller", fake_controller
            ), mock.patch.object(main, "config", SimpleNamespace(autocode=SimpleNamespace(pr_queue_max_batch=1))), mock.patch.object(
                main.asyncio, "to_thread", new=_run_inline
            ):
                stats = asyncio.run(main._drain_pr_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("pr-task:33:test")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertIn("review 反馈", task["result_summary"])
            self.assertEqual([(True, "pr-task:33:test", "")], process_calls)

    def test_retry_pr_task_endpoint_requeues_dead_letter_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="pr_feedback_repair", owner="test")
            queue.enqueue(task_id="pr-task:34:test", target_number=34, source="polling", max_attempts=1)
            claimed = state_store.claim_next_task(task_type="pr_feedback_repair", owner="test")
            self.assertIsNotNone(claimed)
            state_store.mark_task_failed("pr-task:34:test", error="boom", retry_delay_seconds=1)
            with mock.patch.object(main, "pr_task_queue", queue), mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.retry_pr_task("pr-task:34:test"))

            self.assertEqual("ok", result["status"])
            self.assertEqual("queued", result["task"]["status"])

    def test_enqueue_plan_source_items_keeps_existing_duplicate_when_new_copy_is_appended(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] add tests\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )
            first_item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=first_item.key,
                source_name="backlog",
                source_path=first_item.source_path,
                line_number=first_item.line_number,
                title=first_item.title,
                issue_number=701,
                issue_url="https://example.com/issues/701",
            )
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            todo_path.write_text("- [ ] add tests\n- [ ] add tests\n", encoding="utf-8")

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(source_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(
                    main._enqueue_plan_source_items_once(
                        plan_source=backlog,
                        source_name="backlog",
                        priority="background",
                    )
                )

            self.assertEqual({"fetched": 2, "queued": 1}, stats)
            tasks = state_store.list_tasks(task_type="source_issue_create")
            self.assertEqual(1, len(tasks))
            self.assertEqual(2, tasks[0]["payload"]["line_number"])

    def test_enqueue_plan_source_items_requeues_synced_issue_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] add tests\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=item.key,
                source_name="backlog",
                source_path=item.source_path,
                line_number=item.line_number,
                title=item.title,
                issue_number=701,
                issue_url="https://example.com/issues/701",
            )
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            fake_gh = FakeGitHubClient(
                issues_by_number={
                    701: {
                        "number": 701,
                        "state": "open",
                        "updated_at": "2026-03-27T09:10:00Z",
                        "created_at": "2026-03-27T09:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                    }
                }
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    autocode=SimpleNamespace(
                        source_task_max_attempts=3,
                        auto_plan_on_issue_open=True,
                        issue_task_max_attempts=3,
                    )
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(
                    main._enqueue_plan_source_items_once(
                        plan_source=backlog,
                        source_name="backlog",
                        priority="background",
                    )
                )

            self.assertEqual({"fetched": 1, "queued": 0}, stats)
            self.assertEqual([], state_store.list_tasks(task_type="source_issue_create", statuses={"queued", "retry_waiting"}))
            issue_tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(issue_tasks))
            self.assertEqual(701, issue_tasks[0]["target_number"])
            self.assertEqual("source:backlog", issue_tasks[0]["source"])

    def test_enqueue_plan_source_items_revives_dead_letter_issue_task_for_synced_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] add tests\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=item.key,
                source_name="backlog",
                source_path=item.source_path,
                line_number=item.line_number,
                title=item.title,
                issue_number=701,
                issue_url="https://example.com/issues/701",
            )
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            issue_updated_at = "2026-03-27T09:10:00Z"
            task_id = main._build_issue_task_id(
                701,
                issue_updated_at,
                "source:backlog",
                trigger="source_issue_existing",
            )
            issue_queue.enqueue(
                task_id=task_id,
                target_number=701,
                source="source:backlog",
                command="issue_open",
                payload={"explicit_command": False, "issue_updated_at": issue_updated_at},
                max_attempts=1,
            )
            self.assertIsNotNone(issue_queue.claim_next())
            issue_queue.mark_failed(task_id, error="boom", retry_delay_seconds=0)
            fake_gh = FakeGitHubClient(
                issues_by_number={
                    701: {
                        "number": 701,
                        "state": "open",
                        "updated_at": issue_updated_at,
                        "created_at": "2026-03-27T09:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                    }
                }
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    autocode=SimpleNamespace(
                        source_task_max_attempts=3,
                        auto_plan_on_issue_open=True,
                        issue_task_max_attempts=3,
                    )
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline), mock.patch.dict(
                main._source_issue_check_cache, clear=True
            ):
                stats = asyncio.run(
                    main._enqueue_plan_source_items_once(
                        plan_source=backlog,
                        source_name="backlog",
                        priority="background",
                    )
                )

            self.assertEqual({"fetched": 1, "queued": 0}, stats)
            issue_task = state_store.get_task(task_id)
            assert issue_task is not None
            self.assertEqual("queued", issue_task["status"])
            self.assertEqual(0, issue_task["attempt_count"])
            self.assertEqual("", issue_task["last_error"])

    def test_drain_source_task_queue_creates_issue_and_marks_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            queue.enqueue(
                task_id="source-task:test",
                target_number=0,
                source="plan",
                command="create_issue",
                payload={
                    "item_key": "plan:1",
                    "title": "[Plan] add coverage",
                    "body": "body",
                    "source_path": "PLAN.md",
                    "labels": ["autocode", "plan"],
                },
            )
            writer = FakeGitHubWriter()
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        auto_plan_on_issue_open=True,
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    )
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("source-task:test")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertTrue(state_store.has_backlog_item("plan:1"))
            self.assertEqual(1, len(writer.created_issues))
            issue_tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(issue_tasks))
            self.assertEqual("source:plan", issue_tasks[0]["source"])
            self.assertEqual(702, issue_tasks[0]["target_number"])

    def test_drain_source_task_queue_enqueues_created_issue_even_for_bot_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            source_queue.enqueue(
                task_id="source-task:bot",
                target_number=0,
                source="idle_scan",
                command="create_issue",
                payload={
                    "item_key": "idle:1",
                    "title": "[Idle] bot issue",
                    "body": "body",
                    "source_path": "src/app.py",
                    "labels": ["autocode", "idle-scan"],
                },
            )
            writer = SimpleNamespace(
                create_issue=lambda **kwargs: {
                    "number": 999,
                    "html_url": "https://example.com/issues/999",
                    "state": "open",
                    "updated_at": "2026-03-27T09:00:00Z",
                    "created_at": "2026-03-27T09:00:00Z",
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                }
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        auto_plan_on_issue_open=True,
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            issue_tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(issue_tasks))
            self.assertEqual("source:idle_scan", issue_tasks[0]["source"])
            self.assertEqual(999, issue_tasks[0]["target_number"])

    def test_drain_source_task_queue_requeues_issue_task_when_issue_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            source_queue.enqueue(
                task_id="source-task:existing",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:existing",
                    "title": "[TODO] existing issue",
                    "body": "body",
                    "source_path": "TODO.md",
                    "line_number": 7,
                    "labels": ["autocode", "todo"],
                },
            )
            state_store.mark_backlog_item_synced(
                item_key="backlog:existing",
                source_name="backlog",
                source_path="TODO.md",
                line_number=7,
                title="[TODO] existing issue",
                issue_number=812,
                issue_url="https://example.com/issues/812",
            )
            fake_gh = FakeGitHubClient(
                issues_by_number={
                    812: {
                        "number": 812,
                        "state": "open",
                        "updated_at": "2026-03-27T09:10:00Z",
                        "created_at": "2026-03-27T09:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                    }
                }
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main, "gh_writer", FakeGitHubWriter()
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        auto_plan_on_issue_open=True,
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            source_task = state_store.get_task("source-task:existing")
            assert source_task is not None
            self.assertEqual("done", source_task["status"])
            issue_tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(issue_tasks))
            self.assertEqual(812, issue_tasks[0]["target_number"])
            self.assertEqual("source:backlog", issue_tasks[0]["source"])

    def test_drain_source_task_queue_prefers_original_line_when_duplicate_key_is_stolen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] add tests\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )
            original_item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:duplicate-drift",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": original_item.key,
                    "title": original_item.title,
                    "body": original_item.body,
                    "source_path": original_item.source_path,
                    "line_number": original_item.line_number,
                    "labels": original_item.labels,
                },
            )
            writer = FakeGitHubWriter()
            todo_path.write_text("- [ ] add tests\n- [ ] add tests\n", encoding="utf-8")

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            record = state_store.get_backlog_item(original_item.key)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(1, record["line_number"])

    def test_drain_source_task_queue_refreshes_latest_source_payload_before_creating_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:freshen",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:1",
                    "title": "[Backlog] old title",
                    "body": "old body",
                    "source_path": "TODO.md",
                    "labels": ["autocode", "backlog"],
                },
            )
            latest_item = BacklogItem(
                key="backlog:1",
                title="[Backlog] new title",
                body="new body",
                source_path="TODO.md",
                line_number=9,
                labels=["autocode", "backlog", "refreshed"],
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main, "backlog_source", FakeBacklogSource([latest_item])
            ), mock.patch.object(
                main, "controller", None
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                    )
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual("[Bug] new title", writer.created_issues[0]["title"])
            self.assertEqual("这个问题来自待办梳理，建议补充触发条件、影响范围和验收标准后推进处理。", writer.created_issues[0]["body"])
            self.assertEqual(["autocode", "bug"], writer.created_issues[0]["labels"])
            backlog_record = state_store.get_backlog_item("backlog:1")
            self.assertIsNotNone(backlog_record)
            assert backlog_record is not None
            self.assertEqual("[Backlog] new title", backlog_record["title"])
            self.assertEqual("[Bug] new title", backlog_record["issue_title"])

    def test_drain_source_task_queue_uses_cli_draft_before_creating_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:cli-draft",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:cli",
                    "title": "[TODO] old draft",
                    "body": "raw body",
                    "source_path": "TODO.md",
                    "labels": ["autocode", "todo"],
                },
            )
            writer = FakeGitHubWriter()
            planner = FakePlanner(
                json.dumps(
                    {
                        "action": "create",
                        "title": "收敛未映射的 Actions 配置项",
                        "body": "当前有一部分 Actions 配置项没有完成映射，容易导致运行参数和预期不一致。\n\n建议先梳理缺失项，再补齐配置入口和校验。",
                        "labels": ["autocode", "enhancement"],
                        "reason": "rewrite",
                    },
                    ensure_ascii=False,
                )
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "controller", SimpleNamespace(planner=planner)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual("[Feature] 收敛未映射的 Actions 配置项", writer.created_issues[0]["title"])
            self.assertIn("没有完成映射", writer.created_issues[0]["body"])
            self.assertEqual(["autocode", "enhancement"], writer.created_issues[0]["labels"])
            self.assertTrue(planner.prompts)

    def test_drain_source_task_queue_skips_non_actionable_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:skip",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:skip",
                    "title": "[TODO] 这个问题不成立",
                    "body": "raw body",
                    "source_path": "TODO.md",
                    "line_number": 65,
                    "labels": ["autocode", "todo"],
                },
            )
            writer = FakeGitHubWriter()
            planner = FakePlanner(
                json.dumps(
                    {
                        "action": "skip",
                        "title": "",
                        "body": "",
                        "labels": ["autocode", "invalid"],
                        "reason": "这是审计反证结论，不应单独创建 issue",
                    },
                    ensure_ascii=False,
                )
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "controller", SimpleNamespace(planner=planner)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual([], writer.created_issues)
            record = state_store.get_backlog_item("backlog:skip")
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual("skipped", record["sync_status"])
            self.assertIn("不应单独创建 issue", record["sync_reason"])

    def test_draft_source_issue_payload_fallback_infers_type_label(self):
        draft = main._draft_source_issue_payload(
            source_name="backlog",
            item_key="backlog:doc",
            raw_title="简化readme，当前的readme内容太多了",
            raw_body="raw body",
            raw_labels=["autocode", "todo"],
        )

        self.assertEqual("create", draft["action"])
        self.assertEqual("[Docs] 精简 README 结构并下沉细节说明到专题文档", draft["title"])
        self.assertEqual(["autocode", "documentation"], draft["labels"])

    def test_draft_source_issue_payload_prefixes_vague_cli_title(self):
        planner = FakePlanner(
            json.dumps(
                {
                    "action": "create",
                    "title": "优化一下",
                    "body": "当前产品在部分交互或配置流程中会直接暴露 LiteLLM 的概念和实现细节，普通用户需要理解底层代理层才能完成使用或排查问题。",
                    "labels": ["autocode", "enhancement"],
                    "reason": "rewrite",
                },
                ensure_ascii=False,
            )
        )

        with mock.patch.object(main, "controller", SimpleNamespace(planner=planner)):
            draft = main._draft_source_issue_payload(
                source_name="backlog",
                item_key="backlog:feature",
                raw_title="隐藏 LiteLLM 实现细节，简化用户侧模型接入体验",
                raw_body="raw body",
                raw_labels=["autocode", "todo"],
            )

        self.assertEqual("[Feature] 优化模型接入体验，避免用户直接感知 LiteLLM", draft["title"])
        self.assertEqual(["autocode", "enhancement"], draft["labels"])

    def test_draft_source_issue_payload_sanitizes_internal_paths_from_body(self):
        planner = FakePlanner(
            json.dumps(
                {
                    "action": "create",
                    "title": "补齐 Actions 配置映射并增加缺失项校验",
                    "body": "问题出在 `templates/TODO.md` 的同步条目和 `/home/user/projects/demo/app.py:12` 相关逻辑。\n\n来源：`templates/TODO.md:8`\n建议梳理配置入口并补齐映射。",
                    "labels": ["autocode", "enhancement"],
                    "reason": "rewrite",
                },
                ensure_ascii=False,
            )
        )

        with mock.patch.object(main, "controller", SimpleNamespace(planner=planner)):
            draft = main._draft_source_issue_payload(
                source_name="backlog",
                item_key="backlog:sanitize",
                raw_title="Actions 配置映射缺失",
                raw_body="raw body",
                raw_labels=["autocode", "todo"],
            )

        self.assertEqual("[Feature] 补齐 Actions 配置映射并增加缺失项校验", draft["title"])
        self.assertNotIn("TODO.md", draft["body"])
        self.assertNotIn("/home/user/projects", draft["body"])
        self.assertNotIn("来源：", draft["body"])
        self.assertIn("相关代码位置", draft["body"])

    def test_draft_source_issue_payload_forces_security_gap_into_fix_bug(self):
        planner = FakePlanner(
            json.dumps(
                {
                    "action": "create",
                    "title": "为 Discord Webhook 请求增加签名验证",
                    "body": "当前 Discord Webhook 请求处理流程缺少签名验证，无法可靠确认请求是否来自合法来源，存在伪造请求或重放请求被误处理的风险。",
                    "labels": ["autocode", "enhancement"],
                    "reason": "rewrite",
                },
                ensure_ascii=False,
            )
        )

        with mock.patch.object(main, "controller", SimpleNamespace(planner=planner)):
            draft = main._draft_source_issue_payload(
                source_name="idle_scan",
                item_key="idle:security",
                raw_title="Discord Webhook 请求缺少签名验证",
                raw_body="当前 Discord Webhook 请求处理流程缺少签名验证，存在伪造请求和重放风险。",
                raw_labels=["autocode"],
            )

        self.assertEqual("[Bug] 为 Discord Webhook 请求增加签名验证", draft["title"])
        self.assertEqual(["autocode", "bug"], draft["labels"])

    def test_draft_source_issue_payload_fallback_marks_security_gap_as_bug(self):
        draft = main._draft_source_issue_payload(
            source_name="idle_scan",
            item_key="idle:security-fallback",
            raw_title="Discord Webhook 请求缺少签名验证",
            raw_body="当前处理流程缺少签名验证，存在伪造请求和重放风险。",
            raw_labels=["autocode"],
        )

        self.assertEqual("create", draft["action"])
        self.assertTrue(str(draft["title"]).startswith("[Bug] "))
        self.assertEqual(["autocode", "bug"], draft["labels"])

    def test_synced_source_item_keys_matches_exact_item_key_even_if_public_issue_title_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] 简化readme，当前的readme内容太多了\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=item.key,
                source_name="backlog",
                source_path=item.source_path,
                line_number=item.line_number,
                title=item.title,
                issue_title="[Docs] 精简 README 结构并下沉细节说明到专题文档",
                issue_number=883,
                issue_url="https://example.com/issues/883",
            )

            with mock.patch.object(main, "state_store", state_store):
                matched = main._synced_source_item_keys([item], source_name="backlog", require_real_issue=False)

            self.assertEqual({item.key}, matched)

    def test_drain_source_task_queue_writes_issue_link_back_to_backlog_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] 简化readme，当前的readme内容太多了\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            source_queue.enqueue(
                task_id="source-task:backlog-writeback",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": item.labels,
                },
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "gh_writer", FakeGitHubWriter()
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        auto_plan_on_issue_open=True,
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            contents = todo_path.read_text(encoding="utf-8")
            self.assertIn("issue: [#702](https://example.com/issues/702)", contents)

    def test_enqueue_plan_source_items_uses_existing_backlog_issue_link_as_dedupe_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] 简化readme，当前的readme内容太多了\n"
                "  issue: [#883](https://github.com/octo-org/example-repo/issues/883)\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(source_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(
                    main._enqueue_plan_source_items_once(
                        plan_source=backlog,
                        source_name="backlog",
                        priority="background",
                    )
                )

            self.assertEqual({"fetched": 1, "queued": 0}, stats)
            record = state_store.get_backlog_item(item.key)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(883, record["issue_number"])
            self.assertEqual("created", record["sync_status"])
            self.assertEqual([], state_store.list_tasks(task_type="source_issue_create", statuses={"queued", "retry_waiting"}))

    def test_enqueue_plan_source_items_requeues_closed_linked_issue_for_recreation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] 简化readme，当前的readme内容太多了\n"
                "  issue: [#883](https://github.com/octo-org/example-repo/issues/883)\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=item.key,
                source_name="backlog",
                source_path=item.source_path,
                line_number=item.line_number,
                title=item.title,
                issue_title=item.title,
                issue_number=883,
                issue_url="https://github.com/octo-org/example-repo/issues/883",
                sync_status="created",
            )
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            gh = FakeGitHubClient(
                issues_by_number={
                    883: {
                        "number": 883,
                        "title": "[Docs] old",
                        "state": "closed",
                        "updated_at": "2026-03-27T09:53:36Z",
                        "created_at": "2026-03-27T09:53:36Z",
                        "user": {"login": "autocode", "type": "Bot"},
                    }
                }
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh", gh
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "config", SimpleNamespace(autocode=SimpleNamespace(source_task_max_attempts=3))
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline), mock.patch.dict(
                main._source_issue_check_cache, clear=True
            ):
                stats = asyncio.run(
                    main._enqueue_plan_source_items_once(
                        plan_source=backlog,
                        source_name="backlog",
                        priority="background",
                    )
                )

            self.assertEqual({"fetched": 1, "queued": 1}, stats)
            tasks = state_store.list_tasks(task_type="source_issue_create", statuses={"queued"})
            self.assertEqual(1, len(tasks))
            self.assertEqual(item.key, tasks[0]["payload"]["item_key"])

    def test_drain_source_task_queue_recreates_closed_backlog_issue_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] 简化readme，当前的readme内容太多了\n"
                "  issue: [#883](https://github.com/octo-org/example-repo/issues/883)\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id=main._build_source_task_id(item.key),
                target_number=0,
                source="backlog",
                priority="background",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": list(item.labels),
                },
                max_attempts=3,
                replace_existing=True,
            )
            gh = FakeGitHubClient(
                issues_by_number={
                    883: {
                        "number": 883,
                        "title": "[Docs] old",
                        "state": "closed",
                        "updated_at": "2026-03-27T09:53:36Z",
                        "created_at": "2026-03-27T09:53:36Z",
                        "user": {"login": "autocode", "type": "Bot"},
                    }
                },
                listed_issues=[],
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh", gh
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False, poll_interval_seconds=300),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        source_task_max_attempts=3,
                        max_open_autocode_issues=20,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual(1, len(writer.created_issues))
            contents = todo_path.read_text(encoding="utf-8")
            self.assertIn("issue: [#702](https://example.com/issues/702)", contents)

    def test_drain_source_task_queue_recreates_closed_synced_backlog_issue_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] 简化readme，当前的readme内容太多了\n"
                "  issue: [#883](https://github.com/octo-org/example-repo/issues/883)\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key=item.key,
                source_name="backlog",
                source_path=item.source_path,
                line_number=item.line_number,
                title=item.title,
                issue_title=item.title,
                issue_number=883,
                issue_url="https://github.com/octo-org/example-repo/issues/883",
                sync_status="created",
            )
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id=main._build_source_task_id(item.key),
                target_number=0,
                source="backlog",
                priority="background",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": list(item.labels),
                },
                max_attempts=3,
                replace_existing=True,
            )
            gh = FakeGitHubClient(
                issues_by_number={
                    883: {
                        "number": 883,
                        "title": "[Docs] old",
                        "state": "closed",
                        "updated_at": "2026-03-27T09:53:36Z",
                        "created_at": "2026-03-27T09:53:36Z",
                        "user": {"login": "autocode", "type": "Bot"},
                    }
                },
                listed_issues=[],
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh", gh
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False, poll_interval_seconds=300),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        source_task_max_attempts=3,
                        max_open_autocode_issues=20,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual(1, len(writer.created_issues))
            contents = todo_path.read_text(encoding="utf-8")
            self.assertIn("issue: [#702](https://example.com/issues/702)", contents)

    def test_drain_source_task_queue_reuses_matching_open_autocode_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] 单股推送模式会在工作线程里直接并发调用共享 NotificationService。\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            issue_queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            source_queue.enqueue(
                task_id="source-task:reuse-existing",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": item.labels,
                },
            )
            planner = FakePlanner(
                json.dumps(
                    {
                        "action": "create",
                        "title": "修复单股推送模式下共享通知服务的并发调用问题",
                        "body": "单股推送模式下，共享通知服务实例会被多个工作线程并发调用，可能引发状态串扰和重复发送。",
                        "labels": ["autocode", "bug"],
                        "reason": "rewrite",
                    },
                    ensure_ascii=False,
                )
            )
            fake_gh = FakeGitHubClient(
                listed_issues=[
                    {
                        "number": 876,
                        "title": "[Bug] 修复单股推送模式下共享通知服务的并发调用问题",
                        "state": "open",
                        "html_url": "https://github.com/octo-org/example-repo/issues/876",
                        "updated_at": "2026-03-27T10:00:00Z",
                        "created_at": "2026-03-27T10:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                        "labels": [{"name": "autocode"}, {"name": "bug"}],
                    }
                ]
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", issue_queue
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "controller", SimpleNamespace(planner=planner)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False, poll_interval_seconds=300),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                        auto_plan_on_issue_open=True,
                        max_open_autocode_issues=20,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual([], writer.created_issues)
            record = state_store.get_backlog_item(item.key)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(876, record["issue_number"])
            self.assertEqual("[Bug] 修复单股推送模式下共享通知服务的并发调用问题", record["issue_title"])
            self.assertIn("issue: [#876]", todo_path.read_text(encoding="utf-8"))
            issue_tasks = state_store.list_tasks(task_type="issue_execution")
            self.assertEqual(1, len(issue_tasks))
            self.assertEqual(876, issue_tasks[0]["target_number"])

    def test_drain_source_task_queue_defers_non_emergency_issue_when_autocode_issue_pool_is_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text("- [ ] 简化readme，当前的readme内容太多了\n", encoding="utf-8")
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            source_queue.enqueue(
                task_id="source-task:capacity",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": item.labels,
                },
            )
            planner = FakePlanner(
                json.dumps(
                    {
                        "action": "create",
                        "title": "精简 README 并聚焦入门、运行与部署信息",
                        "body": "当前 README 内容偏多，建议拆分为更清晰的入门和专题文档。",
                        "labels": ["autocode", "documentation"],
                        "reason": "rewrite",
                    },
                    ensure_ascii=False,
                )
            )
            fake_gh = FakeGitHubClient(
                listed_issues=[
                    {
                        "number": 800 + idx,
                        "title": f"[Bug] existing {idx}",
                        "state": "open",
                        "html_url": f"https://example.com/issues/{800 + idx}",
                        "updated_at": "2026-03-27T10:00:00Z",
                        "created_at": "2026-03-27T10:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                        "labels": [{"name": "autocode"}, {"name": "bug"}],
                    }
                    for idx in range(20)
                ]
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "controller", SimpleNamespace(planner=planner)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False, poll_interval_seconds=300),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        max_open_autocode_issues=20,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual([], writer.created_issues)
            task = state_store.get_task("source-task:capacity")
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual("retry_waiting", task["status"])
            self.assertIn("autocode issue 已达到 20 个", task["last_error"])

    def test_drain_source_task_queue_allows_emergency_bug_beyond_open_issue_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "# Findings\n\n"
                "## P0\n\n"
                "- [ ] 并发执行时复用同一批共享对象，但关键状态没有统一加锁。\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )
            item = backlog.collect_items()[0]
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            source_queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="source-test")
            source_queue.enqueue(
                task_id="source-task:emergency",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": item.key,
                    "title": item.title,
                    "body": item.body,
                    "source_path": item.source_path,
                    "line_number": item.line_number,
                    "labels": item.labels,
                },
            )
            planner = FakePlanner(
                json.dumps(
                    {
                        "action": "create",
                        "title": "修复并发执行时共享状态缺少统一加锁的问题",
                        "body": "这是 P0 级别的并发问题，可能导致结果串扰和写入污染。",
                        "labels": ["autocode", "bug"],
                        "reason": "rewrite",
                    },
                    ensure_ascii=False,
                )
            )
            fake_gh = FakeGitHubClient(
                listed_issues=[
                    {
                        "number": 800 + idx,
                        "title": f"[Bug] existing {idx}",
                        "state": "open",
                        "html_url": f"https://example.com/issues/{800 + idx}",
                        "updated_at": "2026-03-27T10:00:00Z",
                        "created_at": "2026-03-27T10:00:00Z",
                        "user": {"login": "maintainer", "type": "User"},
                        "labels": [{"name": "autocode"}, {"name": "bug"}],
                    }
                    for idx in range(20)
                ]
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", source_queue
            ), mock.patch.object(
                main, "issue_task_queue", TaskQueue(state_store=state_store, task_type="issue_execution", owner="issue-test")
            ), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main, "gh", fake_gh
            ), mock.patch.object(
                main, "gh_writer", writer
            ), mock.patch.object(
                main, "controller", SimpleNamespace(planner=planner)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False, poll_interval_seconds=300),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                        issue_task_max_attempts=3,
                        auto_plan_on_issue_open=True,
                        max_open_autocode_issues=20,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            self.assertEqual(1, len(writer.created_issues))

    def test_drain_source_task_queue_treats_dry_run_as_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:dry-run",
                target_number=0,
                source="plan",
                command="create_issue",
                payload={
                    "item_key": "plan:dry-run",
                    "title": "[Plan] dry run",
                    "body": "body",
                    "source_path": "PLAN.md",
                    "labels": ["autocode", "plan"],
                },
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(
                main, "gh_writer", SimpleNamespace(create_issue=lambda **kwargs: None)
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=True),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("source-task:dry-run")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("dry_run", task["result_summary"])
            self.assertTrue(state_store.has_backlog_item("plan:dry-run"))
            self.assertFalse(state_store.has_backlog_item("plan:dry-run", require_real_issue=True))

    def test_drain_source_task_queue_skips_first_attempt_when_source_item_has_disappeared(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:missing-first",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:missing-first",
                    "title": "[Backlog] stale item",
                    "body": "body",
                    "source_path": "TODO.md",
                    "labels": ["autocode", "backlog"],
                },
                max_attempts=3,
            )
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main, "backlog_source", FakeBacklogSource([])
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("source-task:missing-first")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("skipped_missing_source", task["result_summary"])
            self.assertEqual(0, len(writer.created_issues))
            self.assertFalse(state_store.has_backlog_item("backlog:missing-first"))

    def test_drain_source_task_queue_skips_retry_when_source_item_has_disappeared(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:stale",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": "backlog:stale",
                    "title": "[Backlog] stale item",
                    "body": "body",
                    "source_path": "TODO.md",
                    "labels": ["autocode", "backlog"],
                },
                max_attempts=3,
            )
            claimed = state_store.claim_next_task(task_type="source_issue_create", owner="test")
            self.assertIsNotNone(claimed)
            state_store.mark_task_failed("source-task:stale", error="boom", retry_delay_seconds=0)
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main, "backlog_source", FakeBacklogSource([])
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=10,
                        plan_max_items_per_cycle=10,
                        idle_scanner_max_items_per_cycle=5,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("source-task:stale")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("skipped_missing_source", task["result_summary"])
            self.assertEqual(0, len(writer.created_issues))
            self.assertFalse(state_store.has_backlog_item("backlog:stale"))

    def test_requeue_source_issue_execution_ignores_skipped_backlog_record_without_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            state_store.mark_backlog_item_synced(
                item_key="backlog:skip",
                source_name="backlog",
                source_path="TODO.md",
                line_number=8,
                title="内部排查备注",
                issue_number=0,
                issue_url="",
                sync_status="skipped",
                sync_reason="不应单独创建 issue",
            )

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "gh", FakeGitHubClient()
            ), mock.patch.object(main.logger, "warning") as warning_log:
                result = asyncio.run(
                    main._requeue_source_issue_execution_if_needed(
                        item_key="backlog:skip",
                        source_name="backlog",
                    )
                )

            self.assertEqual("skipped_non_actionable", result["reason"])
            warning_log.assert_not_called()

    def test_drain_source_task_queue_retry_checks_full_source_not_cycle_limited_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "TODO.md").write_text(
                "- [ ] first task\n"
                "- [ ] second task\n"
                "- [ ] third task\n",
                encoding="utf-8",
            )
            backlog = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=1,
            )
            item_key = backlog._build_item_key("TODO.md", "", "third task")
            state_store = AutoCodeStateStore(str(repo / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(
                task_id="source-task:truncated",
                target_number=0,
                source="backlog",
                command="create_issue",
                payload={
                    "item_key": item_key,
                    "title": "[Backlog] third task",
                    "body": "body",
                    "source_path": "TODO.md",
                    "labels": ["autocode", "backlog"],
                },
                max_attempts=3,
            )
            claimed = state_store.claim_next_task(task_type="source_issue_create", owner="test")
            self.assertIsNotNone(claimed)
            state_store.mark_task_failed("source-task:truncated", error="boom", retry_delay_seconds=0)
            writer = FakeGitHubWriter()

            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", queue
            ), mock.patch.object(main, "gh_writer", writer), mock.patch.object(
                main, "backlog_source", backlog
            ), mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(dry_run=False),
                    autocode=SimpleNamespace(
                        backlog_max_items_per_cycle=1,
                        plan_max_items_per_cycle=1,
                        idle_scanner_max_items_per_cycle=1,
                    ),
                ),
            ), mock.patch.object(main.asyncio, "to_thread", new=_run_inline):
                stats = asyncio.run(main._drain_source_task_queue_once())

            self.assertEqual({"claimed": 1, "processed": 1, "failed": 0}, stats)
            task = state_store.get_task("source-task:truncated")
            assert task is not None
            self.assertEqual("done", task["status"])
            self.assertEqual("issue:702", task["result_summary"])
            self.assertEqual(1, len(writer.created_issues))
            self.assertTrue(state_store.has_backlog_item(item_key))

    def test_retry_source_task_endpoint_requeues_dead_letter_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="source_issue_create", owner="test")
            queue.enqueue(task_id="source-task:test", target_number=0, source="plan", max_attempts=1)
            claimed = state_store.claim_next_task(task_type="source_issue_create", owner="test")
            self.assertIsNotNone(claimed)
            state_store.mark_task_failed("source-task:test", error="boom", retry_delay_seconds=1)
            with mock.patch.object(main, "source_task_queue", queue), mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.retry_source_task("source-task:test"))

            self.assertEqual("ok", result["status"])
            self.assertEqual("queued", result["task"]["status"])

    def test_retry_scheduler_task_endpoint_requeues_task_and_records_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            scheduler = TaskScheduler()

            async def handler():
                return None

            scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
            scheduler._tasks["polling-cycle"].last_status = "dead_letter"
            scheduler._tasks["polling-cycle"].next_run_at = None
            scheduler._tasks["polling-cycle"].consecutive_failures = 2

            with mock.patch.object(main, "scheduler", scheduler), mock.patch.object(main, "state_store", state_store):
                result = asyncio.run(main.retry_scheduler_task("polling-cycle"))

            self.assertEqual("ok", result["status"])
            self.assertEqual("queued", result["task"]["status"])
            self.assertEqual(0, result["task"]["consecutive_failures"])
            observability = state_store.observability_summary()
            self.assertEqual(1, observability["counters"]["scheduler.manual_retry"])
            self.assertEqual("scheduler_recovery", observability["recent_events"][-1]["category"])

    def test_retry_scheduler_task_endpoint_returns_404_for_unknown_task(self):
        scheduler = TaskScheduler()
        with mock.patch.object(main, "scheduler", scheduler):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(main.retry_scheduler_task("missing-task"))

        self.assertEqual(404, ctx.exception.status_code)

    def test_retry_scheduler_task_endpoint_returns_409_for_running_task(self):
        scheduler = TaskScheduler()

        async def handler():
            return None

        scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
        scheduler._tasks["polling-cycle"].lease = TaskLease(owner="test", acquired_at=utc_now_iso())
        with mock.patch.object(main, "scheduler", scheduler):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(main.retry_scheduler_task("polling-cycle"))

        self.assertEqual(409, ctx.exception.status_code)

    def test_run_polling_cycle_records_success_observability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main,
                "_poll_once",
                new=mock.AsyncMock(
                    return_value={
                        "plans": {"fetched": 1, "queued": 0},
                        "backlog": {"fetched": 0, "queued": 0},
                        "issues": {"fetched": 2, "eligible": 1, "processed": 1},
                    }
                ),
            ):
                asyncio.run(main._run_polling_cycle())

            observability = state_store.observability_summary()
            cycles = {item["name"]: item for item in observability["cycles"]}
            self.assertEqual("succeeded", cycles["polling-cycle"]["last_status"])
            self.assertEqual(1, cycles["polling-cycle"]["success_count"])
            self.assertEqual(1, cycles["polling-cycle"]["last_stats"]["plans"]["fetched"])

    def test_poll_once_continues_when_plan_source_sync_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(poll_items_per_page=10),
                    autocode=SimpleNamespace(enable_plan_source=True, enable_backlog_source=False),
                ),
            ), mock.patch.object(main, "gh", object()), mock.patch.object(main, "gh_writer", object()), mock.patch.object(
                main, "state_store", state_store
            ), mock.patch.object(
                main, "_sync_plan_files_once", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
            ), mock.patch.object(
                main, "_sync_backlog_once", new=mock.AsyncMock(return_value={"fetched": 0, "queued": 0})
            ), mock.patch.object(
                main, "_poll_issues_once", new=mock.AsyncMock(return_value={"fetched": 2, "eligible": 1, "processed": 1})
            ) as poll_issues, mock.patch.object(
                main,
                "_reconcile_open_issues_once",
                new=mock.AsyncMock(return_value={"fetched": 3, "eligible": 1, "processed": 1}),
            ) as reconcile_issues, mock.patch.object(
                main, "_poll_issue_comments_once", new=mock.AsyncMock(return_value={"fetched": 1, "eligible": 1, "processed": 1})
            ) as poll_comments:
                result = asyncio.run(main._poll_once())

            poll_issues.assert_awaited_once()
            reconcile_issues.assert_awaited_once()
            poll_comments.assert_awaited_once()
            self.assertEqual(1, result["plans"]["failed"])
            self.assertEqual(1, result["comments"]["processed"])
            self.assertEqual(1, result["reconcile"]["processed"])
            observability = state_store.observability_summary()
            self.assertEqual("source_sync", observability["recent_events"][-1]["category"])
            self.assertEqual("error", observability["recent_events"][-1]["level"])

    def test_run_source_task_cycle_syncs_sources_when_not_polling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(
                main,
                "config",
                SimpleNamespace(
                    server=SimpleNamespace(event_source="webhook"),
                    autocode=SimpleNamespace(enable_plan_source=True, enable_backlog_source=True),
                ),
            ), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "source_task_queue", object()
            ), mock.patch.object(
                main, "gh_writer", object()
            ), mock.patch.object(
                main, "_sync_plan_files_once", new=mock.AsyncMock(return_value={"fetched": 1, "queued": 1})
            ) as sync_plan, mock.patch.object(
                main, "_sync_backlog_once", new=mock.AsyncMock(return_value={"fetched": 2, "queued": 1})
            ) as sync_backlog, mock.patch.object(
                main, "_drain_source_task_queue_once", new=mock.AsyncMock(return_value={"claimed": 1, "processed": 1, "failed": 0})
            ) as drain:
                asyncio.run(main._run_source_task_cycle())

            sync_plan.assert_awaited_once()
            sync_backlog.assert_awaited_once()
            drain.assert_awaited_once()
            observability = state_store.observability_summary()
            cycles = {item["name"]: item for item in observability["cycles"]}
            self.assertEqual("succeeded", cycles["source-task-cycle"]["last_status"])
            self.assertEqual(1, cycles["source-task-cycle"]["last_stats"]["plans"]["queued"])

    def test_run_polling_cycle_records_failure_observability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "_poll_once", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(main._run_polling_cycle())

            observability = state_store.observability_summary()
            cycles = {item["name"]: item for item in observability["cycles"]}
            self.assertEqual("failed", cycles["polling-cycle"]["last_status"])
            self.assertEqual("boom", cycles["polling-cycle"]["last_error"])
            self.assertEqual("scheduler_cycle", observability["recent_events"][-1]["category"])
            self.assertEqual("error", observability["recent_events"][-1]["level"])

    def test_queue_issue_execution_task_skips_new_version_when_terminal_outcome(self):
        """新版本（updated_at 变了）但上次执行已成功时，非 explicit_command 应直接跳过。

        这防止 autocode bot 评论更新 issue.updated_at 后引发的无限自循环：
        bot 评论 → updated_at 变化 → 当前版本不在 processed_updated_ats → 重新入队 → bot 评论 → ...
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
            queue = TaskQueue(state_store=state_store, task_type="issue_execution", owner="test")
            issue_v1 = {
                "number": 983,
                "state": "open",
                "updated_at": "2026-04-01T10:00:00Z",
                "user": {"login": "reporter", "type": "User"},
            }
            issue_v2 = {
                **issue_v1,
                "updated_at": "2026-04-01T10:05:00Z",  # changed by bot comment
            }
            with mock.patch.object(main, "config", SimpleNamespace(
                github=SimpleNamespace(owner="repoowner"),
                autocode=SimpleNamespace(
                    auto_plan_on_issue_open=True,
                    issue_task_max_attempts=3,
                    allow_feature_on_comment=True,
                ),
            )), mock.patch.object(main, "state_store", state_store), mock.patch.object(
                main, "issue_task_queue", queue
            ):
                # Simulate v1 processed with terminal outcome "answered"
                state_store.mark_issue_plan(
                    issue_key="key-v1",
                    issue_number=983,
                    task_type="answer",
                    updated_at="2026-04-01T10:00:00Z",
                    summary="answered",
                    last_outcome="answered",
                )

                # v2 (new version due to bot comment) — should be skipped
                result = main._queue_issue_execution_task(
                    issue_v2,
                    source="reconcile",
                    trigger="issue_reconcile",
                    explicit_command=False,
                )
                self.assertEqual("terminal_outcome_new_version", result.get("reason"), result)
                self.assertFalse(result.get("queued"), result)

                # explicit_command=True should still be allowed (human intent)
                result_explicit = main._queue_issue_execution_task(
                    issue_v2,
                    source="webhook",
                    trigger="issue_implement_command",
                    explicit_command=True,
                )
                self.assertTrue(result_explicit.get("queued"), result_explicit)


if __name__ == "__main__":
    unittest.main()
