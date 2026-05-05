import asyncio
import hashlib
import hmac
import json
import types
import unittest
from unittest import mock

from starlette.requests import Request

import main


def _build_webhook_request(payload: dict, *, secret: str, event_type: str) -> Request:
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


class WebhookFilterTests(unittest.TestCase):
    def setUp(self):
        self.original_agent = main.agent
        self.original_gh = main.gh
        self.original_config = main.config
        self.original_self_login = main.self_login

    def tearDown(self):
        main.agent = self.original_agent
        main.gh = self.original_gh
        main.config = self.original_config
        main.self_login = self.original_self_login

    def test_issue_edited_from_bot_sender_is_ignored_before_dispatch(self):
        secret = "topsecret"
        request = _build_webhook_request(
            {
                "action": "edited",
                "issue": {
                    "number": 41,
                    "title": "Issue title",
                    "body": "Issue body",
                    "user": {"login": "alice", "type": "User"},
                },
                "sender": {"login": "helper-bot[bot]", "type": "Bot"},
            },
            secret=secret,
            event_type="issues",
        )

        main.agent = object()
        main.gh = object()
        main.self_login = "openreview-bot"
        main.config = types.SimpleNamespace(
            server=types.SimpleNamespace(event_source="webhook"),
            github=types.SimpleNamespace(webhook_secret=secret),
        )

        with mock.patch.object(main, "_handle_issue_event", new=mock.AsyncMock()) as handler:
            response = asyncio.run(main.github_webhook(request))

        self.assertEqual({"status": "ok"}, response)
        handler.assert_not_awaited()

    def test_issue_comment_from_self_login_with_marker_is_ignored(self):
        payload = {
            "issue": {
                "number": 53,
                "title": "Issue title",
                "body": "Issue body",
                "user": {"login": "alice", "type": "User"},
            },
            "comment": {
                "id": 802,
                "body": "自动回复\n\n<!-- openreview:idempotency:issue:abc -->",
                "user": {"login": "zhulinsen", "type": "User"},
            },
        }

        main.self_login = "zhulinsen"

        with mock.patch.object(main, "_process_issue", new=mock.AsyncMock()) as process_issue, mock.patch.object(
            main, "_process_pr", new=mock.AsyncMock()
        ) as process_pr:
            asyncio.run(main._handle_issue_comment_event(payload))

        process_issue.assert_not_awaited()
        process_pr.assert_not_awaited()

    def test_pr_edited_from_bot_sender_is_ignored_before_dispatch(self):
        secret = "topsecret"
        request = _build_webhook_request(
            {
                "action": "edited",
                "pull_request": {
                    "number": 42,
                    "title": "PR title",
                    "body": "PR body",
                    "user": {"login": "alice", "type": "User"},
                },
                "sender": {"login": "helper-bot[bot]", "type": "Bot"},
            },
            secret=secret,
            event_type="pull_request",
        )

        main.agent = object()
        main.gh = object()
        main.self_login = "openreview-bot"
        main.config = types.SimpleNamespace(
            server=types.SimpleNamespace(event_source="webhook"),
            github=types.SimpleNamespace(webhook_secret=secret),
        )

        with mock.patch.object(main, "_handle_pr_event", new=mock.AsyncMock()) as handler:
            response = asyncio.run(main.github_webhook(request))

        self.assertEqual({"status": "ok"}, response)
        handler.assert_not_awaited()

    def test_issue_comment_slash_command_is_ignored(self):
        payload = {
            "issue": {
                "number": 52,
                "title": "Issue title",
                "body": "Issue body",
                "user": {"login": "alice", "type": "User"},
            },
            "comment": {
                "id": 801,
                "body": "/implement",
                "user": {"login": "alice", "type": "User"},
            },
        }

        with mock.patch.object(main, "_process_issue", new=mock.AsyncMock()) as process_issue, mock.patch.object(
            main, "_process_pr", new=mock.AsyncMock()
        ) as process_pr:
            asyncio.run(main._handle_issue_comment_event(payload))

        process_issue.assert_not_awaited()
        process_pr.assert_not_awaited()
