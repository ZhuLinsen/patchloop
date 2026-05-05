import unittest

from agent.discussion import (
    build_issue_reference_pr_query,
    build_issue_search_query,
    build_pr_search_query,
    collect_discussion_entries,
    detect_duplicate_issues,
    format_duplicate_issues_context,
    format_related_issues_context,
    format_related_prs_context,
    format_resolution_signals_context,
    pick_latest_external_activity,
)


class DiscussionHelpersTests(unittest.TestCase):
    def test_pick_latest_external_activity_ignores_self_and_bot(self):
        entries = collect_discussion_entries(
            issue_comments=[
                {
                    "id": 1,
                    "body": "机器人回复",
                    "created_at": "2026-03-25T09:00:00Z",
                    "updated_at": "2026-03-25T09:00:00Z",
                    "user": {"login": "openreview-bot", "type": "Bot"},
                },
                {
                    "id": 2,
                    "body": "我们已经按这个方向改了",
                    "created_at": "2026-03-25T10:00:00Z",
                    "updated_at": "2026-03-25T10:00:00Z",
                    "user": {"login": "maintainer", "type": "User"},
                },
                {
                    "id": 3,
                    "body": "最新补充：这个问题在 release 分支也存在",
                    "created_at": "2026-03-25T11:00:00Z",
                    "updated_at": "2026-03-25T11:00:00Z",
                    "user": {"login": "alice", "type": "User"},
                },
            ]
        )

        latest = pick_latest_external_activity(entries, ignored_logins={"maintainer"})

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual("alice", latest.author_login)
        self.assertEqual(3, latest.activity_id)

    def test_pick_latest_external_activity_ignores_only_markered_self_activity(self):
        entries = collect_discussion_entries(
            issue_comments=[
                {
                    "id": 1,
                    "body": "外部用户的较早评论",
                    "created_at": "2026-03-25T09:00:00Z",
                    "updated_at": "2026-03-25T09:00:00Z",
                    "user": {"login": "alice", "type": "User"},
                },
                {
                    "id": 2,
                    "body": "维护者本人补充了新的现象",
                    "created_at": "2026-03-25T10:00:00Z",
                    "updated_at": "2026-03-25T10:00:00Z",
                    "user": {"login": "zhulinsen", "type": "User"},
                },
                {
                    "id": 3,
                    "body": "自动回复\n\n<!-- openreview:idempotency:issue:abc -->",
                    "created_at": "2026-03-25T11:00:00Z",
                    "updated_at": "2026-03-25T11:00:00Z",
                    "user": {"login": "zhulinsen", "type": "User"},
                },
            ]
        )

        latest = pick_latest_external_activity(entries, ignored_logins={"zhulinsen"})

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual("zhulinsen", latest.author_login)
        self.assertEqual(2, latest.activity_id)

    def test_pick_latest_external_activity_ignores_autocode_marker_self_activity(self):
        entries = collect_discussion_entries(
            issue_comments=[
                {
                    "id": 1,
                    "body": "外部用户的较早评论",
                    "created_at": "2026-03-25T09:00:00Z",
                    "updated_at": "2026-03-25T09:00:00Z",
                    "user": {"login": "alice", "type": "User"},
                },
                {
                    "id": 2,
                    "body": "维护者本人补充了新的现象",
                    "created_at": "2026-03-25T10:00:00Z",
                    "updated_at": "2026-03-25T10:00:00Z",
                    "user": {"login": "zhulinsen", "type": "User"},
                },
                {
                    "id": 3,
                    "body": "autocode 自动回复\n\n<!-- autocode:issue-resolution:983:2026-04-02t14-00-00z:answered -->",
                    "created_at": "2026-03-25T11:00:00Z",
                    "updated_at": "2026-03-25T11:00:00Z",
                    "user": {"login": "zhulinsen", "type": "User"},
                },
            ]
        )

        latest = pick_latest_external_activity(entries, ignored_logins={"zhulinsen"})

        self.assertIsNotNone(latest)
        assert latest is not None
        # autocode marker comment should be ignored; the manual comment at id=2 is latest
        self.assertEqual("zhulinsen", latest.author_login)
        self.assertEqual(2, latest.activity_id)

    def test_related_issue_context_filters_current_issue(self):
        context = format_related_issues_context(
            [
                {
                    "number": 12,
                    "state": "closed",
                    "title": "login timeout after retry",
                    "body": "This was fixed by #20",
                    "updated_at": "2026-03-24T00:00:00Z",
                },
                {
                    "number": 18,
                    "state": "open",
                    "title": "current issue should be ignored",
                    "body": "current",
                    "updated_at": "2026-03-25T00:00:00Z",
                },
            ],
            current_issue_number=18,
        )

        self.assertIn("#12", context)
        self.assertNotIn("#18", context)

    def test_detect_duplicate_issues_prefers_high_confidence_matches(self):
        candidates = detect_duplicate_issues(
            "login timeout after retry on release branch",
            "release branch 上也会出现 login timeout after retry，错误栈完全一致",
            [
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
            current_issue_number=99,
        )

        self.assertEqual([12], [issue["number"] for issue in candidates])
        duplicate_context = format_duplicate_issues_context(candidates)
        self.assertIn("#12", duplicate_context)
        self.assertIn("重合线索", duplicate_context)

    def test_detect_duplicate_issues_ignores_generic_config_overlap(self):
        candidates = detect_duplicate_issues(
            "config path question for LOCAL_REPO_PATH",
            "想确认 LOCAL_REPO_PATH 到底在哪配置",
            [
                {
                    "number": 12,
                    "state": "closed",
                    "title": "config path question for PRIMARY_MODEL",
                    "body": "How to configure PRIMARY_MODEL?",
                    "updated_at": "2026-03-24T00:00:00Z",
                }
            ],
            current_issue_number=99,
        )

        self.assertEqual([], candidates)

    def test_issue_search_query_prefers_repo_scoped_terms(self):
        query = build_issue_search_query(
            "demo/repo",
            "登录超时 after retry",
            "release 分支也会出现相同报错",
        )

        self.assertIn("repo:demo/repo", query)
        self.assertIn("is:issue", query)
        self.assertTrue("登录超时" in query or "release" in query)

    def test_pr_search_query_and_reference_query_are_repo_scoped(self):
        search_query = build_pr_search_query(
            "demo/repo",
            "登录超时 after retry",
            "release 分支也会出现相同报错",
        )
        ref_query = build_issue_reference_pr_query("demo/repo", 123)

        self.assertIn("repo:demo/repo", search_query)
        self.assertIn("is:pr", search_query)
        self.assertEqual('repo:demo/repo is:pr "#123"', ref_query)

    def test_related_pr_context_and_resolution_signals_include_merge_state(self):
        prs = [
            {
                "number": 20,
                "state": "closed",
                "title": "fix login timeout",
                "body": "Fixes #12",
                "updated_at": "2026-03-24T00:00:00Z",
                "merged_at": "2026-03-24T01:00:00Z",
                "mergeable_state": "clean",
                "merge_commit_sha": "abcdef1234567890",
            },
            {
                "number": 21,
                "state": "open",
                "title": "investigate retry path",
                "body": "Still debugging",
                "updated_at": "2026-03-25T00:00:00Z",
                "mergeable_state": "dirty",
            },
        ]

        related_context = format_related_prs_context(prs)
        resolution_context = format_resolution_signals_context(prs)

        self.assertIn("PR #20 [merged]", related_context)
        self.assertIn("merged_sha: abcdef1", related_context)
        self.assertIn("mergeable_state: dirty", related_context)
        self.assertIn("已合并", resolution_context)
        self.assertIn("abcdef1", resolution_context)
        self.assertIn("仍在进行中", resolution_context)


if __name__ == "__main__":
    unittest.main()
