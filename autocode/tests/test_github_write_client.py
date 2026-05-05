import unittest

from agent.github_write_client import GitHubWriteClient


class _FakeResponse:
    def __init__(self, payload: dict, *, raw_json: dict | None = None):
        self.payload = payload
        self.raw_json = raw_json

    def raise_for_status(self):
        return None

    def json(self) -> dict:
        if self.raw_json is not None:
            return dict(self.raw_json)
        return {"number": 1, "html_url": "https://example.com/pr/1", "payload": self.payload}


class _RecordingHttpClient:
    def __init__(self):
        self.posts: list[tuple[str, dict]] = []
        self.patches: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    def post(self, url: str, json: dict):
        self.posts.append((url, dict(json)))
        if url.endswith("/graphql"):
            variables = dict(json.get("variables", {}) or {})
            pull_request_id = str(variables.get("pullRequestId", "") or "")
            pr_number = int(pull_request_id.rsplit("_", 1)[-1])
            return _FakeResponse(
                dict(json),
                raw_json={
                    "data": {
                        "markPullRequestReadyForReview": {
                            "pullRequest": {
                                "number": pr_number,
                                "url": f"https://example.com/pr/{pr_number}",
                                "isDraft": False,
                            }
                        }
                    }
                },
            )
        return _FakeResponse(dict(json))

    def patch(self, url: str, json: dict):
        self.patches.append((url, dict(json)))
        return _FakeResponse(dict(json))

    def get(self, url: str):
        self.gets.append(url)
        if url.endswith("/user"):
            return _FakeResponse({}, raw_json={"login": "octocat"})
        if "/pulls/" in url:
            pr_number = int(url.rstrip("/").rsplit("/", 1)[-1])
            return _FakeResponse({}, raw_json={"number": pr_number, "node_id": f"PR_node_{pr_number}"})
        return _FakeResponse({}, raw_json={"login": "octocat"})

    def close(self):
        return None


class GitHubWriteClientTests(unittest.TestCase):
    def test_create_pull_request_defaults_to_ready_pr(self):
        client = GitHubWriteClient(
            token="token",
            owner="demo",
            repo="repo",
            dry_run=False,
            force_draft_pr=False,
            forbid_main_base_branch=False,
        )
        recording_client = _RecordingHttpClient()
        client._client = recording_client

        result = client.create_pull_request(
            title="Fix issue",
            body="body",
            head="autocode/issue-1-fix",
            base="main",
            issue_number=1,
        )

        self.assertEqual(1, len(recording_client.posts))
        self.assertFalse(recording_client.posts[0][1]["draft"])
        self.assertEqual("main", recording_client.posts[0][1]["base"])
        self.assertEqual(1, result["number"])

    def test_create_issue_passes_labels(self):
        client = GitHubWriteClient(
            token="token",
            owner="demo",
            repo="repo",
            dry_run=False,
            force_draft_pr=False,
            forbid_main_base_branch=False,
        )
        recording_client = _RecordingHttpClient()
        client._client = recording_client

        result = client.create_issue(
            title="Backlog item",
            body="body",
            labels=["autocode", "backlog"],
        )

        self.assertEqual(1, len(recording_client.posts))
        self.assertEqual(["autocode", "backlog"], recording_client.posts[0][1]["labels"])
        self.assertEqual(1, result["number"])

    def test_mark_ready_for_review_uses_graphql_mutation(self):
        client = GitHubWriteClient(
            token="token",
            owner="demo",
            repo="repo",
            dry_run=False,
            force_draft_pr=True,
            forbid_main_base_branch=False,
        )
        recording_client = _RecordingHttpClient()
        client._client = recording_client

        result = client.mark_ready_for_review(12)

        self.assertEqual(1, len(recording_client.posts))
        self.assertTrue(recording_client.posts[0][0].endswith("/graphql"))
        self.assertIn("markPullRequestReadyForReview", recording_client.posts[0][1]["query"])
        self.assertEqual({"pullRequestId": "PR_node_12"}, recording_client.posts[0][1]["variables"])
        self.assertEqual(12, result["number"])
        self.assertFalse(result["draft"])
        self.assertEqual("PR_node_12", result["node_id"])

    def test_validate_base_branch_blocks_default_branch_when_strict_mode_enabled(self):
        client = GitHubWriteClient(
            token="token",
            owner="demo",
            repo="repo",
            dry_run=False,
            force_draft_pr=False,
            forbid_main_base_branch=True,
            default_branch="master",
        )

        with self.assertRaisesRegex(RuntimeError, "master"):
            client.create_pull_request(
                title="Fix issue",
                body="body",
                head="autocode/issue-1-fix",
                base="master",
                issue_number=1,
            )

    def test_get_authenticated_login_caches_viewer_login(self):
        client = GitHubWriteClient(
            token="token",
            owner="demo",
            repo="repo",
            dry_run=False,
            force_draft_pr=True,
            forbid_main_base_branch=False,
        )
        recording_client = _RecordingHttpClient()
        client._client = recording_client

        first = client.get_authenticated_login()
        second = client.get_authenticated_login()

        self.assertEqual("octocat", first)
        self.assertEqual("octocat", second)
        self.assertEqual(1, len(recording_client.gets))


if __name__ == "__main__":
    unittest.main()
