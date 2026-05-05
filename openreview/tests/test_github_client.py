import unittest

from agent.github_client import GitHubClient


class _RecentPaginationGitHubClient(GitHubClient):
    def __init__(self, pages: dict[int, tuple[list[dict], str]]):
        super().__init__("token", "demo", "repo")
        self.pages = pages
        self.calls: list[tuple[str, int, int]] = []

    def _get_list_page(self, url: str, params: dict | None = None) -> tuple[list[dict], str]:
        assert params is not None
        page = int(params["page"])
        per_page = int(params["per_page"])
        self.calls.append((url, page, per_page))
        items, link = self.pages.get(page, ([], ""))
        return list(items), link


class _PRFilesPaginationGitHubClient(GitHubClient):
    def __init__(self, pages: dict[int, list[dict]]):
        super().__init__("token", "demo", "repo")
        self.pages = pages
        self.calls: list[tuple[int, int]] = []

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        del url
        assert params is not None
        page = int(params["page"])
        per_page = int(params["per_page"])
        self.calls.append((page, per_page))
        return list(self.pages.get(page, []))


class GitHubClientRecentPaginationTests(unittest.TestCase):
    def test_get_pr_files_reads_all_pages(self):
        client = _PRFilesPaginationGitHubClient(
            {
                1: [{"filename": f"src/file_{index}.py"} for index in range(100)],
                2: [{"filename": "tests/test_file.py"}, {"filename": "README.md"}],
            }
        )
        self.addCleanup(client.close)

        files = client.get_pr_files(123)

        self.assertEqual(102, len(files))
        self.assertEqual("src/file_0.py", files[0])
        self.assertEqual("README.md", files[-1])
        self.assertEqual([(1, 100), (2, 100)], client.calls)

    def test_recent_issue_comments_use_last_page_hint_to_read_only_tail_pages(self):
        client = _RecentPaginationGitHubClient(
            {
                1: (
                    [{"id": 1}, {"id": 2}],
                    (
                        "<https://api.github.com/repos/demo/repo/issues/123/comments?per_page=2&page=2>; rel=\"next\", "
                        "<https://api.github.com/repos/demo/repo/issues/123/comments?per_page=2&page=5>; rel=\"last\""
                    ),
                ),
                4: ([{"id": 7}, {"id": 8}], ""),
                5: ([{"id": 9}], ""),
            }
        )
        self.addCleanup(client.close)

        recent = client.list_recent_issue_comments(123, limit=3, per_page=2)

        self.assertEqual([{"id": 7}, {"id": 8}, {"id": 9}], recent)
        self.assertEqual(
            [
                ("https://api.github.com/repos/demo/repo/issues/123/comments", 1, 2),
                ("https://api.github.com/repos/demo/repo/issues/123/comments", 4, 2),
                ("https://api.github.com/repos/demo/repo/issues/123/comments", 5, 2),
            ],
            client.calls,
        )

    def test_recent_pr_reviews_fallback_to_sequential_paging_without_last_page_hint(self):
        client = _RecentPaginationGitHubClient(
            {
                1: ([{"id": 1}, {"id": 2}], ""),
                2: ([{"id": 3}, {"id": 4}], ""),
                3: ([{"id": 5}], ""),
            }
        )
        self.addCleanup(client.close)

        recent = client.list_recent_pr_reviews(456, limit=3, per_page=2)

        self.assertEqual([{"id": 3}, {"id": 4}, {"id": 5}], recent)
        self.assertEqual(
            [
                ("https://api.github.com/repos/demo/repo/pulls/456/reviews", 1, 2),
                ("https://api.github.com/repos/demo/repo/pulls/456/reviews", 2, 2),
                ("https://api.github.com/repos/demo/repo/pulls/456/reviews", 3, 2),
            ],
            client.calls,
        )


if __name__ == "__main__":
    unittest.main()
