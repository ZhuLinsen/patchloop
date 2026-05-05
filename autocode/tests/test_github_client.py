import unittest

from agent.github_client import GitHubClient


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


class GitHubClientPRFilesTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
