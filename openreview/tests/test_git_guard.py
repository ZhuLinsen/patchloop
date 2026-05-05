import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.git_guard import GitGuard


class GitGuardDiffSnapshotTests(unittest.TestCase):
    def test_build_pr_diff_snapshot_uses_local_merge_base_and_prioritizes_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            self._git(repo, "init", "-b", "main")
            self._git(repo, "config", "user.name", "OpenReview Test")
            self._git(repo, "config", "user.email", "openreview@example.com")

            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "docs").mkdir(parents=True, exist_ok=True)
            (repo / "src" / "app.py").write_text("print('base')\n", encoding="utf-8")
            (repo / "docs" / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
            (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "base")

            self._git(repo, "checkout", "-b", "feature/pr-review")
            (repo / "src" / "app.py").write_text("print('feature')\n", encoding="utf-8")
            (repo / "docs" / "CHANGELOG.md").write_text("# Changelog\n\n- Added local diff\n", encoding="utf-8")
            (repo / "package-lock.json").write_text(
                '{\n  "lockfileVersion": 3,\n  "packages": {\n    "": {\n      "name": "demo"\n    }\n  }\n}\n',
                encoding="utf-8",
            )
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "feature changes")

            guard = GitGuard(str(repo), default_branch="main")
            snapshot = guard.build_pr_diff_snapshot("main", max_excerpt_chars=450)

            self.assertEqual(snapshot.base_ref, "main")
            self.assertEqual(snapshot.compare_ref, "main")
            self.assertCountEqual(
                snapshot.changed_files,
                [
                    "docs/CHANGELOG.md",
                    "package-lock.json",
                    "src/app.py",
                ],
            )
            self.assertIn("docs/CHANGELOG.md", snapshot.diff_excerpt)
            self.assertIn("src/app.py", snapshot.diff_excerpt)
            self.assertNotIn("package-lock.json", snapshot.diff_excerpt)
            self.assertIn("3 files changed", snapshot.diff_stat)

    def _git(self, repo: Path, *args: str) -> str:
        repo.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class GitGuardCompareRefTests(unittest.TestCase):
    def test_resolve_compare_ref_fails_when_non_default_base_cannot_refresh(self):
        class _StubGuard(GitGuard):
            def __init__(self):
                pass

            default_branch = "main"

            def has_remote(self, remote: str) -> bool:
                return remote == "origin"

            def ref_exists(self, ref: str) -> bool:
                return ref in {"origin/release", "release"}

            def fetch(self, ref: str, remote: str = "origin") -> bool:
                return False

        guard = _StubGuard()
        self.assertEqual("", guard._resolve_compare_ref("release"))


if __name__ == "__main__":
    unittest.main()
