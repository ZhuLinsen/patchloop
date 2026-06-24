import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.execution_policy import ExecutionPolicyEngine
from agent.patch_inspector import PatchInspector
from config import AutoCodeConfig


def _run(cmd: list[str], cwd: Path):
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {completed.stderr or completed.stdout}")


class PatchInspectorTests(unittest.TestCase):
    def test_inspect_includes_untracked_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("print('base')\n", encoding="utf-8")
            _run(["git", "add", "app.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)

            tests_dir = repo / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_added.py").write_text("def test_added():\n    assert True\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=True)

            self.assertFalse(result.blocked)
            self.assertIn("tests/test_added.py", result.changed_files)
            self.assertTrue(result.has_test_changes)
            self.assertGreaterEqual(result.added_lines, 2)

    def test_inspect_blocks_when_git_diff_times_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            with mock.patch(
                "agent.patch_inspector.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=120),
            ):
                result = inspector.inspect(repo, require_test_changes=False)

            self.assertTrue(result.blocked)
            self.assertTrue(any("git diff 检查超时" in reason for reason in result.reasons))
            self.assertEqual([], result.changed_files)
            self.assertEqual(0, result.added_lines)
            self.assertEqual(0, result.deleted_lines)

    def test_inspect_excludes_docs_tests_and_env_examples_from_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("print('base')\n", encoding="utf-8")
            _run(["git", "add", "app.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)

            (repo / "app.py").write_text("print('patched')\n", encoding="utf-8")
            (repo / "README.md").write_text("# Notes\n" + ("line\n" * 120), encoding="utf-8")
            (repo / ".env.example").write_text("TOKEN=\nDEBUG=false\n", encoding="utf-8")
            tests_dir = repo / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_added.py").write_text("def test_added():\n    assert True\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=1,
                max_added_lines=5,
                max_deleted_lines=5,
            )

            result = inspector.inspect(repo, require_test_changes=True)

            self.assertFalse(result.blocked)
            self.assertIn(".env.example", result.changed_files)
            self.assertIn("README.md", result.changed_files)
            self.assertIn("tests/test_added.py", result.changed_files)
            self.assertTrue(result.has_test_changes)

    def test_dependency_manifest_bugfix_can_skip_test_file_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "requirements.txt").write_text("exchange-calendars>=4.5.0\n", encoding="utf-8")
            docs_dir = repo / "docs"
            docs_dir.mkdir()
            (docs_dir / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
            _run(["git", "add", "requirements.txt", "docs/CHANGELOG.md"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "requirements.txt").write_text("exchange-calendars>=4.13.0\n", encoding="utf-8")
            (docs_dir / "CHANGELOG.md").write_text(
                "# Changelog\n\n- Fix exchange calendar dependency.\n",
                encoding="utf-8",
            )

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=True)

            self.assertFalse(result.blocked)
            self.assertFalse(result.has_test_changes)
            self.assertIn("requirements.txt", result.changed_files)
            self.assertIn("docs/CHANGELOG.md", result.changed_files)

    def test_code_bugfix_still_requires_test_file_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            _run(["git", "add", "app.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=True)

            self.assertTrue(result.blocked)
            self.assertTrue(any("bug_fix 未包含测试改动" in reason for reason in result.reasons))

    def test_inspect_ignores_transient_pr_body_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("print('base')\n", encoding="utf-8")
            _run(["git", "add", "app.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)

            transient_dir = repo / ".github" / "pr-body"
            transient_dir.mkdir(parents=True, exist_ok=True)
            (transient_dir / "pr-12.md").write_text("temporary pr body\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=False)

            self.assertTrue(result.blocked)
            self.assertEqual([], result.changed_files)
            self.assertIn("未检测到可提交的代码改动", result.reasons)

    def test_inspect_ignores_transient_pr_description_drafts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("print('base')\n", encoding="utf-8")
            _run(["git", "add", "app.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)

            transient_dir = repo / ".github" / "pr-description-drafts"
            transient_dir.mkdir(parents=True, exist_ok=True)
            (transient_dir / "pr-12.md").write_text("temporary pr body draft\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=False)

            self.assertTrue(result.blocked)
            self.assertEqual([], result.changed_files)
            self.assertIn("未检测到可提交的代码改动", result.reasons)

    def test_code_plan_paths_are_preferred_not_hard_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "src").mkdir()
            (repo / "src" / "market_analyzer.py").write_text("VALUE = 1\n", encoding="utf-8")
            _run(["git", "add", "src/market_analyzer.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "src" / "market_analyzer.py").write_text("VALUE = 2\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=False, allowed_paths=["src/services/"])

            self.assertFalse(result.blocked)
            self.assertIn("src/market_analyzer.py", result.changed_files)

    def test_docs_plan_paths_remain_strict_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            (repo / "README.md").write_text("# Base\n", encoding="utf-8")
            _run(["git", "add", "app.py", "README.md"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(repo, require_test_changes=False, allowed_paths=["README.md"])

            self.assertTrue(result.blocked)
            self.assertTrue(any("越界改动: app.py" in reason for reason in result.reasons))

    def test_documentation_only_blocks_code_and_test_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "README.md").write_text("# Base\n", encoding="utf-8")
            (repo / "bot").mkdir()
            (repo / "bot" / "dispatcher.py").write_text("VALUE = 1\n", encoding="utf-8")
            _run(["git", "add", "README.md", "bot/dispatcher.py"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "README.md").write_text("# Updated\n", encoding="utf-8")
            (repo / "bot" / "dispatcher.py").write_text("VALUE = 2\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
            )

            result = inspector.inspect(
                repo,
                require_test_changes=False,
                allowed_paths=["README.md"],
                documentation_only=True,
            )

            self.assertTrue(result.blocked)
            self.assertTrue(any("文档任务禁止代码改动: bot/dispatcher.py" in reason for reason in result.reasons))

    def test_documentation_scope_extensions_are_configurable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            _run(["git", "init"], cwd=repo)
            _run(["git", "config", "user.email", "author@example.com"], cwd=repo)
            _run(["git", "config", "user.name", "Author"], cwd=repo)
            (repo / "guide.adoc").write_text("= Base\n", encoding="utf-8")
            _run(["git", "add", "guide.adoc"], cwd=repo)
            _run(["git", "commit", "-m", "init"], cwd=repo)
            (repo / "guide.adoc").write_text("= Updated\n", encoding="utf-8")

            inspector = PatchInspector(
                policy_engine=ExecutionPolicyEngine(AutoCodeConfig(enabled=True), default_branch="main"),
                max_changed_files=10,
                max_added_lines=50,
                max_deleted_lines=50,
                documentation_extensions=[".adoc"],
            )

            result = inspector.inspect(
                repo,
                require_test_changes=False,
                allowed_paths=["guide.adoc"],
                documentation_only=True,
            )

            self.assertFalse(result.blocked)
            self.assertIn("guide.adoc", result.changed_files)


if __name__ == "__main__":
    unittest.main()
