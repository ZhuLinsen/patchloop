import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agent.workspace_manager import WorkspaceManager


def _run(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {completed.stderr or completed.stdout}")
    return completed.stdout.strip()


class WorkspaceManagerTests(unittest.TestCase):
    def _build_manager(
        self,
        repo_path: Path,
        worktree_root: Path,
        *,
        git_auth_login: str = "",
        git_auth_token: str = "",
    ) -> WorkspaceManager:
        return WorkspaceManager(
            repo_path=str(repo_path),
            worktree_root=str(worktree_root),
            publish_remote="origin",
            branch_prefix="autocode",
            default_branch="main",
            commit_user_name="AutoCode Bot",
            commit_user_email="autocode-bot@example.com",
            git_auth_login=git_auth_login,
            git_auth_token=git_auth_token,
        )

    def _init_remote_repos(self, root: Path) -> tuple[Path, Path, Path]:
        remote = root / "remote.git"
        seed = root / "seed"
        local = root / "local"
        _run(["git", "init", "--bare", str(remote)], cwd=root)
        _run(["git", "clone", str(remote), str(seed)], cwd=root)
        _run(["git", "config", "user.email", "author@example.com"], cwd=seed)
        _run(["git", "config", "user.name", "Author"], cwd=seed)

        (seed / "README.md").write_text("main\n", encoding="utf-8")
        _run(["git", "add", "README.md"], cwd=seed)
        _run(["git", "commit", "-m", "main"], cwd=seed)
        _run(["git", "branch", "-M", "main"], cwd=seed)
        _run(["git", "push", "origin", "main"], cwd=seed)
        _run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=remote)

        _run(["git", "checkout", "-b", "release"], cwd=seed)
        (seed / "release.txt").write_text("release-1\n", encoding="utf-8")
        _run(["git", "add", "release.txt"], cwd=seed)
        _run(["git", "commit", "-m", "release-1"], cwd=seed)
        _run(["git", "push", "origin", "release"], cwd=seed)

        _run(["git", "clone", str(remote), str(local)], cwd=root)
        return remote, seed, local

    def test_prepare_issue_workspace_fetches_latest_non_default_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, seed, local = self._init_remote_repos(root)

            _run(["git", "checkout", "release"], cwd=seed)
            (seed / "release.txt").write_text("release-2\n", encoding="utf-8")
            _run(["git", "commit", "-am", "release-2"], cwd=seed)
            latest_release = _run(["git", "rev-parse", "HEAD"], cwd=seed)
            _run(["git", "push", "origin", "release"], cwd=seed)

            manager = self._build_manager(local, root / "worktrees")
            with manager.prepare_issue_workspace(issue_number=12, title="Refresh release base", base_ref="release") as ws:
                workspace_head = _run(["git", "rev-parse", "HEAD"], cwd=ws.path)

            self.assertEqual(latest_release, workspace_head)

    def test_prepare_issue_workspace_reuses_remote_issue_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, seed, local = self._init_remote_repos(root)

            branch_name = "autocode/issue-7-demo"
            _run(["git", "checkout", "-b", branch_name, "release"], cwd=seed)
            (seed / "resume.txt").write_text("resume-from-existing-branch\n", encoding="utf-8")
            _run(["git", "add", "resume.txt"], cwd=seed)
            _run(["git", "commit", "-m", "resume"], cwd=seed)
            remote_branch_head = _run(["git", "rev-parse", "HEAD"], cwd=seed)
            _run(["git", "push", "origin", branch_name], cwd=seed)

            manager = self._build_manager(local, root / "worktrees")
            with manager.prepare_issue_workspace(issue_number=7, title="demo", base_ref="release") as ws:
                workspace_head = _run(["git", "rev-parse", "HEAD"], cwd=ws.path)

            self.assertEqual(remote_branch_head, workspace_head)

    def test_prepare_issue_workspace_reclaims_stale_managed_branch_worktree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")

            branch_name = manager.build_issue_branch_name(7, "demo")
            stale_path = root / "worktrees" / "stale-issue-7"
            _run(["git", "worktree", "add", "-B", branch_name, str(stale_path), "origin/release"], cwd=local)

            with manager.prepare_issue_workspace(issue_number=7, title="demo", base_ref="release") as ws:
                self.assertTrue(ws.path.exists())
                self.assertNotEqual(stale_path, ws.path)
                self.assertFalse(stale_path.exists())

    def test_prepare_issue_workspace_recovers_when_git_reports_branch_already_checked_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")

            branch_name = manager.build_issue_branch_name(7, "demo")
            stale_path = root / "worktrees" / "stale-issue-7"
            _run(["git", "worktree", "add", "-B", branch_name, str(stale_path), "origin/release"], cwd=local)

            with mock.patch.object(manager, "_cleanup_managed_branch_worktrees", autospec=True) as cleanup_mock:
                cleanup_mock.return_value = None
                with manager.prepare_issue_workspace(issue_number=7, title="demo", base_ref="release") as ws:
                    self.assertTrue(ws.path.exists())
                    self.assertNotEqual(stale_path, ws.path)

            self.assertFalse(stale_path.exists())

    def test_prepare_issue_workspace_rejects_active_branch_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            branch_name = manager.build_issue_branch_name(7, "demo")
            lock_path = manager._branch_lock_path(branch_name)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "branch": branch_name,
                        "pid": os.getpid(),
                        "created_at": 0,
                        "worktree_path": "/tmp/existing-worktree",
                    }
                ),
                encoding="utf-8",
            )

            try:
                with self.assertRaisesRegex(RuntimeError, "已有任务占用"):
                    with manager.prepare_issue_workspace(issue_number=7, title="demo", base_ref="release"):
                        self.fail("expected branch lock conflict")
            finally:
                lock_path.unlink(missing_ok=True)

    def test_prepare_issue_workspace_recovers_stale_branch_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            branch_name = manager.build_issue_branch_name(7, "demo")
            lock_path = manager._branch_lock_path(branch_name)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "branch": branch_name,
                        "pid": 999999999,
                        "created_at": 0,
                        "worktree_path": "/tmp/stale-worktree",
                    }
                ),
                encoding="utf-8",
            )

            with manager.prepare_issue_workspace(issue_number=7, title="demo", base_ref="release") as ws:
                self.assertTrue(ws.path.exists())
                metadata = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(os.getpid(), metadata["pid"])
                self.assertEqual(str(ws.path), metadata["worktree_path"])

            self.assertFalse(lock_path.exists())

    def test_active_branch_lock_returns_live_lock_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            branch_name = manager.build_issue_branch_name(7, "demo")
            lock_path = manager._branch_lock_path(branch_name)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "branch": branch_name,
                        "pid": os.getpid(),
                        "created_at": int(time.time()),
                        "worktree_path": "/tmp/existing-worktree",
                    }
                ),
                encoding="utf-8",
            )

            metadata = manager.active_branch_lock(branch_name)

            assert metadata is not None
            self.assertEqual("/tmp/existing-worktree", metadata["worktree_path"])

    def test_active_branch_lock_removes_stale_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            branch_name = manager.build_issue_branch_name(7, "demo")
            lock_path = manager._branch_lock_path(branch_name)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "branch": branch_name,
                        "pid": 999999999,
                        "created_at": 0,
                        "worktree_path": "/tmp/stale-worktree",
                    }
                ),
                encoding="utf-8",
            )

            metadata = manager.active_branch_lock(branch_name)

            self.assertIsNone(metadata)
            self.assertFalse(lock_path.exists())

    def test_sync_default_branch_pulls_latest_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, seed, local = self._init_remote_repos(root)

            _run(["git", "checkout", "main"], cwd=seed)
            (seed / "README.md").write_text("main-updated\n", encoding="utf-8")
            _run(["git", "commit", "-am", "main-updated"], cwd=seed)
            latest_main = _run(["git", "rev-parse", "HEAD"], cwd=seed)
            _run(["git", "push", "origin", "main"], cwd=seed)

            manager = self._build_manager(local, root / "worktrees")
            manager.sync_default_branch()
            local_head = _run(["git", "rev-parse", "HEAD"], cwd=local)

            self.assertEqual(latest_main, local_head)

    def test_push_branch_rejects_protected_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")

            with self.assertRaisesRegex(RuntimeError, "受保护分支"):
                manager.push_branch(local, "main")

    def test_git_disables_interactive_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(
                local,
                root / "worktrees",
                git_auth_login="octocat",
                git_auth_token="test_git_token",
            )

            with mock.patch("agent.workspace_manager.subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    ["git", "status"],
                    0,
                    stdout="",
                    stderr="",
                )
                manager._git("status", cwd=local)

            _, kwargs = run_mock.call_args
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(kwargs["env"]["GCM_INTERACTIVE"], "never")
            self.assertEqual(kwargs["env"]["GIT_CONFIG_COUNT"], "1")
            self.assertEqual(kwargs["env"]["GIT_CONFIG_KEY_0"], "http.https://github.com/.extraheader")
            self.assertTrue(kwargs["env"]["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: Basic "))

    def test_git_reports_authentication_failure_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")

            with mock.patch(
                "agent.workspace_manager.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["git", "push", "origin", "HEAD:refs/heads/demo"],
                    128,
                    stdout="",
                    stderr="fatal: could not read Username for 'https://github.com': terminal prompts disabled",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "远端 Git 认证失败"):
                    manager._git("push", "origin", "HEAD:refs/heads/demo", cwd=local)

    def test_has_changes_ignores_transient_pr_body_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            transient_file = local / ".github" / "pr-body" / "pr-12.md"
            transient_file.parent.mkdir(parents=True, exist_ok=True)
            transient_file.write_text("temporary pr body\n", encoding="utf-8")

            self.assertFalse(manager.has_changes(local))

    def test_commit_all_excludes_transient_pr_body_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            (local / "README.md").write_text("main updated\n", encoding="utf-8")
            transient_file = local / ".github" / "pr-body" / "pr-12.md"
            transient_file.parent.mkdir(parents=True, exist_ok=True)
            transient_file.write_text("temporary pr body\n", encoding="utf-8")

            manager.commit_all(local, "update readme")

            committed_files = _run(["git", "show", "--name-only", "--pretty=format:"], cwd=local).splitlines()
            self.assertIn("README.md", committed_files)
            self.assertNotIn(".github/pr-body/pr-12.md", committed_files)
            self.assertFalse(transient_file.exists())

    def test_commit_all_excludes_transient_pr_description_drafts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")
            (local / "README.md").write_text("main updated\n", encoding="utf-8")
            transient_file = local / ".github" / "pr-description-drafts" / "pr-12.md"
            transient_file.parent.mkdir(parents=True, exist_ok=True)
            transient_file.write_text("temporary pr body draft\n", encoding="utf-8")

            manager.commit_all(local, "update readme")

            committed_files = _run(["git", "show", "--name-only", "--pretty=format:"], cwd=local).splitlines()
            self.assertIn("README.md", committed_files)
            self.assertNotIn(".github/pr-description-drafts/pr-12.md", committed_files)
            self.assertFalse(transient_file.exists())

    def test_commit_all_can_amend_previous_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, local = self._init_remote_repos(root)
            manager = self._build_manager(local, root / "worktrees")

            (local / "README.md").write_text("main updated\n", encoding="utf-8")
            manager.commit_all(local, "initial update")

            (local / "README.md").write_text("main updated again\n", encoding="utf-8")
            manager.commit_all(local, "fix(review-feedback-1): address latest review comments", amend=True)

            subject = _run(["git", "log", "-1", "--pretty=%s"], cwd=local).strip()
            self.assertEqual("fix(review-feedback-1): address latest review comments", subject)
            self.assertEqual(2, len(_run(["git", "log", "--pretty=%s"], cwd=local).splitlines()))


if __name__ == "__main__":
    unittest.main()
