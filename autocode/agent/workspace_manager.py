"""AutoCode 独立工作区管理。"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_TRANSIENT_WORKSPACE_FILES = {
    "pr.md",
    "pr-body.md",
    "pr_body.md",
    "pull-request.md",
    "pull_request.md",
}
_ALREADY_CHECKED_OUT_PATH_RE = re.compile(
    r"(?:already checked out|already used by worktree) at '([^']+)'",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WorkspaceSession:
    path: Path
    branch_name: str
    base_ref: str


class WorkspaceManager:
    """基于 git worktree 为每次任务创建隔离工作区。"""

    def __init__(
        self,
        *,
        repo_path: str,
        worktree_root: str,
        publish_remote: str,
        branch_prefix: str,
        default_branch: str,
        commit_user_name: str,
        commit_user_email: str,
        git_auth_login: str = "",
        git_auth_token: str = "",
    ):
        self.repo_path = Path(repo_path).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.publish_remote = publish_remote
        self.branch_prefix = branch_prefix.strip("/") or "openreview"
        self.default_branch = default_branch
        self.commit_user_name = commit_user_name
        self.commit_user_email = commit_user_email
        self.git_auth_login = git_auth_login.strip()
        self.git_auth_token = git_auth_token.strip()

    def sync_default_branch(self):
        status = self._git("status", "--porcelain", cwd=self.repo_path)
        if status.stdout.strip():
            raise RuntimeError("本地仓库存在未提交改动，无法在任务开始前同步最新代码")

        self._git("fetch", "origin", self.default_branch, cwd=self.repo_path)
        current_branch = self._git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.repo_path).stdout.strip()
        if current_branch != self.default_branch:
            self._git("checkout", self.default_branch, cwd=self.repo_path)
        result = self._git("pull", "--ff-only", "origin", self.default_branch, cwd=self.repo_path, check=False)
        if result.returncode != 0:
            # ff-only failed (local main diverged) — reset to origin
            logger.warning(
                "workspace-manager: --ff-only pull failed, resetting %s to origin/%s",
                self.default_branch,
                self.default_branch,
            )
            self._git("reset", "--hard", f"origin/{self.default_branch}", cwd=self.repo_path)

    def is_protected_branch(self, branch_name: str) -> bool:
        normalized = branch_name.strip()
        return normalized in {"main", self.default_branch}

    def build_issue_branch_name(self, issue_number: int, title: str) -> str:
        slug = _slugify(title)
        return f"{self.branch_prefix}/issue-{issue_number}-{slug}"

    @contextmanager
    def prepare_issue_workspace(self, *, issue_number: int, title: str, base_ref: str):
        branch_name = self.build_issue_branch_name(issue_number, title)
        with self._prepare_workspace(
            branch_name=branch_name,
            start_point=f"origin/{base_ref}",
            reuse_remote_branch=True,
        ) as session:
            yield session

    @contextmanager
    def prepare_existing_branch_workspace(self, *, branch_name: str, remote: str | None = None):
        if self.is_protected_branch(branch_name):
            raise RuntimeError(f"禁止直接在受保护分支上执行任务: {branch_name}")
        effective_remote = remote or self.publish_remote
        with self._prepare_workspace(
            branch_name=branch_name,
            start_point=f"{effective_remote}/{branch_name}",
        ) as session:
            yield session

    def active_branch_lock(self, branch_name: str) -> dict[str, object] | None:
        lock_path = self._branch_lock_path(branch_name)
        if not lock_path.exists():
            return None
        metadata = self._read_branch_lock(lock_path)
        if self._branch_lock_is_stale(metadata):
            logger.warning(
                "workspace-manager: removing stale branch lock branch=%s lock=%s pid=%s",
                branch_name,
                lock_path,
                metadata.get("pid", "-"),
            )
            lock_path.unlink(missing_ok=True)
            return None
        return metadata

    @contextmanager
    def _prepare_workspace(self, *, branch_name: str, start_point: str, reuse_remote_branch: bool = False):
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        worktree_path = self.worktree_root / f"{branch_name.replace('/', '-')}-{int(time.time())}"
        lock_path = self._acquire_branch_lock(branch_name)

        try:
            start_remote, start_ref = _split_remote_ref(start_point)
            if start_remote and start_ref:
                self._git("fetch", start_remote, start_ref, cwd=self.repo_path)
            else:
                self._git("fetch", "origin", self.default_branch, cwd=self.repo_path)

            effective_start_point = start_point
            if reuse_remote_branch:
                existing_branch = self._git("fetch", self.publish_remote, branch_name, cwd=self.repo_path, check=False)
                if existing_branch.returncode == 0:
                    effective_start_point = f"{self.publish_remote}/{branch_name}"

            self._cleanup_managed_branch_worktrees(branch_name)
            try:
                self._git(
                    "worktree",
                    "add",
                    "-B",
                    branch_name,
                    str(worktree_path),
                    effective_start_point,
                    cwd=self.repo_path,
                )
            except RuntimeError as exc:
                if not self._recover_already_checked_out_worktree(branch_name, exc):
                    raise
                self._git(
                    "worktree",
                    "add",
                    "-B",
                    branch_name,
                    str(worktree_path),
                    effective_start_point,
                    cwd=self.repo_path,
                )
            self._write_branch_lock(lock_path, branch_name, worktree_path)
            self._configure_identity(worktree_path)

            yield WorkspaceSession(path=worktree_path, branch_name=branch_name, base_ref=effective_start_point)
        finally:
            self._git("worktree", "remove", "--force", str(worktree_path), cwd=self.repo_path, check=False)
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)
            self._release_branch_lock(lock_path)

    def has_changes(self, workspace_path: Path) -> bool:
        result = self._git("status", "--porcelain", "--untracked-files=all", cwd=workspace_path)
        return any(
            not _is_transient_workspace_file(rel_path)
            for rel_path in _parse_status_paths(result.stdout)
        )

    def changed_files(self, workspace_path: Path) -> list[str]:
        result = self._git("status", "--porcelain", "--untracked-files=all", cwd=workspace_path)
        return [
            rel_path
            for rel_path in _parse_status_paths(result.stdout)
            if not _is_transient_workspace_file(rel_path)
        ]

    def commit_all(self, workspace_path: Path, message: str, *, amend: bool = False):
        transient_files = self.cleanup_transient_workspace_files(workspace_path)
        if transient_files:
            logger.info(
                "workspace-manager: ignored transient workspace files before commit: %s",
                ", ".join(transient_files),
            )
        self._git("add", "-A", cwd=workspace_path)
        if amend:
            self._git("commit", "--amend", "-m", message, cwd=workspace_path)
            return
        self._git("commit", "-m", message, cwd=workspace_path)

    def last_commit_subject(self, workspace_path: Path) -> str:
        result = self._git("log", "-1", "--pretty=%s", cwd=workspace_path, check=False)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def cleanup_transient_workspace_files(self, workspace_path: Path) -> list[str]:
        result = self._git("status", "--porcelain", "--untracked-files=all", cwd=workspace_path)
        removed: list[str] = []
        for rel_path in _parse_status_paths(result.stdout):
            if not _is_transient_workspace_file(rel_path):
                continue
            tracked = self._git("ls-files", "--error-unmatch", "--", rel_path, cwd=workspace_path, check=False)
            if tracked.returncode == 0:
                self._git("checkout", "--", rel_path, cwd=workspace_path, check=False)
            else:
                file_path = workspace_path / rel_path
                if file_path.exists():
                    file_path.unlink()
                    _cleanup_empty_parent_dirs(file_path.parent, stop_at=workspace_path)
            removed.append(rel_path)
        return removed

    def push_branch(self, workspace_path: Path, branch_name: str, remote: str | None = None, *, force: bool = False):
        if self.is_protected_branch(branch_name):
            raise RuntimeError(f"禁止直接推送到受保护分支: {branch_name}")
        effective_remote = remote or self.publish_remote
        push_args = ["push"]
        if force:
            push_args.append("--force-with-lease")
        push_args.extend([effective_remote, f"HEAD:refs/heads/{branch_name}"])
        self._git(*push_args, cwd=workspace_path)

    # ------------------------------------------------------------------
    # Rebase onto default branch with auto-resolution for doc conflicts
    # ------------------------------------------------------------------

    _AUTO_RESOLVE_PATTERNS: tuple[str, ...] = (
        "docs/changelog.md",
        "changelog.md",
        "docs/full-guide.md",
        "docs/full-guide_en.md",
        "docs/readme_cht.md",
        "docs/readme_en.md",
    )

    def rebase_onto_default_branch(self, workspace_path: Path) -> bool:
        """Rebase the current branch onto origin/<default_branch>.

        Auto-resolves conflicts in CHANGELOG and documentation files by
        accepting the main-branch version (``--ours`` in rebase context).
        Returns True if a rebase was performed (commits were replayed),
        False if the branch was already up-to-date.
        Raises RuntimeError if non-auto-resolvable conflicts remain.
        """
        target = f"origin/{self.default_branch}"
        self._git("fetch", "origin", self.default_branch, cwd=workspace_path)

        # Check if rebase is needed
        merge_base = self._git("merge-base", "HEAD", target, cwd=workspace_path).stdout.strip()
        target_sha = self._git("rev-parse", target, cwd=workspace_path).stdout.strip()
        if merge_base == target_sha:
            logger.info("workspace-manager: branch already up-to-date with %s", target)
            return False

        result = self._git("rebase", target, cwd=workspace_path, check=False)
        if result.returncode == 0:
            logger.info("workspace-manager: rebase onto %s succeeded cleanly", target)
            return True

        # Rebase hit conflicts — try to auto-resolve
        logger.warning(
            "workspace-manager: rebase onto %s had conflicts, attempting auto-resolution",
            target,
        )
        max_rounds = 50  # safety cap for stepwise resolution
        for _ in range(max_rounds):
            conflicted = self._get_conflicted_files(workspace_path)
            if not conflicted:
                break

            non_resolvable = []
            for cf in conflicted:
                if self._is_auto_resolvable(cf):
                    # In rebase context "ours" = the branch we rebase onto (main)
                    self._git("checkout", "--ours", "--", cf, cwd=workspace_path, check=False)
                    self._git("add", "--", cf, cwd=workspace_path)
                    logger.info("workspace-manager: auto-resolved conflict in %s (accepted main version)", cf)
                else:
                    non_resolvable.append(cf)

            if non_resolvable:
                self._git("rebase", "--abort", cwd=workspace_path, check=False)
                raise RuntimeError(
                    f"rebase 冲突无法自动解决，涉及文件: {', '.join(non_resolvable)}"
                )

            # Continue the rebase after resolving this step
            cont = self._git("rebase", "--continue", cwd=workspace_path, check=False)
            if cont.returncode == 0:
                logger.info("workspace-manager: rebase completed after auto-resolution")
                return True
            # If rebase --continue fails, there may be more conflict steps; loop again

        # Exhausted rounds — safety abort
        self._git("rebase", "--abort", cwd=workspace_path, check=False)
        raise RuntimeError("rebase auto-resolution 超过最大轮数，已中止")

    def _get_conflicted_files(self, workspace_path: Path) -> list[str]:
        result = self._git("diff", "--name-only", "--diff-filter=U", cwd=workspace_path, check=False)
        files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        if not files:
            # Also check ls-files --unmerged
            result2 = self._git("ls-files", "--unmerged", cwd=workspace_path, check=False)
            seen: set[str] = set()
            for line in result2.stdout.splitlines():
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[1].strip() not in seen:
                    seen.add(parts[1].strip())
            files = list(seen)
        return files

    def _is_auto_resolvable(self, rel_path: str) -> bool:
        normalized = rel_path.strip().lower().replace("\\", "/")
        return any(normalized.endswith(p) for p in self._AUTO_RESOLVE_PATTERNS)

    def current_head_sha(self, workspace_path: Path) -> str:
        result = self._git("rev-parse", "HEAD", cwd=workspace_path)
        return result.stdout.strip()

    def _configure_identity(self, workspace_path: Path):
        self._git("config", "user.name", self.commit_user_name, cwd=workspace_path)
        self._git("config", "user.email", self.commit_user_email, cwd=workspace_path)

    def _git(
        self,
        *args: str,
        cwd: Path,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("GCM_INTERACTIVE", "never")
        env.setdefault("GIT_EDITOR", "true")  # prevent editor popups during rebase
        auth_header = self._build_git_auth_header()
        if auth_header:
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
            env["GIT_CONFIG_VALUE_0"] = auth_header
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=env,
        )
        if check and result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "git 命令失败"
            normalized_message = message.lower()
            if (
                "could not read username" in normalized_message
                or "terminal prompts disabled" in normalized_message
                or "authentication failed" in normalized_message
                or "could not read password" in normalized_message
            ):
                raise RuntimeError(
                    "git "
                    f"{' '.join(args)} 失败: 远端 Git 认证失败，请先配置可用的 GitHub 凭据；"
                    f"原始错误: {message}"
                )
            raise RuntimeError(f"git {' '.join(args)} 失败: {message}")
        return result

    def _build_git_auth_header(self) -> str:
        if not self.git_auth_token:
            return ""
        login = self.git_auth_login or "x-access-token"
        raw = f"{login}:{self.git_auth_token}".encode("utf-8")
        encoded = base64.b64encode(raw).decode("ascii")
        return f"AUTHORIZATION: Basic {encoded}"

    def _cleanup_managed_branch_worktrees(self, branch_name: str):
        self._git("worktree", "prune", cwd=self.repo_path, check=False)
        branch_ref = f"refs/heads/{branch_name}"
        for path, ref in self._list_worktrees():
            if ref != branch_ref:
                continue
            if path == self.repo_path:
                continue
            self._remove_managed_branch_worktree(branch_name, path)
        self._git("worktree", "prune", cwd=self.repo_path, check=False)

    def _recover_already_checked_out_worktree(self, branch_name: str, exc: RuntimeError) -> bool:
        match = _ALREADY_CHECKED_OUT_PATH_RE.search(str(exc))
        if not match:
            return False
        existing_path = Path(match.group(1)).resolve()
        self._remove_managed_branch_worktree(branch_name, existing_path)
        self._git("worktree", "prune", cwd=self.repo_path, check=False)
        return True

    def _remove_managed_branch_worktree(self, branch_name: str, path: Path):
        if not self._is_managed_worktree(path):
            raise RuntimeError(f"分支 {branch_name} 已在其他工作区使用: {path}")
        logger.warning(
            "workspace-manager: removing stale managed worktree branch=%s path=%s",
            branch_name,
            path,
        )
        self._git("worktree", "remove", "--force", str(path), cwd=self.repo_path, check=False)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def _list_worktrees(self) -> list[tuple[Path, str]]:
        result = self._git("worktree", "list", "--porcelain", cwd=self.repo_path)
        entries: list[tuple[Path, str]] = []
        current_path: Path | None = None
        current_ref = ""
        for line in result.stdout.splitlines():
            if not line.strip():
                if current_path is not None:
                    entries.append((current_path, current_ref))
                current_path = None
                current_ref = ""
                continue
            if line.startswith("worktree "):
                current_path = Path(line.split(" ", 1)[1]).resolve()
            elif line.startswith("branch "):
                current_ref = line.split(" ", 1)[1].strip()
        if current_path is not None:
            entries.append((current_path, current_ref))
        return entries

    def _branch_lock_dir(self) -> Path:
        return self.worktree_root / ".locks"

    def _branch_lock_path(self, branch_name: str) -> Path:
        safe_name = branch_name.replace("/", "__")
        return self._branch_lock_dir() / f"{safe_name}.json"

    def _acquire_branch_lock(self, branch_name: str) -> Path:
        lock_dir = self._branch_lock_dir()
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._branch_lock_path(branch_name)
        while True:
            try:
                payload = {
                    "branch": branch_name,
                    "pid": os.getpid(),
                    "created_at": int(time.time()),
                    "worktree_path": "",
                }
                with lock_path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=True))
                return lock_path
            except FileExistsError:
                pass
            metadata = self.active_branch_lock(branch_name)
            if metadata is None:
                continue
            current_path = str(metadata.get("worktree_path", "") or "").strip()
            detail = f" path={current_path}" if current_path else ""
            raise RuntimeError(f"分支 {branch_name} 当前已有任务占用，无法重复创建工作区。{detail}".rstrip())

    def _write_branch_lock(self, lock_path: Path, branch_name: str, worktree_path: Path):
        payload = {
            "branch": branch_name,
            "pid": os.getpid(),
            "created_at": int(time.time()),
            "worktree_path": str(worktree_path),
        }
        lock_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8", errors="strict")

    def _release_branch_lock(self, lock_path: Path):
        lock_path.unlink(missing_ok=True)

    def _read_branch_lock(self, lock_path: Path) -> dict[str, object]:
        try:
            return json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _branch_lock_is_stale(self, metadata: dict[str, object]) -> bool:
        pid = int(metadata.get("pid", 0) or 0)
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _is_managed_worktree(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.worktree_root)
            return True
        except ValueError:
            return False


def _slugify(text: str, max_length: int = 32) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    if not normalized:
        return "task"
    return normalized[:max_length].rstrip("-") or "task"


def _normalize_rel_path(rel_path: str) -> str:
    normalized = str(rel_path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _is_transient_workspace_file(rel_path: str) -> bool:
    normalized = _normalize_rel_path(rel_path).lower()
    if not normalized:
        return False
    if normalized in _TRANSIENT_WORKSPACE_FILES:
        return True
    if normalized.startswith(".github/pr-body/") and normalized.endswith(".md"):
        return True
    return normalized.startswith(".github/pr-description-drafts/") and normalized.endswith(".md")


def _parse_status_paths(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        rel_path = line[3:].strip()
        if not rel_path:
            continue
        if " -> " in rel_path:
            rel_path = rel_path.split(" -> ", 1)[1].strip()
        rel_path = _normalize_rel_path(rel_path.strip('"'))
        if rel_path:
            paths.append(rel_path)
    return paths


def _cleanup_empty_parent_dirs(path: Path, *, stop_at: Path) -> None:
    boundary = stop_at.resolve()
    current = path
    while current.exists():
        try:
            resolved = current.resolve()
        except OSError:
            break
        if resolved == boundary:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _split_remote_ref(ref: str) -> tuple[str, str]:
    remote, sep, name = ref.partition("/")
    if not sep or not name:
        return "", ""
    return remote, name
