"""
Git 仓库安全守卫

职责:
1. PR 分析时安全切换到目标分支，分析完毕后自动切回原分支
2. 检测并还原 CLI 工具对工作区的意外修改
3. 应用关闭时确保仓库处于干净状态
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BranchPreparation:
    """PR 评审前的本地代码准备结果。"""

    ready: bool = False
    switched: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PRDiffSnapshot:
    """基于本地 git 生成的 PR diff 摘要。"""

    base_ref: str
    compare_ref: str
    merge_base: str
    changed_files: list[str]
    diff_stat: str
    diff_excerpt: str


class GitGuard:
    """Git 仓库安全操作封装。

    确保:
    - 分支切换后始终切回原分支
    - 工作区始终保持干净（无意外修改）
    """

    def __init__(self, repo_path: str, default_branch: str = "main"):
        self.repo_path = Path(repo_path).resolve()
        self.default_branch = default_branch
        # 同一时刻只允许一个线程操作共享本地仓库，避免不同 PR 互相踩 checkout/fetch/clean。
        self._repo_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 底层 git 命令
    # ------------------------------------------------------------------

    def _git_executable(self) -> str:
        git_executable = shutil.which("git")
        if not git_executable:
            logger.error("git 命令未找到，请确认已安装 git 且在 PATH 中")
            raise FileNotFoundError("git")
        return git_executable

    def _git(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """执行 git 命令并返回 CompletedProcess。"""
        if not self.repo_path.exists():
            logger.error("Git 仓库路径不存在: %s", self.repo_path)
            raise FileNotFoundError(self.repo_path)

        cmd = [self._git_executable(), *args]
        try:
            return subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("git %s 超时 (%ds)", " ".join(args), timeout)
            raise
        except FileNotFoundError as exc:
            logger.error("执行 git 命令失败: %s", exc)
            raise

    def clone_if_missing(self, remote_url: str) -> bool:
        """
        当本地仓库目录不存在时自动 clone。

        返回:
            True: 目录原本不存在且已成功 clone
            False: 目录已存在，无需处理
        """
        if self.repo_path.exists():
            return False

        parent = self.repo_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._git_executable(),
            "clone",
            "--branch",
            self.default_branch,
            "--single-branch",
            remote_url,
            str(self.repo_path),
        ]
        logger.warning("LOCAL_REPO_PATH 不存在，正在自动拉取仓库: %s -> %s", remote_url, self.repo_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            logger.error("git clone 超时 (%ss): %s", 300, remote_url)
            raise

        if result.returncode != 0:
            logger.error("git clone 失败: %s", result.stderr.strip())
            raise RuntimeError(f"git clone 失败: {result.stderr.strip()}")

        logger.info("仓库自动拉取完成: %s", self.repo_path)
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def current_branch(self) -> str:
        """获取当前分支名。分离 HEAD 时返回 'HEAD'。"""
        result = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def is_dirty(self) -> bool:
        """工作区是否有未提交的修改。"""
        result = self._git("status", "--porcelain")
        return bool(result.stdout.strip())

    def is_git_repo(self) -> bool:
        """路径是否是一个 git 仓库。"""
        result = self._git("rev-parse", "--is-inside-work-tree")
        return result.returncode == 0 and result.stdout.strip() == "true"

    def current_head_sha(self) -> str:
        """获取当前 HEAD 对应的 commit SHA。"""
        result = self._git("rev-parse", "HEAD")
        return result.stdout.strip() if result.returncode == 0 else ""

    def ref_exists(self, ref: str) -> bool:
        """检查某个 git ref 是否存在。"""
        result = self._git("rev-parse", "--verify", "--quiet", ref)
        return result.returncode == 0

    def has_remote(self, remote: str) -> bool:
        """检查某个远端是否已配置。"""
        result = self._git("remote", "get-url", remote)
        return result.returncode == 0

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def discard_changes(self):
        """丢弃所有未提交修改，恢复工作区到 HEAD 状态。"""
        self._git("checkout", "--", ".")
        self._git("clean", "-fd")

    def checkout(self, ref: str) -> bool:
        """切换到指定分支/引用。"""
        result = self._git("checkout", ref)
        if result.returncode != 0:
            logger.error("checkout %s 失败: %s", ref, result.stderr.strip())
            return False
        logger.debug("已切换到: %s", ref)
        return True

    def fetch(self, ref: str, remote: str = "origin") -> bool:
        """从指定远端拉取引用。"""
        result = self._git("fetch", remote, ref, timeout=120)
        if result.returncode != 0:
            logger.warning("fetch %s %s 失败: %s", remote, ref, result.stderr.strip())
            return False
        return True

    def pull(self, ref: str = "", remote: str = "origin") -> bool:
        """以 fast-forward only 模式同步本地分支。"""
        args = ["pull", "--ff-only", remote]
        if ref:
            args.append(ref)
        result = self._git(*args, timeout=180)
        if result.returncode != 0:
            logger.warning("pull %s %s 失败: %s", remote, ref or "", result.stderr.strip())
            return False
        return True

    def sync_default_branch(self, remote: str = "origin") -> bool:
        """
        同步默认分支到远端最新状态。

        说明:
        - 若当前不在默认分支，会先切到默认分支
        - 完成后保持停留在默认分支
        - 若工作区不干净，返回 False，由调用方决定是否先 stash
        """
        if not self.is_git_repo():
            logger.warning("路径 %s 不是 git 仓库，跳过默认分支同步", self.repo_path)
            return False
        if self.is_dirty():
            logger.warning("工作区有未提交修改，跳过默认分支同步")
            return False

        if self.current_branch() != self.default_branch and not self.checkout(self.default_branch):
            logger.warning("无法切换到默认分支 %s，跳过同步", self.default_branch)
            return False

        if not self.pull(self.default_branch, remote=remote):
            return False

        logger.info("默认分支已同步到最新: %s/%s", remote, self.default_branch)
        return True

    def build_pr_diff_snapshot(
        self,
        base_ref: str,
        *,
        max_excerpt_chars: int = 16000,
    ) -> PRDiffSnapshot:
        """
        基于当前已检出的 PR head，生成完整文件列表与重点 diff 摘要。

        说明:
        - 优先对比 `origin/<base_ref>`；若远端跟踪分支不可用，则回退到本地 `<base_ref>`
        - diff 采用 merge-base 语义，避免把 base 分支后续提交误算进 PR 变更
        """
        normalized_base_ref = base_ref.strip() or self.default_branch
        compare_ref = self._resolve_compare_ref(normalized_base_ref)
        if not compare_ref:
            raise RuntimeError(f"无法解析 PR 基线分支 {normalized_base_ref}")

        merge_base_result = self._git("merge-base", "HEAD", compare_ref)
        if merge_base_result.returncode != 0:
            raise RuntimeError(
                f"无法计算 HEAD 与 {compare_ref} 的 merge-base: {merge_base_result.stderr.strip()}"
            )
        merge_base = merge_base_result.stdout.strip()
        if not merge_base:
            raise RuntimeError(f"无法解析 HEAD 与 {compare_ref} 的 merge-base 输出")

        changed_files_result = self._git(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{merge_base}..HEAD",
        )
        if changed_files_result.returncode != 0:
            raise RuntimeError(f"无法读取 PR 改动文件列表: {changed_files_result.stderr.strip()}")
        changed_files = [line.strip() for line in changed_files_result.stdout.splitlines() if line.strip()]

        diff_stat_result = self._git("diff", "--stat", "--find-renames", f"{merge_base}..HEAD")
        diff_stat = diff_stat_result.stdout.strip() if diff_stat_result.returncode == 0 else ""

        diff_excerpt = self._build_prioritized_diff_excerpt(
            merge_base,
            changed_files,
            max_excerpt_chars=max_excerpt_chars,
        )
        return PRDiffSnapshot(
            base_ref=normalized_base_ref,
            compare_ref=compare_ref,
            merge_base=merge_base,
            changed_files=changed_files,
            diff_stat=diff_stat,
            diff_excerpt=diff_excerpt,
        )

    # ------------------------------------------------------------------
    # 安全上下文管理
    # ------------------------------------------------------------------

    @contextmanager
    def safe_default_branch(self, remote: str = "origin"):
        """
        在默认分支最新代码上执行分析，并在结束后保持停留在默认分支。

        适用于 Issue 分析:
        1. 如有本地改动，先 stash
        2. 切到默认分支并执行 `git pull --ff-only`
        3. yield synced（是否同步成功）
        4. 分析结束后清理工作区并保持在默认分支
        """
        with self._repo_lock:
            if not self.is_git_repo():
                logger.warning("路径 %s 不是 git 仓库，跳过默认分支切换", self.repo_path)
                yield False
                return

            stashed = False
            synced = False

            try:
                if self.is_dirty():
                    logger.warning("工作区有未提交修改，先 stash 保存")
                    r = self._git("stash", "push", "-m", "openreview-auto-stash")
                    stashed = r.returncode == 0

                if self.current_branch() != self.default_branch and not self.checkout(self.default_branch):
                    logger.warning("无法切换到默认分支 %s，使用当前分支继续分析", self.default_branch)
                    yield False
                    return
                synced = self.pull(self.default_branch, remote=remote)
                if synced:
                    logger.info("默认分支已同步到最新: %s/%s", remote, self.default_branch)
                else:
                    logger.warning("默认分支同步失败，继续使用当前代码进行分析")

                yield synced
            finally:
                if self.is_dirty():
                    logger.warning("分析后工作区仍有修改，正在还原...")
                    self.discard_changes()

                if self.current_branch() != self.default_branch and not self.checkout(self.default_branch):
                    logger.error("分析后无法切回默认分支 %s", self.default_branch)

                if stashed:
                    logger.warning(
                        "分析前的本地改动已保存在 stash，未自动恢复，以保持仓库停留在 %s",
                        self.default_branch,
                    )

                logger.info(
                    "默认分支收尾: branch=%s, clean=%s",
                    self.current_branch(),
                    not self.is_dirty(),
                )

    @contextmanager
    def safe_branch(self, target_ref: str, remote_url: str = "", target_sha: str = ""):
        """
        安全分支切换上下文管理器。

        流程:
        1. 工作区脏 → stash 保存
        2. 切到默认分支并 `git pull --ff-only`
        3. fetch + checkout PR 对应的最新 head
        4. 校验当前 HEAD 与目标 sha 一致
        5. yield BranchPreparation（ready=True 表示可以安全评审）
        6. finally: 还原目标分支修改 → 切回默认分支 → 确保干净

        用法::

            with git_guard.safe_branch("feature/xxx", target_sha="abc123") as prep:
                if prep.ready:
                    context = build_context(...)
        """
        with self._repo_lock:
            if not self.is_git_repo():
                logger.warning("路径 %s 不是 git 仓库，跳过分支切换", self.repo_path)
                yield BranchPreparation(reason=f"本地目录 {self.repo_path} 不是 Git 仓库")
                return

            stashed = False
            prep = BranchPreparation()

            try:
                # 1. 如果工作区脏，先 stash 保存
                if self.is_dirty():
                    logger.warning("工作区有未提交修改，先 stash 保存")
                    r = self._git("stash", "push", "-m", "openreview-auto-stash")
                    stashed = r.returncode == 0

                # 2. 先同步默认分支，确保分析基于最新主线
                if not self.sync_default_branch():
                    prep.reason = f"无法同步默认分支 {self.default_branch} 到最新状态"
                    yield prep
                    return

                # 3. fetch 目标分支/提交
                if remote_url and (target_sha or target_ref):
                    fetch_ref = target_sha or target_ref
                    if self.fetch(fetch_ref, remote=remote_url) and self.checkout("FETCH_HEAD"):
                        prep.switched = True
                    else:
                        logger.warning(
                            "无法从 PR 远端拉取目标版本 %s，尝试回退到 origin/%s",
                            fetch_ref,
                            target_ref or fetch_ref,
                        )

                if not prep.switched:
                    if not target_ref:
                        if remote_url and (target_sha or target_ref):
                            prep.reason = (
                                f"无法从 PR 远端拉取目标版本 {target_sha or target_ref}，"
                                "且 PR 缺少 head_ref 无法回退"
                            )
                        else:
                            prep.reason = "PR 缺少 head_ref，无法拉取最新代码"
                        yield prep
                        return
                    if not self.fetch(target_ref):
                        prep.reason = f"无法从 origin 拉取分支 {target_ref}"
                        yield prep
                        return

                    # 优先 checkout 刚更新后的远端分支，兼容没有远端跟踪分支的场景。
                    if self.checkout(f"origin/{target_ref}"):
                        prep.switched = True
                    elif self.checkout(target_ref):
                        prep.switched = True
                    else:
                        prep.reason = f"无法切换到 PR 分支 {target_ref}"
                        yield prep
                        return

                if target_sha:
                    current_sha = self.current_head_sha()
                    if current_sha != target_sha:
                        prep.reason = (
                            "本地检出的代码版本与 PR 最新 head 不一致: "
                            f"expected={target_sha[:12]}, actual={current_sha[:12] or 'unknown'}"
                        )
                        yield prep
                        return

                prep.ready = True
                yield prep

            finally:
                # 4. 还原目标分支上的任何修改
                if prep.switched:
                    self.discard_changes()

                if self.current_branch() != self.default_branch and not self.checkout(self.default_branch):
                    logger.error("无法切回默认分支 %s", self.default_branch)

                # 5. 最终安全检查
                if self.is_dirty():
                    logger.warning("分析后工作区仍有修改，正在还原...")
                    self.discard_changes()

                if stashed:
                    logger.warning(
                        "分析前的本地改动已保存在 stash，未自动恢复，以保持仓库停留在 %s",
                        self.default_branch,
                    )

                logger.info(
                    "分支收尾: branch=%s, clean=%s",
                    self.current_branch(),
                    not self.is_dirty(),
                )

    def _resolve_compare_ref(self, base_ref: str) -> str:
        remote_ref = f"origin/{base_ref}"
        has_origin = self.has_remote("origin")
        should_refresh_remote = has_origin and (
            base_ref != self.default_branch or not self.ref_exists(remote_ref)
        )
        if should_refresh_remote:
            if not self.fetch(base_ref):
                logger.warning("无法刷新 PR 基线分支 %s，停止构建本地 diff", base_ref)
                return ""

        if has_origin and self.ref_exists(remote_ref):
            return remote_ref
        if self.ref_exists(base_ref):
            return base_ref
        return ""

    def _build_prioritized_diff_excerpt(
        self,
        merge_base: str,
        changed_files: list[str],
        *,
        max_excerpt_chars: int,
    ) -> str:
        remaining = max_excerpt_chars
        blocks: list[str] = []
        included_files = 0
        prioritized_files = self._prioritize_review_files(changed_files)
        excerpt_candidates = [path for path in prioritized_files if self._score_review_file(path) >= 0]
        if not excerpt_candidates:
            excerpt_candidates = prioritized_files

        for rel_path in excerpt_candidates:
            diff_result = self._git(
                "diff",
                "--find-renames",
                "--unified=3",
                f"{merge_base}..HEAD",
                "--",
                rel_path,
            )
            if diff_result.returncode != 0:
                logger.warning("无法读取文件 %s 的 diff: %s", rel_path, diff_result.stderr.strip())
                continue

            patch = diff_result.stdout.strip()
            if not patch:
                continue

            block = patch if len(patch) <= remaining else patch[:remaining].rstrip() + "\n... [diff 片段已截断]"
            blocks.append(block)
            included_files += 1
            remaining -= len(block)
            if remaining <= 0:
                break

        if not blocks:
            return ""

        omitted = len(changed_files) - included_files
        if omitted > 0:
            blocks.append(f"... [其余 {omitted} 个文件未展开，已在完整文件清单中列出]")
        return "\n\n".join(blocks)

    def _prioritize_review_files(self, changed_files: list[str]) -> list[str]:
        return sorted(
            changed_files,
            key=lambda path: (-self._score_review_file(path), len(path), path),
        )

    def _score_review_file(self, rel_path: str) -> int:
        lower = rel_path.lower()
        name = Path(rel_path).name.lower()

        score = 0
        if name in {
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "bun.lockb",
            "cargo.lock",
            "poetry.lock",
        }:
            score -= 100
        if lower.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".sh", ".md")):
            score += 12
        if lower.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
            score += 6
        if any(token in lower for token in ("/agent/", "/api/", "/src/", "/tests/", "__tests__")):
            score += 6
        if name in {"readme.md", "changelog.md"}:
            score += 8
        if "test" in lower:
            score += 4
        return score

    # ------------------------------------------------------------------
    # 全局安全检查（可在关闭时调用）
    # ------------------------------------------------------------------

    def ensure_clean(self):
        """
        确保仓库处于干净状态。

        适合在应用关闭（lifespan shutdown）时调用。
        检查:
        - 工作区是否有未提交修改 → 自动还原
        - 当前分支是否是默认分支 → 自动切回
        """
        with self._repo_lock:
            if not self.is_git_repo():
                return

            if self.is_dirty():
                logger.warning("关闭时检测到未提交修改，正在还原...")
                self.discard_changes()

            current = self.current_branch()
            if current != self.default_branch and current not in ("unknown", "HEAD"):
                logger.warning("当前分支 %s ≠ %s，正在切回...", current, self.default_branch)
                self.checkout(self.default_branch)

            logger.info(
                "仓库状态检查: branch=%s, clean=%s",
                self.current_branch(),
                not self.is_dirty(),
            )
