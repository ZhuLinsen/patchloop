"""AutoCode diff 安全检查。"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent.execution_policy import ExecutionPolicyEngine


_DEFAULT_DOCUMENTATION_PATHS = (
    "docs/",
    "doc/",
    "readme.md",
    "changelog.md",
    "agents.md",
    "claude.md",
    "skill.md",
    ".github/pull_request_template.md",
    ".github/copilot-instructions.md",
    ".github/instructions/",
    ".claude/skills/",
)
_DEFAULT_DOCUMENTATION_EXTENSIONS = (".md", ".mdx", ".rst", ".txt")


@dataclass(frozen=True)
class PatchInspectionResult:
    blocked: bool
    reasons: list[str]
    changed_files: list[str]
    added_lines: int
    deleted_lines: int
    has_test_changes: bool


class PatchInspector:
    """基于 git diff 检查改动范围与敏感路径。"""

    def __init__(
        self,
        *,
        policy_engine: ExecutionPolicyEngine,
        max_changed_files: int,
        max_added_lines: int,
        max_deleted_lines: int,
        documentation_paths: list[str] | tuple[str, ...] | None = None,
        documentation_extensions: list[str] | tuple[str, ...] | None = None,
    ):
        self.policy_engine = policy_engine
        self.max_changed_files = max_changed_files
        self.max_added_lines = max_added_lines
        self.max_deleted_lines = max_deleted_lines
        self.documentation_paths = _normalize_scope_values(documentation_paths, _DEFAULT_DOCUMENTATION_PATHS)
        self.documentation_extensions = _normalize_extensions(
            documentation_extensions,
            _DEFAULT_DOCUMENTATION_EXTENSIONS,
        )

    def inspect(
        self,
        workspace_path: str | Path,
        *,
        require_test_changes: bool,
        allowed_paths: list[str] | None = None,
        documentation_only: bool = False,
    ) -> PatchInspectionResult:
        path = Path(workspace_path)
        try:
            changed_by_path = self._collect_changed_files(path)
        except subprocess.TimeoutExpired:
            return PatchInspectionResult(
                blocked=True,
                reasons=["git diff 检查超时，请稍后重试"],
                changed_files=[],
                added_lines=0,
                deleted_lines=0,
                has_test_changes=False,
            )
        changed_files = sorted(changed_by_path)
        added_lines = sum(item[0] for item in changed_by_path.values())
        deleted_lines = sum(item[1] for item in changed_by_path.values())
        budget_files, budget_added_lines, budget_deleted_lines = self._budget_scope(changed_by_path)
        reasons: list[str] = []

        for rel_path in changed_files:
            if self.policy_engine.is_blocked_path(rel_path):
                reasons.append(f"命中受限路径: {rel_path}")

        if allowed_paths and _should_enforce_allowed_paths(allowed_paths):
            out_of_scope = _find_out_of_scope_files(changed_files, allowed_paths)
            if out_of_scope:
                reasons.append(f"越界改动: {', '.join(out_of_scope[:5])}")

        if documentation_only:
            non_documentation_files = [
                rel_path for rel_path in changed_files
                if not self._is_documentation_scope_path(rel_path)
            ]
            if non_documentation_files:
                reasons.append(f"文档任务禁止代码改动: {', '.join(non_documentation_files[:5])}")

        if not changed_files:
            reasons.append("未检测到可提交的代码改动")

        if len(budget_files) > self.max_changed_files:
            reasons.append(f"改动文件数超限: {len(budget_files)} > {self.max_changed_files}")
        if budget_added_lines > self.max_added_lines:
            reasons.append(f"新增行数超限: {budget_added_lines} > {self.max_added_lines}")
        if budget_deleted_lines > self.max_deleted_lines:
            reasons.append(f"删除行数超限: {budget_deleted_lines} > {self.max_deleted_lines}")

        has_test_changes = any(_looks_like_test_file(path) for path in changed_files)
        if require_test_changes and not has_test_changes:
            reasons.append("bug_fix 未包含测试改动，按策略需人工确认")

        return PatchInspectionResult(
            blocked=bool(reasons),
            reasons=reasons,
            changed_files=changed_files,
            added_lines=added_lines,
            deleted_lines=deleted_lines,
            has_test_changes=has_test_changes,
        )

    def _collect_changed_files(self, workspace_path: Path) -> dict[str, tuple[int, int]]:
        changed_by_path: dict[str, tuple[int, int]] = {}

        tracked_diff = subprocess.run(
            ["git", "diff", "--numstat", "--find-renames"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for rel_path, added, deleted in _parse_numstat_output(tracked_diff.stdout):
            if _is_transient_workspace_file(rel_path):
                continue
            changed_by_path[rel_path] = (
                changed_by_path.get(rel_path, (0, 0))[0] + added,
                changed_by_path.get(rel_path, (0, 0))[1] + deleted,
            )

        untracked_files = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for rel_path in (line.strip() for line in untracked_files.stdout.splitlines()):
            if not rel_path:
                continue
            if _is_transient_workspace_file(rel_path):
                continue
            file_path = workspace_path / rel_path
            added = _count_file_lines(file_path)
            changed_by_path[rel_path] = (
                changed_by_path.get(rel_path, (0, 0))[0] + added,
                changed_by_path.get(rel_path, (0, 0))[1],
            )

        return changed_by_path

    def _budget_scope(self, changed_by_path: dict[str, tuple[int, int]]) -> tuple[list[str], int, int]:
        budget_paths = [path for path in changed_by_path if not _is_supporting_budget_file(path)]
        if not budget_paths:
            budget_paths = sorted(changed_by_path)
        else:
            budget_paths = sorted(budget_paths)
        added_lines = sum(changed_by_path[path][0] for path in budget_paths)
        deleted_lines = sum(changed_by_path[path][1] for path in budget_paths)
        return budget_paths, added_lines, deleted_lines

    def _is_documentation_scope_path(self, rel_path: str) -> bool:
        return is_documentation_scope_path(
            rel_path,
            documentation_paths=self.documentation_paths,
            documentation_extensions=self.documentation_extensions,
        )


def _find_out_of_scope_files(changed_files: list[str], allowed_paths: list[str]) -> list[str]:
    normalized_allowed = [_normalize_rel_path(p) for p in allowed_paths if _normalize_rel_path(p)]
    if not normalized_allowed:
        return []
    out_of_scope: list[str] = []
    for rel_path in changed_files:
        normalized = _normalize_rel_path(rel_path)
        if not normalized:
            continue
        if _is_supporting_budget_file(normalized):
            continue
        if any(_path_matches_scope(normalized, scope) for scope in normalized_allowed):
            continue
        out_of_scope.append(rel_path)
    return out_of_scope


def _path_matches_scope(file_path: str, scope: str) -> bool:
    if scope.endswith("/"):
        return file_path == scope.rstrip("/") or file_path.startswith(scope)
    if file_path == scope:
        return True
    if file_path.startswith(scope + "/"):
        return True
    return False


def _should_enforce_allowed_paths(allowed_paths: list[str]) -> bool:
    """Treat planned paths as a hard boundary only for narrow docs/config tasks.

    For implementation work the planner's ``estimated_files`` is a hint, not a
    complete static ownership list.  Hard-blocking every discovered code file
    makes AutoCode fail precisely when it finds the correct existing module.
    Sensitive paths, diff size and test requirements are still enforced
    independently.
    """
    normalized = [_normalize_rel_path(path).lower() for path in allowed_paths if _normalize_rel_path(path)]
    if not normalized:
        return False
    return all(_is_strict_scope_path(path) for path in normalized)


def _is_strict_scope_path(path: str) -> bool:
    if path.endswith("/"):
        return path.startswith(("docs/", "doc/"))
    if path.startswith(("docs/", "doc/")):
        return True
    if path in {
        "readme.md",
        "changelog.md",
        "license",
        "license.md",
        ".env.example",
        ".env.sample",
        ".env.template",
    }:
        return True
    if path.endswith((".md", ".mdx", ".rst", ".txt")):
        return True
    if path.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")):
        return True
    return False


def is_documentation_scope_path(
    rel_path: str,
    *,
    documentation_paths: list[str] | tuple[str, ...] | set[str] | None = None,
    documentation_extensions: list[str] | tuple[str, ...] | set[str] | None = None,
) -> bool:
    """Public helper for callers that need to classify PR/file scope."""
    return _is_documentation_scope_path(
        rel_path,
        documentation_paths=_normalize_scope_values(documentation_paths, _DEFAULT_DOCUMENTATION_PATHS),
        documentation_extensions=_normalize_extensions(
            documentation_extensions,
            _DEFAULT_DOCUMENTATION_EXTENSIONS,
        ),
    )


def _is_documentation_scope_path(
    rel_path: str,
    *,
    documentation_paths: tuple[str, ...],
    documentation_extensions: tuple[str, ...],
) -> bool:
    normalized = _normalize_rel_path(rel_path).lower()
    if not normalized:
        return False
    name = normalized.rsplit("/", 1)[-1]
    if any(_path_matches_scope(normalized, scope) for scope in documentation_paths):
        return True
    if normalized.startswith(".claude/skills/") and name in {"skill.md", "readme.md"}:
        return True
    return normalized.endswith(documentation_extensions)


def _normalize_scope_values(
    values: list[str] | tuple[str, ...] | set[str] | None,
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    raw_values = values if values is not None else defaults
    normalized: list[str] = []
    for item in raw_values:
        cleaned = _normalize_rel_path(str(item or "")).lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _normalize_extensions(
    values: list[str] | tuple[str, ...] | set[str] | None,
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    raw_values = values if values is not None else defaults
    normalized: list[str] = []
    for item in raw_values:
        cleaned = str(item or "").strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _looks_like_test_file(rel_path: str) -> bool:
    normalized = rel_path.lower()
    return (
        "/tests/" in normalized
        or normalized.startswith("tests/")
        or "__tests__" in normalized
        or normalized.endswith("_test.py")
        or normalized.endswith("test.py")
        or normalized.endswith(".spec.ts")
        or normalized.endswith(".test.ts")
        or normalized.endswith(".spec.tsx")
        or normalized.endswith(".test.tsx")
    )


def _is_supporting_budget_file(rel_path: str) -> bool:
    normalized = _normalize_rel_path(rel_path).lower()
    if not normalized:
        return False
    if _looks_like_test_file(normalized):
        return True
    if normalized.endswith(".md") or normalized.startswith("docs/"):
        return True
    if normalized in {".env.example", ".env.sample", ".env.template"}:
        return True
    if "/fixtures/" in normalized or normalized.startswith("fixtures/"):
        return True
    if "/examples/" in normalized or normalized.startswith("examples/"):
        return True
    return False


def _normalize_rel_path(rel_path: str) -> str:
    normalized = str(rel_path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _is_transient_workspace_file(rel_path: str) -> bool:
    normalized = _normalize_rel_path(rel_path).lower()
    if not normalized:
        return False
    if normalized in {"pr.md", "pr-body.md", "pr_body.md", "pull-request.md", "pull_request.md"}:
        return True
    if normalized.startswith(".github/pr-body/") and normalized.endswith(".md"):
        return True
    return normalized.startswith(".github/pr-description-drafts/") and normalized.endswith(".md")


def _parse_numstat_output(output: str) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, rel_path = parts[0], parts[1], parts[2]
        rows.append(
            (
                rel_path,
                int(added) if added.isdigit() else 0,
                int(deleted) if deleted.isdigit() else 0,
            )
        )
    return rows


def _count_file_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0
