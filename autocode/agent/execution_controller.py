"""AutoCode 执行主控。"""
from __future__ import annotations

import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from adapters import create_adapter_from_profiles
from adapters.base import BaseLLMAdapter, LLMError
from agent.autocode_classifier import AutoCodeTriageResult
from agent.autocode_state_store import (
    AutoCodeRunRecord,
    AutoCodeStateStore,
    build_feedback_item_fingerprint,
    build_feedback_key,
    build_issue_plan_key,
    utc_now_iso,
)
from agent.context_builder import LocalContextBuilder
from agent.execution_planner import ExecutionPlan, build_execution_plan, triage_and_plan
from agent.execution_policy import ExecutionPolicyEngine
from agent.github_client import GitHubClient
from agent.github_write_client import GitHubWriteClient
from agent.issue_comment_intent import (
    is_repo_owner_human_issue_comment,
    issue_comment_requests_implementation,
)
from agent.issue_classifier import classify_issue
from agent.patch_inspector import PatchInspectionResult, PatchInspector, is_documentation_scope_path
from agent.validation_runner import ValidationResult, ValidationRunner, ValidationStepResult
from agent.workspace_manager import WorkspaceManager
from config import AppConfig
from prompts import (
    build_execution_failure_reply,
    build_execution_prompt,
    build_review_feedback_prompt,
    format_execution_plan_reply,
)

logger = logging.getLogger(__name__)

_ACTIONABLE_REVIEW_STATES = {"COMMENTED", "CHANGES_REQUESTED"}
_AUTOMATED_REVIEW_BOILERPLATE_MARKERS = (
    "### 💡 codex review",
    "about codex in github",
    "copilot reviewed ",
    "show a summary per file",
    "comments suppressed due to low confidence",
)
_SELF_REVIEW_NON_ACTIONABLE_BODIES = {
    "approved",
    "lgtm",
    "looks good to me",
    "ready to merge",
    "ship it",
    "可以合入",
    "可以合并",
    "通过",
    "无问题",
}
_ACTIONABLE_REVIEW_BODY_MARKERS = (
    "不可直接合入",
    "不能直接合入",
    "不可合入",
    "不能合入",
    "需要先",
    "需先",
    "请修复",
    "请更新",
    "[correctness blocker]",
    "[process blocker]",
    "[security blocker]",
    "[regression blocker]",
    "must fix",
    "needs fix",
    "not ready",
    "please fix",
    "please update",
)
_NON_ACTIONABLE_REVIEW_SUMMARY_MARKERS = (
    "评审结论",
    "必要性",
    "是否有对应 issue",
    "对应 issue",
    "关联 issue",
    "pr 类型",
    "description 完整性",
    "是否可直接合入",
    "可直接合入",
)
_ACTIONABLE_SELF_REVIEW_MARKERS = (
    "不可直接合入",
    "不能直接合入",
    "需要先",
    "需先",
    "请修复",
    "请更新",
    "[correctness blocker]",
    "[process blocker]",
    "[security blocker]",
    "[regression blocker]",
    "must fix",
    "needs fix",
    "not ready",
    "please fix",
    "please update",
)
_ACTIONABLE_SELF_REVIEW_REFERENCE_RE = re.compile(
    r"`[^`]+`|(?:^|[\s(])(?:src|tests|docs|scripts|app|bot|agent|adapters|data_provider|data|projects)/\S+",
    re.IGNORECASE,
)
_ISSUE_TITLE_PREFIX_RE = re.compile(r"^\[(bug|feature|docs|question)\]\s*", re.IGNORECASE)
_INFORMAL_TITLE_RE = re.compile(
    r"[？?吗呢吧嘛]\s*$|[？?吗]|^(?:为什么|为啥|怎么|如何|能否|能不能|能新增|能增加|能加|可以|可否|请问|是否|有没有|是不是)",
)
_VERBOSE_TITLE_MAX_LEN = 30  # titles longer than this prefer plan_goal when available
_PLACEHOLDER_TITLE_RE = re.compile(r"^[\s(#\d)]*$")
_LOW_SIGNAL_TITLE_PREFIX_RE = re.compile(r"^(?:今天|昨天|刚刚|现在|目前|突然|又|还是|这边|这里|那边|这个|这块|好像|似乎)")
_LOW_SIGNAL_TITLE_SUFFIX_RE = re.compile(
    r"(?:跑不动(?:了)?|起不来(?:了)?|用不了(?:了)?|不工作(?:了)?|有问题(?:了)?|出问题(?:了)?|挂了|坏了|不行了|崩了|炸了)$"
)
_PR_LINKED_ISSUE_RE = re.compile(r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)\b", re.IGNORECASE)
_PR_TITLE_ISSUE_RE = re.compile(r"\(#(\d+)\)\s*$")
_SIMPLE_PR_FEEDBACK_MARKERS = (
    "pr description",
    "pull request description",
    "pr body",
    "pull request body",
    "pr 描述",
    "pull request 描述",
    "description 完整性",
    "scope of change",
    "issue link",
    "描述与实际改动内容",
    "changed files list",
    "changed files",
)
_DOC_ONLY_PR_BODY_MARKERS = (
    "docs only",
    "documentation only",
    "docs-only",
    "文档 only",
    "仅文档",
    "只改文档",
    "仅涉及文档",
    "只涉及文档",
    "文档整理",
    "文档重构",
    "文档调整",
    "tests not run",
    "测试未运行",
)
_DOC_ONLY_TITLE_MARKERS = (
    "docs",
    "documentation",
    "readme",
    "guide",
    "文档",
)
_CODE_SCOPE_PATH_RE = re.compile(
    r"^(?:src|app|apps|api|bot|agent|adapters|services|server|web|frontend|backend|tests)/|"
    r"\.(?:py|js|jsx|ts|tsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php)$",
    re.IGNORECASE,
)
_BOT_REVIEWER_LOGINS = frozenset({
    "copilot-pull-request-reviewer",
    "github-actions[bot]",
    "chatgpt-codex-connector",
    "copilot",
    "dependabot[bot]",
})
_BOT_REVIEW_BODY_MARKERS = (
    "<!-- openreview:idempotency:",
    "<!-- openreview:",
)
_ANSWERABLE_ISSUE_CONFIDENCE_THRESHOLD = 0.8
_ISSUE_DISCUSSION_MAX_ITEMS = 8
_ISSUE_DISCUSSION_BODY_LIMIT = 400
_RESOLVED_CONFIRMATION_RE = re.compile(
    r"(?:已解决|解决了|成功了|可以了|好了|恢复了|搞定了|works\s+now|fixed|resolved)",
    re.IGNORECASE,
)
_ACTIONABLE_BUG_RE = re.compile(
    r"(?:无法保存|不能保存|保存失败|无法使用|不能使用|黑屏|白屏|闪退|崩溃|crash|"
    r"不显示|显示.*缺失|数据缺失|计算.*错|公式.*错|负缓存|无过期|没有过期|"
    r"报错|异常|Traceback|Exception|invalid\s+\w+|broken)",
    re.IGNORECASE,
)
_MODEL_PARAMETER_COMPAT_RE = re.compile(
    r"(?:invalid\s+temperature|temperature.*only\s+1|only\s+1\s+is\s+allowed|"
    r"temperature.*只能|temperature.*只(?:能|接受)|只(?:能|接受).*temperature|"
    r"unsupported\s+parameter|参数.*不支持)",
    re.IGNORECASE,
)
_SUPPORT_ONLY_BUG_RE = re.compile(
    r"(?:github\s+actions|workflow|timeout-minutes|maximum execution time|"
    r"环境变量|配置项|怎么配置|如何配置|在哪配置|文档|说明)",
    re.IGNORECASE,
)
_NON_PR_REDIRECT_PATTERNS = (
    re.compile(r"伪需求", re.IGNORECASE),
    re.compile(r"(?:无需|不用|不必|不要|别再)(?:继续)?\s*改代码", re.IGNORECASE),
    re.compile(r"(?:改|调整|修改)[^。；;\n]{0,80}(?:github\s*配置|workflow|timeout-minutes|运行时间配置|运行时配置)[^。；;\n]{0,40}(?:即可|就行|就好)", re.IGNORECASE),
    re.compile(r"(?:只需|只需要)[^。；;\n]{0,60}(?:配置|workflow|timeout-minutes)", re.IGNORECASE),
    re.compile(r"(?:不需要|无需)[^。；;\n]{0,40}(?:pr|pull request)", re.IGNORECASE),
    re.compile(r"(?:关闭|关掉)[^。；;\n]{0,40}(?:pr|pull request)", re.IGNORECASE),
)

# OpenReview 审批检测 —— 用于在 OpenReview 已通过时跳过后续 repair/rebase
_OPENREVIEW_BODY_MARKER = "openreview:idempotency"
_OPENREVIEW_VERDICT_RE = re.compile(
    r"可直接合入\*{0,2}\s*[：:]\s*(不可|不能|可)",
)


@dataclass(frozen=True)
class PlanningResult:
    issue_number: int
    title: str
    body: str
    labels: list[str]
    updated_at: str
    comment: str
    triage: AutoCodeTriageResult
    plan: ExecutionPlan
    issue_key: str

    def to_snapshot_record(self) -> dict:
        return {
            "issue_number": self.issue_number,
            "title": self.title,
            "body": self.body,
            "labels": self.labels,
            "updated_at": self.updated_at,
            "comment": self.comment,
            "triage": self.triage.to_dict(),
            "plan": self.plan.to_dict(),
            "issue_key": self.issue_key,
            "saved_at": utc_now_iso(),
        }

    @classmethod
    def from_snapshot_record(cls, data: dict) -> "PlanningResult":
        return cls(
            issue_number=int(data.get("issue_number", 0) or 0),
            title=str(data.get("title", "") or ""),
            body=str(data.get("body", "") or ""),
            labels=[str(item) for item in data.get("labels", []) if str(item).strip()],
            updated_at=str(data.get("updated_at", "") or ""),
            comment=str(data.get("comment", "") or ""),
            triage=AutoCodeTriageResult.from_dict(dict(data.get("triage", {}) or {})),
            plan=ExecutionPlan.from_dict(dict(data.get("plan", {}) or {})),
            issue_key=str(data.get("issue_key", "") or ""),
        )


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    comment: str
    run_record: AutoCodeRunRecord


@dataclass(frozen=True)
class PullRequestRepairCandidate:
    pr_number: int
    task_id: str
    queue_source: str
    review_state: str
    feedback_key: str
    feedback_updated_at: str
    feedback_item_ids: list[int]
    summary: str
    feedback: list[dict] = field(default_factory=list)
    legacy_feedback_key: str = ""


class ExecutionController:
    """AutoCode 计划、执行、发布与 review feedback 修复主控。"""

    def __init__(
        self,
        *,
        planner: BaseLLMAdapter,
        executor: BaseLLMAdapter,
        gh: GitHubClient,
        gh_writer: GitHubWriteClient,
        state_store: AutoCodeStateStore,
        workspace_manager: WorkspaceManager,
        validation_runner: ValidationRunner,
        patch_inspector: PatchInspector,
        policy_engine: ExecutionPolicyEngine,
        config: AppConfig,
    ):
        self.planner = planner
        self.executor = executor
        self.gh = gh
        self.gh_writer = gh_writer
        self.state_store = state_store
        self.workspace_manager = workspace_manager
        self.validation_runner = validation_runner
        self.patch_inspector = patch_inspector
        self.policy_engine = policy_engine
        self.config = config
        self.context_builder = LocalContextBuilder(
            repo_path=config.local_repo.path,
            max_context_files=config.local_repo.max_context_files,
            max_file_chars=config.local_repo.max_file_chars,
            max_total_chars=config.local_repo.max_total_context_chars,
        )

    def _project_log_path(self) -> Path | None:
        repo = str(getattr(getattr(self.config, "github", None), "repo", "") or "").strip()
        repo_name = repo.rsplit("/", 1)[-1].strip() if repo else ""
        if not repo_name:
            return None
        configured_root = str(getattr(getattr(self.config, "autocode", None), "project_log_root", "") or "").strip()
        base_dir = Path(configured_root).expanduser() if configured_root else Path(__file__).resolve().parents[1] / "projects"
        return base_dir / f"{repo_name}.md"

    def _ensure_project_log_file(self, path: Path):
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        project_name = path.stem
        path.write_text(
            (
                f"# {project_name} Execution Log\n\n"
                "## 已处理记录\n\n"
                "| 日期 | 类型 | 编号 | 主题 | 动作 | 结果 | 备注 |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
            ),
            encoding="utf-8",
        )

    def _project_log_cell(self, value: str, *, limit: int = 160) -> str:
        text = str(value or "").strip()
        if not text:
            return "-"
        text = text.replace("|", "/").replace("\r", " ").replace("\n", "<br>")
        return self._truncate_log_text(text, limit=limit) or "-"

    def _append_project_log_entry(
        self,
        *,
        entry_type: str,
        number: int,
        subject: str,
        action: str,
        result: str,
        note: str = "",
    ):
        path = self._project_log_path()
        if path is None:
            return
        try:
            self._ensure_project_log_file(path)
            text = path.read_text(encoding="utf-8")
            row = (
                f"| {utc_now_iso()[:10]} | {self._project_log_cell(entry_type, limit=24)} | "
                f"{number if int(number or 0) > 0 else '-'} | {self._project_log_cell(subject, limit=80)} | "
                f"{self._project_log_cell(action, limit=32)} | {self._project_log_cell(result, limit=32)} | "
                f"{self._project_log_cell(note, limit=220)} |"
            )
            if row in text:
                return
            placeholder = "| 待补 | Issue/PR | - | - | - | - | 这里补 AutoCode 实际处理过的条目 |"
            if placeholder in text:
                updated = text.replace(placeholder, row, 1)
            else:
                section_start = text.find("## 已处理记录")
                marker = "\n## "
                insert_at = text.find(marker, section_start if section_start >= 0 else 0)
                if insert_at == -1:
                    updated = text.rstrip() + "\n" + row + "\n"
                else:
                    updated = text[:insert_at].rstrip() + "\n" + row + "\n\n" + text[insert_at:].lstrip("\n")
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            logger.warning("project-log: failed to append entry path=%s error=%s", path, exc)

    def plan_issue(self, issue: dict) -> PlanningResult:
        """为单个 issue 生成并冻结一份结构化计划快照。

        这一步只做理解和规划，不进入代码修改；生成的快照可被后续显式执行
        API 复用，避免同一版 issue 反复重新计划。
        """
        self._sync_repo_before_task("issue planning")
        issue_number = int(issue["number"])
        title = str(issue.get("title", "") or "")
        body = str(issue.get("body", "") or "")
        labels = [str(label.get("name", "")) for label in issue.get("labels", [])]
        updated_at = str(issue.get("updated_at", "") or "")
        logger.info(
            "issue-plan: 开始生成结构化计划 issue #%s title=%s labels=%s updated_at=%s",
            issue_number,
            self._truncate_log_text(title, limit=80) or "-",
            self._summarize_list(labels),
            updated_at or "-",
        )
        discussion_comments = self._list_all_issue_comments(issue_number)
        discussion_context = self._format_issue_discussion_context(discussion_comments)
        local_code_context = self.context_builder.build_issue_context(
            title=title,
            body=self._issue_text_with_discussion(body, discussion_context),
        )

        triage, plan = triage_and_plan(
            title=title,
            body=body,
            labels=labels,
            repo_name=self.config.github.repo,
            analyzer=self._planner_with_context(f"issue-plan#{issue_number}:triage+plan"),
            local_code_context=local_code_context,
            discussion_context=discussion_context,
        )
        issue_key = build_issue_plan_key(issue_number, updated_at, triage.task_type)
        comment = format_execution_plan_reply(
            header=self.config.autocode.plan_comment_header,
            task_type=triage.task_type,
            action=triage.action,
            plan_summary=plan.goal,
            risk_level=plan.risk_level,
            estimated_files=plan.estimated_files,
            acceptance_criteria=plan.acceptance_criteria,
            needs_human_approval=plan.needs_human_approval,
            blocked_reasons=plan.blocked_reasons,
        )
        planning = PlanningResult(
            issue_number=issue_number,
            title=title,
            body=body,
            labels=labels,
            updated_at=updated_at,
            comment=comment,
            triage=triage,
            plan=plan,
            issue_key=issue_key,
        )
        self.state_store.save_issue_plan_snapshot(issue_number, planning.to_snapshot_record())
        logger.info(
            "issue-plan: 结构化计划已冻结 issue #%s task_type=%s action=%s risk=%s need_human=%s estimated_files=%s",
            issue_number,
            triage.task_type,
            triage.action,
            plan.risk_level,
            plan.needs_human_approval,
            self._summarize_list(plan.estimated_files),
        )
        return planning

    def _list_all_issue_comments(self, issue_number: int, *, per_page: int = 100) -> list[dict]:
        if issue_number <= 0:
            return []
        comments: list[dict] = []
        page = 1
        while True:
            page_items = self.gh.list_issue_comments(issue_number, per_page=per_page, page=page)
            if not page_items:
                break
            comments.extend(page_items)
            if len(page_items) < per_page:
                break
            page += 1
        return comments

    def _issue_text_with_discussion(self, body: str, discussion_context: str) -> str:
        if not discussion_context:
            return body
        stripped_body = str(body or "").strip()
        if not stripped_body:
            return discussion_context
        return f"{stripped_body}\n\n{discussion_context}"

    def _format_issue_discussion_context(self, comments: list[dict], *, max_items: int = _ISSUE_DISCUSSION_MAX_ITEMS) -> str:
        if not comments:
            return ""
        selected_lines: list[str] = []
        for comment in comments[-max_items:]:
            body = str(comment.get("body", "") or "").strip()
            if not body:
                continue
            login = str(comment.get("user", {}).get("login", "") or "").strip() or "unknown"
            normalized_login = login.lower()
            tags: list[str] = []
            if self._is_repo_owner_login(normalized_login):
                tags.append("owner")
            elif self._is_bot_login(normalized_login):
                tags.append("bot")
            association = str(comment.get("author_association", "") or "").strip().upper()
            if association and association not in {"NONE", "CONTRIBUTOR"}:
                tags.append(association.lower())
            tag_suffix = f" ({', '.join(dict.fromkeys(tags))})" if tags else ""
            snippet = " ".join(body.split())
            if len(snippet) > _ISSUE_DISCUSSION_BODY_LIMIT:
                snippet = snippet[: _ISSUE_DISCUSSION_BODY_LIMIT - 1].rstrip() + "…"
            selected_lines.append(f"- {login}{tag_suffix}: {snippet}")
        if not selected_lines:
            return ""
        return "\n".join(selected_lines)

    def _is_repo_owner_login(self, login: str) -> bool:
        normalized = str(login or "").strip().lower()
        owner = str(getattr(getattr(self.config, "github", None), "owner", "") or "").strip().lower()
        return bool(normalized and owner and normalized == owner)

    def _is_repo_owner_issue_author(self, issue: dict) -> bool:
        author_login = str((issue.get("user", {}) or {}).get("login", "") or "").strip().lower()
        return self._is_repo_owner_login(author_login)

    def _is_bot_login(self, login: str) -> bool:
        normalized = str(login or "").strip().lower()
        if not normalized:
            return False
        bare_login = normalized.removesuffix("[bot]")
        return normalized.endswith("[bot]") or normalized in _BOT_REVIEWER_LOGINS or bare_login in _BOT_REVIEWER_LOGINS

    def _find_authoritative_non_pr_guidance(self, comments: list[dict]) -> str:
        intent, body = self._latest_owner_issue_intent(comments)
        return body if intent == "redirect" else ""

    def _latest_owner_issue_intent(self, comments: list[dict]) -> tuple[str, str]:
        repo_owner = str(getattr(getattr(self.config, "github", None), "owner", "") or "")
        for comment in reversed(comments):
            if not is_repo_owner_human_issue_comment(comment, repo_owner):
                continue
            body = str(comment.get("body", "") or "").strip()
            if not body:
                continue
            if any(pattern.search(body) for pattern in _NON_PR_REDIRECT_PATTERNS):
                return "redirect", body
            if issue_comment_requests_implementation(body):
                return "implement", body
        return "", ""

    def _has_authoritative_owner_implement_command(self, comments: list[dict]) -> bool:
        intent, _ = self._latest_owner_issue_intent(comments)
        return intent == "implement"

    def _latest_external_issue_comment_body(self, comments: list[dict]) -> str:
        for comment in reversed(comments):
            login = str(comment.get("user", {}).get("login", "") or "").strip().lower()
            if not login or self._is_bot_login(login):
                continue
            body = str(comment.get("body", "") or "").strip()
            if body:
                return body
        return ""

    def _issue_discussion_confirms_resolved(self, comments: list[dict]) -> bool:
        latest_body = self._latest_external_issue_comment_body(comments)
        return bool(latest_body and _RESOLVED_CONFIRMATION_RE.search(latest_body))

    def _answerable_issue_should_continue_to_code_flow(
        self,
        *,
        title: str,
        body: str,
        labels: list[str],
        discussion_context: str,
    ) -> bool:
        normalized_labels = {str(label or "").strip().lower() for label in labels}
        combined = "\n".join(
            part for part in (title or "", body or "", discussion_context or "") if str(part or "").strip()
        )
        if not combined.strip():
            return False
        if _MODEL_PARAMETER_COMPAT_RE.search(combined):
            return True
        if "bug" not in normalized_labels:
            return False
        if _SUPPORT_ONLY_BUG_RE.search(combined) and not _ACTIONABLE_BUG_RE.search(combined):
            return False
        return bool(_ACTIONABLE_BUG_RE.search(combined))

    def _finish_non_pr_issue(
        self,
        *,
        issue_number: int,
        title: str,
        labels: list[str],
        updated_at: str,
        status: str,
        summary: str,
    ) -> ExecutionResult:
        # AutoCode must not publish issue comments. Non-PR issues only update
        # internal state so OpenReview or the maintainer can decide follow-up.
        run_record = AutoCodeRunRecord(
            run_id=self._build_run_id("issue", issue_number),
            status=status,
            kind="issue",
            target_number=issue_number,
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
        issue_key = build_issue_plan_key(issue_number, updated_at, status)
        self.state_store.mark_issue_plan(
            issue_key=issue_key,
            issue_number=issue_number,
            task_type=self._task_type_for_non_pr_issue(labels),
            updated_at=updated_at,
            summary=summary,
            last_outcome=status,
        )
        self.state_store.mark_run(run_record)
        self._append_project_log_entry(
            entry_type="Issue",
            number=issue_number,
            subject=title,
            action="skipped" if status == "answered" else "manual_action",
            result=status,
            note=summary,
        )
        return ExecutionResult(False, summary, run_record)

    def _task_type_for_non_pr_issue(self, labels: list[str]) -> str:
        normalized = {str(label or "").strip().lower() for label in labels}
        if "bug" in normalized:
            return "bug_fix"
        if "enhancement" in normalized or "feature" in normalized:
            return "small_feature"
        return "todo_refactor"

    def _maybe_resolve_issue_without_pr(
        self,
        issue: dict,
        *,
        explicit_command: bool,
        comments: list[dict] | None = None,
    ) -> ExecutionResult | None:
        issue_number = int(issue.get("number", 0) or 0)
        if issue_number <= 0:
            return None
        title = str(issue.get("title", "") or "")
        body = str(issue.get("body", "") or "")
        labels = [str(label.get("name", "") or "") for label in issue.get("labels", []) if str(label.get("name", "") or "").strip()]
        updated_at = str(issue.get("updated_at", "") or "")
        comments = list(comments) if comments is not None else self._list_all_issue_comments(issue_number)
        discussion_context = self._format_issue_discussion_context(comments)

        authoritative_guidance = self._find_authoritative_non_pr_guidance(comments)
        if authoritative_guidance:
            return self._finish_non_pr_issue(
                issue_number=issue_number,
                title=title,
                labels=labels,
                updated_at=updated_at,
                status="manual_action",
                summary="维护者已明确指出当前问题更适合走配置/手工路径，不应继续自动生成代码 PR。",
            )

        if self._issue_discussion_confirms_resolved(comments):
            return self._finish_non_pr_issue(
                issue_number=issue_number,
                title=title,
                labels=labels,
                updated_at=updated_at,
                status="resolved_by_user",
                summary="讨论中最新外部评论已确认问题解决，跳过自动实现。",
            )

        if explicit_command:
            return None

        local_code_context = self.context_builder.build_issue_context(
            title=title,
            body=self._issue_text_with_discussion(body, discussion_context),
        )
        classification = classify_issue(
            title,
            body or "",
            labels,
            self._planner_with_context(f"issue-classify#{issue_number}"),
            repo_name=self.config.github.repo,
            local_code_context=local_code_context,
            discussion_context=discussion_context,
        )
        if (
            classification.degraded
            or not classification.is_answerable
            or classification.confidence < _ANSWERABLE_ISSUE_CONFIDENCE_THRESHOLD
        ):
            return None

        if self._answerable_issue_should_continue_to_code_flow(
            title=title,
            body=body,
            labels=labels,
            discussion_context=discussion_context,
        ):
            logger.info(
                "issue-execution: issue #%s classified answerable (%s) but has actionable bug/code signals; continue planning",
                issue_number,
                classification.reason,
            )
            return None

        return self._finish_non_pr_issue(
            issue_number=issue_number,
            title=title,
            labels=labels,
            updated_at=updated_at,
            status="answered",
            summary=f"Issue 可直接回复处理，无需生成代码 PR（{classification.reason}）",
        )

    def execute_issue(
        self,
        issue: dict,
        *,
        explicit_command: bool,
        planning: PlanningResult | None = None,
    ) -> ExecutionResult:
        """执行单个 issue 的主流程。

        流程分为三段：先解析/恢复计划，再做策略判断，最后才在受限 worktree
        中调用 executor 改代码、验证并发布 PR。
        """
        issue_number = int(issue["number"])
        issue_comments = self._list_all_issue_comments(issue_number)
        if not explicit_command and self._has_authoritative_owner_implement_command(issue_comments):
            explicit_command = True
            logger.info(
                "issue-execution: issue #%s promoted to explicit execution by latest owner approval comment",
                issue_number,
            )
        non_pr_result = self._maybe_resolve_issue_without_pr(
            issue,
            explicit_command=explicit_command,
            comments=issue_comments,
        )
        if non_pr_result is not None:
            logger.info(
                "issue-execution: resolved issue #%s without code PR run_status=%s",
                issue_number,
                non_pr_result.run_record.status,
            )
            return non_pr_result

        run_id = self._build_run_id("issue", issue_number)
        started_at = utc_now_iso()
        current_stage = "planning"
        try:
            planning = self._resolve_planning(
                issue,
                explicit_command=explicit_command,
                provided_planning=planning,
            )
        except RuntimeError as exc:
            run_record = AutoCodeRunRecord(
                run_id=run_id,
                status="failed",
                kind="issue",
                target_number=issue_number,
                failure_reason=str(exc),
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(run_record)
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=str(issue.get("title", "") or f"Issue #{issue_number}"),
                action="planning",
                result="failed",
                note=str(exc),
            )
            return ExecutionResult(False, build_execution_failure_reply("planning", str(exc)), run_record)

        title = planning.title
        body = planning.body
        labels = planning.labels
        updated_at = planning.updated_at
        pr_base_branch = self._resolved_pr_base_branch()
        owner_authored_issue = self._is_repo_owner_issue_author(issue)
        logger.info(
            "issue-execution: prepared issue #%s explicit=%s owner_authored=%s mode=%s task_type=%s risk=%s labels=%s updated_at=%s",
            issue_number,
            explicit_command,
            owner_authored_issue,
            self.config.autocode.mode,
            planning.triage.task_type,
            planning.plan.risk_level,
            ",".join(labels) if labels else "-",
            updated_at or "-",
        )

        current_stage = "sync_repo"
        self._sync_repo_before_task("issue execution")

        decision = self.policy_engine.evaluate_issue(
            triage=planning.triage,
            plan=planning.plan,
            labels=labels,
            explicit_command=explicit_command,
            owner_authored=owner_authored_issue,
        )
        publish_target = self.policy_engine.validate_publish_target(pr_base_branch)
        if not publish_target.allowed:
            failure = "；".join(publish_target.reasons)
            run_record = AutoCodeRunRecord(
                run_id=run_id,
                status="failed",
                kind="issue",
                target_number=issue_number,
                failure_reason=failure,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(run_record)
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=title,
                action="publish_target",
                result="failed",
                note=failure,
            )
            return ExecutionResult(False, build_execution_failure_reply("policy", failure), run_record)

        # 自动执行条件不满足时，保留计划快照但不真正修改代码。
        if not explicit_command and not self._should_auto_execute(
            planning.triage,
            owner_authored_issue=owner_authored_issue,
        ):
            logger.info(
                "issue-execution: plan-only issue #%s mode=%s task_type=%s auto_implement=%s owner_authored=%s",
                issue_number,
                self.config.autocode.mode,
                planning.triage.task_type,
                self.config.autocode.auto_implement_on_issue_open,
                owner_authored_issue,
            )
            run_record = AutoCodeRunRecord(
                run_id=run_id,
                status="planned",
                kind="issue",
                target_number=issue_number,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(run_record)
            self.state_store.mark_issue_plan(
                issue_key=planning.issue_key,
                issue_number=issue_number,
                task_type=planning.triage.task_type,
                updated_at=updated_at,
                summary=planning.plan.goal,
                last_outcome="planned",
            )
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=title,
                action="plan_only",
                result="planned",
                note=planning.plan.goal,
            )
            return ExecutionResult(False, planning.comment, run_record)

        # 即使是显式 implement，也仍然要经过策略引擎的边界检查。
        if not decision.allowed:
            failure = "；".join(decision.reasons)
            logger.info(
                "issue-execution: blocked issue #%s task_type=%s reasons=%s",
                issue_number,
                planning.triage.task_type,
                failure,
            )
            run_record = AutoCodeRunRecord(
                run_id=run_id,
                status="blocked",
                kind="issue",
                target_number=issue_number,
                failure_reason=failure,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(run_record)
            self.state_store.mark_issue_plan(
                issue_key=planning.issue_key,
                issue_number=issue_number,
                task_type=planning.triage.task_type,
                updated_at=updated_at,
                summary=planning.plan.goal,
                last_outcome="blocked",
            )
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=title,
                action=planning.triage.action or "auto_fix",
                result="blocked",
                note=failure,
            )
            return ExecutionResult(False, planning.comment + f"\n\n> 自动执行已暂停：{failure}", run_record)

        run_record = AutoCodeRunRecord(
            run_id=run_id,
            status="running",
            kind="issue",
            target_number=issue_number,
            started_at=started_at,
        )
        self.state_store.mark_run(run_record)

        try:
            dry_run = self._is_dry_run()
            current_stage = "prepare_workspace"
            logger.info(
                "issue-execution: issue #%s 准备进入隔离工作区 base=%s goal=%s estimated_files=%s validation=%s",
                issue_number,
                pr_base_branch,
                self._truncate_log_text(planning.plan.goal, limit=120) or "-",
                self._summarize_list(planning.plan.estimated_files),
                self._summarize_list(self._validation_commands()),
            )
            with self.workspace_manager.prepare_issue_workspace(
                issue_number=issue_number,
                title=title,
                base_ref=pr_base_branch,
            ) as workspace:
                logger.info(
                    "issue-execution: issue #%s 已进入隔离工作区 branch=%s workspace=%s base_ref=%s",
                    issue_number,
                    workspace.branch_name,
                    workspace.path,
                    workspace.base_ref,
                )
                executor_settings = self._execution_adapter_settings(planning.triage.task_type)
                logger.info(
                    "issue-execution: issue #%s 执行器配置 profile=%s task_type=%s timeout=%ss idle_limit=%s primary_model=%s primary_fallback=%s fallback_clis=%s",
                    issue_number,
                    executor_settings["profile"],
                    planning.triage.task_type,
                    executor_settings["cli_timeout"],
                    executor_settings["max_idle_seconds"] or "-",
                    executor_settings["primary_model"] or "-",
                    executor_settings["primary_fallback_model"] or "-",
                    executor_settings["fallback_clis"],
                )
                workspace_executor = self._executor_for_workspace(
                    workspace.path,
                    task_type=planning.triage.task_type,
                    log_context=f"issue-execution#{issue_number}",
                )
                workspace_context = self._build_workspace_context(workspace.path, title, body)
                repair_notes = list(self._get_previous_failure_reasons("issue", issue_number))
                max_repair_attempts = max(
                    int(getattr(self.config.autocode, "issue_repair_max_attempts", 2) or 0),
                    0,
                )
                summary_parts: list[str] = []
                validation = ValidationResult(success=True, steps=[])
                inspection: PatchInspectionResult | None = None
                execution_failure = ""

                for attempt_index in range(max_repair_attempts + 1):
                    attempt_no = attempt_index + 1
                    is_repair_attempt = attempt_index > 0
                    prompt = build_execution_prompt(
                        title=title,
                        body=body,
                        repo_name=self.config.github.repo,
                        task_type=planning.triage.task_type,
                        plan_json=planning.plan.to_json(),
                        allowed_paths=planning.plan.estimated_files,
                        validation_commands=self._validation_commands(),
                        local_code_context=workspace_context,
                        max_changed_files=self.config.autocode.max_changed_files,
                        max_added_lines=self.config.autocode.max_added_lines,
                        max_deleted_lines=self.config.autocode.max_deleted_lines,
                        previous_failure_reasons=repair_notes,
                    )
                    current_stage = "cli_repair" if is_repair_attempt else "cli_execute"
                    logger.info(
                        "issue-execution: issue #%s 开始调用%s执行器 branch=%s attempt=%s/%s allowed_paths=%s",
                        issue_number,
                        "修复" if is_repair_attempt else "",
                        workspace.branch_name,
                        attempt_no,
                        max_repair_attempts + 1,
                        self._summarize_list(planning.plan.estimated_files),
                    )
                    summary = workspace_executor.analyze(
                        prompt,
                        system=(
                            "你是受限 AutoCode 修复器，必须先修复上一轮失败原因，只修改必要文件并输出简短摘要。"
                            if is_repair_attempt
                            else "你是受限 AutoCode 执行器，只修改必要文件并输出简短实现摘要。"
                        ),
                    )
                    if summary:
                        summary_parts.append(f"第 {attempt_no} 轮：{summary.strip()}")
                        logger.info(
                            "issue-execution: CLI 输出摘要 for issue #%s attempt=%s: %s",
                            issue_number,
                            attempt_no,
                            self._truncate_log_text(summary, limit=600),
                        )
                    if not self.workspace_manager.has_changes(workspace.path):
                        execution_failure = "执行完成后未检测到代码改动"
                        logger.warning(
                            "issue-execution: issue #%s 执行结束但未检测到代码改动 branch=%s workspace=%s attempt=%s",
                            issue_number,
                            workspace.branch_name,
                            workspace.path,
                            attempt_no,
                        )
                    else:
                        current_stage = "validation"
                        logger.info(
                            "issue-execution: issue #%s 已检测到代码改动，开始本地验证 branch=%s workspace=%s attempt=%s",
                            issue_number,
                            workspace.branch_name,
                            workspace.path,
                            attempt_no,
                        )
                        validation = self.validation_runner.run_all(workspace.path)
                        if not validation.success and not validation.only_test_timed_out:
                            validation_blocker = self._validation_environment_blocked_reason(validation)
                            if validation_blocker:
                                logger.warning(
                                    "issue-execution: issue #%s 本地验证被环境阻塞 branch=%s workspace=%s reason=%s",
                                    issue_number,
                                    workspace.branch_name,
                                    workspace.path,
                                    validation_blocker,
                                )
                                blocked_record = AutoCodeRunRecord(
                                    run_id=run_id,
                                    status="blocked",
                                    kind="issue",
                                    target_number=issue_number,
                                    failure_reason=validation_blocker,
                                    started_at=started_at,
                                    finished_at=utc_now_iso(),
                                )
                                self.state_store.mark_issue_plan(
                                    issue_key=planning.issue_key,
                                    issue_number=issue_number,
                                    task_type=planning.triage.task_type,
                                    updated_at=updated_at,
                                    summary=planning.plan.goal,
                                    last_outcome="blocked",
                                )
                                self.state_store.mark_run(blocked_record)
                                self._append_project_log_entry(
                                    entry_type="Issue",
                                    number=issue_number,
                                    subject=title,
                                    action=planning.triage.action or "auto_fix",
                                    result="blocked",
                                    note=validation_blocker,
                                )
                                return ExecutionResult(
                                    False,
                                    build_execution_failure_reply("validation", validation_blocker),
                                    blocked_record,
                                )
                            execution_failure = self._validation_failure_reason(validation)
                        else:
                            if validation.only_test_timed_out:
                                logger.warning(
                                    "issue-execution: issue #%s 测试超时但 format/lint 通过，继续创建 PR branch=%s",
                                    issue_number,
                                    workspace.branch_name,
                                )
                            current_stage = "inspect_patch"
                            logger.info(
                                "issue-execution: issue #%s 本地验证通过，开始检查补丁边界 branch=%s attempt=%s",
                                issue_number,
                                workspace.branch_name,
                                attempt_no,
                            )
                            inspection = self.patch_inspector.inspect(
                                workspace.path,
                                require_test_changes=planning.triage.task_type == "bug_fix",
                                allowed_paths=planning.plan.estimated_files if planning.plan else None,
                            )
                            execution_failure = "；".join(inspection.reasons) if inspection.blocked else ""

                    if not execution_failure:
                        break
                    if attempt_index >= max_repair_attempts:
                        raise RuntimeError(execution_failure)
                    repair_note = self._format_issue_repair_failure(attempt_no, current_stage, execution_failure)
                    repair_notes.append(repair_note)
                    logger.warning(
                        "issue-execution: issue #%s 第 %s 轮失败，将进入自动修复: %s",
                        issue_number,
                        attempt_no,
                        self._truncate_log_text(execution_failure, limit=400),
                    )

                if inspection is None:
                    raise RuntimeError(execution_failure or "补丁检查未完成")
                summary = "\n".join(summary_parts).strip()
                logger.info(
                    "issue-execution: validation for issue #%s: %s changed_files=%s diff=+%s/-%s tests=%s",
                    issue_number,
                    validation.summarize(),
                    len(inspection.changed_files),
                    inspection.added_lines,
                    inspection.deleted_lines,
                    inspection.has_test_changes,
                )

                current_stage = "publish"
                if not dry_run:
                    logger.info(
                        "issue-execution: issue #%s 准备提交并推送分支 branch=%s",
                        issue_number,
                        workspace.branch_name,
                    )
                    commit_message = self._build_commit_message(issue_number, planning.triage.task_type, title)
                    self.workspace_manager.commit_all(workspace.path, commit_message)
                    self.workspace_manager.push_branch(workspace.path, workspace.branch_name)
                else:
                    logger.info("issue-execution: issue #%s 处于 dry-run，跳过提交与推送", issue_number)
                pr_title = self._build_pr_title(planning.triage.task_type, title, issue_number, plan_goal=planning.plan.goal)
                pr_body = self._build_pr_body(
                    issue_number=issue_number,
                    task_type=planning.triage.task_type,
                    issue_labels=labels,
                    explicit_command=explicit_command,
                    plan=planning.plan,
                    validation=validation,
                    inspection=inspection,
                    execution_summary=summary,
                )
                existing_pr = None if dry_run else self._find_existing_open_pr(workspace.branch_name, pr_base_branch)
                reused_existing_pr = existing_pr is not None
                if existing_pr is not None and self._should_keep_existing_pr_title(
                    str(existing_pr.get("title", "") or ""),
                    planning.triage.task_type,
                    issue_number,
                ):
                    pr_title = str(existing_pr.get("title", "") or pr_title)
                logger.info(
                    "issue-execution: issue #%s 准备%sPR branch=%s base=%s title=%s",
                    issue_number,
                    "复用并更新已有" if reused_existing_pr else "创建新",
                    workspace.branch_name,
                    pr_base_branch,
                    self._truncate_log_text(pr_title, limit=100),
                )
                if existing_pr is not None:
                    pr_number = int(existing_pr.get("number", 0) or 0)
                    updated_pr = self.gh_writer.update_pull_request(pr_number, title=pr_title, body=pr_body)
                    pr = updated_pr or existing_pr
                elif not dry_run:
                    pr_cap_reason = self._check_open_pr_cap()
                    if pr_cap_reason:
                        raise RuntimeError(pr_cap_reason)
                    pr = self.gh_writer.create_pull_request(
                        title=pr_title,
                        body=pr_body,
                        head=workspace.branch_name,
                        base=pr_base_branch,
                        issue_number=issue_number,
                    )
                else:
                    pr = None
                pr_number = int(pr.get("number", 0)) if pr else 0
                if pr_number > 0 and self._should_auto_ready_for_review(pr=pr, reused_existing_pr=reused_existing_pr):
                    logger.info(
                        "issue-execution: issue #%s 准备将 PR #%s 标记为 ready for review",
                        issue_number,
                        pr_number,
                    )
                    ready_pr = self.gh_writer.mark_ready_for_review(pr_number)
                    if ready_pr is not None:
                        pr = ready_pr
                pr_url = str(pr.get("html_url", "") or "") if pr else ""
                ready_for_review = self._is_ready_for_review(pr)
                comment = self._build_issue_success_comment(
                    issue_number=issue_number,
                    branch_name=workspace.branch_name,
                    pr_number=pr_number,
                    pr_url=pr_url,
                    validation=validation,
                    inspection=inspection,
                    reused_existing_pr=reused_existing_pr,
                    ready_for_review=ready_for_review,
                    dry_run=dry_run,
                )
                final_record = AutoCodeRunRecord(
                    run_id=run_id,
                    status=self._issue_run_status(
                        reused_existing_pr=reused_existing_pr,
                        ready_for_review=ready_for_review,
                        dry_run=dry_run,
                    ),
                    kind="issue",
                    target_number=issue_number,
                    branch_name=workspace.branch_name,
                    pr_number=pr_number,
                    pr_url=pr_url,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                )
                self.state_store.mark_issue_plan(
                    issue_key=planning.issue_key,
                    issue_number=issue_number,
                    task_type=planning.triage.task_type,
                    updated_at=updated_at,
                    summary=planning.plan.goal,
                    last_outcome=final_record.status,
                )
                self.state_store.mark_run(final_record)
                note_parts = [planning.plan.goal]
                if workspace.branch_name:
                    note_parts.append(f"branch={workspace.branch_name}")
                if pr_number > 0:
                    note_parts.append(f"pr=#{pr_number}")
                logger.info(
                    "issue-execution: published issue #%s run_status=%s branch=%s pr=%s reused_existing=%s ready=%s dry_run=%s",
                    issue_number,
                    final_record.status,
                    workspace.branch_name,
                    pr_number,
                    reused_existing_pr,
                    ready_for_review,
                    dry_run,
                )
                self._append_project_log_entry(
                    entry_type="Issue",
                    number=issue_number,
                    subject=title,
                    action=planning.triage.action or "auto_fix",
                    result=final_record.status,
                    note="; ".join(note_parts),
                )
                return ExecutionResult(True, comment, final_record)
        except (RuntimeError, LLMError) as exc:
            logger.error(
                "issue-execution: failed issue #%s run_id=%s stage=%s error=%s",
                issue_number,
                run_id,
                current_stage,
                str(exc),
            )
            failure_record = AutoCodeRunRecord(
                run_id=run_id,
                status="failed",
                kind="issue",
                target_number=issue_number,
                failure_reason=str(exc),
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(failure_record)
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=title,
                action=planning.triage.action if planning is not None else "auto_fix",
                result="failed",
                note=str(exc),
            )
            return ExecutionResult(False, build_execution_failure_reply("execution", str(exc)), failure_record)
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "issue-execution: unexpected failure issue #%s run_id=%s stage=%s error=%s",
                issue_number,
                run_id,
                current_stage,
                failure_reason,
            )
            failure_record = AutoCodeRunRecord(
                run_id=run_id,
                status="failed",
                kind="issue",
                target_number=issue_number,
                failure_reason=failure_reason,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            self.state_store.mark_run(failure_record)
            self._append_project_log_entry(
                entry_type="Issue",
                number=issue_number,
                subject=title,
                action=planning.triage.action if planning is not None else "auto_fix",
                result="failed",
                note=failure_reason,
            )
            return ExecutionResult(
                False,
                build_execution_failure_reply("execution", f"未预期异常：{failure_reason}"),
                failure_record,
            )

    def _resolve_planning(
        self,
        issue: dict,
        *,
        explicit_command: bool,
        provided_planning: PlanningResult | None,
    ) -> PlanningResult:
        if provided_planning is not None:
            return provided_planning

        issue_number = int(issue["number"])
        current_updated_at = str(issue.get("updated_at", "") or "")
        # 自动流和显式流都复用缓存：只要 updated_at 未变，已冻结的计划仍然有效。
        snapshot = self.state_store.get_issue_plan_snapshot(issue_number)
        if snapshot:
            restored = PlanningResult.from_snapshot_record(snapshot)
            if restored.updated_at == current_updated_at:
                logger.info(
                    "issue-execution: 复用已缓存计划 issue #%s updated_at=%s explicit=%s",
                    issue_number,
                    current_updated_at,
                    explicit_command,
                )
                return restored

        return self.plan_issue(issue)

    def process_pull_request_queue_cycle(self) -> list[str]:
        results: list[str] = []
        tracked_prs = self._collect_tracked_pull_requests()
        repair_budget = (
            self.config.autocode.pr_queue_max_batch
            if (self.config.autocode.enable_pr_queue or self.config.autocode.poll_review_feedback)
            else 0
        )

        for pr in tracked_prs:
            pr_number = pr.get("number")
            if not isinstance(pr_number, int):
                continue
            claim = self.state_store.try_claim_tracked_pr(pr_number)
            if claim != "claimed":
                continue
            try:
                summary, consumed_budget = self._process_tracked_pull_request(
                    pr,
                    allow_repair=repair_budget > 0,
                )
                if summary:
                    results.append(summary)
                if consumed_budget:
                    repair_budget -= 1
            finally:
                self.state_store.release_tracked_pr(pr_number)
        return results

    def process_review_feedback_cycle(self) -> list[str]:
        """兼容旧入口，内部统一走 PR 队列扫描。"""
        return self.process_pull_request_queue_cycle()

    def process_pull_request(
        self,
        pr: dict,
        *,
        allow_repair: bool = True,
        current_task_id: str = "",
        current_feedback_key: str = "",
        cached_feedback: list[dict] | None = None,
        cached_review_state: str = "",
        cached_legacy_feedback_key: str = "",
    ) -> str:
        pr_number = int(pr.get("number", 0) or 0)
        if pr_number <= 0:
            return "缺少有效的 PR 编号"

        queue_source = self._tracked_pr_source(pr)
        if not queue_source:
            return f"PR #{pr_number} 不在 AutoCode 跟踪范围，跳过"

        head = pr.get("head", {}) or {}
        base = pr.get("base", {}) or {}
        self.state_store.upsert_tracked_pr(
            pr_number=pr_number,
            branch_name=str(head.get("ref", "") or ""),
            base_ref=str(base.get("ref", "") or ""),
            head_sha=str(head.get("sha", "") or ""),
            author_login=str(pr.get("user", {}).get("login", "") or ""),
            pr_url=str(pr.get("html_url", "") or ""),
            queue_source=queue_source,
        )
        claim = self.state_store.try_claim_tracked_pr(pr_number)
        if claim != "claimed":
            return f"PR #{pr_number} 当前已有任务在执行中"

        try:
            summary, _ = self._process_tracked_pull_request(
                pr,
                allow_repair=allow_repair,
                current_task_id=current_task_id,
                current_feedback_key=current_feedback_key,
                cached_feedback=cached_feedback,
                cached_review_state=cached_review_state,
                cached_legacy_feedback_key=cached_legacy_feedback_key,
            )
            return summary
        finally:
            self.state_store.release_tracked_pr(pr_number)

    def collect_tracked_pull_requests(self) -> list[dict]:
        return self._collect_tracked_pull_requests()

    def rebase_tracked_pr_if_needed(self, pr: dict) -> str | None:
        """Rebase a tracked PR branch onto latest main if it's behind.

        Returns a summary message if rebase was performed/attempted, or None if
        no rebase was needed. This is a lightweight operation that does not require
        review feedback and does not consume repair budget.
        """
        pr_number = int(pr.get("number", 0) or 0)
        if pr_number <= 0:
            return None
        head = pr.get("head", {}) or {}
        head_ref = str(head.get("ref", "") or "")
        head_sha = str(head.get("sha", "") or "")
        if not head_ref:
            return None
        queue_source = self._tracked_pr_source(pr)
        if not queue_source:
            return None
        if self.state_store.is_tracked_pr_paused(pr_number):
            return None
        tracked = self.state_store.get_tracked_pr(pr_number) or {}
        last_failed_head_sha = str(tracked.get("last_rebase_failed_head_sha", "") or "")
        if head_sha and last_failed_head_sha == head_sha:
            logger.info(
                "pr-rebase: PR #%s head=%s 上次 rebase 已失败且 head 未变化，跳过重复尝试",
                pr_number,
                head_sha[:12],
            )
            return None
        # If there is an active (queued/running) repair task for this PR,
        # skip rebase to avoid interfering with in-flight work.
        if self.state_store.has_active_tasks_for_target(pr_number, task_type="pr_feedback_repair"):
            logger.debug(
                "pr-rebase: PR #%s 有活跃修复任务，跳过 rebase",
                pr_number,
            )
            return None
        active_branch_lock = self.workspace_manager.active_branch_lock(head_ref)
        if active_branch_lock is not None:
            logger.info(
                "pr-rebase: PR #%s branch=%s 当前工作区被占用，跳过 rebase path=%s",
                pr_number,
                head_ref,
                str(active_branch_lock.get("worktree_path", "") or "") or "-",
            )
            return None

        # Only rebase PRs that actually have merge conflicts.
        # Fetch the single-PR detail to get mergeable_state (list endpoint
        # does not include it).  Skip rebase when the PR has no conflicts
        # to avoid unnecessary force-pushes that re-run CI and disrupt
        # review history.
        try:
            pr_detail = self.gh.get_pr(pr_number)
            mergeable_state = str(pr_detail.get("mergeable_state", "") or "").lower()
            if mergeable_state and mergeable_state not in ("dirty", "unknown"):
                self.state_store.clear_tracked_pr_rebase_failure(pr_number)
                logger.debug(
                    "pr-rebase: PR #%s mergeable_state=%s (无冲突)，跳过 rebase",
                    pr_number,
                    mergeable_state,
                )
                return None
            if mergeable_state:
                logger.info(
                    "pr-rebase: PR #%s mergeable_state=%s，需要 rebase",
                    pr_number,
                    mergeable_state,
                )
        except Exception as exc:
            # If we can't fetch PR detail, fall through and let the local
            # rebase logic decide (it returns False if already up-to-date).
            logger.debug(
                "pr-rebase: PR #%s 无法获取 mergeable_state (%s)，继续本地检查",
                pr_number,
                exc,
            )

        try:
            self._sync_repo_before_task("pr rebase check")
            with self.workspace_manager.prepare_existing_branch_workspace(branch_name=head_ref) as workspace:
                rebased = self.workspace_manager.rebase_onto_default_branch(workspace.path)
                if not rebased:
                    self.state_store.clear_tracked_pr_rebase_failure(pr_number)
                    return None
                self.workspace_manager.push_branch(workspace.path, workspace.branch_name, force=True)
                self.state_store.clear_tracked_pr_rebase_failure(pr_number)
                logger.info(
                    "pr-rebase: PR #%s 已成功 rebase 并推送 branch=%s",
                    pr_number,
                    head_ref,
                )
                return f"PR #{pr_number} 已自动 rebase 到最新 {self.workspace_manager.default_branch}"
        except RuntimeError as exc:
            self.state_store.mark_tracked_pr_rebase_failure(
                pr_number,
                head_sha=head_sha,
                summary=str(exc),
            )
            logger.warning(
                "pr-rebase: PR #%s rebase 失败: %s",
                pr_number,
                exc,
            )
            return f"PR #{pr_number} rebase 失败: {exc}"

    def plan_pull_request_repair(self, pr: dict) -> tuple[str, PullRequestRepairCandidate | None]:
        pr_number = int(pr.get("number", 0) or 0)
        if pr_number <= 0:
            return "缺少有效的 PR 编号", None

        queue_source = self._tracked_pr_source(pr)
        if not queue_source:
            return f"PR #{pr_number} 不在 AutoCode 跟踪范围，跳过", None

        head = pr.get("head", {}) or {}
        base = pr.get("base", {}) or {}
        current_head_sha = str(head.get("sha", "") or "")
        self.state_store.upsert_tracked_pr(
            pr_number=pr_number,
            branch_name=str(head.get("ref", "") or ""),
            base_ref=str(base.get("ref", "") or ""),
            head_sha=current_head_sha,
            author_login=str(pr.get("user", {}).get("login", "") or ""),
            pr_url=str(pr.get("html_url", "") or ""),
            queue_source=queue_source,
            pr_updated_at=str(pr.get("updated_at", "") or ""),
        )

        latest_reviews_by_user = self._latest_reviews_by_user(pr_number)
        review_state = self._summarize_latest_review_state(latest_reviews_by_user)
        tracked_record = self.state_store.get_tracked_pr(pr_number) or {}
        base_ref = str(pr.get("base", {}).get("ref", "") or "")
        pr_author_login = str(pr.get("user", {}).get("login", "") or "").strip().lower()

        if self.state_store.is_tracked_pr_paused(pr_number):
            pause = self.state_store.get_tracked_pr_pause(pr_number) or {}
            summary = f"PR #{pr_number} 已暂停自动修复: {pause.get('reason', '无说明')}"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="paused",
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        protected_branch = self._protected_publish_branch()
        if self.config.autocode.forbid_main_base_branch and base_ref == protected_branch:
            summary = f"PR #{pr_number} base={protected_branch}，按策略跳过自动修复"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="blocked",
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        if self.state_store.has_active_tasks_for_target(pr_number, task_type="pr_feedback_repair"):
            summary = f"PR #{pr_number} 当前已有 review feedback 修复任务在执行中或队列中"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="review_feedback_pending",
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        feedback = self._collect_pr_feedback(
            pr_number,
            after_updated_at=str(tracked_record.get("last_feedback_updated_at", "") or ""),
            after_item_ids=[
                int(item)
                for item in tracked_record.get("last_feedback_item_ids", [])
                if str(item).strip()
            ],
            pr_author_login=pr_author_login,
            latest_reviews_by_user=latest_reviews_by_user,
            current_head_sha=current_head_sha,
        )
        if not feedback:
            openreview_ok = self._has_openreview_approval(
                pr_number,
                latest_reviews_by_user,
                current_head_sha=current_head_sha,
            )
            status = "approved" if (review_state == "approved" or openreview_ok) else "waiting_review"
            summary = (
                f"PR #{pr_number} 已通过最新 review，继续保留监控"
                if status == "approved"
                else f"PR #{pr_number} 暂无新的可执行检视意见，等待 reviewer 进一步反馈"
            )
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status=status,
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        redirect_reason = self._find_feedback_non_pr_redirect(feedback)
        if redirect_reason:
            summary = f"PR #{pr_number} 收到 owner 明确指示，停止继续自动改代码：{redirect_reason}"
            logger.info("pr-feedback: %s", summary)
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="blocked",
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        # OpenReview 已通过 + 仅剩 bot 反馈 → 视为已就绪，跳过修复
        openreview_ok = self._has_openreview_approval(
            pr_number,
            latest_reviews_by_user,
            current_head_sha=current_head_sha,
        )
        if openreview_ok and self._is_bot_only_feedback(feedback):
            summary = (
                f"PR #{pr_number} 已通过 OpenReview 审批，忽略剩余 bot 反馈 "
                f"(bot_items={len(feedback)})"
            )
            logger.info("pr-feedback: %s", summary)
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="approved",
                summary=summary,
                review_state=review_state,
            )
            return summary, None

        self._reset_repair_limit_for_trusted_feedback_if_needed(pr_number, feedback, context="plan")
        max_repair_reason = self._check_repair_limit(pr_number, review_state)
        if max_repair_reason:
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            return max_repair_reason, None

        bot_throttle_reason = self._check_bot_feedback_throttle(pr_number, feedback, review_state)
        if bot_throttle_reason:
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            return bot_throttle_reason, None

        feedback_ids = [item["id"] for item in feedback if isinstance(item.get("id"), int)]
        feedback_marker_values = [self._feedback_marker(item) for item in feedback]
        legacy_feedback_updated_at_values = [str(item.get("updated_at", "")) for item in feedback]
        feedback_key = build_feedback_key(
            pr_number,
            feedback_ids,
            feedback_marker_values,
            item_fingerprints=[build_feedback_item_fingerprint(item) for item in feedback],
        )
        legacy_feedback_key = build_feedback_key(pr_number, feedback_ids, legacy_feedback_updated_at_values)
        task_summary = self._existing_feedback_task_summary(
            pr_number,
            feedback_key=feedback_key,
            legacy_feedback_key=legacy_feedback_key,
            review_state=review_state,
        )
        if task_summary is not None:
            summary, status = task_summary
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status=status,
                summary=summary,
                review_state=review_state,
            )
            return summary, None
        feedback_claim = self.state_store.try_claim_feedback(
            feedback_key,
            extra_feedback_keys=[legacy_feedback_key],
            owner=self._feedback_task_id(feedback_key),
        )
        if feedback_claim == "processed":
            summary = f"PR #{pr_number} 的最新检视意见已处理，等待 reviewer 复核"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="waiting_review",
                summary=summary,
                review_state=review_state,
            )
            return summary, None
        if feedback_claim == "inflight":
            return f"PR #{pr_number} 当前已有修复任务在执行中", None

        feedback_updated_at, feedback_item_ids = self._feedback_cursor(feedback)
        summary = f"PR #{pr_number} 已进入修复队列，等待后台任务消费 (feedback_items={len(feedback)})"
        logger.info(
            "pr-feedback: queued repair for PR #%s review_state=%s latest_reviews=[%s] feedback=[%s]",
            pr_number,
            review_state,
            self._latest_reviews_log_summary(latest_reviews_by_user),
            self._feedback_log_summary(feedback),
        )
        self.state_store.mark_tracked_pr_status(
            pr_number,
            status="review_feedback_pending",
            summary=summary,
            review_state=review_state,
        )
        return summary, PullRequestRepairCandidate(
            pr_number=pr_number,
            task_id=f"pr-task:{feedback_key}",
            queue_source=queue_source,
            review_state=review_state,
            feedback_key=feedback_key,
            feedback_updated_at=feedback_updated_at,
            feedback_item_ids=feedback_item_ids,
            summary=summary,
            feedback=feedback,
            legacy_feedback_key=legacy_feedback_key,
        )

    def _collect_tracked_pull_requests(self) -> list[dict]:
        per_page = max(30, self.config.server.poll_items_per_page)
        prs = self._list_all_pull_requests(
            state="open",
            sort="updated",
            direction="desc",
            per_page=per_page,
        )
        tracked_open_prs: dict[int, dict] = {}

        for pr in prs:
            pr_number = pr.get("number")
            if not isinstance(pr_number, int):
                continue
            queue_source = self._tracked_pr_source(pr)
            if not queue_source:
                continue
            head = pr.get("head", {}) or {}
            base = pr.get("base", {}) or {}
            tracked_open_prs[pr_number] = pr
            self.state_store.upsert_tracked_pr(
                pr_number=pr_number,
                branch_name=str(head.get("ref", "") or ""),
                base_ref=str(base.get("ref", "") or ""),
                head_sha=str(head.get("sha", "") or ""),
                author_login=str(pr.get("user", {}).get("login", "") or ""),
                pr_url=str(pr.get("html_url", "") or ""),
                queue_source=queue_source,
                pr_updated_at=str(pr.get("updated_at", "") or ""),
            )

        tracked_records = self.state_store.list_tracked_prs()
        tracked_by_number = {
            int(item.get("pr_number", 0) or 0): item
            for item in tracked_records
            if int(item.get("pr_number", 0) or 0) > 0
        }
        for pr_number, record in tracked_by_number.items():
            if pr_number in tracked_open_prs:
                continue
            if str(record.get("status", "") or "") == "closed":
                continue
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="closed",
                summary="PR 已关闭或已不再处于 open 状态。",
            )

        return sorted(
            tracked_open_prs.values(),
            key=lambda item: (
                str(tracked_by_number.get(int(item.get("number", 0) or 0), {}).get("first_seen_at", "")),
                int(item.get("number", 0) or 0),
            ),
        )

    def _process_tracked_pull_request(
        self,
        pr: dict,
        *,
        allow_repair: bool,
        current_task_id: str = "",
        current_feedback_key: str = "",
        cached_feedback: list[dict] | None = None,
        cached_review_state: str = "",
        cached_legacy_feedback_key: str = "",
    ) -> tuple[str, bool]:
        pr_number = int(pr["number"])
        head = pr.get("head", {}) or {}
        current_head_sha = str(head.get("sha", "") or "")
        base_ref = str(pr.get("base", {}).get("ref", "") or "")
        # 当队列传入了 cached_review_state，跳过重新拉取 reviews（plan 阶段已获取）。
        if cached_review_state:
            review_state = cached_review_state
            latest_reviews_by_user: dict[str, dict] = {}
        else:
            latest_reviews_by_user = self._latest_reviews_by_user(pr_number)
            review_state = self._summarize_latest_review_state(latest_reviews_by_user)
        tracked_record = self.state_store.get_tracked_pr(pr_number) or {}
        pr_author_login = str(pr.get("user", {}).get("login", "") or "").strip().lower()

        if self.state_store.is_tracked_pr_paused(pr_number):
            pause = self.state_store.get_tracked_pr_pause(pr_number) or {}
            summary = f"PR #{pr_number} 已暂停自动修复: {pause.get('reason', '无说明')}"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="paused",
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        protected_branch = self._protected_publish_branch()
        if self.config.autocode.forbid_main_base_branch and base_ref == protected_branch:
            summary = f"PR #{pr_number} base={protected_branch}，按策略跳过自动修复"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="blocked",
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        # 当队列传入了 cached_feedback，直接使用，跳过重新拉取。
        if cached_feedback is not None:
            feedback = self._filter_stale_openreview_feedback(
                pr_number,
                cached_feedback,
                current_head_sha=current_head_sha,
            )
        else:
            feedback = self._collect_pr_feedback(
                pr_number,
                after_updated_at=str(tracked_record.get("last_feedback_updated_at", "") or ""),
                after_item_ids=[
                    int(item)
                    for item in tracked_record.get("last_feedback_item_ids", [])
                    if str(item).strip()
                ],
                pr_author_login=pr_author_login,
                latest_reviews_by_user=latest_reviews_by_user,
                current_head_sha=current_head_sha,
            )
        if not feedback:
            openreview_ok = self._has_openreview_approval(
                pr_number,
                latest_reviews_by_user,
                current_head_sha=current_head_sha,
            )
            status = "approved" if (review_state == "approved" or openreview_ok) else "waiting_review"
            summary = (
                f"PR #{pr_number} 已通过最新 review，继续保留监控"
                if status == "approved"
                else f"PR #{pr_number} 暂无新的可执行检视意见，等待 reviewer 进一步反馈"
            )
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status=status,
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        redirect_reason = self._find_feedback_non_pr_redirect(feedback)
        if redirect_reason:
            summary = f"PR #{pr_number} 收到 owner 明确指示，停止继续自动改代码：{redirect_reason}"
            logger.info("pr-feedback: %s", summary)
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="blocked",
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        # OpenReview 已通过 + 仅剩 bot 反馈 → 视为已就绪，跳过修复
        openreview_ok = self._has_openreview_approval(
            pr_number,
            latest_reviews_by_user,
            current_head_sha=current_head_sha,
        )
        if openreview_ok and self._is_bot_only_feedback(feedback):
            summary = (
                f"PR #{pr_number} 已通过 OpenReview 审批，忽略剩余 bot 反馈 "
                f"(bot_items={len(feedback)})"
            )
            logger.info("pr-feedback: %s", summary)
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="approved",
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        self._reset_repair_limit_for_trusted_feedback_if_needed(pr_number, feedback, context="execute")
        max_repair_reason = self._check_repair_limit(pr_number, review_state)
        if max_repair_reason:
            # 推进 cursor 避免每轮 poll 重复评估同一批 feedback
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            return max_repair_reason, False

        bot_throttle_reason = self._check_bot_feedback_throttle(pr_number, feedback, review_state)
        if bot_throttle_reason:
            # 推进 cursor 避免每轮 poll 重复评估同一批 feedback
            self._advance_feedback_cursor_on_block(pr_number, feedback)
            return bot_throttle_reason, False

        feedback_ids = [item["id"] for item in feedback if isinstance(item.get("id"), int)]
        feedback_marker_values = [self._feedback_marker(item) for item in feedback]
        legacy_feedback_updated_at_values = [str(item.get("updated_at", "")) for item in feedback]
        feedback_key = build_feedback_key(
            pr_number,
            feedback_ids,
            feedback_marker_values,
            item_fingerprints=[build_feedback_item_fingerprint(item) for item in feedback],
        )
        legacy_feedback_key = build_feedback_key(pr_number, feedback_ids, legacy_feedback_updated_at_values)
        task_summary = self._existing_feedback_task_summary(
            pr_number,
            feedback_key=feedback_key,
            legacy_feedback_key=legacy_feedback_key,
            review_state=review_state,
            ignore_task_ids={str(current_task_id or "").strip()} if str(current_task_id or "").strip() else None,
        )
        if task_summary is not None:
            summary, status = task_summary
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status=status,
                summary=summary,
                review_state=review_state,
            )
            return summary, False
        claim_owner = str(current_task_id or "").strip() or str(current_feedback_key or "").strip()
        if not claim_owner:
            claim_owner = self._feedback_task_id(feedback_key)
        owned_by_current_task = bool(str(current_task_id or "").strip()) and self.state_store.feedback_claim_owned_by(
            feedback_key,
            owner=str(current_task_id or "").strip(),
            extra_feedback_keys=[legacy_feedback_key],
        )
        if owned_by_current_task:
            feedback_claim = "claimed"
        else:
            feedback_claim = self.state_store.try_claim_feedback(
                feedback_key,
                extra_feedback_keys=[legacy_feedback_key],
                owner=claim_owner,
            )

        if feedback_claim == "processed":
            summary = f"PR #{pr_number} 的最新检视意见已处理，等待 reviewer 复核"
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="waiting_review",
                summary=summary,
                review_state=review_state,
            )
            return summary, False
        if feedback_claim == "inflight":
            return f"PR #{pr_number} 当前已有修复任务在执行中", False
        if not allow_repair:
            self.state_store.release_feedback(feedback_key, owner=claim_owner)
            summary = f"PR #{pr_number} 已进入修复队列，等待轮到本次批次处理 (feedback_items={len(feedback)})"
            logger.info(
                "pr-feedback: pending repair for PR #%s review_state=%s latest_reviews=[%s] feedback=[%s]",
                pr_number,
                review_state,
                self._latest_reviews_log_summary(latest_reviews_by_user),
                self._feedback_log_summary(feedback),
            )
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="review_feedback_pending",
                summary=summary,
                review_state=review_state,
            )
            return summary, False

        try:
            summary = self._apply_review_feedback(pr, feedback)
            self.state_store.mark_feedback_processed(
                feedback_key=feedback_key,
                pr_number=pr_number,
                summary=summary,
                extra_feedback_keys=[legacy_feedback_key],
            )
            latest_feedback_updated_at, latest_feedback_ids = self._feedback_cursor(feedback)
            self.state_store.update_tracked_pr_feedback_cursor(
                pr_number,
                updated_at=latest_feedback_updated_at,
                item_ids=latest_feedback_ids,
            )
            return summary, True
        except (RuntimeError, LLMError):
            # 修复失败时也推进 cursor，避免同一批 feedback 无限重试消耗 LLM。
            # 下次新 review 会产生新 feedback 重新进入队列。
            latest_feedback_updated_at, latest_feedback_ids = self._feedback_cursor(feedback)
            self.state_store.update_tracked_pr_feedback_cursor(
                pr_number,
                updated_at=latest_feedback_updated_at,
                item_ids=latest_feedback_ids,
            )
            raise
        finally:
            self.state_store.release_feedback(feedback_key, owner=claim_owner)

    def _tracked_pr_source(self, pr: dict) -> str:
        if not (self.config.autocode.enable_pr_queue or self.config.autocode.poll_review_feedback):
            return ""
        head = pr.get("head", {}) or {}
        head_ref = str(head.get("ref", "") or "")
        head_repo = head.get("repo", {}) or {}
        head_repo_full_name = str(head_repo.get("full_name", "") or "")
        if head_repo_full_name and head_repo_full_name != self.config.github.repo:
            return ""

        if head_ref.startswith(f"{self.config.autocode.branch_prefix}/"):
            return "autocode_branch"

        author_login = str(pr.get("user", {}).get("login", "") or "").lower()
        if (
            self.config.autocode.enable_pr_queue
            and author_login
            and author_login in self.config.autocode.tracked_pr_authors
        ):
            return "author_allowlist"
        return ""

    def _summarize_review_state(self, pr_number: int) -> str:
        return self._summarize_latest_review_state(self._latest_reviews_by_user(pr_number))

    def _summarize_latest_review_state(self, latest_reviews_by_user: dict[str, dict]) -> str:
        latest_states = {
            str(review.get("state", "") or "").upper()
            for review in latest_reviews_by_user.values()
            if str(review.get("state", "") or "").strip()
        }
        if "CHANGES_REQUESTED" in latest_states:
            return "changes_requested"
        if "APPROVED" in latest_states:
            return "approved"
        if "COMMENTED" in latest_states:
            return "commented"
        return "pending"

    def _has_openreview_approval(
        self,
        pr_number: int,
        latest_reviews_by_user: dict[str, dict] | None = None,
        *,
        current_head_sha: str = "",
    ) -> bool:
        """Return True if the latest OpenReview verdict is '可直接合入：可'.

        Scans all reviews (not just latest-per-user) to find the most recent
        review carrying the ``openreview:idempotency`` marker, then parses the
        verdict line.
        """
        reviews = self._list_all_pr_reviews(pr_number) if latest_reviews_by_user is None else []

        # Prefer scanning all reviews when available; fall back to
        # latest_reviews_by_user for the zero-extra-API-call path.
        candidates: list[dict] = []
        if reviews:
            candidates = list(reversed(reviews))
        elif latest_reviews_by_user:
            candidates = list(latest_reviews_by_user.values())

        for review in candidates:
            body = str(review.get("body", "") or "")
            if _OPENREVIEW_BODY_MARKER not in body:
                continue
            if not self._review_targets_current_head(review, current_head_sha=current_head_sha):
                continue
            m = _OPENREVIEW_VERDICT_RE.search(body)
            if m:
                return m.group(1) == "可"
            # Marker present but no parseable verdict — treat as unknown.
            return False
        return False

    def _apply_review_feedback(self, pr: dict, feedback: list[dict]) -> str:
        pr_number = int(pr["number"])
        head_ref = str(pr.get("head", {}).get("ref", "") or "")
        title = str(pr.get("title", "") or "")
        body = str(pr.get("body", "") or "")
        dry_run = self._is_dry_run()
        if self.workspace_manager.is_protected_branch(head_ref):
            raise RuntimeError(f"禁止直接修改受保护分支: {head_ref}")
        run_id = self._build_run_id("feedback", pr_number)
        started_at = utc_now_iso()
        record = AutoCodeRunRecord(
            run_id=run_id,
            status="running",
            kind="review_feedback",
            target_number=pr_number,
            branch_name=head_ref,
            started_at=started_at,
        )
        self.state_store.mark_run(record)
        comments = [self._feedback_text(item) for item in feedback]
        comments = [item for item in comments if item]
        if not comments:
            return f"PR #{pr_number} 无可执行 review 反馈"
        pr_title = str(pr.get("title", "") or f"PR #{pr_number}")
        pr_planning = self._planning_for_pull_request(pr)
        current_stage = "sync_repo"
        logger.info(
            "pr-feedback: starting repair for PR #%s branch=%s feedback=[%s]",
            pr_number,
            head_ref,
            self._feedback_log_summary(feedback),
        )

        try:
            self._sync_repo_before_task("review feedback repair")
            current_stage = "prepare_workspace"
            logger.info(
                "pr-feedback: PR #%s 准备进入修复工作区 branch=%s feedback_items=%s",
                pr_number,
                head_ref,
                len(comments),
            )
            with self.workspace_manager.prepare_existing_branch_workspace(branch_name=head_ref) as workspace:
                logger.info(
                    "pr-feedback: PR #%s 已进入修复工作区 branch=%s workspace=%s base_ref=%s",
                    pr_number,
                    workspace.branch_name,
                    workspace.path,
                    workspace.base_ref,
                )
                # Rebase onto latest main to resolve merge conflicts
                current_stage = "rebase"
                try:
                    rebased = self.workspace_manager.rebase_onto_default_branch(workspace.path)
                    if rebased:
                        logger.info(
                            "pr-feedback: PR #%s 已成功 rebase 到最新 %s",
                            pr_number,
                            self.workspace_manager.default_branch,
                        )
                except RuntimeError as rebase_exc:
                    logger.warning(
                        "pr-feedback: PR #%s rebase 失败，继续在当前分支上修复: %s",
                        pr_number,
                        rebase_exc,
                    )
                    rebased = False
                workspace_executor = self._review_feedback_executor_for_workspace(
                    workspace.path,
                    feedback=feedback,
                    log_context=f"pr-feedback#{pr_number}",
                )
                feedback_context = self._build_workspace_context(workspace.path, title, body)
                current_repair_round = self.state_store.get_tracked_pr_repair_count(pr_number)
                previous_repair_notes = list(self._get_previous_failure_reasons("review_feedback", pr_number))
                max_repair_attempts = max(
                    int(getattr(self.config.autocode, "review_feedback_repair_max_attempts", 2) or 0),
                    0,
                )
                changed_files_from_body = self._extract_pr_changed_files_from_body(body)
                remote_changed_files = self._safe_get_pr_files(pr_number)
                prompt_changed_files = remote_changed_files or changed_files_from_body
                documentation_only_scope = self._is_documentation_only_pr(
                    pr,
                    feedback=feedback,
                    changed_files=remote_changed_files,
                )
                documentation_allowed_paths = self._documentation_feedback_allowed_paths(
                    feedback=feedback,
                    changed_files=remote_changed_files + changed_files_from_body,
                )
                if documentation_only_scope:
                    contaminating_files = [
                        path for path in remote_changed_files
                        if path and not self._is_documentation_scope_path(path)
                    ]
                    if contaminating_files and not self._feedback_requests_scope_cleanup(feedback, contaminating_files):
                        raise RuntimeError(
                            "文档/治理 PR 已包含非文档改动，暂停自动修复以避免继续扩大范围: "
                            + ", ".join(contaminating_files[:5])
                        )
                    logger.info(
                        "pr-feedback: PR #%s 识别为文档/治理范围，启用硬范围锁 allowed_paths=%s",
                        pr_number,
                        self._summarize_list(documentation_allowed_paths),
                    )
                scope_guard = self._review_feedback_scope_guard(
                    documentation_only=documentation_only_scope,
                    allowed_paths=documentation_allowed_paths,
                )
                summary_parts: list[str] = []
                cli_output = ""
                validation = ValidationResult(success=True, steps=[])
                inspection: PatchInspectionResult | None = None
                execution_failure = ""

                for attempt_index in range(max_repair_attempts + 1):
                    attempt_no = attempt_index + 1
                    is_repair_attempt = attempt_index > 0
                    prompt = build_review_feedback_prompt(
                        pr_title=title,
                        pr_body=body,
                        repo_name=self.config.github.repo,
                        review_feedback=comments,
                        local_code_context=feedback_context,
                        changed_files=prompt_changed_files,
                        scope_guard=scope_guard,
                        repair_round=current_repair_round + attempt_no,
                        previous_repair_notes=previous_repair_notes,
                    )
                    current_stage = "cli_repair" if is_repair_attempt else "cli_execute"
                    logger.info(
                        "pr-feedback: PR #%s 开始调用%s修复执行器 attempt=%s/%s feedback_items=%s",
                        pr_number,
                        "再次" if is_repair_attempt else "",
                        attempt_no,
                        max_repair_attempts + 1,
                        len(comments),
                    )
                    attempt_output = workspace_executor.analyze(
                        prompt,
                        system=(
                            "你是 AutoCode review feedback 二次修复器，必须先修复上一轮失败原因，只处理评论中明确要求的问题。"
                            if is_repair_attempt
                            else "你是 AutoCode review feedback 修复器，只修复评论中明确要求的问题。"
                        ),
                    )
                    if attempt_output:
                        summary_parts.append(f"第 {attempt_no} 轮：{attempt_output.strip()}")
                        cli_output = "\n".join(summary_parts).strip()
                        logger.info(
                            "pr-feedback: CLI 输出摘要 for PR #%s attempt=%s: %s",
                            pr_number,
                            attempt_no,
                            self._truncate_log_text(attempt_output, limit=600),
                        )
                    execution_failure = ""
                    if not self.workspace_manager.has_changes(workspace.path):
                        _can_body_refresh = pr_planning is not None
                        _is_body_only = self._is_pr_body_only_feedback(feedback)
                        if _is_body_only and _can_body_refresh:
                            # If we rebased, push the rebased branch even without new code changes.
                            if rebased and not dry_run:
                                self.workspace_manager.push_branch(workspace.path, workspace.branch_name, force=True)
                                logger.info(
                                    "pr-feedback: PR #%s 无新代码改动但已推送 rebase 结果以解决合并冲突",
                                    pr_number,
                                )
                            validation = ValidationResult(success=True, steps=[])
                            inspection, existing_pr_body = self._live_patch_inspection_for_pr(pr)
                            current_stage = "publish"
                            logger.info(
                                "pr-feedback: PR #%s 未检测到代码改动，反馈仅涉及 PR 描述，改为刷新 PR 描述",
                                pr_number,
                            )
                            if not dry_run:
                                refreshed_body = self._build_pr_body(
                                    issue_number=pr_planning.issue_number,
                                    task_type=pr_planning.triage.task_type,
                                    issue_labels=pr_planning.labels,
                                    explicit_command=False,
                                    plan=pr_planning.plan,
                                    validation=validation,
                                    inspection=inspection,
                                    execution_summary=cli_output,
                                    existing_pr_body=existing_pr_body,
                                )
                                self.gh_writer.update_pull_request(pr_number, body=refreshed_body)
                                logger.info(
                                    "pr-feedback: PR #%s 已在无代码 diff 情况下刷新 PR 描述 issue=%s",
                                    pr_number,
                                    pr_planning.issue_number,
                                )
                            else:
                                logger.info(
                                    "pr-feedback: PR #%s review 反馈无代码改动且处于 dry-run，跳过远端更新",
                                    pr_number,
                                )
                            self.state_store.mark_tracked_pr_status(
                                pr_number,
                                status="dry_run" if dry_run else "updated",
                                summary=(
                                    f"PR #{pr_number} dry-run 已完成 review 反馈检查，建议刷新 PR 描述"
                                    if dry_run
                                    else f"PR #{pr_number} 已根据 review 反馈刷新 PR 描述 (body_only)"
                                ),
                                review_state="commented",
                            )
                            final_record = AutoCodeRunRecord(
                                run_id=run_id,
                                status="dry_run" if dry_run else "done",
                                kind="review_feedback",
                                target_number=pr_number,
                                branch_name=workspace.branch_name,
                                started_at=started_at,
                                finished_at=utc_now_iso(),
                            )
                            self.state_store.mark_run(final_record)
                            self._append_project_log_entry(
                                entry_type="PR",
                                number=pr_number,
                                subject=pr_title,
                                action="review_feedback",
                                result="dry_run" if dry_run else "body_only",
                                note=f"{len(comments)} feedback item(s)",
                            )
                            if dry_run:
                                return f"PR #{pr_number} dry-run 已完成 review 反馈检查，建议刷新 PR 描述"
                            return f"PR #{pr_number} 已根据 review 反馈刷新 PR 描述"
                        execution_failure = "review feedback 执行后未产生改动"
                    else:
                        if documentation_only_scope:
                            current_stage = "inspect_patch"
                            logger.info(
                                "pr-feedback: PR #%s 已检测到文档范围改动，先检查补丁边界 attempt=%s",
                                pr_number,
                                attempt_no,
                            )
                            inspection = self.patch_inspector.inspect(
                                workspace.path,
                                require_test_changes=False,
                                allowed_paths=documentation_allowed_paths,
                                documentation_only=True,
                            )
                            execution_failure = "；".join(inspection.reasons) if inspection.blocked else ""
                            if not execution_failure:
                                current_stage = "validation"
                                logger.info(
                                    "pr-feedback: PR #%s 文档范围检查通过，执行轻量文档验证 attempt=%s",
                                    pr_number,
                                    attempt_no,
                                )
                                validation = self._run_documentation_only_validation(workspace.path)
                                if not validation.success:
                                    execution_failure = self._validation_failure_reason(validation)
                        else:
                            current_stage = "validation"
                            logger.info("pr-feedback: PR #%s 已检测到代码改动，开始本地验证 attempt=%s", pr_number, attempt_no)
                            validation = self.validation_runner.run_all(
                                workspace.path,
                                test_timeout_override=self.config.autocode.repair_test_timeout_seconds,
                            )
                            if not validation.success and not validation.only_test_timed_out:
                                execution_failure = self._validation_failure_reason(validation)
                            else:
                                if validation.only_test_timed_out:
                                    logger.warning(
                                        "pr-feedback: PR #%s 测试超时但 format/lint 通过，继续推送修复",
                                        pr_number,
                                    )
                                current_stage = "inspect_patch"
                                logger.info("pr-feedback: PR #%s 本地验证通过，开始检查补丁边界 attempt=%s", pr_number, attempt_no)
                                inspection = self.patch_inspector.inspect(workspace.path, require_test_changes=False)
                                execution_failure = "；".join(inspection.reasons) if inspection.blocked else ""
                    if not execution_failure:
                        break
                    if attempt_index >= max_repair_attempts:
                        raise RuntimeError(execution_failure)
                    repair_note = self._format_review_feedback_repair_failure(attempt_no, current_stage, execution_failure)
                    previous_repair_notes.append(repair_note)
                    logger.warning(
                        "pr-feedback: PR #%s 第 %s 轮失败，将进入自动修复: %s",
                        pr_number,
                        attempt_no,
                        self._truncate_log_text(execution_failure, limit=400),
                    )

                if inspection is None:
                    raise RuntimeError(execution_failure or "review feedback 补丁检查未完成")
                current_stage = "publish"
                if not dry_run:
                    logger.info(
                        "pr-feedback: PR #%s 准备提交并推送修复 branch=%s",
                        pr_number,
                        workspace.branch_name,
                    )
                    self.workspace_manager.commit_all(
                        workspace.path,
                        f"fix(review-feedback-{pr_number}): address latest review comments",
                    )
                    self.workspace_manager.push_branch(workspace.path, workspace.branch_name, force=rebased)
                    if pr_planning is not None:
                        # 等待 GitHub 重新计算 diff 统计，避免 PR 描述与实际不一致
                        time.sleep(12)
                        live_inspection, existing_pr_body = self._live_patch_inspection_for_pr(pr)
                        refreshed_body = self._build_pr_body(
                            issue_number=pr_planning.issue_number,
                            task_type=pr_planning.triage.task_type,
                            issue_labels=pr_planning.labels,
                            explicit_command=False,
                            plan=pr_planning.plan,
                            validation=validation,
                            inspection=live_inspection,
                            execution_summary=cli_output,
                            existing_pr_body=existing_pr_body,
                        )
                        self.gh_writer.update_pull_request(pr_number, body=refreshed_body)
                        logger.info(
                            "pr-feedback: PR #%s 已根据最新 diff 刷新 PR 描述 issue=%s",
                            pr_number,
                            pr_planning.issue_number,
                        )
                else:
                    logger.info("pr-feedback: PR #%s 处于 dry-run，跳过提交与推送", pr_number)
                if not dry_run:
                    # 代码修复成功推送后递增修复计数
                    repair_round = self.state_store.increment_tracked_pr_repair_count(
                        pr_number, feedback_count=len(comments),
                    )
                    logger.info(
                        "pr-feedback: PR #%s 修复计数已更新 repair_round=%s feedback_count=%s",
                        pr_number, repair_round, len(comments),
                    )
                    if self.state_store.is_repair_diverging(pr_number):
                        logger.warning(
                            "pr-feedback: PR #%s feedback 数量连续递增，判定为发散，停止自动修复",
                            pr_number,
                        )
                        self.state_store.mark_tracked_pr_status(
                            pr_number,
                            status="repair_limit_reached",
                            summary=f"PR #{pr_number} feedback 数量连续递增（越改越差），停止自动修复",
                            review_state="changes_requested",
                        )
                self.state_store.mark_tracked_pr_status(
                    pr_number,
                    status="dry_run" if dry_run else "updated",
                    summary=(
                        f"PR #{pr_number} dry-run 已完成 review 反馈修复验证，未实际推送分支"
                        if dry_run
                        else f"PR #{pr_number} 已推送最新修复，等待 reviewer 复核"
                    ),
                    review_state="commented",
                )
                final_record = AutoCodeRunRecord(
                    run_id=run_id,
                    status="dry_run" if dry_run else "done",
                    kind="review_feedback",
                    target_number=pr_number,
                    branch_name=workspace.branch_name,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                )
                self.state_store.mark_run(final_record)
                self._append_project_log_entry(
                    entry_type="PR",
                    number=pr_number,
                    subject=pr_title,
                    action="review_feedback",
                    result="dry_run" if dry_run else "updated",
                    note=f"{len(comments)} feedback item(s)",
                )
                if dry_run:
                    return f"PR #{pr_number} dry-run 已完成 review 反馈修复验证"
                return f"PR #{pr_number} 已根据 review 反馈更新分支"
        except (RuntimeError, LLMError) as exc:
            logger.error(
                "pr-feedback: failed PR #%s run_id=%s stage=%s error=%s",
                pr_number,
                run_id,
                current_stage,
                str(exc),
            )
            # 失败也计入修复计数，防止无限重试（dry-run 不递增，与成功路径保持一致）
            if not dry_run:
                self.state_store.increment_tracked_pr_repair_count(
                    pr_number, feedback_count=len(comments),
                )
            self.state_store.mark_run(
                AutoCodeRunRecord(
                    run_id=run_id,
                    status="failed",
                    kind="review_feedback",
                    target_number=pr_number,
                    branch_name=head_ref,
                    failure_reason=str(exc),
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                )
            )
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="failed",
                summary=f"PR #{pr_number} review 反馈处理失败: {exc}；下一步：{self._failure_next_step(current_stage, str(exc))}",
                review_state="changes_requested",
            )
            self._append_project_log_entry(
                entry_type="PR",
                number=pr_number,
                subject=pr_title,
                action="review_feedback",
                result="failed",
                note=str(exc),
            )
            raise RuntimeError(f"PR #{pr_number} review 反馈处理失败: {exc}") from exc
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "pr-feedback: unexpected failure PR #%s run_id=%s stage=%s error=%s",
                pr_number,
                run_id,
                current_stage,
                failure_reason,
            )
            self.state_store.mark_run(
                AutoCodeRunRecord(
                    run_id=run_id,
                    status="failed",
                    kind="review_feedback",
                    target_number=pr_number,
                    branch_name=head_ref,
                    failure_reason=failure_reason,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                )
            )
            self.state_store.mark_tracked_pr_status(
                pr_number,
                status="failed",
                summary=(
                    f"PR #{pr_number} review 反馈处理出现未预期异常: {failure_reason}；"
                    f"下一步：{self._failure_next_step(current_stage, failure_reason)}"
                ),
                review_state="changes_requested",
            )
            self._append_project_log_entry(
                entry_type="PR",
                number=pr_number,
                subject=pr_title,
                action="review_feedback",
                result="failed",
                note=failure_reason,
            )
            raise RuntimeError(f"PR #{pr_number} review 反馈处理出现未预期异常: {failure_reason}") from exc

    def _collect_pr_feedback(
        self,
        pr_number: int,
        *,
        after_updated_at: str = "",
        after_item_ids: list[int] | None = None,
        pr_author_login: str = "",
        latest_reviews_by_user: dict[str, dict] | None = None,
        current_head_sha: str = "",
    ) -> list[dict]:
        after_ids = {int(item) for item in (after_item_ids or []) if int(item) > 0}
        feedback: list[dict] = []
        latest_reviews_by_user = latest_reviews_by_user or self._latest_reviews_by_user(pr_number)
        actionable_latest_reviews: dict[str, dict] = {}
        skipped_stale_openreview = 0
        for user_login, review in latest_reviews_by_user.items():
            if str(review.get("state", "") or "").upper() not in _ACTIONABLE_REVIEW_STATES:
                continue
            if self._is_stale_openreview_review(review, current_head_sha=current_head_sha):
                skipped_stale_openreview += 1
                continue
            actionable_latest_reviews[user_login] = review
        if skipped_stale_openreview:
            logger.info(
                "pr-feedback: PR #%s 跳过非当前 head 的 OpenReview 评审 items=%s",
                pr_number,
                skipped_stale_openreview,
            )

        allowed_review_ids = {
            int(review.get("id", 0) or 0)
            for review in actionable_latest_reviews.values()
            if int(review.get("id", 0) or 0) > 0
        }

        for review in actionable_latest_reviews.values():
            body = str(review.get("body", "") or "").strip()
            if (
                body
                and self._should_include_review_body(review, pr_author_login=pr_author_login)
                and self._is_feedback_item_pending(review, after_updated_at=after_updated_at, after_ids=after_ids)
            ):
                feedback.append(review)

        for comment in self._list_all_pr_review_comments(pr_number):
            body = str(comment.get("body", "") or "").strip()
            if not body:
                continue
            if self._is_outdated_review_comment(comment):
                continue
            review_id = int(comment.get("pull_request_review_id", 0) or 0)
            if review_id > 0:
                if review_id in allowed_review_ids and self._is_feedback_item_pending(
                    comment,
                    after_updated_at=after_updated_at,
                    after_ids=after_ids,
                ):
                    feedback.append(comment)
                continue

            if self._is_feedback_item_pending(comment, after_updated_at=after_updated_at, after_ids=after_ids):
                feedback.append(comment)
        feedback.sort(key=self._feedback_sort_key)
        return self._focus_feedback_on_latest_openreview(feedback)

    def _filter_stale_openreview_feedback(
        self,
        pr_number: int,
        feedback: list[dict],
        *,
        current_head_sha: str = "",
    ) -> list[dict]:
        if not feedback or not current_head_sha:
            return list(feedback)
        stale_review_ids: set[int] = set()
        filtered: list[dict] = []
        skipped = 0
        for item in feedback:
            if self._is_stale_openreview_review(item, current_head_sha=current_head_sha):
                review_id = int(item.get("id", 0) or 0)
                if review_id > 0:
                    stale_review_ids.add(review_id)
                skipped += 1
                continue
            filtered.append(item)
        if stale_review_ids:
            linked_filtered: list[dict] = []
            for item in filtered:
                review_id = int(item.get("pull_request_review_id", 0) or 0)
                if review_id in stale_review_ids:
                    skipped += 1
                    continue
                linked_filtered.append(item)
            filtered = linked_filtered
        if skipped:
            logger.info(
                "pr-feedback: PR #%s 跳过非当前 head 的 OpenReview cached feedback items=%s",
                pr_number,
                skipped,
            )
        return filtered

    def _is_stale_openreview_review(self, review: dict, *, current_head_sha: str = "") -> bool:
        body = str(review.get("body", "") or "")
        if not self._looks_like_openreview_review_body(body):
            return False
        return not self._review_targets_current_head(review, current_head_sha=current_head_sha)

    def _review_targets_current_head(self, review: dict, *, current_head_sha: str = "") -> bool:
        current = str(current_head_sha or "").strip()
        if not current:
            return True
        review_sha = self._review_commit_sha(review)
        if not review_sha:
            return True
        return review_sha == current

    def _review_commit_sha(self, review: dict) -> str:
        for key in ("commit_id", "commit_sha", "head_sha"):
            value = str(review.get(key, "") or "").strip()
            if value:
                return value
        commit = review.get("commit", {}) or {}
        if isinstance(commit, dict):
            for key in ("oid", "sha"):
                value = str(commit.get(key, "") or "").strip()
                if value:
                    return value
        return ""

    def _feedback_text(self, item: dict) -> str:
        body = str(item.get("body", "") or "").strip()
        if not body:
            return ""
        path = str(item.get("path", "") or "").strip()
        if path:
            line = int(item.get("original_line", 0) or item.get("line", 0) or 0)
            side = str(item.get("side", "") or "").strip()
            if line > 0 and side:
                return f"{path}:{line} ({side}): {body}"
            if line > 0:
                return f"{path}:{line}: {body}"
            return f"{path}: {body}"
        return body

    def _latest_reviews_by_user(self, pr_number: int) -> dict[str, dict]:
        latest_by_user: dict[str, dict] = {}
        for review in self._list_all_pr_reviews(pr_number):
            user_login = str(review.get("user", {}).get("login", "") or "").lower()
            if not user_login:
                continue
            existing = latest_by_user.get(user_login)
            if existing is None or self._feedback_sort_key(review) >= self._feedback_sort_key(existing):
                latest_by_user[user_login] = review
        return latest_by_user

    def _build_workspace_context(self, workspace_path: Path, title: str, body: str) -> str:
        builder = LocalContextBuilder(
            repo_path=str(workspace_path),
            max_context_files=self.config.local_repo.max_context_files,
            max_file_chars=self.config.local_repo.max_file_chars,
            max_total_chars=self.config.local_repo.max_total_context_chars,
        )
        return builder.build_issue_context(title=title, body=body)

    def _planner_with_context(self, log_context: str) -> BaseLLMAdapter:
        return self.planner.with_log_context(log_context)

    def _execution_adapter_settings(self, task_type: str) -> dict[str, str | int]:
        primary_model = self.config.cli.primary_model
        primary_fallback_model = self.config.cli.primary_fallback_model
        cli_timeout = self.config.cli.executor_cli_timeout
        max_idle_seconds = int(getattr(self.config.cli, "executor_max_idle_seconds", 0) or 0)
        if max_idle_seconds <= 0:
            max_idle_seconds = min(int(cli_timeout or 0), 300)
        profile = "default"
        if task_type == "todo_refactor":
            profile = "todo_refactor"
            cli_timeout = self.config.cli.todo_refactor_executor_cli_timeout
            max_idle_seconds = self.config.cli.todo_refactor_executor_max_idle_seconds
            preferred_model = self.config.cli.todo_refactor_executor_model
            if preferred_model:
                primary_model = preferred_model
                if preferred_model != self.config.cli.primary_model:
                    primary_fallback_model = self.config.cli.primary_model
        return {
            "profile": profile,
            "cli_timeout": cli_timeout,
            "max_idle_seconds": max_idle_seconds,
            "primary_model": primary_model,
            "primary_fallback_model": primary_fallback_model,
            "fallback_clis": ",".join(self.config.cli.fallback_clis) or "-",
        }

    def _build_custom_execution_adapter(
        self,
        *,
        workspace_path: Path,
        primary_model: str,
        primary_fallback_model: str,
        cli_timeout: int,
        max_idle_seconds: int,
        log_context: str = "",
    ) -> BaseLLMAdapter:
        adapter = create_adapter_from_profiles(
            cli_profiles=self.config.cli.cli_profiles,
            primary_cli=self.config.cli.primary_cli,
            fallback_clis=self.config.cli.fallback_clis,
            cli_timeout=cli_timeout,
            max_idle_seconds=max_idle_seconds,
            working_dir=str(workspace_path),
            execution_mode=True,
            primary_model_override=primary_model,
            primary_fallback_model_override=primary_fallback_model,
        )
        if log_context:
            adapter = adapter.with_log_context(log_context)
        return adapter

    def _executor_for_workspace(
        self,
        workspace_path: Path,
        *,
        task_type: str = "",
        log_context: str = "",
    ) -> BaseLLMAdapter:
        settings = self._execution_adapter_settings(task_type)
        if settings["profile"] == "todo_refactor":
            adapter = self._build_custom_execution_adapter(
                workspace_path=workspace_path,
                primary_model=str(settings["primary_model"] or ""),
                primary_fallback_model=str(settings["primary_fallback_model"] or ""),
                cli_timeout=int(settings["cli_timeout"] or 0),
                max_idle_seconds=int(settings["max_idle_seconds"] or 0),
            )
        else:
            adapter = self.executor.with_working_dir(str(workspace_path))
        if log_context:
            adapter = adapter.with_log_context(log_context)
        return adapter

    def _review_feedback_executor_for_workspace(
        self,
        workspace_path: Path,
        *,
        feedback: list[dict],
        log_context: str = "",
    ) -> BaseLLMAdapter:
        # simple_model 优先级: 全局 > primary profile simple_model
        simple_model = str(getattr(self.config.cli, "simple_executor_model", "") or "").strip()
        if not simple_model:
            primary_profile = self.config.cli.cli_profiles.get(self.config.cli.primary_cli)
            simple_model = (primary_profile.simple_model if primary_profile else "") or ""
        max_items = max(int(getattr(self.config.cli, "simple_executor_max_feedback_items", 0) or 0), 0)
        if simple_model and max_items > 0 and len(feedback) <= max_items and self._is_simple_review_feedback(feedback):
            primary_fallback_model = self.config.cli.primary_model
            if simple_model == self.config.cli.primary_model:
                primary_fallback_model = self.config.cli.primary_fallback_model
            logger.info(
                "pr-feedback: using simple executor model=%s feedback_items=%s threshold=%s",
                simple_model,
                len(feedback),
                max_items,
            )
            return self._build_custom_execution_adapter(
                workspace_path=workspace_path,
                primary_model=simple_model,
                primary_fallback_model=str(primary_fallback_model or ""),
                cli_timeout=int(self.config.cli.executor_cli_timeout or 0),
                max_idle_seconds=max(
                    int(getattr(self.config.cli, "executor_max_idle_seconds", 0) or 0),
                    min(int(self.config.cli.executor_cli_timeout or 0), 300),
                ),
                log_context=log_context,
            )
        adapter = self.executor.with_working_dir(str(workspace_path))
        if log_context:
            adapter = adapter.with_log_context(log_context)
        return adapter

    def _is_simple_review_feedback(self, feedback: list[dict]) -> bool:
        if not feedback:
            return False
        for item in feedback:
            path = str(item.get("path", "") or "").strip().lower()
            body = str(item.get("body", "") or "").strip().lower()
            if path.endswith(".md") or path.startswith("docs/") or path == "readme.md":
                continue
            if any(marker in body for marker in _SIMPLE_PR_FEEDBACK_MARKERS):
                continue
            return False
        return True

    def _extract_issue_number_from_pr(self, pr: dict) -> int:
        body = str(pr.get("body", "") or "")
        title = str(pr.get("title", "") or "")
        match = _PR_LINKED_ISSUE_RE.search(body)
        if match:
            return int(match.group(1))
        match = _PR_TITLE_ISSUE_RE.search(title)
        if match:
            return int(match.group(1))
        return 0

    def _planning_for_pull_request(self, pr: dict) -> PlanningResult | None:
        issue_number = self._extract_issue_number_from_pr(pr)
        if issue_number <= 0:
            return None
        snapshot = self.state_store.get_issue_plan_snapshot(issue_number)
        if not snapshot:
            return None
        try:
            return PlanningResult.from_snapshot_record(snapshot)
        except Exception:
            logger.warning("pr-feedback: failed to parse issue plan snapshot for issue #%s", issue_number)
            return None

    def _list_all_pull_requests(
        self,
        *,
        state: str,
        sort: str,
        direction: str,
        per_page: int,
    ) -> list[dict]:
        prs: list[dict] = []
        page = 1
        while True:
            batch = self.gh.list_pull_requests(
                state=state,
                sort=sort,
                direction=direction,
                per_page=per_page,
                page=page,
            )
            if not batch:
                break
            prs.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return prs

    def _list_all_pr_reviews(self, pr_number: int, *, per_page: int = 100) -> list[dict]:
        reviews: list[dict] = []
        page = 1
        while True:
            batch = self.gh.list_pr_reviews(pr_number, per_page=per_page, page=page)
            if not batch:
                break
            reviews.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return reviews

    def _list_all_pr_review_comments(self, pr_number: int, *, per_page: int = 100) -> list[dict]:
        comments: list[dict] = []
        page = 1
        while True:
            batch = self.gh.list_pr_review_comments(pr_number, per_page=per_page, page=page)
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return comments

    def _should_auto_execute(self, triage: AutoCodeTriageResult, *, owner_authored_issue: bool = False) -> bool:
        mode = self.config.autocode.mode
        if mode == "manual":
            return False
        # 普通用户的 feature 仍保守处理；repo owner 自己创建的 issue 视为已授权执行。
        auto_types = {"bug_fix", "todo_refactor"}
        if owner_authored_issue:
            auto_types.update({"small_feature", "high_risk_feature"})
        if mode == "semi_auto":
            return triage.task_type in auto_types and self.config.autocode.auto_implement_on_issue_open
        return self.config.autocode.auto_implement_on_issue_open and triage.task_type in auto_types

    def _build_run_id(self, kind: str, number: int) -> str:
        return f"{kind}-{number}-{uuid.uuid4().hex[:10]}"

    def _validation_commands(self) -> list[str]:
        commands: list[str] = []
        for group in (
            self.config.autocode.format_commands,
            self.config.autocode.lint_commands,
            self.config.autocode.test_commands,
        ):
            commands.extend(" ".join(command) for command in group)
        return commands

    def _build_commit_message(self, issue_number: int, task_type: str, title: str) -> str:
        prefix = self._TASK_TYPE_PREFIX_MAP.get(task_type, "chore")
        short_title = "-".join(title.lower().split())[:40].strip("-") or "task"
        return f"{prefix}(issue-{issue_number}): {short_title}"

    _TASK_TYPE_PREFIX_MAP: dict[str, str] = {
        "bug_fix": "fix",
        "small_feature": "feat",
        "high_risk_feature": "feat",
        "todo_refactor": "refactor",
    }

    def _normalize_pr_title_issue_text(self, title: str) -> str:
        normalized = _ISSUE_TITLE_PREFIX_RE.sub("", title.strip()).strip()
        # 如果去掉 [Bug]/[Feature] 前缀后内容不足 5 个字符，视为空泛标题返回空字符串。
        if len(normalized) < 5:
            return ""
        return normalized

    @staticmethod
    def _is_usable_pr_title(text: str) -> bool:
        """Return False if text is empty, placeholder-only, or a user question."""
        if not text or _PLACEHOLDER_TITLE_RE.match(text):
            return False
        if _INFORMAL_TITLE_RE.search(text):
            return False
        compact = re.sub(r"\s+", "", text)
        if _LOW_SIGNAL_TITLE_PREFIX_RE.match(compact):
            return False
        if _LOW_SIGNAL_TITLE_SUFFIX_RE.search(compact):
            return False
        return True

    @staticmethod
    def _truncate_title(text: str, max_len: int = 50) -> str:
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        # Try word boundary for mixed CJK/Latin text
        for sep in ("，", "、", "；", ",", " "):
            head, _, _ = cut.rpartition(sep)
            if head and len(head) >= max_len // 2:
                cut = head
                break
        return cut.rstrip("，。、,. ") + "…"

    def _build_pr_title(self, task_type: str, title: str, issue_number: int, *, plan_goal: str = "") -> str:
        prefix = self._TASK_TYPE_PREFIX_MAP.get(task_type, "chore")
        short_title = self._normalize_pr_title_issue_text(title)
        goal_first_sentence = plan_goal.split("。")[0].split("\n")[0].strip() if plan_goal else ""
        if not self._is_usable_pr_title(short_title) or (
            len(short_title) > _VERBOSE_TITLE_MAX_LEN and goal_first_sentence
        ):
            short_title = goal_first_sentence
        if not short_title:
            short_title = f"resolve issue #{issue_number}"
        short_title = self._truncate_title(short_title)
        return f"{prefix}: {short_title} (#{issue_number})"

    def _should_keep_existing_pr_title(self, existing_title: str, task_type: str, issue_number: int) -> bool:
        title = str(existing_title or "").strip()
        if not title:
            return False
        expected_prefix = f"{self._TASK_TYPE_PREFIX_MAP.get(task_type, 'chore')}: "
        if not title.lower().startswith(expected_prefix):
            return False
        match = _PR_TITLE_ISSUE_RE.search(title)
        if not match or int(match.group(1)) != issue_number:
            return False
        short_title = title[len(expected_prefix) : match.start()].strip()
        # 去掉 [Bug]/[Feature] 前缀后不足 5 个有效字符的标题视为空泛。
        cleaned = _ISSUE_TITLE_PREFIX_RE.sub("", short_title).strip()
        if len(cleaned) < 5:
            return False
        return self._is_usable_pr_title(short_title)

    def _build_pr_body(
        self,
        *,
        issue_number: int,
        task_type: str,
        issue_labels: list[str],
        explicit_command: bool,
        plan: ExecutionPlan,
        validation: ValidationResult,
        inspection: PatchInspectionResult,
        execution_summary: str,
        existing_pr_body: str = "",
    ) -> str:
        pr_type = self._resolve_pr_type(task_type, issue_labels, inspection.changed_files)
        verification_commands = [" ".join(step.command) for step in validation.steps if step.command]
        trigger_source = "显式 implement/API 执行" if explicit_command else "Issue 自动执行"
        preserved_verification_section = (
            self._extract_pr_body_section(existing_pr_body, "Verification Commands And Results")
            if existing_pr_body and not verification_commands
            else ""
        )
        # 当 inspection 无数据（如 body-only 刷新时 API 失败）时，保留已有的 Background/Scope 段落。
        preserved_scope_section = (
            self._extract_pr_body_section(existing_pr_body, "Scope Of Change")
            if existing_pr_body and not inspection.changed_files
            else ""
        )
        preserved_background_section = (
            self._extract_pr_body_section(existing_pr_body, "Background And Problem")
            if existing_pr_body and not inspection.changed_files
            else ""
        )
        lines = [
            "## PR Type",
            f"- [{'x' if pr_type == 'fix' else ' '}] fix",
            f"- [{'x' if pr_type == 'feat' else ' '}] feat",
            f"- [{'x' if pr_type == 'refactor' else ' '}] refactor",
            f"- [{'x' if pr_type == 'docs' else ' '}] docs",
            f"- [{'x' if pr_type == 'chore' else ' '}] chore",
            f"- [{'x' if pr_type == 'test' else ' '}] test",
            "",
            "## Background And Problem",
        ]
        if preserved_background_section:
            lines.extend(preserved_background_section.splitlines())
        else:
            lines.extend(
                [
                    f"- 当前问题：{plan.goal}",
                    f"- 影响范围：本次改动涉及 {len(inspection.changed_files)} 个文件，Diff 为 `+{inspection.added_lines} / -{inspection.deleted_lines}`。",
                    f"- 触发来源：{trigger_source}（Issue #{issue_number}）。",
                ]
            )
        lines.extend(["", "## Scope Of Change"])
        if inspection.changed_files:
            lines.extend(f"- `{path}`" for path in inspection.changed_files[:12])
            if len(inspection.changed_files) > 12:
                lines.append(f"- ... and {len(inspection.changed_files) - 12} more files")
        elif preserved_scope_section:
            lines.extend(preserved_scope_section.splitlines())
        else:
            lines.append("- 无")
        lines.extend(
            [
                "",
                "## Documentation And Changelog",
                self._build_pr_documentation_note(inspection.changed_files),
                "",
                "## Issue Link",
                f"Closes #{issue_number}",
                "",
                "## Verification Commands And Results",
            ]
        )
        if verification_commands:
            lines.extend(
                [
                    "```bash",
                    *verification_commands,
                    "```",
                    "",
                    "关键输出/结论 / Key output & conclusion:",
                    f"- {validation.summarize()}",
                ]
            )
        elif preserved_verification_section:
            lines.extend(preserved_verification_section.splitlines())
        else:
            lines.extend(
                [
                    "```bash",
                    "# no validation commands recorded",
                    "```",
                    "",
                    "关键输出/结论 / Key output & conclusion:",
                    f"- {validation.summarize()}",
                ]
            )
        lines.extend(
            [
                "",
                "## Compatibility And Risk",
                self._build_pr_risk_note(plan.risk_level, inspection.changed_files, plan.assumptions),
                "",
                "## Rollback Plan",
                self._build_pr_rollback_plan(inspection.changed_files),
                "",
                "## Acceptance Criteria",
            ]
        )
        if plan.acceptance_criteria:
            lines.extend(f"- {item}" for item in plan.acceptance_criteria)
        else:
            lines.append("- 修复目标问题并通过本地验证")
        if plan.risk_level == "high" or plan.needs_human_approval:
            lines.extend(["", "## Notes"])
            if plan.risk_level == "high":
                lines.append("- Risk is marked as high. Please review scope, validation coverage, and release impact carefully.")
            if plan.needs_human_approval:
                lines.append("- The structured plan originally requested human approval; this PR keeps review-only publication semantics.")
        lines.extend(
            [
                "",
                "## Checklist",
                "- [x] 本 PR 有明确动机和业务价值 / This PR has a clear motivation and value",
                "- [x] 已提供可复现的验证命令与结果 / Reproducible verification commands and results are included",
                "- [x] 已评估兼容性与风险 / Compatibility and risk have been assessed",
                "- [x] 已提供回滚方案 / A rollback plan is provided",
                self._build_pr_documentation_checklist_item(inspection.changed_files),
            ]
        )
        return "\n".join(lines).strip()

    def _sanitize_execution_summary(self, text: str) -> str:
        """Strip local worktree paths from LLM output before embedding in PR body."""
        if not text:
            return text
        if not hasattr(self, "_worktree_path_re"):
            root = re.escape(self.config.autocode.worktree_root.rstrip("/"))
            self._worktree_path_re = re.compile(root + r"/[^/\s)\]\"']+/")
        return self._worktree_path_re.sub("", text)

    def _resolve_pr_type(self, task_type: str, issue_labels: list[str], changed_files: list[str]) -> str:
        normalized_labels = {label.strip().lower() for label in issue_labels if label.strip()}
        if "documentation" in normalized_labels or "docs" in normalized_labels:
            return "docs"
        if "test" in normalized_labels or (
            changed_files and all(path.startswith("tests/") for path in changed_files)
        ):
            return "test"
        if task_type == "bug_fix":
            return "fix"
        if task_type in {"small_feature", "high_risk_feature"}:
            return "feat"
        if task_type == "todo_refactor":
            return "refactor"
        return "chore"

    def _live_patch_inspection_for_pr(self, pr: dict) -> tuple[PatchInspectionResult, str]:
        pr_number = int(pr.get("number", 0) or 0)
        refreshed_pr = dict(pr or {})
        if pr_number > 0:
            try:
                fetched_pr = self.gh.get_pr(pr_number)
            except Exception as exc:
                logger.warning("pr-feedback: failed to refresh PR #%s metadata before body rebuild: %s", pr_number, exc)
            else:
                if isinstance(fetched_pr, dict) and fetched_pr:
                    refreshed_pr = fetched_pr
        body = str(refreshed_pr.get("body", "") or pr.get("body", "") or "")
        changed_files: list[str] = []
        if pr_number > 0:
            try:
                changed_files = [str(path) for path in self.gh.get_pr_files(pr_number) if str(path).strip()]
            except Exception as exc:
                logger.warning("pr-feedback: failed to fetch PR #%s file list before body rebuild: %s", pr_number, exc)
        if not changed_files:
            changed_files = self._extract_pr_changed_files_from_body(body)
        inspection = PatchInspectionResult(
            blocked=False,
            reasons=[],
            changed_files=changed_files,
            added_lines=int(refreshed_pr.get("additions", 0) or pr.get("additions", 0) or 0),
            deleted_lines=int(refreshed_pr.get("deletions", 0) or pr.get("deletions", 0) or 0),
            has_test_changes=any(str(path).startswith("tests/") for path in changed_files),
        )
        return inspection, body

    def _build_pr_risk_note(self, risk_level: str, changed_files: list[str], assumptions: list[str] | None = None) -> str:
        normalized = risk_level.strip().lower()
        focus = ", ".join(f"`{f}`" for f in changed_files[:6]) or "无"
        lines: list[str] = []
        if normalized == "high":
            lines.append(f"- **High**：涉及 {focus}，请重点审查变更范围、验证覆盖与发布影响。")
        elif normalized == "medium":
            lines.append(f"- **Medium**：涉及 {focus}，建议按文件范围复核。")
        else:
            lines.append(f"- **Low**：涉及 {focus}，未识别额外兼容性风险。")
        if assumptions:
            lines.append("- 前提假设：")
            lines.extend(f"  - {a}" for a in assumptions[:4])
        return "\n".join(lines)

    @staticmethod
    def _build_pr_rollback_plan(changed_files: list[str]) -> str:
        if not changed_files:
            return "- `git revert <merge-commit>` 回滚本 PR 提交。"
        focus = ", ".join(f"`{f}`" for f in changed_files[:4])
        return f"- `git revert <merge-commit>` 回滚本 PR 提交，重点确认 {focus} 恢复正常。"

    @staticmethod
    def _normalize_changed_path(path: str) -> str:
        return str(path or "").strip().replace("\\", "/").lower().lstrip("/")

    def _documentation_changed_files(self, changed_files: list[str]) -> list[str]:
        docs: list[str] = []
        for path in changed_files:
            normalized = self._normalize_changed_path(path)
            if not normalized:
                continue
            if normalized == "readme.md" or normalized.endswith("/readme.md"):
                docs.append(path)
                continue
            if normalized == "docs/changelog.md" or normalized.endswith("/changelog.md"):
                docs.append(path)
                continue
            if normalized.startswith("docs/"):
                docs.append(path)
        return docs

    def _has_changelog_update(self, changed_files: list[str]) -> bool:
        return any(
            normalized == "docs/changelog.md" or normalized.endswith("/changelog.md")
            for normalized in (self._normalize_changed_path(path) for path in changed_files)
        )

    def _has_user_doc_update(self, changed_files: list[str]) -> bool:
        return any(
            normalized == "readme.md"
            or normalized.endswith("/readme.md")
            or (normalized.startswith("docs/") and normalized != "docs/changelog.md")
            for normalized in (self._normalize_changed_path(path) for path in changed_files)
        )

    def _build_pr_documentation_note(self, changed_files: list[str]) -> str:
        doc_files = self._documentation_changed_files(changed_files)
        if not doc_files:
            return (
                "- 当前补丁未包含 `README.md`、`docs/` 或 `docs/CHANGELOG.md` 更新；"
                "如果本次变更涉及 CLI、API、配置、日志语义或其他用户可见行为，请在合并前补充原因与文档落点。"
            )
        focus = ", ".join(f"`{path}`" for path in doc_files[:6])
        lines = [f"- 已同步更新文档/变更记录：{focus}。"]
        if len(doc_files) > 6:
            lines.append(f"- 其余文档文件：还有 {len(doc_files) - 6} 个未在此处展开。")
        if not self._has_changelog_update(changed_files):
            lines.append("- 当前补丁尚未包含 `docs/CHANGELOG.md`；如对外行为有变化，请在合并前补齐。")
        if not self._has_user_doc_update(changed_files):
            lines.append("- 当前补丁尚未包含 `README.md` 或专题文档；如用户可见行为发生变化，请补充文档落点。")
        return "\n".join(lines)

    def _build_pr_documentation_checklist_item(self, changed_files: list[str]) -> str:
        if self._has_changelog_update(changed_files) and self._has_user_doc_update(changed_files):
            return (
                "- [x] 已同步更新相关文档与 `docs/CHANGELOG.md`，并在 PR 描述中说明文档落点 / "
                "Relevant docs and `docs/CHANGELOG.md` are updated, and the documentation location is stated in this PR"
            )
        return (
            "- [ ] 文档与 `docs/CHANGELOG.md` 同步仍需确认；如涉及用户可见变更，请在合并前补充原因与文档落点 / "
            "Documentation and `docs/CHANGELOG.md` sync still needs confirmation before merge when user-visible behavior changes"
        )

    @staticmethod
    def _extract_pr_body_section(pr_body: str, heading: str) -> str:
        capture = False
        lines: list[str] = []
        heading_line = f"## {heading}".strip()
        for raw_line in str(pr_body or "").splitlines():
            line = raw_line.rstrip("\n")
            if line.strip().startswith("## "):
                if capture:
                    break
                capture = line.strip() == heading_line
                continue
            if capture:
                lines.append(line)
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _extract_pr_changed_files_from_body(pr_body: str) -> list[str]:
        files: list[str] = []
        in_scope = False
        for raw_line in str(pr_body or "").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                if in_scope:
                    break
                in_scope = line.lower() == "## scope of change"
                continue
            if not in_scope or not line.startswith("- "):
                continue
            item = line[2:].strip().strip("`").strip()
            if not item or item == "无" or item.startswith("... and "):
                continue
            files.append(item)
        return files

    def _safe_get_pr_files(self, pr_number: int) -> list[str]:
        try:
            return self.gh.get_pr_files(pr_number)
        except Exception as exc:
            logger.warning("pr-feedback: failed to fetch PR #%s file list for scope guard: %s", pr_number, exc)
            return []

    def _is_documentation_only_pr(
        self,
        pr: dict,
        *,
        feedback: list[dict],
        changed_files: list[str],
    ) -> bool:
        title = str(pr.get("title", "") or "")
        body = str(pr.get("body", "") or "")
        normalized_title = title.lower()
        normalized_body = body.lower()
        if any(marker in normalized_body for marker in _DOC_ONLY_PR_BODY_MARKERS):
            return True
        if changed_files and all(self._is_documentation_scope_path(path) for path in changed_files):
            return True
        feedback_paths = [str(item.get("path", "") or "") for item in feedback if str(item.get("path", "") or "").strip()]
        if feedback_paths and all(self._is_documentation_scope_path(path) for path in feedback_paths):
            return True
        if any(marker in normalized_title for marker in _DOC_ONLY_TITLE_MARKERS):
            return not any(_CODE_SCOPE_PATH_RE.search(path or "") for path in changed_files)
        return False

    def _documentation_feedback_allowed_paths(
        self,
        *,
        feedback: list[dict],
        changed_files: list[str],
    ) -> list[str]:
        allowed: list[str] = []
        for path in changed_files:
            if path and self._is_documentation_scope_path(path):
                allowed.append(path)
        for item in feedback:
            path = str(item.get("path", "") or "").strip()
            if path and self._is_documentation_scope_path(path):
                allowed.append(path)
        return sorted(dict.fromkeys(allowed))

    def _is_documentation_scope_path(self, rel_path: str) -> bool:
        return is_documentation_scope_path(
            rel_path,
            documentation_paths=getattr(self.config.autocode, "documentation_scope_paths", None),
            documentation_extensions=getattr(self.config.autocode, "documentation_scope_extensions", None),
        )

    @staticmethod
    def _feedback_requests_scope_cleanup(feedback: list[dict], contaminating_files: list[str]) -> bool:
        if not contaminating_files:
            return False
        text = "\n".join(str(item.get("body", "") or "") for item in feedback).lower()
        if not any(marker in text for marker in ("remove", "revert", "drop", "拆分", "移除", "回滚", "去掉", "代码改动")):
            return False
        return any(path.lower() in text for path in contaminating_files)

    @staticmethod
    def _review_feedback_scope_guard(*, documentation_only: bool, allowed_paths: list[str]) -> str:
        if not documentation_only:
            return ""
        allowed_text = "、".join(f"`{path}`" for path in allowed_paths[:12]) if allowed_paths else "`docs/**`、`README.md`、`AGENTS.md` 等文档/治理文件"
        return (
            "本 PR/反馈已判定为文档或治理范围。"
            f"只允许修改文档/治理文件；优先限制在：{allowed_text}。"
            "禁止修改运行时代码、测试代码、构建脚本或依赖配置。"
            "即使本地测试失败，也不得为了让测试通过改代码；应先回退越界改动，最多只修正文档或说明验证限制。"
        )

    @staticmethod
    def _run_documentation_only_validation(workspace_path: Path) -> ValidationResult:
        command = ["git", "diff", "--check"]
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = f"命令超时 (120s): {' '.join(command)}"
            if exc.stdout:
                output += f"\n{exc.stdout}"
            if exc.stderr:
                output += f"\n{exc.stderr}"
            return ValidationResult(
                success=False,
                steps=[
                    ValidationStepResult(
                        name="diff-check",
                        command=command,
                        success=False,
                        output=output.strip(),
                        is_timeout=True,
                    )
                ],
            )
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return ValidationResult(
            success=completed.returncode == 0,
            steps=[
                ValidationStepResult(
                    name="diff-check",
                    command=command,
                    success=completed.returncode == 0,
                    output=output.strip(),
                )
            ],
        )

    def _build_issue_success_comment(
        self,
        *,
        issue_number: int,
        branch_name: str,
        pr_number: int,
        pr_url: str,
        validation: ValidationResult,
        inspection: PatchInspectionResult,
        reused_existing_pr: bool,
        ready_for_review: bool,
        dry_run: bool,
    ) -> str:
        lines = [
            "## AutoCode dry-run 完成"
            if dry_run
            else ("## AutoCode 已更新 PR" if reused_existing_pr else "## AutoCode 已提交 PR"),
            "",
            f"- Issue: #{issue_number}",
            f"- 分支: `{branch_name}`",
            f"- 验证结果: {validation.summarize()}",
            f"- 改动文件数: {len(inspection.changed_files)}",
            "- 安全约束: 不会 merge PR，也不会直接 push 到 `main` 或仓库默认分支",
        ]
        if pr_number and pr_url:
            lines.append(f"- PR: #{pr_number} {pr_url}")
        else:
            lines.append("- PR: dry-run 模式下未实际创建")
        if ready_for_review:
            lines.append("- 发布状态: 已自动转为 ready for review")
        return "\n".join(lines)

    def _find_existing_open_pr(self, head_branch: str, base_branch: str) -> dict | None:
        head = f"{self.config.github.repo.split('/', 1)[0]}:{head_branch}"
        existing = self.gh.list_pull_requests(
            state="open",
            sort="updated",
            direction="desc",
            head=head,
            base=base_branch,
            per_page=1,
            page=1,
        )
        return existing[0] if existing else None

    def _check_open_pr_cap(self) -> str:
        """检查 open autocode PR 是否已达上限。返回空字符串表示允许，否则返回阻断原因。

        只统计 autocode/ 分支前缀的 PR（即本系统自动创建的），不包含
        tracked_pr_authors 白名单中人工创建的 PR。
        """
        cap = max(int(getattr(self.config.autocode, "max_open_autocode_prs", 20) or 0), 0)
        if cap <= 0:
            return ""
        try:
            open_prs = self.gh.list_pull_requests(
                state="open",
                sort="updated",
                direction="desc",
                per_page=100,
                page=1,
            )
            autocode_prs = [
                pr for pr in open_prs
                if str(pr.get("head", {}).get("ref", "") or "").startswith("autocode/")
            ]
            if len(autocode_prs) >= cap:
                return (
                    f"当前 open autocode PR 已达到上限 ({len(autocode_prs)}/{cap})，"
                    "需要先合并或关闭已有 PR 再创建新的"
                )
        except Exception as exc:
            logger.warning("pr-cap: 检查 open PR 数量失败: %s", exc)
        return ""

    def _resolved_pr_base_branch(self) -> str:
        configured = str(self.config.autocode.pr_base_branch or "").strip()
        if configured:
            return configured
        return str(self.config.local_repo.default_branch).strip()

    def _protected_publish_branch(self) -> str:
        return str(self.config.local_repo.default_branch or "main").strip() or "main"

    def _should_auto_ready_for_review(self, *, pr: dict | None, reused_existing_pr: bool) -> bool:
        if not bool(getattr(self.config.autocode, "auto_ready_for_review", False)):
            return False
        if pr is None:
            return False
        if bool(pr.get("draft", False)):
            return True
        return bool(getattr(self.config.autocode, "force_draft_pr", False)) and not reused_existing_pr

    def _is_ready_for_review(self, pr: dict | None) -> bool:
        if pr is None:
            return False
        if "draft" in pr:
            return not bool(pr.get("draft", False))
        return False

    def _issue_run_status(self, *, reused_existing_pr: bool, ready_for_review: bool, dry_run: bool) -> str:
        if dry_run:
            return "dry_run"
        if ready_for_review:
            return "pr_ready"
        return "pr_updated" if reused_existing_pr else "pr_opened"

    def _is_dry_run(self) -> bool:
        return bool(
            getattr(getattr(self.config, "server", None), "dry_run", False)
            or getattr(self.gh_writer, "dry_run", False)
        )

    def _sync_repo_before_task(self, task_name: str):
        try:
            self.workspace_manager.sync_default_branch()
        except RuntimeError as exc:
            raise RuntimeError(f"{task_name} 前同步最新代码失败: {exc}") from exc

    def _format_issue_repair_failure(self, attempt_no: int, stage: str, reason: str) -> str:
        compact_reason = " ".join(str(reason or "").split())
        if len(compact_reason) > 800:
            compact_reason = compact_reason[:797] + "..."
        return f"issue 执行第 {attempt_no} 轮在 {stage} 阶段失败：{compact_reason}"

    def _format_review_feedback_repair_failure(self, attempt_no: int, stage: str, reason: str) -> str:
        compact_reason = " ".join(str(reason or "").split())
        if len(compact_reason) > 800:
            compact_reason = compact_reason[:797] + "..."
        return f"review feedback 第 {attempt_no} 轮在 {stage} 阶段失败：{compact_reason}"

    def _failure_next_step(self, stage: str, reason: str) -> str:
        text = f"{stage}\n{reason}".lower()
        if "command not found" in text or "no module named" in text or "环境未就绪" in reason or "缺少" in reason:
            return "补齐验证依赖或说明该失败可忽略后重试"
        if "验证失败" in reason or "pytest" in text or "lint" in text:
            return "根据失败输出修复测试/lint，或补充说明后让 AutoCode 继续修"
        if "未产生改动" in reason or "未检测到代码改动" in reason:
            return "补充明确目标文件、期望 diff 或复现步骤后重试"
        if "越界" in reason or "blocked" in text or "禁止" in reason:
            return "确认是否允许扩大改动范围或拆小任务"
        return "补充上下文或确认执行范围后重试"

    def _validation_failure_reason(self, validation: ValidationResult) -> str:
        summary = validation.failed_steps_summary(limit=4, output_limit=600)
        return f"验证失败:\n{summary}" if summary else "验证失败"

    def _validation_environment_blocked_reason(self, validation: ValidationResult) -> str:
        blocker = validation.environment_blocker()
        if not blocker:
            return ""
        return f"本地验证环境未就绪：{blocker}"

    def _feedback_marker(self, item: dict) -> str:
        return str(item.get("updated_at") or item.get("submitted_at") or item.get("created_at") or "")

    def _feedback_sort_key(self, item: dict) -> tuple[str, int]:
        return self._feedback_marker(item), int(item.get("id", 0) or 0)

    def _feedback_task_id(self, feedback_key: str) -> str:
        return f"pr-task:{feedback_key}"

    def _existing_feedback_task_summary(
        self,
        pr_number: int,
        *,
        feedback_key: str,
        legacy_feedback_key: str,
        review_state: str,
        ignore_task_ids: set[str] | None = None,
    ) -> tuple[str, str] | None:
        ignored = {item for item in (ignore_task_ids or set()) if str(item or "").strip()}
        task_records = []
        for candidate_key in (feedback_key, legacy_feedback_key):
            if not candidate_key:
                continue
            task = self.state_store.get_task(self._feedback_task_id(candidate_key))
            if task is not None and str(task.get("task_id", "") or "") not in ignored:
                task_records.append(task)
        if not task_records:
            return None
        task = sorted(
            task_records,
            key=lambda item: (
                str(item.get("updated_at", "") or ""),
                str(item.get("task_id", "") or ""),
            ),
            reverse=True,
        )[0]
        status = str(task.get("status", "") or "")
        task_id = str(task.get("task_id", "") or "")
        if status in {"queued", "running"}:
            verb = "正在执行" if status == "running" else "已在队列中等待执行"
            return (
                f"PR #{pr_number} 的相同检视意见{verb} (task={task_id})",
                "review_feedback_pending",
            )
        if status == "retry_waiting":
            return (
                f"PR #{pr_number} 的相同检视意见上一轮修复失败，等待自动重试 "
                f"(task={task_id}, last_error={self._truncate_log_text(str(task.get('last_error', '') or ''), limit=180)})",
                "review_feedback_pending",
            )
        if status == "dead_letter":
            if review_state == "approved":
                return (
                    f"PR #{pr_number} 已通过最新 review（之前修复尝试已耗尽），继续保留监控 (task={task_id})",
                    "approved",
                )
            return (
                f"PR #{pr_number} 的相同检视意见已达到最大自动修复重试次数，等待人工处理 "
                f"(task={task_id}, last_error={self._truncate_log_text(str(task.get('last_error', '') or ''), limit=180)})",
                "failed",
            )
        if status == "done":
            return (
                f"PR #{pr_number} 的相同检视意见已处理完成，等待 reviewer 复核 (task={task_id})",
                "approved" if review_state == "approved" else "waiting_review",
            )
        return None

    def _latest_reviews_log_summary(self, latest_reviews_by_user: dict[str, dict]) -> str:
        parts: list[str] = []
        for user_login, review in sorted(latest_reviews_by_user.items()):
            state = str(review.get("state", "") or "").upper() or "UNKNOWN"
            review_id = int(review.get("id", 0) or 0)
            marker = self._feedback_marker(review) or "unknown"
            body = self._truncate_log_text(str(review.get("body", "") or "").strip(), limit=80)
            segment = f"{user_login}:{state}"
            if review_id > 0:
                segment += f"#{review_id}"
            segment += f"@{marker}"
            if body:
                segment += f":{body}"
            parts.append(segment)
        return "; ".join(parts) if parts else "none"

    def _feedback_log_summary(self, feedback: list[dict], *, limit: int = 4) -> str:
        parts: list[str] = []
        for item in feedback[:limit]:
            item_id = int(item.get("id", 0) or 0)
            user_login = str(item.get("user", {}).get("login", "") or "").strip().lower() or "unknown"
            path = str(item.get("path", "") or "").strip()
            kind = "comment" if path else "review"
            marker = self._feedback_marker(item) or "unknown"
            location = ""
            line = int(item.get("line", 0) or item.get("original_line", 0) or 0)
            if path:
                location = f" {path}"
                if line > 0:
                    location += f":{line}"
            body = self._truncate_log_text(str(item.get("body", "") or "").strip(), limit=100)
            parts.append(f"{kind}#{item_id} {user_login}@{marker}{location}: {body}".strip())
        remaining = len(feedback) - limit
        if remaining > 0:
            parts.append(f"+{remaining} more")
        return "; ".join(parts) if parts else "none"

    def _should_include_review_body(self, review: dict, *, pr_author_login: str) -> bool:
        reviewer_login = str(review.get("user", {}).get("login", "") or "").strip().lower()
        review_state = str(review.get("state", "") or "").upper()
        body = str(review.get("body", "") or "").strip()
        if self._looks_like_non_actionable_review_summary_body(body):
            return False
        if reviewer_login and pr_author_login and reviewer_login == pr_author_login and review_state == "COMMENTED":
            return self._looks_like_actionable_self_review_body(body)
        if not body:
            return False
        return not self._looks_like_automated_review_boilerplate(body)

    def _looks_like_actionable_self_review_body(self, body: str) -> bool:
        normalized = " ".join(str(body or "").strip().lower().split())
        if not normalized:
            return False
        if normalized in _SELF_REVIEW_NON_ACTIONABLE_BODIES:
            return False
        if self._looks_like_automated_review_boilerplate(body):
            return False
        if any(marker in normalized for marker in _ACTIONABLE_SELF_REVIEW_MARKERS):
            return True
        return _ACTIONABLE_SELF_REVIEW_REFERENCE_RE.search(str(body or "")) is not None

    def _looks_like_non_actionable_review_summary_body(self, body: str) -> bool:
        normalized = " ".join(str(body or "").strip().lower().split())
        if not normalized:
            return False
        if "评审结论" not in normalized:
            return False
        if any(marker in normalized for marker in _ACTIONABLE_REVIEW_BODY_MARKERS):
            return False
        return any(marker in normalized for marker in _NON_ACTIONABLE_REVIEW_SUMMARY_MARKERS)

    def _looks_like_automated_review_boilerplate(self, body: str) -> bool:
        normalized = " ".join(str(body or "").strip().lower().split())
        if not normalized:
            return False
        return any(marker in normalized for marker in _AUTOMATED_REVIEW_BOILERPLATE_MARKERS)

    def _is_pr_body_only_feedback(self, feedback: list[dict]) -> bool:
        if not feedback:
            return False
        for item in feedback:
            if str(item.get("path", "") or "").strip():
                return False
            body = str(item.get("body", "") or "").strip().lower()
            if not body:
                return False
            if not any(marker in body for marker in _SIMPLE_PR_FEEDBACK_MARKERS):
                return False
        return True

    def _is_bot_only_feedback(self, feedback: list[dict]) -> bool:
        """所有反馈项都来自已知 bot reviewer 时返回 True。"""
        if not feedback:
            return False
        for item in feedback:
            if not self._is_bot_review_item(item):
                return False
        return True

    def _is_bot_review_item(self, item: dict) -> bool:
        """单条反馈来自已知 bot（按 login 或 body 标记判断）。"""
        login = str(item.get("user", {}).get("login", "") or "").strip().lower()
        # Match with and without [bot] suffix for consistent detection
        bare_login = login.removesuffix("[bot]")
        if bare_login and (login in _BOT_REVIEWER_LOGINS or bare_login in _BOT_REVIEWER_LOGINS):
            return True
        body = str(item.get("body", "") or "")
        return any(marker in body for marker in _BOT_REVIEW_BODY_MARKERS)

    def _find_feedback_non_pr_redirect(self, feedback: list[dict]) -> str:
        for item in reversed(feedback):
            if self._is_bot_review_item(item):
                continue
            login = str(item.get("user", {}).get("login", "") or "").strip().lower()
            if not self._is_repo_owner_login(login):
                continue
            raw_body = str(item.get("body", "") or "")
            if self._looks_like_openreview_review_body(raw_body):
                continue
            body = " ".join(raw_body.split())
            if not body:
                continue
            if any(pattern.search(body) for pattern in _NON_PR_REDIRECT_PATTERNS):
                return self._truncate_log_text(body, limit=180)
            return ""
        return ""

    def _focus_feedback_on_latest_openreview(self, feedback: list[dict]) -> list[dict]:
        """Prefer the newest actionable OpenReview verdict over older bot-only notes."""
        actionable_openreview = [
            item for item in feedback
            if self._is_actionable_openreview_feedback_item(item)
        ]
        if not actionable_openreview:
            return feedback
        latest_openreview_key = max(self._feedback_sort_key(item) for item in actionable_openreview)
        openreview_item_ids = {id(item) for item in actionable_openreview}
        focused: list[dict] = []
        dropped = 0
        for item in feedback:
            if id(item) in openreview_item_ids:
                focused.append(item)
                continue
            if not self._is_bot_review_item(item):
                focused.append(item)
                continue
            if self._feedback_sort_key(item) >= latest_openreview_key:
                focused.append(item)
                continue
            dropped += 1
        if dropped:
            logger.info(
                "pr-feedback: 最新 OpenReview 评审已覆盖旧 bot 反馈，跳过旧 bot feedback items=%s",
                dropped,
            )
        return focused

    def _looks_like_openreview_review_body(self, body: str) -> bool:
        raw = str(body or "")
        normalized = " ".join(raw.strip().lower().split())
        if not normalized:
            return False
        if _OPENREVIEW_BODY_MARKER in normalized or "<!-- openreview:" in normalized:
            return True
        has_review_structure = "评审结论" in normalized and (
            "是否可直接合入" in normalized or "主要问题" in normalized
        )
        if not has_review_structure:
            return False
        has_actionable_verdict = bool(_OPENREVIEW_VERDICT_RE.search(raw)) or any(
            marker in normalized for marker in _ACTIONABLE_REVIEW_BODY_MARKERS
        )
        return has_actionable_verdict

    def _is_actionable_openreview_feedback_item(self, item: dict) -> bool:
        body = str(item.get("body", "") or "")
        if not self._looks_like_openreview_review_body(body):
            return False
        verdict = _OPENREVIEW_VERDICT_RE.search(body)
        if verdict and verdict.group(1) == "可":
            return False
        normalized = " ".join(body.strip().lower().split())
        if verdict and verdict.group(1) in {"不可", "不能"}:
            return True
        return any(marker in normalized for marker in _ACTIONABLE_REVIEW_BODY_MARKERS)

    def _has_actionable_openreview_feedback(self, feedback: list[dict]) -> bool:
        return any(self._is_actionable_openreview_feedback_item(item) for item in feedback)

    def _feedback_resets_repair_limit(self, feedback: list[dict]) -> bool:
        return bool(feedback) and (
            not self._is_bot_only_feedback(feedback)
            or self._has_actionable_openreview_feedback(feedback)
        )

    def _reset_repair_limit_for_trusted_feedback_if_needed(
        self,
        pr_number: int,
        feedback: list[dict],
        *,
        context: str,
    ) -> bool:
        max_rounds = max(int(getattr(self.config.autocode, "pr_max_repair_rounds", 8) or 0), 0)
        if max_rounds <= 0:
            return False
        current = self.state_store.get_tracked_pr_repair_count(pr_number)
        if current < max_rounds:
            return False
        if not self._feedback_resets_repair_limit(feedback):
            return False
        self.state_store.reset_tracked_pr_repair_count(pr_number)
        logger.info(
            "pr-feedback: PR #%s 收到新的可信 review feedback，修复计数已自动重置 "
            "(context=%s, was=%s/%s)",
            pr_number,
            context,
            current,
            max_rounds,
        )
        return True

    def _advance_feedback_cursor_on_block(self, pr_number: int, feedback: list[dict]) -> None:
        """推进 feedback cursor 使已被阻断的 feedback 不再被下轮 poll 重复评估。"""
        if not feedback:
            return
        cursor_updated_at, cursor_item_ids = self._feedback_cursor(feedback)
        if cursor_updated_at:
            self.state_store.update_tracked_pr_feedback_cursor(
                pr_number,
                updated_at=cursor_updated_at,
                item_ids=cursor_item_ids,
            )

    def _check_repair_limit(self, pr_number: int, review_state: str) -> str:
        """检查 PR 修复轮数是否已达上限，返回空字符串表示允许，否则返回阻断摘要。"""
        max_rounds = max(int(getattr(self.config.autocode, "pr_max_repair_rounds", 8) or 0), 0)
        if max_rounds <= 0:
            return ""
        current = self.state_store.get_tracked_pr_repair_count(pr_number)
        if current < max_rounds:
            return ""
        summary = (
            f"PR #{pr_number} 已达到最大修复轮数 ({current}/{max_rounds})，"
            "需要人工介入或通过 pause/retry 接口重置"
        )
        # 已处于 repair_limit_reached 状态时降级为 debug，避免每轮 poll 重复 warning
        tracked = self.state_store.get_tracked_pr(pr_number) or {}
        if str(tracked.get("status", "")) == "repair_limit_reached":
            logger.debug("pr-feedback: %s", summary)
        else:
            logger.warning("pr-feedback: %s", summary)
        self.state_store.mark_tracked_pr_status(
            pr_number,
            status="repair_limit_reached",
            summary=summary,
            review_state=review_state,
        )
        return summary

    def _check_bot_feedback_throttle(self, pr_number: int, feedback: list[dict], review_state: str) -> str:
        """纯 bot 反馈超过修复上限时返回阻断摘要，否则返回空字符串。"""
        if not self._is_bot_only_feedback(feedback):
            return ""
        if self._has_actionable_openreview_feedback(feedback):
            return ""
        max_rounds = max(int(getattr(self.config.autocode, "bot_feedback_max_repair_rounds", 4) or 0), 0)
        if max_rounds <= 0:
            return ""
        current = self.state_store.get_tracked_pr_repair_count(pr_number)
        if current < max_rounds:
            return ""
        summary = (
            f"PR #{pr_number} 当前反馈均来自 bot reviewer，且已达到 bot 修复上限 "
            f"({current}/{max_rounds})，跳过自动修复"
        )
        # 已经在 waiting_review 且上次也是 bot throttle 时降级为 debug
        tracked = self.state_store.get_tracked_pr(pr_number) or {}
        prev_summary = str(tracked.get("last_summary", "") or "")
        if "bot 修复上限" in prev_summary:
            logger.debug("pr-feedback: %s", summary)
        else:
            logger.info("pr-feedback: %s", summary)
        self.state_store.mark_tracked_pr_status(
            pr_number,
            status="waiting_review",
            summary=summary,
            review_state=review_state,
        )
        return summary

    def _is_outdated_review_comment(self, comment: dict) -> bool:
        if not str(comment.get("path", "") or "").strip():
            return False
        line = comment.get("line")
        original_line = int(comment.get("original_line", 0) or 0)
        original_position = int(comment.get("original_position", 0) or 0)
        if line in (None, "", 0) and (original_line > 0 or original_position > 0):
            return True
        return False

    def _truncate_log_text(self, text: str, *, limit: int = 160) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 3, 1)] + "..."

    def _summarize_list(self, items: list[str], *, limit: int = 6) -> str:
        normalized = [str(item).strip() for item in items if str(item).strip()]
        if not normalized:
            return "-"
        if len(normalized) <= limit:
            return ", ".join(normalized)
        return ", ".join(normalized[:limit]) + f", ...(+{len(normalized) - limit})"

    def _is_feedback_item_pending(self, item: dict, *, after_updated_at: str, after_ids: set[int]) -> bool:
        if not after_updated_at:
            return True
        marker = self._feedback_marker(item)
        item_id = int(item.get("id", 0) or 0)
        marker_dt = self._parse_iso_timestamp(marker)
        after_dt = self._parse_iso_timestamp(after_updated_at)
        if marker_dt is not None and after_dt is not None:
            if marker_dt > after_dt:
                return True
            if marker_dt == after_dt and item_id > 0 and item_id not in after_ids:
                return True
            return False
        if marker > after_updated_at:
            return True
        if marker == after_updated_at and item_id > 0 and item_id not in after_ids:
            return True
        return False

    def _feedback_cursor(self, feedback: list[dict]) -> tuple[str, list[int]]:
        if not feedback:
            return "", []
        latest_marker = max(self._feedback_marker(item) for item in feedback)
        latest_ids = sorted(
            int(item.get("id", 0) or 0)
            for item in feedback
            if self._feedback_marker(item) == latest_marker and int(item.get("id", 0) or 0) > 0
        )
        return latest_marker, latest_ids

    def _get_previous_failure_reasons(self, kind: str, target_number: int) -> list[str]:
        """从历史 run 记录中提取最近失败原因，用于重试时注入 prompt 上下文。"""
        failed_runs = self.state_store.list_failed_runs_for_target(kind, target_number, limit=3)
        reasons: list[str] = []
        for run in failed_runs:
            reason = str(run.get("failure_reason", "") or "").strip()
            if self._should_suppress_previous_failure_reason(kind, reason):
                continue
            if reason:
                # 截断过长的错误信息
                if len(reason) > 200:
                    reason = reason[:200] + "…"
                reasons.append(reason)
        return reasons

    @staticmethod
    def _should_suppress_previous_failure_reason(kind: str, reason: str) -> bool:
        """Drop stale issue-scope failures that would mislead the next implement attempt."""
        if str(kind or "").strip() != "issue":
            return False
        return str(reason or "").strip().startswith("越界改动:")

    def _parse_iso_timestamp(self, value: str) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
