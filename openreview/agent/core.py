"""
Agent 核心调度逻辑

职责:
1. 接收事件网关传来的结构化事件
2. 调用分类器判断是否应该处理
3. 组装 prompt 并调用 CLI 分析器
4. 返回处理结果（不直接操作 GitHub API，由上层决定是否发布）

安全原则:
- 本模块只做 **分析和生成回复文本**
- 绝不执行任何代码、提交、合并等操作
"""
import logging
import re
from dataclasses import dataclass

from adapters.base import BaseLLMAdapter, LLMError
from agent.classifier import classify_issue, ClassificationResult, quick_classify
from agent.context_builder import LocalContextBuilder
from agent.git_guard import GitGuard, PRDiffSnapshot
from prompts import (
    build_issue_analysis_prompt,
    build_pr_review_prompt,
    build_pr_followup_prompt,
    build_unanswerable_reply,
    wrap_followup_reply,
    wrap_pr_review_reply,
    wrap_reply,
)

logger = logging.getLogger(__name__)

_ISSUE_REFERENCE_RE = re.compile(
    r"(?i)\b(?:fix(?:es|ed)?|close(?:s|d)?|refs?|reference(?:s|d)?|resolve(?:s|d)?)"
    r"\s*(?:issue\s*)?(?:[:：]\s*)?#\d+\b"
)
_DOC_LIKE_REVIEW_EXTENSIONS = {
    ".md", ".mdx", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
}
_DOC_OR_COMMENT_ONLY_CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx"}
_EXTERNAL_MODEL_RISK_MARKERS = (
    "openai_base_url",
    "base_url",
    "api_base",
    "litellm_model",
    "agent_litellm_model",
    "vision_model",
    "llm_channels",
    "llm_",
    "model:",
    "models=",
    "provider/",
    "deepseek",
    "gemini",
    "anthropic",
    "openai",
    "ollama",
    "dashscope",
    "moonshot",
    "qwen",
    "kimi",
    "minimax",
    "deprecated",
    "废弃",
)
_RUNTIME_CONFIG_MIGRATION_MARKERS = (
    "sanitize",
    "clear",
    "filter",
    "清空",
    "过滤",
    "迁移",
    "默认",
    "default",
    "preset",
    "fallback",
)


@dataclass(frozen=True)
class PRReviewResult:
    """PR 处理结果。"""

    reply: str = ""
    blocked_reason: str = ""


class ReviewAgent:
    """代码审查与 Issue 分析 Agent"""

    def __init__(
        self,
        analyzer: BaseLLMAdapter,
        repo_name: str,
        max_comment_length: int = 4000,
        context_builder: LocalContextBuilder | None = None,
        git_guard: GitGuard | None = None,
    ):
        self.analyzer = analyzer
        self.repo_name = repo_name
        self.max_comment_length = max_comment_length
        self.context_builder = context_builder
        self.git_guard = git_guard

    # ------------------------------------------------------------------
    # Issue 处理
    # ------------------------------------------------------------------

    def handle_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
        *,
        triage_context: str = "",
        discussion_context: str = "",
        allow_unanswerable_reply: bool = True,
        author_association: str = "",
    ) -> tuple[str, ClassificationResult]:
        """
        处理 Issue 事件。

        Returns:
            (reply_text, classification) — 回复文本和分类结果
        """
        if self.git_guard is not None:
            with self.git_guard.safe_default_branch():
                return self._do_issue_analysis(
                    title,
                    body,
                    labels,
                    triage_context=triage_context,
                    discussion_context=discussion_context,
                    allow_unanswerable_reply=allow_unanswerable_reply,
                    author_association=author_association,
                )
        return self._do_issue_analysis(
            title,
            body,
            labels,
            triage_context=triage_context,
            discussion_context=discussion_context,
            allow_unanswerable_reply=allow_unanswerable_reply,
            author_association=author_association,
        )

    # ------------------------------------------------------------------
    # PR Review 处理
    # ------------------------------------------------------------------

    def handle_pr(
        self,
        title: str,
        body: str,
        base_ref: str,
        head_ref: str = "",
        head_repo_clone_url: str = "",
        head_sha: str = "",
        discussion_context: str = "",
        review_facts: list[str] | None = None,
    ) -> PRReviewResult:
        """
        处理 PR 事件，返回 review 结果。

        只有在本地仓库成功同步并检出到 PR 最新 head 后，才会生成 review。
        否则返回 blocked_reason，由上层决定如何通知。
        """
        if self.git_guard is None:
            reason = "本地 Git 仓库未就绪，无法在评审前同步到 PR 最新代码"
            logger.warning(reason)
            return PRReviewResult(blocked_reason=reason)

        with self.git_guard.safe_branch(
            head_ref,
            remote_url=head_repo_clone_url,
            target_sha=head_sha,
        ) as prep:
            if not prep.ready:
                reason = prep.reason or "无法切换到 PR 最新代码"
                logger.warning("PR 评审前同步失败，跳过 review: %s", reason)
                return PRReviewResult(blocked_reason=reason)

            logger.info(
                "已切换到 PR 最新代码进行分析: ref=%s, sha=%s",
                head_ref or "unknown",
                head_sha[:12] or "unknown",
            )
            try:
                return PRReviewResult(
                    reply=self._do_pr_review(
                        title,
                        body,
                        base_ref,
                        discussion_context=discussion_context,
                        review_facts=review_facts,
                    )
                )
            except RuntimeError as e:
                reason = str(e).strip() or "无法构建 PR 本地 diff"
                logger.warning("PR 评审前本地 diff 构建失败，跳过 review: %s", reason)
                return PRReviewResult(blocked_reason=reason)

    def handle_pr_followup(
        self,
        title: str,
        body: str,
        base_ref: str,
        discussion_context: str,
        head_ref: str = "",
        head_repo_clone_url: str = "",
        head_sha: str = "",
        review_facts: list[str] | None = None,
    ) -> PRReviewResult:
        """
        处理 PR 后续讨论，返回继续评审结果。

        与首轮 review 一样，只有在本地切换到最新 head 后才会生成跟进评论。
        """
        if self.git_guard is None:
            reason = "本地 Git 仓库未就绪，无法在评审前同步到 PR 最新代码"
            logger.warning(reason)
            return PRReviewResult(blocked_reason=reason)

        with self.git_guard.safe_branch(
            head_ref,
            remote_url=head_repo_clone_url,
            target_sha=head_sha,
        ) as prep:
            if not prep.ready:
                reason = prep.reason or "无法切换到 PR 最新代码"
                logger.warning("PR 跟进评审前同步失败，跳过 review: %s", reason)
                return PRReviewResult(blocked_reason=reason)

            logger.info(
                "已切换到 PR 最新代码进行跟进分析: ref=%s, sha=%s",
                head_ref or "unknown",
                head_sha[:12] or "unknown",
            )
            try:
                return PRReviewResult(
                    reply=self._do_pr_followup_review(
                        title,
                        body,
                        base_ref,
                        discussion_context=discussion_context,
                        review_facts=review_facts,
                    )
                )
            except RuntimeError as e:
                reason = str(e).strip() or "无法构建 PR 本地 diff"
                logger.warning("PR 跟进评审前本地 diff 构建失败，跳过 review: %s", reason)
                return PRReviewResult(blocked_reason=reason)

    def _do_issue_analysis(
        self,
        title: str,
        body: str,
        labels: list[str],
        *,
        triage_context: str = "",
        discussion_context: str = "",
        allow_unanswerable_reply: bool = True,
        author_association: str = "",
    ) -> tuple[str, ClassificationResult]:
        """执行 Issue 分类与回复生成。"""
        local_code_context = ""
        if self.context_builder is not None:
            extended_body = "\n\n".join(
                part for part in (body or "", triage_context, discussion_context) if part.strip()
            )
            local_code_context = self.context_builder.build_issue_context(title=title, body=extended_body)

        # 跟进讨论（有讨论上下文 + 不允许 UNANSWERABLE 回复）直接跳过 LLM 分类
        is_followup = not allow_unanswerable_reply and bool(discussion_context)

        # 先尝试轻量级预过滤（即使跟进也要尊重高置信 UNANSWERABLE 判断）
        quick_result = quick_classify(title, body or "", labels, author_association=author_association)
        if quick_result is not None:
            classification = quick_result
            logger.info(
                "Issue 预过滤分类: %s (reason: %s)",
                classification.classification,
                classification.reason,
            )
        elif is_followup:
            classification = ClassificationResult(
                classification="ANSWERABLE",
                reason="跟进讨论，跳过分类直接分析",
                confidence=0.9,
            )
            logger.info("Issue 跟进讨论，跳过 LLM 分类直接进入分析")
        else:
            classification = classify_issue(
                title,
                body or "",
                labels,
                self.analyzer,
                repo_name=self.repo_name,
                local_code_context=local_code_context,
                triage_context=triage_context,
                discussion_context=discussion_context,
                author_association=author_association,
            )

        if classification.degraded:
            logger.error(
                "Issue 分类失败，跳过自动回复: %s",
                classification.internal_error or classification.reason,
            )
            return "", classification

        if not classification.is_answerable:
            logger.info("Issue 分类为不可回答: %s", classification.reason)
            if not allow_unanswerable_reply:
                return "", classification
            reply = build_unanswerable_reply(classification.reason)
            return reply, classification

        try:
            logger.info("Issue 分类为可回答，开始生成详细回复")
            prompt = build_issue_analysis_prompt(
                title=title,
                body=body or "",
                labels=labels,
                repo_name=self.repo_name,
                local_code_context=local_code_context,
                triage_context=triage_context,
                discussion_context=discussion_context,
                is_followup=is_followup,
                author_association=author_association,
            )
            raw_reply = self.analyzer.analyze(
                prompt,
                system=(
                    "你是 GitHub Issue 回复助手，只输出最终评论正文。"
                    "不要输出分析过程、内部大纲、工具调用、Read/Grep/List/Explore 轨迹，"
                    "也不要输出“是否合理”“是否是 issue”“建议动作”等内部字段。"
                ),
            )
            self._verify_repo_clean()
            reply = wrap_reply(self._truncate(raw_reply))
            logger.info("Issue 分析完成，回复长度: %d", len(reply))
            return reply, classification
        except LLMError as e:
            logger.error("Issue 分析 CLI 调用失败: %s", e)
            return "", classification

    def _do_pr_review(
        self,
        title: str,
        body: str,
        base_ref: str,
        *,
        discussion_context: str = "",
        review_facts: list[str] | None = None,
    ) -> str:
        """执行 PR review 的实际逻辑。"""
        if self.git_guard is None:
            return ""

        snapshot = self.git_guard.build_pr_diff_snapshot(base_ref)
        changed_files = snapshot.changed_files
        if not changed_files:
            logger.warning("PR 本地 diff 为空，跳过 review")
            return ""

        local_code_context = ""
        if self.context_builder is not None:
            local_code_context = self.context_builder.build_pr_context(changed_files=changed_files)

        logger.info("开始生成 PR review")
        local_snapshot_facts = self._build_local_snapshot_review_facts(snapshot)
        discussion_facts = self._build_discussion_review_facts(discussion_context)
        prompt = build_pr_review_prompt(
            title=title,
            body=body or "",
            repo_name=self.repo_name,
            diff_excerpt=snapshot.diff_excerpt,
            diff_stat=snapshot.diff_stat,
            changed_files=changed_files,
            review_facts=self._build_pr_review_facts(
                body=body or "",
                changed_files=changed_files,
                diff_excerpt=snapshot.diff_excerpt,
                extra_facts=local_snapshot_facts + list(review_facts or []) + discussion_facts,
            ),
            local_code_context=local_code_context,
            discussion_context=discussion_context,
        )
        try:
            raw_reply = self.analyzer.analyze(
                prompt,
                system=(
                    "你是 GitHub PR reviewer，只输出最终 review 评论正文。"
                    "不要输出思考过程、工具调用、Read/Grep/List/Explore 轨迹或文件查看过程。"
                ),
            )
            self._verify_repo_clean()
            reply = wrap_pr_review_reply(self._truncate(raw_reply))
            logger.info("PR review 完成，回复长度: %d", len(reply))
            return reply
        except LLMError as e:
            logger.error("PR review CLI 调用失败: %s", e)
            return ""

    def _do_pr_followup_review(
        self,
        title: str,
        body: str,
        base_ref: str,
        *,
        discussion_context: str,
        review_facts: list[str] | None = None,
    ) -> str:
        """执行 PR 后续讨论的继续评审。"""
        if self.git_guard is None:
            return ""

        snapshot = self.git_guard.build_pr_diff_snapshot(base_ref)
        changed_files = snapshot.changed_files
        if not changed_files:
            logger.warning("PR 本地 diff 为空，跳过跟进评审")
            return ""

        local_code_context = ""
        if self.context_builder is not None:
            local_code_context = self.context_builder.build_pr_context(changed_files=changed_files)

        logger.info("开始生成 PR 跟进评审")
        local_snapshot_facts = self._build_local_snapshot_review_facts(snapshot)
        discussion_facts = self._build_discussion_review_facts(discussion_context)
        prompt = build_pr_followup_prompt(
            title=title,
            body=body or "",
            repo_name=self.repo_name,
            discussion_context=discussion_context,
            diff_excerpt=snapshot.diff_excerpt,
            diff_stat=snapshot.diff_stat,
            changed_files=changed_files,
            review_facts=self._build_pr_review_facts(
                body=body or "",
                changed_files=changed_files,
                diff_excerpt=snapshot.diff_excerpt,
                extra_facts=local_snapshot_facts + list(review_facts or []) + discussion_facts,
            ),
            local_code_context=local_code_context,
        )
        try:
            raw_reply = self.analyzer.analyze(
                prompt,
                system=(
                    "你是 GitHub PR reviewer，只输出最终继续评审评论正文。"
                    "不要输出思考过程、工具调用、Read/Grep/List/Explore 轨迹或文件查看过程。"
                ),
            )
            self._verify_repo_clean()
            reply = wrap_followup_reply(self._truncate(raw_reply))
            logger.info("PR 跟进评审完成，回复长度: %d", len(reply))
            return reply
        except LLMError as e:
            logger.error("PR 跟进评审 CLI 调用失败: %s", e)
            return ""

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _build_pr_review_facts(
        self,
        *,
        body: str,
        changed_files: list[str],
        diff_excerpt: str = "",
        extra_facts: list[str] | None = None,
    ) -> list[str]:
        changed_set = set(changed_files)
        issue_refs = self._extract_issue_references(body)
        review_text = self._review_risk_text(body=body, changed_files=changed_files, diff_excerpt=diff_excerpt)
        facts = [
            f"完整改动文件数：{len(changed_files)}",
            f"`README.md` 在改动文件中：{'是' if 'README.md' in changed_set else '否'}",
            f"`docs/CHANGELOG.md` 在改动文件中：{'是' if 'docs/CHANGELOG.md' in changed_set else '否'}",
            f"PR 描述提到 `ci_gate.sh`：{'是' if self._mentions_ci_gate(body) else '否'}",
            f"PR 描述中的 issue 关联：{issue_refs or '未检测到 Fixes/Closes/Refs 语句'}",
            self._review_scope_fact(changed_files),
        ]
        facts.extend(
            self._external_model_review_facts(
                review_text,
                docs_only=self._is_doc_like_review_change(changed_files),
            )
        )
        for fact in extra_facts or []:
            normalized = str(fact or "").strip()
            if normalized and normalized not in facts:
                facts.append(normalized)
        return facts

    def _build_local_snapshot_review_facts(self, snapshot: PRDiffSnapshot) -> list[str]:
        merge_base_short = snapshot.merge_base[:12] if snapshot.merge_base else "unknown"
        return [
            "本地评审方式：已拉取并检出 PR 最新 head 后分析",
            f"本地 diff 比较基线：{snapshot.compare_ref}",
            f"本地 merge-base：{merge_base_short}",
            "完整改动文件列表来源：本地 git diff --name-only",
        ]

    def _build_discussion_review_facts(self, discussion_context: str) -> list[str]:
        text = str(discussion_context or "").lower()
        if not text:
            return []
        process_markers = (
            "rollback",
            "回滚",
            "description",
            "描述",
            "readme",
            "文档",
            "官方来源",
            "source",
            "验证证据",
            "verification",
        )
        openreview_mentions = text.count("openreview") + text.count("自动生成")
        marker_hits = sum(1 for marker in process_markers if marker in text)
        if openreview_mentions >= 1 and marker_hits >= 2:
            return [
                "流程型阻断节流提示：讨论中已有自动评审反复提到描述、回滚、文档、来源或验证证据等流程问题；若当前代码/CI/兼容性无真实风险，不要再次把纯流程问题作为阻断，应降级为建议或交由维护者判断。"
            ]
        return []

    def _extract_issue_references(self, body: str) -> str:
        references: list[str] = []
        seen: set[str] = set()
        for match in _ISSUE_REFERENCE_RE.finditer(body):
            reference = match.group(0).strip()
            lowered = reference.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            references.append(reference)
        return " | ".join(references[:3])

    def _mentions_ci_gate(self, body: str) -> bool:
        return "ci_gate.sh" in body.lower()

    def _review_scope_fact(self, changed_files: list[str]) -> str:
        if not changed_files:
            return "低风险文档/注释类审查提示：无改动文件，需先确认 diff"
        lower_paths = [path.strip().lower() for path in changed_files if str(path or "").strip()]
        doc_like = [path for path in lower_paths if self._is_doc_like_review_path(path)]
        code_like = [
            path for path in lower_paths
            if path.endswith(tuple(_DOC_OR_COMMENT_ONLY_CODE_EXTENSIONS))
        ]
        if len(doc_like) == len(lower_paths):
            return "低风险文档/注释类审查提示：当前改动文件均为文档或图片资源，流程性缺口不能单独阻断"
        if doc_like and code_like and len(doc_like) + len(code_like) == len(lower_paths):
            return "低风险文档/注释类审查提示：若代码文件仅改注释或 docstring，应按文档/注释类处理，回滚方案等流程性缺口不能单独阻断"
        return "低风险文档/注释类审查提示：当前包含运行时代码或无法确认仅为注释，按常规风险审查"

    def _review_risk_text(self, *, body: str, changed_files: list[str], diff_excerpt: str) -> str:
        return "\n".join([str(body or ""), *[str(path or "") for path in changed_files], str(diff_excerpt or "")]).lower()

    def _is_doc_like_review_path(self, path: str) -> bool:
        normalized = str(path or "").strip().lower()
        return (
            normalized.startswith(("docs/", "doc/"))
            or normalized in {"readme.md", "changelog.md"}
            or normalized.endswith(tuple(_DOC_LIKE_REVIEW_EXTENSIONS))
        )

    def _is_doc_like_review_change(self, changed_files: list[str]) -> bool:
        lower_paths = [path.strip().lower() for path in changed_files if str(path or "").strip()]
        return bool(lower_paths) and all(self._is_doc_like_review_path(path) for path in lower_paths)

    def _external_model_review_facts(self, text: str, *, docs_only: bool = False) -> list[str]:
        facts: list[str] = []
        if any(marker in text for marker in _EXTERNAL_MODEL_RISK_MARKERS):
            facts.append(
                (
                    "文档型外部模型/API 提示：当前改动文件均为文档或资源；只有本次 diff 新增或改变模型名、provider、"
                    "Base URL、SDK/依赖默认值、废弃日期等用户可执行兼容性声明时，才要求官方来源或运行时兼容证据。"
                    "若只是搬运/恢复既有仓库文档，且 PR 描述明确无运行时变更、无新增外部兼容性语义并给出仓库内依据，"
                    "不要仅因缺少外部官方来源判为阻断。"
                )
                if docs_only
                else (
                    "外部模型/API 兼容风险提示：检测到模型名、provider、Base URL、废弃日期、LiteLLM 或 LLM 配置相关改动；"
                    "评审必须核验官方来源、当前依赖/运行时兼容性、旧配置迁移和回退路径，不能只因 CI 成功或 PR 描述完整就判低风险。"
                )
            )
        if any(marker in text for marker in ("deprecated", "废弃", "v4", "v3", "model", "模型")) and any(
            provider in text for provider in ("deepseek", "gemini", "openai", "anthropic", "ollama", "dashscope", "moonshot")
        ):
            facts.append(
                (
                    "文档型外部事实核验提示：PR 文档涉及第三方模型/API 名称或生命周期信息；仅当本次 diff 新增或改变外部事实、"
                    "默认推荐或废弃时间时，才要求官方文档/公告来源。复述既有仓库说明时可接受仓库内证据。"
                )
                if docs_only
                else "外部事实核验提示：PR 涉及第三方模型/API 名称或生命周期信息；若描述未附官方文档/公告来源，需列为待澄清或验证缺口。"
            )
        if any(marker in text for marker in _RUNTIME_CONFIG_MIGRATION_MARKERS) and any(
            key in text
            for key in (
                "litellm_model",
                "agent_litellm_model",
                "vision_model",
                "fallback",
                "runtimeconfig",
                "运行时",
                "用户配置",
            )
        ):
            facts.append(
                (
                    "文档型运行时配置提示：PR 文档提到默认模型、fallback、运行时配置或保存行为；只有新增/改变配置语义、"
                    "迁移步骤或用户可执行示例时，才要求迁移/回退说明。单纯整理既有说明不应被当成运行时代码迁移。"
                )
                if docs_only
                else "运行时配置迁移提示：检测到默认模型、运行时模型或保存前清理逻辑变更；需检查是否会静默清空/迁移用户配置、是否有用户可见提示，以及对应回归测试。"
            )
        return facts

    def _verify_repo_clean(self):
        """检查分析操作是否意外修改了仓库文件，如有则还原。"""
        if self.git_guard and self.git_guard.is_dirty():
            logger.warning("CLI 分析后检测到仓库文件被修改，正在还原...")
            self.git_guard.discard_changes()

    def _truncate(self, text: str) -> str:
        """截断超长文本，在换行边界处截断以保证 Markdown 结构完整。"""
        if len(text) > self.max_comment_length:
            cutoff = self.max_comment_length - 50
            # 在换行处截断，避免切割到代码块中间
            newline_pos = text.rfind("\n", 0, cutoff)
            if newline_pos > cutoff // 2:
                cutoff = newline_pos
            return text[:cutoff] + "\n\n... [回复过长，已截断]"
        return text
