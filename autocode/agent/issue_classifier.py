"""Issue 分类器 — 基于内容与本地上下文判断是否适合直接回复。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from adapters.base import BaseLLMAdapter, LLMError
from agent.json_utils import extract_json_object
from prompts import build_issue_classify_prompt

logger = logging.getLogger(__name__)

_FEATURE_LABELS = frozenset({
    "feature", "enhancement", "feature request", "feature-request",
})
_SKIP_LABELS = frozenset({
    "wontfix", "won't fix", "duplicate", "invalid",
})
_FEATURE_TITLE_PREFIXES = (
    "feature:", "enhancement:", "rfc:", "[feature]", "[enhancement]",
    "[rfc]", "feature request:",
)
_QUESTION_SIGNALS = (
    "?", "怎么", "如何", "为什么", "为何", "不能", "不行",
    "失败", "报错", "错误", "异常", "bug", "问题",
    "how", "why", "error", "fail", "crash", "broken",
)


@dataclass(frozen=True)
class ClassificationResult:
    classification: str
    reason: str
    confidence: float
    degraded: bool = False
    internal_error: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.classification == "ANSWERABLE"


def quick_classify(title: str, body: str, labels: list[str]) -> ClassificationResult | None:
    """高置信轻量预过滤。"""
    combined_text = f"{title}\n{body}".lower()
    if any(signal in combined_text for signal in _QUESTION_SIGNALS):
        return None

    lower_labels = {str(label or "").lower().strip() for label in labels}
    lower_title = str(title or "").lower().strip()

    hit_skip = lower_labels & _SKIP_LABELS
    if hit_skip:
        return ClassificationResult(
            classification="UNANSWERABLE",
            reason=f"预过滤: 标签 [{', '.join(sorted(hit_skip))}] 且无问答信号",
            confidence=0.9,
        )

    hit_feature = lower_labels & _FEATURE_LABELS
    has_feature_prefix = any(lower_title.startswith(prefix) for prefix in _FEATURE_TITLE_PREFIXES)
    if hit_feature or has_feature_prefix:
        reason_parts: list[str] = []
        if hit_feature:
            reason_parts.append(f"标签 [{', '.join(sorted(hit_feature))}]")
        if has_feature_prefix:
            reason_parts.append("标题前缀")
        return ClassificationResult(
            classification="UNANSWERABLE",
            reason=f"预过滤: {' + '.join(reason_parts)} 表明为功能请求",
            confidence=0.85,
        )

    return None


def classify_issue(
    title: str,
    body: str,
    labels: list[str],
    analyzer: BaseLLMAdapter,
    *,
    repo_name: str = "",
    local_code_context: str = "",
    triage_context: str = "",
    discussion_context: str = "",
) -> ClassificationResult:
    """判断 issue 是否更适合直接回复而不是进入代码 PR 流程。"""
    quick_result = quick_classify(title, body, labels)
    if quick_result is not None:
        logger.info(
            "issue-classifier: quick result=%s confidence=%.2f reason=%s",
            quick_result.classification,
            quick_result.confidence,
            quick_result.reason,
        )
        return quick_result

    prompt = build_issue_classify_prompt(
        title=title,
        body=body or "",
        labels=labels,
        repo_name=repo_name,
        local_code_context=local_code_context,
        triage_context=triage_context,
        discussion_context=discussion_context,
    )
    try:
        raw_response = analyzer.analyze(prompt, system="你是一个精确的分类助手，只输出 JSON。")
        data = extract_json_object(raw_response, context="issue classify 回复")
        classification = str(data.get("classification", "UNANSWERABLE") or "UNANSWERABLE").upper()
        if classification not in {"ANSWERABLE", "UNANSWERABLE"}:
            classification = "UNANSWERABLE"
        confidence = float(data.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(confidence, 1.0))
        return ClassificationResult(
            classification=classification,
            reason=str(data.get("reason", "") or "未知"),
            confidence=confidence,
        )
    except (LLMError, ValueError) as exc:
        logger.warning("issue-classifier: degraded to UNANSWERABLE because classify failed: %s", exc)
        return ClassificationResult(
            classification="UNANSWERABLE",
            reason="自动分类服务暂时不可用",
            confidence=0.0,
            degraded=True,
            internal_error=str(exc),
        )
