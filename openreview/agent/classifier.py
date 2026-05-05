"""Issue 分类器 — 基于内容与本地上下文判断是否适合自动回答。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from adapters.base import BaseLLMAdapter, LLMError
from prompts import build_issue_classify_prompt

logger = logging.getLogger(__name__)

# 轻量级预过滤常量 — 仅在多重信号吻合时跳过 LLM 分类
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
_TRUSTED_AUTHOR_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_ACTIONABLE_FEATURE_MARKERS = (
    "目标",
    "建议实施",
    "实施顺序",
    "验收",
    "重点文件",
    "非目标",
    "验证建议",
    "scope",
    "non-goal",
    "acceptance",
    "implementation",
    "validation",
)
_QUESTION_SIGNALS = (
    "?", "怎么", "如何", "为什么", "为何", "不能", "不行",
    "失败", "报错", "错误", "异常", "bug", "问题",
    "how", "why", "error", "fail", "crash", "broken",
)


@dataclass
class ClassificationResult:
    classification: str  # "ANSWERABLE" or "UNANSWERABLE"
    reason: str
    confidence: float
    degraded: bool = False
    internal_error: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.classification == "ANSWERABLE"


def quick_classify(
    title: str,
    body: str,
    labels: list[str],
    author_association: str = "",
) -> ClassificationResult | None:
    """
    轻量级预过滤：对高置信非问答类 Issue 直接返回 UNANSWERABLE，
    省去 LLM 分类调用。仅在多重信号吻合时生效（无问答信号 + 标签/标题匹配）。

    Returns None 表示需要 LLM 分类。
    """
    combined_text = f"{title}\n{body}".lower()

    if any(signal in combined_text for signal in _QUESTION_SIGNALS):
        return None

    lower_labels = {l.lower().strip() for l in labels}
    lower_title = title.lower().strip()

    hit_skip = lower_labels & _SKIP_LABELS
    if hit_skip:
        return ClassificationResult(
            classification="UNANSWERABLE",
            reason=f"预过滤: 标签 [{', '.join(sorted(hit_skip))}] 且无问答信号",
            confidence=0.9,
        )

    hit_feature = lower_labels & _FEATURE_LABELS
    has_feature_prefix = any(lower_title.startswith(p) for p in _FEATURE_TITLE_PREFIXES)

    if hit_feature or has_feature_prefix:
        if _is_trusted_actionable_feature_spec(body, author_association):
            return ClassificationResult(
                classification="ANSWERABLE",
                reason="功能实施分析",
                confidence=0.85,
            )
        reason_parts = []
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


def _is_trusted_actionable_feature_spec(body: str, author_association: str) -> bool:
    if str(author_association or "").upper() not in _TRUSTED_AUTHOR_ASSOCIATIONS:
        return False
    text = str(body or "").strip()
    if len(text) < 180:
        return False
    lowered = text.lower()
    marker_hits = sum(1 for marker in _ACTIONABLE_FEATURE_MARKERS if marker in lowered or marker in text)
    return marker_hits >= 2


def classify_issue(
    title: str,
    body: str,
    labels: list[str],
    analyzer: BaseLLMAdapter,
    repo_name: str = "",
    local_code_context: str = "",
    triage_context: str = "",
    discussion_context: str = "",
    author_association: str = "",
) -> ClassificationResult:
    """
    综合分类 Issue。

    先尝试轻量级预过滤，命中则跳过 LLM 调用；
    否则交给模型结合 Issue 内容、标签和本地代码上下文判断。
    """
    quick_result = quick_classify(title, body, labels, author_association=author_association)
    if quick_result is not None:
        logger.info(
            "Issue 预过滤分类: %s (reason: %s)",
            quick_result.classification,
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
        author_association=author_association,
    )
    try:
        raw_response = analyzer.analyze(prompt, system="你是一个精确的分类助手，只输出 JSON。")
        result = _parse_classification(raw_response)
        logger.info("Issue CLI 分类: %s (confidence: %.2f, reason: %s)",
                     result.classification, result.confidence, result.reason)
        return result
    except (LLMError, ValueError) as e:
        logger.warning("LLM 分类失败，默认为 UNANSWERABLE: %s", e)
        # 安全降级: 无法判断时不公开回复，避免将系统错误暴露给用户
        return ClassificationResult(
            classification="UNANSWERABLE",
            reason="自动分类服务暂时不可用",
            confidence=0.0,
            degraded=True,
            internal_error=str(e),
        )


def _parse_classification(raw: str) -> ClassificationResult:
    """解析 LLM 返回的 JSON 分类结果"""
    # 尝试提取 JSON 块
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 兜底: 在整个文本中寻找 JSON 对象
        import re
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"无法从 LLM 回复中解析 JSON: {raw[:200]}")

    classification = data.get("classification", "UNANSWERABLE").upper()
    if classification not in ("ANSWERABLE", "UNANSWERABLE"):
        classification = "UNANSWERABLE"

    return ClassificationResult(
        classification=classification,
        reason=data.get("reason", "未知"),
        confidence=float(data.get("confidence", 0.5)),
    )
