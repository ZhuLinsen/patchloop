"""AutoCode 任务分诊器。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from adapters.base import BaseLLMAdapter, LLMError
from agent.json_utils import extract_json_object
from prompts import build_execution_triage_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoCodeTriageResult:
    task_type: str
    action: str
    risk_level: str
    reason: str
    confidence: float
    degraded: bool = False
    internal_error: str = ""

    @property
    def is_rejected(self) -> bool:
        return self.action == "reject"

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "action": self.action,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "confidence": self.confidence,
            "degraded": self.degraded,
            "internal_error": self.internal_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutoCodeTriageResult":
        task_type = str(data.get("task_type", "high_risk_feature") or "high_risk_feature").strip()
        if task_type not in {"bug_fix", "todo_refactor", "small_feature", "high_risk_feature"}:
            task_type = "high_risk_feature"
        action = str(data.get("action", "plan_only") or "plan_only").strip()
        if action not in {"auto_fix", "implement", "plan_only", "reject"}:
            action = "plan_only"
        risk_level = str(data.get("risk_level", "high") or "high").strip()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "high"
        return cls(
            task_type=task_type,
            action=action,
            risk_level=risk_level,
            reason=str(data.get("reason", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            degraded=bool(data.get("degraded", False)),
            internal_error=str(data.get("internal_error", "") or ""),
        )


def triage_issue_for_execution(
    *,
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    analyzer: BaseLLMAdapter,
    local_code_context: str = "",
) -> AutoCodeTriageResult:
    prompt = build_execution_triage_prompt(
        title=title,
        body=body or "",
        labels=labels,
        repo_name=repo_name,
        local_code_context=local_code_context,
    )
    try:
        raw = analyzer.analyze(prompt, system="你是精确的 AutoCode JSON 分诊器，只输出 JSON。")
        return _parse_triage_response(raw)
    except (LLMError, ValueError) as exc:
        logger.warning("AutoCode 分诊失败，降级为 plan_only: %s", exc)
        return AutoCodeTriageResult(
            task_type="high_risk_feature",
            action="plan_only",
            risk_level="high",
            reason="自动分诊失败，保守降级",
            confidence=0.0,
            degraded=True,
            internal_error=str(exc),
        )


def _parse_triage_response(raw: str) -> AutoCodeTriageResult:
    data = extract_json_object(raw, context="triage 回复")

    task_type = str(data.get("task_type", "high_risk_feature")).strip()
    if task_type not in {"bug_fix", "todo_refactor", "small_feature", "high_risk_feature"}:
        task_type = "high_risk_feature"

    action = str(data.get("action", "plan_only")).strip()
    if action not in {"auto_fix", "implement", "plan_only", "reject"}:
        action = "plan_only"

    risk_level = str(data.get("risk_level", "high")).strip()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "high"

    reason = str(data.get("reason", "")).strip() or "未提供理由"
    confidence = float(data.get("confidence", 0.0) or 0.0)
    return AutoCodeTriageResult(
        task_type=task_type,
        action=action,
        risk_level=risk_level,
        reason=reason,
        confidence=confidence,
    )
