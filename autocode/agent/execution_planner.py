"""AutoCode 结构化计划生成。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from adapters.base import BaseLLMAdapter, LLMError
from agent.autocode_classifier import AutoCodeTriageResult
from agent.json_utils import extract_json_object
from prompts import build_combined_triage_plan_prompt, build_execution_plan_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPlan:
    goal: str
    assumptions: list[str]
    acceptance_criteria: list[str]
    risk_level: str
    estimated_files: list[str]
    suggested_tests: list[str]
    needs_human_approval: bool
    blocked_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "assumptions": self.assumptions,
            "acceptance_criteria": self.acceptance_criteria,
            "risk_level": self.risk_level,
            "estimated_files": self.estimated_files,
            "suggested_tests": self.suggested_tests,
            "needs_human_approval": self.needs_human_approval,
            "blocked_reasons": self.blocked_reasons,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        risk_level = str(data.get("risk_level", "high")).strip() or "high"
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "high"
        return cls(
            goal=str(data.get("goal", "")).strip(),
            assumptions=[str(item).strip() for item in data.get("assumptions", []) if str(item).strip()],
            acceptance_criteria=[
                str(item).strip() for item in data.get("acceptance_criteria", []) if str(item).strip()
            ],
            risk_level=risk_level,
            estimated_files=[str(item).strip() for item in data.get("estimated_files", []) if str(item).strip()],
            suggested_tests=[str(item).strip() for item in data.get("suggested_tests", []) if str(item).strip()],
            needs_human_approval=bool(data.get("needs_human_approval", False)),
            blocked_reasons=[str(item).strip() for item in data.get("blocked_reasons", []) if str(item).strip()],
        )


def build_execution_plan(
    *,
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    task_type: str,
    analyzer: BaseLLMAdapter,
    local_code_context: str = "",
) -> ExecutionPlan:
    prompt = build_execution_plan_prompt(
        title=title,
        body=body or "",
        labels=labels,
        repo_name=repo_name,
        task_type=task_type,
        local_code_context=local_code_context,
    )
    try:
        raw = analyzer.analyze(prompt, system="你是精确的 JSON 规划助手，只输出 JSON。")
        return _parse_execution_plan(raw)
    except (LLMError, ValueError) as exc:
        logger.warning("AutoCode 计划生成失败，返回保守计划: %s", exc)
        return ExecutionPlan(
            goal=title.strip() or "待人工确认",
            assumptions=["自动计划失败，需人工确认实现范围"],
            acceptance_criteria=["明确需求边界并人工确认后再执行"],
            risk_level="high",
            estimated_files=[],
            suggested_tests=[],
            needs_human_approval=True,
            blocked_reasons=["自动计划服务暂不可用"],
        )


def triage_and_plan(
    *,
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    analyzer: BaseLLMAdapter,
    local_code_context: str = "",
    discussion_context: str = "",
) -> tuple[AutoCodeTriageResult, ExecutionPlan]:
    """合并分诊和计划为单次 LLM 调用。

    返回 (triage, plan) 元组。失败时返回保守降级结果。
    """
    prompt = build_combined_triage_plan_prompt(
        title=title,
        body=body or "",
        labels=labels,
        repo_name=repo_name,
        local_code_context=local_code_context,
        discussion_context=discussion_context,
    )
    try:
        raw = analyzer.analyze(prompt, system="你是精确的 AutoCode JSON 分析器，只输出 JSON。")
        return _parse_combined_response(raw, title)
    except (LLMError, ValueError) as exc:
        logger.warning("AutoCode 合并分诊+计划失败，保守降级: %s", exc)
        triage = AutoCodeTriageResult(
            task_type="high_risk_feature",
            action="plan_only",
            risk_level="high",
            reason="自动分诊失败，保守降级",
            confidence=0.0,
            degraded=True,
            internal_error=str(exc),
        )
        plan = ExecutionPlan(
            goal=title.strip() or "待人工确认",
            assumptions=["自动计划失败，需人工确认实现范围"],
            acceptance_criteria=["明确需求边界并人工确认后再执行"],
            risk_level="high",
            estimated_files=[],
            suggested_tests=[],
            needs_human_approval=True,
            blocked_reasons=["自动计划服务暂不可用"],
        )
        return triage, plan


def _parse_combined_response(raw: str, fallback_title: str) -> tuple[AutoCodeTriageResult, ExecutionPlan]:
    data = extract_json_object(raw, context="combined triage+plan 回复")

    triage_data = data.get("triage", {})
    if not isinstance(triage_data, dict):
        triage_data = {}
    triage = AutoCodeTriageResult.from_dict(triage_data)

    plan_data = data.get("plan", {})
    if not isinstance(plan_data, dict):
        plan_data = {}
    plan = ExecutionPlan.from_dict(plan_data)

    return triage, plan


def _parse_execution_plan(raw: str) -> ExecutionPlan:
    data = extract_json_object(raw, context="execution plan 回复")

    risk_level = str(data.get("risk_level", "high")).strip()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "high"

    data["risk_level"] = risk_level
    return ExecutionPlan.from_dict(data)
