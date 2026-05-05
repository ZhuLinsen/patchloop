"""AutoCode 策略引擎。"""
from __future__ import annotations

from dataclasses import dataclass

from config import AutoCodeConfig
from agent.autocode_classifier import AutoCodeTriageResult
from agent.execution_planner import ExecutionPlan


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reasons: list[str]


class ExecutionPolicyEngine:
    """根据配置、分诊结果和计划结果决定是否允许自动执行。"""

    def __init__(self, config: AutoCodeConfig, default_branch: str):
        self.config = config
        self.default_branch = default_branch

    def evaluate_issue(
        self,
        *,
        triage: AutoCodeTriageResult,
        plan: ExecutionPlan,
        labels: list[str],
        explicit_command: bool = False,
        owner_authored: bool = False,
    ) -> PolicyDecision:
        hard_reasons: list[str] = []
        approval_reasons: list[str] = []
        has_owner_authorization = explicit_command or owner_authored

        if triage.is_rejected:
            hard_reasons.append("任务被分诊器判定为不适合自动执行")
            return PolicyDecision(allowed=False, requires_approval=True, reasons=hard_reasons)

        if self.config.allowed_issue_labels:
            lowered = {label.lower() for label in labels}
            allowed_labels = {label.lower() for label in self.config.allowed_issue_labels}
            if lowered.isdisjoint(allowed_labels):
                hard_reasons.append("Issue 未命中允许自动执行的标签")

        if triage.task_type in set(self.config.require_approval_for):
            approval_reasons.append(f"任务类型 `{triage.task_type}` 需要人工批准")

        if plan.needs_human_approval:
            if self.config.allow_high_risk_autorun:
                # 高风险自动执行已开启：仅当 blocked_reasons 非空时才阻塞，
                # needs_human_approval 本身不再构成硬阻塞。
                if plan.blocked_reasons:
                    approval_reasons.extend(plan.blocked_reasons)
            else:
                approval_reasons.append("结构化计划要求人工批准")
                approval_reasons.extend(plan.blocked_reasons)

        if (triage.risk_level == "high" or plan.risk_level == "high") and not self.config.allow_high_risk_autorun:
            approval_reasons.append("风险等级为 high")

        if triage.task_type == "small_feature" and not has_owner_authorization and not self._allow_feature_autorun():
            approval_reasons.append("feature 默认不在 issue opened 时自动执行")

        reasons = list(hard_reasons)
        if not has_owner_authorization:
            reasons.extend(approval_reasons)

        allowed = not reasons
        requires_approval = bool(approval_reasons) or bool(hard_reasons)
        return PolicyDecision(allowed=allowed, requires_approval=requires_approval, reasons=reasons)

    def validate_publish_target(self, base_branch: str) -> PolicyDecision:
        reasons: list[str] = []
        protected_branch = str(self.default_branch or "main").strip() or "main"
        if self.config.forbid_main_base_branch and base_branch == protected_branch:
            reasons.append(f"默认分支为 {protected_branch}，当前策略不允许作为发布目标")
        return PolicyDecision(allowed=not reasons, requires_approval=bool(reasons), reasons=reasons)

    def is_blocked_path(self, path: str) -> bool:
        normalized = _normalize_rel_path(path)
        for blocked_prefix in self.config.blocked_paths:
            candidate = _normalize_rel_path(blocked_prefix)
            if not candidate:
                continue
            if candidate.endswith("/"):
                directory = candidate.rstrip("/")
                if normalized == directory or normalized.startswith(candidate):
                    return True
                continue
            if normalized == candidate or normalized.startswith(candidate + "/"):
                return True
        return False

    def _allow_feature_autorun(self) -> bool:
        # feature 不再自动执行，统一由作者评论"实现"触发
        return False


def _normalize_rel_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")
