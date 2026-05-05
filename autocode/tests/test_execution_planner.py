import json
import unittest

from adapters.base import BaseLLMAdapter, LLMError
from agent.execution_planner import ExecutionPlan, triage_and_plan


class StubAdapter(BaseLLMAdapter):
    def __init__(self, response: str):
        self._response = response

    def analyze(self, prompt: str, system: str = "") -> str:
        del prompt, system
        return self._response

    def with_working_dir(self, working_dir: str | None) -> "StubAdapter":
        del working_dir
        return self


class ExplodingAdapter(BaseLLMAdapter):
    def __init__(self, exc: Exception):
        self._exc = exc

    def analyze(self, prompt: str, system: str = "") -> str:
        del prompt, system
        raise self._exc

    def with_working_dir(self, working_dir: str | None) -> "ExplodingAdapter":
        del working_dir
        return self


class TriageAndPlanTests(unittest.TestCase):
    def _call(self, adapter):
        return triage_and_plan(
            title="Fix empty payload crash",
            body="Handle None payload in webhook handler",
            labels=["bug"],
            repo_name="demo/repo",
            analyzer=adapter,
        )

    def test_parses_valid_combined_response(self):
        response = json.dumps({
            "triage": {
                "task_type": "bug_fix",
                "action": "auto_fix",
                "risk_level": "low",
                "reason": "明确的空值处理缺陷",
                "confidence": 0.9,
            },
            "plan": {
                "goal": "修复 webhook handler 中的 None payload 崩溃",
                "assumptions": ["现有测试覆盖该路径"],
                "acceptance_criteria": ["None payload 不再抛异常"],
                "risk_level": "low",
                "estimated_files": ["src/webhook.py"],
                "suggested_tests": ["pytest tests/test_webhook.py"],
                "needs_human_approval": False,
                "blocked_reasons": [],
            },
        })
        triage, plan = self._call(StubAdapter(response))
        self.assertEqual(triage.task_type, "bug_fix")
        self.assertEqual(triage.action, "auto_fix")
        self.assertFalse(triage.degraded)
        self.assertEqual(plan.goal, "修复 webhook handler 中的 None payload 崩溃")
        self.assertFalse(plan.needs_human_approval)

    def test_degrades_on_llm_error(self):
        triage, plan = self._call(ExplodingAdapter(LLMError("timeout")))
        self.assertTrue(triage.degraded)
        self.assertEqual(triage.task_type, "high_risk_feature")
        self.assertEqual(triage.action, "plan_only")
        self.assertTrue(plan.needs_human_approval)

    def test_degrades_on_invalid_json(self):
        triage, plan = self._call(StubAdapter("this is not JSON at all"))
        self.assertTrue(triage.degraded)
        self.assertEqual(triage.task_type, "high_risk_feature")
        self.assertTrue(plan.needs_human_approval)

    def test_handles_missing_triage_key(self):
        response = json.dumps({
            "plan": {
                "goal": "修复问题",
                "assumptions": [],
                "acceptance_criteria": [],
                "risk_level": "low",
                "estimated_files": [],
                "suggested_tests": [],
                "needs_human_approval": False,
                "blocked_reasons": [],
            },
        })
        triage, plan = self._call(StubAdapter(response))
        # triage defaults from empty dict
        self.assertIsNotNone(triage.task_type)
        self.assertEqual(plan.goal, "修复问题")

    def test_handles_missing_plan_key(self):
        response = json.dumps({
            "triage": {
                "task_type": "bug_fix",
                "action": "auto_fix",
                "risk_level": "low",
                "reason": "明确缺陷",
                "confidence": 0.85,
            },
        })
        triage, plan = self._call(StubAdapter(response))
        self.assertEqual(triage.task_type, "bug_fix")
        # plan defaults from empty dict
        self.assertIsNotNone(plan.goal)

    def test_handles_wrapped_json_with_markdown(self):
        inner = json.dumps({
            "triage": {
                "task_type": "todo_refactor",
                "action": "auto_fix",
                "risk_level": "low",
                "reason": "文档清理",
                "confidence": 0.8,
            },
            "plan": {
                "goal": "清理 TODO 注释",
                "assumptions": [],
                "acceptance_criteria": ["无残留 TODO"],
                "risk_level": "low",
                "estimated_files": ["src/app.py"],
                "suggested_tests": [],
                "needs_human_approval": False,
                "blocked_reasons": [],
            },
        })
        response = f"```json\n{inner}\n```"
        triage, plan = self._call(StubAdapter(response))
        self.assertEqual(triage.task_type, "todo_refactor")
        self.assertEqual(plan.goal, "清理 TODO 注释")


if __name__ == "__main__":
    unittest.main()
