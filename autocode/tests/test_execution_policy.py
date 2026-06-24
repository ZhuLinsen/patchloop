import unittest

from agent.execution_policy import ExecutionPolicyEngine
from agent.autocode_classifier import AutoCodeTriageResult
from agent.execution_planner import ExecutionPlan
from config import AutoCodeConfig


class ExecutionPolicyTests(unittest.TestCase):
    def test_validate_publish_target_blocks_main_when_strict_mode_enabled(self):
        config = AutoCodeConfig(
            enabled=True,
            forbid_main_base_branch=True,
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")

        decision = policy.validate_publish_target("main")

        self.assertFalse(decision.allowed)
        self.assertTrue(any("main" in reason for reason in decision.reasons))

    def test_validate_publish_target_blocks_non_main_default_branch_when_strict_mode_enabled(self):
        config = AutoCodeConfig(
            enabled=True,
            forbid_main_base_branch=True,
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="master")

        decision = policy.validate_publish_target("master")

        self.assertFalse(decision.allowed)
        self.assertEqual(["默认分支为 master，当前策略不允许作为发布目标"], decision.reasons)

    def test_issue_policy_requires_approval_for_small_feature(self):
        config = AutoCodeConfig(
            enabled=True,
            require_approval_for=["small_feature"],
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="small_feature",
            action="implement",
            risk_level="medium",
            reason="feature",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="add dashboard",
            assumptions=[],
            acceptance_criteria=["feature works"],
            risk_level="medium",
            estimated_files=["src/dashboard.tsx"],
            suggested_tests=["npm test"],
            needs_human_approval=False,
            blocked_reasons=[],
        )

        decision = policy.evaluate_issue(triage=triage, plan=plan, labels=[], explicit_command=False)

        self.assertFalse(decision.allowed)
        self.assertTrue(any("small_feature" in reason for reason in decision.reasons))

    def test_explicit_small_feature_is_not_blocked_by_legacy_comment_flag(self):
        config = AutoCodeConfig(
            enabled=True,
            allow_feature_on_comment=False,
            require_approval_for=[],
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="small_feature",
            action="implement",
            risk_level="low",
            reason="feature",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="add dashboard",
            assumptions=[],
            acceptance_criteria=["feature works"],
            risk_level="low",
            estimated_files=["src/dashboard.tsx"],
            suggested_tests=["npm test"],
            needs_human_approval=False,
            blocked_reasons=[],
        )

        decision = policy.evaluate_issue(triage=triage, plan=plan, labels=[], explicit_command=True)

        self.assertTrue(decision.allowed)
        self.assertEqual([], decision.reasons)

    def test_owner_authored_small_feature_still_requires_explicit_approval(self):
        config = AutoCodeConfig(
            enabled=True,
            allow_feature_on_comment=False,
            require_approval_for=["small_feature"],
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="small_feature",
            action="plan_only",
            risk_level="medium",
            reason="owner requested feature",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="add dashboard",
            assumptions=[],
            acceptance_criteria=["feature works"],
            risk_level="medium",
            estimated_files=["src/dashboard.tsx"],
            suggested_tests=["npm test"],
            needs_human_approval=False,
            blocked_reasons=[],
        )

        decision = policy.evaluate_issue(
            triage=triage,
            plan=plan,
            labels=[],
            explicit_command=False,
            owner_authored=True,
        )

        self.assertFalse(decision.allowed)
        self.assertTrue(any("small_feature" in reason for reason in decision.reasons))
        self.assertTrue(any("feature 默认" in reason for reason in decision.reasons))

    def test_explicit_command_counts_as_manual_approval_for_high_risk_bugfix(self):
        config = AutoCodeConfig(
            enabled=True,
            require_approval_for=["high_risk_feature"],
            allow_high_risk_autorun=False,
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="bug_fix",
            action="implement",
            risk_level="high",
            reason="security bug",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="fix auth regression",
            assumptions=[],
            acceptance_criteria=["auth works"],
            risk_level="high",
            estimated_files=["auth.py"],
            suggested_tests=["pytest tests/test_auth.py"],
            needs_human_approval=True,
            blocked_reasons=[],
        )

        manual_decision = policy.evaluate_issue(triage=triage, plan=plan, labels=[], explicit_command=True)
        automatic_decision = policy.evaluate_issue(triage=triage, plan=plan, labels=[], explicit_command=False)

        self.assertTrue(manual_decision.allowed)
        self.assertEqual([], manual_decision.reasons)
        self.assertFalse(automatic_decision.allowed)
        self.assertTrue(any("风险等级" in reason for reason in automatic_decision.reasons))

    def test_issue_policy_allows_high_risk_when_config_enabled(self):
        config = AutoCodeConfig(
            enabled=True,
            require_approval_for=["small_feature"],
            allow_high_risk_autorun=True,
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="bug_fix",
            action="implement",
            risk_level="high",
            reason="security bug",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="fix auth regression",
            assumptions=[],
            acceptance_criteria=["auth works"],
            risk_level="high",
            estimated_files=["auth.py"],
            suggested_tests=["pytest tests/test_auth.py"],
            needs_human_approval=False,
            blocked_reasons=[],
        )

        decision = policy.evaluate_issue(triage=triage, plan=plan, labels=[], explicit_command=False)

        self.assertTrue(decision.allowed)
        self.assertEqual([], decision.reasons)

    def test_issue_policy_does_not_block_on_blocked_reasons_without_human_approval(self):
        config = AutoCodeConfig(
            enabled=True,
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")
        triage = AutoCodeTriageResult(
            task_type="todo_refactor",
            action="auto_fix",
            risk_level="medium",
            reason="docs cleanup",
            confidence=0.9,
        )
        plan = ExecutionPlan(
            goal="trim readme",
            assumptions=["保留高频内容"],
            acceptance_criteria=["README 更聚焦"],
            risk_level="medium",
            estimated_files=["README.md"],
            suggested_tests=["manual review"],
            needs_human_approval=False,
            blocked_reasons=["README 信息优先级存在主观判断"],
        )

        decision = policy.evaluate_issue(triage=triage, plan=plan, labels=["documentation"], explicit_command=False)

        self.assertTrue(decision.allowed)
        self.assertEqual([], decision.reasons)

    def test_is_blocked_path_does_not_treat_env_example_as_real_env(self):
        config = AutoCodeConfig(
            enabled=True,
            blocked_paths=[".env", "deploy/"],
            pr_base_branch="release",
        )
        policy = ExecutionPolicyEngine(config, default_branch="main")

        self.assertTrue(policy.is_blocked_path(".env"))
        self.assertFalse(policy.is_blocked_path(".env.example"))
        self.assertTrue(policy.is_blocked_path("deploy/app.yaml"))


if __name__ == "__main__":
    unittest.main()
