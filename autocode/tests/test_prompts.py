import unittest

from prompts import (
    _truncate_pr_body_for_feedback,
    build_execution_failure_reply,
    build_execution_prompt,
    build_review_feedback_prompt,
    format_execution_plan_reply,
    sanitize_public_reply,
)


class PromptFactsTests(unittest.TestCase):
    def test_execution_prompt_includes_restricted_paths_and_validation_commands(self):
        prompt = build_execution_prompt(
            title="Fix webhook payload",
            body="Handle empty payload safely",
            repo_name="demo/repo",
            task_type="bug_fix",
            plan_json='{"goal": "fix"}',
            allowed_paths=["main.py", "tests/test_webhook.py"],
            validation_commands=["pytest tests/test_webhook.py", "ruff check ."],
            max_changed_files=3,
            max_added_lines=120,
            max_deleted_lines=40,
        )

        self.assertIn("禁止使用 git merge", prompt)
        self.assertIn("不要破坏现有公共接口、导入路径、CLI 参数或模块级导出", prompt)
        self.assertIn("不要新增更严格的运行前提", prompt)
        self.assertIn("main.py", prompt)
        self.assertIn("tests/test_webhook.py", prompt)
        self.assertIn("pytest tests/test_webhook.py", prompt)
        self.assertIn("ruff check .", prompt)
        self.assertIn("改动文件数不超过 3 个", prompt)
        self.assertIn("新增行数不超过 120 行，删除行数不超过 40 行", prompt)

    def test_docs_todo_prompt_enforces_doc_only_scope_and_external_validation(self):
        prompt = build_execution_prompt(
            title="Trim README",
            body="Move details into docs",
            repo_name="demo/repo",
            task_type="todo_refactor",
            plan_json='{"goal": "trim docs"}',
            allowed_paths=["README.md", "docs/full-guide.md"],
            validation_commands=["./scripts/ci_gate.sh flake8", "./scripts/ci_gate.sh offline-tests"],
        )

        self.assertIn("本任务只允许修改以下路径", prompt)
        self.assertIn("README.md", prompt)
        self.assertIn("docs/full-guide.md", prompt)
        self.assertIn("这是纯文档任务，禁止修改文档范围之外的代码、测试、配置和数据文件", prompt)
        self.assertIn("必须运行与本次改动直接相关的测试命令并确认通过", prompt)
        self.assertIn("返回后系统会统一执行的校验命令", prompt)

    def test_execution_plan_reply_includes_blocked_reasons(self):
        reply = format_execution_plan_reply(
            header="## AutoCode Plan",
            task_type="todo_refactor",
            action="auto_fix",
            plan_summary="精简 README 并下沉细节",
            risk_level="medium",
            estimated_files=["README.md"],
            acceptance_criteria=["README 更聚焦"],
            needs_human_approval=False,
            blocked_reasons=["README 信息优先级仍需人工拍板"],
        )

        self.assertIn("当前阻塞/注意事项", reply)
        self.assertIn("README 信息优先级仍需人工拍板", reply)

    def test_execution_failure_reply_includes_actionable_next_step(self):
        reply = build_execution_failure_reply(
            "validation",
            "验证失败:\ntest:FAIL pytest tests/test_app.py\nAssertionError",
        )

        self.assertIn("建议下一步", reply)
        self.assertIn("继续修", reply)
        self.assertIn("修复测试或 lint", reply)

    def test_execution_prompt_includes_previous_failure_reasons(self):
        prompt = build_execution_prompt(
            title="Fix webhook payload",
            body="Handle empty payload safely",
            repo_name="demo/repo",
            task_type="bug_fix",
            plan_json='{"goal": "fix"}',
            allowed_paths=["main.py"],
            validation_commands=["pytest"],
            previous_failure_reasons=["验证失败: lint error in main.py", "补丁超出范围限制"],
        )

        self.assertIn("前几轮执行失败原因", prompt)
        self.assertIn("验证失败: lint error in main.py", prompt)
        self.assertIn("补丁超出范围限制", prompt)

    def test_execution_prompt_omits_failure_section_when_no_history(self):
        prompt = build_execution_prompt(
            title="Fix webhook payload",
            body="Handle empty payload safely",
            repo_name="demo/repo",
            task_type="bug_fix",
            plan_json='{"goal": "fix"}',
            allowed_paths=["main.py"],
            validation_commands=["pytest"],
            previous_failure_reasons=[],
        )

        self.assertNotIn("前几轮执行失败原因", prompt)

    def test_review_feedback_prompt_includes_repair_round_and_history(self):
        prompt = build_review_feedback_prompt(
            pr_title="fix: handle empty payload (#10)",
            pr_body="## Background\n- 修复空 payload 处理",
            repo_name="demo/repo",
            review_feedback=["请更新错误处理逻辑"],
            repair_round=3,
            previous_repair_notes=["review feedback 执行后未产生改动"],
        )

        self.assertIn("第 3 轮", prompt)
        self.assertIn("前几轮修复情况", prompt)
        self.assertIn("review feedback 执行后未产生改动", prompt)

    def test_review_feedback_prompt_omits_history_when_first_round(self):
        prompt = build_review_feedback_prompt(
            pr_title="fix: handle empty payload (#10)",
            pr_body="## Background\n- 修复空 payload 处理",
            repo_name="demo/repo",
            review_feedback=["请更新错误处理逻辑"],
        )

        self.assertNotIn("前几轮修复情况", prompt)
        self.assertNotIn("当前修复轮次", prompt)


class TruncatePrBodyTests(unittest.TestCase):
    def test_short_body_unchanged(self):
        body = "## Goal\n修复空 payload 处理"
        self.assertEqual(body, _truncate_pr_body_for_feedback(body))

    def test_empty_body_returns_default(self):
        self.assertEqual("无", _truncate_pr_body_for_feedback(""))
        self.assertEqual("无", _truncate_pr_body_for_feedback(None))

    def test_long_body_truncated(self):
        body = "## Goal\n" + "x" * 700
        result = _truncate_pr_body_for_feedback(body, max_chars=100)
        self.assertIn("... [截断]", result)
        self.assertLessEqual(len(result), 200)

    def test_skips_rollback_section(self):
        body = (
            "## Goal\n修复空 payload\n"
            "## 回滚方案\n很长的回滚说明 " * 20 + "\n"
            "## Summary\n总结"
        )
        result = _truncate_pr_body_for_feedback(body, max_chars=600)
        self.assertNotIn("回滚方案", result)
        self.assertIn("Goal", result)

    def test_skips_verification_section(self):
        body = (
            "## Goal\n修复空 payload\n"
            "## Verification\nAll tests passed " + "x" * 100 + "\n"
            "## Summary\n总结"
        )
        result = _truncate_pr_body_for_feedback(body, max_chars=80)
        self.assertNotIn("Verification", result)

    def test_skips_risk_section_chinese(self):
        body = (
            "## Goal\n修复空 payload\n"
            "## 兼容性与风险\n高风险改动 " + "x" * 100 + "\n"
            "## Summary\n总结"
        )
        result = _truncate_pr_body_for_feedback(body, max_chars=80)
        self.assertNotIn("兼容性与风险", result)
        self.assertIn("Summary", result)

    def test_resumes_after_skipped_section(self):
        body = (
            "## Goal\n修复空 payload\n"
            "## 验证结果\nPASS " + "x" * 100 + "\n"
            "## Next Steps\n跟进"
        )
        result = _truncate_pr_body_for_feedback(body, max_chars=80)
        self.assertNotIn("验证结果", result)
        self.assertIn("Next Steps", result)

class SanitizePublicReplyTests(unittest.TestCase):
    """Regression tests for tool-trace and local-path stripping."""

    def test_box_drawing_traces_stripped(self):
        raw = (
            '│ "def prefetch_stock_names" (data_provider)\n'
            "└ 11 lines found\n"
            "● Search (grep)\n"
            "├ foo\n"
            "实际有用的内容。\n"
        )
        result = sanitize_public_reply(raw)
        self.assertNotIn("│", result)
        self.assertNotIn("└", result)
        self.assertNotIn("●", result)
        self.assertNotIn("├", result)
        self.assertIn("实际有用的内容", result)

    def test_local_paths_replaced(self):
        raw = "文件位于 /home/user/autocode-example_repo/src/data/stock_mapping.py 中"
        result = sanitize_public_reply(raw)
        self.assertNotIn("/home/user", result)
        self.assertIn("stock_mapping.py", result)

    def test_mixed_trace_and_content_preserved(self):
        raw = (
            "│ src/data/stock_mapping.py\n"
            "│ data_provider/base.py\n"
            "│ wc -l /home/user/foo.py\n"
            "\n"
            "当前版本已基本解决。\n"
            "\n"
            "**建议排查**\n"
            "- 确认版本\n"
        )
        result = sanitize_public_reply(raw)
        self.assertIn("已基本解决", result)
        self.assertIn("建议排查", result)
        self.assertNotIn("│", result)
        self.assertNotIn("/home/user", result)

    def test_bot_footer_stripped(self):
        raw = (
            "这是一条公开回复。\n"
            "\n"
            "---\n"
            "*🤖 此回复由 AutoCode Bot 自动生成，仅供参考。如有疑问请 @维护者。*"
        )
        result = sanitize_public_reply(raw)
        self.assertEqual("这是一条公开回复。", result)


if __name__ == "__main__":
    unittest.main()
