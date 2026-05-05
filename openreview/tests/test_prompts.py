import unittest

from prompts import (
    build_issue_analysis_prompt,
    build_pr_followup_prompt,
    build_pr_review_prompt,
    build_unanswerable_reply,
    format_followup_reply,
)


class PromptFactsTests(unittest.TestCase):
    def test_issue_analysis_prompt_includes_duplicate_and_resolved_modes(self):
        prompt = build_issue_analysis_prompt(
            title="config not loading",
            body="这个问题是不是已经修过了？",
            labels=["bug"],
            repo_name="demo/repo",
            triage_context=(
                "疑似重复 Issue（高置信）:\n"
                "- #12 [closed] config not loading\n"
                "\n修复状态线索:\n"
                "- PR #77 已合并，可能已经包含修复：fix config loading，相关提交: abcdef1"
            ),
            local_code_context="## 当前本地仓库上下文\n### 文件: README.md\n",
        )

        self.assertIn("重复 issue", prompt)
        self.assertIn("已修复", prompt)
        self.assertIn("使用/配置问题", prompt)
        self.assertIn("commit", prompt)

    def test_prompt_includes_structured_facts_and_complete_file_list(self):
        prompt = build_pr_review_prompt(
            title="Test PR",
            body="Ran ./scripts/ci_gate.sh\nRefs #602",
            repo_name="demo/repo",
            diff_excerpt="diff --git a/README.md b/README.md",
            diff_stat=" README.md | 1 +\n 1 file changed, 1 insertion(+)",
            changed_files=["README.md", "docs/CHANGELOG.md"],
            review_facts=[
                "完整改动文件数：2",
                "`docs/CHANGELOG.md` 在改动文件中：是",
            ],
            discussion_context="- [issue_comment] @alice (2026-03-25T10:00:00Z)\n  已补充验证结果",
        )

        self.assertIn("README.md", prompt)
        self.assertIn("docs/CHANGELOG.md", prompt)
        self.assertIn("结构化事实", prompt)
        self.assertIn("完整改动文件数：2", prompt)
        self.assertIn("本地 diff 统计", prompt)
        self.assertIn("当前讨论", prompt)
        self.assertIn("如果当前 CI 未完成、CI 失败，或 PR 暂时有冲突", prompt)
        self.assertIn("mergeable_state=blocked", prompt)
        self.assertIn("bypass permission", prompt)
        self.assertIn("不要仅凭这一点就把“是否可直接合入”判为不可", prompt)
        self.assertIn("本地已检出的 PR 最新 head", prompt)
        self.assertIn("先看当前 diff 新引入的真实风险", prompt)
        self.assertIn("行为/正确性风险 > 兼容性/迁移风险 > 验证缺口 > 文档/描述/模板缺口", prompt)
        self.assertIn("已有通用文档，但本次新增语义未说明", prompt)
        self.assertIn("[Correctness blocker]", prompt)
        self.assertIn("不要暗示这是第一次出现", prompt)
        self.assertIn("不能替代完整改动文件列表和本地文件快照", prompt)
        self.assertIn("低风险文档/注释类 PR", prompt)
        self.assertIn("缺少回滚方案", prompt)
        self.assertIn("不能单独作为合入阻断", prompt)
        self.assertIn("外部模型/API 兼容风险", prompt)
        self.assertIn("官方来源链接", prompt)
        self.assertIn("锁定依赖/运行时兼容验证", prompt)
        self.assertIn("文档型外部模型/API 提示", prompt)
        self.assertIn("不要仅因缺少外部官方来源判为不可合入", prompt)
        self.assertIn("用户可见提示", prompt)
        self.assertIn("静默清空", prompt)

    def test_followup_prompt_includes_discussion_context(self):
        prompt = build_pr_followup_prompt(
            title="Test PR",
            body="Fix race condition",
            repo_name="demo/repo",
            discussion_context="- [review_comment] @alice (2026-03-25T10:00:00Z)\n  这里已经修过了",
            diff_excerpt="diff --git a/app.py b/app.py",
            changed_files=["app.py"],
            review_facts=["完整改动文件数：1"],
        )

        self.assertIn("继续评审", prompt)
        self.assertIn("当前讨论", prompt)
        self.assertIn("@alice", prompt)
        self.assertIn("如果当前 CI 未完成、CI 失败，或 PR 暂时有冲突", prompt)
        self.assertIn("mergeable_state=blocked", prompt)
        self.assertIn("bypass permission", prompt)
        self.assertIn("不要仅凭这一点就把“是否仍有阻断”判为有", prompt)
        self.assertIn("本地已检出的 PR 最新 head", prompt)
        self.assertIn("现有文档已覆盖通用能力", prompt)
        self.assertIn("不要维持过重结论", prompt)
        self.assertIn("简单文档/注释修改是否需要复杂回滚方案", prompt)
        self.assertIn("缺少回滚方案只能作为建议补充", prompt)
        self.assertIn("外部模型/API 兼容风险", prompt)
        self.assertIn("当前依赖/运行时兼容", prompt)
        self.assertIn("文档型外部模型/API 提示", prompt)
        self.assertIn("要先接受已补充的 docs-only 范围说明和仓库内依据", prompt)
        self.assertIn("静默清空", prompt)

    def test_followup_reply_is_wrapped_into_consistent_sections(self):
        reply = format_followup_reply(
            "结论：部分接受，补充说明里有成立的点\n"
            "是否仍有阻断：有，`src/core.py` 的边界仍未覆盖\n"
            "说明\n"
            "- `src/core.py` 里空输入分支仍会绕过校验\n"
            "- `tests/test_core.py` 还缺对应回归\n"
        )

        self.assertIn("**跟进结论**", reply)
        self.assertIn("- **结论**：部分接受，补充说明里有成立的点", reply)
        self.assertIn("- **是否仍有阻断**：有，`src/core.py` 的边界仍未覆盖", reply)
        self.assertIn("**说明**", reply)
        self.assertIn("1. `src/core.py` 里空输入分支仍会绕过校验", reply)
        self.assertIn("2. `tests/test_core.py` 还缺对应回归", reply)

    def test_feature_unanswerable_reply_mentions_autocode_handoff(self):
        reply = build_unanswerable_reply("这是一个新功能请求，需要维护者决策")

        self.assertIn("AutoCode", reply)
        self.assertIn("继续按队列", reply)
        self.assertNotIn("暂时无法对此类问题给出合适的答复", reply)


if __name__ == "__main__":
    unittest.main()
