import unittest
from contextlib import contextmanager

from adapters.base import BaseLLMAdapter
from agent.core import ReviewAgent
from agent.git_guard import PRDiffSnapshot


class _StubAnalyzer(BaseLLMAdapter):
    def analyze(self, prompt: str, system: str = "") -> str:
        return "OK"


class _IssueClassifierStubAnalyzer(BaseLLMAdapter):
    def analyze(self, prompt: str, system: str = "") -> str:
        if "Issue 分类助手" in prompt:
            return '{"classification":"UNANSWERABLE","reason":"新功能请求","confidence":0.9}'
        return "不应被调用"


class _RecordingIssueAnalyzer(BaseLLMAdapter):
    def __init__(self):
        self.prompts = []

    def analyze(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        return (
            "这个功能应按主闭环拆成事件协议、意图解析、会话状态、首批工具和诊断入口五段推进。"
            "\n- 先固化 SSE 事件协议，避免 Web Chat 继续只等最终回答。"
            "\n- 复用现有解析、历史和持仓服务，不新增平行业务逻辑。"
            "\n- 验证重点放在旧客户端 final 兼容、参数脱敏、低置信度确认和工具失败降级。"
        )


class ReviewAgentFactExtractionTests(unittest.TestCase):
    def test_extracts_flexible_issue_references_and_ci_gate_mentions(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")
        body = """
        Verification:
        - scripts/ci_gate.sh
        - Refs Issue：#602
        - Closes #603
        """

        facts = agent._build_pr_review_facts(
            body=body,
            changed_files=["README.md", "docs/CHANGELOG.md"],
            extra_facts=["当前 CI 状态：pending", "当前 merge 状态：存在冲突"],
        )

        self.assertIn("PR 描述提到 `ci_gate.sh`：是", facts)
        self.assertIn("PR 描述中的 issue 关联：Refs Issue：#602 | Closes #603", facts)
        self.assertIn("当前 CI 状态：pending", facts)
        self.assertIn("当前 merge 状态：存在冲突", facts)
        self.assertTrue(any("低风险文档/注释类审查提示" in fact for fact in facts))

    def test_builds_local_snapshot_review_facts(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")
        snapshot = PRDiffSnapshot(
            base_ref="main",
            compare_ref="origin/main",
            merge_base="1234567890abcdef",
            changed_files=["src/app.py"],
            diff_stat="1 file changed",
            diff_excerpt="diff --git a/src/app.py b/src/app.py",
        )

        facts = agent._build_local_snapshot_review_facts(snapshot)

        self.assertIn("本地评审方式：已拉取并检出 PR 最新 head 后分析", facts)
        self.assertIn("本地 diff 比较基线：origin/main", facts)
        self.assertIn("本地 merge-base：1234567890ab", facts)
        self.assertIn("完整改动文件列表来源：本地 git diff --name-only", facts)

    def test_review_scope_fact_relaxes_docs_and_docstring_only_process_blockers(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")

        docs_fact = agent._review_scope_fact(["docs/guide.md", "docs/img_11.png"])
        mixed_fact = agent._review_scope_fact(["docs/guide.md", "src/sender.py"])

        self.assertIn("均为文档或图片资源", docs_fact)
        self.assertIn("流程性缺口不能单独阻断", docs_fact)
        self.assertIn("若代码文件仅改注释或 docstring", mixed_fact)
        self.assertIn("流程性缺口不能单独阻断", mixed_fact)

    def test_external_model_changes_add_compatibility_review_facts(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")

        facts = agent._build_pr_review_facts(
            body=(
                "DeepSeek 官方推荐 deepseek-v4-flash，并说明 deepseek-chat "
                "will be deprecated on 2026/07/24"
            ),
            changed_files=["src/config.py", "apps/dsa-web/src/components/settings/LLMChannelEditor.tsx"],
            diff_excerpt=(
                "- OPENAI_BASE_URL=https://api.deepseek.com/v1\n"
                "+ OPENAI_BASE_URL=https://api.deepseek.com\n"
                "- LITELLM_MODEL=deepseek/deepseek-chat\n"
                "+ LITELLM_MODEL=deepseek/deepseek-v4-flash\n"
                "+ sanitizeRuntimeConfigForSave(runtimeConfig, availableModels)"
            ),
        )

        joined = "\n".join(facts)
        self.assertIn("外部模型/API 兼容风险提示", joined)
        self.assertIn("外部事实核验提示", joined)
        self.assertIn("运行时配置迁移提示", joined)
        self.assertIn("不能只因 CI 成功或 PR 描述完整就判低风险", joined)

    def test_docs_only_model_mentions_are_scoped_to_documentation_risk(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")

        facts = agent._build_pr_review_facts(
            body=(
                "Docs only. No runtime changes and no new external compatibility semantics. "
                "Existing LLM provider and Base URL behavior is documented from in-repo config."
            ),
            changed_files=["README.md", "docs/full-guide.md"],
            diff_excerpt=(
                "+ OPENAI_BASE_URL=https://api.deepseek.com\n"
                "+ LITELLM_MODEL=deepseek/deepseek-chat\n"
                "+ fallback models keep existing behavior\n"
            ),
        )

        joined = "\n".join(facts)
        self.assertIn("文档型外部模型/API 提示", joined)
        self.assertIn("文档型运行时配置提示", joined)
        self.assertIn("不要仅因缺少外部官方来源判为阻断", joined)
        self.assertNotIn("评审必须核验官方来源", joined)
        self.assertNotIn("保存前清理逻辑变更", joined)

    def test_discussion_facts_throttle_repeated_process_blockers(self):
        agent = ReviewAgent(analyzer=_StubAnalyzer(), repo_name="demo/repo")

        facts = agent._build_discussion_review_facts(
            "[review] OpenReview Bot 自动生成：缺少 description 回滚方案。\n"
            "[review] OpenReview Bot 自动生成：README 和验证证据还需要补充。"
        )

        self.assertEqual(1, len(facts))
        self.assertIn("流程型阻断节流提示", facts[0])
        self.assertIn("降级为建议", facts[0])


class ReviewAgentBlockedReasonTests(unittest.TestCase):
    def test_handle_pr_returns_blocked_reason_when_local_diff_build_fails(self):
        class _StubGitGuard:
            @contextmanager
            def safe_branch(self, *_args, **_kwargs):
                yield type("Prep", (), {"ready": True, "reason": "", "switched": True})()

            def build_pr_diff_snapshot(self, _base_ref: str):
                raise RuntimeError("无法解析 PR 基线分支 release")

        agent = ReviewAgent(
            analyzer=_StubAnalyzer(),
            repo_name="demo/repo",
            git_guard=_StubGitGuard(),
        )

        result = agent.handle_pr(
            "Test PR",
            "Body",
            "release",
            "feature/ref",
            "",
            "abc123",
        )

        self.assertEqual("", result.reply)
        self.assertEqual("无法解析 PR 基线分支 release", result.blocked_reason)


class ReviewAgentIssueFollowupTests(unittest.TestCase):
    def test_owner_actionable_feature_issue_gets_implementation_analysis(self):
        analyzer = _RecordingIssueAnalyzer()
        agent = ReviewAgent(
            analyzer=analyzer,
            repo_name="demo/repo",
        )

        reply, classification = agent.handle_issue(
            "[Feature] 优先做 Agent 主闭环",
            (
                "## 背景\n当前已经具备 Agent chat API、Web Chat 和 tool registry。\n"
                "## 目标\n让 Web Chat 展示意图、实体、工具调用和上下文继承。\n"
                "## 建议实施顺序\n1. 固化 Agent SSE 事件协议。\n2. 前置意图识别与实体解析。\n"
                "## 验收\n旧客户端仍能消费 final，工具参数必须脱敏。\n"
                "## 非目标\n本 issue 不包含 MCP Server 和多用户权限隔离。"
            ),
            ["enhancement"],
            author_association="OWNER",
        )

        self.assertEqual("ANSWERABLE", classification.classification)
        self.assertEqual("功能实施分析", classification.reason)
        self.assertEqual(1, len(analyzer.prompts))
        self.assertIn("功能实施分析", analyzer.prompts[0])
        self.assertIn("作者关系**: OWNER", analyzer.prompts[0])
        self.assertIn("主闭环", reply)
        self.assertNotIn("OpenReview 不会在这里替维护者拍板", reply)

    def test_followup_unanswerable_issue_does_not_repeat_placeholder(self):
        agent = ReviewAgent(
            analyzer=_IssueClassifierStubAnalyzer(),
            repo_name="demo/repo",
        )

        reply, classification = agent.handle_issue(
            "希望增加一个全新的发布流程",
            "这个需求需要单独设计",
            ["enhancement"],
            discussion_context="- [issue_comment] @alice (2026-03-25T10:00:00Z)\n  能否支持一下？",
            allow_unanswerable_reply=False,
        )

        self.assertEqual("", reply)
        self.assertEqual("UNANSWERABLE", classification.classification)


if __name__ == "__main__":
    unittest.main()
