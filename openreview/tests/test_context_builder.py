import tempfile
import unittest
from pathlib import Path

from agent.context_builder import LocalContextBuilder


class LocalContextBuilderTests(unittest.TestCase):
    def test_issue_context_extracts_agents_sections_by_title_not_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "AGENTS.md").write_text(
                "\n".join(
                    [
                        "# Demo",
                        "## 2. Issue 分析原则",
                        "- 优先结合仓库现状回答",
                        "## 6. 建议评审输出格式",
                        "- 先给结论，再给建议",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_path / "README.md").write_text("# demo\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=2,
                max_file_chars=500,
                max_total_chars=1200,
            )

            context = builder.build_issue_context(
                title="这个配置为什么没生效",
                body="想看 issue 回复时会不会带上 AGENTS 约束",
            )

            self.assertIn("## 仓库 AGENTS 规范", context)
            self.assertIn("## 2. Issue 分析原则", context)
            self.assertIn("## 6. 建议评审输出格式", context)
            self.assertIn("先给结论，再给建议", context)

    def test_issue_context_prefers_targeted_match_snippets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "config.py").write_text(
                "\n".join(
                    [
                        "APP_NAME = 'demo'",
                        "LOCAL_REPO_PATH = '/tmp/repo'",
                        "LOCAL_REPO_DEFAULT_BRANCH = 'main'",
                        "PRIMARY_MODEL = 'gpt-5.4'",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_path / "README.md").write_text("# demo\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_issue_context(
                title="LOCAL_REPO_PATH 在哪里配置",
                body="想确认 LOCAL_REPO_DEFAULT_BRANCH 也在哪生效",
            )

            self.assertIn("## 可能相关的本地代码位置", context)
            self.assertIn("文件: config.py", context)
            self.assertIn("LOCAL_REPO_PATH", context)
            self.assertIn("2: LOCAL_REPO_PATH", context)

    def test_issue_context_prefers_docs_for_usage_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "README.md").write_text(
                "\n".join(
                    [
                        "# demo",
                        "## polling",
                        "设置 `POLL_INTERVAL_SECONDS` 后即可启用 polling 模式。",
                    ]
                ),
                encoding="utf-8",
            )
            docs_dir = repo_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "usage.md").write_text(
                "使用说明：`EVENT_SOURCE=polling` 时会读取轮询配置。",
                encoding="utf-8",
            )
            (repo_path / "service.py").write_text(
                "def run():\n    return 'ok'\n",
                encoding="utf-8",
            )

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_issue_context(
                title="怎么开启 polling 模式",
                body="需要配置什么环境变量，为什么现在没有生效？",
            )

            self.assertIn("文件: README.md", context)
            self.assertIn("文件: docs/usage.md", context)
            self.assertIn("POLL_INTERVAL_SECONDS", context)

    def test_issue_context_does_not_promote_unrelated_docs_only_by_usage_bias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "README.md").write_text(
                "# demo\n这是一个通用介绍，没有配置项。\n",
                encoding="utf-8",
            )
            (repo_path / ".env.example").write_text(
                "EVENT_SOURCE=polling\nPOLL_INTERVAL_SECONDS=30\n",
                encoding="utf-8",
            )

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=1,
                max_file_chars=500,
                max_total_chars=1200,
            )

            context = builder.build_issue_context(
                title="怎么配置 EVENT_SOURCE",
                body="想确认 polling 模式需要哪些环境变量",
            )

            self.assertIn("文件: .env.example", context)
            self.assertNotIn("文件: README.md", context)

    def test_issue_context_adds_locator_hints_for_ui_bug_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            src_dir = repo_path / "src" / "desktop"
            src_dir.mkdir(parents=True)
            (src_dir / "settings_panel.tsx").write_text(
                "export function NotificationSettings() { return <div>agent settings</div>; }\n",
                encoding="utf-8",
            )
            (repo_path / "README.md").write_text("# demo\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_issue_context(
                title="Windows 设置页通知 agent 黑屏",
                body="release 免安装包进入通知设置后黑屏，无法继续使用。",
            )

            self.assertIn("## 缺陷定位提示", context)
            self.assertIn("不要只要求用户补日志", context)
            self.assertIn("settings", context)
            self.assertIn("src/desktop/settings_panel.tsx", context)

    def test_pr_context_includes_related_existing_docs_for_workflow_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            workflows_dir = repo_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)
            (workflows_dir / "review.yml").write_text(
                "\n".join(
                    [
                        "name: review",
                        "env:",
                        "  LLM_OLLAMA_BASE_URL: ${{ vars.LLM_OLLAMA_BASE_URL }}",
                        "  LLM_OLLAMA_MODELS: ${{ vars.LLM_OLLAMA_MODELS }}",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_path / "README.md").write_text(
                "使用前请配置 `LLM_OLLAMA_BASE_URL`，并确认模型来源可访问。\n",
                encoding="utf-8",
            )
            (repo_path / ".env.example").write_text(
                "LLM_OLLAMA_MODELS=qwen2.5-coder:latest\n",
                encoding="utf-8",
            )

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_pr_context(
                changed_files=[".github/workflows/review.yml"],
            )

            self.assertIn("## PR 对应的本地文件快照", context)
            self.assertIn("文件: .github/workflows/review.yml", context)
            self.assertIn("## 可能相关的现有文档/配置", context)
            self.assertIn("文件: README.md", context)
            self.assertIn("文件: .env.example", context)
            self.assertIn("LLM_OLLAMA_BASE_URL", context)

    def test_pr_context_includes_evidence_pack_with_related_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            src_dir = repo_path / "src"
            tests_dir = repo_path / "tests"
            src_dir.mkdir()
            tests_dir.mkdir()
            (src_dir / "service.py").write_text("def run_service():\n    return True\n", encoding="utf-8")
            (tests_dir / "test_service.py").write_text("def test_run_service():\n    assert True\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_pr_context(changed_files=["src/service.py"])

            self.assertIn("## 自动证据包", context)
            self.assertIn("可能相关测试", context)
            self.assertIn("`tests/test_service.py`", context)

    def test_pr_context_skips_unrelated_existing_docs_for_workflow_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            workflows_dir = repo_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True)
            docs_dir = repo_path / "docs"
            docs_dir.mkdir()
            (workflows_dir / "review.yml").write_text(
                "\n".join(
                    [
                        "name: review",
                        "env:",
                        "  LLM_OLLAMA_BASE_URL: ${{ vars.LLM_OLLAMA_BASE_URL }}",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_path / "README.md").write_text(
                "# demo\n这里没有相关配置说明。\n",
                encoding="utf-8",
            )
            (docs_dir / "usage.md").write_text(
                "这是一个通用使用介绍，没有提到 OLLAMA 配置。\n",
                encoding="utf-8",
            )

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_pr_context(
                changed_files=[".github/workflows/review.yml"],
            )

            self.assertIn("文件: .github/workflows/review.yml", context)
            self.assertNotIn("## 可能相关的现有文档/配置", context)

    def test_pr_context_expands_multiple_changed_files_until_size_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            src_dir = repo_path / "src"
            src_dir.mkdir(parents=True)
            for index in range(3):
                (src_dir / f"file_{index}.py").write_text(
                    (
                        f"VALUE_{index} = {index}\n"
                        f"print(VALUE_{index})\n"
                        f"print('line-{index}-extra-context')\n"
                    ),
                    encoding="utf-8",
                )

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=1,
                max_file_chars=120,
                max_total_chars=170,
            )

            context = builder.build_pr_context(
                changed_files=[
                    "src/file_0.py",
                    "src/file_1.py",
                    "src/file_2.py",
                ],
            )

            self.assertIn("文件: src/file_0.py", context)
            self.assertIn("其余", context)
            self.assertIn("完整改动文件列表已在上文给出", context)


if __name__ == "__main__":
    unittest.main()
