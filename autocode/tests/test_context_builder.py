import tempfile
import unittest
from pathlib import Path

from agent.context_builder import LocalContextBuilder


class LocalContextBuilderTests(unittest.TestCase):
    def test_issue_context_includes_repo_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            instructions_dir = repo_path / ".github" / "instructions"
            instructions_dir.mkdir(parents=True)
            (instructions_dir / "python.md").write_text(
                "保持公共导入路径稳定，避免无理由调整模块级导出。",
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
                title="修复启动失败日志问题",
                body="需要先看仓库现有编码规范",
            )

            self.assertIn("## 目标仓库编码规范 (.github/instructions)", context)
            self.assertIn("### python.md", context)
            self.assertIn("保持公共导入路径稳定", context)

    def test_pr_context_includes_repo_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            instructions_dir = repo_path / ".github" / "instructions"
            instructions_dir.mkdir(parents=True)
            (instructions_dir / "review.txt").write_text(
                "修改 review feedback 时优先最小改动，不改变现有 API。",
                encoding="utf-8",
            )
            src_dir = repo_path / "src"
            src_dir.mkdir()
            (src_dir / "service.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=2,
                max_file_chars=500,
                max_total_chars=1200,
            )

            context = builder.build_pr_context(["src/service.py"])

            self.assertIn("## 目标仓库编码规范 (.github/instructions)", context)
            self.assertIn("### review.txt", context)
            self.assertIn("不改变现有 API", context)

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
                body="想看 issue 规划时会不会带上 AGENTS 约束",
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

    def test_issue_context_includes_evidence_pack_for_paths_and_related_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            src_dir = repo_path / "src"
            tests_dir = repo_path / "tests"
            src_dir.mkdir()
            tests_dir.mkdir()
            (src_dir / "service.py").write_text("class TimeoutError(Exception):\n    pass\n", encoding="utf-8")
            (tests_dir / "test_service.py").write_text("def test_timeout():\n    assert True\n", encoding="utf-8")

            builder = LocalContextBuilder(
                repo_path=str(repo_path),
                max_context_files=4,
                max_file_chars=500,
                max_total_chars=2000,
            )

            context = builder.build_issue_context(
                title="src/service.py:12 抛 TimeoutError",
                body="调用 run_service 时超时，需要找到相关测试。",
            )

            self.assertIn("## 自动证据包", context)
            self.assertIn("明确提到的文件", context)
            self.assertIn("`src/service.py`", context)
            self.assertIn("关键错误/标识符线索", context)
            self.assertIn("`TimeoutError`", context)
            self.assertIn("可能相关测试", context)
            self.assertIn("`tests/test_service.py`", context)

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


if __name__ == "__main__":
    unittest.main()
