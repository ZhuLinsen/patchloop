import os
import unittest
from pathlib import Path
from unittest import mock

import config


class ConfigTests(unittest.TestCase):
    def _base_env(self) -> dict[str, str]:
        return {
            "GITHUB_TOKEN": "test-token",
            "GITHUB_REPO": "example/repo",
            "ENABLE_WEBHOOK": "false",
            "EVENT_SOURCE": "polling",
            "PRIMARY_CLI": "codex",
            "LOCAL_REPO_PATH": "../target-repo",
            "EXECUTION_WORKTREE_ROOT": ".tmp-worktrees",
        }

    def test_backlog_repo_path_defaults_to_local_repo_path(self):
        with mock.patch.dict(os.environ, self._base_env(), clear=True):
            loaded = config.load_config()

        self.assertEqual(loaded.local_repo.path, loaded.autocode.backlog_repo_path)

    def test_backlog_repo_path_resolves_relative_to_config_dir(self):
        env = self._base_env()
        env["EXECUTION_BACKLOG_REPO_PATH"] = "."

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        expected = str(Path(config.__file__).resolve().parent)
        self.assertEqual(expected, loaded.autocode.backlog_repo_path)

    def test_blank_backlog_repo_path_falls_back_to_local_repo_path(self):
        env = self._base_env()
        env["EXECUTION_BACKLOG_REPO_PATH"] = "   "

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(loaded.local_repo.path, loaded.autocode.backlog_repo_path)

    def test_allow_high_risk_autorun_reads_from_env(self):
        env = self._base_env()
        env["EXECUTION_ALLOW_HIGH_RISK_AUTORUN"] = "true"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertTrue(loaded.autocode.allow_high_risk_autorun)

    def test_cli_timeout_defaults_both_planner_and_executor(self):
        env = self._base_env()
        env["CLI_TIMEOUT"] = "480"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(480, loaded.cli.planner_cli_timeout)
        self.assertEqual(480, loaded.cli.executor_cli_timeout)

    def test_planner_and_executor_timeout_can_be_overridden_separately(self):
        env = self._base_env()
        env["CLI_TIMEOUT"] = "480"
        env["PLANNER_CLI_TIMEOUT"] = "300"
        env["EXECUTOR_CLI_TIMEOUT"] = "1500"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(300, loaded.cli.planner_cli_timeout)
        self.assertEqual(1500, loaded.cli.executor_cli_timeout)

    def test_todo_refactor_executor_profile_defaults_from_executor_settings(self):
        env = self._base_env()
        env["EXECUTOR_CLI_TIMEOUT"] = "1500"
        env["PRIMARY_MODEL"] = "gpt-5.4"
        env["PRIMARY_FALLBACK_MODEL"] = "gpt-5.3-codex"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(900, loaded.cli.todo_refactor_executor_cli_timeout)
        self.assertEqual("gpt-5.3-codex", loaded.cli.todo_refactor_executor_model)
        self.assertEqual(180, loaded.cli.todo_refactor_executor_max_idle_seconds)

    def test_todo_refactor_executor_profile_can_be_overridden(self):
        env = self._base_env()
        env["EXECUTOR_CLI_TIMEOUT"] = "1500"
        env["TODO_REFACTOR_EXECUTOR_CLI_TIMEOUT"] = "720"
        env["TODO_REFACTOR_EXECUTOR_MODEL"] = "gpt-5.2"
        env["TODO_REFACTOR_EXECUTOR_MAX_IDLE_SECONDS"] = "240"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(720, loaded.cli.todo_refactor_executor_cli_timeout)
        self.assertEqual("gpt-5.2", loaded.cli.todo_refactor_executor_model)
        self.assertEqual(240, loaded.cli.todo_refactor_executor_max_idle_seconds)

    def test_simple_executor_profile_reads_from_env(self):
        env = self._base_env()
        env["SIMPLE_EXECUTOR_MODEL"] = "gpt-5.3-codex-spark"
        env["SIMPLE_EXECUTOR_MAX_FEEDBACK_ITEMS"] = "2"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual("gpt-5.3-codex-spark", loaded.cli.simple_executor_model)
        self.assertEqual(2, loaded.cli.simple_executor_max_feedback_items)

    def test_system_scope_and_recovery_policy_read_from_env(self):
        env = self._base_env()
        env["EXECUTION_VALIDATION_TRANSIENT_ARTIFACT_DIRS"] = "node_modules,generated"
        env["EXECUTION_DOCUMENTATION_SCOPE_EXTENSIONS"] = ".md,.adoc"
        env["EXECUTION_RECOVERABLE_DEAD_LETTER_MARKERS"] = "越界改动,validation noise"
        env["EXECUTION_DEAD_LETTER_AUTO_REQUEUE_LIMIT"] = "3"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(["node_modules", "generated"], loaded.autocode.validation_transient_artifact_dirs)
        self.assertEqual([".md", ".adoc"], loaded.autocode.documentation_scope_extensions)
        self.assertEqual(["越界改动", "validation noise"], loaded.autocode.recoverable_dead_letter_markers)
        self.assertEqual(3, loaded.autocode.dead_letter_auto_requeue_limit)

    def test_bot_feedback_max_repair_rounds_reads_from_env(self):
        env = self._base_env()
        env["EXECUTION_BOT_FEEDBACK_MAX_REPAIR_ROUNDS"] = "6"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(6, loaded.autocode.bot_feedback_max_repair_rounds)

    # --- 新 CLI profile 配置格式测试 ---

    def test_cli_profiles_built_from_new_format(self):
        env = self._base_env()
        env["CLI_CODEX_MODELS"] = "gpt-5.4,gpt-5.3-codex-spark"
        env["CLI_CODEX_BILLING"] = "token"
        env["FALLBACK_CLIS"] = "copilot"
        env["CLI_COPILOT_MODELS"] = "gpt-5.3-codex,gpt-4.1"
        env["CLI_COPILOT_BILLING"] = "call"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertIn("codex", loaded.cli.cli_profiles)
        self.assertIn("copilot", loaded.cli.cli_profiles)
        codex_profile = loaded.cli.cli_profiles["codex"]
        self.assertEqual("gpt-5.4", codex_profile.primary_model)
        self.assertEqual(["gpt-5.3-codex-spark"], codex_profile.fallback_models)
        self.assertEqual("token", codex_profile.billing)
        copilot_profile = loaded.cli.cli_profiles["copilot"]
        self.assertEqual("gpt-5.3-codex", copilot_profile.primary_model)
        self.assertEqual(["gpt-4.1"], copilot_profile.fallback_models)
        self.assertEqual("call", copilot_profile.billing)

    def test_cli_profiles_backward_compat_with_old_env_vars(self):
        env = self._base_env()
        env["PRIMARY_MODEL"] = "gpt-5.4"
        env["PRIMARY_FALLBACK_MODEL"] = "gpt-5.3-codex"
        env["FALLBACK_CLI"] = "copilot"
        env["FALLBACK_MODEL"] = "gpt-4.1"
        env["FALLBACK_FALLBACK_MODEL"] = "gpt-3.5"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        codex_profile = loaded.cli.cli_profiles["codex"]
        self.assertEqual("gpt-5.4", codex_profile.primary_model)
        self.assertEqual(["gpt-5.3-codex"], codex_profile.fallback_models)
        copilot_profile = loaded.cli.cli_profiles["copilot"]
        self.assertEqual("gpt-4.1", copilot_profile.primary_model)
        self.assertEqual(["gpt-3.5"], copilot_profile.fallback_models)

    def test_fallback_clis_supports_multiple_cli_names(self):
        env = self._base_env()
        env["FALLBACK_CLIS"] = "copilot,opencode"
        env["CLI_COPILOT_MODELS"] = "gpt-5.3-codex"
        env["CLI_OPENCODE_MODELS"] = "claude-4"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(["copilot", "opencode"], loaded.cli.fallback_clis)
        self.assertIn("copilot", loaded.cli.cli_profiles)
        self.assertIn("opencode", loaded.cli.cli_profiles)

    def test_compat_properties_read_from_profiles(self):
        env = self._base_env()
        env["CLI_CODEX_MODELS"] = "gpt-5.4,gpt-5.3-codex-spark"
        env["FALLBACK_CLIS"] = "copilot"
        env["CLI_COPILOT_MODELS"] = "gpt-5.3-codex,gpt-4.1"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual("gpt-5.4", loaded.cli.primary_model)
        self.assertEqual("gpt-5.3-codex-spark", loaded.cli.primary_fallback_model)
        self.assertEqual("copilot", loaded.cli.fallback_cli)
        self.assertEqual("gpt-5.3-codex", loaded.cli.fallback_model)
        self.assertEqual("gpt-4.1", loaded.cli.fallback_fallback_model)

    def test_cli_billing_defaults_by_cli_name(self):
        env = self._base_env()

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        codex_profile = loaded.cli.cli_profiles["codex"]
        self.assertEqual("token", codex_profile.billing)

    def test_custom_cli_accepted_with_info_log(self):
        env = self._base_env()
        env["PRIMARY_CLI"] = "codex"
        env["FALLBACK_CLIS"] = "kiro"
        env["CLI_KIRO_MODELS"] = "claude-4,claude-3.5"
        env["CLI_KIRO_BILLING"] = "token"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertIn("kiro", loaded.cli.cli_profiles)
        kiro_profile = loaded.cli.cli_profiles["kiro"]
        self.assertEqual("claude-4", kiro_profile.primary_model)
        self.assertEqual(["claude-3.5"], kiro_profile.fallback_models)

    def test_per_cli_todo_and_simple_model(self):
        env = self._base_env()
        env["PRIMARY_CLI"] = "copilot"
        env["FALLBACK_CLIS"] = "codex"
        env["CLI_COPILOT_MODELS"] = "claude-ops-4.6,gpt-5.4"
        env["CLI_COPILOT_TODO_MODEL"] = "gpt-5.3-codex"
        env["CLI_COPILOT_SIMPLE_MODEL"] = "gpt-5.3-codex"
        env["CLI_CODEX_MODELS"] = "gpt-5.4,gpt-5.3-codex-spark"
        env["CLI_CODEX_TODO_MODEL"] = "gpt-5.3-codex-spark"
        env["CLI_CODEX_SIMPLE_MODEL"] = "gpt-5.3-codex-spark"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        copilot_profile = loaded.cli.cli_profiles["copilot"]
        self.assertEqual("gpt-5.3-codex", copilot_profile.todo_model)
        self.assertEqual("gpt-5.3-codex", copilot_profile.simple_model)
        codex_profile = loaded.cli.cli_profiles["codex"]
        self.assertEqual("gpt-5.3-codex-spark", codex_profile.todo_model)
        self.assertEqual("gpt-5.3-codex-spark", codex_profile.simple_model)
        # 全局 todo_refactor_model 应自动从 primary CLI profile 解析
        self.assertEqual("gpt-5.3-codex", loaded.cli.todo_refactor_executor_model)

    def test_global_todo_model_overrides_profile(self):
        env = self._base_env()
        env["CLI_CODEX_MODELS"] = "gpt-5.4"
        env["CLI_CODEX_TODO_MODEL"] = "gpt-5.3-codex-spark"
        env["TODO_REFACTOR_EXECUTOR_MODEL"] = "gpt-4.1"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        # 全局显式指定时覆盖 profile 配置
        self.assertEqual("gpt-4.1", loaded.cli.todo_refactor_executor_model)


if __name__ == "__main__":
    unittest.main()
