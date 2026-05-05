"""
配置管理 — 从 .env 加载所有配置项
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent
_LOCAL_ENV_PATH = _BASE_DIR / ".env"
load_dotenv(dotenv_path=_LOCAL_ENV_PATH, override=False)


def _split_command(value: str) -> list[str]:
    return shlex.split(value) if value.strip() else []


def _split_csv(value: str, *, sep: str = ",") -> list[str]:
    return [item.strip() for item in value.split(sep) if item.strip()]


def _split_command_lines(value: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        commands.append(shlex.split(stripped))
    return commands


_DEFAULT_VALIDATION_TRANSIENT_ARTIFACT_DIRS = (
    ".mypy_cache,.pytest_cache,.ruff_cache,.tox,.venv,build,dist,node_modules,venv"
)
_DEFAULT_DOCUMENTATION_SCOPE_PATHS = (
    "docs/,doc/,README.md,CHANGELOG.md,AGENTS.md,CLAUDE.md,SKILL.md,"
    ".github/pull_request_template.md,.github/copilot-instructions.md,"
    ".github/instructions/,.claude/skills/"
)
_DEFAULT_DOCUMENTATION_SCOPE_EXTENSIONS = ".md,.mdx,.rst,.txt"
_DEFAULT_RECOVERABLE_DEAD_LETTER_MARKERS = "node_modules,.venv,临时依赖目录,transient,validation noise"


@dataclass(frozen=True)
class GitHubConfig:
    token: str = field(default_factory=lambda: os.environ["GITHUB_TOKEN"])
    write_token: str = field(default_factory=lambda: os.getenv("GITHUB_WRITE_TOKEN", "").strip())
    webhook_secret: str = field(default_factory=lambda: os.getenv("GITHUB_WEBHOOK_SECRET", ""))
    repo: str = field(default_factory=lambda: os.environ["GITHUB_REPO"])
    notify_login: str = field(
        default_factory=lambda: os.getenv("GITHUB_NOTIFY_LOGIN", "").strip().lstrip("@")
    )

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0]

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[1]

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.repo}.git"


# 已知 CLI 的默认计费模型
_DEFAULT_BILLING: dict[str, str] = {
    "copilot": "call",
    "cursor": "call",
    "codex": "token",
    "opencode": "token",
}

# 已知 CLI 可直接实例化；未知 CLI 需要至少指定 prompt_flag
_BUILTIN_CLIS = {"copilot", "cursor", "codex", "opencode"}


def _load_cli_profile(name: str, *, is_primary: bool = False) -> "CLIProfile":
    """从 CLI_<NAME>_* 环境变量加载单个 CLI profile，兼容旧 PRIMARY_*/FALLBACK_* 格式。"""
    upper = name.upper()

    # --- models: 新格式 CLI_<NAME>_MODELS (CSV) 优先 ---
    models_csv = os.getenv(f"CLI_{upper}_MODELS", "").strip()
    if models_csv:
        all_models = _split_csv(models_csv)
        primary_model = all_models[0] if all_models else ""
        fallback_models = all_models[1:]
    else:
        # 兼容旧格式
        if is_primary:
            primary_model = os.getenv(f"CLI_{upper}_PRIMARY_MODEL",
                                      os.getenv("PRIMARY_MODEL", "")).strip()
            fb_csv = os.getenv(f"CLI_{upper}_FALLBACK_MODELS",
                               os.getenv("PRIMARY_FALLBACK_MODEL", "")).strip()
        else:
            primary_model = os.getenv(f"CLI_{upper}_PRIMARY_MODEL",
                                      os.getenv("FALLBACK_MODEL", "")).strip()
            fb_csv = os.getenv(f"CLI_{upper}_FALLBACK_MODELS",
                               os.getenv("FALLBACK_FALLBACK_MODEL", "")).strip()
        fallback_models = _split_csv(fb_csv, sep=",") if fb_csv else []

    model_flag = os.getenv(
        f"CLI_{upper}_MODEL_FLAG",
        os.getenv("PRIMARY_MODEL_FLAG" if is_primary else "FALLBACK_MODEL_FLAG", ""),
    ).strip()
    command = _split_command(os.getenv(
        f"CLI_{upper}_COMMAND",
        os.getenv("PRIMARY_CLI_COMMAND" if is_primary else "FALLBACK_CLI_COMMAND", ""),
    ))
    billing = os.getenv(f"CLI_{upper}_BILLING", "").strip().lower()
    if billing not in ("call", "token"):
        billing = _DEFAULT_BILLING.get(name.lower(), "token")

    # --- 任务级模型: CLI_<NAME>_TODO_MODEL / CLI_<NAME>_SIMPLE_MODEL ---
    todo_model = os.getenv(f"CLI_{upper}_TODO_MODEL", "").strip()
    simple_model = os.getenv(f"CLI_{upper}_SIMPLE_MODEL", "").strip()

    return CLIProfile(
        name=name.lower(),
        primary_model=primary_model,
        fallback_models=fallback_models,
        model_flag=model_flag,
        command=command,
        billing=billing,
        todo_model=todo_model,
        simple_model=simple_model,
    )


@dataclass(frozen=True)
class CLIProfile:
    """单个 CLI 工具的完整配置。"""
    name: str
    primary_model: str = ""
    fallback_models: list[str] = field(default_factory=list)
    model_flag: str = ""
    command: list[str] = field(default_factory=list)
    billing: str = "token"  # "token" (按量) | "call" (按次)
    todo_model: str = ""     # todo_refactor / 轻量任务优先使用的模型
    simple_model: str = ""   # 简单 PR 反馈优先使用的模型


@dataclass(frozen=True)
class CLIConfig:
    planner_cli_timeout: int = field(
        default_factory=lambda: int(os.getenv("PLANNER_CLI_TIMEOUT", os.getenv("CLI_TIMEOUT", "180")))
    )
    executor_cli_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXECUTOR_CLI_TIMEOUT", os.getenv("CLI_TIMEOUT", "180")))
    )
    executor_max_idle_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTOR_MAX_IDLE_SECONDS", "0"))
    )
    todo_refactor_executor_cli_timeout: int = field(
        default_factory=lambda: int(os.getenv("TODO_REFACTOR_EXECUTOR_CLI_TIMEOUT", "0"))
    )
    todo_refactor_executor_model: str = field(
        default_factory=lambda: os.getenv("TODO_REFACTOR_EXECUTOR_MODEL", "").strip()
    )
    todo_refactor_executor_max_idle_seconds: int = field(
        default_factory=lambda: int(os.getenv("TODO_REFACTOR_EXECUTOR_MAX_IDLE_SECONDS", "0"))
    )
    simple_executor_model: str = field(
        default_factory=lambda: os.getenv("SIMPLE_EXECUTOR_MODEL", "").strip()
    )
    simple_executor_max_feedback_items: int = field(
        default_factory=lambda: int(os.getenv("SIMPLE_EXECUTOR_MAX_FEEDBACK_ITEMS", "1"))
    )
    # --- 新格式：primary + fallback 链 ---
    primary_cli: str = field(default_factory=lambda: os.getenv("PRIMARY_CLI", "copilot").strip().lower())
    fallback_clis: list[str] = field(default_factory=lambda: _split_csv(
        os.getenv("FALLBACK_CLIS", os.getenv("FALLBACK_CLI", ""))
    ))
    cli_profiles: dict[str, CLIProfile] = field(default_factory=dict)

    # --- 兼容属性：旧代码通过 config.cli.primary_model 等访问仍然可用 ---
    @property
    def primary_profile(self) -> CLIProfile:
        return self.cli_profiles.get(self.primary_cli, CLIProfile(name=self.primary_cli))

    @property
    def primary_model(self) -> str:
        return self.primary_profile.primary_model

    @property
    def primary_fallback_model(self) -> str:
        models = self.primary_profile.fallback_models
        return models[0] if models else ""

    @property
    def primary_model_flag(self) -> str:
        return self.primary_profile.model_flag

    @property
    def primary_command(self) -> list[str]:
        return self.primary_profile.command

    @property
    def fallback_cli(self) -> str:
        return self.fallback_clis[0] if self.fallback_clis else ""

    @property
    def fallback_model(self) -> str:
        if not self.fallback_clis:
            return ""
        profile = self.cli_profiles.get(self.fallback_clis[0], CLIProfile(name=self.fallback_clis[0]))
        return profile.primary_model

    @property
    def fallback_fallback_model(self) -> str:
        if not self.fallback_clis:
            return ""
        profile = self.cli_profiles.get(self.fallback_clis[0], CLIProfile(name=self.fallback_clis[0]))
        return profile.fallback_models[0] if profile.fallback_models else ""

    @property
    def fallback_model_flag(self) -> str:
        if not self.fallback_clis:
            return ""
        profile = self.cli_profiles.get(self.fallback_clis[0], CLIProfile(name=self.fallback_clis[0]))
        return profile.model_flag

    @property
    def fallback_command(self) -> list[str]:
        if not self.fallback_clis:
            return []
        profile = self.cli_profiles.get(self.fallback_clis[0], CLIProfile(name=self.fallback_clis[0]))
        return profile.command


@dataclass(frozen=True)
class ServerConfig:
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8001")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true")
    event_source: str = field(default_factory=lambda: os.getenv("EVENT_SOURCE", "").strip().lower())
    enable_webhook: bool = field(default_factory=lambda: os.getenv("ENABLE_WEBHOOK", "false").lower() == "true")
    poll_interval_seconds: int = field(default_factory=lambda: int(os.getenv("POLL_INTERVAL_SECONDS", "60")))
    poll_items_per_page: int = field(default_factory=lambda: int(os.getenv("POLL_ITEMS_PER_PAGE", "30")))
    state_file_path: str = field(default_factory=lambda: os.getenv("STATE_FILE_PATH", ".autocode-service-state.json"))
    max_comment_length: int = field(default_factory=lambda: int(os.getenv("MAX_COMMENT_LENGTH", "4000")))


@dataclass(frozen=True)
class LocalRepoConfig:
    path: str = field(default_factory=lambda: os.getenv("LOCAL_REPO_PATH", os.getcwd()))
    default_branch: str = field(default_factory=lambda: os.getenv("LOCAL_REPO_DEFAULT_BRANCH", "main"))
    max_context_files: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_CONTEXT_FILES", "6")))
    max_file_chars: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_FILE_CHARS", "1800")))
    max_total_context_chars: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_TOTAL_CONTEXT_CHARS", "12000")))


@dataclass(frozen=True)
class AutoCodeConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_EXECUTION", "false").lower() == "true")
    mode: str = field(default_factory=lambda: os.getenv("EXECUTION_MODE", "manual").strip().lower())
    state_file_path: str = field(
        default_factory=lambda: os.getenv("EXECUTION_STATE_FILE_PATH", ".autocode-state.json")
    )
    auto_plan_on_issue_open: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_AUTO_PLAN_ON_ISSUE_OPEN", "true").lower() == "true"
    )
    auto_implement_on_issue_open: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_AUTO_IMPLEMENT_ON_ISSUE_OPEN", "false").lower() == "true"
    )
    poll_review_feedback: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_POLL_REVIEW_FEEDBACK", "true").lower() == "true"
    )
    review_feedback_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_REVIEW_FEEDBACK_INTERVAL_SECONDS", "300"))
    )
    plan_comment_header: str = field(
        default_factory=lambda: os.getenv("EXECUTION_PLAN_COMMENT_HEADER", "## AutoCode Plan")
    )
    branch_prefix: str = field(default_factory=lambda: os.getenv("EXECUTION_BRANCH_PREFIX", "autocode"))
    publish_remote: str = field(default_factory=lambda: os.getenv("EXECUTION_PUSH_REMOTE", "origin"))
    pr_base_branch: str = field(default_factory=lambda: os.getenv("EXECUTION_PR_BASE_BRANCH", "").strip())
    force_draft_pr: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_FORCE_DRAFT_PR", "false").lower() == "true"
    )
    auto_ready_for_review: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_AUTO_READY_FOR_REVIEW", "false").lower() == "true"
    )
    forbid_main_base_branch: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_FORBID_MAIN_BASE_BRANCH", "false").lower() == "true"
    )
    allow_feature_on_comment: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ALLOW_FEATURE_ON_COMMENT", "true").lower() == "true"
    )
    allowed_issue_labels: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_ALLOWED_LABELS", ""))
    )
    require_approval_for: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("EXECUTION_REQUIRE_APPROVAL_FOR", "")
        )
    )
    allow_high_risk_autorun: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ALLOW_HIGH_RISK_AUTORUN", "false").lower() == "true"
    )
    blocked_paths: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                "EXECUTION_BLOCKED_PATHS",
                ".github/workflows/,infra/,deploy/,migrations/,secrets/,.env",
            )
        )
    )
    max_changed_files: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_CHANGED_FILES", "8"))
    )
    max_added_lines: int = field(default_factory=lambda: int(os.getenv("EXECUTION_MAX_ADDED_LINES", "400")))
    max_deleted_lines: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_DELETED_LINES", "200"))
    )
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "1200")))
    test_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_TEST_TIMEOUT_SECONDS", "0"))
    )
    repair_test_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_REPAIR_TEST_TIMEOUT_SECONDS", "600"))
    )
    issue_repair_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_ISSUE_REPAIR_MAX_ATTEMPTS", "2"))
    )
    review_feedback_repair_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_REVIEW_FEEDBACK_REPAIR_MAX_ATTEMPTS", "2"))
    )
    bot_feedback_max_repair_rounds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_BOT_FEEDBACK_MAX_REPAIR_ROUNDS", "4"))
    )
    validation_fail_fast: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_VALIDATION_FAIL_FAST", "false").lower() == "true"
    )
    validation_transient_artifact_dirs: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                "EXECUTION_VALIDATION_TRANSIENT_ARTIFACT_DIRS",
                _DEFAULT_VALIDATION_TRANSIENT_ARTIFACT_DIRS,
            )
        )
    )
    documentation_scope_paths: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("EXECUTION_DOCUMENTATION_SCOPE_PATHS", _DEFAULT_DOCUMENTATION_SCOPE_PATHS)
        )
    )
    documentation_scope_extensions: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("EXECUTION_DOCUMENTATION_SCOPE_EXTENSIONS", _DEFAULT_DOCUMENTATION_SCOPE_EXTENSIONS)
        )
    )
    recoverable_dead_letter_markers: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("EXECUTION_RECOVERABLE_DEAD_LETTER_MARKERS", _DEFAULT_RECOVERABLE_DEAD_LETTER_MARKERS)
        )
    )
    dead_letter_auto_requeue_limit: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_DEAD_LETTER_AUTO_REQUEUE_LIMIT", "5"))
    )
    worktree_root: str = field(default_factory=lambda: os.getenv("EXECUTION_WORKTREE_ROOT", "/tmp/autocode-runs"))
    commit_user_name: str = field(
        default_factory=lambda: os.getenv("EXECUTION_COMMIT_USER_NAME", "AutoCode Bot")
    )
    commit_user_email: str = field(
        default_factory=lambda: os.getenv("EXECUTION_COMMIT_USER_EMAIL", "autocode-bot@example.com")
    )
    test_commands: list[list[str]] = field(
        default_factory=lambda: _split_command_lines(os.getenv("EXECUTION_TEST_COMMANDS", ""))
    )
    lint_commands: list[list[str]] = field(
        default_factory=lambda: _split_command_lines(os.getenv("EXECUTION_LINT_COMMANDS", ""))
    )
    format_commands: list[list[str]] = field(
        default_factory=lambda: _split_command_lines(os.getenv("EXECUTION_FORMAT_COMMANDS", ""))
    )
    enable_pr_queue: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ENABLE_PR_QUEUE", "true").lower() == "true"
    )
    tracked_pr_authors: list[str] = field(
        default_factory=lambda: [item.lower() for item in _split_csv(os.getenv("EXECUTION_TRACK_PR_AUTHORS", ""))]
    )
    pr_queue_max_batch: int = field(default_factory=lambda: int(os.getenv("EXECUTION_PR_QUEUE_MAX_BATCH", "1")))
    enable_backlog_source: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ENABLE_BACKLOG_SOURCE", "false").lower() == "true"
    )
    backlog_repo_path: str = field(
        default_factory=lambda: os.getenv(
            "EXECUTION_BACKLOG_REPO_PATH",
            os.getenv("LOCAL_REPO_PATH", os.getcwd()),
        )
    )
    backlog_file_paths: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_BACKLOG_FILE_PATHS", ""))
    )
    backlog_labels: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_BACKLOG_LABELS", "autocode"))
    )
    backlog_title_prefix: str = field(
        default_factory=lambda: os.getenv("EXECUTION_BACKLOG_TITLE_PREFIX", "")
    )
    backlog_max_items_per_cycle: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_BACKLOG_MAX_ITEMS_PER_CYCLE", "10"))
    )
    max_open_autocode_issues: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_OPEN_AUTOCODE_ISSUES", "20"))
    )
    enable_plan_source: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ENABLE_PLAN_SOURCE", "false").lower() == "true"
    )
    plan_file_paths: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_PLAN_FILE_PATHS", ""))
    )
    plan_labels: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_PLAN_LABELS", "autocode"))
    )
    plan_title_prefix: str = field(
        default_factory=lambda: os.getenv("EXECUTION_PLAN_TITLE_PREFIX", "")
    )
    plan_max_items_per_cycle: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_PLAN_MAX_ITEMS_PER_CYCLE", "10"))
    )
    enable_idle_scanner: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_ENABLE_IDLE_SCANNER", "false").lower() == "true"
    )
    idle_scanner_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_IDLE_SCANNER_INTERVAL_SECONDS", "1800"))
    )
    idle_scanner_enable_heuristic_analysis: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_IDLE_ENABLE_HEURISTIC_ANALYSIS", "true").lower() == "true"
    )
    idle_scanner_enable_llm_enrichment: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_IDLE_ENABLE_LLM_ENRICHMENT", "true").lower() == "true"
    )
    idle_scanner_advanced_only_when_todo_clear: bool = field(
        default_factory=lambda: os.getenv("EXECUTION_IDLE_ADVANCED_ONLY_WHEN_TODO_CLEAR", "true").lower() == "true"
    )
    idle_scanner_long_function_lines: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_IDLE_LONG_FUNCTION_LINES", "80"))
    )
    idle_scanner_llm_max_context_files: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_IDLE_LLM_MAX_CONTEXT_FILES", "4"))
    )
    idle_scanner_labels: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EXECUTION_IDLE_SCANNER_LABELS", "autocode"))
    )
    idle_scanner_title_prefix: str = field(
        default_factory=lambda: os.getenv("EXECUTION_IDLE_SCANNER_TITLE_PREFIX", "")
    )
    idle_scanner_max_items_per_cycle: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_IDLE_SCANNER_MAX_ITEMS_PER_CYCLE", "5"))
    )
    issue_task_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_ISSUE_TASK_INTERVAL_SECONDS", "5"))
    )
    issue_task_max_batch: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_ISSUE_TASK_MAX_BATCH", "1"))
    )
    issue_task_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_ISSUE_TASK_MAX_ATTEMPTS", "3"))
    )
    pr_task_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_PR_TASK_INTERVAL_SECONDS", "5"))
    )
    pr_task_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_PR_TASK_MAX_ATTEMPTS", "3"))
    )
    pr_max_repair_rounds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_PR_MAX_REPAIR_ROUNDS", "8"))
    )
    max_open_autocode_prs: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_OPEN_AUTOCODE_PRS", "20"))
    )
    source_task_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_SOURCE_TASK_INTERVAL_SECONDS", "5"))
    )
    source_task_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_SOURCE_TASK_MAX_ATTEMPTS", "3"))
    )


@dataclass(frozen=True)
class AppConfig:
    github: GitHubConfig = field(default_factory=GitHubConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    local_repo: LocalRepoConfig = field(default_factory=LocalRepoConfig)
    autocode: AutoCodeConfig = field(default_factory=AutoCodeConfig)


def load_config() -> AppConfig:
    """加载并验证配置"""
    config = AppConfig()
    base_dir = _BASE_DIR
    resolved_event_source = config.server.event_source or ("webhook" if config.server.enable_webhook else "polling")
    resolved_enable_webhook = config.server.enable_webhook or resolved_event_source == "webhook"
    object.__setattr__(config.server, "event_source", resolved_event_source)
    object.__setattr__(config.server, "enable_webhook", resolved_enable_webhook)
    state_file_path = Path(config.server.state_file_path).expanduser()
    if not state_file_path.is_absolute():
        state_file_path = (base_dir / state_file_path).resolve()
    object.__setattr__(config.server, "state_file_path", str(state_file_path))
    local_repo_path = Path(config.local_repo.path).expanduser()
    if not local_repo_path.is_absolute():
        local_repo_path = (base_dir / local_repo_path).resolve()
    object.__setattr__(config.local_repo, "path", str(local_repo_path))
    backlog_repo_path_raw = str(config.autocode.backlog_repo_path or "").strip()
    if backlog_repo_path_raw:
        backlog_repo_path = Path(backlog_repo_path_raw).expanduser()
        if not backlog_repo_path.is_absolute():
            backlog_repo_path = (base_dir / backlog_repo_path).resolve()
    else:
        backlog_repo_path = local_repo_path
    object.__setattr__(config.autocode, "backlog_repo_path", str(backlog_repo_path))
    worktree_root = Path(config.autocode.worktree_root).expanduser()
    if not worktree_root.is_absolute():
        worktree_root = (base_dir / worktree_root).resolve()
    object.__setattr__(config.autocode, "worktree_root", str(worktree_root))
    autocode_state_file_path = Path(config.autocode.state_file_path).expanduser()
    if not autocode_state_file_path.is_absolute():
        autocode_state_file_path = (base_dir / autocode_state_file_path).resolve()
    object.__setattr__(config.autocode, "state_file_path", str(autocode_state_file_path))
    if not config.autocode.pr_base_branch:
        object.__setattr__(config.autocode, "pr_base_branch", config.local_repo.default_branch)
    # --- 构建 CLI profiles ---
    all_cli_names = [config.cli.primary_cli] + list(config.cli.fallback_clis)
    cli_profiles: dict[str, CLIProfile] = {}
    for idx, cli_name in enumerate(all_cli_names):
        normalized = cli_name.strip().lower()
        if not normalized:
            continue
        if normalized not in cli_profiles:
            cli_profiles[normalized] = _load_cli_profile(normalized, is_primary=(idx == 0))
    object.__setattr__(config.cli, "cli_profiles", cli_profiles)
    # 验证必填项
    assert config.github.token, "GITHUB_TOKEN 未设置"
    assert config.github.repo, "GITHUB_REPO 未设置"
    assert config.server.event_source in {"webhook", "polling"}, "EVENT_SOURCE 仅支持 webhook 或 polling"
    assert config.server.poll_interval_seconds > 0, "POLL_INTERVAL_SECONDS 必须大于 0"
    assert 1 <= config.server.poll_items_per_page <= 100, "POLL_ITEMS_PER_PAGE 必须在 1-100 之间"
    if config.server.enable_webhook:
        assert config.github.webhook_secret, "ENABLE_WEBHOOK=true 时必须设置 GITHUB_WEBHOOK_SECRET"
    # 已知 CLI 可直接使用；未知 CLI 需要在环境变量中配置 prompt_flag
    assert config.cli.primary_cli, "PRIMARY_CLI 未设置"
    for cli_name in all_cli_names:
        normalized = cli_name.strip().lower()
        if not normalized:
            continue
        if normalized not in _BUILTIN_CLIS:
            # 未知 CLI 仍可使用，仅发出日志提示；运行时适配器工厂会处理
            import logging as _logging
            _logging.getLogger(__name__).info("自定义 CLI '%s' 未内置，将使用通用适配器", normalized)
    assert config.cli.planner_cli_timeout > 0, "PLANNER_CLI_TIMEOUT 必须大于 0"
    assert config.cli.executor_cli_timeout > 0, "EXECUTOR_CLI_TIMEOUT 必须大于 0"
    executor_idle_seconds = int(getattr(config.cli, "executor_max_idle_seconds", 0) or 0)
    if executor_idle_seconds <= 0:
        executor_idle_seconds = min(config.cli.executor_cli_timeout, 300)
    else:
        executor_idle_seconds = min(executor_idle_seconds, config.cli.executor_cli_timeout)
    object.__setattr__(config.cli, "executor_max_idle_seconds", executor_idle_seconds)
    assert config.cli.executor_max_idle_seconds > 0, "EXECUTOR_MAX_IDLE_SECONDS 必须大于 0"
    todo_refactor_timeout = int(getattr(config.cli, "todo_refactor_executor_cli_timeout", 0) or 0)
    if todo_refactor_timeout <= 0:
        todo_refactor_timeout = min(config.cli.executor_cli_timeout, 900)
    object.__setattr__(config.cli, "todo_refactor_executor_cli_timeout", todo_refactor_timeout)
    # todo_refactor model 优先级: 全局 > primary profile todo_model > primary fallback_model > primary_model
    todo_refactor_model = str(getattr(config.cli, "todo_refactor_executor_model", "") or "").strip()
    if not todo_refactor_model:
        primary_profile = cli_profiles.get(config.cli.primary_cli)
        todo_refactor_model = (
            (primary_profile.todo_model if primary_profile else "")
            or config.cli.primary_fallback_model
            or config.cli.primary_model
        )
    object.__setattr__(config.cli, "todo_refactor_executor_model", todo_refactor_model)
    todo_refactor_idle_seconds = int(getattr(config.cli, "todo_refactor_executor_max_idle_seconds", 0) or 0)
    if todo_refactor_idle_seconds <= 0:
        todo_refactor_idle_seconds = min(todo_refactor_timeout, 180)
    else:
        todo_refactor_idle_seconds = min(todo_refactor_idle_seconds, todo_refactor_timeout)
    object.__setattr__(config.cli, "todo_refactor_executor_max_idle_seconds", todo_refactor_idle_seconds)
    assert config.cli.todo_refactor_executor_cli_timeout > 0, "TODO_REFACTOR_EXECUTOR_CLI_TIMEOUT 必须大于 0"
    assert config.cli.todo_refactor_executor_max_idle_seconds > 0, "TODO_REFACTOR_EXECUTOR_MAX_IDLE_SECONDS 必须大于 0"
    assert config.autocode.mode in {"manual", "semi_auto", "auto"}, "EXECUTION_MODE 仅支持 manual/semi_auto/auto"
    assert config.autocode.review_feedback_interval_seconds > 0, "EXECUTION_REVIEW_FEEDBACK_INTERVAL_SECONDS 必须大于 0"
    assert config.autocode.max_changed_files > 0, "EXECUTION_MAX_CHANGED_FILES 必须大于 0"
    assert config.autocode.max_added_lines >= 0, "EXECUTION_MAX_ADDED_LINES 不能小于 0"
    assert config.autocode.max_deleted_lines >= 0, "EXECUTION_MAX_DELETED_LINES 不能小于 0"
    assert config.autocode.timeout_seconds > 0, "EXECUTION_TIMEOUT_SECONDS 必须大于 0"
    assert config.autocode.pr_queue_max_batch > 0, "EXECUTION_PR_QUEUE_MAX_BATCH 必须大于 0"
    assert config.autocode.backlog_max_items_per_cycle > 0, "EXECUTION_BACKLOG_MAX_ITEMS_PER_CYCLE 必须大于 0"
    assert config.autocode.plan_max_items_per_cycle > 0, "EXECUTION_PLAN_MAX_ITEMS_PER_CYCLE 必须大于 0"
    assert config.autocode.idle_scanner_interval_seconds > 0, "EXECUTION_IDLE_SCANNER_INTERVAL_SECONDS 必须大于 0"
    assert config.autocode.idle_scanner_max_items_per_cycle > 0, "EXECUTION_IDLE_SCANNER_MAX_ITEMS_PER_CYCLE 必须大于 0"
    assert config.autocode.idle_scanner_long_function_lines > 0, "EXECUTION_IDLE_LONG_FUNCTION_LINES 必须大于 0"
    assert config.autocode.idle_scanner_llm_max_context_files > 0, "EXECUTION_IDLE_LLM_MAX_CONTEXT_FILES 必须大于 0"
    assert config.autocode.issue_task_interval_seconds > 0, "EXECUTION_ISSUE_TASK_INTERVAL_SECONDS 必须大于 0"
    assert config.autocode.issue_task_max_batch > 0, "EXECUTION_ISSUE_TASK_MAX_BATCH 必须大于 0"
    assert config.autocode.issue_task_max_attempts > 0, "EXECUTION_ISSUE_TASK_MAX_ATTEMPTS 必须大于 0"
    assert config.autocode.pr_task_interval_seconds > 0, "EXECUTION_PR_TASK_INTERVAL_SECONDS 必须大于 0"
    assert config.autocode.pr_task_max_attempts > 0, "EXECUTION_PR_TASK_MAX_ATTEMPTS 必须大于 0"
    assert config.autocode.pr_max_repair_rounds >= 0, "EXECUTION_PR_MAX_REPAIR_ROUNDS 不能小于 0"
    assert config.autocode.bot_feedback_max_repair_rounds >= 0, "EXECUTION_BOT_FEEDBACK_MAX_REPAIR_ROUNDS 不能小于 0"
    assert config.autocode.max_open_autocode_prs >= 0, "EXECUTION_MAX_OPEN_AUTOCODE_PRS 不能小于 0"
    assert config.autocode.dead_letter_auto_requeue_limit >= 0, "EXECUTION_DEAD_LETTER_AUTO_REQUEUE_LIMIT 不能小于 0"
    assert config.autocode.source_task_interval_seconds > 0, "EXECUTION_SOURCE_TASK_INTERVAL_SECONDS 必须大于 0"
    assert config.autocode.source_task_max_attempts > 0, "EXECUTION_SOURCE_TASK_MAX_ATTEMPTS 必须大于 0"
    return config


def load_autocode_config() -> AppConfig:
    """加载 AutoCode 入口配置，并验证关键安全约束。"""
    config = load_config()
    assert config.autocode.enabled, "ENABLE_EXECUTION=false，AutoCode 入口未启用"
    if not config.server.dry_run:
        assert config.github.write_token, "AutoCode 非 dry_run 时必须设置 GITHUB_WRITE_TOKEN"
    return config
