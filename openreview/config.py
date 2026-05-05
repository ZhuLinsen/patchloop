"""
配置管理 — 从 .env 加载所有配置项
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _split_command(value: str) -> list[str]:
    return value.split() if value.strip() else []


@dataclass(frozen=True)
class GitHubConfig:
    token: str = field(default_factory=lambda: os.environ["GITHUB_TOKEN"])
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


@dataclass(frozen=True)
class CLIConfig:
    cli_timeout: int = field(default_factory=lambda: int(os.getenv("CLI_TIMEOUT", "300")))
    primary_cli: str = field(default_factory=lambda: os.getenv("PRIMARY_CLI", "copilot"))
    primary_model: str = field(default_factory=lambda: os.getenv("PRIMARY_MODEL", ""))
    primary_fallback_model: str = field(default_factory=lambda: os.getenv("PRIMARY_FALLBACK_MODEL", ""))
    primary_model_flag: str = field(default_factory=lambda: os.getenv("PRIMARY_MODEL_FLAG", ""))
    primary_command: list[str] = field(default_factory=lambda: _split_command(os.getenv("PRIMARY_CLI_COMMAND", "")))
    fallback_cli: str = field(default_factory=lambda: os.getenv("FALLBACK_CLI", ""))
    fallback_model: str = field(default_factory=lambda: os.getenv("FALLBACK_MODEL", ""))
    fallback_fallback_model: str = field(default_factory=lambda: os.getenv("FALLBACK_FALLBACK_MODEL", ""))
    fallback_model_flag: str = field(default_factory=lambda: os.getenv("FALLBACK_MODEL_FLAG", ""))
    fallback_command: list[str] = field(default_factory=lambda: _split_command(os.getenv("FALLBACK_CLI_COMMAND", "")))


@dataclass(frozen=True)
class ServerConfig:
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true")
    event_source: str = field(default_factory=lambda: os.getenv("EVENT_SOURCE", "").strip().lower())
    enable_webhook: bool = field(default_factory=lambda: os.getenv("ENABLE_WEBHOOK", "false").lower() == "true")
    poll_interval_seconds: int = field(default_factory=lambda: int(os.getenv("POLL_INTERVAL_SECONDS", "60")))
    poll_items_per_page: int = field(default_factory=lambda: int(os.getenv("POLL_ITEMS_PER_PAGE", "30")))
    poll_pending_pr_batch_size: int = field(default_factory=lambda: int(os.getenv("POLL_PENDING_PR_BATCH_SIZE", "20")))
    poll_pending_pr_max_age_days: int = field(default_factory=lambda: int(os.getenv("POLL_PENDING_PR_MAX_AGE_DAYS", "30")))
    poll_review_submission_batch_size: int = field(
        default_factory=lambda: int(os.getenv("POLL_REVIEW_SUBMISSION_BATCH_SIZE", "20"))
    )
    poll_review_submission_max_age_days: int = field(
        default_factory=lambda: int(os.getenv("POLL_REVIEW_SUBMISSION_MAX_AGE_DAYS", "21"))
    )
    state_file_path: str = field(default_factory=lambda: os.getenv("STATE_FILE_PATH", ".openreview-state.json"))
    max_comment_length: int = field(default_factory=lambda: int(os.getenv("MAX_COMMENT_LENGTH", "4000")))
    pr_head_stabilize_seconds: int = field(default_factory=lambda: int(os.getenv("PR_HEAD_STABILIZE_SECONDS", "300")))


@dataclass(frozen=True)
class LocalRepoConfig:
    path: str = field(default_factory=lambda: os.getenv("LOCAL_REPO_PATH", os.getcwd()))
    default_branch: str = field(default_factory=lambda: os.getenv("LOCAL_REPO_DEFAULT_BRANCH", "main"))
    max_context_files: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_CONTEXT_FILES", "6")))
    max_file_chars: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_FILE_CHARS", "1800")))
    max_total_context_chars: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_TOTAL_CONTEXT_CHARS", "12000")))


@dataclass(frozen=True)
class AppConfig:
    github: GitHubConfig = field(default_factory=GitHubConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    local_repo: LocalRepoConfig = field(default_factory=LocalRepoConfig)


def load_config() -> AppConfig:
    """加载并验证配置"""
    config = AppConfig()
    base_dir = Path(__file__).resolve().parent
    resolved_event_source = config.server.event_source or ("webhook" if config.server.enable_webhook else "polling")
    object.__setattr__(config.server, "event_source", resolved_event_source)
    object.__setattr__(config.server, "enable_webhook", resolved_event_source == "webhook")
    state_file_path = Path(config.server.state_file_path).expanduser()
    if not state_file_path.is_absolute():
        state_file_path = (base_dir / state_file_path).resolve()
    object.__setattr__(config.server, "state_file_path", str(state_file_path))
    local_repo_path = Path(config.local_repo.path).expanduser()
    if not local_repo_path.is_absolute():
        local_repo_path = (base_dir / local_repo_path).resolve()
    object.__setattr__(config.local_repo, "path", str(local_repo_path))
    # 验证必填项
    assert config.github.token, "GITHUB_TOKEN 未设置"
    assert config.github.repo, "GITHUB_REPO 未设置"
    assert config.server.event_source in {"webhook", "polling"}, "EVENT_SOURCE 仅支持 webhook 或 polling"
    assert config.server.poll_interval_seconds > 0, "POLL_INTERVAL_SECONDS 必须大于 0"
    assert 1 <= config.server.poll_items_per_page <= 100, "POLL_ITEMS_PER_PAGE 必须在 1-100 之间"
    assert config.server.poll_pending_pr_batch_size > 0, "POLL_PENDING_PR_BATCH_SIZE 必须大于 0"
    assert config.server.poll_pending_pr_max_age_days > 0, "POLL_PENDING_PR_MAX_AGE_DAYS 必须大于 0"
    assert config.server.poll_review_submission_batch_size > 0, "POLL_REVIEW_SUBMISSION_BATCH_SIZE 必须大于 0"
    assert config.server.poll_review_submission_max_age_days > 0, "POLL_REVIEW_SUBMISSION_MAX_AGE_DAYS 必须大于 0"
    if config.server.enable_webhook:
        assert config.github.webhook_secret, "ENABLE_WEBHOOK=true 时必须设置 GITHUB_WEBHOOK_SECRET"
    supported_clis = {"copilot", "cursor", "codex", "opencode"}
    assert config.cli.primary_cli in supported_clis, f"不支持的 PRIMARY_CLI: {config.cli.primary_cli}"
    if config.cli.fallback_cli:
        assert config.cli.fallback_cli in supported_clis, f"不支持的 FALLBACK_CLI: {config.cli.fallback_cli}"
    return config
