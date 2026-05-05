"""CLI 适配器工厂。"""
from .base import BaseLLMAdapter, LLMError
from .cli_adapter import CopilotCLIAdapter, CursorCLIAdapter, CodexCLIAdapter, OpenCodeCLIAdapter
from .failover_adapter import FailoverAdapter

__all__ = ["create_adapter", "BaseLLMAdapter", "LLMError"]


def create_adapter(**kwargs) -> BaseLLMAdapter:
    """创建带多级模型/CLI 回退能力的适配器。"""
    timeout = kwargs.get("cli_timeout", 120)
    working_dir = kwargs.get("working_dir")
    candidates: list[BaseLLMAdapter] = []
    seen: set[tuple[str, tuple[str, ...], str, str, int]] = set()

    def add_candidate(cli_name: str, command: list[str] | None, model: str, model_flag: str):
        normalized = cli_name.strip().lower()
        if not normalized:
            return
        key = (normalized, tuple(command or ()), model.strip(), model_flag.strip(), timeout)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            _create_single_adapter(
                cli_name=normalized,
                command=command,
                model=model,
                model_flag=model_flag,
                timeout=timeout,
                working_dir=working_dir,
            )
        )

    add_candidate(
        cli_name=kwargs["primary_cli"],
        command=kwargs.get("primary_command"),
        model=kwargs.get("primary_model", ""),
        model_flag=kwargs.get("primary_model_flag", ""),
    )
    add_candidate(
        cli_name=kwargs["primary_cli"],
        command=kwargs.get("primary_command"),
        model=kwargs.get("primary_fallback_model", ""),
        model_flag=kwargs.get("primary_model_flag", ""),
    )

    fallback_cli = kwargs.get("fallback_cli", "")
    if fallback_cli:
        add_candidate(
            cli_name=fallback_cli,
            command=kwargs.get("fallback_command"),
            model=kwargs.get("fallback_model", ""),
            model_flag=kwargs.get("fallback_model_flag", ""),
        )
        add_candidate(
            cli_name=fallback_cli,
            command=kwargs.get("fallback_command"),
            model=kwargs.get("fallback_fallback_model", ""),
            model_flag=kwargs.get("fallback_model_flag", ""),
        )

    primary = candidates[0]
    if len(candidates) == 1:
        return primary
    return FailoverAdapter(primary=primary, fallbacks=candidates[1:])


def _create_single_adapter(
    cli_name: str,
    command: list[str] | None,
    model: str,
    model_flag: str,
    timeout: int,
    working_dir: str | None,
) -> BaseLLMAdapter:
    normalized = cli_name.strip().lower()
    effective_model_flag = model_flag.strip()
    if normalized == "copilot":
        return CopilotCLIAdapter(
            timeout=timeout,
            command=command,
            model=model,
            model_flag=effective_model_flag or None,
            working_dir=working_dir,
        )
    if normalized == "cursor":
        return CursorCLIAdapter(
            timeout=timeout,
            command=command,
            model=model,
            model_flag=effective_model_flag or None,
            working_dir=working_dir,
        )
    if normalized == "codex":
        return CodexCLIAdapter(
            timeout=timeout,
            command=command,
            model=model,
            model_flag=effective_model_flag or None,
            working_dir=working_dir,
        )
    if normalized == "opencode":
        return OpenCodeCLIAdapter(
            timeout=timeout,
            command=command,
            model=model,
            model_flag=effective_model_flag or None,
            working_dir=working_dir,
        )
    raise ValueError(f"不支持的 CLI: {cli_name}")
