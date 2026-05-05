"""CLI 适配器工厂。"""
from __future__ import annotations

from .base import BaseLLMAdapter, LLMError
from .cli_adapter import (
    CLIAdapter,
    CopilotCLIAdapter,
    CursorCLIAdapter,
    CodexCLIAdapter,
    OpenCodeCLIAdapter,
)
from .execution_adapter import (
    ExecutionCodexCLIAdapter,
    ExecutionCopilotCLIAdapter,
    ExecutionCursorCLIAdapter,
    ExecutionOpenCodeCLIAdapter,
)
from .failover_adapter import FailoverAdapter

__all__ = ["create_adapter", "create_execution_adapter", "create_adapter_from_profiles", "BaseLLMAdapter", "LLMError"]

# --- 已知 CLI 名称到适配器类的映射 ---
_ANALYSIS_ADAPTER_MAP: dict[str, type[CLIAdapter]] = {
    "copilot": CopilotCLIAdapter,
    "cursor": CursorCLIAdapter,
    "codex": CodexCLIAdapter,
    "opencode": OpenCodeCLIAdapter,
}
_EXECUTION_ADAPTER_MAP: dict[str, type[CLIAdapter]] = {
    "copilot": ExecutionCopilotCLIAdapter,
    "cursor": ExecutionCursorCLIAdapter,
    "codex": ExecutionCodexCLIAdapter,
    "opencode": ExecutionOpenCodeCLIAdapter,
}


def create_adapter_from_profiles(
    *,
    cli_profiles: dict,
    primary_cli: str,
    fallback_clis: list[str],
    cli_timeout: int = 120,
    max_idle_seconds: int = 0,
    working_dir: str | None = None,
    execution_mode: bool = False,
    primary_model_override: str = "",
    primary_fallback_model_override: str = "",
) -> BaseLLMAdapter:
    """从 CLIProfile 字典构建带多级 CLI/模型回退的适配器。

    这是新的主入口，直接消费 config.cli 中的 cli_profiles。
    primary_model_override / primary_fallback_model_override 用于 execution_controller
    在运行时覆盖 primary CLI 的模型（如 todo_refactor / simple_executor）。
    """
    candidates: list[BaseLLMAdapter] = []
    seen: set[tuple[str, tuple[str, ...], str, str, int, int]] = set()

    def _add(cli_name: str, profile, model: str):
        normalized = cli_name.strip().lower()
        if not normalized:
            return
        command = profile.command if profile else []
        model_flag = profile.model_flag if profile else ""
        billing = profile.billing if profile else "token"
        key = (normalized, tuple(command or ()), model.strip(), model_flag.strip(), cli_timeout, max_idle_seconds)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            _create_single_adapter(
                execution_mode=execution_mode,
                cli_name=normalized,
                command=command or None,
                model=model,
                model_flag=model_flag,
                timeout=cli_timeout,
                max_idle_seconds=max_idle_seconds,
                working_dir=working_dir,
                billing_type=billing,
            )
        )

    # primary CLI
    primary_profile = cli_profiles.get(primary_cli)
    if primary_profile:
        pm = primary_model_override or primary_profile.primary_model
        _add(primary_cli, primary_profile, pm)
        if primary_fallback_model_override:
            _add(primary_cli, primary_profile, primary_fallback_model_override)
        for fm in primary_profile.fallback_models:
            _add(primary_cli, primary_profile, fm)

    # fallback CLIs
    for fb_cli in fallback_clis:
        fb_profile = cli_profiles.get(fb_cli.strip().lower())
        if not fb_profile:
            continue
        _add(fb_cli, fb_profile, fb_profile.primary_model)
        for fm in fb_profile.fallback_models:
            _add(fb_cli, fb_profile, fm)

    if not candidates:
        raise ValueError(f"未能构建任何 CLI 适配器: primary={primary_cli}, fallbacks={fallback_clis}")
    if len(candidates) == 1:
        return candidates[0]
    return FailoverAdapter(primary=candidates[0], fallbacks=candidates[1:])


def create_adapter(**kwargs) -> BaseLLMAdapter:
    """创建带多级模型/CLI 回退能力的适配器（兼容旧调用方式）。"""
    return _create_failover_adapter(execution_mode=False, **kwargs)


def create_execution_adapter(**kwargs) -> BaseLLMAdapter:
    """创建允许在受限工作区内修改文件的执行适配器（兼容旧调用方式）。"""
    return _create_failover_adapter(execution_mode=True, **kwargs)


def _create_failover_adapter(*, execution_mode: bool, **kwargs) -> BaseLLMAdapter:
    """创建带多级模型/CLI 回退能力的适配器（兼容旧 kwargs 接口）。"""
    timeout = kwargs.get("cli_timeout", 120)
    max_idle_seconds = kwargs.get("max_idle_seconds", 0)
    working_dir = kwargs.get("working_dir")
    candidates: list[BaseLLMAdapter] = []
    seen: set[tuple[str, tuple[str, ...], str, str, int, int]] = set()

    def add_candidate(cli_name: str, command: list[str] | None, model: str, model_flag: str, billing_type: str = "token"):
        normalized = cli_name.strip().lower()
        if not normalized:
            return
        key = (normalized, tuple(command or ()), model.strip(), model_flag.strip(), timeout, max_idle_seconds)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            _create_single_adapter(
                execution_mode=execution_mode,
                cli_name=normalized,
                command=command,
                model=model,
                model_flag=model_flag,
                timeout=timeout,
                max_idle_seconds=max_idle_seconds,
                working_dir=working_dir,
                billing_type=billing_type,
            )
        )

    add_candidate(
        cli_name=kwargs["primary_cli"],
        command=kwargs.get("primary_command"),
        model=kwargs.get("primary_model", ""),
        model_flag=kwargs.get("primary_model_flag", ""),
        billing_type=kwargs.get("primary_billing_type", "token"),
    )
    add_candidate(
        cli_name=kwargs["primary_cli"],
        command=kwargs.get("primary_command"),
        model=kwargs.get("primary_fallback_model", ""),
        model_flag=kwargs.get("primary_model_flag", ""),
        billing_type=kwargs.get("primary_billing_type", "token"),
    )

    fallback_cli = kwargs.get("fallback_cli", "")
    if fallback_cli:
        add_candidate(
            cli_name=fallback_cli,
            command=kwargs.get("fallback_command"),
            model=kwargs.get("fallback_model", ""),
            model_flag=kwargs.get("fallback_model_flag", ""),
            billing_type=kwargs.get("fallback_billing_type", "token"),
        )
        add_candidate(
            cli_name=fallback_cli,
            command=kwargs.get("fallback_command"),
            model=kwargs.get("fallback_fallback_model", ""),
            model_flag=kwargs.get("fallback_model_flag", ""),
            billing_type=kwargs.get("fallback_billing_type", "token"),
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
    max_idle_seconds: int,
    working_dir: str | None,
    execution_mode: bool = False,
    billing_type: str = "token",
) -> BaseLLMAdapter:
    normalized = cli_name.strip().lower()
    effective_model_flag = model_flag.strip()
    adapter_map = _EXECUTION_ADAPTER_MAP if execution_mode else _ANALYSIS_ADAPTER_MAP

    if normalized in adapter_map:
        adapter_cls = adapter_map[normalized]
        return adapter_cls(
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            command=command,
            model=model,
            model_flag=effective_model_flag or None,
            working_dir=working_dir,
            billing_type=billing_type,
        )

    # 未知 CLI：使用通用 CLIAdapter (analysis) 或 ExecutionCLIAdapter (execution)
    from .execution_adapter import ExecutionCLIAdapter
    base_cls: type[CLIAdapter] = ExecutionCLIAdapter if execution_mode else CLIAdapter
    return base_cls(
        command=command or [normalized],
        prompt_flag="--prompt",
        prompt_mode="flag",
        timeout=timeout,
        max_idle_seconds=max_idle_seconds,
        model=model,
        model_flag=effective_model_flag or "",
        working_dir=working_dir,
        billing_type=billing_type,
    )
