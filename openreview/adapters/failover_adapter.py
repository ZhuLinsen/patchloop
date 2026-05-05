"""主备 CLI 适配器。"""
from __future__ import annotations

import logging

from .base import BaseLLMAdapter, LLMError

logger = logging.getLogger(__name__)


def _adapter_label(adapter: BaseLLMAdapter) -> str:
    command = getattr(adapter, "command", None)
    model = str(getattr(adapter, "model", "") or "").strip()
    cli_name = ""
    if isinstance(command, list) and command:
        cli_name = str(command[0] or "").strip()
    if cli_name and model:
        return f"{cli_name}/{model}"
    if cli_name:
        return cli_name
    if model:
        return model
    return adapter.__class__.__name__


class FailoverAdapter(BaseLLMAdapter):
    """按顺序尝试多个 CLI/模型组合。"""

    def __init__(
        self,
        primary: BaseLLMAdapter,
        secondary: BaseLLMAdapter | None = None,
        fallbacks: list[BaseLLMAdapter] | None = None,
    ):
        self.adapters = [primary]
        if secondary is not None:
            self.adapters.append(secondary)
        if fallbacks:
            self.adapters.extend(fallbacks)

    def analyze(self, prompt: str, system: str = "") -> str:
        last_error: LLMError | None = None
        total = len(self.adapters)
        for index, adapter in enumerate(self.adapters):
            current_label = _adapter_label(adapter)
            try:
                result = adapter.analyze(prompt, system=system)
                if index > 0:
                    logger.info(
                        "回退候选调用成功: candidate=%d/%d, adapter=%s",
                        index + 1,
                        total,
                        current_label,
                    )
                return result
            except LLMError as error:
                last_error = error
                if index == len(self.adapters) - 1:
                    break
                next_label = _adapter_label(self.adapters[index + 1])
                logger.warning(
                    "候选调用失败: candidate=%d/%d, adapter=%s, error=%s; 切换到 candidate=%d/%d, adapter=%s",
                    index + 1,
                    total,
                    current_label,
                    error,
                    index + 2,
                    total,
                    next_label,
                )

        if last_error is not None:
            logger.error(
                "所有 CLI/模型候选均失败: total=%d, last_adapter=%s, error=%s",
                total,
                _adapter_label(self.adapters[-1]),
                last_error,
            )
            raise last_error
        raise LLMError("未配置可用的 CLI 适配器")
