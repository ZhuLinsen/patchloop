"""主备 CLI 适配器。"""
from __future__ import annotations

import logging

from .base import BaseLLMAdapter, LLMError

logger = logging.getLogger(__name__)


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
        # 计费模型以主适配器为准
        self.billing_type = getattr(primary, "billing_type", "token")

    def analyze(self, prompt: str, system: str = "") -> str:
        last_error: LLMError | None = None
        for index, adapter in enumerate(self.adapters):
            try:
                return adapter.analyze(prompt, system=system)
            except LLMError as error:
                last_error = error
                if index == len(self.adapters) - 1:
                    break
                logger.warning("当前 CLI/模型调用失败，切换到下一候选: %s", error)

        if last_error is not None:
            raise last_error
        raise LLMError("未配置可用的 CLI 适配器")

    def with_working_dir(self, working_dir: str | None) -> "FailoverAdapter":
        rebound = [adapter.with_working_dir(working_dir) for adapter in self.adapters]
        return FailoverAdapter(primary=rebound[0], fallbacks=rebound[1:])

    def with_log_context(self, log_context: str) -> "FailoverAdapter":
        rebound = [adapter.with_log_context(log_context) for adapter in self.adapters]
        return FailoverAdapter(primary=rebound[0], fallbacks=rebound[1:])
