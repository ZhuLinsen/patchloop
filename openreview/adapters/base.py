"""分析适配器基类。"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLLMAdapter(ABC):
    """分析适配器统一接口。"""

    @abstractmethod
    def analyze(self, prompt: str, system: str = "") -> str:
        """
        发送 prompt 给 CLI 分析器并返回文本回复。

        Args:
            prompt: 用户 prompt
            system: 可选的 system message

        Returns:
            分析器的文本回复

        Raises:
            LLMError: 当分析器调用失败时
        """
        pass

    def chat(self, prompt: str, system: str = "") -> str:
        """兼容旧调用，内部仍统一走 `analyze()`。"""
        return self.analyze(prompt, system=system)

    def health_check(self) -> bool:
        """检查分析后端是否可用。"""
        try:
            resp = self.analyze("请回复 OK", system="你是一个测试助手")
            return len(resp.strip()) > 0
        except Exception as e:
            logger.error("LLM health check failed: %s", e)
            return False


class LLMError(Exception):
    """LLM 调用异常"""
    pass
