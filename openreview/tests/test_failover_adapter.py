import unittest

from adapters.base import BaseLLMAdapter, LLMError
from adapters.failover_adapter import FailoverAdapter


class _StubAdapter(BaseLLMAdapter):
    def __init__(self, *, cli: str, model: str, response: str = "", error: str = ""):
        self.command = [cli]
        self.model = model
        self._response = response
        self._error = error

    def analyze(self, prompt: str, system: str = "") -> str:
        if self._error:
            raise LLMError(self._error)
        return self._response


class FailoverAdapterLoggingTests(unittest.TestCase):
    def test_logs_candidate_switch_and_fallback_success(self):
        adapter = FailoverAdapter(
            primary=_StubAdapter(cli="codex", model="gpt-5.4", error="CLI 工具超时 (300s)"),
            fallbacks=[_StubAdapter(cli="codex", model="gpt-5.3-codex", response="ok")],
        )

        with self.assertLogs("adapters.failover_adapter", level="INFO") as captured:
            result = adapter.analyze("prompt")

        self.assertEqual("ok", result)
        joined = "\n".join(captured.output)
        self.assertIn("candidate=1/2, adapter=codex/gpt-5.4", joined)
        self.assertIn("candidate=2/2, adapter=codex/gpt-5.3-codex", joined)
        self.assertIn("回退候选调用成功", joined)

    def test_logs_final_failure_after_all_candidates_fail(self):
        adapter = FailoverAdapter(
            primary=_StubAdapter(cli="codex", model="gpt-5.4", error="CLI 工具超时 (300s)"),
            fallbacks=[_StubAdapter(cli="copilot", model="gpt-4.1", error="CLI 工具未找到: copilot")],
        )

        with self.assertLogs("adapters.failover_adapter", level="INFO") as captured:
            with self.assertRaisesRegex(LLMError, "CLI 工具未找到: copilot"):
                adapter.analyze("prompt")

        joined = "\n".join(captured.output)
        self.assertIn("所有 CLI/模型候选均失败", joined)
        self.assertIn("last_adapter=copilot/gpt-4.1", joined)
