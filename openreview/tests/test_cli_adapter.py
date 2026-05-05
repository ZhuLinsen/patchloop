import subprocess
import unittest
from unittest.mock import patch

from adapters.base import LLMError
from adapters.cli_adapter import CLIAdapter, _summarize_cli_output


class _FakeProcess:
    def __init__(self):
        self.returncode = 0
        self._calls = 0

    def communicate(self, timeout=None):
        self._calls += 1
        if self._calls == 1:
            raise subprocess.TimeoutExpired(
                ["codex"],
                timeout,
                output="phase 1\nphase 2\n",
                stderr="warming up\n",
            )
        return ("done\n", "")

    def kill(self):
        self.returncode = -9


class _TimeoutAdapter(CLIAdapter):
    def _run_command(self, command, stdin_path=None, prompt_chars=0):
        raise subprocess.TimeoutExpired(
            command,
            self.timeout,
            output="step a\nstep b\nstep c\nstep d\nstep e\n",
            stderr="permission denied\ntrace line\n",
        )


class CLIAdapterLoggingTests(unittest.TestCase):
    def test_summarize_cli_output_keeps_tail(self):
        summary = _summarize_cli_output(
            "line1\nline2\nline3\nline4\nline5\n",
            max_lines=3,
            max_chars=100,
        )
        self.assertEqual("line3\nline4\nline5", summary)

    def test_run_command_logs_partial_output_during_progress(self):
        adapter = CLIAdapter(
            command=["codex"],
            prompt_flag="",
            prompt_mode="positional",
            timeout=30,
        )

        with patch("adapters.cli_adapter.subprocess.Popen", return_value=_FakeProcess()):
            with self.assertLogs("adapters.cli_adapter", level="INFO") as captured:
                result = adapter._run_command(["codex", "hello"], prompt_chars=5)

        self.assertEqual("done\n", result.stdout)
        joined = "\n".join(captured.output)
        self.assertIn("CLI 运行中 stdout 摘要", joined)
        self.assertIn("phase 1", joined)
        self.assertIn("CLI 运行中 stderr 摘要", joined)
        self.assertIn("warming up", joined)

    def test_timeout_logs_stdout_and_stderr_summaries(self):
        adapter = _TimeoutAdapter(
            command=["codex"],
            prompt_flag="",
            prompt_mode="positional",
            timeout=30,
            model="gpt-5.4",
        )

        with self.assertLogs("adapters.cli_adapter", level="INFO") as captured:
            with self.assertRaisesRegex(LLMError, "CLI 工具超时"):
                adapter.analyze("prompt")

        joined = "\n".join(captured.output)
        self.assertIn("stdout_chars=", joined)
        self.assertIn("CLI 超时前 stdout 摘要", joined)
        self.assertIn("step e", joined)
        self.assertIn("CLI 超时前 stderr 摘要", joined)
        self.assertIn("permission denied", joined)
