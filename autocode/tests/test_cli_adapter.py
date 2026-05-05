import os
import subprocess
import time
import unittest
from unittest import mock

from adapters.base import LLMError
from adapters.cli_adapter import CLIAdapter, CopilotCLIAdapter, _safe_unlink, _stream_progress_summary
from adapters.execution_adapter import ExecutionCLIAdapter, ExecutionCodexCLIAdapter, ExecutionCopilotCLIAdapter


class CLIAdapterTests(unittest.TestCase):
    def test_analyze_timeout_message_includes_cli_and_model(self):
        adapter = CLIAdapter(command=["codex"], timeout=45, model="gpt-5.4", model_flag="--model")
        with mock.patch.object(adapter, "_run_command", side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=45)):
            with self.assertRaises(LLMError) as exc:
                adapter.analyze("hello")
        self.assertIn("cli=codex", str(exc.exception))
        self.assertIn("model=gpt-5.4", str(exc.exception))
        self.assertIn("timeout=45s", str(exc.exception))

    def test_execution_adapter_timeout_message_includes_cli_and_model(self):
        adapter = ExecutionCLIAdapter(command=["codex"], timeout=20, model="gpt-5.4", model_flag="--model")
        with mock.patch.object(adapter, "_run_command", side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=20)):
            with self.assertRaises(LLMError) as exc:
                adapter.analyze("hello")
        self.assertIn("cli=codex", str(exc.exception))
        self.assertIn("model=gpt-5.4", str(exc.exception))
        self.assertIn("timeout=20s", str(exc.exception))

    def test_analyze_timeout_message_includes_recent_output_and_context(self):
        adapter = CLIAdapter(
            command=["codex"],
            timeout=45,
            model="gpt-5.4",
            model_flag="--model",
        ).with_log_context("issue-execution#880")
        timeout_exc = subprocess.TimeoutExpired(
            cmd=["codex"],
            timeout=45,
            output="working on patch",
            stderr="still running",
        )
        with mock.patch.object(adapter, "_run_command", side_effect=timeout_exc):
            with self.assertRaises(LLMError) as exc:
                adapter.analyze("hello")
        self.assertIn("context=issue-execution#880", str(exc.exception))
        self.assertIn("recent_stdout=working on patch", str(exc.exception))
        self.assertIn("recent_stderr=still running", str(exc.exception))

    def test_idle_limit_message_includes_context_and_recent_output(self):
        adapter = CLIAdapter(
            command=["codex"],
            timeout=45,
            max_idle_seconds=180,
            model="gpt-5.4",
            model_flag="--model",
        ).with_log_context("issue-execution#883")
        idle_exc = subprocess.TimeoutExpired(
            cmd=["codex"],
            timeout=45,
            output="",
            stderr="same diff keeps repeating",
        )
        setattr(idle_exc, "autocode_reason", "idle")
        setattr(idle_exc, "autocode_idle_seconds", 180)

        with mock.patch.object(adapter, "_run_command", side_effect=idle_exc):
            with self.assertRaises(LLMError) as exc:
                adapter.analyze("hello")

        self.assertIn("CLI 长时间无新输出", str(exc.exception))
        self.assertIn("idle=180s", str(exc.exception))
        self.assertIn("context=issue-execution#883", str(exc.exception))
        self.assertIn("recent_stderr=same diff keeps repeating", str(exc.exception))

    def test_safe_unlink_ignores_os_errors(self):
        with mock.patch("adapters.cli_adapter.os.unlink", side_effect=OSError("busy")):
            _safe_unlink("/tmp/non-removable")

    def test_analyze_logs_output_preview(self):
        adapter = CLIAdapter(command=["codex"], timeout=45, model="gpt-5.4", model_flag="--model")
        completed = subprocess.CompletedProcess(["codex"], 0, "final answer\n", "")
        with mock.patch.object(adapter, "_run_command", return_value=completed):
            with self.assertLogs("adapters.cli_adapter", level="INFO") as logs:
                result = adapter.analyze("hello")
        self.assertEqual("final answer", result)
        self.assertTrue(any("CLI 输出摘要" in line for line in logs.output))

    def test_with_log_context_is_reflected_in_logs(self):
        adapter = CLIAdapter(
            command=["codex"],
            timeout=45,
            model="gpt-5.4",
            model_flag="--model",
        ).with_log_context("issue-plan#880:triage")
        completed = subprocess.CompletedProcess(["codex"], 0, "final answer\n", "")
        with mock.patch.object(adapter, "_run_command", return_value=completed):
            with self.assertLogs("adapters.cli_adapter", level="INFO") as logs:
                adapter.analyze("hello")
        self.assertTrue(any("context=issue-plan#880:triage" in line for line in logs.output))

    def test_execution_codex_adapter_defaults_to_workspace_write_sandbox(self):
        adapter = ExecutionCodexCLIAdapter()
        self.assertEqual(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write"],
            adapter.command,
        )

    def test_execution_codex_adapter_respects_explicit_sandbox(self):
        adapter = ExecutionCodexCLIAdapter(
            command=["codex", "exec", "--skip-git-repo-check", "--sandbox", "danger-full-access"]
        )
        self.assertEqual(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "danger-full-access"],
            adapter.command,
        )

    def test_execution_copilot_adapter_defaults_to_allow_all(self):
        adapter = ExecutionCopilotCLIAdapter()
        self.assertEqual(
            ["copilot", "--allow-all"],
            adapter.command,
        )

    def test_execution_copilot_adapter_respects_explicit_allow_all(self):
        adapter = ExecutionCopilotCLIAdapter(command=["copilot", "--allow-all"])
        self.assertEqual(["copilot", "--allow-all"], adapter.command)

    def test_execution_copilot_adapter_no_duplicate_flags(self):
        adapter = ExecutionCopilotCLIAdapter(
            command=["copilot", "--allow-all"]
        )
        self.assertEqual(
            ["copilot", "--allow-all"],
            adapter.command,
        )

    def test_run_command_starts_new_session(self):
        adapter = CLIAdapter(command=["codex"], timeout=45, model="gpt-5.4", model_flag="--model")

        class ImmediateProcess:
            def __init__(self):
                self.pid = 1234
                self.returncode = 0
                self.stdout = None
                self.stderr = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        process = ImmediateProcess()
        with mock.patch("adapters.cli_adapter.subprocess.Popen", return_value=process) as popen:
            result = adapter._run_command(["codex", "hello"])
        self.assertEqual(0, result.returncode)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_command_preview_for_logs_keeps_command_readable(self):
        positional = CLIAdapter(
            command=["codex", "exec"],
            prompt_mode="positional",
            timeout=45,
            model="gpt-5.4",
            model_flag="--model",
        )
        preview = positional._command_preview_for_logs(["codex", "exec", "very secret prompt"])
        self.assertIn("<prompt>", preview)
        self.assertNotIn("very secret prompt", preview)

        flag = CLIAdapter(
            command=["cursor"],
            prompt_flag="--prompt",
            prompt_mode="flag",
            timeout=45,
            model="gpt-5.4",
            model_flag="--model",
        )
        preview = flag._command_preview_for_logs(["cursor", "--prompt", "very secret prompt", "--model", "gpt-5.4"])
        self.assertIn("<prompt>", preview)
        self.assertNotIn("very secret prompt", preview)

    def test_prompt_summary_for_logs_keeps_preview_and_tail(self):
        adapter = CLIAdapter(command=["codex"], timeout=45, model="gpt-5.4", model_flag="--model")
        prompt = "start " + ("x" * 500) + " end"
        summary = adapter._prompt_summary_for_logs(prompt, stdin_path="/tmp/prompt.txt")
        self.assertEqual("stdin", summary["transport"])
        self.assertEqual("/tmp/prompt.txt", summary["stdin_path"])
        self.assertEqual(str(len(prompt)), summary["chars"])
        self.assertIn("start", summary["preview"])
        self.assertIn("end", summary["tail"])

    def test_handle_timeout_logs_cleanup_details(self):
        adapter = CLIAdapter(command=["codex"], timeout=45, model="gpt-5.4", model_flag="--model")

        class TimeoutProcess:
            def __init__(self):
                self.pid = 4321
                self.returncode = None
                self.stdout = None
                self.stderr = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

            def terminate(self):
                return None

            def kill(self):
                self.returncode = -9
                return None

        with mock.patch("adapters.cli_adapter.os.getpgid", return_value=4321), mock.patch(
            "adapters.cli_adapter.os.killpg"
        ):
            with self.assertLogs("adapters.cli_adapter", level="WARNING") as logs:
                adapter._handle_timeout(
                    TimeoutProcess(),
                    command=["codex"],
                    started_at=time.monotonic() - 45,
                    prompt_summary=adapter._prompt_summary_for_logs("debug this timeout"),
                    stdout_chunks=["partial stdout"],
                    stderr_chunks=["partial stderr"],
                    stdout_thread=None,
                    stderr_thread=None,
                )
        combined_logs = "\n".join(logs.output)
        self.assertIn("CLI 调用超时，准备终止进程", combined_logs)
        self.assertIn("prompt_transport=argv", combined_logs)
        self.assertIn("prompt_preview=debug this timeout", combined_logs)
        self.assertIn("cleanup=pgid=4321,sent=SIGTERM,exit_after_term=-15,returncode=-15", combined_logs)
        self.assertIn("active_stream=stdout+stderr", combined_logs)
        self.assertIn("stdout=partial stdout", combined_logs)
        self.assertIn("stderr=partial stderr", combined_logs)

    def test_stream_progress_summary_prefers_active_stderr_output(self):
        summary = _stream_progress_summary(
            [],
            ["codex is diffing files"],
            last_stdout_chars=0,
            last_stderr_chars=0,
        )

        self.assertEqual("stderr", summary["active_stream"])
        self.assertEqual("codex is diffing files", summary["recent_output"])
        self.assertEqual(str(len("codex is diffing files")), summary["new_output_chars"])

    def test_copilot_run_with_stdin_uses_pipe_not_file(self):
        """CopilotCLIAdapter._run_with_stdin 使用 os.pipe 而非临时文件传递 stdin。"""
        adapter = CopilotCLIAdapter(timeout=10, model="gpt-4.1", model_flag="--model")
        captured_stdin = {}

        original_popen = subprocess.Popen

        class FakeProcess:
            def __init__(self, *a, **kw):
                import stat
                stdin_fd = kw.get("stdin")
                if stdin_fd is not None:
                    fd_stat = os.fstat(stdin_fd.fileno())
                    captured_stdin["is_pipe"] = stat.S_ISFIFO(fd_stat.st_mode)
                    captured_stdin["content"] = stdin_fd.read()
                self.pid = 9999
                self.returncode = 0
                self.stdout = None
                self.stderr = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        with mock.patch("adapters.cli_adapter.subprocess.Popen", FakeProcess):
            result = adapter._run_with_stdin(["copilot", "--model", "gpt-4.1"], "hello from pipe")

        self.assertTrue(captured_stdin.get("is_pipe"), "stdin should be a pipe (FIFO), not a regular file")
        self.assertEqual("hello from pipe", captured_stdin.get("content"))
        self.assertEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
