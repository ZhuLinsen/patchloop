"""CLI 命令行工具适配器。"""
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .base import BaseLLMAdapter, LLMError

logger = logging.getLogger(__name__)

# 命令行参数最大安全长度 (保守值)
_MAX_ARG_LENGTH = 8000
_PROGRESS_LOG_INTERVAL_SECONDS = 10
_PARTIAL_OUTPUT_LOG_LINES = 4
_PARTIAL_OUTPUT_LOG_CHARS = 400

# 安全前缀：防止 CLI 工具修改文件
_SAFETY_PREFIX = (
    "【重要】你是一个只读分析助手。"
    "绝对不要修改、创建或删除任何文件。"
    "只输出分析文本。\n\n"
)


def _summarize_cli_output(
    text: str,
    *,
    max_lines: int = _PARTIAL_OUTPUT_LOG_LINES,
    max_chars: int = _PARTIAL_OUTPUT_LOG_CHARS,
) -> str:
    """提取 CLI 最近输出的短摘要，便于在日志中定位卡住的位置。"""
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return ""
    summary = "\n".join(lines[-max_lines:])
    if len(summary) > max_chars:
        summary = "..." + summary[-(max_chars - 3):]
    return summary


class CLIAdapter(BaseLLMAdapter):
    """通用 CLI 工具适配器"""

    def __init__(
        self,
        command: list[str],
        prompt_flag: str = "--prompt",
        prompt_mode: str = "flag",
        timeout: int = 120,
        model: str = "",
        model_flag: str = "",
        working_dir: str | None = None,
    ):
        """
        Args:
            command: 基础命令, 如 ["copilot"] 或 ["codex", "run"]
            prompt_flag: prompt 参数的 flag, 如 "-p" 或 "--prompt"
            prompt_mode: prompt 传递方式: flag | positional | stdin
            timeout: 子进程超时秒数
            model: CLI 内要使用的模型名
            model_flag: 模型参数 flag；为空则不传模型参数
            working_dir: CLI 执行目录，通常应指向被分析的仓库根目录
        """
        self.command = command
        self.prompt_flag = prompt_flag
        self.prompt_mode = prompt_mode
        self.timeout = timeout
        self.model = model.strip()
        self.model_flag = model_flag.strip()
        self.working_dir = str(Path(working_dir).resolve()) if working_dir else None

    def analyze(self, prompt: str, system: str = "") -> str:
        full_prompt = _SAFETY_PREFIX
        if system:
            full_prompt += f"{system}\n\n"
        full_prompt += prompt
        base_cmd = list(self.command)
        if self.model and self.model_flag:
            base_cmd.extend([self.model_flag, self.model])

        try:
            if len(full_prompt) > _MAX_ARG_LENGTH:
                result = self._run_with_stdin(base_cmd, full_prompt)
            else:
                cmd = self._build_command(base_cmd, full_prompt)
                result = self._run_command(cmd, prompt_chars=len(full_prompt))
            if result.returncode != 0:
                logger.error("CLI tool stderr: %s", result.stderr)
                raise LLMError(f"CLI 工具返回非零退出码 {result.returncode}: {result.stderr[:500]}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            stdout = str(getattr(exc, "output", "") or "")
            stderr = str(getattr(exc, "stderr", "") or "")
            logger.warning(
                "CLI 调用超时: cli=%s, model=%s, timeout=%ss, cwd=%s, stdout_chars=%d, stderr_chars=%d",
                self.command[0],
                self.model or "default",
                self.timeout,
                self.working_dir or os.getcwd(),
                len(stdout),
                len(stderr),
            )
            stdout_summary = _summarize_cli_output(stdout)
            stderr_summary = _summarize_cli_output(stderr)
            if stdout_summary:
                logger.warning("CLI 超时前 stdout 摘要: %s", stdout_summary)
            if stderr.strip():
                logger.warning("CLI 超时前 stderr 摘要: %s", stderr_summary or stderr.strip()[:300])
            raise LLMError(f"CLI 工具超时 ({self.timeout}s)")
        except FileNotFoundError:
            raise LLMError(f"CLI 工具未找到: {self.command[0]}")

    def _build_command(self, command: list[str], prompt: str) -> list[str]:
        """根据不同 CLI 的参数风格组装命令。"""
        if self.prompt_mode == "flag":
            return command + [self.prompt_flag, prompt]
        if self.prompt_mode == "positional":
            return command + [prompt]
        if self.prompt_mode == "stdin":
            return command
        raise LLMError(f"不支持的 prompt_mode: {self.prompt_mode}")

    def _run_with_stdin(self, command: list[str], prompt: str) -> subprocess.CompletedProcess:
        """将 prompt 写入 stdin 传给 CLI。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            tmp_path = f.name
        try:
            return self._run_command(command, stdin_path=tmp_path, prompt_chars=len(prompt))
        finally:
            os.unlink(tmp_path)

    def _run_command(
        self,
        command: list[str],
        stdin_path: str | None = None,
        prompt_chars: int = 0,
    ) -> subprocess.CompletedProcess:
        """执行 CLI 命令，并在长时间运行时输出心跳日志。"""
        stdin_handle = None
        started_at = time.monotonic()
        last_stdout_summary = ""
        last_stderr_summary = ""
        transport = "stdin" if stdin_path else self.prompt_mode
        logger.info(
            "CLI 调用开始: cli=%s, model=%s, timeout=%ss, cwd=%s, prompt_chars=%d, transport=%s",
            self.command[0],
            self.model or "default",
            self.timeout,
            self.working_dir or os.getcwd(),
            max(prompt_chars, 0),
            transport,
        )
        try:
            if stdin_path:
                stdin_handle = open(stdin_path, "r", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
            )
            while True:
                elapsed = time.monotonic() - started_at
                remaining = self.timeout - elapsed
                if remaining <= 0:
                    process.kill()
                    stdout, stderr = process.communicate()
                    raise subprocess.TimeoutExpired(command, self.timeout, output=stdout, stderr=stderr)
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(_PROGRESS_LOG_INTERVAL_SECONDS, remaining)
                    )
                    duration = time.monotonic() - started_at
                    logger.info(
                        "CLI 调用完成: cli=%s, model=%s, exit_code=%s, elapsed=%.1fs",
                        self.command[0],
                        self.model or "default",
                        process.returncode,
                        duration,
                    )
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired as exc:
                    elapsed_now = time.monotonic() - started_at
                    stdout_summary = _summarize_cli_output(str(getattr(exc, "output", "") or ""))
                    stderr_summary = _summarize_cli_output(str(getattr(exc, "stderr", "") or ""))
                    if stdout_summary and stdout_summary != last_stdout_summary:
                        logger.info(
                            "CLI 运行中 stdout 摘要: cli=%s, model=%s, elapsed=%.1fs, output=%s",
                            self.command[0],
                            self.model or "default",
                            elapsed_now,
                            stdout_summary,
                        )
                        last_stdout_summary = stdout_summary
                    if stderr_summary and stderr_summary != last_stderr_summary:
                        logger.info(
                            "CLI 运行中 stderr 摘要: cli=%s, model=%s, elapsed=%.1fs, output=%s",
                            self.command[0],
                            self.model or "default",
                            elapsed_now,
                            stderr_summary,
                        )
                        last_stderr_summary = stderr_summary
                    logger.info(
                        "CLI 仍在执行: cli=%s, model=%s, elapsed=%.1fs",
                        self.command[0],
                        self.model or "default",
                        elapsed_now,
                    )
        finally:
            if stdin_handle is not None:
                stdin_handle.close()


class CopilotCLIAdapter(CLIAdapter):
    """GitHub Copilot CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
    ):
        super().__init__(
            command=command or ["copilot"],
            prompt_flag="-p",
            prompt_mode="flag",
            timeout=timeout,
            model=model,
            model_flag=model_flag or "",
            working_dir=working_dir,
        )


class CursorCLIAdapter(CLIAdapter):
    """Cursor CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
    ):
        super().__init__(
            command=command or ["cursor"],
            prompt_flag="--prompt",
            prompt_mode="flag",
            timeout=timeout,
            model=model,
            model_flag=model_flag or "",
            working_dir=working_dir,
        )


class CodexCLIAdapter(CLIAdapter):
    """Codex CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
    ):
        super().__init__(
            command=command or ["codex", "exec", "--skip-git-repo-check"],
            prompt_flag="",
            prompt_mode="positional",
            timeout=timeout,
            model=model,
            model_flag=model_flag or "--model",
            working_dir=working_dir,
        )

    def analyze(self, prompt: str, system: str = "") -> str:
        full_prompt = _SAFETY_PREFIX
        if system:
            full_prompt += f"{system}\n\n"
        full_prompt += prompt
        base_cmd = list(self.command)
        if self.model and self.model_flag:
            base_cmd.extend([self.model_flag, self.model])

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            cmd = base_cmd + ["--output-last-message", output_path]
            if len(full_prompt) > _MAX_ARG_LENGTH:
                result = self._run_with_stdin(cmd, full_prompt)
            else:
                result = self._run_command(self._build_command(cmd, full_prompt), prompt_chars=len(full_prompt))
            if result.returncode != 0:
                logger.error("CLI tool stderr: %s", result.stderr)
                raise LLMError(f"CLI 工具返回非零退出码 {result.returncode}: {result.stderr[:500]}")

            try:
                output = open(output_path, "r", encoding="utf-8").read().strip()
            except OSError:
                output = ""
            return output or result.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            stdout = str(getattr(exc, "output", "") or "")
            stderr = str(getattr(exc, "stderr", "") or "")
            logger.warning(
                "CLI 调用超时: cli=%s, model=%s, timeout=%ss, cwd=%s, stdout_chars=%d, stderr_chars=%d",
                self.command[0],
                self.model or "default",
                self.timeout,
                self.working_dir or os.getcwd(),
                len(stdout),
                len(stderr),
            )
            stdout_summary = _summarize_cli_output(stdout)
            stderr_summary = _summarize_cli_output(stderr)
            if stdout_summary:
                logger.warning("CLI 超时前 stdout 摘要: %s", stdout_summary)
            if stderr.strip():
                logger.warning("CLI 超时前 stderr 摘要: %s", stderr_summary or stderr.strip()[:300])
            raise LLMError(f"CLI 工具超时 ({self.timeout}s)")
        except FileNotFoundError:
            raise LLMError(f"CLI 工具未找到: {self.command[0]}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class OpenCodeCLIAdapter(CLIAdapter):
    """OpenCode CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
    ):
        super().__init__(
            command=command or ["opencode"],
            prompt_flag="--prompt",
            prompt_mode="flag",
            timeout=timeout,
            model=model,
            model_flag=model_flag or "--model",
            working_dir=working_dir,
        )
