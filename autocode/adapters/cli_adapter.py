"""CLI 命令行工具适配器。"""
from __future__ import annotations

import copy
import logging
import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .base import BaseLLMAdapter, LLMError

logger = logging.getLogger(__name__)

# 命令行参数最大安全长度 (Linux MAX_ARG_STRLEN=128KB，留余量取 120KB)
_MAX_ARG_LENGTH = 120000
_PROGRESS_LOG_INTERVAL_SECONDS = 10
_TERMINATION_GRACE_SECONDS = 3
_KILL_WAIT_SECONDS = 5
_STREAM_JOIN_TIMEOUT_SECONDS = 1

# 安全前缀：防止 CLI 工具修改文件
_SAFETY_PREFIX = (
    "【重要】你是一个只读分析助手。"
    "绝对不要修改、创建或删除任何文件。"
    "只输出分析文本。\n\n"
)


def _safe_unlink(path: str):
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("临时文件删除失败: path=%s error=%s", path, exc)


def _preview_text(text: str, *, limit: int = 240) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 1)] + "..."


def _preview_text_from_end(text: str, *, limit: int = 240) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return "..." + normalized[-max(limit - 3, 1):]


def _preview_command(command: list[str], *, limit: int = 240) -> str:
    if not command:
        return "-"
    try:
        rendered = shlex.join(command)
    except Exception:
        rendered = " ".join(str(part) for part in command)
    return _preview_text(rendered, limit=limit)


def _stream_progress_summary(
    stdout_chunks: list[str],
    stderr_chunks: list[str],
    *,
    last_stdout_chars: int = 0,
    last_stderr_chars: int = 0,
) -> dict[str, str]:
    stdout_chars = sum(len(chunk) for chunk in stdout_chunks)
    stderr_chars = sum(len(chunk) for chunk in stderr_chunks)
    new_stdout_chars = max(stdout_chars - last_stdout_chars, 0)
    new_stderr_chars = max(stderr_chars - last_stderr_chars, 0)
    stdout_preview = _preview_recent_chunks(stdout_chunks, limit=80) or ""
    stderr_preview = _preview_recent_chunks(stderr_chunks, limit=80) or ""

    if new_stdout_chars > 0 and new_stderr_chars == 0:
        active_stream = "stdout"
        recent_output = stdout_preview or "-"
        new_output_chars = new_stdout_chars
    elif new_stderr_chars > 0 and new_stdout_chars == 0:
        # Codex/Copilot 等 CLI 往往把过程输出写到 stderr，stderr 不等于出错。
        active_stream = "stderr"
        recent_output = stderr_preview or "-"
        new_output_chars = new_stderr_chars
    elif new_stdout_chars > 0 and new_stderr_chars > 0:
        active_stream = "stdout+stderr"
        recent_output = _combine_stream_previews(stdout_preview, stderr_preview)
        new_output_chars = new_stdout_chars + new_stderr_chars
    elif stderr_preview and not stdout_preview:
        active_stream = "stderr"
        recent_output = stderr_preview
        new_output_chars = new_stderr_chars
    elif stdout_preview and not stderr_preview:
        active_stream = "stdout"
        recent_output = stdout_preview
        new_output_chars = new_stdout_chars
    elif stdout_preview and stderr_preview:
        active_stream = "stdout+stderr"
        recent_output = _combine_stream_previews(stdout_preview, stderr_preview)
        new_output_chars = new_stdout_chars + new_stderr_chars
    else:
        active_stream = "-"
        recent_output = "-"
        new_output_chars = 0

    return {
        "active_stream": active_stream,
        "recent_output": recent_output,
        "new_output_chars": str(new_output_chars),
        "stdout_chars": str(stdout_chars),
        "stderr_chars": str(stderr_chars),
    }


def _combine_stream_previews(stdout_preview: str, stderr_preview: str) -> str:
    parts: list[str] = []
    if stdout_preview:
        parts.append(f"stdout={stdout_preview}")
    if stderr_preview:
        parts.append(f"stderr={stderr_preview}")
    return " | ".join(parts) or "-"


class CLIAdapter(BaseLLMAdapter):
    """通用 CLI 工具适配器"""

    def __init__(
        self,
        command: list[str],
        prompt_flag: str = "--prompt",
        prompt_mode: str = "flag",
        timeout: int = 120,
        max_idle_seconds: int = 0,
        model: str = "",
        model_flag: str = "",
        working_dir: str | None = None,
        log_context: str = "",
        billing_type: str = "token",
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
            billing_type: 计费模型 "token" (按量) | "call" (按次)
        """
        self.command = command
        self.prompt_flag = prompt_flag
        self.prompt_mode = prompt_mode
        self.timeout = timeout
        self.max_idle_seconds = max(int(max_idle_seconds or 0), 0)
        self.model = model.strip()
        self.model_flag = model_flag.strip()
        self.working_dir = str(Path(working_dir).resolve()) if working_dir else None
        self.log_context = log_context.strip()
        self.billing_type = billing_type if billing_type in ("token", "call") else "token"

    def _billing_instruction(self) -> str:
        """返回按次计费 CLI 的额外 prompt 指令，按量计费时为空。"""
        if self.billing_type != "call":
            return ""
        return (
            "【计费约束】当前 CLI 按次计费，请在本次回复中一次性完成所有分析和改动，"
            "不要拆分成多步或要求后续交互。如果任务复杂，优先给出最关键的完整实现。\n\n"
        )

    def _timeout_error_message(self, *, output: str = "", stderr: str = "") -> str:
        message = f"CLI 工具超时: cli={self.command[0]}, model={self.model or 'default'}, timeout={self.timeout}s"
        if self.log_context:
            message += f", context={self.log_context}"
        if output:
            message += f", recent_stdout={_preview_text(output, limit=160)}"
        if stderr:
            message += f", recent_stderr={_preview_text(stderr, limit=160)}"
        return message

    def _idle_error_message(self, *, idle_seconds: float, output: str = "", stderr: str = "") -> str:
        message = (
            f"CLI 长时间无新输出: cli={self.command[0]}, model={self.model or 'default'}, "
            f"idle={int(max(idle_seconds, 0))}s"
        )
        if self.log_context:
            message += f", context={self.log_context}"
        if output:
            message += f", recent_stdout={_preview_text(output, limit=160)}"
        if stderr:
            message += f", recent_stderr={_preview_text(stderr, limit=160)}"
        return message

    def _limit_error_message(self, exc: subprocess.TimeoutExpired) -> str:
        output = str(getattr(exc, "output", "") or "")
        stderr = str(getattr(exc, "stderr", "") or "")
        if str(getattr(exc, "autocode_reason", "") or "") == "idle":
            idle_seconds = float(getattr(exc, "autocode_idle_seconds", 0) or self.max_idle_seconds or 0)
            return self._idle_error_message(idle_seconds=idle_seconds, output=output, stderr=stderr)
        return self._timeout_error_message(output=output, stderr=stderr)

    def analyze(self, prompt: str, system: str = "") -> str:
        full_prompt = _SAFETY_PREFIX
        billing_hint = self._billing_instruction()
        if billing_hint:
            full_prompt += billing_hint
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
                result = self._run_command(cmd, prompt=full_prompt)
            if result.returncode != 0:
                self._log_failure(result)
                raise LLMError(f"CLI 工具返回非零退出码 {result.returncode}: {result.stderr[:500]}")
            output = result.stdout.strip()
            if output:
                self._log_output_summary(output)
            return output
        except subprocess.TimeoutExpired as exc:
            raise LLMError(self._limit_error_message(exc))
        except FileNotFoundError:
            raise LLMError(f"CLI 工具未找到: {self.command[0]}")
        except OSError as exc:
            logger.exception(
                "CLI 调用异常: context=%s cli=%s, model=%s, error=%s",
                self.log_context or "-",
                self.command[0],
                self.model or "default",
                exc,
            )
            raise LLMError(f"CLI 调用异常: cli={self.command[0]}, error={exc}") from exc

    def with_working_dir(self, working_dir: str | None) -> "CLIAdapter":
        cloned = copy.copy(self)
        cloned.command = list(self.command)
        cloned.working_dir = str(Path(working_dir).resolve()) if working_dir else None
        return cloned

    def with_log_context(self, log_context: str) -> "CLIAdapter":
        cloned = copy.copy(self)
        cloned.command = list(self.command)
        cloned.log_context = str(log_context or "").strip()
        return cloned

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
            return self._run_command(command, stdin_path=tmp_path, prompt=prompt)
        finally:
            _safe_unlink(tmp_path)

    def _run_command(
        self,
        command: list[str],
        stdin_path: str | None = None,
        prompt: str = "",
        stdin_handle=None,
    ) -> subprocess.CompletedProcess:
        """执行 CLI 命令，并在长时间运行时输出心跳日志。"""
        _owns_stdin_handle = stdin_handle is None
        started_at = time.monotonic()
        stdin_label = stdin_path or ("pipe" if stdin_handle else None)
        command_preview = self._command_preview_for_logs(command, stdin_path=stdin_label)
        prompt_summary = self._prompt_summary_for_logs(prompt, stdin_path=stdin_label)
        logger.info(
            "CLI 调用开始: context=%s cli=%s, model=%s, timeout=%ss, cwd=%s, argv=%s, prompt_transport=%s, prompt_chars=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            self.timeout,
            self.working_dir or os.getcwd(),
            command_preview,
            prompt_summary["transport"],
            prompt_summary["chars"],
        )
        logger.info(
            "CLI Prompt 摘要: context=%s stdin_path=%s preview=%s tail=%s",
            self.log_context or "-",
            prompt_summary["stdin_path"],
            prompt_summary["preview"],
            prompt_summary["tail"],
        )
        try:
            if stdin_path and stdin_handle is None:
                stdin_handle = open(stdin_path, "r", encoding="utf-8")
                _owns_stdin_handle = True
            process = subprocess.Popen(
                command,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                start_new_session=True,
            )
            logger.info(
                "CLI 子进程已启动: context=%s cli=%s, model=%s, pid=%s",
                self.log_context or "-",
                self.command[0],
                self.model or "default",
                process.pid,
            )
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            stdout_thread = self._start_stream_reader(process.stdout, stdout_chunks)
            stderr_thread = self._start_stream_reader(process.stderr, stderr_chunks)
            last_stdout_chars = 0
            last_stderr_chars = 0
            last_activity_at = started_at
            while True:
                elapsed = time.monotonic() - started_at
                remaining = self.timeout - elapsed
                if remaining <= 0:
                    self._handle_timeout(
                        process,
                        command=command,
                        started_at=started_at,
                        prompt_summary=prompt_summary,
                        stdout_chunks=stdout_chunks,
                        stderr_chunks=stderr_chunks,
                        stdout_thread=stdout_thread,
                        stderr_thread=stderr_thread,
                    )
                    stdout = "".join(stdout_chunks)
                    stderr = "".join(stderr_chunks)
                    raise subprocess.TimeoutExpired(command, self.timeout, output=stdout, stderr=stderr)
                try:
                    process.wait(timeout=min(_PROGRESS_LOG_INTERVAL_SECONDS, remaining))
                    self._join_stream_reader(stdout_thread, stderr_thread)
                    stdout = "".join(stdout_chunks)
                    stderr = "".join(stderr_chunks)
                    duration = time.monotonic() - started_at
                    logger.info(
                        "CLI 调用完成: context=%s cli=%s, model=%s, exit_code=%s, elapsed=%.1fs, stdout_chars=%s, stderr_chars=%s",
                        self.log_context or "-",
                        self.command[0],
                        self.model or "default",
                        process.returncode,
                        duration,
                        len(stdout or ""),
                        len(stderr or ""),
                    )
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    progress = _stream_progress_summary(
                        stdout_chunks,
                        stderr_chunks,
                        last_stdout_chars=last_stdout_chars,
                        last_stderr_chars=last_stderr_chars,
                    )
                    new_output_chars = int(progress["new_output_chars"] or "0")
                    if new_output_chars > 0:
                        last_activity_at = now
                    idle_for = max(now - last_activity_at, 0.0)
                    if self.max_idle_seconds > 0 and idle_for >= self.max_idle_seconds:
                        self._handle_idle_limit(
                            process,
                            command=command,
                            started_at=started_at,
                            idle_for=idle_for,
                            prompt_summary=prompt_summary,
                            stdout_chunks=stdout_chunks,
                            stderr_chunks=stderr_chunks,
                            stdout_thread=stdout_thread,
                            stderr_thread=stderr_thread,
                        )
                        stdout = "".join(stdout_chunks)
                        stderr = "".join(stderr_chunks)
                        exc = subprocess.TimeoutExpired(command, self.timeout, output=stdout, stderr=stderr)
                        setattr(exc, "autocode_reason", "idle")
                        setattr(exc, "autocode_idle_seconds", idle_for)
                        raise exc
                    logger.info(
                        "CLI 仍在执行: context=%s cli=%s, model=%s, elapsed=%.1fs, idle_for=%.1fs, active_stream=%s, new_output_chars=%s, stdout_chars=%s, stderr_chars=%s, recent_output=%s",
                        self.log_context or "-",
                        self.command[0],
                        self.model or "default",
                        now - started_at,
                        idle_for,
                        progress["active_stream"],
                        progress["new_output_chars"],
                        progress["stdout_chars"],
                        progress["stderr_chars"],
                        progress["recent_output"],
                    )
                    last_stdout_chars = int(progress["stdout_chars"])
                    last_stderr_chars = int(progress["stderr_chars"])
        finally:
            if _owns_stdin_handle and stdin_handle is not None:
                stdin_handle.close()

    def _handle_idle_limit(
        self,
        process: subprocess.Popen,
        *,
        command: list[str],
        started_at: float,
        idle_for: float,
        prompt_summary: dict[str, str],
        stdout_chunks: list[str],
        stderr_chunks: list[str],
        stdout_thread: threading.Thread | None,
        stderr_thread: threading.Thread | None,
    ) -> None:
        progress = _stream_progress_summary(stdout_chunks, stderr_chunks)
        logger.warning(
            "CLI 长时间无新输出，准备终止进程: context=%s cli=%s, model=%s, pid=%s, elapsed=%.1fs, idle_for=%.1fs, active_stream=%s, recent_output=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            getattr(process, "pid", "-"),
            time.monotonic() - started_at,
            idle_for,
            progress["active_stream"],
            progress["recent_output"],
        )
        command_preview = self._command_preview_for_logs(command)
        cleanup = self._terminate_process_tree(process)
        self._close_process_streams(process)
        alive_threads = self._join_stream_reader(stdout_thread, stderr_thread)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        final_progress = _stream_progress_summary(stdout_chunks, stderr_chunks)
        logger.error(
            "CLI 长时间无新输出: context=%s cli=%s, model=%s, pid=%s, idle_limit=%ss, idle_for=%.1fs, argv=%s, prompt_transport=%s, prompt_chars=%s, prompt_preview=%s, prompt_tail=%s, cleanup=%s, reader_threads_alive=%s, active_stream=%s, stdout_chars=%s, stderr_chars=%s, recent_output=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            getattr(process, "pid", "-"),
            self.max_idle_seconds,
            idle_for,
            command_preview,
            prompt_summary["transport"],
            prompt_summary["chars"],
            prompt_summary["preview"],
            prompt_summary["tail"],
            cleanup,
            ",".join(alive_threads) if alive_threads else "-",
            final_progress["active_stream"],
            len(stdout or ""),
            len(stderr or ""),
            final_progress["recent_output"],
        )

    def _handle_timeout(
        self,
        process: subprocess.Popen,
        *,
        command: list[str],
        started_at: float,
        prompt_summary: dict[str, str],
        stdout_chunks: list[str],
        stderr_chunks: list[str],
        stdout_thread: threading.Thread | None,
        stderr_thread: threading.Thread | None,
    ) -> None:
        progress = _stream_progress_summary(stdout_chunks, stderr_chunks)
        logger.warning(
            "CLI 调用超时，准备终止进程: context=%s cli=%s, model=%s, pid=%s, elapsed=%.1fs, active_stream=%s, recent_output=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            getattr(process, "pid", "-"),
            time.monotonic() - started_at,
            progress["active_stream"],
            progress["recent_output"],
        )
        command_preview = self._command_preview_for_logs(command)
        cleanup = self._terminate_process_tree(process)
        self._close_process_streams(process)
        alive_threads = self._join_stream_reader(stdout_thread, stderr_thread)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        final_progress = _stream_progress_summary(stdout_chunks, stderr_chunks)
        logger.error(
            "CLI 调用超时: context=%s cli=%s, model=%s, pid=%s, timeout=%ss, argv=%s, prompt_transport=%s, prompt_chars=%s, prompt_preview=%s, prompt_tail=%s, cleanup=%s, reader_threads_alive=%s, active_stream=%s, stdout_chars=%s, stderr_chars=%s, recent_output=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            getattr(process, "pid", "-"),
            self.timeout,
            command_preview,
            prompt_summary["transport"],
            prompt_summary["chars"],
            prompt_summary["preview"],
            prompt_summary["tail"],
            cleanup,
            ",".join(alive_threads) if alive_threads else "-",
            final_progress["active_stream"],
            len(stdout or ""),
            len(stderr or ""),
            final_progress["recent_output"],
        )

    def _terminate_process_tree(self, process: subprocess.Popen) -> str:
        pid = getattr(process, "pid", None)
        notes: list[str] = []
        pgid: int | None = None
        if process.poll() is not None:
            return f"already_exited(returncode={process.returncode})"
        if pid:
            try:
                pgid = os.getpgid(pid)
                notes.append(f"pgid={pgid}")
            except (ProcessLookupError, OSError) as exc:
                notes.append(f"pgid_unavailable={exc}")
        if pgid and process.poll() is None:
            try:
                os.killpg(pgid, signal.SIGTERM)
                notes.append("sent=SIGTERM")
            except ProcessLookupError:
                notes.append("sigterm=process_group_missing")
            except OSError as exc:
                notes.append(f"sigterm_failed={exc}")
        elif process.poll() is None:
            try:
                process.terminate()
                notes.append("sent=terminate")
            except OSError as exc:
                notes.append(f"terminate_failed={exc}")
        if process.poll() is None:
            try:
                process.wait(timeout=_TERMINATION_GRACE_SECONDS)
                notes.append(f"exit_after_term={process.returncode}")
            except subprocess.TimeoutExpired:
                notes.append(f"term_grace_expired={_TERMINATION_GRACE_SECONDS}s")
        if process.poll() is None and pgid:
            try:
                os.killpg(pgid, signal.SIGKILL)
                notes.append("sent=SIGKILL")
            except ProcessLookupError:
                notes.append("sigkill=process_group_missing")
            except OSError as exc:
                notes.append(f"sigkill_failed={exc}")
        elif process.poll() is None:
            try:
                process.kill()
                notes.append("sent=kill")
            except OSError as exc:
                notes.append(f"kill_failed={exc}")
        if process.poll() is None:
            try:
                process.wait(timeout=_KILL_WAIT_SECONDS)
                notes.append(f"exit_after_kill={process.returncode}")
            except subprocess.TimeoutExpired:
                notes.append(f"kill_wait_expired={_KILL_WAIT_SECONDS}s")
        if process.poll() is not None:
            notes.append(f"returncode={process.returncode}")
        return ",".join(notes) if notes else "cleanup=noop"

    def _log_failure(self, result: subprocess.CompletedProcess):
        logger.error(
            "CLI 调用失败: context=%s cli=%s, model=%s, stdout=%s, stderr=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            _preview_text(result.stdout, limit=200),
            _preview_text(result.stderr, limit=500),
        )

    def _log_output_summary(self, output: str):
        logger.info(
            "CLI 输出摘要: context=%s cli=%s, model=%s, chars=%s, preview=%s",
            self.log_context or "-",
            self.command[0],
            self.model or "default",
            len(output),
            _preview_text(output, limit=300),
        )

    def _start_stream_reader(
        self,
        stream,
        chunks: list[str],
    ) -> threading.Thread | None:
        if stream is None:
            return None
        thread = threading.Thread(target=_consume_stream, args=(stream, chunks), daemon=True)
        thread.start()
        return thread

    def _join_stream_reader(self, *threads: threading.Thread | None) -> list[str]:
        alive_threads: list[str] = []
        for thread in threads:
            if thread is not None:
                thread.join(timeout=_STREAM_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    alive_threads.append(thread.name)
        return alive_threads

    def _close_process_streams(self, process: subprocess.Popen):
        _safe_close_stream(getattr(process, "stdout", None))
        _safe_close_stream(getattr(process, "stderr", None))

    def _command_preview_for_logs(self, command: list[str], *, stdin_path: str | None = None) -> str:
        sanitized = list(command)
        if stdin_path:
            return _preview_command(sanitized)
        if self.prompt_mode == "positional" and sanitized:
            sanitized[-1] = "<prompt>"
            return _preview_command(sanitized)
        if self.prompt_mode == "flag" and self.prompt_flag:
            for index in range(len(sanitized) - 1):
                if sanitized[index] == self.prompt_flag:
                    sanitized[index + 1] = "<prompt>"
                    break
        return _preview_command(sanitized)

    def _prompt_summary_for_logs(self, prompt: str, *, stdin_path: str | None = None) -> dict[str, str]:
        normalized_prompt = str(prompt or "")
        return {
            "transport": "stdin" if stdin_path else "argv",
            "stdin_path": stdin_path or "-",
            "chars": str(len(normalized_prompt)),
            "preview": _preview_text(normalized_prompt, limit=400) or "-",
            "tail": _preview_text_from_end(normalized_prompt, limit=220) or "-",
        }


class CopilotCLIAdapter(CLIAdapter):
    """GitHub Copilot CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        max_idle_seconds: int = 0,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
        billing_type: str = "call",
    ):
        super().__init__(
            command=command or ["copilot"],
            prompt_flag="-p",
            prompt_mode="flag",
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            model=model,
            model_flag=model_flag or "",
            working_dir=working_dir,
            billing_type=billing_type,
        )

    def _run_with_stdin(self, command: list[str], prompt: str) -> subprocess.CompletedProcess:
        """Copilot CLI 要求 stdin 是真实管道而非普通文件，否则会拒绝读取。"""
        read_fd, write_fd = os.pipe()

        def _write_and_close():
            try:
                with os.fdopen(write_fd, "w", encoding="utf-8") as writer:
                    writer.write(prompt)
            except OSError:
                pass

        write_thread = threading.Thread(target=_write_and_close, daemon=True)
        write_thread.start()

        read_handle = os.fdopen(read_fd, "r", encoding="utf-8")
        try:
            return self._run_command(command, stdin_handle=read_handle, prompt=prompt)
        finally:
            read_handle.close()
            write_thread.join(timeout=5)


class CursorCLIAdapter(CLIAdapter):
    """Cursor CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        max_idle_seconds: int = 0,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
        billing_type: str = "call",
    ):
        super().__init__(
            command=command or ["cursor"],
            prompt_flag="--prompt",
            prompt_mode="flag",
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            model=model,
            model_flag=model_flag or "",
            working_dir=working_dir,
            billing_type=billing_type,
        )


class CodexCLIAdapter(CLIAdapter):
    """Codex CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        max_idle_seconds: int = 0,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
        billing_type: str = "token",
    ):
        super().__init__(
            command=command or ["codex", "exec", "--skip-git-repo-check"],
            prompt_flag="",
            prompt_mode="positional",
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            model=model,
            model_flag=model_flag or "--model",
            working_dir=working_dir,
            billing_type=billing_type,
        )

    def analyze(self, prompt: str, system: str = "") -> str:
        full_prompt = _SAFETY_PREFIX
        billing_hint = self._billing_instruction()
        if billing_hint:
            full_prompt += billing_hint
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
                result = self._run_command(self._build_command(cmd, full_prompt), prompt=full_prompt)
            if result.returncode != 0:
                self._log_failure(result)
                raise LLMError(f"CLI 工具返回非零退出码 {result.returncode}: {result.stderr[:500]}")

            try:
                with open(output_path, "r", encoding="utf-8") as handle:
                    output = handle.read().strip()
            except OSError:
                output = ""
            final_output = output or result.stdout.strip()
            if final_output:
                self._log_output_summary(final_output)
            return final_output
        except subprocess.TimeoutExpired as exc:
            raise LLMError(self._limit_error_message(exc))
        except FileNotFoundError:
            raise LLMError(f"CLI 工具未找到: {self.command[0]}")
        finally:
            if os.path.exists(output_path):
                _safe_unlink(output_path)


class OpenCodeCLIAdapter(CLIAdapter):
    """OpenCode CLI 适配器"""
    def __init__(
        self,
        timeout: int = 120,
        max_idle_seconds: int = 0,
        command: list[str] | None = None,
        model: str = "",
        model_flag: str | None = None,
        working_dir: str | None = None,
        billing_type: str = "token",
    ):
        super().__init__(
            command=command or ["opencode"],
            prompt_flag="--prompt",
            prompt_mode="flag",
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            model=model,
            model_flag=model_flag or "--model",
            working_dir=working_dir,
            billing_type=billing_type,
        )


def _consume_stream(stream, chunks: list[str]):
    try:
        while True:
            try:
                chunk = stream.read(4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        _safe_close_stream(stream)


def _safe_close_stream(stream) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        return


def _preview_recent_chunks(chunks: list[str], *, limit: int = 200) -> str:
    if not chunks:
        return ""
    collected: list[str] = []
    total = 0
    for chunk in reversed(chunks):
        collected.append(chunk)
        total += len(chunk)
        if total >= max(limit * 2, 400):
            break
    return _preview_text("".join(reversed(collected)), limit=limit)
