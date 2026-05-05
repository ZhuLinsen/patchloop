"""可写工作区执行适配器。"""
from __future__ import annotations

import subprocess
import tempfile

from .base import LLMError
from .cli_adapter import (
    CLIAdapter,
    CodexCLIAdapter,
    CopilotCLIAdapter,
    CursorCLIAdapter,
    OpenCodeCLIAdapter,
    _MAX_ARG_LENGTH,
    _safe_unlink,
)

_EXECUTION_PREFIX = (
    "【重要】你正在受限工作区内执行自动编码任务。"
    "你可以修改当前工作区内的文件来完成需求，但必须遵守以下规则：\n"
    "1. 只允许修改当前仓库工作区内的文件，不要访问工作区外路径。\n"
    "2. 不要执行 git merge、git push 到默认分支、删除历史、改写远端历史。\n"
    "3. 不要创建或修改任何会提升权限的配置，例如 secrets、deployment、workflow。\n"
    "4. 完成后只输出简短实现摘要，不要输出思考过程。\n\n"
)


def _ensure_codex_workspace_write(command: list[str] | None) -> list[str]:
    effective = list(command) if command else ["codex", "exec", "--skip-git-repo-check"]
    if len(effective) < 2 or effective[0] != "codex" or effective[1] != "exec":
        return effective
    if "--dangerously-bypass-approvals-and-sandbox" in effective or "--full-auto" in effective:
        return effective
    if "--sandbox" in effective or "-s" in effective:
        return effective
    return [*effective, "--sandbox", "workspace-write"]


def _ensure_copilot_execution_permissions(command: list[str] | None) -> list[str]:
    """Copilot CLI 非交互模式需要 --allow-all 来启用所有权限（包括工具、路径、URL）。
    --allow-all 等价于 --allow-all-tools --allow-all-paths --allow-all-urls"""
    effective = list(command) if command else ["copilot"]
    if "--allow-all" not in effective:
        effective.append("--allow-all")
    return effective


class ExecutionCLIAdapter(CLIAdapter):
    """允许在受限工作区内修改文件的通用 CLI 适配器。"""

    def analyze(self, prompt: str, system: str = "") -> str:
        execution_prompt = _EXECUTION_PREFIX
        billing_hint = self._billing_instruction()
        if billing_hint:
            execution_prompt += billing_hint
        if system:
            execution_prompt += f"{system}\n\n"
        execution_prompt += prompt
        base_cmd = list(self.command)
        if self.model and self.model_flag:
            base_cmd.extend([self.model_flag, self.model])

        try:
            if len(execution_prompt) > _MAX_ARG_LENGTH:
                result = self._run_with_stdin(base_cmd, execution_prompt)
            else:
                cmd = self._build_command(base_cmd, execution_prompt)
                result = self._run_command(cmd, prompt=execution_prompt)
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
            raise LLMError(f"CLI 调用异常: cli={self.command[0]}, error={exc}") from exc


class ExecutionCopilotCLIAdapter(ExecutionCLIAdapter, CopilotCLIAdapter):
    """Copilot 写代码模式。"""

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
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            command=_ensure_copilot_execution_permissions(command),
            model=model,
            model_flag=model_flag or "",
            working_dir=working_dir,
            billing_type=billing_type,
        )


class ExecutionCursorCLIAdapter(ExecutionCLIAdapter, CursorCLIAdapter):
    """Cursor 写代码模式。"""


class ExecutionCodexCLIAdapter(ExecutionCLIAdapter, CodexCLIAdapter):
    """Codex 写代码模式。"""

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
            timeout=timeout,
            max_idle_seconds=max_idle_seconds,
            command=_ensure_codex_workspace_write(command),
            model=model,
            model_flag=model_flag or None,
            working_dir=working_dir,
            billing_type=billing_type,
        )

    def analyze(self, prompt: str, system: str = "") -> str:
        execution_prompt = _EXECUTION_PREFIX
        billing_hint = self._billing_instruction()
        if billing_hint:
            execution_prompt += billing_hint
        if system:
            execution_prompt += f"{system}\n\n"
        execution_prompt += prompt
        base_cmd = list(self.command)
        if self.model and self.model_flag:
            base_cmd.extend([self.model_flag, self.model])

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            cmd = base_cmd + ["--output-last-message", output_path]
            if len(execution_prompt) > _MAX_ARG_LENGTH:
                result = self._run_with_stdin(cmd, execution_prompt)
            else:
                result = self._run_command(self._build_command(cmd, execution_prompt), prompt=execution_prompt)
            if result.returncode != 0:
                self._log_failure(result)
                raise LLMError(f"CLI 工具返回非零退出码 {result.returncode}: {result.stderr[:500]}")
            try:
                output = open(output_path, "r", encoding="utf-8").read().strip()
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
        except OSError as exc:
            raise LLMError(f"CLI 调用异常: cli={self.command[0]}, error={exc}") from exc
        finally:
            _safe_unlink(output_path)


class ExecutionOpenCodeCLIAdapter(ExecutionCLIAdapter, OpenCodeCLIAdapter):
    """OpenCode 写代码模式。"""
