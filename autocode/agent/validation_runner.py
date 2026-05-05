"""AutoCode 本地验证执行器。"""
from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

_ENVIRONMENT_BLOCKER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?mi)(?:^|[\s/:])([A-Za-z0-9_.+-]+): command not found"), "缺少命令 `{match}`"),
    (re.compile(r"(?mi)No module named ['\"]?([A-Za-z0-9_.-]+)['\"]?"), "缺少 Python 模块 `{match}`"),
    (re.compile(r"(?mi)can't open file ['\"]([^'\"]+)['\"]"), "缺少脚本 `{match}`"),
)
_ENVIRONMENT_BLOCKER_MARKERS: tuple[tuple[str, str], ...] = (
    ("no such file or directory", "命令或脚本不存在"),
    ("not installed", "依赖未安装"),
)
_DEFAULT_TRANSIENT_ARTIFACT_DIRS = frozenset({
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "venv",
})
_PROCESS_TERMINATE_GRACE_SECONDS = 3
_PROCESS_KILL_GRACE_SECONDS = 2


@dataclass(frozen=True)
class ValidationStepResult:
    name: str
    command: list[str]
    success: bool
    output: str
    is_timeout: bool = False

    def environment_blocker(self) -> str:
        if self.success:
            return ""
        return _detect_environment_blocker(self.command, self.output)


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    steps: list[ValidationStepResult]

    def first_failed_step(self) -> ValidationStepResult | None:
        return next((step for step in self.steps if not step.success), None)

    def summarize(self) -> str:
        if not self.steps:
            return "未配置验证命令"
        parts: list[str] = []
        for step in self.steps:
            if step.success:
                status = "PASS"
            elif step.is_timeout:
                status = "TIMEOUT"
            else:
                status = "FAIL"
            parts.append(f"{step.name}:{status}")
        return ", ".join(parts)

    @property
    def only_test_timed_out(self) -> bool:
        """True when the only failures are test-step timeouts (format/lint all passed)."""
        if self.success:
            return False
        failed_steps = [s for s in self.steps if not s.success]
        return bool(failed_steps) and all(s.is_timeout and s.name == "test" for s in failed_steps)

    def environment_blocker(self) -> str:
        failed = self.first_failed_step()
        if failed is None:
            return ""
        return failed.environment_blocker()

    def failed_steps_summary(self, *, limit: int = 4, output_limit: int = 500) -> str:
        failed_steps = [step for step in self.steps if not step.success]
        if not failed_steps:
            return ""
        lines: list[str] = []
        for step in failed_steps[:limit]:
            command = " ".join(step.command).strip() or "验证命令"
            output = " ".join(str(step.output or "无输出").split())
            if len(output) > output_limit:
                output = output[: max(output_limit - 3, 1)] + "..."
            status = "TIMEOUT" if step.is_timeout else "FAIL"
            lines.append(f"{step.name}:{status} {command}\n{output}")
        if len(failed_steps) > limit:
            lines.append(f"... 还有 {len(failed_steps) - limit} 个失败步骤未展开")
        return "\n\n".join(lines)


class ValidationRunner:
    """按顺序执行 format/lint/test 验证。"""

    def __init__(
        self,
        *,
        format_commands: list[list[str]],
        lint_commands: list[list[str]],
        test_commands: list[list[str]],
        timeout_seconds: int,
        test_timeout_seconds: int = 0,
        fail_fast: bool = False,
        transient_artifact_dirs: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        self.format_commands = format_commands
        self.lint_commands = lint_commands
        self.test_commands = test_commands
        self.timeout_seconds = timeout_seconds
        self.test_timeout_seconds = test_timeout_seconds or timeout_seconds
        self.fail_fast = fail_fast
        self.transient_artifact_dirs = _normalize_artifact_dirs(transient_artifact_dirs)

    def run_all(self, workspace_path: str | Path, *, test_timeout_override: int = 0) -> ValidationResult:
        path = Path(workspace_path)
        steps: list[ValidationStepResult] = []
        effective_test_timeout = test_timeout_override or self.test_timeout_seconds
        for name, commands in (
            ("format", self.format_commands),
            ("lint", self.lint_commands),
            ("test", self.test_commands),
        ):
            step_timeout = effective_test_timeout if name == "test" else self.timeout_seconds
            for command in commands:
                logger.info(
                    "validation: 开始执行%s校验 workspace=%s command=%s",
                    name,
                    path,
                    " ".join(command),
                )
                try:
                    completed = self._run_command(path, command, step_timeout)
                except subprocess.TimeoutExpired as exc:
                    output_parts = [f"命令超时 ({step_timeout}s): {' '.join(command)}"]
                    if exc.stdout:
                        output_parts.append(str(exc.stdout).strip())
                    if exc.stderr:
                        output_parts.append(str(exc.stderr).strip())
                    steps.append(
                        ValidationStepResult(
                            name=name,
                            command=command,
                            success=False,
                            output="\n".join(part for part in output_parts if part).strip(),
                            is_timeout=True,
                        )
                    )
                    logger.warning(
                        "validation: %s校验超时 workspace=%s timeout=%ss command=%s",
                        name,
                        path,
                        step_timeout,
                        " ".join(command),
                    )
                    if self.fail_fast:
                        return ValidationResult(success=False, steps=steps)
                    continue
                output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
                if completed.returncode != 0:
                    cleaned_artifacts = _cleanup_artifacts_referenced_by_output(
                        path,
                        output,
                        self.transient_artifact_dirs,
                    )
                    if cleaned_artifacts:
                        logger.warning(
                            "validation: %s校验失败命中临时依赖目录，已清理后重试 workspace=%s artifacts=%s command=%s",
                            name,
                            path,
                            ", ".join(cleaned_artifacts),
                            " ".join(command),
                        )
                        try:
                            completed = self._run_command(path, command, step_timeout)
                        except subprocess.TimeoutExpired as exc:
                            output_parts = [f"命令超时 ({step_timeout}s): {' '.join(command)}"]
                            if exc.stdout:
                                output_parts.append(str(exc.stdout).strip())
                            if exc.stderr:
                                output_parts.append(str(exc.stderr).strip())
                            steps.append(
                                ValidationStepResult(
                                    name=name,
                                    command=command,
                                    success=False,
                                    output="\n".join(part for part in output_parts if part).strip(),
                                    is_timeout=True,
                                )
                            )
                            logger.warning(
                                "validation: %s校验清理临时依赖目录后仍超时 workspace=%s timeout=%ss command=%s",
                                name,
                                path,
                                step_timeout,
                                " ".join(command),
                            )
                            if self.fail_fast:
                                return ValidationResult(success=False, steps=steps)
                            continue
                        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
                step_result = ValidationStepResult(
                    name=name,
                    command=command,
                    success=completed.returncode == 0,
                    output=output.strip(),
                )
                steps.append(step_result)
                if completed.returncode == 0:
                    logger.info(
                        "validation: %s校验通过 workspace=%s command=%s",
                        name,
                        path,
                        " ".join(command),
                    )
                else:
                    logger.warning(
                        "validation: %s校验失败 workspace=%s command=%s output=%s",
                        name,
                        path,
                        " ".join(command),
                        _preview_output(step_result.output),
                    )
                    blocker = step_result.environment_blocker()
                    if blocker:
                        logger.warning(
                            "validation: %s校验检测到环境阻塞 workspace=%s command=%s blocker=%s",
                            name,
                            path,
                            " ".join(command),
                            blocker,
                        )
                if completed.returncode != 0 and self.fail_fast:
                    return ValidationResult(success=False, steps=steps)
        return ValidationResult(
            success=not any(not step.success for step in steps),
            steps=steps,
        )

    @staticmethod
    def _run_command(path: Path, command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            cwd=str(path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout = exc.stdout or exc.output
                stderr = exc.stderr
            raise subprocess.TimeoutExpired(
                command,
                timeout_seconds,
                output=_string_output(stdout),
                stderr=_string_output(stderr),
            ) from exc
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode or 0,
            stdout=stdout or "",
            stderr=stderr or "",
        )


def _preview_output(text: str, *, limit: int = 200) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 1)] + "..."


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    pgid: int | None = None
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            logger.warning("validation: 终止超时命令进程组失败 pid=%s pgid=%s error=%s", process.pid, pgid, exc)
        else:
            try:
                process.wait(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
                return
            except subprocess.TimeoutExpired:
                pass

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as exc:
            logger.warning("validation: 强制终止超时命令进程组失败 pid=%s pgid=%s error=%s", process.pid, pgid, exc)
            return
        try:
            process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("validation: 超时命令进程组未能完全退出 pid=%s pgid=%s", process.pid, pgid)
        return

    try:
        process.terminate()
        process.wait(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
    except (ProcessLookupError, OSError):
        return
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            logger.warning("validation: 超时命令进程未能完全退出 pid=%s", process.pid)


def _string_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _detect_environment_blocker(command: list[str], output: str) -> str:
    text = str(output or "")
    if not text.strip():
        return ""

    hints: list[str] = []
    for pattern, template in _ENVIRONMENT_BLOCKER_PATTERNS:
        for match in pattern.findall(text):
            hints.append(template.format(match=match))

    lowered = text.lower()
    for marker, hint in _ENVIRONMENT_BLOCKER_MARKERS:
        if marker in lowered:
            hints.append(hint)

    unique_hints: list[str] = []
    for hint in hints:
        normalized = str(hint).strip()
        if normalized and normalized not in unique_hints:
            unique_hints.append(normalized)
    if not unique_hints:
        return ""

    command_text = " ".join(command).strip() or "验证命令"
    return f"{command_text} 依赖环境未就绪：{'；'.join(unique_hints)}"


def _cleanup_artifacts_referenced_by_output(
    workspace_path: Path,
    output: str,
    artifact_dirs: set[str],
) -> list[str]:
    """Remove transient dependency/build dirs only when a failed validator references them."""
    referenced = _referenced_artifact_dirs(output, artifact_dirs)
    cleaned: list[str] = []
    for rel_dir in sorted(referenced):
        artifact_path = (workspace_path / rel_dir).resolve()
        try:
            artifact_path.relative_to(workspace_path.resolve())
        except ValueError:
            continue
        if not artifact_path.is_dir():
            continue
        try:
            shutil.rmtree(artifact_path)
        except OSError as exc:
            logger.warning("validation: 清理临时依赖目录失败 path=%s error=%s", artifact_path, exc)
            continue
        cleaned.append(rel_dir)
    return cleaned


def _referenced_artifact_dirs(output: str, artifact_dirs: set[str]) -> set[str]:
    text = str(output or "").replace("\\", "/")
    referenced: set[str] = set()
    for artifact_name in artifact_dirs:
        pattern = re.compile(rf"(?:(?:^|[\s'\"`:(])((?:[\w.+-]+/)*){re.escape(artifact_name)})(?:/|$)")
        for match in pattern.finditer(text):
            prefix = match.group(1) or ""
            rel_dir = f"{prefix}{artifact_name}".lstrip("/")
            if ".." in Path(rel_dir).parts:
                continue
            referenced.add(rel_dir)
    return referenced


def _normalize_artifact_dirs(value: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    raw_values = value if value is not None else _DEFAULT_TRANSIENT_ARTIFACT_DIRS
    normalized: set[str] = set()
    for item in raw_values:
        cleaned = str(item or "").strip().strip("/")
        if not cleaned or cleaned in {".", ".."}:
            continue
        if ".." in Path(cleaned).parts:
            continue
        normalized.add(cleaned)
    return normalized
